#![recursion_limit = "20"]

pub(crate) mod utils {
    use std::mem::MaybeUninit;

    #[inline]
    pub unsafe fn leaky_iterator_to_array<I: Iterator,const D: usize>(iter: I) -> [I::Item; D] {
        // SAFETY: an array of MaybeUninits doesn't require initialization as they may be uninitialized
        let mut arr: [MaybeUninit<I::Item>; D] = unsafe { MaybeUninit::uninit().assume_init() };
                
        // SAFETY: old vals aren't dropped as dropping a MaybeUninit does nothing
        for (idx,val) in iter.enumerate() {
            arr[idx] = MaybeUninit::new(val);
        }

        // SAFETY: VERY UNSAFE, SHOULD BE CHANGED WHEN ARRAY ASSUME INIT IS STABLE
        // 1: [MaybeUninit<T>; D] == [T; D] in mem
        // 2: 1 -> *const [MaybeUninit<T>; D] == *const [T; D]
        // 3: 2 -> transmute is safe
        // 4: forget doesn't cause a leak because the initialized_arr and arr alias and so arr will be dropped with initialized_arr
        //     4+: necessary because otherwise arr would be dropped twice invalidating initialized_arr and causing a double free
        //
        // addendum: pointers are only used because transmute thinks size_of::<T> != size_of::<T> 
        let initialized_arr;
        unsafe { 
            initialized_arr = std::ptr::read(std::mem::transmute(&arr as *const [MaybeUninit<I::Item>; D]));
            std::mem::forget(arr);
        }

        initialized_arr
    }

    pub struct IterInitArray<T,const D: usize>{pub array: [MaybeUninit<T>; D], pub last_index: Option<usize>}

    impl<T,const D: usize> IterInitArray<T,D> {
        #[inline]
        pub unsafe fn assume_init(self) -> [T; D] {
            // SAFETY: VERY UNSAFE, SHOULD BE CHANGED WHEN ARRAY ASSUME INIT IS STABLE
            // 1: [MaybeUninit<T>; D] == [T; D] in mem
            // 2: 1 -> *const [MaybeUninit<T>; D] == *const [T; D]
            // 3: 2 -> transmute is safe
            // 4: forget doesn't cause a leak because the initialized_arr and arr alias and so arr will be dropped with initialized_arr
            //     4+: necessary because otherwise arr would be dropped twice invalidating initialized_arr and causing a double free
            //
            // addendum: pointers are only used because transmute thinks size_of::<T> != size_of::<T> 
            let initialized_arr;
            unsafe {
                initialized_arr = std::ptr::read(std::mem::transmute(&self.array as *const [MaybeUninit<T>; D]));
                std::mem::forget(self);
            }
            initialized_arr
        }
    }

    //Also both of these are actually unsafe
    //no check for if the value is initialized
    impl<T,const D: usize> std::ops::Index<usize> for IterInitArray<T,D> where [MaybeUninit<T>; D]: std::ops::Index<usize>, usize: std::slice::SliceIndex<[MaybeUninit<T>]> {
        type Output = <[MaybeUninit<T>; D] as std::ops::Index<usize>>::Output;

        #[inline]
        fn index(&self, index: usize) -> &Self::Output {
            self.array.index(index)
        }
    }

    //To ensure Safety, any val recieved must be written to with a valid value
    impl<T,const D: usize> std::ops::IndexMut<usize> for IterInitArray<T,D> where [MaybeUninit<T>; D]: std::ops::Index<usize>, usize: std::slice::SliceIndex<[MaybeUninit<T>]> {
        #[inline]
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            self.last_index = Some(index);
            self.array.index_mut(index)
        }
    }

    impl<T,const D: usize> std::ops::Drop for IterInitArray<T,D> {
        #[inline]
        fn drop(&mut self) {
            if let Some(last_index) = self.last_index {
                for i in 0..=last_index {
                    unsafe { self.array[i].assume_init_drop() };
                }
            }
        }
    }

    #[inline]
    pub unsafe fn iterator_to_array<I: Iterator,const D: usize>(iter: I) -> [I::Item; D] {
        let mut arr : IterInitArray<I::Item, D>= IterInitArray{array: unsafe {MaybeUninit::uninit().assume_init()},last_index: None};

        for (idx,val) in iter.enumerate() {
            arr[idx] = MaybeUninit::new(val);
        }

        arr.assume_init()
    }

    pub struct UncheckedCell<T>(std::cell::UnsafeCell<T>);

    impl<T> UncheckedCell<T> {
        #[inline]
        pub fn new(val: T) -> UncheckedCell<T> {
            UncheckedCell(std::cell::UnsafeCell::new(val))
        }

        #[allow(dead_code)]
        #[inline]
        pub unsafe fn borrow<'a>(&'a self) -> &'a T {
            unsafe { &*self.0.get()}
        }

        #[inline]
        pub unsafe fn borrow_mut<'a>(&'a self) -> &'a mut T {
            unsafe { &mut *self.0.get()}
        }
    }
}


#[derive(PartialEq,Debug)]
#[derive(Clone,Copy)]
#[repr(transparent)]
pub struct Scalar<T>(pub T);

//CHECK
impl<T> Scalar<T> {
    pub fn unwrap(self) -> T {
        self.0
    }

    pub fn new(val: T) -> Self {
        Scalar(val)
    }
}

pub mod scalar_math {
    use std::ops::*;
    use super::Scalar;

    macro_rules! overload_basic_operator_for_scalar {
        ($Op:ident,$OpF:ident) => {
            impl<S1,S2> $Op<Scalar<S2>> for Scalar<S1> where S1: $Op<S2> {
                type Output = Scalar<S1::Output>;

                #[inline]
                fn $OpF(self,rhs: Scalar<S2>) -> Self::Output {
                    Scalar(self.0.$OpF(rhs.0))
                }
            }

            impl<'a,S1,S2> $Op<Scalar<S2>> for &'a Scalar<S1> where &'a S1: $Op<S2> {
                type Output = Scalar<<&'a S1 as $Op<S2>>::Output>;

                #[inline]
                fn $OpF(self,rhs: Scalar<S2>) -> Self::Output {
                    Scalar((&self.0).$OpF(rhs.0))
                }
            }

            impl<'a,S1,S2> $Op<&'a Scalar<S2>> for Scalar<S1> where S1: $Op<&'a S2> {
                type Output = Scalar<S1::Output>;

                #[inline]
                fn $OpF(self,rhs: &'a Scalar<S2>) -> Self::Output {
                    Scalar(self.0.$OpF(&rhs.0))
                }
            }

            impl<'a,S1,S2> $Op<&'a Scalar<S2>> for &'a Scalar<S1> where &'a S1: $Op<&'a S2> {
                type Output = Scalar<<&'a S1 as $Op<&'a S2>>::Output>;

                #[inline]
                fn $OpF(self,rhs: &'a Scalar<S2>) -> Self::Output {
                    Scalar((&self.0).$OpF(&rhs.0))
                }
            }
        };
    }
    macro_rules! overload_assign_operator_for_scalar {
        ($Op:ident,$OpF:ident) => {
            impl<S1,S2> $Op<Scalar<S2>> for Scalar<S1> where S1: $Op<S2> {
                #[inline]
                fn $OpF(&mut self,rhs: Scalar<S2>) {
                    self.0.$OpF(rhs.0);
                }
            }

            impl<'a,S1,S2> $Op<&'a Scalar<S2>> for Scalar<S1> where S1: $Op<&'a S2> {
                #[inline]
                fn $OpF(&mut self,rhs: &'a Scalar<S2>) {
                    self.0.$OpF(&rhs.0);
                }
            }
        };
    }

    overload_basic_operator_for_scalar!(Add,add);
    overload_basic_operator_for_scalar!(Sub,sub);
    overload_basic_operator_for_scalar!(Mul,mul);
    overload_basic_operator_for_scalar!(Div,div);
    overload_basic_operator_for_scalar!(Rem,rem);
    overload_assign_operator_for_scalar!(AddAssign,add_assign);
    overload_assign_operator_for_scalar!(SubAssign,sub_assign);
    overload_assign_operator_for_scalar!(MulAssign,mul_assign);
    overload_assign_operator_for_scalar!(DivAssign,div_assign);
    overload_assign_operator_for_scalar!(RemAssign,rem_assign);
}

#[derive(PartialEq,Debug)]
#[derive(Clone)] 
pub struct ColumnVec<T: IntoIterator,const D: usize>(T);

impl<T: IntoIterator,const D: usize> ColumnVec<T,D> {
    pub fn unwrap(self) -> T {
        self.0
    }

    pub unsafe fn unchecked_from(iterable: T) -> Self {
        ColumnVec(iterable)
    }

    pub fn transpose(self) -> RowVec<T,D> {
        RowVec(self.0)
    }
}

impl<T,const D: usize> From<[T; D]> for ColumnVec<[T; D],D> {
    #[inline]
    fn from(value: [T; D]) -> Self {
        ColumnVec(value)
    }
}

impl<T,const D: usize> TryFrom<Vec<T>> for ColumnVec<Vec<T>,D> {
    type Error = ();

    #[inline]
    fn try_from(value: Vec<T>) -> Result<Self, Self::Error> {
        if value.len() == D {
            Ok(ColumnVec(value))
        } else {
            Err(())
        }
    }
}

//CHECK
impl<Idx,T: IntoIterator + std::ops::Index<Idx>,const D: usize> std::ops::Index<Idx> for ColumnVec<T,D> where T::Output: Sized {
    type Output = Scalar<T::Output>;
    
    #[inline]
    fn index(&self, index: Idx) -> &Self::Output {
        //SAFETY: scalar is repr(transparant) so the representaion of Scalar<T> 
        //is identical to T and as such &Scalar<T> is identical to &T or so I
        //think. DOUBLE CHECK THIS ZACH, double checked, SHOULD BE CHECKED MORE
        unsafe { std::mem::transmute((&self.0).index(index)) }
    }
}

#[derive(Clone)]
pub struct ColVecIter<I: Iterator>(I);

impl<I: Iterator> Iterator for ColVecIter<I> {
    type Item = Scalar<I::Item>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(val) = self.0.next() {
            Some(Scalar(val))
        } else {
            None
        }
    }
}

impl<T: IntoIterator,const D: usize> IntoIterator for ColumnVec<T,D> {
    type IntoIter = ColVecIter<T::IntoIter>;
    type Item = Scalar<T::Item>;
    
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        ColVecIter(self.0.into_iter())
    }
}

impl<'a,T: IntoIterator,const D: usize> IntoIterator for &'a ColumnVec<T,D> where &'a T: IntoIterator {
    type IntoIter = ColVecIter<<&'a T as IntoIterator>::IntoIter>;
    type Item = Scalar<<&'a T as IntoIterator>::Item>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        ColVecIter((&self.0).into_iter())
    }
}

impl<'a,T: IntoIterator,const D: usize> IntoIterator for &'a mut ColumnVec<T,D> where &'a mut T: IntoIterator {
    type IntoIter = ColVecIter<<&'a mut T as IntoIterator>::IntoIter>;
    type Item = Scalar<<&'a mut T as IntoIterator>::Item>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        ColVecIter((&mut self.0).into_iter())
    }
}

 
pub struct RowVec<T: IntoIterator,const D: usize>(pub T);


pub trait Dot<T> {
    ///A trait for implementing a dot product function
    ///Implementations of It generally makes the assumption that
    ///the output type has a default function which results
    ///in a value akin to 0 (ex. 0,0.0,"")
    type Output;

    fn dot(self,rhs: T) -> Self::Output;
}

pub mod col_vec_iterators {
    use std::ops::*;
    use super::*;

    pub struct EvalColVec<T,const D: usize>(pub(crate) T);

    impl<I: Iterator,const D: usize> IntoIterator for EvalColVec<I,D> {
        type IntoIter = ColVecIter<I>;
        type Item = Scalar<I::Item>;

        #[inline]
        fn into_iter(self) -> Self::IntoIter {
            ColVecIter(self.0)
        }
    }


    pub trait EvalsToColVec<const D: usize> {
        type Item;
        
        fn eval(self) -> ColumnVec<Vec<Self::Item>,D>;

        fn leaky_eval_array(self) -> ColumnVec<[Self::Item; D],D>;

        fn eval_array(self) -> ColumnVec<[Self::Item; D],D>;
    }

    impl<T: EvalsToColVec<D>,const D: usize> EvalsToColVec<D> for EvalColVec<T,D> {
        type Item = T::Item;
        
        #[inline]
        fn eval(self) -> ColumnVec<Vec<Self::Item>,D> {
            self.0.eval()
        }

        #[inline]
        fn leaky_eval_array(self) -> ColumnVec<[Self::Item; D],D> {
            self.0.leaky_eval_array()
        }

        #[inline]
        fn eval_array(self) -> ColumnVec<[Self::Item; D],D> {
            self.0.eval_array()
        }
    }

    
    pub trait SyncEvalsToColVec {
        type EvalOut;
        type EvalArrayOut;
    
        fn sync_eval(self) -> Self::EvalOut;
        fn sync_leaky_eval_array(self) -> Self::EvalArrayOut;
        fn sync_eval_array(self) -> Self::EvalArrayOut;
    }

    #[macro_export]
    macro_rules! impl_sync_evals_to_col_vec {
        ($($i:ident)*,$($i_buf:ident)*) => {
            impl<$($i: Iterator),*,const D: usize> SyncEvalsToColVec for ($(EvalColVec<$i,D>),*) {
                type EvalOut = ($(ColumnVec<Vec<$i::Item>,D>),*);
                type EvalArrayOut = ($(ColumnVec<[$i::Item; D],D>),*);
    
                #[allow(non_snake_case)]
                #[inline]
                fn sync_eval(self) -> Self::EvalOut {
                    let ($(EvalColVec(mut $i)),*) = self;
                    $(
                        let mut $i_buf = Vec::with_capacity(D);
                    )*
                    for _ in 0..D {
                        $(
                            $i_buf.push($i.next().expect("internal_error: The first D elements in a EvalColVec must be valid"));
                        )*
                    }
                    ($(ColumnVec($i_buf)),*)
                }
    
                #[allow(non_snake_case)]
                #[inline]
                fn sync_leaky_eval_array(self) -> Self::EvalArrayOut {
                    let ($(EvalColVec(mut $i)),*) = self;
                    $(
                        let mut $i_buf: [std::mem::MaybeUninit<$i::Item>; D] = unsafe {std::mem::MaybeUninit::uninit().assume_init()};
                    )*
                    for i in 0..D {
                        $(
                            $i_buf[i] = std::mem::MaybeUninit::new($i.next().expect("internal_error: The first D elements in a EvalColVec must be valid"));
                        )*
                    }
                    ($(ColumnVec(unsafe {
                        let initialized_arr: [$i::Item; D] = std::ptr::read(std::mem::transmute(&$i_buf as *const [std::mem::MaybeUninit<$i::Item>; D]));
                        std::mem::forget($i_buf);
                        initialized_arr
                    })),*)
                }

                #[allow(non_snake_case)]
                #[inline]
                fn sync_eval_array(self) -> Self::EvalArrayOut {
                    let ($(EvalColVec(mut $i)),*) = self;
                    $(
                        let mut $i_buf: utils::IterInitArray<$i::Item, D>= utils::IterInitArray{array: unsafe {std::mem::MaybeUninit::uninit().assume_init()},last_index: None};
                    )*

                    for i in 0..D {
                        $(
                            $i_buf[i] = std::mem::MaybeUninit::new($i.next().expect("internal_error: The first D elements in a EvalColVec must be valid"));
                        )*
                    }
                    ($(ColumnVec(unsafe {$i_buf.assume_init()})),*)
                }
            }
        };
    }
    
    macro_rules! recursive_impl_sync_evals_to_col_vec {
        ($i:ident,$i_buf:ident) => {};
        ($i_first:ident $($i:ident)* ,$i_buf_first:ident $($i_buf:ident)* ) => {
            impl_sync_evals_to_col_vec!($i_first $($i)*,$i_buf_first $($i_buf)*);
            recursive_impl_sync_evals_to_col_vec!($($i)*,$($i_buf)*);
        };
    } 

    recursive_impl_sync_evals_to_col_vec!(A B C NotD E F G H I J K L M N O P,A_buf B_buf C_buf D_buf E_buf F_buf G_buf H_buf I_buf J_buf K_buf L_buf M_buf N_buf O_buf P_buf);


    macro_rules! impl_eval_col_vec {
        ($t:ty,1,$ti:ty,$tr:path) => {
            impl<I: Iterator,S: Copy,const D: usize> EvalsToColVec<D> for $t where $ti: $tr {
                type Item = <$t as Iterator>::Item;

                #[inline]
                fn eval(self) -> ColumnVec<Vec<Self::Item>,D> {
                    let mut vec = Vec::with_capacity(D);
                    for output in self {
                        vec.push(output)
                    }
                    ColumnVec(vec)
                }
                
                #[inline]
                fn leaky_eval_array(self) -> ColumnVec<[Self::Item; D],D> {
                    ColumnVec(unsafe { utils::leaky_iterator_to_array(self) })
                }

                #[inline]
                fn eval_array(self) -> ColumnVec<[Self::Item; D],D> {
                    ColumnVec(unsafe { utils::iterator_to_array(self) })
                }
            }
        };
        ($t:ty,2,$ti:ty,$tr:path) => {
            impl<I1: Iterator,I2: Iterator,const D: usize> EvalsToColVec<D> for $t where $ti: $tr {
                type Item = <$t as Iterator>::Item;

                #[inline]
                fn eval(self) -> ColumnVec<Vec<Self::Item>,D> {
                    let mut vec = Vec::with_capacity(D);
                    for output in self {
                        vec.push(output)
                    }
                    ColumnVec(vec)
                }

                #[inline]
                fn leaky_eval_array(self) -> ColumnVec<[Self::Item; D],D> {
                    ColumnVec(unsafe { utils::leaky_iterator_to_array(self) })
                }

                #[inline]
                fn eval_array(self) -> ColumnVec<[Self::Item; D],D> {
                    ColumnVec(unsafe { utils::iterator_to_array(self) })
                }
            }
        };
    }

    macro_rules! overload_given_operator_for_vec {
        ($Op:ident,$OpF:ident,$OpI:ident) => {
            impl<I1: IntoIterator,I2: IntoIterator,const D: usize> $Op<ColumnVec<I2,D>> for ColumnVec<I1,D> where I1::Item: $Op<I2::Item> {
                type Output = EvalColVec<$OpI<I1::IntoIter,I2::IntoIter,D>,D>;

                #[inline]
                fn $OpF(self,other: ColumnVec<I2,D>) -> Self::Output {
                    EvalColVec($OpI(
                        self.0.into_iter(),
                        other.0.into_iter()
                    ))
                }
            }

            impl<'a,I1: IntoIterator,I2: IntoIterator,const D: usize> $Op<ColumnVec<I2,D>> for &'a ColumnVec<I1,D> 
            where &'a I1: IntoIterator, <&'a I1 as IntoIterator>::Item: $Op<I2::Item> {
                type Output = EvalColVec<$OpI<<&'a I1 as IntoIterator>::IntoIter,I2::IntoIter,D>,D>;

                #[inline]
                fn $OpF(self,other: ColumnVec<I2,D>) -> Self::Output {
                    EvalColVec($OpI(
                        (&self.0).into_iter(),
                        other.0.into_iter()
                    ))
                }
            }

            impl<'a,I1: IntoIterator,I2: IntoIterator,const D: usize> $Op<&'a ColumnVec<I2,D>> for ColumnVec<I1,D> 
            where &'a I2: IntoIterator, I1::Item: $Op<<&'a I2 as IntoIterator>::Item> {
                type Output = EvalColVec<$OpI<I1::IntoIter,<&'a I2 as IntoIterator>::IntoIter,D>,D>;

                #[inline]
                fn $OpF(self,other: &'a ColumnVec<I2,D>) -> Self::Output {
                    EvalColVec($OpI(
                        self.0.into_iter(),
                        (&other.0).into_iter()
                    ))
                }
            }

            impl<'a,I1: IntoIterator,I2: IntoIterator,const D: usize> $Op<&'a ColumnVec<I2,D>> for &'a ColumnVec<I1,D> 
            where &'a I1: IntoIterator, &'a I2: IntoIterator, <&'a I1 as IntoIterator>::Item: $Op<<&'a I2 as IntoIterator>::Item> {
                type Output = EvalColVec<$OpI<<&'a I1 as IntoIterator>::IntoIter,<&'a I2 as IntoIterator>::IntoIter,D>,D>;

                #[inline]
                fn $OpF(self,other: &'a ColumnVec<I2,D>) -> Self::Output {
                    EvalColVec($OpI(
                        (&self.0).into_iter(),
                        (&other.0).into_iter()
                    ))
                }
            }
        

            impl<I1: Iterator,I2: IntoIterator,const D: usize> $Op<ColumnVec<I2,D>> for EvalColVec<I1,D> where I1::Item: $Op<I2::Item> {
                type Output = EvalColVec<$OpI<I1,I2::IntoIter,D>,D>;

                #[inline]
                fn $OpF(self,other: ColumnVec<I2,D>) -> Self::Output {
                    EvalColVec($OpI(
                        self.0,
                        other.0.into_iter()
                    ))
                }
            }

            impl<'a,I1: Iterator,I2: IntoIterator,const D: usize> $Op<&'a ColumnVec<I2,D>> for EvalColVec<I1,D> 
            where &'a I2: IntoIterator, I1::Item: $Op<<&'a I2 as IntoIterator>::Item> {
                type Output = EvalColVec<$OpI<I1,<&'a I2 as IntoIterator>::IntoIter,D>,D>;

                #[inline]
                fn $OpF(self,other: &'a ColumnVec<I2,D>) -> Self::Output {
                    EvalColVec($OpI(
                        self.0,
                        (&other.0).into_iter()
                    ))
                }
            }

            impl<I1: IntoIterator,I2: Iterator,const D: usize> $Op<EvalColVec<I2,D>> for ColumnVec<I1,D> where I1::Item: $Op<I2::Item> {
                type Output = EvalColVec<$OpI<I1::IntoIter,I2,D>,D>;

                #[inline]
                fn $OpF(self,other: EvalColVec<I2,D>) -> Self::Output {
                    EvalColVec($OpI(
                        self.0.into_iter(),
                        other.0
                    ))
                }
            }

            impl<'a,I1: IntoIterator,I2: Iterator,const D: usize> $Op<EvalColVec<I2,D>> for &'a ColumnVec<I1,D> 
            where &'a I1: IntoIterator, <&'a I1 as IntoIterator>::Item: $Op<I2::Item> {
                type Output = EvalColVec<$OpI<<&'a I1 as IntoIterator>::IntoIter,I2,D>,D>;

                #[inline]
                fn $OpF(self,other: EvalColVec<I2,D>) -> Self::Output {
                    EvalColVec($OpI(
                        (&self.0).into_iter(),
                        other.0
                    ))
                }
            }


            impl<I1: Iterator,I2: Iterator,const D: usize> $Op<EvalColVec<I2,D>> for EvalColVec<I1,D> 
            where I1::Item: $Op<I2::Item> {
                type Output = EvalColVec<$OpI<I1,I2,D>,D>;

                #[inline]
                fn $OpF(self,other: EvalColVec<I2,D>) -> Self::Output {
                    EvalColVec($OpI(
                        self.0,
                        other.0
                    ))
                }
            }
        };
        ($Op:ident,$OpF:ident,$OpI:ident,0) => {
            impl<I: IntoIterator,S: Copy,const D: usize> $Op<Scalar<S>> for ColumnVec<I,D> where I::Item: $Op<S> {
                type Output = EvalColVec<$OpI<I::IntoIter,S,D>,D>;

                #[inline]
                fn $OpF(self,rhs: Scalar<S>) -> Self::Output {
                    EvalColVec($OpI(
                        self.0.into_iter(),
                        rhs.0
                    ))
                }
            }

            impl<'a,I: IntoIterator,S: Copy,const D: usize> $Op<Scalar<S>> for &'a ColumnVec<I,D> 
            where &'a I: IntoIterator, <&'a I as IntoIterator>::Item: $Op<S> {
                type Output = EvalColVec<$OpI<<&'a I as IntoIterator>::IntoIter,S,D>,D>;

                #[inline]
                fn $OpF(self,rhs: Scalar<S>) -> Self::Output {
                    EvalColVec($OpI(
                        (&self.0).into_iter(),
                        rhs.0
                    ))
                }
            }

            impl<'a,I: IntoIterator,S: Copy,const D: usize> $Op<&'a Scalar<S>> for ColumnVec<I,D> where I::Item: $Op<&'a S> {
                type Output = EvalColVec<$OpI<I::IntoIter,&'a S,D>,D>;

                #[inline]
                fn $OpF(self,rhs: &'a Scalar<S>) -> Self::Output {
                    EvalColVec($OpI(
                        self.0.into_iter(),
                        &rhs.0
                    ))
                }
            }

            impl<'a,I: IntoIterator,S: Copy,const D: usize> $Op<&'a Scalar<S>> for &'a ColumnVec<I,D> 
            where &'a I: IntoIterator, <&'a I as IntoIterator>::Item: $Op<&'a S> {
                type Output = EvalColVec<$OpI<<&'a I as IntoIterator>::IntoIter,&'a S,D>,D>;

                #[inline]
                fn $OpF(self,rhs: &'a Scalar<S>) -> Self::Output {
                    EvalColVec($OpI(
                        (&self.0).into_iter(),
                        &rhs.0
                    ))
                }
            }


            impl<I: Iterator,S: Copy,const D: usize> $Op<Scalar<S>> for EvalColVec<I,D> where I::Item: $Op<S> {
                type Output = EvalColVec<$OpI<I,S,D>,D>;

                #[inline]
                fn $OpF(self,rhs: Scalar<S>) -> Self::Output {
                    EvalColVec($OpI(
                        self.0,
                        rhs.0
                    ))
                }
            }

            impl<'a,I: Iterator,S: Copy,const D: usize> $Op<&'a Scalar<S>> for EvalColVec<I,D> where I::Item: $Op<&'a S> {
                type Output = EvalColVec<$OpI<I,&'a S,D>,D>;

                #[inline]
                fn $OpF(self,rhs: &'a Scalar<S>) -> Self::Output {
                    EvalColVec($OpI(
                        self.0,
                        &rhs.0
                    ))
                }
            }
        };
        ($Op:ident,$OpF:ident,$OpI:ident,1) => {
            impl<I: IntoIterator,S: Copy,const D: usize> $Op<ColumnVec<I,D>> for Scalar<S> where S: $Op<I::Item> {
                type Output = EvalColVec<$OpI<S,I::IntoIter,D>,D>;

                #[inline]
                fn $OpF(self,rhs: ColumnVec<I,D>) -> Self::Output {
                    EvalColVec($OpI(
                        rhs.0.into_iter(),
                        self.0
                    ))
                }
            }

            impl<'a,I: IntoIterator,S: Copy,const D: usize> $Op<ColumnVec<I,D>> for &'a Scalar<S> where &'a S: $Op<I::Item> {
                type Output = EvalColVec<$OpI<&'a S,I::IntoIter,D>,D>;

                #[inline]
                fn $OpF(self,rhs: ColumnVec<I,D>) -> Self::Output {
                    EvalColVec($OpI(
                        rhs.0.into_iter(),
                        &self.0
                    ))
                }
            }

            impl<'a,I: IntoIterator,S: Copy,const D: usize> $Op<&'a ColumnVec<I,D>> for Scalar<S> 
            where &'a I: IntoIterator, S: $Op<<&'a I as IntoIterator>::Item> {
                type Output = EvalColVec<$OpI<S,<&'a I as IntoIterator>::IntoIter,D>,D>;

                #[inline]
                fn $OpF(self,rhs: &'a ColumnVec<I,D>) -> Self::Output {
                    EvalColVec($OpI(
                        (&rhs.0).into_iter(),
                        self.0
                    ))
                }
            }

            impl<'a,I: IntoIterator,S: Copy,const D: usize> $Op<&'a ColumnVec<I,D>> for &'a Scalar<S> 
            where &'a I: IntoIterator, &'a S: $Op<<&'a I as IntoIterator>::Item> {
                type Output = EvalColVec<$OpI<&'a S,<&'a I as IntoIterator>::IntoIter,D>,D>;

                #[inline]
                fn $OpF(self,rhs: &'a ColumnVec<I,D>) -> Self::Output {
                    EvalColVec($OpI(
                        (&rhs.0).into_iter(),
                        &self.0
                    ))
                }
            }
        
            
            impl<I: Iterator,S: Copy,const D: usize> $Op<EvalColVec<I,D>> for Scalar<S> where S: $Op<I::Item> {
                type Output = EvalColVec<$OpI<S,I,D>,D>;

                #[inline]
                fn $OpF(self,rhs: EvalColVec<I,D>) -> Self::Output {
                    EvalColVec($OpI(
                        rhs.0,
                        self.0
                    ))
                }
            }

            impl<'a, I: Iterator,S: Copy,const D: usize> $Op<EvalColVec<I,D>> for &'a Scalar<S> where &'a S: $Op<I::Item> {
                type Output = EvalColVec<$OpI<&'a S,I,D>,D>;

                #[inline]
                fn $OpF(self,rhs: EvalColVec<I,D>) -> Self::Output {
                    EvalColVec($OpI(
                        rhs.0,
                        &self.0
                    ))
                }
            }
        }
    }
    
    pub struct ColVecAdd<I1: Iterator,I2: Iterator,const D: usize>(I1,I2) where I1::Item: Add<I2::Item>;

    impl<I1: Iterator,I2: Iterator,const D: usize> Iterator for ColVecAdd<I1,I2,D> where I1::Item: Add<I2::Item> {
        type Item = <I1::Item as Add<I2::Item>>::Output;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            if let (Some(left),Some(right)) = (self.0.next(),self.1.next()) {
                Some(left + right)
            } else {
                None
            }
        }
    }
    
    impl_eval_col_vec!(ColVecAdd<I1,I2,D>,2,I1::Item,Add<I2::Item>);
    overload_given_operator_for_vec!(Add,add,ColVecAdd);

    pub struct ColVecSub<I1: Iterator,I2: Iterator,const D: usize>(I1,I2) where I1::Item: Sub<I2::Item>;
    
    impl<I1: Iterator,I2: Iterator,const D: usize> Iterator for ColVecSub<I1,I2,D> where I1::Item: Sub<I2::Item> {
        type Item = <I1::Item as Sub<I2::Item>>::Output;
        
        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            if let (Some(left),Some(right)) = (self.0.next(),self.1.next()) {
                Some(left - right)
            } else {
                None
            }
        }
    }
    
    impl_eval_col_vec!(ColVecSub<I1,I2,D>,2,I1::Item,Sub<I2::Item>);
    overload_given_operator_for_vec!(Sub,sub,ColVecSub);

    pub struct ColVecMulAsFirst<I: Iterator,S: Copy,const D: usize>(I,S) where I::Item: Mul<S>;

    impl<I: Iterator,S: Copy,const D: usize> Iterator for ColVecMulAsFirst<I,S,D> where I::Item: Mul<S> {
        type Item = <I::Item as Mul<S>>::Output;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            if let Some(val) = self.0.next() {
                Some(val * self.1)
            } else {
                None
            }
        }
    }
    
    impl_eval_col_vec!(ColVecMulAsFirst<I,S,D>,1,I::Item,Mul<S>);
    overload_given_operator_for_vec!(Mul,mul,ColVecMulAsFirst,0);

    pub struct ColVecMulAsSecond<S: Copy,I: Iterator,const D: usize>(I,S) where S: Mul<I::Item>;

    impl<S: Copy,I: Iterator,const D: usize> Iterator for ColVecMulAsSecond<S,I,D> where S: Mul<I::Item> {
        type Item = S::Output;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            if let Some(val) = self.0.next() {
                Some(self.1 * val)
            } else {
                None
            }
        }
    }

    impl_eval_col_vec!(ColVecMulAsSecond<S,I,D>,1,S,Mul<I::Item>);
    overload_given_operator_for_vec!(Mul,mul,ColVecMulAsSecond,1);

    pub struct ColVecDivAsDividend<I: Iterator,S: Copy,const D: usize>(I,S) where I::Item: Div<S>;

    impl<I: Iterator,S: Copy,const D: usize> Iterator for ColVecDivAsDividend<I,S,D> where I::Item: Div<S> {
        type Item = <I::Item as Div<S>>::Output;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            if let Some(val) = self.0.next() {
                Some(val / self.1)
            } else {
                None
            }
        }
    }

    impl_eval_col_vec!(ColVecDivAsDividend<I,S,D>,1,I::Item,Div<S>);
    overload_given_operator_for_vec!(Div,div,ColVecDivAsDividend,0);

    pub struct ColVecDivAsDivisor<S: Copy,I: Iterator,const D: usize>(I,S) where S: Div<I::Item>; 

    impl<S: Copy,I: Iterator,const D: usize> Iterator for ColVecDivAsDivisor<S,I,D> where S: Div<I::Item> {
        type Item = S::Output;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            if let Some(val) = self.0.next() {
                Some(self.1 / val)
            } else {
                None
            }
        }
    }

    impl_eval_col_vec!(ColVecDivAsDivisor<S,I,D>,1,S,Div<I::Item>);
    overload_given_operator_for_vec!(Div,div,ColVecDivAsDivisor,1);

    pub struct ColVecRemAsDividend<I: Iterator,S: Copy,const D: usize>(I,S) where I::Item: Rem<S>;

    impl<I: Iterator,S: Copy,const D: usize> Iterator for ColVecRemAsDividend<I,S,D> where I::Item: Rem<S> {
        type Item = <I::Item as Rem<S>>::Output;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            if let Some(val) = self.0.next() {
                Some(val % self.1)
            } else {
                None
            }
        }
    }

    impl_eval_col_vec!(ColVecRemAsDividend<I,S,D>,1,I::Item,Rem<S>);
    overload_given_operator_for_vec!(Rem,rem,ColVecRemAsDividend,0);

    pub struct ColVecRemAsDivisor<S: Copy,I: Iterator,const D: usize>(I,S) where S: Rem<I::Item>;

    impl<S: Copy,I: Iterator,const D: usize> Iterator for ColVecRemAsDivisor<S,I,D> where S: Rem<I::Item> {
        type Item = S::Output;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            if let Some(val) = self.0.next() {
                Some(self.1 % val)
            } else {
                None
            }
        }
    }

    impl_eval_col_vec!(ColVecRemAsDivisor<S,I,D>,1,S,Rem<I::Item>);
    overload_given_operator_for_vec!(Rem,rem,ColVecRemAsDivisor,1);

    //Assign operators
    macro_rules! impl_assign_operator_for_vec {
        ($Op:ident,$OpF:ident,0) => {
            impl<I1: IntoIterator,I2: IntoIterator,O,const D: usize> $Op<ColumnVec<I2,D>> for ColumnVec<I1,D> 
            where for<'a> &'a mut I1: IntoIterator<Item = &'a mut O>,O: $Op<I2::Item> {
                #[inline]
                fn $OpF(&mut self,rhs: ColumnVec<I2,D>) {
                    for (self_val,other_val) in (&mut self.0).into_iter().zip(rhs.0.into_iter()) {
                        (self_val).$OpF(other_val)
                    }
                }
            }

            impl<'b,I1: IntoIterator,I2: IntoIterator,O,const D: usize> $Op<&'b ColumnVec<I2,D>> for ColumnVec<I1,D> 
            where &'b I2: IntoIterator,for<'a> &'a mut I1: IntoIterator<Item = &'a mut O>,O: $Op<<&'b I2 as IntoIterator>::Item> {
                #[inline]
                fn $OpF(&mut self,rhs: &'b ColumnVec<I2,D>) {
                    for (self_val,other_val) in (&mut self.0).into_iter().zip((&rhs.0).into_iter()) {
                        (self_val).$OpF(other_val)
                    }
                }
            }

            impl<I1: IntoIterator,I2: Iterator,O,const D: usize> $Op<EvalColVec<I2,D>> for ColumnVec<I1,D> 
            where for<'a> &'a mut I1: IntoIterator<Item = &'a mut O>,O: $Op<I2::Item> {
                #[inline]
                fn $OpF(&mut self,rhs: EvalColVec<I2,D>) {
                    for (self_val,other_val) in (&mut self.0).into_iter().zip(rhs.0) {
                        (self_val).$OpF(other_val)
                    }
                }
            }
        };
        ($Op:ident,$OpF:ident,1) => {
            impl<I: IntoIterator,O,S: Copy,const D: usize> $Op<Scalar<S>> for ColumnVec<I,D> 
            where for<'a> &'a mut I: IntoIterator<Item = &'a mut O>,for<'a> O: $Op<S> {
                #[inline]
                fn $OpF(&mut self,rhs: Scalar<S>) {
                    for val in (&mut self.0).into_iter() {
                        (val).$OpF(rhs.0)
                    }
                }
            }

            impl<'b,I: IntoIterator,O,S: Copy,const D: usize> $Op<&'b Scalar<S>> for ColumnVec<I,D> 
            where for<'a> &'a mut I: IntoIterator<Item = &'a mut O>,for<'a> O: $Op<&'b S> {
                #[inline]
                fn $OpF(&mut self,rhs: &'b Scalar<S>) {
                    for val in (&mut self.0).into_iter() {
                        (val).$OpF(&rhs.0)
                    }
                }
            }
        }
    }

    impl_assign_operator_for_vec!(AddAssign,add_assign,0);
    impl_assign_operator_for_vec!(SubAssign,sub_assign,0);
    impl_assign_operator_for_vec!(MulAssign,mul_assign,1);
    impl_assign_operator_for_vec!(DivAssign,div_assign,1);
    impl_assign_operator_for_vec!(RemAssign,rem_assign,1);

    //dot product

    //assumes the default is something akin to 0
    impl<I1: IntoIterator,I2: IntoIterator,O: AddAssign<O> + std::default::Default ,const D: usize> Dot<ColumnVec<I2,D>> for ColumnVec<I1,D>
    where I1::Item: Mul<I2::Item,Output = O>  {
        type Output = Scalar<O>;

        #[inline]
        fn dot(self,rhs: ColumnVec<I2,D>) -> Self::Output {
            let mut output = <O>::default();
            for (left_val,right_val) in self.0.into_iter().zip(rhs.0.into_iter()) {
                output += left_val * right_val;
            }
            Scalar(output)
        }
    }

    impl<'a, I1: IntoIterator,I2: IntoIterator,O: AddAssign<O> + std::default::Default ,const D: usize> Dot<ColumnVec<I2,D>> for &'a ColumnVec<I1,D>
    where &'a I1: IntoIterator,<&'a I1 as IntoIterator>::Item: Mul<I2::Item,Output = O>  {
        type Output = Scalar<O>;

        #[inline]
        fn dot(self,rhs: ColumnVec<I2,D>) -> Self::Output {
            let mut output = <O>::default();
            for (left_val,right_val) in (&self.0).into_iter().zip(rhs.0.into_iter()) {
                output += left_val * right_val;
            }
            Scalar(output)
        }
    }

    impl<'a, I1: IntoIterator,I2: IntoIterator,O: AddAssign<O> + std::default::Default ,const D: usize> Dot<&'a ColumnVec<I2,D>> for ColumnVec<I1,D>
    where &'a I2: IntoIterator, I1::Item: Mul<<&'a I2 as IntoIterator>::Item,Output = O>  {
        type Output = Scalar<O>;

        #[inline]
        fn dot(self,rhs: &'a ColumnVec<I2,D>) -> Self::Output {
            let mut output = <O>::default();
            for (left_val,right_val) in self.0.into_iter().zip((&rhs.0).into_iter()) {
                output += left_val * right_val;
            }
            Scalar(output)
        }
    }

    impl<'a, I1: IntoIterator,I2: IntoIterator,O: AddAssign<O> + std::default::Default ,const D: usize> Dot<&'a ColumnVec<I2,D>> for &'a ColumnVec<I1,D>
    where &'a I2: IntoIterator, &'a I1: IntoIterator, <&'a I1 as IntoIterator>::Item: Mul<<&'a I2 as IntoIterator>::Item,Output = O>  {
        type Output = Scalar<O>;

        fn dot(self,rhs: &'a ColumnVec<I2,D>) -> Self::Output {
            let mut output = <O>::default();
            for (left_val,right_val) in (&self.0).into_iter().zip((&rhs.0).into_iter()) {
                output += left_val * right_val;
            }
            Scalar(output)
        }
    }


    impl<I1: Iterator,I2: IntoIterator,O: AddAssign<O> + std::default::Default,const D: usize> Dot<ColumnVec<I2,D>> for EvalColVec<I1,D> 
    where I1::Item: Mul<I2::Item,Output = O> {
        type Output = Scalar<O>;

        #[inline]
        fn dot(self,rhs: ColumnVec<I2,D>) -> Self::Output {
            let mut output = <O>::default();
            for (left_val,right_val) in self.0.zip(rhs.0.into_iter()) {
                output += left_val * right_val;
            }
            Scalar(output)
        }
    }

    impl<'a,I1: Iterator,I2: IntoIterator,O: AddAssign<O> + std::default::Default,const D: usize> Dot<&'a ColumnVec<I2,D>> for EvalColVec<I1,D> 
    where &'a I2: IntoIterator, I1::Item: Mul<<&'a I2 as IntoIterator>::Item,Output = O> {
        type Output = Scalar<O>;

        #[inline]
        fn dot(self,rhs: &'a ColumnVec<I2,D>) -> Self::Output {
            let mut output = <O>::default();
            for (left_val,right_val) in self.0.zip((&rhs.0).into_iter()) {
                output += left_val * right_val;
            }
            Scalar(output)
        }
    }

    impl<I1: IntoIterator,I2: Iterator,O: AddAssign<O> + std::default::Default,const D: usize> Dot<EvalColVec<I2,D>> for ColumnVec<I1,D> 
    where I1::Item: Mul<I2::Item,Output = O> {
        type Output = Scalar<O>;

        #[inline]
        fn dot(self,rhs: EvalColVec<I2,D>) -> Self::Output {
            let mut output = <O>::default();
            for (left_val,right_val) in self.0.into_iter().zip(rhs.0) {
                output += left_val * right_val;
            }
            Scalar(output)
        }
    }

    impl<'a, I1: IntoIterator,I2: Iterator,O: AddAssign<O> + std::default::Default,const D: usize> Dot<EvalColVec<I2,D>> for &'a ColumnVec<I1,D> 
    where &'a I1: IntoIterator,<&'a I1 as IntoIterator>::Item: Mul<I2::Item,Output = O> {
        type Output = Scalar<O>;

        #[inline]
        fn dot(self,rhs: EvalColVec<I2,D>) -> Self::Output {
            let mut output = <O>::default();
            for (left_val,right_val) in (&self.0).into_iter().zip(rhs.0) {
                output += left_val * right_val;
            }
            Scalar(output)
        }
    }


    impl<I1: Iterator,I2: Iterator,O: AddAssign<O> + std::default::Default,const D: usize> Dot<EvalColVec<I2,D>> for EvalColVec<I1,D> 
    where I1::Item: Mul<I2::Item,Output = O> {
        type Output = Scalar<O>;

        #[inline]
        fn dot(self,rhs: EvalColVec<I2,D>) -> Self::Output {
            let mut output = <O>::default();
            for (left_val,right_val) in self.0.zip(rhs.0) {
                output += left_val * right_val;
            }
            Scalar(output)
        }
    }

    //Cloned and Copied
    pub struct ClonedColVec<'a,I: Iterator<Item = &'a T>,T: 'a,const D: usize>(I) where T: Clone;

    impl<'a,I: Iterator<Item = &'a T>,T: 'a,const D: usize> Iterator for ClonedColVec<'a,I,T,D> where T: Clone {
        type Item = T;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            self.0.next().cloned()
        }
    }

    impl<'a,I: Iterator<Item = &'a T>,T: 'a,const D: usize> EvalsToColVec<D> for ClonedColVec<'a,I,T,D> where T: Clone {
        type Item = T;

        #[inline]
        fn eval(self) -> ColumnVec<Vec<Self::Item>,D> {
            let mut vec = Vec::with_capacity(D);
            for output in self {
                vec.push(output)
            }
            ColumnVec(vec)
        }

        #[inline]
        fn leaky_eval_array(self) -> ColumnVec<[Self::Item; D],D> {
            ColumnVec(unsafe { utils::leaky_iterator_to_array(self) })
        }

        #[inline]
        fn eval_array(self) -> ColumnVec<[Self::Item; D],D> {
            ColumnVec(unsafe { utils::iterator_to_array(self) })
        }
    }


    pub struct CopiedColVec<'a,I: Iterator<Item = &'a T>,T: 'a,const D: usize>(I) where T: Copy;

    impl<'a,I: Iterator<Item = &'a T>,T: 'a,const D: usize> Iterator for CopiedColVec<'a,I,T,D> where T: Copy {
        type Item = T;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            self.0.next().copied()
        }
    }

    impl<'a,I: Iterator<Item = &'a T>,T: 'a,const D: usize> EvalsToColVec<D> for CopiedColVec<'a,I,T,D> where T: Copy {
        type Item = T;

        #[inline]
        fn eval(self) -> ColumnVec<Vec<Self::Item>,D> {
            let mut vec = Vec::with_capacity(D);
            for output in self {
                vec.push(output)
            }
            ColumnVec(vec)
        }

        #[inline]
        fn leaky_eval_array(self) -> ColumnVec<[Self::Item; D],D> {
            ColumnVec(unsafe { utils::leaky_iterator_to_array(self) })
        }

        #[inline]
        fn eval_array(self) -> ColumnVec<[Self::Item; D],D> {
            ColumnVec(unsafe { utils::iterator_to_array(self) })
        }
    }


    impl<'a,I: Iterator<Item = &'a T>,T: 'a,const D: usize> EvalColVec<I,D> {
        #[inline]
        pub fn cloned(self) -> EvalColVec<ClonedColVec<'a,I,T,D>,D> where T: Clone{
            EvalColVec(ClonedColVec(
                self.0
            ))
        }

        #[inline]
        pub fn copied(self) -> EvalColVec<CopiedColVec<'a,I,T,D>,D> where T: Copy{
            EvalColVec(CopiedColVec(
                self.0
            ))
        }
    }

    impl<'a,I: IntoIterator<Item = &'a T>,T: 'a,const D: usize> ColumnVec<I,D> {
        #[inline]
        pub fn cloned(self) -> EvalColVec<ClonedColVec<'a,I::IntoIter,T,D>,D> where T: Clone{
            EvalColVec(ClonedColVec(
                self.0.into_iter()
            ))
        }

        #[inline]
        pub fn copied(self) -> EvalColVec<CopiedColVec<'a,I::IntoIter,T,D>,D> where T: Copy{
            EvalColVec(CopiedColVec(
                self.0.into_iter()
            ))
        }
    }
    
    //Duplicate
    struct DuplicatedColVec<I: Iterator,const D: usize> where I::Item: Clone {
        iterator: I,
        buffer: Option<Option<I::Item>>,
    }

    pub struct FirstDuplicateColVec<I: Iterator,const D: usize>(std::rc::Rc<std::cell::RefCell<DuplicatedColVec<I,D>>>) where I::Item: Clone;

    pub struct SecondDuplicateColVec<I: Iterator,const D: usize>(std::rc::Rc<std::cell::RefCell<DuplicatedColVec<I,D>>>) where I::Item: Clone;

    impl<I: Iterator,const D: usize> Iterator for FirstDuplicateColVec<I,D> where I::Item: Clone {
        type Item = I::Item;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            let val = self.0.borrow_mut().iterator.next();
            if let Some(_) = self.0.borrow().buffer {
                panic!("math_vector: Error, FirstDuplicateColVec<...> must be read before SecondDuplicateColVec<...> for each item within\n\nThe FirstDuplicateColVec is likely being fully read before the other (ie finding the dot product with one)")
            }
            self.0.borrow_mut().buffer = Some(val.clone());
            val
        }
    }

    impl<I: Iterator,const D: usize> Iterator for SecondDuplicateColVec<I,D> where I::Item: Clone {
        type Item = I::Item;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            let val = std::mem::replace(&mut self.0.borrow_mut().buffer,None);
            match val {
                Some(inner_val) => inner_val,
                None => {panic!("math_vector: Error, FirstDuplicateColVec<...> must be read before SecondDuplicateColVec<...> for each item within\n\nEither you need to switch around the order of these 2 or \none of these 2 is being fully read before the other (ie finding the dot product with one)")}
            }
        }
    }


    impl<I: Iterator,const D: usize> EvalColVec<I,D> where I::Item: Clone {
        #[inline]
        pub fn duplicate(self) -> (EvalColVec<FirstDuplicateColVec<I,D>,D>,EvalColVec<SecondDuplicateColVec<I,D>,D>) {
            let shared_iter = std::rc::Rc::new(std::cell::RefCell::new(DuplicatedColVec{
                iterator: self.0,
                buffer: None
            }));
            (
                EvalColVec(FirstDuplicateColVec(shared_iter.clone())),
                EvalColVec(SecondDuplicateColVec(shared_iter))
            )
        }
    }

    impl<I: IntoIterator,const D: usize> ColumnVec<I,D> where I::Item: Clone {
        #[inline]
        pub fn duplicate(self) -> (EvalColVec<FirstDuplicateColVec<I::IntoIter,D>,D>,EvalColVec<SecondDuplicateColVec<I::IntoIter,D>,D>) {
            let shared_iter = std::rc::Rc::new(std::cell::RefCell::new(DuplicatedColVec{
                iterator: self.0.into_iter(),
                buffer: None
            }));
            (
                EvalColVec(FirstDuplicateColVec(shared_iter.clone())),
                EvalColVec(SecondDuplicateColVec(shared_iter))
            )
        }
        
        #[inline]
        pub fn duplicate_ref<'a>(&'a self) -> (EvalColVec<FirstDuplicateColVec<<&'a I as IntoIterator>::IntoIter,D>,D>,EvalColVec<SecondDuplicateColVec<<&'a I as IntoIterator>::IntoIter,D>,D>) 
        where &'a I: IntoIterator, <&'a I as IntoIterator>::Item: Clone {
            let shared_iter = std::rc::Rc::new(std::cell::RefCell::new(DuplicatedColVec{
                iterator: (&self.0).into_iter(),
                buffer: None
            }));
            (
                EvalColVec(FirstDuplicateColVec(shared_iter.clone())),
                EvalColVec(SecondDuplicateColVec(shared_iter))
            )
        }
    }
    
    //Unchecked Duplicate
    struct UncheckedDuplicatedColVec<I: Iterator,const D: usize> where I::Item: Clone {
        iterator: I,
        buffer: Option<I::Item>
    }

    pub struct FirstUncheckedDuplicateColVec<I: Iterator,const D: usize>(std::rc::Rc<utils::UncheckedCell<UncheckedDuplicatedColVec<I,D>>>) where I::Item: Clone;

    pub struct SecondUncheckedDuplicateColVec<I: Iterator,const D: usize>(std::rc::Rc<utils::UncheckedCell<UncheckedDuplicatedColVec<I,D>>>) where I::Item: Clone;

    impl<I: Iterator,const D: usize> Iterator for FirstUncheckedDuplicateColVec<I,D> where I::Item: Clone {
        type Item = I::Item;
        
        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            let val = unsafe { self.0.borrow_mut().iterator.next() };
            unsafe { self.0.borrow_mut().buffer = val.clone() };
            val
        }
    }

    impl<I: Iterator,const D: usize> Iterator for SecondUncheckedDuplicateColVec<I,D> where I::Item: Clone {
        type Item = I::Item;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            std::mem::replace( unsafe { &mut self.0.borrow_mut().buffer }, None)
        }
    }


    impl<I: Iterator,const D: usize> EvalColVec<I,D> where I::Item: Clone {
        #[inline]
        pub unsafe fn unchecked_duplicate(self) -> (EvalColVec<FirstUncheckedDuplicateColVec<I,D>,D>,EvalColVec<SecondUncheckedDuplicateColVec<I,D>,D>) {
            let shared_iter = std::rc::Rc::new(utils::UncheckedCell::new(UncheckedDuplicatedColVec{
                iterator: self.0,
                buffer: None
            }));
            (
                EvalColVec(FirstUncheckedDuplicateColVec(shared_iter.clone())),
                EvalColVec(SecondUncheckedDuplicateColVec(shared_iter))
            )
        }
    }

    impl<I: IntoIterator,const D: usize> ColumnVec<I,D> where I::Item: Clone {
        #[inline]
        pub unsafe fn unchecked_duplicate(self) -> (EvalColVec<FirstUncheckedDuplicateColVec<I::IntoIter,D>,D>,EvalColVec<SecondUncheckedDuplicateColVec<I::IntoIter,D>,D>) {
            let shared_iter = std::rc::Rc::new(utils::UncheckedCell::new(UncheckedDuplicatedColVec{
                iterator: self.0.into_iter(),
                buffer: None
            }));
            (
                EvalColVec(FirstUncheckedDuplicateColVec(shared_iter.clone())),
                EvalColVec(SecondUncheckedDuplicateColVec(shared_iter))
            )
        }
        
        #[inline]
        pub unsafe fn unchecked_duplicate_ref<'a>(&'a self) -> (EvalColVec<FirstUncheckedDuplicateColVec<<&'a I as IntoIterator>::IntoIter,D>,D>,EvalColVec<SecondUncheckedDuplicateColVec<<&'a I as IntoIterator>::IntoIter,D>,D>) 
        where &'a I: IntoIterator, <&'a I as IntoIterator>::Item: Clone {
            let shared_iter = std::rc::Rc::new(utils::UncheckedCell::new(UncheckedDuplicatedColVec{
                iterator: (&self.0).into_iter(),
                buffer: None
            }));
            (
                EvalColVec(FirstUncheckedDuplicateColVec(shared_iter.clone())),
                EvalColVec(SecondUncheckedDuplicateColVec(shared_iter))
            )
        }
    }
 
    //buffer ops (clone_to_buffer,copy_to_buffer)
    pub struct BufferedColVec<T,const D: usize>(utils::IterInitArray<T,D>);

    impl<T,const D: usize> EvalsToColVec<D> for BufferedColVec<T,D> {
        type Item = T;

        fn eval(self) -> ColumnVec<Vec<Self::Item>,D> {
            if self.0.last_index == Some(D-1) {
                ColumnVec(unsafe { self.0.assume_init() }.into())
            } else if self.0.last_index == Some(0) {
                panic!("A BufferedColVec must be written to before being evaluated");
            } else {
                panic!("A BufferedColVec must be written to only once");
            }
        }

        fn leaky_eval_array(self) -> ColumnVec<[Self::Item; D],D> {
            eprintln!("math_vector: Warning, BufferedColVec's leaky_eval_array is equivilent to its eval_array. \n\tUse LeakyUncheckedBufferedColVec for that instead");
            self.eval_array()
        }

        fn eval_array(self) -> ColumnVec<[Self::Item; D],D> {
            if self.0.last_index == Some(D-1) {
                ColumnVec(unsafe { self.0.assume_init() })
            } else if self.0.last_index == Some(0) {
                panic!("A BufferedColVec must be written to before being evaluated");
            } else {
                panic!("A BufferedColVec must be written to only once");
            }
        }
    }
    
    pub fn get_col_vec_buffer<T,const D: usize>() -> EvalColVec<BufferedColVec<T,D>,D> {
        EvalColVec(BufferedColVec(utils::IterInitArray{array: unsafe {std::mem::MaybeUninit::uninit().assume_init()},last_index: None}))
    }
    
        //Clone
    pub struct CloneToBufferColVec<'a,I: Iterator,const D: usize> where I::Item: Clone {iter: I, buffer: &'a mut BufferedColVec<I::Item,D>}

    impl<'a,I: Iterator,const D: usize> Iterator for CloneToBufferColVec<'a,I,D> where I::Item: Clone {
        type Item = I::Item;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            let val = self.iter.next();
            if let Some(inner_val) = val.clone() {
                match self.buffer.0.last_index {
                    None => {self.buffer.0[0] = std::mem::MaybeUninit::new(inner_val)}
                    Some(innerer_val) => {self.buffer.0[innerer_val+1] = std::mem::MaybeUninit::new(inner_val)}
                }
            }
            val
        }
    }
            //No reason not to implement this but its kinda dumb
    impl<'a,I: Iterator,const D: usize> EvalsToColVec<D> for CloneToBufferColVec<'a,I,D> where I::Item: Clone {
        type Item = I::Item;

        #[inline]
        fn eval(self) -> ColumnVec<Vec<Self::Item>,D> {
            let mut vec = Vec::with_capacity(D);
            for output in self {
                vec.push(output)
            }
            ColumnVec(vec)
        }

        #[inline]
        fn leaky_eval_array(self) -> ColumnVec<[Self::Item; D],D> {
            ColumnVec(unsafe { utils::leaky_iterator_to_array(self) })
        }

        #[inline]
        fn eval_array(self) -> ColumnVec<[Self::Item; D],D> {
            ColumnVec(unsafe { utils::iterator_to_array(self) })
        }
    }
    
        //Copy
    pub struct CopyToBufferColVec<'a,I: Iterator,const D: usize> where I::Item: Copy {iter: I, buffer: &'a mut BufferedColVec<I::Item,D>}

    impl<'a,I: Iterator,const D: usize> Iterator for CopyToBufferColVec<'a,I,D> where I::Item: Copy {
        type Item = I::Item;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            let val = self.iter.next();
            if let Some(inner_val) = val {
                match self.buffer.0.last_index {
                    None => {self.buffer.0[0] = std::mem::MaybeUninit::new(inner_val)}
                    Some(innerer_val) => {self.buffer.0[innerer_val+1] = std::mem::MaybeUninit::new(inner_val)}
                }
            }
            val
        }
    }
            //No reason not to implement this but its kinda dumb
    impl<'a,I: Iterator,const D: usize> EvalsToColVec<D> for CopyToBufferColVec<'a,I,D> where I::Item: Copy {
        type Item = I::Item;

        #[inline]
        fn eval(self) -> ColumnVec<Vec<Self::Item>,D> {
            let mut vec = Vec::with_capacity(D);
            for output in self {
                vec.push(output)
            }
            ColumnVec(vec)
        }

        #[inline]
        fn leaky_eval_array(self) -> ColumnVec<[Self::Item; D],D> {
            ColumnVec(unsafe { utils::leaky_iterator_to_array(self) })
        }

        #[inline]
        fn eval_array(self) -> ColumnVec<[Self::Item; D],D> {
            ColumnVec(unsafe { utils::iterator_to_array(self) })
        }
    }
        

    impl<I: Iterator,const D: usize> EvalColVec<I,D> {
        pub fn clone_to_buffer<'a>(self,buf: &'a mut EvalColVec<BufferedColVec<I::Item,D>,D>) -> EvalColVec<CloneToBufferColVec<'a,I,D>,D> where I::Item: Clone {
            EvalColVec(CloneToBufferColVec{
                iter: self.0,
                buffer: &mut buf.0
            })
        }

        pub fn copy_to_buffer<'a>(self,buf: &'a mut EvalColVec<BufferedColVec<I::Item,D>,D>) -> EvalColVec<CopyToBufferColVec<'a,I,D>,D> where I::Item: Copy {
            EvalColVec(CopyToBufferColVec{
                iter: self.0,
                buffer: &mut buf.0
            })
        }
    }

    //Map
    pub struct ColVecMap<I: Iterator,F: FnMut(I::Item) -> O,O,const D: usize>(I,F);

    impl<I: Iterator,F: FnMut(I::Item) -> O,O,const D: usize> Iterator for ColVecMap<I,F,O,D> {
        type Item = O;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            if let Some(val) = self.0.next() {
                Some(self.1(val))
            } else {
                None
            }
        }
    }

    impl<I: Iterator,F: FnMut(I::Item) -> O,O,const D: usize> EvalsToColVec<D> for ColVecMap<I,F,O,D> {
        type Item = O;

        #[inline]
        fn eval(self) -> ColumnVec<Vec<Self::Item>,D> {
            let mut vec = Vec::with_capacity(D);
            for output in self {
                vec.push(output)
            }
            ColumnVec(vec)
        }

        #[inline]
        fn leaky_eval_array(self) -> ColumnVec<[Self::Item; D],D> {
            ColumnVec(unsafe { utils::leaky_iterator_to_array(self) })
        }

        #[inline]
        fn eval_array(self) -> ColumnVec<[Self::Item; D],D> {
            ColumnVec(unsafe { utils::iterator_to_array(self) })
        }
    }


    impl<I: Iterator,const D: usize> EvalColVec<I,D> {
        #[inline]
        pub fn map<F: FnMut(I::Item) -> O,O>(self,f: F) -> ColVecMap<I,F,O,D> {
            ColVecMap(self.0,f)
        }
    }
    
    impl<I: IntoIterator,const D: usize> ColumnVec<I,D> {
        #[inline]
        pub fn map<F: FnMut(I::Item) -> O,O>(self,f: F) -> ColVecMap<I::IntoIter,F,O,D> {
            ColVecMap(self.0.into_iter(),f)
        }

        #[inline]
        pub fn map_ref<'a,F: FnMut(<&'a I as IntoIterator>::Item) -> O,O>(&'a self,f: F) -> ColVecMap<<&'a I as IntoIterator>::IntoIter,F,O,D> where &'a I: IntoIterator {
            ColVecMap((&self.0).into_iter(),f)
        }
    }
    
    //Map_and_store
    pub struct ColVecMapAndStore<'a,I: Iterator,F,O,const D: usize> where for<'b> F: FnMut(&'b I::Item) -> O {iter: I, f: F, buffer: &'a mut BufferedColVec<I::Item,D> }

    impl<'a,I: Iterator,F,O,const D: usize> Iterator for ColVecMapAndStore<'a,I,F,O,D> where for<'b> F: FnMut(&'b I::Item) -> O {
        type Item = O;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            match self.iter.next() {
                Some(val) => {
                    let out = (self.f)(&val); 
                    match self.buffer.0.last_index {
                        None => {self.buffer.0[0] = std::mem::MaybeUninit::new(val)}
                        Some(innerer_val) => {self.buffer.0[innerer_val+1] = std::mem::MaybeUninit::new(val)}
                    }
                    Some(out)
                }
                None => None
            }
        }
    }

    impl<'a,I: Iterator,F,O,const D: usize> EvalsToColVec<D> for ColVecMapAndStore<'a,I,F,O,D> where for<'b> F: FnMut(&'b I::Item) -> O {
        type Item = O;

        #[inline]
        fn eval(self) -> ColumnVec<Vec<Self::Item>,D> {
            let mut vec = Vec::with_capacity(D);
            for output in self {
                vec.push(output)
            }
            ColumnVec(vec)
        }

        #[inline]
        fn leaky_eval_array(self) -> ColumnVec<[Self::Item; D],D> {
            ColumnVec(unsafe { utils::leaky_iterator_to_array(self) })
        }

        #[inline]
        fn eval_array(self) -> ColumnVec<[Self::Item; D],D> {
            ColumnVec(unsafe { utils::iterator_to_array(self) })
        }
    }


    impl<I: Iterator,const D: usize> EvalColVec<I,D> {
        pub fn map_and_store<'a,F,O>(self,f: F,buffer: &'a mut BufferedColVec<I::Item,D>) -> EvalColVec<ColVecMapAndStore<'a,I,F,O,D>,D> where for<'b> F: FnMut(&'b I::Item) -> O {
            EvalColVec(ColVecMapAndStore{
                iter: self.0,
                f,
                buffer
            })
        }
    }

    //mut map_and_store
    pub struct ColVecMutMapAndStore<'a,I: Iterator,F,O,const D: usize> where for<'b> F: FnMut(&'b mut I::Item) -> O {iter: I, f: F, buffer: &'a mut BufferedColVec<I::Item,D> }

    impl<'a,I: Iterator,F,O,const D: usize> Iterator for ColVecMutMapAndStore<'a,I,F,O,D> where for<'b> F: FnMut(&'b mut I::Item) -> O {
        type Item = O;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            match self.iter.next() {
                Some(mut val) => {
                    let out = (self.f)(&mut val); 
                    match self.buffer.0.last_index {
                        None => {self.buffer.0[0] = std::mem::MaybeUninit::new(val)}
                        Some(innerer_val) => {self.buffer.0[innerer_val+1] = std::mem::MaybeUninit::new(val)}
                    }
                    Some(out)
                }
                None => None
            }
        }
    }

    impl<'a,I: Iterator,F,O,const D: usize> EvalsToColVec<D> for ColVecMutMapAndStore<'a,I,F,O,D> where for<'b> F: FnMut(&'b mut I::Item) -> O {
        type Item = O;

        #[inline]
        fn eval(self) -> ColumnVec<Vec<Self::Item>,D> {
            let mut vec = Vec::with_capacity(D);
            for output in self {
                vec.push(output)
            }
            ColumnVec(vec)
        }

        #[inline]
        fn leaky_eval_array(self) -> ColumnVec<[Self::Item; D],D> {
            ColumnVec(unsafe { utils::leaky_iterator_to_array(self) })
        }

        #[inline]
        fn eval_array(self) -> ColumnVec<[Self::Item; D],D> {
            ColumnVec(unsafe { utils::iterator_to_array(self) })
        }
    }


    impl<I: Iterator,const D: usize> EvalColVec<I,D> {
        pub fn mut_map_and_store<'a,F,O>(self,f: F,buffer: &'a mut BufferedColVec<I::Item,D>) -> EvalColVec<ColVecMutMapAndStore<'a,I,F,O,D>,D> where for<'b> F: FnMut(&'b mut I::Item) -> O {
            EvalColVec(ColVecMutMapAndStore{
                iter: self.0,
                f,
                buffer
            })
        }
    }


    //neg
    pub struct NegatedColVec<I: Iterator,const D: usize>(I) where I::Item: Neg;

    impl<I: Iterator,const D: usize> Iterator for NegatedColVec<I,D> where I::Item: Neg {
        type Item = <I::Item as Neg>::Output;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            if let Some(val) = self.0.next() {
                Some(-val)
            } else {
                None
            }
        }
    }

    impl<I: Iterator,const D: usize> EvalsToColVec<D> for NegatedColVec<I,D> where I::Item: Neg {
        type Item = <I::Item as Neg>::Output;

        #[inline]
        fn eval(self) -> ColumnVec<Vec<Self::Item>,D> {
            let mut vec = Vec::with_capacity(D);
            for output in self {
                vec.push(output)
            }
            ColumnVec(vec)
        }

        #[inline]
        fn leaky_eval_array(self) -> ColumnVec<[Self::Item; D],D> {
            ColumnVec(unsafe { utils::leaky_iterator_to_array(self) })
        }

        #[inline]
        fn eval_array(self) -> ColumnVec<[Self::Item; D],D> {
            ColumnVec(unsafe { utils::iterator_to_array(self) })
        }
    }


    impl<I: IntoIterator,const D: usize> Neg for ColumnVec<I,D> where I::Item: Neg {
        type Output = EvalColVec<NegatedColVec<I::IntoIter,D>,D>;

        #[inline]
        fn neg(self) -> Self::Output {
            EvalColVec(NegatedColVec(self.0.into_iter()))
        }
    }

    impl<'a,I: IntoIterator,const D: usize> Neg for &'a ColumnVec<I,D> where &'a I: IntoIterator, <&'a I as IntoIterator>::Item: Neg{
        type Output = EvalColVec<NegatedColVec<<&'a I as IntoIterator>::IntoIter,D>,D>;

        #[inline]
        fn neg(self) -> Self::Output {
            EvalColVec(NegatedColVec((&self.0).into_iter()))
        }
    }

    impl<I: Iterator,const D: usize> Neg for EvalColVec<I,D> where I::Item: Neg {
        type Output = EvalColVec<NegatedColVec<I,D>,D>;

        #[inline]
        fn neg(self) -> Self::Output {
            EvalColVec(NegatedColVec(self.0))
        }
    }
}

#[cfg(test)]
mod test {
    use crate::col_vec_iterators::{get_col_vec_buffer, SyncEvalsToColVec};

    use super::*;
    use super::col_vec_iterators::EvalsToColVec;

    use rand::Rng;

    #[test]
    fn basic_ops() {
        let x = ColumnVec::from([1;5]);
        let y = ColumnVec::from([2;5]);
        let z = std::ops::Add::<ColumnVec<[i32;5],5>>::add(x,y).eval();
        assert_eq!(ColumnVec::try_from(vec![3,3,3,3,3]).unwrap(),z);

        let x = ColumnVec::from([1;5]);
        let y = ColumnVec::from([2;5]);
        let z = std::ops::Sub::<ColumnVec<[i32;5],5>>::sub(x,y).eval();
        assert_eq!(ColumnVec::try_from(vec![-1,-1,-1,-1,-1]).unwrap(),z);

        let x = ColumnVec::from([2;5]);
        assert_eq!(ColumnVec::try_from(vec![6;5]).unwrap(),(x * Scalar(3)).eval());

        let x = ColumnVec::from([6;5]);
        assert_eq!(ColumnVec::try_from(vec![2;5]).unwrap(),(x / Scalar(3)).eval());

        let x = ColumnVec::from([5;5]);
        assert_eq!(ColumnVec::try_from(vec![2;5]).unwrap(),(x % Scalar(3)).eval());

        let x = ColumnVec::from([2;5]);
        let y = std::ops::Mul::<ColumnVec<[i32;5],5>>::mul(Scalar(3),x).eval();
        assert_eq!(ColumnVec::try_from(vec![6;5]).unwrap(),y);

        let x = ColumnVec::from([2;5]);
        let y = std::ops::Div::<ColumnVec<[i32;5],5>>::div(Scalar(6),x).eval();
        assert_eq!(ColumnVec::try_from(vec![3;5]).unwrap(),y);

        let x = ColumnVec::from([2;5]);
        let y = std::ops::Rem::<ColumnVec<[i32;5],5>>::rem(Scalar(3),x).eval();
        assert_eq!(ColumnVec::try_from(vec![1;5]).unwrap(),y);
    }

    #[test]
    fn stacking_ops() {
        let a = ColumnVec::from([4;5]);
        let b = ColumnVec::from([2;5]);
        let c = ColumnVec::from([1;5]);
        let test = 
            std::ops::Sub::<ColumnVec<[i32;5],5>>::sub(
                std::ops::Add::<ColumnVec<[i32; 5], 5>>::add(a,b),
                c)
            *Scalar(3)
            /Scalar(5)
            %Scalar(2);
        let test = test.eval();
        assert_eq!(ColumnVec::try_from(vec![1;5]).unwrap(),test);
    }

    #[test]
    fn assign_ops() {
        let mut x: ColumnVec<[i32; 5], 5> = ColumnVec::from([4; 5]);
        std::ops::AddAssign::<ColumnVec<[i32; 5],5>>::add_assign(&mut x,ColumnVec::from([3; 5]));
        assert_eq!(ColumnVec::from([7; 5]),x);
        std::ops::SubAssign::<ColumnVec<[i32; 5],5>>::sub_assign(&mut x, ColumnVec::from([5; 5]));
        assert_eq!(ColumnVec::from([2; 5]),x);
        x *= Scalar(4);
        assert_eq!(ColumnVec::from([8; 5]),x);
        x /= Scalar(2);
        assert_eq!(ColumnVec::from([4; 5]),x);
        x %= Scalar(3);
        assert_eq!(ColumnVec::from([1; 5]),x);
    }

    #[test]
    fn custom_ops() {
        let x = ColumnVec::from([3; 5]);
        assert_eq!(ColumnVec::try_from(vec![-3; 5]).unwrap(),(-x).eval());

        let x = ColumnVec::from([3; 5]);
        let y = ColumnVec::from([2; 5]);
        assert_eq!(Scalar(30),Dot::<ColumnVec<[i32; 5],5>>::dot(x,y));

        let x = ColumnVec::from([&3; 5]);
        assert_eq!(ColumnVec::try_from(vec![3; 5]).unwrap(),x.cloned().eval());

        let x = ColumnVec::from([&3; 5]);
        assert_eq!(ColumnVec::try_from(vec![3; 5]).unwrap(),x.copied().eval());

        let x = ColumnVec::from([3; 5]);
        assert_eq!(ColumnVec::try_from(vec![4; 5]).unwrap(),x.map(|val| {val + 1}).eval())
    }

    #[test]
    fn duplicate_test() {
        let (x,y) = ColumnVec::from([1,2,3,4,5]).duplicate();

        assert_eq!(ColumnVec::try_from(vec![2,4,6,8,10]).unwrap(),std::ops::Add::<col_vec_iterators::EvalColVec<col_vec_iterators::SecondDuplicateColVec<std::array::IntoIter<i32,5>,5>,5>>::add(x,y).eval());
    }

    #[test]
    fn unchecked_duplicate_test() {
        let (x,y) = unsafe { ColumnVec::from([1,2,3,4,5]).unchecked_duplicate() };

        assert_eq!(ColumnVec::try_from(vec![2,4,6,8,10]).unwrap(),std::ops::Add::<col_vec_iterators::EvalColVec<col_vec_iterators::SecondUncheckedDuplicateColVec<std::array::IntoIter<i32,5>,5>,5>>::add(x,y).eval());
    }

    #[test]
    fn synchronize_test() {
        let (a,b) = (ColumnVec::from([1,2,3,4,5,6,7,8,9,10]) * Scalar(2)).duplicate();
        let (c,d) = (a*Scalar(2),b*Scalar(3)).sync_eval();
        assert_eq!(ColumnVec::try_from(vec![4,8,12,16,20,24,28,32,36,40]).unwrap(),c);
        assert_eq!(ColumnVec::try_from(vec![6,12,18,24,30,36,42,48,54,60]).unwrap(),d);

        let (a,b) = (ColumnVec::from([1,2,3,4,5,6,7,8,9,10]) * Scalar(2)).duplicate();
        let (c,d) = (a*Scalar(2),b*Scalar(3)).sync_leaky_eval_array();
        assert_eq!(ColumnVec::from([4,8,12,16,20,24,28,32,36,40]),c);
        assert_eq!(ColumnVec::from([6,12,18,24,30,36,42,48,54,60]),d);

        let (a,b) = (ColumnVec::from([1,2,3,4,5,6,7,8,9,10]) * Scalar(2)).duplicate();
        let (c,d) = (a*Scalar(2),b*Scalar(3)).sync_eval_array();
        assert_eq!(ColumnVec::from([4,8,12,16,20,24,28,32,36,40]),c);
        assert_eq!(ColumnVec::from([6,12,18,24,30,36,42,48,54,60]),d);
    }

    //Doesnt check leaky-ness
    #[test]
    fn eval_array_test() {
        let vec = ColumnVec::from([1,2,3,4,5,6,7,8,9,10]);
        assert_eq!(ColumnVec::from([2,4,6,8,10,12,14,16,18,20]),(vec*Scalar(2)).eval_array());

        let vec = ColumnVec::from([1,2,3,4,5,6,7,8,9,10]);
        assert_eq!(ColumnVec::from([2,4,6,8,10,12,14,16,18,20]),(vec*Scalar(2)).leaky_eval_array());
    }

    #[test]
    fn clone_and_copy_to_buffer_test() {
        let mut buf = get_col_vec_buffer();
        let vec = ColumnVec::from([1,2,3,4,5,6,7,8,9,10]);
        assert_eq!(ColumnVec::from([4,8,12,16,20,24,28,32,36,40]),((vec * Scalar(2)).clone_to_buffer(&mut buf) * Scalar(2)).eval_array());
        assert_eq!(ColumnVec::from([2,4,6,8,10,12,14,16,18,20]),buf.eval_array());

        let mut buf = get_col_vec_buffer();
        let vec = ColumnVec::from([1,2,3,4,5,6,7,8,9,10]);
        assert_eq!(ColumnVec::from([4,8,12,16,20,24,28,32,36,40]),((vec * Scalar(2)).copy_to_buffer(&mut buf) * Scalar(2)).eval_array());
        assert_eq!(ColumnVec::from([2,4,6,8,10,12,14,16,18,20]),buf.eval_array());
    }


    #[test]
    fn duplicate_stress_test() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000000 {
            let original: ColumnVec<[u32; 10], 10> = ColumnVec::from([
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000)
            ]);
            let (clone_a,clone_b) = original.clone().duplicate();
            assert_eq!((original*Scalar(2)).eval(),std::ops::Add::<col_vec_iterators::EvalColVec<col_vec_iterators::SecondDuplicateColVec<std::array::IntoIter<u32,10>,10>,10>>::add(clone_a,clone_b).eval());
        }
    }

    #[test]
    fn unchecked_duplicate_stress_test() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000000 {
            let original: ColumnVec<[u32; 10], 10> = ColumnVec::from([
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000)
            ]);
            let (clone_a,clone_b) = unsafe { original.clone().unchecked_duplicate() };
            assert_eq!((original*Scalar(2)).eval(),std::ops::Add::<col_vec_iterators::EvalColVec<col_vec_iterators::SecondUncheckedDuplicateColVec<std::array::IntoIter<u32,10>,10>,10>>::add(clone_a,clone_b).eval());
        }
    }

    #[test]
    fn eval_array_stress_test() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000000 {
            let vec_nums = (
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000)
            );
            let vec1 = ColumnVec::from(<[u32; 10]>::from(vec_nums));
            let vec2 = ColumnVec::from([vec_nums.0 * 2,vec_nums.1 * 2,vec_nums.2 * 2,vec_nums.3 * 2,vec_nums.4 * 2,vec_nums.5 * 2,vec_nums.6 * 2,vec_nums.7 * 2,vec_nums.8 * 2,vec_nums.9 * 2]);
            assert_eq!(vec2,(vec1 * Scalar(2)).eval_array());
        }
    }

    #[test]
    fn leaky_eval_array_stress_test() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000000 {
            let vec_nums = (
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000),
                rng.gen_range(1..1000000000)
            );
            let vec1 = ColumnVec::from(<[u32; 10]>::from(vec_nums));
            let vec2 = ColumnVec::from([vec_nums.0 * 2,vec_nums.1 * 2,vec_nums.2 * 2,vec_nums.3 * 2,vec_nums.4 * 2,vec_nums.5 * 2,vec_nums.6 * 2,vec_nums.7 * 2,vec_nums.8 * 2,vec_nums.9 * 2]);
            assert_eq!(vec2,(vec1 * Scalar(2)).leaky_eval_array());
        }
    }

    #[test]
    fn synchronize_stress_test() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000000 {
            let original: ColumnVec<[u32; 10], 10> = ColumnVec::from([
                rng.gen_range(1..100000000),
                rng.gen_range(1..100000000),
                rng.gen_range(1..100000000),
                rng.gen_range(1..100000000),
                rng.gen_range(1..100000000),
                rng.gen_range(1..100000000),
                rng.gen_range(1..100000000),
                rng.gen_range(1..100000000),
                rng.gen_range(1..100000000),
                rng.gen_range(1..100000000)
            ]);
            let copy1 = original.clone();
            let copy2 = original.clone();
            let copy3 = original.clone();
            let copy4 = original.clone();
            let copy5 = original.clone();
            let (dupe1,dupe2) = original.duplicate();
            let (dupe2,dupe3) = dupe2.duplicate();
            let (dupe3,dupe4) = dupe3.duplicate();
            let (dupe4,dupe5) = dupe4.duplicate();
            assert_eq!(
                (
                    (copy1 * Scalar(2)).eval_array(),
                    (copy2 * Scalar(3)).eval_array(),
                    (copy3 * Scalar(4)).eval_array(),
                    (copy4 * Scalar(5)).eval_array(),
                    (copy5 * Scalar(6)).eval_array()
                ),
                (
                    dupe1 * Scalar(2),
                    dupe2 * Scalar(3),
                    dupe3 * Scalar(4),
                    dupe4 * Scalar(5),
                    dupe5 * Scalar(6)
                ).sync_eval_array()
            );
        }
    }

    #[ignore]
    #[test]
    fn iter_copy_to_slice_speed_test() {
        let mut custom = std::time::Duration::new(0,0);
        let mut builtin = std::time::Duration::new(0,0);
        for _ in 0..1000000000 {
            let x: &[i32] = &[20; 1000];
            let y: &mut [i32] = &mut [0; 1000];
            let now = std::time::Instant::now();
            let x = x.into_iter();
            for (x_val,y_val) in x.zip(y.iter_mut()) {
                *y_val = *x_val;
            }
            let next_time = now.elapsed();
            custom += next_time;

            
            let x: &[i32] = &[20; 1000];
            let y: &mut [i32] = &mut [0; 1000];
            let now = std::time::Instant::now();
            y.copy_from_slice(x);
            let next_time = now.elapsed();
            builtin += next_time;
        }
        println!("{}",custom.as_nanos());
        println!("{}",builtin.as_nanos());
    }
}