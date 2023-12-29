#![allow(clippy::type_complexity)]
#![allow(clippy::into_iter_on_ref)]
#![allow(clippy::useless_conversion)]

#[allow(clippy::missing_safety_doc)]
pub(crate) mod utils {
    use std::mem::MaybeUninit;

    pub struct IterInitArray<T,const D: usize>{pub array: [MaybeUninit<T>; D], pub last_index: usize}

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
            // FIXME: pointers are only used because transmute thinks size_of::<T> != size_of::<T>, only true is T is a DST, which it cant be
            let initialized_arr;
            unsafe {
                initialized_arr = std::ptr::read(std::mem::transmute(&self.array as *const [MaybeUninit<T>; D]));
                std::mem::forget(self);
            }
            initialized_arr
        }

        #[inline]
        pub fn new() -> Self {
            IterInitArray{
                array: unsafe {MaybeUninit::uninit().assume_init()},
                last_index: 0
            }
        }

        #[inline]
        pub fn assign_next(&mut self, val: T) {
            let index = self.last_index;
            self[index] = std::mem::MaybeUninit::new(val);
        }
    }

    //Also both of these are actually unsafe
    //no check for if the value is initialized
    impl<T,const D: usize> std::ops::Index<usize> for IterInitArray<T,D> where [MaybeUninit<T>; D]: std::ops::Index<usize>, usize: std::slice::SliceIndex<[MaybeUninit<T>]> {
        type Output = <usize as std::slice::SliceIndex<[MaybeUninit<T>]>>::Output;

        #[inline]
        fn index(&self, index: usize) -> &Self::Output {
            unsafe {self.array.get_unchecked(index)}
        }
    }

    //To ensure Safety, any val recieved must be written to with a valid value
    impl<T,const D: usize> std::ops::IndexMut<usize> for IterInitArray<T,D> where [MaybeUninit<T>; D]: std::ops::Index<usize>, usize: std::slice::SliceIndex<[MaybeUninit<T>]> {
        #[inline]
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            let output = unsafe {self.array.get_unchecked_mut(index)};
            self.last_index += 1;
            output
        }
    }

    impl<T,const D: usize> std::ops::Drop for IterInitArray<T,D> {
        #[inline]
        fn drop(&mut self) {
            for i in 0..self.last_index {
                unsafe { self.array[i].assume_init_drop() };
            }
        }
    }

    #[inline]
    pub unsafe fn iterator_to_array<I: Iterator,const D: usize>(iter: I) -> [I::Item; D] {
        let mut arr : IterInitArray<I::Item, D>= IterInitArray::new();

        for val in iter {
            arr.assign_next(val)
        }

        arr.assume_init()
    }


    pub struct NoneIter<T>(std::marker::PhantomData<T>);

    #[allow(dead_code)]
    impl<T> NoneIter<T> {
        #[inline]
        fn new() -> Self {
            NoneIter(std::marker::PhantomData)
        }
    }

    impl<T> Iterator for NoneIter<T> {
        type Item = T;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            None
        }
    }
}


#[derive(PartialEq,Debug)]
#[derive(Clone,Copy)]
#[repr(transparent)]
pub struct Scalar<T>(pub T);

//CHECK
impl<T> Scalar<T> {
    #[inline]
    pub fn new(val: T) -> Self {
        Scalar(val)
    }
}

impl<T> std::ops::Deref for Scalar<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for Scalar<T> {
    
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
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
pub struct ColumnVec<T,const D: usize>([T; D]);

impl<T,const D: usize> ColumnVec<T,D> {
    #[inline]
    pub fn unwrap(self) -> [T; D] {
        self.0
    }

    #[inline]
    pub fn transpose(self) -> RowVec<T,D> {
        RowVec(self.0)
    }
}

impl<C,T,const D: usize> From<C> for ColumnVec<T,D> where [T; D]: From<C> {
    #[inline]
    fn from(value: C) -> Self {
        ColumnVec(<[T; D]>::from(value))
    }
}


impl<T,const D: usize> std::ops::Deref for ColumnVec<T,D> {
    type Target = [T;D];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T,const D: usize> std::ops::DerefMut for ColumnVec<T,D> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

//CHECK
impl<Idx,T,const D: usize> std::ops::Index<Idx> for ColumnVec<T,D> 
where [T; D]: std::ops::Index<Idx>, <[T; D] as std::ops::Index<Idx>>::Output: Sized {
    type Output = Scalar<<[T; D] as std::ops::Index<Idx>>::Output>;
    
    #[inline]
    fn index(&self, index: Idx) -> &Self::Output {
        //SAFETY: scalar is repr(transparant) so the representaion of Scalar<T> 
        //is identical to T and as such &Scalar<T> is identical to &T or so I
        //think. DOUBLE CHECK THIS ZACH, double checked, SHOULD BE CHECKED MORE
        unsafe { std::mem::transmute(self.0.index(index)) }
    }
}

#[derive(Clone)]
pub struct ColVecIter<I: Iterator>(I);

impl<I: Iterator> Iterator for ColVecIter<I> {
    type Item = Scalar<I::Item>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(Scalar)
    }
}

impl<T,const D: usize> IntoIterator for ColumnVec<T,D> {
    type IntoIter = ColVecIter<<[T; D] as IntoIterator>::IntoIter>;
    type Item = Scalar<<[T; D] as IntoIterator>::Item>;
    
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        ColVecIter(self.0.into_iter())
    }
}

impl<'a,T,const D: usize> IntoIterator for &'a ColumnVec<T,D> {
    type IntoIter = ColVecIter<<&'a [T; D] as IntoIterator>::IntoIter>;
    type Item = Scalar<<&'a [T; D] as IntoIterator>::Item>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        ColVecIter((&self.0).into_iter())
    }
}

impl<'a,T,const D: usize> IntoIterator for &'a mut ColumnVec<T,D> {
    type IntoIter = ColVecIter<<&'a mut [T; D] as IntoIterator>::IntoIter>;
    type Item = Scalar<<&'a mut [T; D] as IntoIterator>::Item>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        ColVecIter((&mut self.0).into_iter())
    }
}


pub struct RowVec<T,const D: usize>([T; D]);


pub trait Dot<T> {
    type Output;

    fn dot(self,rhs: T) -> Self::Output;
}

pub trait Zip<T> {
    type Output;

    fn zip(self,other: T) -> Self::Output;
}

pub trait Unzippable<T> {
    fn unzip(data: T) -> Self;
}

pub mod col_vec_iterators {
    use std::ops::*;
    use super::*;

    #[must_use]
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

        fn eval(self) -> ColumnVec<Self::Item,D>;
    }

    impl<T: EvalsToColVec<D>,const D: usize> EvalsToColVec<D> for EvalColVec<T,D> {
        type Item = T::Item;

        #[inline]
        fn eval(self) -> ColumnVec<Self::Item,D> {
            self.0.eval()
        }
    }

    
    pub trait SyncEvalsToColVec {
        type EvalArrayOut;
    
        fn sync_leaky_eval(self) -> Self::EvalArrayOut;
        fn sync_eval(self) -> Self::EvalArrayOut;
    }

    #[macro_export]
    macro_rules! impl_sync_evals_to_col_vec {
        ($($i:ident)*,$($i_buf:ident)*) => {
            impl<$($i: Iterator),*,const D: usize> SyncEvalsToColVec for ($(EvalColVec<$i,D>),*) {
                type EvalArrayOut = ($(ColumnVec<$i::Item,D>),*);
    
                #[allow(non_snake_case)]
                #[inline]
                fn sync_leaky_eval(self) -> Self::EvalArrayOut {
                    let ($(EvalColVec(mut $i)),*) = self;
                    $(
                        let mut $i_buf: [std::mem::MaybeUninit<$i::Item>; D] = unsafe {std::mem::MaybeUninit::uninit().assume_init()};
                    )*
                    for i in 0..D {
                        $(
                            $i_buf[i] = std::mem::MaybeUninit::new($i.next().expect("math_vector internal_error: The first D elements in a EvalColVec must be valid"));
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
                fn sync_eval(self) -> Self::EvalArrayOut {
                    let ($(EvalColVec(mut $i)),*) = self;
                    $(
                        let mut $i_buf: utils::IterInitArray<$i::Item, D>= utils::IterInitArray{array: unsafe {std::mem::MaybeUninit::uninit().assume_init()},last_index: 0};
                    )*

                    for i in 0..D {
                        $(
                            $i_buf[i] = std::mem::MaybeUninit::new($i.next().expect("math_vector internal_error: The first D elements in a EvalColVec must be valid"));
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
                fn eval(self) -> ColumnVec<Self::Item,D> {
                    ColumnVec(unsafe { utils::iterator_to_array(self) })
                }
            }
        };
        ($t:ty,2,$ti:ty,$tr:path) => {
            impl<I1: Iterator,I2: Iterator,const D: usize> EvalsToColVec<D> for $t where $ti: $tr {
                type Item = <$t as Iterator>::Item;

                #[inline]
                fn eval(self) -> ColumnVec<Self::Item,D> {
                    ColumnVec(unsafe { utils::iterator_to_array(self) })
                }
            }
        };
    }

    macro_rules! overload_given_operator_for_vec {
        ($Op:ident,$OpF:ident,$OpI:ident) => {
            impl<T1,T2,const D: usize> $Op<ColumnVec<T2,D>> for ColumnVec<T1,D> where <[T1; D] as IntoIterator>::Item: $Op<<[T2; D] as IntoIterator>::Item> {
                type Output = EvalColVec<$OpI<<[T1; D] as IntoIterator>::IntoIter,<[T2; D] as IntoIterator>::IntoIter,D>,D>;

                #[inline]
                fn $OpF(self,other: ColumnVec<T2,D>) -> Self::Output {
                    EvalColVec($OpI(
                        self.0.into_iter(),
                        other.0.into_iter()
                    ))
                }
            }

            impl<'a,T1,T2,const D: usize> $Op<ColumnVec<T2,D>> for &'a ColumnVec<T1,D> 
            where <&'a [T1; D] as IntoIterator>::Item: $Op<<[T2; D] as IntoIterator>::Item> {
                type Output = EvalColVec<$OpI<<&'a [T1; D] as IntoIterator>::IntoIter,<[T2; D] as IntoIterator>::IntoIter,D>,D>;

                #[inline]
                fn $OpF(self,other: ColumnVec<T2,D>) -> Self::Output {
                    EvalColVec($OpI(
                        (&self.0).into_iter(),
                        other.0.into_iter()
                    ))
                }
            }

            impl<'a,T1,T2,const D: usize> $Op<&'a ColumnVec<T2,D>> for ColumnVec<T1,D> 
            where <[T1; D] as IntoIterator>::Item: $Op<<&'a [T2; D] as IntoIterator>::Item> {
                type Output = EvalColVec<$OpI<<[T1; D] as IntoIterator>::IntoIter,<&'a [T2; D] as IntoIterator>::IntoIter,D>,D>;

                #[inline]
                fn $OpF(self,other: &'a ColumnVec<T2,D>) -> Self::Output {
                    EvalColVec($OpI(
                        self.0.into_iter(),
                        (&other.0).into_iter()
                    ))
                }
            }

            impl<'a,T1,T2,const D: usize> $Op<&'a ColumnVec<T2,D>> for &'a ColumnVec<T1,D> 
            where <&'a [T1; D] as IntoIterator>::Item: $Op<<&'a [T2; D] as IntoIterator>::Item> {
                type Output = EvalColVec<$OpI<<&'a [T1; D] as IntoIterator>::IntoIter,<&'a [T2; D] as IntoIterator>::IntoIter,D>,D>;

                #[inline]
                fn $OpF(self,other: &'a ColumnVec<T2,D>) -> Self::Output {
                    EvalColVec($OpI(
                        (&self.0).into_iter(),
                        (&other.0).into_iter()
                    ))
                }
            }
        

            impl<I1: Iterator,T2,const D: usize> $Op<ColumnVec<T2,D>> for EvalColVec<I1,D> where I1::Item: $Op<<[T2; D] as IntoIterator>::Item> {
                type Output = EvalColVec<$OpI<I1,<[T2; D] as IntoIterator>::IntoIter,D>,D>;

                #[inline]
                fn $OpF(self,other: ColumnVec<T2,D>) -> Self::Output {
                    EvalColVec($OpI(
                        self.0,
                        other.0.into_iter()
                    ))
                }
            }

            impl<'a,I1: Iterator,T2,const D: usize> $Op<&'a ColumnVec<T2,D>> for EvalColVec<I1,D> 
            where I1::Item: $Op<<&'a [T2; D] as IntoIterator>::Item> {
                type Output = EvalColVec<$OpI<I1,<&'a [T2; D] as IntoIterator>::IntoIter,D>,D>;

                #[inline]
                fn $OpF(self,other: &'a ColumnVec<T2,D>) -> Self::Output {
                    EvalColVec($OpI(
                        self.0,
                        (&other.0).into_iter()
                    ))
                }
            }

            impl<T1,I2: Iterator,const D: usize> $Op<EvalColVec<I2,D>> for ColumnVec<T1,D> where <[T1; D] as IntoIterator>::Item: $Op<I2::Item> {
                type Output = EvalColVec<$OpI<<[T1; D] as IntoIterator>::IntoIter,I2,D>,D>;

                #[inline]
                fn $OpF(self,other: EvalColVec<I2,D>) -> Self::Output {
                    EvalColVec($OpI(
                        self.0.into_iter(),
                        other.0
                    ))
                }
            }

            impl<'a,T1,I2: Iterator,const D: usize> $Op<EvalColVec<I2,D>> for &'a ColumnVec<T1,D> 
            where <&'a [T1; D] as IntoIterator>::Item: $Op<I2::Item> {
                type Output = EvalColVec<$OpI<<&'a [T1; D] as IntoIterator>::IntoIter,I2,D>,D>;

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
            impl<T,S: Copy,const D: usize> $Op<Scalar<S>> for ColumnVec<T,D> where <[T; D] as IntoIterator>::Item: $Op<S> {
                type Output = EvalColVec<$OpI<<[T; D] as IntoIterator>::IntoIter,S,D>,D>;

                #[inline]
                fn $OpF(self,rhs: Scalar<S>) -> Self::Output {
                    EvalColVec($OpI(
                        self.0.into_iter(),
                        rhs.0
                    ))
                }
            }

            impl<'a,T,S: Copy,const D: usize> $Op<Scalar<S>> for &'a ColumnVec<T,D> 
            where <&'a [T; D] as IntoIterator>::Item: $Op<S> {
                type Output = EvalColVec<$OpI<<&'a [T; D] as IntoIterator>::IntoIter,S,D>,D>;

                #[inline]
                fn $OpF(self,rhs: Scalar<S>) -> Self::Output {
                    EvalColVec($OpI(
                        (&self.0).into_iter(),
                        rhs.0
                    ))
                }
            }

            impl<'a,T,S: Copy,const D: usize> $Op<&'a Scalar<S>> for ColumnVec<T,D> where <[T; D] as IntoIterator>::Item: $Op<&'a S> {
                type Output = EvalColVec<$OpI<<[T; D] as IntoIterator>::IntoIter,&'a S,D>,D>;

                #[inline]
                fn $OpF(self,rhs: &'a Scalar<S>) -> Self::Output {
                    EvalColVec($OpI(
                        self.0.into_iter(),
                        &rhs.0
                    ))
                }
            }

            impl<'a,T,S: Copy,const D: usize> $Op<&'a Scalar<S>> for &'a ColumnVec<T,D> 
            where <&'a [T; D] as IntoIterator>::Item: $Op<&'a S> {
                type Output = EvalColVec<$OpI<<&'a [T; D] as IntoIterator>::IntoIter,&'a S,D>,D>;

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
            impl<T,S: Copy,const D: usize> $Op<ColumnVec<T,D>> for Scalar<S> where S: $Op<<[T; D] as IntoIterator>::Item> {
                type Output = EvalColVec<$OpI<S,<[T; D] as IntoIterator>::IntoIter,D>,D>;

                #[inline]
                fn $OpF(self,rhs: ColumnVec<T,D>) -> Self::Output {
                    EvalColVec($OpI(
                        rhs.0.into_iter(),
                        self.0
                    ))
                }
            }

            impl<'a,T,S: Copy,const D: usize> $Op<ColumnVec<T,D>> for &'a Scalar<S> where &'a S: $Op<<[T; D] as IntoIterator>::Item> {
                type Output = EvalColVec<$OpI<&'a S,<[T; D] as IntoIterator>::IntoIter,D>,D>;

                #[inline]
                fn $OpF(self,rhs: ColumnVec<T,D>) -> Self::Output {
                    EvalColVec($OpI(
                        rhs.0.into_iter(),
                        &self.0
                    ))
                }
            }

            impl<'a,T,S: Copy,const D: usize> $Op<&'a ColumnVec<T,D>> for Scalar<S> 
            where S: $Op<<&'a [T; D] as IntoIterator>::Item> {
                type Output = EvalColVec<$OpI<S,<&'a [T; D] as IntoIterator>::IntoIter,D>,D>;

                #[inline]
                fn $OpF(self,rhs: &'a ColumnVec<T,D>) -> Self::Output {
                    EvalColVec($OpI(
                        (&rhs.0).into_iter(),
                        self.0
                    ))
                }
            }

            impl<'a,T,S: Copy,const D: usize> $Op<&'a ColumnVec<T,D>> for &'a Scalar<S> 
            where &'a S: $Op<<&'a [T; D] as IntoIterator>::Item> {
                type Output = EvalColVec<$OpI<&'a S,<&'a [T; D] as IntoIterator>::IntoIter,D>,D>;

                #[inline]
                fn $OpF(self,rhs: &'a ColumnVec<T,D>) -> Self::Output {
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
            impl<T1,T2,O,const D: usize> $Op<ColumnVec<T2,D>> for ColumnVec<T1,D> 
            where for<'a> &'a mut [T1; D]: IntoIterator<Item = &'a mut O>,O: $Op<<[T2; D] as IntoIterator>::Item> {
                #[inline]
                fn $OpF(&mut self,rhs: ColumnVec<T2,D>) {
                    for (self_val,other_val) in (&mut self.0).into_iter().zip(rhs.0.into_iter()) {
                        (self_val).$OpF(other_val)
                    }
                }
            }

            impl<'b,T1,T2,O,const D: usize> $Op<&'b ColumnVec<T2,D>> for ColumnVec<T1,D> 
            where for<'a> &'a mut [T1; D]: IntoIterator<Item = &'a mut O>,O: $Op<<&'b [T2; D] as IntoIterator>::Item> {
                #[inline]
                fn $OpF(&mut self,rhs: &'b ColumnVec<T2,D>) {
                    for (self_val,other_val) in (&mut self.0).into_iter().zip((&rhs.0).into_iter()) {
                        (self_val).$OpF(other_val)
                    }
                }
            }

            impl<T1,I2: Iterator,O,const D: usize> $Op<EvalColVec<I2,D>> for ColumnVec<T1,D> 
            where for<'a> &'a mut [T1; D]: IntoIterator<Item = &'a mut O>,O: $Op<I2::Item> {
                #[inline]
                fn $OpF(&mut self,rhs: EvalColVec<I2,D>) {
                    for (self_val,other_val) in (&mut self.0).into_iter().zip(rhs.0) {
                        (self_val).$OpF(other_val)
                    }
                }
            }
        };
        ($Op:ident,$OpF:ident,1) => {
            impl<T,O,S: Copy,const D: usize> $Op<Scalar<S>> for ColumnVec<T,D> 
            where for<'a> &'a mut [T; D]: IntoIterator<Item = &'a mut O>,for<'a> O: $Op<S> {
                #[inline]
                fn $OpF(&mut self,rhs: Scalar<S>) {
                    for val in (&mut self.0).into_iter() {
                        (val).$OpF(rhs.0)
                    }
                }
            }

            impl<'b,T,O,S: Copy,const D: usize> $Op<&'b Scalar<S>> for ColumnVec<T,D> 
            where for<'a> &'a mut [T; D]: IntoIterator<Item = &'a mut O>,for<'a> O: $Op<&'b S> {
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




    //comp_mul
    pub struct ColVecCompMul<I1: Iterator,I2: Iterator,const D: usize>(I1,I2) where I1::Item: Mul<I2::Item>;

    impl<I1: Iterator,I2: Iterator,const D: usize> Iterator for ColVecCompMul<I1,I2,D> where I1::Item: Mul<I2::Item> {
        type Item = <I1::Item as Mul<I2::Item>>::Output;

        fn next(&mut self) -> Option<Self::Item> {
            match (self.0.next(),self.1.next()) {
                (Some(x1),Some(x2)) => Some(x1*x2),
                _ => None
            }
        }
    }

    impl<I1: Iterator,I2: Iterator,const D: usize> EvalsToColVec<D> for ColVecCompMul<I1,I2,D> where I1::Item: Mul<I2::Item> {
        type Item = <I1::Item as Mul<I2::Item>>::Output;

        fn eval(self) -> ColumnVec<Self::Item,D> {
            ColumnVec(unsafe { utils::iterator_to_array(self) })
        }
    }

    pub struct ColVecCompDiv<I1: Iterator,I2: Iterator,const D: usize>(I1,I2) where I1::Item: Div<I2::Item>;

    impl<I1: Iterator,I2: Iterator,const D: usize> Iterator for ColVecCompDiv<I1,I2,D> where I1::Item: Div<I2::Item> {
        type Item = <I1::Item as Div<I2::Item>>::Output;

        fn next(&mut self) -> Option<Self::Item> {
            match (self.0.next(),self.1.next()) {
                (Some(x1),Some(x2)) => Some(x1/x2),
                _ => None
            }
        }
    }
    
    impl<I1: Iterator,I2: Iterator,const D: usize> EvalsToColVec<D> for ColVecCompDiv<I1,I2,D> where I1::Item: Div<I2::Item> {
        type Item = <I1::Item as Div<I2::Item>>::Output;

        fn eval(self) -> ColumnVec<Self::Item,D> {
            ColumnVec(unsafe { utils::iterator_to_array(self) })
        }
    }

    //dot product
    impl<T1,T2,O: AddAssign<O> ,const D: usize> Dot<ColumnVec<T2,D>> for ColumnVec<T1,D>
    where <[T1; D] as IntoIterator>::Item: Mul<<[T2; D] as IntoIterator>::Item,Output = O>  {
        type Output = Scalar<O>;

        #[inline]
        fn dot(self,rhs: ColumnVec<T2,D>) -> Self::Output {
            let mut iter = self.0.into_iter().zip(rhs.0.into_iter());
            let (first_self_val,first_other_val) = iter.next().expect("The ColumnVecs should have a element in them both");
            let mut output = first_self_val * first_other_val;
            for (self_val,other_val) in iter {
                output += self_val * other_val;
            }
            Scalar(output)
        }
    }

    impl<'a,T1,T2,O: AddAssign<O> ,const D: usize> Dot<ColumnVec<T2,D>> for &'a ColumnVec<T1,D>
    where <&'a [T1; D] as IntoIterator>::Item: Mul<<[T2; D] as IntoIterator>::Item,Output = O>  {
        type Output = Scalar<O>;

        #[inline]
        fn dot(self,rhs: ColumnVec<T2,D>) -> Self::Output {
            let mut iter = (&self.0).into_iter().zip(rhs.0.into_iter());
            let (first_self_val,first_other_val) = iter.next().expect("The ColumnVecs should have a element in them both");
            let mut output = first_self_val * first_other_val;
            for (self_val,other_val) in iter {
                output += self_val * other_val;
            }
            Scalar(output)
        }
    }

    impl<'a,T1,T2,O: AddAssign<O> ,const D: usize> Dot<&'a ColumnVec<T2,D>> for ColumnVec<T1,D>
    where <[T1; D] as IntoIterator>::Item: Mul<<&'a [T2; D] as IntoIterator>::Item,Output = O>  {
        type Output = Scalar<O>;

        #[inline]
        fn dot(self,rhs: &'a ColumnVec<T2,D>) -> Self::Output {
            let mut iter = self.0.into_iter().zip((&rhs.0).into_iter());
            let (first_self_val,first_other_val) = iter.next().expect("The ColumnVecs should have a element in them both");
            let mut output = first_self_val * first_other_val;
            for (self_val,other_val) in iter {
                output += self_val * other_val;
            }
            Scalar(output)
        }
    }

    impl<'a,T1,T2,O: AddAssign<O> ,const D: usize> Dot<&'a ColumnVec<T2,D>> for &'a ColumnVec<T1,D>
    where <&'a [T1; D] as IntoIterator>::Item: Mul<<&'a [T2; D] as IntoIterator>::Item,Output = O>  {
        type Output = Scalar<O>;

        #[inline]
        fn dot(self,rhs: &'a ColumnVec<T2,D>) -> Self::Output {
            let mut iter = (&self.0).into_iter().zip((&rhs.0).into_iter());
            let (first_self_val,first_other_val) = iter.next().expect("The ColumnVecs should have a element in them both");
            let mut output = first_self_val * first_other_val;
            for (self_val,other_val) in iter {
                output += self_val * other_val;
            }
            Scalar(output)
        }
    }

    
    impl<I1: Iterator,T2,O: AddAssign<O> ,const D: usize> Dot<ColumnVec<T2,D>> for EvalColVec<I1,D>
    where I1::Item: Mul<<[T2; D] as IntoIterator>::Item,Output = O>  {
        type Output = Scalar<O>;

        #[inline]
        fn dot(self,rhs: ColumnVec<T2,D>) -> Self::Output {
            let mut iter = self.0.zip(rhs.0.into_iter());
            let (first_self_val,first_other_val) = iter.next().expect("The ColumnVecs should have a element in them both");
            let mut output = first_self_val * first_other_val;
            for (self_val,other_val) in iter {
                output += self_val * other_val;
            }
            Scalar(output)
        }
    }

    impl<'a,I1: Iterator,T2,O: AddAssign<O> ,const D: usize> Dot<&'a ColumnVec<T2,D>> for EvalColVec<I1,D>
    where I1::Item: Mul<<&'a [T2; D] as IntoIterator>::Item,Output = O>  {
        type Output = Scalar<O>;

        #[inline]
        fn dot(self,rhs: &'a ColumnVec<T2,D>) -> Self::Output {
            let mut iter = self.0.zip((&rhs.0).into_iter());
            let (first_self_val,first_other_val) = iter.next().expect("The ColumnVecs should have a element in them both");
            let mut output = first_self_val * first_other_val;
            for (self_val,other_val) in iter {
                output += self_val * other_val;
            }
            Scalar(output)
        }
    }

    impl<T1,I2: Iterator,O: AddAssign<O> ,const D: usize> Dot<EvalColVec<I2,D>> for ColumnVec<T1,D>
    where <[T1; D] as IntoIterator>::Item: Mul<I2::Item,Output = O>  {
        type Output = Scalar<O>;

        #[inline]
        fn dot(self,rhs: EvalColVec<I2,D>) -> Self::Output {
            let mut iter = self.0.into_iter().zip(rhs.0);
            let (first_self_val,first_other_val) = iter.next().expect("The ColumnVecs should have a element in them both");
            let mut output = first_self_val * first_other_val;
            for (self_val,other_val) in iter {
                output += self_val * other_val;
            }
            Scalar(output)
        }
    }

    impl<'a,T1,I2: Iterator,O: AddAssign<O> ,const D: usize> Dot<EvalColVec<I2,D>> for &'a ColumnVec<T1,D>
    where <&'a [T1; D] as IntoIterator>::Item: Mul<I2::Item,Output = O>  {
        type Output = Scalar<O>;

        #[inline]
        fn dot(self,rhs: EvalColVec<I2,D>) -> Self::Output {
            let mut iter = (&self.0).into_iter().zip(rhs.0);
            let (first_self_val,first_other_val) = iter.next().expect("The ColumnVecs should have a element in them both");
            let mut output = first_self_val * first_other_val;
            for (self_val,other_val) in iter {
                output += self_val * other_val;
            }
            Scalar(output)
        }
    }


    impl<I1: Iterator,I2: Iterator,O: AddAssign<O> ,const D: usize> Dot<EvalColVec<I2,D>> for EvalColVec<I1,D>
    where I1::Item: Mul<I2::Item,Output = O>  {
        type Output = Scalar<O>;

        #[inline]
        fn dot(self,rhs: EvalColVec<I2,D>) -> Self::Output {
            let mut iter = self.0.zip(rhs.0);
            let (first_self_val,first_other_val) = iter.next().expect("The ColumnVecs should have a element in them both");
            let mut output = first_self_val * first_other_val;
            for (self_val,other_val) in iter {
                output += self_val * other_val;
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
        fn eval(self) -> ColumnVec<Self::Item,D> {
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
        fn eval(self) -> ColumnVec<Self::Item,D> {
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


    //buffer ops (clone_to_buffer,copy_to_buffer)
    pub struct BufferedColVec<T,const D: usize>{buf: utils::IterInitArray<T,D>, is_unlinked: bool}

    impl<T,const D: usize> EvalsToColVec<D> for BufferedColVec<T,D> {
        type Item = T;

        #[inline]
        fn eval(self) -> ColumnVec<Self::Item,D> {
            if self.buf.last_index == D {
                ColumnVec(unsafe { self.buf.assume_init() })
            } else {
                panic!("\n\tmath_vector: Error, A BufferedColVec was not fully written to before being evaluated, only {} elements written",self.buf.last_index);
            }
        }
    }
    
    pub fn get_col_vec_buffer<T,const D: usize>() -> EvalColVec<BufferedColVec<T,D>,D> {
        EvalColVec(BufferedColVec{
            buf: utils::IterInitArray::new(),
            is_unlinked: true
        })
    }
    
        //Clone
    pub struct CloneToBufferColVec<'a,I: Iterator,const D: usize> where I::Item: Clone {iter: std::iter::Enumerate<I>, buffer: &'a mut BufferedColVec<I::Item,D>}

    impl<'a,I: Iterator,const D: usize> Iterator for CloneToBufferColVec<'a,I,D> where I::Item: Clone {
        type Item = I::Item;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            self.iter.next().map(|(idx,val)| {
                self.buffer.buf[idx] = std::mem::MaybeUninit::new(val.clone());
                val
            })
        }
    }
            //No reason not to implement this but its kinda dumb
    impl<'a,I: Iterator,const D: usize> EvalsToColVec<D> for CloneToBufferColVec<'a,I,D> where I::Item: Clone {
        type Item = I::Item;

        #[inline]
        fn eval(self) -> ColumnVec<Self::Item,D> {
            ColumnVec(unsafe { utils::iterator_to_array(self) })
        }
    }
    
        //Copy
    pub struct CopyToBufferColVec<'a,I: Iterator,const D: usize> where I::Item: Copy {iter: std::iter::Enumerate<I>, buffer: &'a mut BufferedColVec<I::Item,D>}

    impl<'a,I: Iterator,const D: usize> Iterator for CopyToBufferColVec<'a,I,D> where I::Item: Copy {
        type Item = I::Item;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            self.iter.next().map(|(idx,val)| {
                self.buffer.buf[idx] = std::mem::MaybeUninit::new(val);
                val
            })
        }
    }
            //No reason not to implement this but its kinda dumb
    impl<'a,I: Iterator,const D: usize> EvalsToColVec<D> for CopyToBufferColVec<'a,I,D> where I::Item: Copy {
        type Item = I::Item;

        #[inline]
        fn eval(self) -> ColumnVec<Self::Item,D> {
            ColumnVec(unsafe { utils::iterator_to_array(self) })
        }
    }
        

    impl<I: Iterator,const D: usize> EvalColVec<I,D> {
        pub fn clone_to_buffer(self,buf: &mut EvalColVec<BufferedColVec<I::Item,D>,D>) -> EvalColVec<CloneToBufferColVec<'_,I,D>,D> where I::Item: Clone {
            assert!(buf.0.is_unlinked);
            buf.0.is_unlinked = false;
            EvalColVec(CloneToBufferColVec{
                iter: self.0.enumerate(),
                buffer: &mut buf.0
            })
        }

        pub fn copy_to_buffer(self,buf: &mut EvalColVec<BufferedColVec<I::Item,D>,D>) -> EvalColVec<CopyToBufferColVec<'_,I,D>,D> where I::Item: Copy {
            assert!(buf.0.is_unlinked);
            buf.0.is_unlinked = false;
            EvalColVec(CopyToBufferColVec{
                iter: self.0.enumerate(),
                buffer: &mut buf.0
            })
        }
    }

        //MapBuf
    pub struct ColVecMapBuf<'a,I: Iterator,F: FnMut(I::Item) -> (O1,O2),O1,O2,const D: usize> {
        iter: std::iter::Enumerate<I>,
        func: F,
        buffer: &'a mut BufferedColVec<O1,D>
    }

    impl<'a,I: Iterator,F: FnMut(I::Item) -> (O1,O2),O1,O2,const D: usize> Iterator for ColVecMapBuf<'a,I,F,O1,O2,D> {
        type Item = O2;

        fn next(&mut self) -> Option<Self::Item> {
            self.iter.next().map(|(idx,x)| {
                let (y1,y2) = (self.func)(x); 
                self.buffer.buf[idx] = std::mem::MaybeUninit::new(y1);
                y2
            })
        }
    }

    impl<'a,I: Iterator,F: FnMut(I::Item) -> (O1,O2),O1,O2,const D: usize> EvalsToColVec<D> for ColVecMapBuf<'a,I,F,O1,O2,D> {
        type Item = O2;

        #[inline]
        fn eval(self) -> ColumnVec<Self::Item,D> {
            ColumnVec(unsafe { utils::iterator_to_array(self) })
        }
    }

    impl<I: Iterator,const D: usize> EvalColVec<I,D> {
        pub fn map_buf<F: FnMut(I::Item) -> (O1,O2),O1,O2>(self,buf: &mut EvalColVec<BufferedColVec<O1,D>,D>,func: F) -> EvalColVec<ColVecMapBuf<'_,I,F,O1,O2,D>,D> {
            EvalColVec(ColVecMapBuf{
                iter: self.0.enumerate(),
                func,
                buffer: &mut buf.0
            })
        }
    }

    impl<T,const D: usize> ColumnVec<T,D> {
        pub fn map_buf<F: FnMut(T) -> (O1,O2),O1,O2>(self,buf: &mut EvalColVec<BufferedColVec<O1,D>,D>,func: F) -> EvalColVec<ColVecMapBuf<'_,<[T; D] as IntoIterator>::IntoIter,F,O1,O2,D>,D> {
            assert!(buf.0.is_unlinked);
            buf.0.is_unlinked = false;
            EvalColVec(ColVecMapBuf{
                iter: self.0.into_iter().enumerate(),
                func,
                buffer: &mut buf.0
            })
        }

        pub fn map_buf_ref<'a,F: FnMut(&'a T) -> (O1,O2),O1,O2>(&'a self,buf: &'a mut EvalColVec<BufferedColVec<O1,D>,D>,func: F) -> EvalColVec<ColVecMapBuf<'a,<&'a [T; D] as IntoIterator>::IntoIter,F,O1,O2,D>,D> {
            assert!(buf.0.is_unlinked);
            buf.0.is_unlinked = false;
            EvalColVec(ColVecMapBuf{
                iter: self.0.iter().enumerate(),
                func,
                buffer: &mut buf.0
            })
        }

        pub fn map_buf_ref_mut<'a,F: FnMut(&'a mut T) -> (O1,O2),O1,O2>(&'a mut self,buf: &'a mut EvalColVec<BufferedColVec<O1,D>,D>,func: F) -> EvalColVec<ColVecMapBuf<'a,<&'a mut [T; D] as IntoIterator>::IntoIter,F,O1,O2,D>,D> {
            assert!(buf.0.is_unlinked);
            buf.0.is_unlinked = false;
            EvalColVec(ColVecMapBuf{
                iter: self.0.iter_mut().enumerate(),
                func,
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
        fn eval(self) -> ColumnVec<Self::Item,D> {
            ColumnVec(unsafe { utils::iterator_to_array(self) })
        }
    }


    impl<I: Iterator,const D: usize> EvalColVec<I,D> {
        #[inline]
        pub fn map<F: FnMut(I::Item) -> O,O>(self,f: F) -> ColVecMap<I,F,O,D> {
            ColVecMap(self.0,f)
        }
    }
    
    impl<T,const D: usize> ColumnVec<T,D> {
        #[inline]
        pub fn map<F: FnMut(<[T; D] as IntoIterator>::Item) -> O,O>(self,f: F) -> ColVecMap<<[T; D] as IntoIterator>::IntoIter,F,O,D> {
            ColVecMap(self.0.into_iter(),f)
        }

        #[inline]
        pub fn map_ref<'a,F: FnMut(<&'a [T; D] as IntoIterator>::Item) -> O,O>(&'a self,f: F) -> ColVecMap<<&'a [T; D] as IntoIterator>::IntoIter,F,O,D> {
            ColVecMap((&self.0).into_iter(),f)
        }

        #[inline]
        pub fn map_mut_ref<'a,F: FnMut(<&'a mut [T; D] as IntoIterator>::Item) -> O,O>(&'a mut self,f: F) -> ColVecMap<<&'a mut [T; D] as IntoIterator>::IntoIter,F,O,D> {
            ColVecMap((&mut self.0).into_iter(),f)
        }
    }
    
    
    //neg
    pub struct NegatedColVec<I: Iterator,const D: usize>(I) where I::Item: Neg;

    impl<I: Iterator,const D: usize> Iterator for NegatedColVec<I,D> where I::Item: Neg {
        type Item = <I::Item as Neg>::Output;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            self.0.next().map(|val| -val)
        }
    }

    impl<I: Iterator,const D: usize> EvalsToColVec<D> for NegatedColVec<I,D> where I::Item: Neg {
        type Item = <I::Item as Neg>::Output;

        #[inline]
        fn eval(self) -> ColumnVec<Self::Item,D> {
            ColumnVec(unsafe { utils::iterator_to_array(self) })
        }
    }


    impl<T,const D: usize> Neg for ColumnVec<T,D> where <[T; D] as IntoIterator>::Item: Neg {
        type Output = EvalColVec<NegatedColVec<<[T; D] as IntoIterator>::IntoIter,D>,D>;

        #[inline]
        fn neg(self) -> Self::Output {
            EvalColVec(NegatedColVec(self.0.into_iter()))
        }
    }

    impl<'a,T,const D: usize> Neg for &'a ColumnVec<T,D> where <&'a [T; D] as IntoIterator>::Item: Neg {
        type Output = EvalColVec<NegatedColVec<<&'a [T; D] as IntoIterator>::IntoIter,D>,D>;

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

    //zip
    pub struct ZippedColVec<I1: Iterator,I2: Iterator,const D: usize>(I1,I2);

    impl<I1: Iterator,I2: Iterator,const D: usize> Iterator for ZippedColVec<I1,I2,D> {
        type Item = (I1::Item,I2::Item);

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            if let (Some(val1),Some(val2)) = (self.0.next(),self.1.next()) {
                Some((val1,val2))
            } else {
                None
            }
        }
    }

    impl<I1: Iterator,I2: Iterator,const D: usize> EvalsToColVec<D> for ZippedColVec<I1,I2,D> {
        type Item = (I1::Item,I2::Item);

        #[inline]
        fn eval(self) -> ColumnVec<Self::Item,D> {
            ColumnVec(unsafe { utils::iterator_to_array(self) })
        }
    }


    impl<I1: Iterator,I2: Iterator,const D: usize> Zip<EvalColVec<I2,D>> for EvalColVec<I1,D> {
        type Output = EvalColVec<ZippedColVec<I1,I2,D>,D>;

        #[inline]
        fn zip(self,other: EvalColVec<I2,D>) -> Self::Output {
            EvalColVec(ZippedColVec(
                self.0,
                other.0
            ))
        }
    }


    impl<I1: Iterator,T2,const D: usize> Zip<ColumnVec<T2,D>> for EvalColVec<I1,D> {
        type Output = EvalColVec<ZippedColVec<I1,<[T2; D] as IntoIterator>::IntoIter,D>,D>;

        #[inline]
        fn zip(self,other: ColumnVec<T2,D>) -> Self::Output {
            EvalColVec(ZippedColVec(
                self.0,
                other.0.into_iter()
            ))
        }
    }

    impl<'a, I1: Iterator,T2,const D: usize> Zip<&'a ColumnVec<T2,D>> for EvalColVec<I1,D> {
        type Output = EvalColVec<ZippedColVec<I1,<&'a [T2; D] as IntoIterator>::IntoIter,D>,D>;

        #[inline]
        fn zip(self,other: &'a ColumnVec<T2,D>) -> Self::Output {
            EvalColVec(ZippedColVec(
                self.0,
                (&other.0).into_iter()
            ))
        }
    }

    impl<'a, I1: Iterator,T2,const D: usize> Zip<&'a mut ColumnVec<T2,D>> for EvalColVec<I1,D> {
        type Output = EvalColVec<ZippedColVec<I1,<&'a mut [T2; D] as IntoIterator>::IntoIter,D>,D>;

        #[inline]
        fn zip(self,other: &'a mut ColumnVec<T2,D>) -> Self::Output {
            EvalColVec(ZippedColVec(
                self.0,
                (&mut other.0).into_iter()
            ))
        }
    }


    impl<T1,I2: Iterator,const D: usize> Zip<EvalColVec<I2,D>> for ColumnVec<T1,D> {
        type Output = EvalColVec<ZippedColVec<<[T1; D] as IntoIterator>::IntoIter,I2,D>,D>;

        #[inline]
        fn zip(self,other: EvalColVec<I2,D>) -> Self::Output {
            EvalColVec(ZippedColVec(
                self.0.into_iter(),
                other.0
            ))
        }
    }

    impl<'a,T1,I2: Iterator,const D: usize> Zip<EvalColVec<I2,D>> for &'a ColumnVec<T1,D> {
        type Output = EvalColVec<ZippedColVec<<&'a [T1; D] as IntoIterator>::IntoIter,I2,D>,D>;

        #[inline]
        fn zip(self,other: EvalColVec<I2,D>) -> Self::Output {
            EvalColVec(ZippedColVec(
                (&self.0).into_iter(),
                other.0
            ))
        }
    }

    impl<'a,T1,I2: Iterator,const D: usize> Zip<EvalColVec<I2,D>> for &'a mut ColumnVec<T1,D> {
        type Output = EvalColVec<ZippedColVec<<&'a mut [T1; D] as IntoIterator>::IntoIter,I2,D>,D>;

        #[inline]
        fn zip(self,other: EvalColVec<I2,D>) -> Self::Output {
            EvalColVec(ZippedColVec(
                (&mut self.0).into_iter(),
                other.0
            ))
        }
    }


    impl<T1,T2,const D: usize> Zip<ColumnVec<T2,D>> for ColumnVec<T1,D> {
        type Output = EvalColVec<ZippedColVec<<[T1; D] as IntoIterator>::IntoIter,<[T2; D] as IntoIterator>::IntoIter,D>,D>;

        #[inline]
        fn zip(self,other: ColumnVec<T2,D>) -> Self::Output {
            EvalColVec(ZippedColVec(
                self.0.into_iter(),
                other.0.into_iter()
            ))
        }
    }

    impl<'a,T1,T2,const D: usize> Zip<&'a ColumnVec<T2,D>> for ColumnVec<T1,D> {
        type Output = EvalColVec<ZippedColVec<<[T1; D] as IntoIterator>::IntoIter,<&'a [T2; D] as IntoIterator>::IntoIter,D>,D>;

        #[inline]
        fn zip(self,other: &'a ColumnVec<T2,D>) -> Self::Output {
            EvalColVec(ZippedColVec(
                self.0.into_iter(),
                (&other.0).into_iter()
            ))
        }
    }

    impl<'a,T1,T2,const D: usize> Zip<&'a mut ColumnVec<T2,D>> for ColumnVec<T1,D> {
        type Output = EvalColVec<ZippedColVec<<[T1; D] as IntoIterator>::IntoIter,<&'a mut [T2; D] as IntoIterator>::IntoIter,D>,D>;

        #[inline]
        fn zip(self,other: &'a mut ColumnVec<T2,D>) -> Self::Output {
            EvalColVec(ZippedColVec(
                self.0.into_iter(),
                (&mut other.0).into_iter()
            ))
        }
    }


    impl<'a,T1,T2,const D: usize> Zip<ColumnVec<T2,D>> for &'a ColumnVec<T1,D> {
        type Output = EvalColVec<ZippedColVec<<&'a [T1; D] as IntoIterator>::IntoIter,<[T2; D] as IntoIterator>::IntoIter,D>,D>;

        #[inline]
        fn zip(self,other: ColumnVec<T2,D>) -> Self::Output {
            EvalColVec(ZippedColVec(
                (&self.0).into_iter(),
                other.0.into_iter()
            ))
        }
    }

    impl<'a,T1,T2,const D: usize> Zip<&'a ColumnVec<T2,D>> for &'a ColumnVec<T1,D> {
        type Output = EvalColVec<ZippedColVec<<&'a [T1; D] as IntoIterator>::IntoIter,<&'a [T2; D] as IntoIterator>::IntoIter,D>,D>;

        #[inline]
        fn zip(self,other: &'a ColumnVec<T2,D>) -> Self::Output {
            EvalColVec(ZippedColVec(
                (&self.0).into_iter(),
                (&other.0).into_iter()
            ))
        }
    }

    impl<'a,T1,T2,const D: usize> Zip<&'a mut ColumnVec<T2,D>> for &'a ColumnVec<T1,D> {
        type Output = EvalColVec<ZippedColVec<<&'a [T1; D] as IntoIterator>::IntoIter,<&'a mut [T2; D] as IntoIterator>::IntoIter,D>,D>;

        #[inline]
        fn zip(self,other: &'a mut ColumnVec<T2,D>) -> Self::Output {
            EvalColVec(ZippedColVec(
                (&self.0).into_iter(),
                (&mut other.0).into_iter()
            ))
        }
    }


    impl<'a,T1,T2,const D: usize> Zip<ColumnVec<T2,D>> for &'a mut ColumnVec<T1,D> {
        type Output = EvalColVec<ZippedColVec<<&'a mut [T1; D] as IntoIterator>::IntoIter,<[T2; D] as IntoIterator>::IntoIter,D>,D>;

        #[inline]
        fn zip(self,other: ColumnVec<T2,D>) -> Self::Output {
            EvalColVec(ZippedColVec(
                (&mut self.0).into_iter(),
                other.0.into_iter()
            ))
        }
    }

    impl<'a,T1,T2,const D: usize> Zip<&'a ColumnVec<T2,D>> for &'a mut ColumnVec<T1,D> {
        type Output = EvalColVec<ZippedColVec<<&'a mut [T1; D] as IntoIterator>::IntoIter,<&'a [T2; D] as IntoIterator>::IntoIter,D>,D>;

        #[inline]
        fn zip(self,other: &'a ColumnVec<T2,D>) -> Self::Output {
            EvalColVec(ZippedColVec(
                (&mut self.0).into_iter(),
                (&other.0).into_iter()
            ))
        }
    }

    impl<'a,T1,T2,const D: usize> Zip<&'a mut ColumnVec<T2,D>> for &'a mut ColumnVec<T1,D> {
        type Output = EvalColVec<ZippedColVec<<&'a mut [T1; D] as IntoIterator>::IntoIter,<&'a mut [T2; D] as IntoIterator>::IntoIter,D>,D>;

        #[inline]
        fn zip(self,other: &'a mut ColumnVec<T2,D>) -> Self::Output {
            EvalColVec(ZippedColVec(
                (&mut self.0).into_iter(),
                (&mut other.0).into_iter()
            ))
        }
    }

    //unzip 
    pub struct UnzippedColVec<'a,I: Iterator<Item=(T1,T2)>,T1,T2,const D: usize> {
        iter: I,
        buffer: &'a mut BufferedColVec<T1,D>
    }

    impl<'a,I: Iterator<Item=(T1,T2)>,T1,T2,const D: usize> Iterator for UnzippedColVec<'a,I,T1,T2,D> {
        type Item = T2;

        fn next(&mut self) -> Option<Self::Item> {
            self.iter.next().map(|(y1,y2)| {
                self.buffer.buf.assign_next(y1);
                y2
            })
        }
    }

    impl<'a,I: Iterator<Item=(T1,T2)>,T1,T2,const D: usize> EvalsToColVec<D> for UnzippedColVec<'a,I,T1,T2,D> {
        type Item = T2;

        #[inline]
        fn eval(self) -> ColumnVec<Self::Item,D> {
            ColumnVec(unsafe { utils::iterator_to_array(self) })
        }
    }

    impl<I: Iterator + EvalsToColVec<D>,const D: usize> Unzippable<EvalColVec<I,D>> for ColumnVec<<I as EvalsToColVec<D>>::Item,D> {
        fn unzip(data: EvalColVec<I,D>) -> Self {
            data.eval()
        }
    }

    impl<I: Iterator<Item=(T1,T2)> + EvalsToColVec<D>,T1,T2,O2,const D: usize> Unzippable<EvalColVec<I,D>> for (ColumnVec<T1,D>,O2)
    where for<'a> O2: Unzippable<EvalColVec<UnzippedColVec<'a,I,T1,T2,D>,D>> {
        fn unzip(data: EvalColVec<I,D>) -> Self {
            let mut buffer = get_col_vec_buffer();
            let right_output = <O2>::unzip(EvalColVec(UnzippedColVec{iter: data.0,buffer: &mut buffer.0}));
            (buffer.eval(),right_output)
        }
    }

    impl<I: Iterator + EvalsToColVec<D>,const D: usize> EvalColVec<I,D> {
        pub fn unzip_eval<O>(self) -> O where O: Unzippable<EvalColVec<I,D>> {
            <O as Unzippable<Self>>::unzip(self)
        }
    }

    impl<T1,T2,const D: usize> ColumnVec<(T1,T2),D> {
        pub fn unzip<O>(self) -> (ColumnVec<T1,D>,O) where for<'a> O: Unzippable<EvalColVec<UnzippedColVec<'a,<[(T1,T2); D] as IntoIterator>::IntoIter,T1,T2,D>,D>> {
            let mut buffer = get_col_vec_buffer();
            let right_output = <O as Unzippable<EvalColVec<UnzippedColVec<<[(T1,T2); D] as IntoIterator>::IntoIter,T1,T2,D>,D>>>::unzip(EvalColVec(UnzippedColVec{iter: self.0.into_iter(),buffer: &mut buffer.0}));
            (buffer.eval(),right_output)
        }
    }

    //fold
    impl<I: Iterator,const D: usize> EvalColVec<I,D> {
        pub fn fold<T,F: FnMut(T,I::Item) -> T>(self,mut init: T,mut f: F) -> T {
            for item in self.0 {
                init = (f)(init,item)
            }
            init
        }
    }

    impl<T,const D: usize> ColumnVec<T,D> {
        pub fn fold<O,F: FnMut(O,<[T;D] as IntoIterator>::Item) -> O>(self,mut init: O,mut f: F) -> O {
            for item in self.0 {
                init = (f)(init,item)
            }
            init
        }

        pub fn fold_ref<'a,O,F: FnMut(O,<&'a [T;D] as IntoIterator>::Item) -> O>(&'a self,mut init: O, mut f: F) -> O {
            for item in &self.0 {
                init = (f)(init,item)
            }
            init
        }

        pub fn fold_ref_mut<'a,O,F: FnMut(O,<&'a mut [T;D] as IntoIterator>::Item) -> O>(&'a mut self,mut init: O, mut f: F) -> O {
            for item in &mut self.0 {
                init = (f)(init,item)
            }
            init
        }
    }

    //sum
    impl<I: Iterator,const D: usize> EvalColVec<I,D> {
        pub fn sum<O>(self) -> O where O: std::iter::Sum<I::Item> {
            <O>::sum(self.0)
        }
    }

    impl<T,const D: usize> ColumnVec<T,D> {
        pub fn sum<O>(self) -> O where O: std::iter::Sum<<[T; D] as IntoIterator>::Item> {
            <O>::sum(self.0.into_iter())
        }

        pub fn sum_ref<'a,O>(&'a self) -> O where O: std::iter::Sum<<&'a [T; D] as IntoIterator>::Item> {
            <O>::sum(self.0.iter())
        }
    }

    //sqr_mag
    impl<I: Iterator,const D: usize> EvalColVec<I,D> {
        pub fn sqr_mag<O>(self) -> O where I::Item: Mul<I::Item> + Copy, O: std::iter::Sum<<I::Item as Mul<I::Item>>::Output> {
            self.map(|x| x*x).sum()
        }
    }

    impl<T,const D: usize> ColumnVec<T,D> {
        pub fn sqr_mag<O>(self) -> O where T: Mul<T> + Copy, O: std::iter::Sum<<T as Mul<T>>::Output> {
            self.map(|x| x*x).sum()
        }

        pub fn sqr_mag_ref<'a,O>(&'a self) -> O where &'a T: Mul<&'a T>, O: std::iter::Sum<<&'a T as Mul<&'a T>>::Output> {
            self.map_ref(|x| x*x).sum()
        }
    }
}

#[cfg(test)]
mod test {
    use crate::col_vec_iterators::get_col_vec_buffer;

    use super::*;
    use super::col_vec_iterators::EvalsToColVec;

    use rand::Rng;

    #[test]
    fn basic_ops() {
        let x = ColumnVec::from([1; 5]);
        let y = ColumnVec::from([2; 5]);
        let z = (x+y).eval();
        assert_eq!(ColumnVec::from([3,3,3,3,3]),z);

        let x = ColumnVec::from([1; 5]);
        let y = ColumnVec::from([2; 5]);
        let z = (x-y).eval();
        assert_eq!(ColumnVec::from([-1,-1,-1,-1,-1]),z);

        let x = ColumnVec::from([2; 5]);
        assert_eq!(ColumnVec::from([6; 5]),(x * Scalar(3)).eval());

        let x = ColumnVec::from([6; 5]);
        assert_eq!(ColumnVec::from([2; 5]),(x / Scalar(3)).eval());

        let x = ColumnVec::from([5; 5]);
        assert_eq!(ColumnVec::from([2; 5]),(x % Scalar(3)).eval());

        let x = ColumnVec::from([2; 5]);
        let y = (Scalar(3) * x).eval();
        assert_eq!(ColumnVec::from([6; 5]),y);

        let x = ColumnVec::from([2; 5]);
        let y = (Scalar(6) / x).eval();
        assert_eq!(ColumnVec::from([3; 5]),y);

        let x = ColumnVec::from([2; 5]);
        let y = (Scalar(3) % x).eval();
        assert_eq!(ColumnVec::from([1; 5]),y);
    }

    #[test]
    fn stacking_ops() {
        let a = ColumnVec::from([4; 5]);
        let b = ColumnVec::from([2; 5]);
        let c = ColumnVec::from([1; 5]);
        let test = 
            std::ops::Sub::<ColumnVec<i32,5>>::sub(
                std::ops::Add::<ColumnVec<i32, 5>>::add(a,b),
                c)
            *Scalar(3)
            /Scalar(5)
            %Scalar(2);
        let test = test.eval();
        assert_eq!(ColumnVec::from([1;5]),test);
    }

    #[test]
    fn assign_ops() {
        let mut x: ColumnVec<i32, 5> = ColumnVec::from([4; 5]);
        std::ops::AddAssign::<ColumnVec<i32,5>>::add_assign(&mut x,ColumnVec::from([3; 5]));
        assert_eq!(ColumnVec::from([7; 5]),x);
        std::ops::SubAssign::<ColumnVec<i32,5>>::sub_assign(&mut x, ColumnVec::from([5; 5]));
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
        assert_eq!(ColumnVec::from([-3; 5]),(-x).eval());

        let x = ColumnVec::from([3; 5]);
        let y = ColumnVec::from([2; 5]);
        assert_eq!(Scalar(30),Dot::<ColumnVec<i32,5>>::dot(x,y));

        /*
        let x = ColumnVec::from([&3; 5]);
        assert_eq!(ColumnVec::from([3; 5]),x.cloned().eval());

        let x = ColumnVec::from([&3; 5]);
        assert_eq!(ColumnVec::from([3; 5]),x.copied().eval());
        */

        let x = ColumnVec::from([3; 5]);
        assert_eq!(ColumnVec::from([4; 5]),x.map(|val| {val + 1}).eval())
    }

    //Doesnt check leaky-ness
    #[test]
    fn eval_test() {
        let vec = ColumnVec::from([1,2,3,4,5,6,7,8,9,10]);
        assert_eq!(ColumnVec::from([2,4,6,8,10,12,14,16,18,20]),(vec*Scalar(2)).eval());
    }

    #[test]
    fn clone_and_copy_to_buffer_test() {
        let mut buf = get_col_vec_buffer();
        let vec = ColumnVec::from([1,2,3,4,5,6,7,8,9,10]);
        assert_eq!(ColumnVec::from([4,8,12,16,20,24,28,32,36,40]),((vec * Scalar(2)).clone_to_buffer(&mut buf) * Scalar(2)).eval());
        assert_eq!(ColumnVec::from([2,4,6,8,10,12,14,16,18,20]),buf.eval());

        let mut buf = get_col_vec_buffer();
        let vec = ColumnVec::from([1,2,3,4,5,6,7,8,9,10]);
        assert_eq!(ColumnVec::from([4,8,12,16,20,24,28,32,36,40]),((vec * Scalar(2)).copy_to_buffer(&mut buf) * Scalar(2)).eval());
        assert_eq!(ColumnVec::from([2,4,6,8,10,12,14,16,18,20]),buf.eval());
    }

    #[test]
    fn zip_and_unzip_test() {
        let x = ColumnVec::from([1,2,3,4,5,6,7,8,9,10]);
        let y = ColumnVec::from([10,9,8,7,6,5,4,3,2,1]);
        let z = x.zip(y).eval();
        assert_eq!(ColumnVec::from([(1,10),(2,9),(3,8),(4,7),(5,6),(6,5),(7,4),(8,3),(9,2),(10,1)]),z);

        let (x,y) = z.clone().unzip();
        assert_eq!(ColumnVec::from([1,2,3,4,5,6,7,8,9,10]),x);
        assert_eq!(ColumnVec::from([10,9,8,7,6,5,4,3,2,1]),y);
    }

    #[test]
    fn eval_stress_test() {
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
            let vec1 = ColumnVec::from(<[u32;10]>::from(vec_nums));
            let vec2 = ColumnVec::from([vec_nums.0 * 2,vec_nums.1 * 2,vec_nums.2 * 2,vec_nums.3 * 2,vec_nums.4 * 2,vec_nums.5 * 2,vec_nums.6 * 2,vec_nums.7 * 2,vec_nums.8 * 2,vec_nums.9 * 2]);
            assert_eq!(vec2,(vec1 * Scalar(2)).eval());
        }
    }

    #[test]
    fn zip_and_unzip_stress_test() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000000 {
            let x: ColumnVec<u32, 10> = ColumnVec::from([
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
            let y: ColumnVec<u32, 10> = ColumnVec::from([
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

            let (x_prime,y_prime): (_,ColumnVec<u32, 10>) = x.clone().zip(y.clone()).unzip_eval();
            assert_eq!(x_prime,x);
            assert_eq!(y_prime,y);
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

    #[ignore]
    #[test]
    fn col_vec_vs_vec_speed_test() {
        let mut custom = std::time::Duration::new(0,0);
        let mut builtin = std::time::Duration::new(0,0);
        for _ in 0..1000000 {
            let x = ColumnVec::from([15; 1000]);
            let y = ColumnVec::from([15; 1000]);
            let now = std::time::Instant::now();
            let z = (x+y).eval();
            let next_time = now.elapsed();
            assert_eq!(z,ColumnVec::from([30; 1000]));
            custom += next_time;

            
            let x = vec![15;1000];
            let y = vec![15;1000];
            let now = std::time::Instant::now();
            let mut z = Vec::with_capacity(1000);
            for (x_val,y_val) in x.into_iter().zip(y.into_iter()) {
                z.push(x_val + y_val);
            }
            let next_time = now.elapsed();
            assert_eq!(z,vec![30; 1000]);
            builtin += next_time;
        }
        println!("{}",custom.as_nanos() as f64/1000000.0);
        println!("{}",builtin.as_nanos() as f64/1000000.0);
    }

    #[ignore]
    #[test]
    fn new_buffer_vs_reused_buffer() {
        let mut reuse = std::time::Duration::new(0,0);
        let mut new = std::time::Duration::new(0,0);
        for _ in 0..1000000 {
            let x = ColumnVec::from([15; 10000]);
            let y = ColumnVec::from([15; 10000]);
            let now = std::time::Instant::now();
            let _z = (x+y).eval();
            let next_time = now.elapsed();
            new += next_time;

            let mut x = [15; 10000];
            let y = [15; 10000];
            let now = std::time::Instant::now();
            for (x_val,y_val) in x.iter_mut().zip(y.into_iter()) {
                *x_val = *x_val+y_val;
            }
            let next_time = now.elapsed();
            reuse += next_time;
        }
        println!("{}",new.as_nanos() as f64/1000000.0);
        println!("{}",reuse.as_nanos() as f64/1000000.0);
    }
    
}