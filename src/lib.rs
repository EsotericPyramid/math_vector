#![recursion_limit = "20"]

#[derive(PartialEq,Debug)]
#[derive(Clone,Copy)]
#[repr(transparent)]
pub struct Scalar<T>(pub T);

//CHECK
impl<T> Scalar<T> {
    //CHECK
    pub fn move_ref_out<'a>(scalar: Scalar<&'a T>) -> &'a Self {
        //DOUBLE CHECK THIS ZACH
        //Scalar<&T> == &T == &Scalar<T>
        unsafe{ std::mem::transmute(scalar) }
    }

    pub fn clone_inner<'a>(scalar: Scalar<&'a T>) -> Self where T: Clone {
        Scalar(scalar.0.clone())
    }

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

                fn $OpF(self,rhs: Scalar<S2>) -> Self::Output {
                    Scalar(self.0.$OpF(rhs.0))
                }
            }

            impl<'a,S1,S2> $Op<Scalar<S2>> for &'a Scalar<S1> where &'a S1: $Op<S2> {
                type Output = Scalar<<&'a S1 as $Op<S2>>::Output>;

                fn $OpF(self,rhs: Scalar<S2>) -> Self::Output {
                    Scalar((&self.0).$OpF(rhs.0))
                }
            }

            impl<'a,S1,S2> $Op<&'a Scalar<S2>> for Scalar<S1> where S1: $Op<&'a S2> {
                type Output = Scalar<S1::Output>;

                fn $OpF(self,rhs: &'a Scalar<S2>) -> Self::Output {
                    Scalar(self.0.$OpF(&rhs.0))
                }
            }

            impl<'a,S1,S2> $Op<&'a Scalar<S2>> for &'a Scalar<S1> where &'a S1: $Op<&'a S2> {
                type Output = Scalar<<&'a S1 as $Op<&'a S2>>::Output>;

                fn $OpF(self,rhs: &'a Scalar<S2>) -> Self::Output {
                    Scalar((&self.0).$OpF(&rhs.0))
                }
            }
        };
    }
    macro_rules! overload_assign_operator_for_scalar {
        ($Op:ident,$OpF:ident) => {
            impl<S1,S2> $Op<Scalar<S2>> for Scalar<S1> where S1: $Op<S2> {
                fn $OpF(&mut self,rhs: Scalar<S2>) {
                    self.0.$OpF(rhs.0);
                }
            }

            impl<'a,S1,S2> $Op<&'a Scalar<S2>> for Scalar<S1> where S1: $Op<&'a S2> {
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
pub struct ColumnVec<T: IntoIterator,const D: usize>(T);

impl<T: IntoIterator,const D: usize> ColumnVec<T,D> {
    pub fn unwrap(self) -> T {
        self.0
    }

    pub fn unchecked_from(iterable: T) -> Self {
        ColumnVec(iterable)
    }

    pub fn transpose(self) -> RowVec<T,D> {
        RowVec(self.0)
    }
}

impl<T,const D: usize> From<[T; D]> for ColumnVec<[T; D],D> {
    fn from(value: [T; D]) -> Self {
        ColumnVec(value)
    }
}

impl<T,const D: usize> TryFrom<Vec<T>> for ColumnVec<Vec<T>,D> {
    type Error = ();

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
    
    fn into_iter(self) -> Self::IntoIter {
        ColVecIter(self.0.into_iter())
    }
}

impl<'a,T: IntoIterator,const D: usize> IntoIterator for &'a ColumnVec<T,D> where &'a T: IntoIterator {
    type IntoIter = ColVecIter<<&'a T as IntoIterator>::IntoIter>;
    type Item = Scalar<<&'a T as IntoIterator>::Item>;

    fn into_iter(self) -> Self::IntoIter {
        ColVecIter((&self.0).into_iter())
    }
}

impl<'a,T: IntoIterator,const D: usize> IntoIterator for &'a mut ColumnVec<T,D> where &'a mut T: IntoIterator {
    type IntoIter = ColVecIter<<&'a mut T as IntoIterator>::IntoIter>;
    type Item = Scalar<<&'a mut T as IntoIterator>::Item>;

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

    pub struct EvalColVec<T,const D: usize>(pub(super) T);

    impl<I: Iterator,const D: usize> IntoIterator for EvalColVec<I,D> {
        type IntoIter = ColVecIter<I>;
        type Item = Scalar<I::Item>;

        fn into_iter(self) -> Self::IntoIter {
            ColVecIter(self.0)
        }
    }


    pub trait EvalsToColVec<const D: usize> {
        type Output: IntoIterator;
    
        fn eval(self) -> ColumnVec<Self::Output,D>;
    }

    impl<T: EvalsToColVec<D>,const D: usize> EvalsToColVec<D> for EvalColVec<T,D> {
        type Output = T::Output;
        
        fn eval(self) -> ColumnVec<Self::Output,D> {
            self.0.eval()
        }
    }

    macro_rules! impl_eval_col_vec {
        ($t:ty,1,$ti:ty,$tr:path) => {
            impl<I: Iterator,S: Copy,const D: usize> EvalsToColVec<D> for $t where $ti: $tr {
                type Output =  Vec<<$t as Iterator>::Item>;

                fn eval(self) -> ColumnVec<Self::Output,D> {
                    let mut vec = Vec::with_capacity(D);
                    for output in self {
                        vec.push(output)
                    }
                    ColumnVec(vec)
                }
            }
        };
        ($t:ty,2,$ti:ty,$tr:path) => {
            impl<I1: Iterator,I2: Iterator,const D: usize> EvalsToColVec<D> for $t where $ti: $tr {
                type Output = Vec<<$t as Iterator>::Item>;

                fn eval(self) -> ColumnVec<Self::Output,D> {
                    let mut vec = Vec::with_capacity(D);
                    for output in self {
                        vec.push(output)
                    }
                    ColumnVec(vec)
                }
            }
        };
    }

    macro_rules! overload_given_operator_for_vec {
        ($Op:ident,$OpF:ident,$OpI:ident) => {
            impl<I1: IntoIterator,I2: IntoIterator,const D: usize> $Op<ColumnVec<I2,D>> for ColumnVec<I1,D> where I1::Item: $Op<I2::Item> {
                type Output = EvalColVec<$OpI<I1::IntoIter,I2::IntoIter,D>,D>;

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

                fn $OpF(self,other: &'a ColumnVec<I2,D>) -> Self::Output {
                    EvalColVec($OpI(
                        (&self.0).into_iter(),
                        (&other.0).into_iter()
                    ))
                }
            }
        

            impl<I1: Iterator + EvalsToColVec<D>,I2: IntoIterator,const D: usize> $Op<ColumnVec<I2,D>> for EvalColVec<I1,D> where I1::Item: $Op<I2::Item> {
                type Output = EvalColVec<$OpI<I1,I2::IntoIter,D>,D>;

                fn $OpF(self,other: ColumnVec<I2,D>) -> Self::Output {
                    EvalColVec($OpI(
                        self.0,
                        other.0.into_iter()
                    ))
                }
            }

            impl<'a,I1: Iterator + EvalsToColVec<D>,I2: IntoIterator,const D: usize> $Op<&'a ColumnVec<I2,D>> for EvalColVec<I1,D> 
            where &'a I2: IntoIterator, I1::Item: $Op<<&'a I2 as IntoIterator>::Item> {
                type Output = EvalColVec<$OpI<I1,<&'a I2 as IntoIterator>::IntoIter,D>,D>;

                fn $OpF(self,other: &'a ColumnVec<I2,D>) -> Self::Output {
                    EvalColVec($OpI(
                        self.0,
                        (&other.0).into_iter()
                    ))
                }
            }

            impl<I1: IntoIterator,I2: Iterator + EvalsToColVec<D>,const D: usize> $Op<EvalColVec<I2,D>> for ColumnVec<I1,D> where I1::Item: $Op<I2::Item> {
                type Output = EvalColVec<$OpI<I1::IntoIter,I2,D>,D>;

                fn $OpF(self,other: EvalColVec<I2,D>) -> Self::Output {
                    EvalColVec($OpI(
                        self.0.into_iter(),
                        other.0
                    ))
                }
            }

            impl<'a,I1: IntoIterator,I2: Iterator + EvalsToColVec<D>,const D: usize> $Op<EvalColVec<I2,D>> for &'a ColumnVec<I1,D> 
            where &'a I1: IntoIterator, <&'a I1 as IntoIterator>::Item: $Op<I2::Item> {
                type Output = EvalColVec<$OpI<<&'a I1 as IntoIterator>::IntoIter,I2,D>,D>;

                fn $OpF(self,other: EvalColVec<I2,D>) -> Self::Output {
                    EvalColVec($OpI(
                        (&self.0).into_iter(),
                        other.0
                    ))
                }
            }


            impl<I1: Iterator + EvalsToColVec<D>,I2: Iterator + EvalsToColVec<D>,const D: usize> $Op<EvalColVec<I2,D>> for EvalColVec<I1,D> 
            where I1::Item: $Op<I2::Item> {
                type Output = EvalColVec<$OpI<I1,I2,D>,D>;

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

                fn $OpF(self,rhs: Scalar<S>) -> Self::Output {
                    EvalColVec($OpI(
                        (&self.0).into_iter(),
                        rhs.0
                    ))
                }
            }

            impl<'a,I: IntoIterator,S: Copy,const D: usize> $Op<&'a Scalar<S>> for ColumnVec<I,D> where I::Item: $Op<&'a S> {
                type Output = EvalColVec<$OpI<I::IntoIter,&'a S,D>,D>;

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

                fn $OpF(self,rhs: &'a Scalar<S>) -> Self::Output {
                    EvalColVec($OpI(
                        (&self.0).into_iter(),
                        &rhs.0
                    ))
                }
            }


            impl<I: Iterator + EvalsToColVec<D>,S: Copy,const D: usize> $Op<Scalar<S>> for EvalColVec<I,D> where I::Item: $Op<S> {
                type Output = EvalColVec<$OpI<I,S,D>,D>;

                fn $OpF(self,rhs: Scalar<S>) -> Self::Output {
                    EvalColVec($OpI(
                        self.0,
                        rhs.0
                    ))
                }
            }

            impl<'a,I: Iterator + EvalsToColVec<D>,S: Copy,const D: usize> $Op<&'a Scalar<S>> for EvalColVec<I,D> where I::Item: $Op<&'a S> {
                type Output = EvalColVec<$OpI<I,&'a S,D>,D>;

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

                fn $OpF(self,rhs: ColumnVec<I,D>) -> Self::Output {
                    EvalColVec($OpI(
                        rhs.0.into_iter(),
                        self.0
                    ))
                }
            }

            impl<'a,I: IntoIterator,S: Copy,const D: usize> $Op<ColumnVec<I,D>> for &'a Scalar<S> where &'a S: $Op<I::Item> {
                type Output = EvalColVec<$OpI<&'a S,I::IntoIter,D>,D>;

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

                fn $OpF(self,rhs: &'a ColumnVec<I,D>) -> Self::Output {
                    EvalColVec($OpI(
                        (&rhs.0).into_iter(),
                        &self.0
                    ))
                }
            }
        
            
            impl<I: Iterator + EvalsToColVec<D>,S: Copy,const D: usize> $Op<EvalColVec<I,D>> for Scalar<S> where S: $Op<I::Item> {
                type Output = EvalColVec<$OpI<S,I,D>,D>;

                fn $OpF(self,rhs: EvalColVec<I,D>) -> Self::Output {
                    EvalColVec($OpI(
                        rhs.0,
                        self.0
                    ))
                }
            }

            impl<'a, I: Iterator + EvalsToColVec<D>,S: Copy,const D: usize> $Op<EvalColVec<I,D>> for &'a Scalar<S> where &'a S: $Op<I::Item> {
                type Output = EvalColVec<$OpI<&'a S,I,D>,D>;

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
                fn $OpF(&mut self,rhs: ColumnVec<I2,D>) {
                    for (self_val,other_val) in (&mut self.0).into_iter().zip(rhs.0.into_iter()) {
                        (self_val).$OpF(other_val)
                    }
                }
            }

            impl<'b,I1: IntoIterator,I2: IntoIterator,O,const D: usize> $Op<&'b ColumnVec<I2,D>> for ColumnVec<I1,D> 
            where &'b I2: IntoIterator,for<'a> &'a mut I1: IntoIterator<Item = &'a mut O>,O: $Op<<&'b I2 as IntoIterator>::Item> {
                fn $OpF(&mut self,rhs: &'b ColumnVec<I2,D>) {
                    for (self_val,other_val) in (&mut self.0).into_iter().zip((&rhs.0).into_iter()) {
                        (self_val).$OpF(other_val)
                    }
                }
            }

            impl<I1: IntoIterator,I2: Iterator,O,const D: usize> $Op<EvalColVec<I2,D>> for ColumnVec<I1,D> 
            where for<'a> &'a mut I1: IntoIterator<Item = &'a mut O>,O: $Op<I2::Item> {
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
                fn $OpF(&mut self,rhs: Scalar<S>) {
                    for val in (&mut self.0).into_iter() {
                        (val).$OpF(rhs.0)
                    }
                }
            }

            impl<'b,I: IntoIterator,O,S: Copy,const D: usize> $Op<&'b Scalar<S>> for ColumnVec<I,D> 
            where for<'a> &'a mut I: IntoIterator<Item = &'a mut O>,for<'a> O: $Op<&'b S> {
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


    impl<I1: Iterator + EvalsToColVec<D>,I2: IntoIterator,O: AddAssign<O> + std::default::Default,const D: usize> Dot<ColumnVec<I2,D>> for EvalColVec<I1,D> 
    where I1::Item: Mul<I2::Item,Output = O> {
        type Output = Scalar<O>;

        fn dot(self,rhs: ColumnVec<I2,D>) -> Self::Output {
            let mut output = <O>::default();
            for (left_val,right_val) in self.0.zip(rhs.0.into_iter()) {
                output += left_val * right_val;
            }
            Scalar(output)
        }
    }

    impl<'a,I1: Iterator + EvalsToColVec<D>,I2: IntoIterator,O: AddAssign<O> + std::default::Default,const D: usize> Dot<&'a ColumnVec<I2,D>> for EvalColVec<I1,D> 
    where &'a I2: IntoIterator, I1::Item: Mul<<&'a I2 as IntoIterator>::Item,Output = O> {
        type Output = Scalar<O>;

        fn dot(self,rhs: &'a ColumnVec<I2,D>) -> Self::Output {
            let mut output = <O>::default();
            for (left_val,right_val) in self.0.zip((&rhs.0).into_iter()) {
                output += left_val * right_val;
            }
            Scalar(output)
        }
    }

    impl<I1: IntoIterator,I2: Iterator + EvalsToColVec<D>,O: AddAssign<O> + std::default::Default,const D: usize> Dot<EvalColVec<I2,D>> for ColumnVec<I1,D> 
    where I1::Item: Mul<I2::Item,Output = O> {
        type Output = Scalar<O>;

        fn dot(self,rhs: EvalColVec<I2,D>) -> Self::Output {
            let mut output = <O>::default();
            for (left_val,right_val) in self.0.into_iter().zip(rhs.0) {
                output += left_val * right_val;
            }
            Scalar(output)
        }
    }

    impl<'a, I1: IntoIterator,I2: Iterator + EvalsToColVec<D>,O: AddAssign<O> + std::default::Default,const D: usize> Dot<EvalColVec<I2,D>> for &'a ColumnVec<I1,D> 
    where &'a I1: IntoIterator,<&'a I1 as IntoIterator>::Item: Mul<I2::Item,Output = O> {
        type Output = Scalar<O>;

        fn dot(self,rhs: EvalColVec<I2,D>) -> Self::Output {
            let mut output = <O>::default();
            for (left_val,right_val) in (&self.0).into_iter().zip(rhs.0) {
                output += left_val * right_val;
            }
            Scalar(output)
        }
    }


    impl<I1: Iterator + EvalsToColVec<D>,I2: Iterator + EvalsToColVec<D>,O: AddAssign<O> + std::default::Default,const D: usize> Dot<EvalColVec<I2,D>> for EvalColVec<I1,D> 
    where I1::Item: Mul<I2::Item,Output = O> {
        type Output = Scalar<O>;

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

        fn next(&mut self) -> Option<Self::Item> {
            self.0.next().cloned()
        }
    }

    impl<'a,I: Iterator<Item = &'a T>,T: 'a,const D: usize> EvalsToColVec<D> for ClonedColVec<'a,I,T,D> where T: Clone {
        type Output =  Vec<T>;

        fn eval(self) -> ColumnVec<Self::Output,D> {
            let mut vec = Vec::with_capacity(D);
            for output in self {
                vec.push(output)
            }
            ColumnVec(vec)
        }
    }


    pub struct CopiedColVec<'a,I: Iterator<Item = &'a T>,T: 'a,const D: usize>(I) where T: Copy;

    impl<'a,I: Iterator<Item = &'a T>,T: 'a,const D: usize> Iterator for CopiedColVec<'a,I,T,D> where T: Copy {
        type Item = T;

        fn next(&mut self) -> Option<Self::Item> {
            self.0.next().copied()
        }
    }

    impl<'a,I: Iterator<Item = &'a T>,T: 'a,const D: usize> EvalsToColVec<D> for CopiedColVec<'a,I,T,D> where T: Copy {
        type Output =  Vec<T>;

        fn eval(self) -> ColumnVec<Self::Output,D> {
            let mut vec = Vec::with_capacity(D);
            for output in self {
                vec.push(output)
            }
            ColumnVec(vec)
        }
    }


    impl<'a,I: Iterator<Item = &'a T> + EvalsToColVec<D>,T: 'a,const D: usize> EvalColVec<I,D> {
        pub fn cloned(self) -> EvalColVec<ClonedColVec<'a,I,T,D>,D> where T: Clone{
            EvalColVec(ClonedColVec(
                self.0
            ))
        }

        pub fn copied(self) -> EvalColVec<CopiedColVec<'a,I,T,D>,D> where T: Copy{
            EvalColVec(CopiedColVec(
                self.0
            ))
        }
    }

    impl<'a,I: IntoIterator<Item = &'a T>,T: 'a,const D: usize> ColumnVec<I,D> {
        pub fn cloned(self) -> EvalColVec<ClonedColVec<'a,I::IntoIter,T,D>,D> where T: Clone{
            EvalColVec(ClonedColVec(
                self.0.into_iter()
            ))
        }

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

        fn next(&mut self) -> Option<Self::Item> {
            let val = self.0.borrow_mut().iterator.next();
            self.0.borrow_mut().buffer = Some(val.clone());
            val
        }
    }

    impl<I: Iterator,const D: usize> Iterator for SecondDuplicateColVec<I,D> where I::Item: Clone {
        type Item = I::Item;

        fn next(&mut self) -> Option<Self::Item> {
            match self.0.borrow_mut().buffer {
                Some(ref mut val) => {std::mem::replace(val,None)}
                None => {panic!("math_vector Error: FirstDuplicateColVec<...> must be read before SecondDuplicateColVec<...> for each item within\n\nEither you need to switch around the order of these 2 or \none of these 2 is being fully read before the other (ie finding the dot product with one)")}
            }
        }
    }

        //TODO: remove these after iterator ops only need Iters in EvalColVec and dont require EvalsToColVec
    impl<I: Iterator,const D: usize> EvalsToColVec<D> for FirstDuplicateColVec<I,D> where I::Item: Clone {
        type Output = Vec<I::Item>;

        fn eval(self) -> ColumnVec<Self::Output,D> {
            panic!("DuplicateColVecs dont actually implement EvalsToColVec but have to because of implementation details, \nthere may one day be a method to have a iterator which clones its elements to a buffer and then outputs that")
        }
    }

    impl<I: Iterator,const D: usize> EvalsToColVec<D> for SecondDuplicateColVec<I,D> where I::Item: Clone {
        type Output = Vec<I::Item>;

        fn eval(self) -> ColumnVec<Self::Output,D> {
            panic!("DuplicateColVecs dont actually implement EvalsToColVec but have to because of implementation details, \nthere may one day be a method to have a iterator which clones its elements to a buffer and then outputs that")
        }
    }
    

    impl<I: Iterator + EvalsToColVec<D>,const D: usize> EvalColVec<I,D> where I::Item: Clone {
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
    }

    //Map
    pub struct ColVecMap<I: Iterator,F: FnMut(I::Item) -> O,O,const D: usize>(I,F);

    impl<I: Iterator,F: FnMut(I::Item) -> O,O,const D: usize> Iterator for ColVecMap<I,F,O,D> {
        type Item = O;

        fn next(&mut self) -> Option<Self::Item> {
            if let Some(val) = self.0.next() {
                Some(self.1(val))
            } else {
                None
            }
        }
    }

    impl<I: Iterator,F: FnMut(I::Item) -> O,O,const D: usize> EvalsToColVec<D> for ColVecMap<I,F,O,D> {
        type Output = Vec<O>;

        fn eval(self) -> ColumnVec<Self::Output,D> {
            let mut vec = Vec::with_capacity(D);
            for output in self {
                vec.push(output)
            }
            ColumnVec(vec)
        }
    }


    impl<I: Iterator + EvalsToColVec<D>,const D: usize> EvalColVec<I,D> {
        pub fn map<F: FnMut(I::Item) -> O,O>(self,f: F) -> ColVecMap<I,F,O,D> {
            ColVecMap(self.0,f)
        }
    }
    
    impl<I: IntoIterator,const D: usize> ColumnVec<I,D> {
        pub fn map<F: FnMut(I::Item) -> O,O>(self,f: F) -> ColVecMap<I::IntoIter,F,O,D> {
            ColVecMap(self.0.into_iter(),f)
        }

        pub fn map_ref<'a,F: FnMut(<&'a I as IntoIterator>::Item) -> O,O>(&'a self,f: F) -> ColVecMap<<&'a I as IntoIterator>::IntoIter,F,O,D> where &'a I: IntoIterator {
            ColVecMap((&self.0).into_iter(),f)
        }
    }
    
    //neg
    pub struct NegatedColVec<I: Iterator,const D: usize>(I) where I::Item: Neg;

    impl<I: Iterator,const D: usize> Iterator for NegatedColVec<I,D> where I::Item: Neg {
        type Item = <I::Item as Neg>::Output;

        fn next(&mut self) -> Option<Self::Item> {
            if let Some(val) = self.0.next() {
                Some(-val)
            } else {
                None
            }
        }
    }

    impl<I: Iterator,const D: usize> EvalsToColVec<D> for NegatedColVec<I,D> where I::Item: Neg {
        type Output = Vec<<I::Item as Neg>::Output>;

        fn eval(self) -> ColumnVec<Self::Output,D> {
            let mut vec = Vec::with_capacity(D);
            for output in self {
                vec.push(output)
            }
            ColumnVec(vec)
        }
    }


    impl<I: IntoIterator,const D: usize> Neg for ColumnVec<I,D> where I::Item: Neg {
        type Output = EvalColVec<NegatedColVec<I::IntoIter,D>,D>;

        fn neg(self) -> Self::Output {
            EvalColVec(NegatedColVec(self.0.into_iter()))
        }
    }

    impl<'a,I: IntoIterator,const D: usize> Neg for &'a ColumnVec<I,D> where &'a I: IntoIterator, <&'a I as IntoIterator>::Item: Neg{
        type Output = EvalColVec<NegatedColVec<<&'a I as IntoIterator>::IntoIter,D>,D>;

        fn neg(self) -> Self::Output {
            EvalColVec(NegatedColVec((&self.0).into_iter()))
        }
    }

    impl<I: Iterator,const D: usize> Neg for EvalColVec<I,D> where I::Item: Neg {
        type Output = EvalColVec<NegatedColVec<I,D>,D>;

        fn neg(self) -> Self::Output {
            EvalColVec(NegatedColVec(self.0))
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use super::col_vec_iterators::EvalsToColVec;

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
}