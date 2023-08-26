
#[derive(Clone,Copy)]
#[repr(transparent)]
pub struct Scalar<T>(pub T);

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

pub struct ColumnVec<T: IntoIterator,const D: usize>(T);

impl<T: IntoIterator,const D: usize> ColumnVec<T,D> {
    pub fn unwrap(self) -> T {
        self.0
    }

    pub fn transpose(self) -> RowVec<T,D> {
        RowVec(self.0)
    }
}

impl<T,const D: usize> From<[T; D]> for ColumnVec<[T;D],D> {
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

impl<Idx,T: IntoIterator + std::ops::Index<Idx>,const D: usize> std::ops::Index<Idx> for ColumnVec<T,D> where T::Output: Sized {
    type Output = Scalar<T::Output>;
    
    fn index(&self, index: Idx) -> &Self::Output {
        //SAFETY: scalar is repr(transparant) so the representaion of Scalar<T> 
        //is identical to T and as such &Scalar<T> is identical to &T or so I
        //think. DOUBLE CHECK THIS ZACH
        unsafe { std::mem::transmute(Scalar((&self.0).index(index))) }
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

pub struct EvalVec<T,const D: usize>(T);

impl<T,const D: usize> EvalVec<T,D> {
    pub fn unwrap(self) -> T {
        self.0
    }
}

pub mod col_vec_iterators {
    use std::ops::*;
    use super::*;

    pub trait EvalColVec<const D: usize> {
        type Output: IntoIterator;
    
        fn eval(self) -> ColumnVec<Self::Output,D>;
    }

    impl<T: EvalColVec<D>,const D: usize> EvalColVec<D> for EvalVec<T,D> {
        type Output = T::Output;
        
        fn eval(self) -> ColumnVec<Self::Output,D> {
            self.0.eval()
        }
    }

    macro_rules! impl_eval_col_vec {
        ($t:ty,1,$ti:ty,$tr:path) => {
            impl<I: Iterator,S: Copy,const D: usize> EvalColVec<D> for $t where $ti: $tr {
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
            impl<I1: Iterator,I2: Iterator,const D: usize> EvalColVec<D> for $t where $ti: $tr {
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
                type Output = EvalVec<$OpI<I1::IntoIter,I2::IntoIter,D>,D>;

                fn $OpF(self,other: ColumnVec<I2,D>) -> Self::Output {
                    EvalVec($OpI(
                        self.0.into_iter(),
                        other.0.into_iter()
                    ))
                }
            }

            impl<'a,I1: IntoIterator,I2: IntoIterator,const D: usize> $Op<ColumnVec<I2,D>> for &'a ColumnVec<I1,D> 
            where &'a I1: IntoIterator, <&'a I1 as IntoIterator>::Item: $Op<I2::Item> {
                type Output = EvalVec<$OpI<<&'a I1 as IntoIterator>::IntoIter,I2::IntoIter,D>,D>;

                fn $OpF(self,other: ColumnVec<I2,D>) -> Self::Output {
                    EvalVec($OpI(
                        (&self.0).into_iter(),
                        other.0.into_iter()
                    ))
                }
            }

            impl<'a,I1: IntoIterator,I2: IntoIterator,const D: usize> $Op<&'a ColumnVec<I2,D>> for ColumnVec<I1,D> 
            where &'a I2: IntoIterator, I1::Item: $Op<<&'a I2 as IntoIterator>::Item> {
                type Output = EvalVec<$OpI<I1::IntoIter,<&'a I2 as IntoIterator>::IntoIter,D>,D>;

                fn $OpF(self,other: &'a ColumnVec<I2,D>) -> Self::Output {
                    EvalVec($OpI(
                        self.0.into_iter(),
                        (&other.0).into_iter()
                    ))
                }
            }

            impl<'a,I1: IntoIterator,I2: IntoIterator,const D: usize> $Op<&'a ColumnVec<I2,D>> for &'a ColumnVec<I1,D> 
            where &'a I1: IntoIterator, &'a I2: IntoIterator, <&'a I1 as IntoIterator>::Item: $Op<<&'a I2 as IntoIterator>::Item> {
                type Output = EvalVec<$OpI<<&'a I1 as IntoIterator>::IntoIter,<&'a I2 as IntoIterator>::IntoIter,D>,D>;

                fn $OpF(self,other: &'a ColumnVec<I2,D>) -> Self::Output {
                    EvalVec($OpI(
                        (&self.0).into_iter(),
                        (&other.0).into_iter()
                    ))
                }
            }
        

            impl<I1: Iterator + EvalColVec<D>,I2: IntoIterator,const D: usize> $Op<ColumnVec<I2,D>> for EvalVec<I1,D> where I1::Item: $Op<I2::Item> {
                type Output = EvalVec<$OpI<I1,I2::IntoIter,D>,D>;

                fn $OpF(self,other: ColumnVec<I2,D>) -> Self::Output {
                    EvalVec($OpI(
                        self.0,
                        other.0.into_iter()
                    ))
                }
            }

            impl<'a,I1: Iterator + EvalColVec<D>,I2: IntoIterator,const D: usize> $Op<&'a ColumnVec<I2,D>> for EvalVec<I1,D> 
            where &'a I2: IntoIterator, I1::Item: $Op<<&'a I2 as IntoIterator>::Item> {
                type Output = EvalVec<$OpI<I1,<&'a I2 as IntoIterator>::IntoIter,D>,D>;

                fn $OpF(self,other: &'a ColumnVec<I2,D>) -> Self::Output {
                    EvalVec($OpI(
                        self.0,
                        (&other.0).into_iter()
                    ))
                }
            }

            impl<I1: IntoIterator,I2: Iterator + EvalColVec<D>,const D: usize> $Op<EvalVec<I2,D>> for ColumnVec<I1,D> where I1::Item: $Op<I2::Item> {
                type Output = EvalVec<$OpI<I1::IntoIter,I2,D>,D>;

                fn $OpF(self,other: EvalVec<I2,D>) -> Self::Output {
                    EvalVec($OpI(
                        self.0.into_iter(),
                        other.0
                    ))
                }
            }

            impl<'a,I1: IntoIterator,I2: Iterator + EvalColVec<D>,const D: usize> $Op<EvalVec<I2,D>> for &'a ColumnVec<I1,D> 
            where &'a I1: IntoIterator, <&'a I1 as IntoIterator>::Item: $Op<I2::Item> {
                type Output = EvalVec<$OpI<<&'a I1 as IntoIterator>::IntoIter,I2,D>,D>;

                fn $OpF(self,other: EvalVec<I2,D>) -> Self::Output {
                    EvalVec($OpI(
                        (&self.0).into_iter(),
                        other.0
                    ))
                }
            }


            impl<I1: Iterator + EvalColVec<D>,I2: Iterator + EvalColVec<D>,const D: usize> $Op<EvalVec<I2,D>> for EvalVec<I1,D> 
            where I1::Item: $Op<I2::Item> {
                type Output = EvalVec<$OpI<I1,I2,D>,D>;

                fn $OpF(self,other: EvalVec<I2,D>) -> Self::Output {
                    EvalVec($OpI(
                        self.0,
                        other.0
                    ))
                }
            }
        };
        ($Op:ident,$OpF:ident,$OpI:ident,0) => {
            impl<I: IntoIterator,S: Copy,const D: usize> $Op<Scalar<S>> for ColumnVec<I,D> where I::Item: $Op<S> {
                type Output = EvalVec<$OpI<I::IntoIter,S,D>,D>;

                fn $OpF(self,rhs: Scalar<S>) -> Self::Output {
                    EvalVec($OpI(
                        self.0.into_iter(),
                        rhs.0
                    ))
                }
            }

            impl<'a,I: IntoIterator,S: Copy,const D: usize> $Op<Scalar<S>> for &'a ColumnVec<I,D> 
            where &'a I: IntoIterator, <&'a I as IntoIterator>::Item: $Op<S> {
                type Output = EvalVec<$OpI<<&'a I as IntoIterator>::IntoIter,S,D>,D>;

                fn $OpF(self,rhs: Scalar<S>) -> Self::Output {
                    EvalVec($OpI(
                        (&self.0).into_iter(),
                        rhs.0
                    ))
                }
            }

            impl<'a,I: IntoIterator,S: Copy,const D: usize> $Op<&'a Scalar<S>> for ColumnVec<I,D> where I::Item: $Op<&'a S> {
                type Output = EvalVec<$OpI<I::IntoIter,&'a S,D>,D>;

                fn $OpF(self,rhs: &'a Scalar<S>) -> Self::Output {
                    EvalVec($OpI(
                        self.0.into_iter(),
                        &rhs.0
                    ))
                }
            }

            impl<'a,I: IntoIterator,S: Copy,const D: usize> $Op<&'a Scalar<S>> for &'a ColumnVec<I,D> 
            where &'a I: IntoIterator, <&'a I as IntoIterator>::Item: $Op<&'a S> {
                type Output = EvalVec<$OpI<<&'a I as IntoIterator>::IntoIter,&'a S,D>,D>;

                fn $OpF(self,rhs: &'a Scalar<S>) -> Self::Output {
                    EvalVec($OpI(
                        (&self.0).into_iter(),
                        &rhs.0
                    ))
                }
            }


            impl<I: Iterator + EvalColVec<D>,S: Copy,const D: usize> $Op<Scalar<S>> for EvalVec<I,D> where I::Item: $Op<S> {
                type Output = EvalVec<$OpI<I,S,D>,D>;

                fn $OpF(self,rhs: Scalar<S>) -> Self::Output {
                    EvalVec($OpI(
                        self.0,
                        rhs.0
                    ))
                }
            }

            impl<'a,I: Iterator + EvalColVec<D>,S: Copy,const D: usize> $Op<&'a Scalar<S>> for EvalVec<I,D> where I::Item: $Op<&'a S> {
                type Output = EvalVec<$OpI<I,&'a S,D>,D>;

                fn $OpF(self,rhs: &'a Scalar<S>) -> Self::Output {
                    EvalVec($OpI(
                        self.0,
                        &rhs.0
                    ))
                }
            }
        };
        ($Op:ident,$OpF:ident,$OpI:ident,1) => {
            impl<I: IntoIterator,S: Copy,const D: usize> $Op<ColumnVec<I,D>> for Scalar<S> where S: $Op<I::Item> {
                type Output = EvalVec<$OpI<S,I::IntoIter,D>,D>;

                fn $OpF(self,rhs: ColumnVec<I,D>) -> Self::Output {
                    EvalVec($OpI(
                        rhs.0.into_iter(),
                        self.0
                    ))
                }
            }

            impl<'a,I: IntoIterator,S: Copy,const D: usize> $Op<ColumnVec<I,D>> for &'a Scalar<S> where &'a S: $Op<I::Item> {
                type Output = EvalVec<$OpI<&'a S,I::IntoIter,D>,D>;

                fn $OpF(self,rhs: ColumnVec<I,D>) -> Self::Output {
                    EvalVec($OpI(
                        rhs.0.into_iter(),
                        &self.0
                    ))
                }
            }

            impl<'a,I: IntoIterator,S: Copy,const D: usize> $Op<&'a ColumnVec<I,D>> for Scalar<S> 
            where &'a I: IntoIterator, S: $Op<<&'a I as IntoIterator>::Item> {
                type Output = EvalVec<$OpI<S,<&'a I as IntoIterator>::IntoIter,D>,D>;

                fn $OpF(self,rhs: &'a ColumnVec<I,D>) -> Self::Output {
                    EvalVec($OpI(
                        (&rhs.0).into_iter(),
                        self.0
                    ))
                }
            }

            impl<'a,I: IntoIterator,S: Copy,const D: usize> $Op<&'a ColumnVec<I,D>> for &'a Scalar<S> 
            where &'a I: IntoIterator, &'a S: $Op<<&'a I as IntoIterator>::Item> {
                type Output = EvalVec<$OpI<&'a S,<&'a I as IntoIterator>::IntoIter,D>,D>;

                fn $OpF(self,rhs: &'a ColumnVec<I,D>) -> Self::Output {
                    EvalVec($OpI(
                        (&rhs.0).into_iter(),
                        &self.0
                    ))
                }
            }
        
            
            impl<I: Iterator + EvalColVec<D>,S: Copy,const D: usize> $Op<EvalVec<I,D>> for Scalar<S> where S: $Op<I::Item> {
                type Output = EvalVec<$OpI<S,I,D>,D>;

                fn $OpF(self,rhs: EvalVec<I,D>) -> Self::Output {
                    EvalVec($OpI(
                        rhs.0,
                        self.0
                    ))
                }
            }

            impl<'a, I: Iterator + EvalColVec<D>,S: Copy,const D: usize> $Op<EvalVec<I,D>> for &'a Scalar<S> where &'a S: $Op<I::Item> {
                type Output = EvalVec<$OpI<&'a S,I,D>,D>;

                fn $OpF(self,rhs: EvalVec<I,D>) -> Self::Output {
                    EvalVec($OpI(
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


    impl<I1: Iterator + EvalColVec<D>,I2: IntoIterator,O: AddAssign<O> + std::default::Default,const D: usize> Dot<ColumnVec<I2,D>> for EvalVec<I1,D> 
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

    impl<'a,I1: Iterator + EvalColVec<D>,I2: IntoIterator,O: AddAssign<O> + std::default::Default,const D: usize> Dot<&'a ColumnVec<I2,D>> for EvalVec<I1,D> 
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

    impl<I1: IntoIterator,I2: Iterator + EvalColVec<D>,O: AddAssign<O> + std::default::Default,const D: usize> Dot<EvalVec<I2,D>> for ColumnVec<I1,D> 
    where I1::Item: Mul<I2::Item,Output = O> {
        type Output = Scalar<O>;

        fn dot(self,rhs: EvalVec<I2,D>) -> Self::Output {
            let mut output = <O>::default();
            for (left_val,right_val) in self.0.into_iter().zip(rhs.0) {
                output += left_val * right_val;
            }
            Scalar(output)
        }
    }

    impl<'a, I1: IntoIterator,I2: Iterator + EvalColVec<D>,O: AddAssign<O> + std::default::Default,const D: usize> Dot<EvalVec<I2,D>> for &'a ColumnVec<I1,D> 
    where &'a I1: IntoIterator,<&'a I1 as IntoIterator>::Item: Mul<I2::Item,Output = O> {
        type Output = Scalar<O>;

        fn dot(self,rhs: EvalVec<I2,D>) -> Self::Output {
            let mut output = <O>::default();
            for (left_val,right_val) in (&self.0).into_iter().zip(rhs.0) {
                output += left_val * right_val;
            }
            Scalar(output)
        }
    }


    impl<I1: Iterator + EvalColVec<D>,I2: Iterator + EvalColVec<D>,O: AddAssign<O> + std::default::Default,const D: usize> Dot<EvalVec<I2,D>> for EvalVec<I1,D> 
    where I1::Item: Mul<I2::Item,Output = O> {
        type Output = Scalar<O>;

        fn dot(self,rhs: EvalVec<I2,D>) -> Self::Output {
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

    impl<'a,I: Iterator<Item = &'a T>,T: 'a,const D: usize> EvalColVec<D> for ClonedColVec<'a,I,T,D> where T: Clone {
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

    impl<'a,I: Iterator<Item = &'a T>,T: 'a,const D: usize> EvalColVec<D> for CopiedColVec<'a,I,T,D> where T: Copy {
        type Output =  Vec<T>;

        fn eval(self) -> ColumnVec<Self::Output,D> {
            let mut vec = Vec::with_capacity(D);
            for output in self {
                vec.push(output)
            }
            ColumnVec(vec)
        }
    }


    impl<'a,I: Iterator<Item = &'a T>,T: 'a,const D: usize> EvalVec<I,D> {
        pub fn cloned(self) -> EvalVec<ClonedColVec<'a,I,T,D>,D> where T: Clone{
            EvalVec(ClonedColVec(
                self.0
            ))
        }

        pub fn copied(self) -> EvalVec<CopiedColVec<'a,I,T,D>,D> where T: Copy{
            EvalVec(CopiedColVec(
                self.0
            ))
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

    impl<I: Iterator,F: FnMut(I::Item) -> O,O,const D: usize> EvalColVec<D> for ColVecMap<I,F,O,D> {
        type Output = Vec<O>;

        fn eval(self) -> ColumnVec<Self::Output,D> {
            let mut vec = Vec::with_capacity(D);
            for output in self {
                vec.push(output)
            }
            ColumnVec(vec)
        }
    }
}