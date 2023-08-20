

pub use std::ops::*;

pub struct Scalar<T: Copy>(T);


pub struct ColumnVec<I: IntoIterator,const D: usize>(pub I);

pub struct RowVec<I: IntoIterator,const D: usize>(pub I);


pub mod column_vec_math {
    use std::ops::*;

    use super::ColumnVec;
    use super::Scalar;
    //ColumnVec basic arithmetic
    pub trait EvalColumn<const D: usize>: Iterator {
        type IterType: IntoIterator;
    
        fn eval(self) -> ColumnVec<Self::IterType,D>;
    }
    
    //Addition
    pub struct ColumnVecAdd<I1: Iterator, I2: Iterator,const D: usize> where I1::Item: Add<I2::Item> {left: I1,right: I2}
    
    impl<I1: Iterator,I2: Iterator,const D: usize> Iterator for ColumnVecAdd<I1,I2,D> 
    where I1::Item: Add<I2::Item> {
        type Item = <I1::Item as Add<I2::Item>>::Output;
    
        fn next(&mut self) -> Option<Self::Item> {
            if let Some(left_out) = self.left.next() {
                if let Some(right_out) = self.right.next() {
                    return Some(left_out+right_out)
                }
                panic!("cant get the next value on 2 different sized vectors")
            }
            None
        }
    }
    
    impl<I1: Iterator,I2: Iterator,const D: usize> EvalColumn<D> for ColumnVecAdd<I1,I2,D> 
    where I1::Item: Add<I2::Item> {
        type IterType = Vec<<<I1 as Iterator>::Item as Add<I2::Item>>::Output>;
    
        fn eval(self) -> ColumnVec<Vec<<<I1 as Iterator>::Item as Add<I2::Item>>::Output>,D> {
            let mut vec = Vec::with_capacity(D);
            for (val1,val2) in self.left.zip(self.right) {
                vec.push(val1+val2)
            }
            ColumnVec(vec)
        }
    }
    
    impl<I1: IntoIterator,I2: IntoIterator,const D: usize> Add<ColumnVec<I2,D>> for ColumnVec<I1,D> 
    where I1::Item: Add<I2::Item> {
        type Output = ColumnVecAdd<I1::IntoIter,I2::IntoIter,D>;
    
        fn add(self,other: ColumnVec<I2,D>) -> Self::Output {
            ColumnVecAdd{
                left: self.0.into_iter(),
                right: other.0.into_iter()
            }
        }
    }
    
    
    impl<'a,I1: IntoIterator,I2: IntoIterator,const D: usize> Add<ColumnVec<I2,D>> for &'a ColumnVec<I1,D> 
    where &'a I1: IntoIterator, <&'a I1 as IntoIterator>::Item: Add<I2::Item> {
        type Output = ColumnVecAdd<<&'a I1 as IntoIterator>::IntoIter,I2::IntoIter,D>;
    
        fn add(self,other: ColumnVec<I2,D>) -> Self::Output {
            ColumnVecAdd{
                left: (&self.0).into_iter(),
                right: other.0.into_iter()
            }
        }
    }
    
    impl<'a,I1: IntoIterator,I2: IntoIterator,const D: usize> Add<&'a ColumnVec<I2,D>> for ColumnVec<I1,D> 
    where &'a I2: IntoIterator, I1::Item: Add<<&'a I2 as IntoIterator>::Item> {
        type Output = ColumnVecAdd<I1::IntoIter,<&'a I2 as IntoIterator>::IntoIter,D>;
    
        fn add(self,other: &'a ColumnVec<I2,D>) -> Self::Output {
            ColumnVecAdd{
                left: self.0.into_iter(),
                right: (&other.0).into_iter()
            }
        }
    }
    
    impl<'a,I1: IntoIterator,I2: IntoIterator,const D: usize> Add<&'a ColumnVec<I2,D>> for &'a ColumnVec<I1,D> 
    where &'a I1: IntoIterator, &'a I2: IntoIterator, <&'a I1 as IntoIterator>::Item: Add<<&'a I2 as IntoIterator>::Item> {
        type Output = ColumnVecAdd<<&'a I1 as IntoIterator>::IntoIter,<&'a I2 as IntoIterator>::IntoIter,D>;
    
        fn add(self,other: &'a ColumnVec<I2,D>) -> Self::Output {
            ColumnVecAdd{
                left: (&self.0).into_iter(),
                right: (&other.0).into_iter()
            }
        }
    }
    
    
    //Subtraction
    pub struct ColumnVecSub<I1: Iterator, I2: Iterator,const D: usize> where I1::Item: Sub<I2::Item> {left: I1,right: I2}
    
    impl<I1: Iterator,I2: Iterator,const D: usize> Iterator for ColumnVecSub<I1,I2,D> 
    where I1::Item: Sub<I2::Item> {
        type Item = <I1::Item as Sub<I2::Item>>::Output;
    
        fn next(&mut self) -> Option<Self::Item> {
            if let Some(left_out) = self.left.next() {
                if let Some(right_out) = self.right.next() {
                    return Some(left_out-right_out)
                }
                panic!("cant get the next value on 2 different sized vectors")
            }
            None
        }
    }
    
    impl<I1: Iterator,I2: Iterator,const D: usize> EvalColumn<D> for ColumnVecSub<I1,I2,D> 
    where I1::Item: Sub<I2::Item> {
        type IterType = Vec<<<I1 as Iterator>::Item as Sub<I2::Item>>::Output>;
    
        fn eval(self) -> ColumnVec<Vec<<<I1 as Iterator>::Item as Sub<I2::Item>>::Output>,D> {
            let mut vec = Vec::with_capacity(D);
            for (val1,val2) in self.left.zip(self.right) {
                vec.push(val1-val2)
            }
            ColumnVec(vec)
        }
    }
    
    impl<I1: IntoIterator,I2: IntoIterator,const D: usize> Sub<ColumnVec<I2,D>> for ColumnVec<I1,D> 
    where I1::Item: Sub<I2::Item> {
        type Output = ColumnVecSub<I1::IntoIter,I2::IntoIter,D>;
    
        fn sub(self,other: ColumnVec<I2,D>) -> Self::Output {
            ColumnVecSub{
                left: self.0.into_iter(),
                right: other.0.into_iter()
            }
        }
    }
    
    
    impl<'a,I1: IntoIterator,I2: IntoIterator,const D: usize> Sub<ColumnVec<I2,D>> for &'a ColumnVec<I1,D> 
    where &'a I1: IntoIterator, <&'a I1 as IntoIterator>::Item: Sub<I2::Item> {
        type Output = ColumnVecSub<<&'a I1 as IntoIterator>::IntoIter,I2::IntoIter,D>;
    
        fn sub(self,other: ColumnVec<I2,D>) -> Self::Output {
            ColumnVecSub{
                left: (&self.0).into_iter(),
                right: other.0.into_iter()
            }
        }
    }
    
    impl<'a,I1: IntoIterator,I2: IntoIterator,const D: usize> Sub<&'a ColumnVec<I2,D>> for ColumnVec<I1,D> 
    where &'a I2: IntoIterator, I1::Item: Sub<<&'a I2 as IntoIterator>::Item> {
        type Output = ColumnVecSub<I1::IntoIter,<&'a I2 as IntoIterator>::IntoIter,D>;
    
        fn sub(self,other: &'a ColumnVec<I2,D>) -> Self::Output {
            ColumnVecSub{
                left: self.0.into_iter(),
                right: (&other.0).into_iter()
            }
        }
    }
    
    impl<'a,I1: IntoIterator,I2: IntoIterator,const D: usize> Sub<&'a ColumnVec<I2,D>> for &'a ColumnVec<I1,D> 
    where &'a I1: IntoIterator, &'a I2: IntoIterator, <&'a I1 as IntoIterator>::Item: Sub<<&'a I2 as IntoIterator>::Item> {
        type Output = ColumnVecSub<<&'a I1 as IntoIterator>::IntoIter,<&'a I2 as IntoIterator>::IntoIter,D>;
    
        fn sub(self,other: &'a ColumnVec<I2,D>) -> Self::Output {
            ColumnVecSub{
                left: (&self.0).into_iter(),
                right: (&other.0).into_iter()
            }
        }
    }
    
    
    //Multiplication where Vec is first factor
    pub struct ColumnVecMulAsFirstFactor<I: Iterator, S: Copy,const D: usize> where I::Item: Mul<S> {vec: I,factor: S}
    
    impl<I: Iterator, S: Copy, const D: usize> Iterator for ColumnVecMulAsFirstFactor<I,S,D> 
    where I::Item: Mul<S> {
        type Item = <I::Item as Mul<S>>::Output;
    
        fn next(&mut self) -> Option<Self::Item> {
            if let Some(val) = self.vec.next() {
                Some(val * self.factor)
            } else {
                None
            }
        }
    }
    
    impl<I: Iterator, S: Copy, const D: usize> EvalColumn<D> for ColumnVecMulAsFirstFactor<I,S,D>
    where I::Item: Mul<S> {
        type IterType = Vec<<I::Item as Mul<S>>::Output>;
    
        fn eval(self) -> ColumnVec<Vec<<I::Item as Mul<S>>::Output>,D> {
            let mut out_vec = Vec::with_capacity(D);
            for val in self.vec {
                out_vec.push(val * self.factor)
            }
            ColumnVec(out_vec)
        }
    }
    
    impl<I: IntoIterator, S: Copy, const D: usize> Mul<Scalar<S>> for ColumnVec<I,D>
    where I::Item: Mul<S> {
        type Output = ColumnVecMulAsFirstFactor<I::IntoIter,S,D>;
    
        fn mul(self, factor: Scalar<S>) -> Self::Output {
            ColumnVecMulAsFirstFactor{
                vec: self.0.into_iter(),
                factor: factor.0
            }
        }
    }
    
    
    impl<'a,I: IntoIterator, S: Copy, const D: usize> Mul<Scalar<S>> for &'a ColumnVec<I,D> 
    where &'a I: IntoIterator, <&'a I as IntoIterator>::Item: Mul<S> {
        type Output = ColumnVecMulAsFirstFactor<<&'a I as IntoIterator>::IntoIter,S,D>;
    
        fn mul(self, factor: Scalar<S>) -> Self::Output {
            ColumnVecMulAsFirstFactor{
                vec: (&self.0).into_iter(),
                factor: factor.0
            }
        }
    }
    
    impl<'a,I: IntoIterator, S: Copy, const D: usize> Mul<&'a Scalar<S>> for ColumnVec<I,D>
    where I::Item: Mul<&'a S> {
        type Output = ColumnVecMulAsFirstFactor<I::IntoIter,&'a S,D>;
    
        fn mul(self, factor: &'a Scalar<S>) -> Self::Output {
            ColumnVecMulAsFirstFactor{
                vec: self.0.into_iter(),
                factor: &factor.0
            }
        }
    }
    
    impl<'a,I: IntoIterator, S: Copy, const D: usize> Mul<&'a Scalar<S>> for &'a ColumnVec<I,D>
    where &'a I: IntoIterator, <&'a I as IntoIterator>::Item: Mul<&'a S> {
        type Output = ColumnVecMulAsFirstFactor<<&'a I as IntoIterator>::IntoIter,&'a S,D>;
    
        fn mul(self, factor: &'a Scalar<S>) -> Self::Output {
            ColumnVecMulAsFirstFactor{
                vec: (&self.0).into_iter(),
                factor: &factor.0
            }
        }
    }
    
    
    //Multiplication where Vec is the Second factor
    pub struct ColumnVecMulAsSecondFactor<I: Iterator, S: Copy,const D: usize> where S: Mul<I::Item> {vec: I,factor: S}
    
    impl<I: Iterator, S: Copy, const D: usize> Iterator for ColumnVecMulAsSecondFactor<I,S,D> 
    where S: Mul<I::Item> {
        type Item = <S as Mul<I::Item>>::Output;
    
        fn next(&mut self) -> Option<Self::Item> {
            if let Some(val) = self.vec.next() {
                Some(self.factor * val)
            } else {
                None
            }
        }
    }
    
    impl<I: Iterator, S: Copy, const D: usize> EvalColumn<D> for ColumnVecMulAsSecondFactor<I,S,D>
    where S: Mul<I::Item> {
        type IterType = Vec<<S as Mul<I::Item>>::Output>;
    
        fn eval(self) -> ColumnVec<Vec<<S as Mul<I::Item>>::Output>,D> {
            let mut out_vec = Vec::with_capacity(D);
            for val in self.vec {
                out_vec.push(self.factor * val)
            }
            ColumnVec(out_vec)
        }
    }
    
    impl<I: IntoIterator, S: Copy, const D: usize> Mul<ColumnVec<I,D>> for Scalar<S>
    where S: Mul<I::Item> {
        type Output = ColumnVecMulAsSecondFactor<I::IntoIter,S,D>;
    
        fn mul(self, vec: ColumnVec<I,D>) -> Self::Output {
            ColumnVecMulAsSecondFactor{
                vec: vec.0.into_iter(),
                factor: self.0
            }
        }
    }
    
    
    impl<'a,I: IntoIterator, S: Copy, const D: usize> Mul<&'a ColumnVec<I,D>> for Scalar<S>
    where &'a I: IntoIterator, S: Mul<<&'a I as IntoIterator>::Item> {
        type Output = ColumnVecMulAsSecondFactor<<&'a I as IntoIterator>::IntoIter,S,D>;
    
        fn mul(self, vec: &'a ColumnVec<I,D>) -> Self::Output {
            ColumnVecMulAsSecondFactor{
                vec: (&vec.0).into_iter(),
                factor: self.0
            }
        }
    }
    
    impl<'a,I: IntoIterator, S: Copy, const D: usize> Mul<ColumnVec<I,D>> for &'a Scalar<S> 
    where &'a S: Mul<I::Item> {
        type Output = ColumnVecMulAsSecondFactor<I::IntoIter,&'a S,D>;
    
        fn mul(self, vec: ColumnVec<I,D>) -> Self::Output {
            ColumnVecMulAsSecondFactor{
                vec: vec.0.into_iter(),
                factor: &self.0
            }
        }
    }
    
    impl<'a,I: IntoIterator, S: Copy, const D: usize> Mul<&'a ColumnVec<I,D>> for &'a Scalar<S>
    where &'a I: IntoIterator, &'a S: Mul<<&'a I as IntoIterator>::Item> {
        type Output = ColumnVecMulAsSecondFactor<<&'a I as IntoIterator>::IntoIter,&'a S,D>;
    
        fn mul(self, vec: &'a ColumnVec<I,D>) -> Self::Output {
            ColumnVecMulAsSecondFactor{
                vec: (&vec.0).into_iter(),
                factor: &self.0
            }
        }
    }
    
    
    //Division where vec is dividend
    pub struct ColumnVecDivAsDividend<I: Iterator, S: Copy,const D: usize> where I::Item: Div<S> {vec: I,divisor: S}
    
    impl<I: Iterator, S: Copy, const D: usize> Iterator for ColumnVecDivAsDividend<I,S,D> 
    where I::Item: Div<S> {
        type Item = <I::Item as Div<S>>::Output;
    
        fn next(&mut self) -> Option<Self::Item> {
            if let Some(val) = self.vec.next() {
                Some(val / self.divisor)
            } else {
                None
            }
        }
    }
    
    impl<I: Iterator, S: Copy, const D: usize> EvalColumn<D> for ColumnVecDivAsDividend<I,S,D>
    where I::Item: Div<S> {
        type IterType = Vec<<I::Item as Div<S>>::Output>;
    
        fn eval(self) -> ColumnVec<Vec<<I::Item as Div<S>>::Output>,D> {
            let mut out_vec = Vec::with_capacity(D);
            for val in self.vec {
                out_vec.push(val / self.divisor)
            }
            ColumnVec(out_vec)
        }
    }
    
    impl<I: IntoIterator, S: Copy, const D: usize> Div<Scalar<S>> for ColumnVec<I,D>
    where I::Item: Div<S> {
        type Output = ColumnVecDivAsDividend<I::IntoIter,S,D>;
    
        fn div(self, divisor: Scalar<S>) -> Self::Output {
            ColumnVecDivAsDividend{
                vec: self.0.into_iter(),
                divisor: divisor.0
            }
        }
    }
    
    
    impl<'a,I: IntoIterator, S: Copy, const D: usize> Div<Scalar<S>> for &'a ColumnVec<I,D> 
    where &'a I: IntoIterator, <&'a I as IntoIterator>::Item: Div<S> {
        type Output = ColumnVecDivAsDividend<<&'a I as IntoIterator>::IntoIter,S,D>;
    
        fn div(self, divisor: Scalar<S>) -> Self::Output {
            ColumnVecDivAsDividend{
                vec: (&self.0).into_iter(),
                divisor: divisor.0
            }
        }
    }
    
    impl<'a,I: IntoIterator, S: Copy, const D: usize> Div<&'a Scalar<S>> for ColumnVec<I,D>
    where I::Item: Div<&'a S> {
        type Output = ColumnVecDivAsDividend<I::IntoIter,&'a S,D>;
    
        fn div(self, divisor: &'a Scalar<S>) -> Self::Output {
            ColumnVecDivAsDividend{
                vec: self.0.into_iter(),
                divisor: &divisor.0
            }
        }
    }
    
    impl<'a,I: IntoIterator, S: Copy, const D: usize> Div<&'a Scalar<S>> for &'a ColumnVec<I,D>
    where &'a I: IntoIterator, <&'a I as IntoIterator>::Item: Div<&'a S> {
        type Output = ColumnVecDivAsDividend<<&'a I as IntoIterator>::IntoIter,&'a S,D>;
    
        fn div(self, divisor: &'a Scalar<S>) -> Self::Output {
            ColumnVecDivAsDividend{
                vec: (&self.0).into_iter(),
                divisor: &divisor.0
            }
        }
    }
    
    
    //Division where vec is divisor
    pub struct ColumnVecDivAsDivisor<I: Iterator, S: Copy,const D: usize> where S: Div<I::Item> {vec: I,dividend: S}
    
    impl<I: Iterator, S: Copy, const D: usize> Iterator for ColumnVecDivAsDivisor<I,S,D> 
    where S: Div<I::Item> {
        type Item = <S as Div<I::Item>>::Output;
    
        fn next(&mut self) -> Option<Self::Item> {
            if let Some(val) = self.vec.next() {
                Some(self.dividend / val)
            } else {
                None
            }
        }
    }
    
    impl<I: Iterator, S: Copy, const D: usize> EvalColumn<D> for ColumnVecDivAsDivisor<I,S,D>
    where S: Div<I::Item> {
        type IterType = Vec<<S as Div<I::Item>>::Output>;
    
        fn eval(self) -> ColumnVec<Vec<<S as Div<I::Item>>::Output>,D> {
            let mut out_vec = Vec::with_capacity(D);
            for val in self.vec {
                out_vec.push(self.dividend / val)
            }
            ColumnVec(out_vec)
        }
    }
    
    impl<I: IntoIterator, S: Copy, const D: usize> Div<ColumnVec<I,D>> for Scalar<S>
    where S: Div<I::Item> {
        type Output = ColumnVecDivAsDivisor<I::IntoIter,S,D>;
    
        fn div(self, divisor: ColumnVec<I,D>) -> Self::Output {
            ColumnVecDivAsDivisor{
                vec: divisor.0.into_iter(),
                dividend: self.0
            }
        }
    }
    
    
    impl<'a,I: IntoIterator, S: Copy, const D: usize> Div<&'a ColumnVec<I,D>> for Scalar<S>
    where &'a I: IntoIterator, S: Div<<&'a I as IntoIterator>::Item> {
        type Output = ColumnVecDivAsDivisor<<&'a I as IntoIterator>::IntoIter,S,D>;
    
        fn div(self, vec: &'a ColumnVec<I,D>) -> Self::Output {
            ColumnVecDivAsDivisor{
                vec: (&vec.0).into_iter(),
                dividend: self.0
            }
        }
    }
    
    impl<'a,I: IntoIterator, S: Copy, const D: usize> Div<ColumnVec<I,D>> for &'a Scalar<S> 
    where &'a S: Div<I::Item> {
        type Output = ColumnVecDivAsDivisor<I::IntoIter,&'a S,D>;
    
        fn div(self, vec: ColumnVec<I,D>) -> Self::Output {
            ColumnVecDivAsDivisor{
                vec: vec.0.into_iter(),
                dividend: &self.0
            }
        }
    }
    
    impl<'a,I: IntoIterator, S: Copy, const D: usize> Div<&'a ColumnVec<I,D>> for &'a Scalar<S>
    where &'a I: IntoIterator, &'a S: Div<<&'a I as IntoIterator>::Item> {
        type Output = ColumnVecDivAsDivisor<<&'a I as IntoIterator>::IntoIter,&'a S,D>;
    
        fn div(self, vec: &'a ColumnVec<I,D>) -> Self::Output {
            ColumnVecDivAsDivisor{
                vec: (&vec.0).into_iter(),
                dividend: &self.0
            }
        }
    }
    
    
    //Remainder where vec is dividend
    pub struct ColumnVecRemAsDividend<I: Iterator, S: Copy,const D: usize> where I::Item: Rem<S> {vec: I,divisor: S}
    
    impl<I: Iterator, S: Copy, const D: usize> Iterator for ColumnVecRemAsDividend<I,S,D> 
    where I::Item: Rem<S> {
        type Item = <I::Item as Rem<S>>::Output;
    
        fn next(&mut self) -> Option<Self::Item> {
            if let Some(val) = self.vec.next() {
                Some(val % self.divisor)
            } else {
                None
            }
        }
    }
    
    impl<I: Iterator, S: Copy, const D: usize> EvalColumn<D> for ColumnVecRemAsDividend<I,S,D>
    where I::Item: Rem<S> {
        type IterType = Vec<<I::Item as Rem<S>>::Output>;
    
        fn eval(self) -> ColumnVec<Vec<<I::Item as Rem<S>>::Output>,D> {
            let mut out_vec = Vec::with_capacity(D);
            for val in self.vec {
                out_vec.push(val % self.divisor)
            }
            ColumnVec(out_vec)
        }
    }
    
    impl<I: IntoIterator, S: Copy, const D: usize> Rem<Scalar<S>> for ColumnVec<I,D>
    where I::Item: Rem<S> {
        type Output = ColumnVecRemAsDividend<I::IntoIter,S,D>;
    
        fn rem(self, divisor: Scalar<S>) -> Self::Output {
            ColumnVecRemAsDividend{
                vec: self.0.into_iter(),
                divisor: divisor.0
            }
        }
    }
    
    
    impl<'a,I: IntoIterator, S: Copy, const D: usize> Rem<Scalar<S>> for &'a ColumnVec<I,D> 
    where &'a I: IntoIterator, <&'a I as IntoIterator>::Item: Rem<S> {
        type Output = ColumnVecRemAsDividend<<&'a I as IntoIterator>::IntoIter,S,D>;
    
        fn rem(self, divisor: Scalar<S>) -> Self::Output {
            ColumnVecRemAsDividend{
                vec: (&self.0).into_iter(),
                divisor: divisor.0
            }
        }
    }
    
    impl<'a,I: IntoIterator, S: Copy, const D: usize> Rem<&'a Scalar<S>> for ColumnVec<I,D>
    where I::Item: Rem<&'a S> {
        type Output = ColumnVecRemAsDividend<I::IntoIter,&'a S,D>;
    
        fn rem(self, divisor: &'a Scalar<S>) -> Self::Output {
            ColumnVecRemAsDividend{
                vec: self.0.into_iter(),
                divisor: &divisor.0
            }
        }
    }
    
    impl<'a,I: IntoIterator, S: Copy, const D: usize> Rem<&'a Scalar<S>> for &'a ColumnVec<I,D>
    where &'a I: IntoIterator, <&'a I as IntoIterator>::Item: Rem<&'a S> {
        type Output = ColumnVecRemAsDividend<<&'a I as IntoIterator>::IntoIter,&'a S,D>;
    
        fn rem(self, divisor: &'a Scalar<S>) -> Self::Output {
            ColumnVecRemAsDividend{
                vec: (&self.0).into_iter(),
                divisor: &divisor.0
            }
        }
    }
    
    
    //Remainder where vec is divisor
    pub struct ColumnVecRemAsDivisor<I: Iterator, S: Copy,const D: usize> where S: Rem<I::Item> {vec: I,dividend: S}
    
    impl<I: Iterator, S: Copy, const D: usize> Iterator for ColumnVecRemAsDivisor<I,S,D> 
    where S: Rem<I::Item> {
        type Item = <S as Rem<I::Item>>::Output;
    
        fn next(&mut self) -> Option<Self::Item> {
            if let Some(val) = self.vec.next() {
                Some(self.dividend % val)
            } else {
                None
            }
        }
    }
    
    impl<I: Iterator, S: Copy, const D: usize> EvalColumn<D> for ColumnVecRemAsDivisor<I,S,D>
    where S: Rem<I::Item> {
        type IterType = Vec<<S as Rem<I::Item>>::Output>;
    
        fn eval(self) -> ColumnVec<Vec<<S as Rem<I::Item>>::Output>,D> {
            let mut out_vec = Vec::with_capacity(D);
            for val in self.vec {
                out_vec.push(self.dividend % val)
            }
            ColumnVec(out_vec)
        }
    }
    
    impl<I: IntoIterator, S: Copy, const D: usize> Rem<ColumnVec<I,D>> for Scalar<S>
    where S: Rem<I::Item> {
        type Output = ColumnVecRemAsDivisor<I::IntoIter,S,D>;
    
        fn rem(self, divisor: ColumnVec<I,D>) -> Self::Output {
            ColumnVecRemAsDivisor{
                vec: divisor.0.into_iter(),
                dividend: self.0
            }
        }
    }
    
    
    impl<'a,I: IntoIterator, S: Copy, const D: usize> Rem<&'a ColumnVec<I,D>> for Scalar<S>
    where &'a I: IntoIterator, S: Rem<<&'a I as IntoIterator>::Item> {
        type Output = ColumnVecRemAsDivisor<<&'a I as IntoIterator>::IntoIter,S,D>;
    
        fn rem(self, vec: &'a ColumnVec<I,D>) -> Self::Output {
            ColumnVecRemAsDivisor{
                vec: (&vec.0).into_iter(),
                dividend: self.0
            }
        }
    }
    
    impl<'a,I: IntoIterator, S: Copy, const D: usize> Rem<ColumnVec<I,D>> for &'a Scalar<S> 
    where &'a S: Rem<I::Item> {
        type Output = ColumnVecRemAsDivisor<I::IntoIter,&'a S,D>;
    
        fn rem(self, vec: ColumnVec<I,D>) -> Self::Output {
            ColumnVecRemAsDivisor{
                vec: vec.0.into_iter(),
                dividend: &self.0
            }
        }
    }
    
    impl<'a,I: IntoIterator, S: Copy, const D: usize> Rem<&'a ColumnVec<I,D>> for &'a Scalar<S>
    where &'a I: IntoIterator, &'a S: Rem<<&'a I as IntoIterator>::Item> {
        type Output = ColumnVecRemAsDivisor<<&'a I as IntoIterator>::IntoIter,&'a S,D>;
    
        fn rem(self, vec: &'a ColumnVec<I,D>) -> Self::Output {
            ColumnVecRemAsDivisor{
                vec: (&vec.0).into_iter(),
                dividend: &self.0
            }
        }
    }

    //Math implementations for a ColumnVec and one of these iterators

    //Ordering variation reuires violating the orphan rules, can be added once associated consts wor with generic consts
    impl<I1: IntoIterator,I2: Iterator + EvalColumn<D>,const D: usize> Add<I2> for ColumnVec<I1,D> 
    where I1::Item: Add<I2::Item> {
        type Output = ColumnVecAdd<I1::IntoIter,I2,D>;

        fn add(self, other: I2) -> Self::Output {
            ColumnVecAdd{
                left: self.0.into_iter(),
                right: other
            }
        }
    }

    impl<'a,I1: IntoIterator,I2: Iterator + EvalColumn<D>,const D: usize> Add<I2> for &'a ColumnVec<I1,D> 
    where &'a I1: IntoIterator, <&'a I1 as IntoIterator>::Item: Add<I2::Item> {
        type Output = ColumnVecAdd<<&'a I1 as IntoIterator>::IntoIter,I2,D>;

        fn add(self, other: I2) -> Self::Output {
            ColumnVecAdd{
                left: (&self.0).into_iter(),
                right: other
            }
        }
    }

    impl<I1: IntoIterator,I2: Iterator + EvalColumn<D>,const D: usize> Sub<I2> for ColumnVec<I1,D> 
    where I1::Item: Sub<I2::Item> {
        type Output = ColumnVecSub<I1::IntoIter,I2,D>;

        fn sub(self, other: I2) -> Self::Output {
            ColumnVecSub{
                left: self.0.into_iter(),
                right: other
            }
        }
    }

    impl<'a,I1: IntoIterator,I2: Iterator + EvalColumn<D>,const D: usize> Sub<I2> for &'a ColumnVec<I1,D> 
    where &'a I1: IntoIterator, <&'a I1 as IntoIterator>::Item: Sub<I2::Item> {
        type Output = ColumnVecSub<<&'a I1 as IntoIterator>::IntoIter,I2,D>;

        fn sub(self, other: I2) -> Self::Output {
            ColumnVecSub{
                left: (&self.0).into_iter(),
                right: other
            }
        }
    }

    //Math implementations for these dual combinations of these iterators

    impl<I1: Iterator, I2: Iterator, I3: Iterator + EvalColumn<D>,const D: usize> Add<I3> for ColumnVecAdd<I1,I2,D> 
    where I1::Item: Add<I2::Item>, <I1::Item as Add<I2::Item>>::Output: Add<I3::Item> {
        type Output = ColumnVecAdd<ColumnVecAdd<I1,I2,D>,I3,D>;

        fn add(self, other: I3) -> Self::Output {
            ColumnVecAdd{
                left: self,
                right: other
            }
        }
    }

    impl<I1: Iterator, I2: Iterator, I3: Iterator + EvalColumn<D>,const D: usize> Add<I3> for ColumnVecSub<I1,I2,D> 
    where I1::Item: Sub<I2::Item>, <I1::Item as Sub<I2::Item>>::Output: Add<I3::Item> {
        type Output = ColumnVecAdd<ColumnVecSub<I1,I2,D>,I3,D>;

        fn add(self, other: I3) -> Self::Output {
            ColumnVecAdd{
                left: self,
                right: other
            }
        }
    }

    impl<I1: Iterator,I2: Iterator + EvalColumn<D>,S: Copy,const D: usize> Add<I2> for ColumnVecMulAsFirstFactor<I1,S,D>
    where I1::Item: Mul<S>, <I1::Item as Mul<S>>::Output: Add<I2::Item> {
        type Output = ColumnVecAdd<ColumnVecMulAsFirstFactor<I1,S,D>,I2,D>;

        fn add(self, other: I2) -> Self::Output {
            ColumnVecAdd{
                left: self,
                right: other
            }
        }
    }

    impl<I1: Iterator,I2: Iterator + EvalColumn<D>,S: Copy,const D: usize> Add<I2> for ColumnVecMulAsSecondFactor<I1,S,D>
    where S: Mul<I1::Item>, <S as Mul<I1::Item>>::Output: Add<I2::Item> {
        type Output = ColumnVecAdd<ColumnVecMulAsSecondFactor<I1,S,D>,I2,D>;

        fn add(self, other: I2) -> Self::Output {
            ColumnVecAdd{
                left: self,
                right: other
            }
        }
    }

    impl<I1: Iterator,I2: Iterator + EvalColumn<D>,S: Copy,const D: usize> Add<I2> for ColumnVecDivAsDividend<I1,S,D>
    where I1::Item: Div<S>, <I1::Item as Div<S>>::Output: Add<I2::Item> {
        type Output = ColumnVecAdd<ColumnVecDivAsDividend<I1,S,D>,I2,D>;

        fn add(self, other: I2) -> Self::Output {
            ColumnVecAdd{
                left: self,
                right: other
            }
        }
    }

    impl<I1: Iterator,I2: Iterator + EvalColumn<D>,S: Copy,const D: usize> Add<I2> for ColumnVecDivAsDivisor<I1,S,D>
    where S: Div<I1::Item>, <S as Div<I1::Item>>::Output: Add<I2::Item> {
        type Output = ColumnVecAdd<ColumnVecDivAsDivisor<I1,S,D>,I2,D>;

        fn add(self, other: I2) -> Self::Output {
            ColumnVecAdd{
                left: self,
                right: other
            }
        }
    }

    impl<I1: Iterator,I2: Iterator + EvalColumn<D>,S: Copy,const D: usize> Add<I2> for ColumnVecRemAsDividend<I1,S,D>
    where I1::Item: Rem<S>, <I1::Item as Rem<S>>::Output: Add<I2::Item> {
        type Output = ColumnVecAdd<ColumnVecRemAsDividend<I1,S,D>,I2,D>;

        fn add(self, other: I2) -> Self::Output {
            ColumnVecAdd{
                left: self,
                right: other
            }
        }
    }

    impl<I1: Iterator,I2: Iterator + EvalColumn<D>,S: Copy,const D: usize> Add<I2> for ColumnVecRemAsDivisor<I1,S,D>
    where S: Rem<I1::Item>, <S as Rem<I1::Item>>::Output: Add<I2::Item> {
        type Output = ColumnVecAdd<ColumnVecRemAsDivisor<I1,S,D>,I2,D>;

        fn add(self, other: I2) -> Self::Output {
            ColumnVecAdd{
                left: self,
                right: other
            }
        }
    }


    impl<I1: Iterator, I2: Iterator, I3: Iterator + EvalColumn<D>,const D: usize> Sub<I3> for ColumnVecAdd<I1,I2,D> 
    where I1::Item: Add<I2::Item>, <I1::Item as Add<I2::Item>>::Output: Sub<I3::Item> {
        type Output = ColumnVecSub<ColumnVecAdd<I1,I2,D>,I3,D>;

        fn sub(self, other: I3) -> Self::Output {
            ColumnVecSub{
                left: self,
                right: other
            }
        }
    }

    impl<I1: Iterator, I2: Iterator, I3: Iterator + EvalColumn<D>,const D: usize> Sub<I3> for ColumnVecSub<I1,I2,D> 
    where I1::Item: Sub<I2::Item>, <I1::Item as Sub<I2::Item>>::Output: Sub<I3::Item> {
        type Output = ColumnVecSub<ColumnVecSub<I1,I2,D>,I3,D>;

        fn sub(self, other: I3) -> Self::Output {
            ColumnVecSub{
                left: self,
                right: other
            }
        }
    }

    impl<I1: Iterator,I2: Iterator + EvalColumn<D>,S: Copy,const D: usize> Sub<I2> for ColumnVecMulAsFirstFactor<I1,S,D>
    where I1::Item: Mul<S>, <I1::Item as Mul<S>>::Output: Sub<I2::Item> {
        type Output = ColumnVecSub<ColumnVecMulAsFirstFactor<I1,S,D>,I2,D>;

        fn sub(self, other: I2) -> Self::Output {
            ColumnVecSub{
                left: self,
                right: other
            }
        }
    }

    impl<I1: Iterator,I2: Iterator + EvalColumn<D>,S: Copy,const D: usize> Sub<I2> for ColumnVecMulAsSecondFactor<I1,S,D>
    where S: Mul<I1::Item>, <S as Mul<I1::Item>>::Output: Sub<I2::Item> {
        type Output = ColumnVecSub<ColumnVecMulAsSecondFactor<I1,S,D>,I2,D>;

        fn sub(self, other: I2) -> Self::Output {
            ColumnVecSub{
                left: self,
                right: other
            }
        }
    }

    impl<I1: Iterator,I2: Iterator + EvalColumn<D>,S: Copy,const D: usize> Sub<I2> for ColumnVecDivAsDividend<I1,S,D>
    where I1::Item: Div<S>, <I1::Item as Div<S>>::Output: Sub<I2::Item> {
        type Output = ColumnVecSub<ColumnVecDivAsDividend<I1,S,D>,I2,D>;

        fn sub(self, other: I2) -> Self::Output {
            ColumnVecSub{
                left: self,
                right: other
            }
        }
    }

    impl<I1: Iterator,I2: Iterator + EvalColumn<D>,S: Copy,const D: usize> Sub<I2> for ColumnVecDivAsDivisor<I1,S,D>
    where S: Div<I1::Item>, <S as Div<I1::Item>>::Output: Sub<I2::Item> {
        type Output = ColumnVecSub<ColumnVecDivAsDivisor<I1,S,D>,I2,D>;

        fn sub(self, other: I2) -> Self::Output {
            ColumnVecSub{
                left: self,
                right: other
            }
        }
    }

    impl<I1: Iterator,I2: Iterator + EvalColumn<D>,S: Copy,const D: usize> Sub<I2> for ColumnVecRemAsDividend<I1,S,D>
    where I1::Item: Rem<S>, <I1::Item as Rem<S>>::Output: Sub<I2::Item> {
        type Output = ColumnVecSub<ColumnVecRemAsDividend<I1,S,D>,I2,D>;

        fn sub(self, other: I2) -> Self::Output {
            ColumnVecSub{
                left: self,
                right: other
            }
        }
    }

    impl<I1: Iterator,I2: Iterator + EvalColumn<D>,S: Copy,const D: usize> Sub<I2> for ColumnVecRemAsDivisor<I1,S,D>
    where S: Rem<I1::Item>, <S as Rem<I1::Item>>::Output: Sub<I2::Item> {
        type Output = ColumnVecSub<ColumnVecRemAsDivisor<I1,S,D>,I2,D>;

        fn sub(self, other: I2) -> Self::Output {
            ColumnVecSub{
                left: self,
                right: other
            }
        }
    }

    //math implementations for one of the iterators and a scalar combo

    impl<I1: Iterator,I2: Iterator,S: Copy,const D: usize> Mul<Scalar<S>> for ColumnVecAdd<I1,I2,D>
    where I1::Item: Add<I2::Item>, <I1::Item as Add<I2::Item>>::Output: Mul<S> {
        type Output = ColumnVecMulAsFirstFactor<ColumnVecAdd<I1,I2,D>,S,D>;

        fn mul(self, factor: Scalar<S>) -> Self::Output {
            ColumnVecMulAsFirstFactor{
                vec: self,
                factor: factor.0
            }
        }
    }

    impl<'a,I1: Iterator,I2: Iterator,S: Copy, const D: usize> Mul<&'a Scalar<S>> for ColumnVecAdd<I1,I2,D>
    where I1::Item: Add<I2::Item>, <I1::Item as Add<I2::Item>>::Output: Mul<&'a S> {
        type Output = ColumnVecMulAsFirstFactor<ColumnVecAdd<I1,I2,D>,&'a S,D>;

        fn mul(self, factor: &'a Scalar<S>) -> Self::Output {
            ColumnVecMulAsFirstFactor{
                vec: self,
                factor: &factor.0
            }
        }
    }

    impl<I1: Iterator,I2: Iterator,S: Copy,const D: usize> Mul<ColumnVecAdd<I1,I2,D>> for Scalar<S>
    where I1::Item: Add<I2::Item>, S: Mul<<I1::Item as Add<I2::Item>>::Output> {
        type Output = ColumnVecMulAsSecondFactor<ColumnVecAdd<I1,I2,D>,S,D>;

        fn mul(self, vec: ColumnVecAdd<I1,I2,D>) -> Self::Output {
            ColumnVecMulAsSecondFactor{
                vec: vec,
                factor: self.0
            }
        }
    }

    impl<'a,I1: Iterator,I2: Iterator,S: Copy,const D: usize> Mul<ColumnVecAdd<I1,I2,D>> for &'a Scalar<S>
    where I1::Item: Add<I2::Item>, &'a S: Mul<<I1::Item as Add<I2::Item>>::Output> {
        type Output = ColumnVecMulAsSecondFactor<ColumnVecAdd<I1,I2,D>,&'a S,D>;

        fn mul(self, vec: ColumnVecAdd<I1,I2,D>) -> Self::Output {
            ColumnVecMulAsSecondFactor{
                vec: vec,
                factor: &self.0
            }
        }
    }


    impl<I1: Iterator,I2: Iterator,S: Copy,const D: usize> Div<Scalar<S>> for ColumnVecAdd<I1,I2,D>
    where I1::Item: Add<I2::Item>, <I1::Item as Add<I2::Item>>::Output: Div<S> {
        type Output = ColumnVecDivAsDividend<ColumnVecAdd<I1,I2,D>,S,D>;

        fn div(self, divisor: Scalar<S>) -> Self::Output {
            ColumnVecDivAsDividend{
                vec: self,
                divisor: divisor.0
            }
        }
    }

    impl<'a,I1: Iterator,I2: Iterator,S: Copy, const D: usize> Div<&'a Scalar<S>> for ColumnVecAdd<I1,I2,D>
    where I1::Item: Add<I2::Item>, <I1::Item as Add<I2::Item>>::Output: Div<&'a S> {
        type Output = ColumnVecDivAsDividend<ColumnVecAdd<I1,I2,D>,&'a S,D>;

        fn div(self, divisor: &'a Scalar<S>) -> Self::Output {
            ColumnVecDivAsDividend{
                vec: self,
                divisor: &divisor.0
            }
        }
    }

    impl<I1: Iterator,I2: Iterator,S: Copy,const D: usize> Div<ColumnVecAdd<I1,I2,D>> for Scalar<S>
    where I1::Item: Add<I2::Item>, S: Div<<I1::Item as Add<I2::Item>>::Output> {
        type Output = ColumnVecDivAsDivisor<ColumnVecAdd<I1,I2,D>,S,D>;

        fn div(self, vec: ColumnVecAdd<I1,I2,D>) -> Self::Output {
            ColumnVecDivAsDivisor{
                vec: vec,
                dividend: self.0
            }
        }
    }

    impl<'a,I1: Iterator,I2: Iterator,S: Copy,const D: usize> Div<ColumnVecAdd<I1,I2,D>> for &'a Scalar<S>
    where I1::Item: Add<I2::Item>, &'a S: Div<<I1::Item as Add<I2::Item>>::Output> {
        type Output = ColumnVecDivAsDivisor<ColumnVecAdd<I1,I2,D>,&'a S,D>;

        fn div(self, vec: ColumnVecAdd<I1,I2,D>) -> Self::Output {
            ColumnVecDivAsDivisor{
                vec: vec,
                dividend: &self.0
            }
        }
    }


    impl<I1: Iterator,I2: Iterator,S: Copy,const D: usize> Rem<Scalar<S>> for ColumnVecAdd<I1,I2,D>
    where I1::Item: Add<I2::Item>, <I1::Item as Add<I2::Item>>::Output: Rem<S> {
        type Output = ColumnVecRemAsDividend<ColumnVecAdd<I1,I2,D>,S,D>;

        fn rem(self, divisor: Scalar<S>) -> Self::Output {
            ColumnVecRemAsDividend{
                vec: self,
                divisor: divisor.0
            }
        }
    }

    impl<'a,I1: Iterator,I2: Iterator,S: Copy, const D: usize> Rem<&'a Scalar<S>> for ColumnVecAdd<I1,I2,D>
    where I1::Item: Add<I2::Item>, <I1::Item as Add<I2::Item>>::Output: Rem<&'a S> {
        type Output = ColumnVecRemAsDividend<ColumnVecAdd<I1,I2,D>,&'a S,D>;

        fn rem(self, divisor: &'a Scalar<S>) -> Self::Output {
            ColumnVecRemAsDividend{
                vec: self,
                divisor: &divisor.0
            }
        }
    }

    impl<I1: Iterator,I2: Iterator,S: Copy,const D: usize> Rem<ColumnVecAdd<I1,I2,D>> for Scalar<S>
    where I1::Item: Add<I2::Item>, S: Rem<<I1::Item as Add<I2::Item>>::Output> {
        type Output = ColumnVecRemAsDivisor<ColumnVecAdd<I1,I2,D>,S,D>;

        fn rem(self, vec: ColumnVecAdd<I1,I2,D>) -> Self::Output {
            ColumnVecRemAsDivisor{
                vec: vec,
                dividend: self.0
            }
        }
    }

    impl<'a,I1: Iterator,I2: Iterator,S: Copy,const D: usize> Rem<ColumnVecAdd<I1,I2,D>> for &'a Scalar<S>
    where I1::Item: Add<I2::Item>, &'a S: Rem<<I1::Item as Add<I2::Item>>::Output> {
        type Output = ColumnVecRemAsDivisor<ColumnVecAdd<I1,I2,D>,&'a S,D>;

        fn rem(self, vec: ColumnVecAdd<I1,I2,D>) -> Self::Output {
            ColumnVecRemAsDivisor{
                vec: vec,
                dividend: &self.0
            }
        }
    }



    impl<I1: Iterator,I2: Iterator,S: Copy,const D: usize> Mul<Scalar<S>> for ColumnVecSub<I1,I2,D>
    where I1::Item: Sub<I2::Item>, <I1::Item as Sub<I2::Item>>::Output: Mul<S> {
        type Output = ColumnVecMulAsFirstFactor<ColumnVecSub<I1,I2,D>,S,D>;

        fn mul(self, factor: Scalar<S>) -> Self::Output {
            ColumnVecMulAsFirstFactor{
                vec: self,
                factor: factor.0
            }
        }
    }

    impl<'a,I1: Iterator,I2: Iterator,S: Copy, const D: usize> Mul<&'a Scalar<S>> for ColumnVecSub<I1,I2,D>
    where I1::Item: Sub<I2::Item>, <I1::Item as Sub<I2::Item>>::Output: Mul<&'a S> {
        type Output = ColumnVecMulAsFirstFactor<ColumnVecSub<I1,I2,D>,&'a S,D>;

        fn mul(self, factor: &'a Scalar<S>) -> Self::Output {
            ColumnVecMulAsFirstFactor{
                vec: self,
                factor: &factor.0
            }
        }
    }

    impl<I1: Iterator,I2: Iterator,S: Copy,const D: usize> Mul<ColumnVecSub<I1,I2,D>> for Scalar<S>
    where I1::Item: Sub<I2::Item>, S: Mul<<I1::Item as Sub<I2::Item>>::Output> {
        type Output = ColumnVecMulAsSecondFactor<ColumnVecSub<I1,I2,D>,S,D>;

        fn mul(self, vec: ColumnVecSub<I1,I2,D>) -> Self::Output {
            ColumnVecMulAsSecondFactor{
                vec: vec,
                factor: self.0
            }
        }
    }

    impl<'a,I1: Iterator,I2: Iterator,S: Copy,const D: usize> Mul<ColumnVecSub<I1,I2,D>> for &'a Scalar<S>
    where I1::Item: Sub<I2::Item>, &'a S: Mul<<I1::Item as Sub<I2::Item>>::Output> {
        type Output = ColumnVecMulAsSecondFactor<ColumnVecSub<I1,I2,D>,&'a S,D>;

        fn mul(self, vec: ColumnVecSub<I1,I2,D>) -> Self::Output {
            ColumnVecMulAsSecondFactor{
                vec: vec,
                factor: &self.0
            }
        }
    }


    impl<I1: Iterator,I2: Iterator,S: Copy,const D: usize> Div<Scalar<S>> for ColumnVecSub<I1,I2,D>
    where I1::Item: Sub<I2::Item>, <I1::Item as Sub<I2::Item>>::Output: Div<S> {
        type Output = ColumnVecDivAsDividend<ColumnVecSub<I1,I2,D>,S,D>;

        fn div(self, divisor: Scalar<S>) -> Self::Output {
            ColumnVecDivAsDividend{
                vec: self,
                divisor: divisor.0
            }
        }
    }

    impl<'a,I1: Iterator,I2: Iterator,S: Copy, const D: usize> Div<&'a Scalar<S>> for ColumnVecSub<I1,I2,D>
    where I1::Item: Sub<I2::Item>, <I1::Item as Sub<I2::Item>>::Output: Div<&'a S> {
        type Output = ColumnVecDivAsDividend<ColumnVecSub<I1,I2,D>,&'a S,D>;

        fn div(self, divisor: &'a Scalar<S>) -> Self::Output {
            ColumnVecDivAsDividend{
                vec: self,
                divisor: &divisor.0
            }
        }
    }

    impl<I1: Iterator,I2: Iterator,S: Copy,const D: usize> Div<ColumnVecSub<I1,I2,D>> for Scalar<S>
    where I1::Item: Sub<I2::Item>, S: Div<<I1::Item as Sub<I2::Item>>::Output> {
        type Output = ColumnVecDivAsDivisor<ColumnVecSub<I1,I2,D>,S,D>;

        fn div(self, vec: ColumnVecSub<I1,I2,D>) -> Self::Output {
            ColumnVecDivAsDivisor{
                vec: vec,
                dividend: self.0
            }
        }
    }

    impl<'a,I1: Iterator,I2: Iterator,S: Copy,const D: usize> Div<ColumnVecSub<I1,I2,D>> for &'a Scalar<S>
    where I1::Item: Sub<I2::Item>, &'a S: Div<<I1::Item as Sub<I2::Item>>::Output> {
        type Output = ColumnVecDivAsDivisor<ColumnVecSub<I1,I2,D>,&'a S,D>;

        fn div(self, vec: ColumnVecSub<I1,I2,D>) -> Self::Output {
            ColumnVecDivAsDivisor{
                vec: vec,
                dividend: &self.0
            }
        }
    }


    impl<I1: Iterator,I2: Iterator,S: Copy,const D: usize> Rem<Scalar<S>> for ColumnVecSub<I1,I2,D>
    where I1::Item: Sub<I2::Item>, <I1::Item as Sub<I2::Item>>::Output: Rem<S> {
        type Output = ColumnVecRemAsDividend<ColumnVecSub<I1,I2,D>,S,D>;

        fn rem(self, divisor: Scalar<S>) -> Self::Output {
            ColumnVecRemAsDividend{
                vec: self,
                divisor: divisor.0
            }
        }
    }

    impl<'a,I1: Iterator,I2: Iterator,S: Copy, const D: usize> Rem<&'a Scalar<S>> for ColumnVecSub<I1,I2,D>
    where I1::Item: Sub<I2::Item>, <I1::Item as Sub<I2::Item>>::Output: Rem<&'a S> {
        type Output = ColumnVecRemAsDividend<ColumnVecSub<I1,I2,D>,&'a S,D>;

        fn rem(self, divisor: &'a Scalar<S>) -> Self::Output {
            ColumnVecRemAsDividend{
                vec: self,
                divisor: &divisor.0
            }
        }
    }

    impl<I1: Iterator,I2: Iterator,S: Copy,const D: usize> Rem<ColumnVecSub<I1,I2,D>> for Scalar<S>
    where I1::Item: Sub<I2::Item>, S: Rem<<I1::Item as Sub<I2::Item>>::Output> {
        type Output = ColumnVecRemAsDivisor<ColumnVecSub<I1,I2,D>,S,D>;

        fn rem(self, vec: ColumnVecSub<I1,I2,D>) -> Self::Output {
            ColumnVecRemAsDivisor{
                vec: vec,
                dividend: self.0
            }
        }
    }

    impl<'a,I1: Iterator,I2: Iterator,S: Copy,const D: usize> Rem<ColumnVecSub<I1,I2,D>> for &'a Scalar<S>
    where I1::Item: Sub<I2::Item>, &'a S: Rem<<I1::Item as Sub<I2::Item>>::Output> {
        type Output = ColumnVecRemAsDivisor<ColumnVecSub<I1,I2,D>,&'a S,D>;

        fn rem(self, vec: ColumnVecSub<I1,I2,D>) -> Self::Output {
            ColumnVecRemAsDivisor{
                vec: vec,
                dividend: &self.0
            }
        }
    }



    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<Scalar<S2>> for ColumnVecMulAsFirstFactor<I,S1,D> 
    where I::Item: Mul<S1>, <I::Item as Mul<S1>>::Output: Mul<S2> {
        type Output = ColumnVecMulAsFirstFactor<ColumnVecMulAsFirstFactor<I,S1,D>,S2,D>;

        fn mul(self, factor: Scalar<S2>) -> Self::Output {
            ColumnVecMulAsFirstFactor{
                vec: self,
                factor: factor.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<&'a Scalar<S2>> for ColumnVecMulAsFirstFactor<I,S1,D>
    where I::Item: Mul<S1>, <I::Item as Mul<S1>>::Output: Mul<&'a S2> {
        type Output = ColumnVecMulAsFirstFactor<ColumnVecMulAsFirstFactor<I,S1,D>,&'a S2,D>;

        fn mul(self, factor: &'a Scalar<S2>) -> Self::Output {
            ColumnVecMulAsFirstFactor{
                vec: self,
                factor: &factor.0
            }
        }
    }

    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<ColumnVecMulAsFirstFactor<I,S1,D>> for Scalar<S2>
    where I::Item: Mul<S1>, S2: Mul<<I::Item as Mul<S1>>::Output> {
        type Output = ColumnVecMulAsSecondFactor<ColumnVecMulAsFirstFactor<I,S1,D>,S2,D>;

        fn mul(self, vec: ColumnVecMulAsFirstFactor<I,S1,D>) -> Self::Output {
            ColumnVecMulAsSecondFactor{
                vec: vec,
                factor: self.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<ColumnVecMulAsFirstFactor<I,S1,D>> for &'a Scalar<S2>
    where I::Item: Mul<S1>, &'a S2: Mul<<I::Item as Mul<S1>>::Output> {
        type Output = ColumnVecMulAsSecondFactor<ColumnVecMulAsFirstFactor<I,S1,D>,&'a S2,D>;

        fn mul(self, vec: ColumnVecMulAsFirstFactor<I,S1,D>) -> Self::Output {
            ColumnVecMulAsSecondFactor{
                vec: vec,
                factor: &self.0
            }
        }
    }


    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<Scalar<S2>> for ColumnVecMulAsFirstFactor<I,S1,D> 
    where I::Item: Mul<S1>, <I::Item as Mul<S1>>::Output: Div<S2> {
        type Output = ColumnVecDivAsDividend<ColumnVecMulAsFirstFactor<I,S1,D>,S2,D>;

        fn div(self, divisor: Scalar<S2>) -> Self::Output {
            ColumnVecDivAsDividend{
                vec: self,
                divisor: divisor.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<&'a Scalar<S2>> for ColumnVecMulAsFirstFactor<I,S1,D>
    where I::Item: Mul<S1>, <I::Item as Mul<S1>>::Output: Div<&'a S2> {
        type Output = ColumnVecDivAsDividend<ColumnVecMulAsFirstFactor<I,S1,D>,&'a S2,D>;

        fn div(self, divisor: &'a Scalar<S2>) -> Self::Output {
            ColumnVecDivAsDividend{
                vec: self,
                divisor: &divisor.0
            }
        }
    }

    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<ColumnVecMulAsFirstFactor<I,S1,D>> for Scalar<S2>
    where I::Item: Mul<S1>, S2: Div<<I::Item as Mul<S1>>::Output> {
        type Output = ColumnVecDivAsDivisor<ColumnVecMulAsFirstFactor<I,S1,D>,S2,D>;

        fn div(self, vec: ColumnVecMulAsFirstFactor<I,S1,D>) -> Self::Output {
            ColumnVecDivAsDivisor{
                vec: vec,
                dividend: self.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<ColumnVecMulAsFirstFactor<I,S1,D>> for &'a Scalar<S2>
    where I::Item: Mul<S1>, &'a S2: Div<<I::Item as Mul<S1>>::Output> {
        type Output = ColumnVecDivAsDivisor<ColumnVecMulAsFirstFactor<I,S1,D>,&'a S2,D>;

        fn div(self, vec: ColumnVecMulAsFirstFactor<I,S1,D>) -> Self::Output {
            ColumnVecDivAsDivisor{
                vec: vec,
                dividend: &self.0
            }
        }
    }


    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<Scalar<S2>> for ColumnVecMulAsFirstFactor<I,S1,D> 
    where I::Item: Mul<S1>, <I::Item as Mul<S1>>::Output: Rem<S2> {
        type Output = ColumnVecRemAsDividend<ColumnVecMulAsFirstFactor<I,S1,D>,S2,D>;

        fn rem(self, divisor: Scalar<S2>) -> Self::Output {
            ColumnVecRemAsDividend{
                vec: self,
                divisor: divisor.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<&'a Scalar<S2>> for ColumnVecMulAsFirstFactor<I,S1,D>
    where I::Item: Mul<S1>, <I::Item as Mul<S1>>::Output: Rem<&'a S2> {
        type Output = ColumnVecRemAsDividend<ColumnVecMulAsFirstFactor<I,S1,D>,&'a S2,D>;

        fn rem(self, divisor: &'a Scalar<S2>) -> Self::Output {
            ColumnVecRemAsDividend{
                vec: self,
                divisor: &divisor.0
            }
        }
    }

    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<ColumnVecMulAsFirstFactor<I,S1,D>> for Scalar<S2>
    where I::Item: Mul<S1>, S2: Rem<<I::Item as Mul<S1>>::Output> {
        type Output = ColumnVecRemAsDivisor<ColumnVecMulAsFirstFactor<I,S1,D>,S2,D>;

        fn rem(self, vec: ColumnVecMulAsFirstFactor<I,S1,D>) -> Self::Output {
            ColumnVecRemAsDivisor{
                vec: vec,
                dividend: self.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<ColumnVecMulAsFirstFactor<I,S1,D>> for &'a Scalar<S2>
    where I::Item: Mul<S1>, &'a S2: Rem<<I::Item as Mul<S1>>::Output> {
        type Output = ColumnVecRemAsDivisor<ColumnVecMulAsFirstFactor<I,S1,D>,&'a S2,D>;

        fn rem(self, vec: ColumnVecMulAsFirstFactor<I,S1,D>) -> Self::Output {
            ColumnVecRemAsDivisor{
                vec: vec,
                dividend: &self.0
            }
        }
    }



    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<Scalar<S2>> for ColumnVecMulAsSecondFactor<I,S1,D> 
    where S1: Mul<I::Item>, <S1 as Mul<I::Item>>::Output: Mul<S2> {
        type Output = ColumnVecMulAsFirstFactor<ColumnVecMulAsSecondFactor<I,S1,D>,S2,D>;

        fn mul(self, factor: Scalar<S2>) -> Self::Output {
            ColumnVecMulAsFirstFactor{
                vec: self,
                factor: factor.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<&'a Scalar<S2>> for ColumnVecMulAsSecondFactor<I,S1,D>
    where S1: Mul<I::Item>, <S1 as Mul<I::Item>>::Output: Mul<&'a S2> {
        type Output = ColumnVecMulAsFirstFactor<ColumnVecMulAsSecondFactor<I,S1,D>,&'a S2,D>;

        fn mul(self, factor: &'a Scalar<S2>) -> Self::Output {
            ColumnVecMulAsFirstFactor{
                vec: self,
                factor: &factor.0
            }
        }
    }

    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<ColumnVecMulAsSecondFactor<I,S1,D>> for Scalar<S2>
    where S1: Mul<I::Item>, S2: Mul<<S1 as Mul<I::Item>>::Output> {
        type Output = ColumnVecMulAsSecondFactor<ColumnVecMulAsSecondFactor<I,S1,D>,S2,D>;

        fn mul(self, vec: ColumnVecMulAsSecondFactor<I,S1,D>) -> Self::Output {
            ColumnVecMulAsSecondFactor{
                vec: vec,
                factor: self.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<ColumnVecMulAsSecondFactor<I,S1,D>> for &'a Scalar<S2>
    where S1: Mul<I::Item>, &'a S2: Mul<<S1 as Mul<I::Item>>::Output> {
        type Output = ColumnVecMulAsSecondFactor<ColumnVecMulAsSecondFactor<I,S1,D>,&'a S2,D>;

        fn mul(self, vec: ColumnVecMulAsSecondFactor<I,S1,D>) -> Self::Output {
            ColumnVecMulAsSecondFactor{
                vec: vec,
                factor: &self.0
            }
        }
    }


    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<Scalar<S2>> for ColumnVecMulAsSecondFactor<I,S1,D> 
    where S1: Mul<I::Item>, <S1 as Mul<I::Item>>::Output: Div<S2> {
        type Output = ColumnVecDivAsDividend<ColumnVecMulAsSecondFactor<I,S1,D>,S2,D>;

        fn div(self, divisor: Scalar<S2>) -> Self::Output {
            ColumnVecDivAsDividend{
                vec: self,
                divisor: divisor.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<&'a Scalar<S2>> for ColumnVecMulAsSecondFactor<I,S1,D>
    where S1: Mul<I::Item>, <S1 as Mul<I::Item>>::Output: Div<&'a S2> {
        type Output = ColumnVecDivAsDividend<ColumnVecMulAsSecondFactor<I,S1,D>,&'a S2,D>;

        fn div(self, divisor: &'a Scalar<S2>) -> Self::Output {
            ColumnVecDivAsDividend{
                vec: self,
                divisor: &divisor.0
            }
        }
    }

    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<ColumnVecMulAsSecondFactor<I,S1,D>> for Scalar<S2>
    where S1: Mul<I::Item>, S2: Div<<S1 as Mul<I::Item>>::Output> {
        type Output = ColumnVecDivAsDivisor<ColumnVecMulAsSecondFactor<I,S1,D>,S2,D>;

        fn div(self, vec: ColumnVecMulAsSecondFactor<I,S1,D>) -> Self::Output {
            ColumnVecDivAsDivisor{
                vec: vec,
                dividend: self.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<ColumnVecMulAsSecondFactor<I,S1,D>> for &'a Scalar<S2>
    where S1: Mul<I::Item>, &'a S2: Div<<S1 as Mul<I::Item>>::Output> {
        type Output = ColumnVecDivAsDivisor<ColumnVecMulAsSecondFactor<I,S1,D>,&'a S2,D>;

        fn div(self, vec: ColumnVecMulAsSecondFactor<I,S1,D>) -> Self::Output {
            ColumnVecDivAsDivisor{
                vec: vec,
                dividend: &self.0
            }
        }
    }


    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<Scalar<S2>> for ColumnVecMulAsSecondFactor<I,S1,D> 
    where S1: Mul<I::Item>, <S1 as Mul<I::Item>>::Output: Rem<S2> {
        type Output = ColumnVecRemAsDividend<ColumnVecMulAsSecondFactor<I,S1,D>,S2,D>;

        fn rem(self, divisor: Scalar<S2>) -> Self::Output {
            ColumnVecRemAsDividend{
                vec: self,
                divisor: divisor.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<&'a Scalar<S2>> for ColumnVecMulAsSecondFactor<I,S1,D>
    where S1: Mul<I::Item>, <S1 as Mul<I::Item>>::Output: Rem<&'a S2> {
        type Output = ColumnVecRemAsDividend<ColumnVecMulAsSecondFactor<I,S1,D>,&'a S2,D>;

        fn rem(self, divisor: &'a Scalar<S2>) -> Self::Output {
            ColumnVecRemAsDividend{
                vec: self,
                divisor: &divisor.0
            }
        }
    }

    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<ColumnVecMulAsSecondFactor<I,S1,D>> for Scalar<S2>
    where S1: Mul<I::Item>, S2: Rem<<S1 as Mul<I::Item>>::Output> {
        type Output = ColumnVecRemAsDivisor<ColumnVecMulAsSecondFactor<I,S1,D>,S2,D>;

        fn rem(self, vec: ColumnVecMulAsSecondFactor<I,S1,D>) -> Self::Output {
            ColumnVecRemAsDivisor{
                vec: vec,
                dividend: self.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<ColumnVecMulAsSecondFactor<I,S1,D>> for &'a Scalar<S2>
    where S1: Mul<I::Item>, &'a S2: Rem<<S1 as Mul<I::Item>>::Output> {
        type Output = ColumnVecRemAsDivisor<ColumnVecMulAsSecondFactor<I,S1,D>,&'a S2,D>;

        fn rem(self, vec: ColumnVecMulAsSecondFactor<I,S1,D>) -> Self::Output {
            ColumnVecRemAsDivisor{
                vec: vec,
                dividend: &self.0
            }
        }
    }



    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<Scalar<S2>> for ColumnVecDivAsDividend<I,S1,D> 
    where I::Item: Div<S1>, <I::Item as Div<S1>>::Output: Mul<S2> {
        type Output = ColumnVecMulAsFirstFactor<ColumnVecDivAsDividend<I,S1,D>,S2,D>;

        fn mul(self, factor: Scalar<S2>) -> Self::Output {
            ColumnVecMulAsFirstFactor{
                vec: self,
                factor: factor.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<&'a Scalar<S2>> for ColumnVecDivAsDividend<I,S1,D>
    where I::Item: Div<S1>, <I::Item as Div<S1>>::Output: Mul<&'a S2> {
        type Output = ColumnVecMulAsFirstFactor<ColumnVecDivAsDividend<I,S1,D>,&'a S2,D>;

        fn mul(self, factor: &'a Scalar<S2>) -> Self::Output {
            ColumnVecMulAsFirstFactor{
                vec: self,
                factor: &factor.0
            }
        }
    }

    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<ColumnVecDivAsDividend<I,S1,D>> for Scalar<S2>
    where I::Item: Div<S1>, S2: Mul<<I::Item as Div<S1>>::Output> {
        type Output = ColumnVecMulAsSecondFactor<ColumnVecDivAsDividend<I,S1,D>,S2,D>;

        fn mul(self, vec: ColumnVecDivAsDividend<I,S1,D>) -> Self::Output {
            ColumnVecMulAsSecondFactor{
                vec: vec,
                factor: self.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<ColumnVecDivAsDividend<I,S1,D>> for &'a Scalar<S2>
    where I::Item: Div<S1>, &'a S2: Mul<<I::Item as Div<S1>>::Output> {
        type Output = ColumnVecMulAsSecondFactor<ColumnVecDivAsDividend<I,S1,D>,&'a S2,D>;

        fn mul(self, vec: ColumnVecDivAsDividend<I,S1,D>) -> Self::Output {
            ColumnVecMulAsSecondFactor{
                vec: vec,
                factor: &self.0
            }
        }
    }


    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<Scalar<S2>> for ColumnVecDivAsDividend<I,S1,D> 
    where I::Item: Div<S1>, <I::Item as Div<S1>>::Output: Div<S2> {
        type Output = ColumnVecDivAsDividend<ColumnVecDivAsDividend<I,S1,D>,S2,D>;

        fn div(self, divisor: Scalar<S2>) -> Self::Output {
            ColumnVecDivAsDividend{
                vec: self,
                divisor: divisor.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<&'a Scalar<S2>> for ColumnVecDivAsDividend<I,S1,D>
    where I::Item: Div<S1>, <I::Item as Div<S1>>::Output: Div<&'a S2> {
        type Output = ColumnVecDivAsDividend<ColumnVecDivAsDividend<I,S1,D>,&'a S2,D>;

        fn div(self, divisor: &'a Scalar<S2>) -> Self::Output {
            ColumnVecDivAsDividend{
                vec: self,
                divisor: &divisor.0
            }
        }
    }

    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<ColumnVecDivAsDividend<I,S1,D>> for Scalar<S2>
    where I::Item: Div<S1>, S2: Div<<I::Item as Div<S1>>::Output> {
        type Output = ColumnVecDivAsDivisor<ColumnVecDivAsDividend<I,S1,D>,S2,D>;

        fn div(self, vec: ColumnVecDivAsDividend<I,S1,D>) -> Self::Output {
            ColumnVecDivAsDivisor{
                vec: vec,
                dividend: self.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<ColumnVecDivAsDividend<I,S1,D>> for &'a Scalar<S2>
    where I::Item: Div<S1>, &'a S2: Div<<I::Item as Div<S1>>::Output> {
        type Output = ColumnVecDivAsDivisor<ColumnVecDivAsDividend<I,S1,D>,&'a S2,D>;

        fn div(self, vec: ColumnVecDivAsDividend<I,S1,D>) -> Self::Output {
            ColumnVecDivAsDivisor{
                vec: vec,
                dividend: &self.0
            }
        }
    }


    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<Scalar<S2>> for ColumnVecDivAsDividend<I,S1,D> 
    where I::Item: Div<S1>, <I::Item as Div<S1>>::Output: Rem<S2> {
        type Output = ColumnVecRemAsDividend<ColumnVecDivAsDividend<I,S1,D>,S2,D>;

        fn rem(self, divisor: Scalar<S2>) -> Self::Output {
            ColumnVecRemAsDividend{
                vec: self,
                divisor: divisor.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<&'a Scalar<S2>> for ColumnVecDivAsDividend<I,S1,D>
    where I::Item: Div<S1>, <I::Item as Div<S1>>::Output: Rem<&'a S2> {
        type Output = ColumnVecRemAsDividend<ColumnVecDivAsDividend<I,S1,D>,&'a S2,D>;

        fn rem(self, divisor: &'a Scalar<S2>) -> Self::Output {
            ColumnVecRemAsDividend{
                vec: self,
                divisor: &divisor.0
            }
        }
    }

    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<ColumnVecDivAsDividend<I,S1,D>> for Scalar<S2>
    where I::Item: Div<S1>, S2: Rem<<I::Item as Div<S1>>::Output> {
        type Output = ColumnVecRemAsDivisor<ColumnVecDivAsDividend<I,S1,D>,S2,D>;

        fn rem(self, vec: ColumnVecDivAsDividend<I,S1,D>) -> Self::Output {
            ColumnVecRemAsDivisor{
                vec: vec,
                dividend: self.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<ColumnVecDivAsDividend<I,S1,D>> for &'a Scalar<S2>
    where I::Item: Div<S1>, &'a S2: Rem<<I::Item as Div<S1>>::Output> {
        type Output = ColumnVecRemAsDivisor<ColumnVecDivAsDividend<I,S1,D>,&'a S2,D>;

        fn rem(self, vec: ColumnVecDivAsDividend<I,S1,D>) -> Self::Output {
            ColumnVecRemAsDivisor{
                vec: vec,
                dividend: &self.0
            }
        }
    }



    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<Scalar<S2>> for ColumnVecDivAsDivisor<I,S1,D> 
    where S1: Div<I::Item>, <S1 as Div<I::Item>>::Output: Mul<S2> {
        type Output = ColumnVecMulAsFirstFactor<ColumnVecDivAsDivisor<I,S1,D>,S2,D>;

        fn mul(self, factor: Scalar<S2>) -> Self::Output {
            ColumnVecMulAsFirstFactor{
                vec: self,
                factor: factor.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<&'a Scalar<S2>> for ColumnVecDivAsDivisor<I,S1,D>
    where S1: Div<I::Item>, <S1 as Div<I::Item>>::Output: Mul<&'a S2> {
        type Output = ColumnVecMulAsFirstFactor<ColumnVecDivAsDivisor<I,S1,D>,&'a S2,D>;

        fn mul(self, factor: &'a Scalar<S2>) -> Self::Output {
            ColumnVecMulAsFirstFactor{
                vec: self,
                factor: &factor.0
            }
        }
    }

    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<ColumnVecDivAsDivisor<I,S1,D>> for Scalar<S2>
    where S1: Div<I::Item>, S2: Mul<<S1 as Div<I::Item>>::Output> {
        type Output = ColumnVecMulAsSecondFactor<ColumnVecDivAsDivisor<I,S1,D>,S2,D>;

        fn mul(self, vec: ColumnVecDivAsDivisor<I,S1,D>) -> Self::Output {
            ColumnVecMulAsSecondFactor{
                vec: vec,
                factor: self.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<ColumnVecDivAsDivisor<I,S1,D>> for &'a Scalar<S2>
    where S1: Div<I::Item>, &'a S2: Mul<<S1 as Div<I::Item>>::Output> {
        type Output = ColumnVecMulAsSecondFactor<ColumnVecDivAsDivisor<I,S1,D>,&'a S2,D>;

        fn mul(self, vec: ColumnVecDivAsDivisor<I,S1,D>) -> Self::Output {
            ColumnVecMulAsSecondFactor{
                vec: vec,
                factor: &self.0
            }
        }
    }


    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<Scalar<S2>> for ColumnVecDivAsDivisor<I,S1,D> 
    where S1: Div<I::Item>, <S1 as Div<I::Item>>::Output: Div<S2> {
        type Output = ColumnVecDivAsDividend<ColumnVecDivAsDivisor<I,S1,D>,S2,D>;

        fn div(self, divisor: Scalar<S2>) -> Self::Output {
            ColumnVecDivAsDividend{
                vec: self,
                divisor: divisor.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<&'a Scalar<S2>> for ColumnVecDivAsDivisor<I,S1,D>
    where S1: Div<I::Item>, <S1 as Div<I::Item>>::Output: Div<&'a S2> {
        type Output = ColumnVecDivAsDividend<ColumnVecDivAsDivisor<I,S1,D>,&'a S2,D>;

        fn div(self, divisor: &'a Scalar<S2>) -> Self::Output {
            ColumnVecDivAsDividend{
                vec: self,
                divisor: &divisor.0
            }
        }
    }

    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<ColumnVecDivAsDivisor<I,S1,D>> for Scalar<S2>
    where S1: Div<I::Item>, S2: Div<<S1 as Div<I::Item>>::Output> {
        type Output = ColumnVecDivAsDivisor<ColumnVecDivAsDivisor<I,S1,D>,S2,D>;

        fn div(self, vec: ColumnVecDivAsDivisor<I,S1,D>) -> Self::Output {
            ColumnVecDivAsDivisor{
                vec: vec,
                dividend: self.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<ColumnVecDivAsDivisor<I,S1,D>> for &'a Scalar<S2>
    where S1: Div<I::Item>, &'a S2: Div<<S1 as Div<I::Item>>::Output> {
        type Output = ColumnVecDivAsDivisor<ColumnVecDivAsDivisor<I,S1,D>,&'a S2,D>;

        fn div(self, vec: ColumnVecDivAsDivisor<I,S1,D>) -> Self::Output {
            ColumnVecDivAsDivisor{
                vec: vec,
                dividend: &self.0
            }
        }
    }


    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<Scalar<S2>> for ColumnVecDivAsDivisor<I,S1,D> 
    where S1: Div<I::Item>, <S1 as Div<I::Item>>::Output: Rem<S2> {
        type Output = ColumnVecRemAsDividend<ColumnVecDivAsDivisor<I,S1,D>,S2,D>;

        fn rem(self, divisor: Scalar<S2>) -> Self::Output {
            ColumnVecRemAsDividend{
                vec: self,
                divisor: divisor.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<&'a Scalar<S2>> for ColumnVecDivAsDivisor<I,S1,D>
    where S1: Div<I::Item>, <S1 as Div<I::Item>>::Output: Rem<&'a S2> {
        type Output = ColumnVecRemAsDividend<ColumnVecDivAsDivisor<I,S1,D>,&'a S2,D>;

        fn rem(self, divisor: &'a Scalar<S2>) -> Self::Output {
            ColumnVecRemAsDividend{
                vec: self,
                divisor: &divisor.0
            }
        }
    }

    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<ColumnVecDivAsDivisor<I,S1,D>> for Scalar<S2>
    where S1: Div<I::Item>, S2: Rem<<S1 as Div<I::Item>>::Output> {
        type Output = ColumnVecRemAsDivisor<ColumnVecDivAsDivisor<I,S1,D>,S2,D>;

        fn rem(self, vec: ColumnVecDivAsDivisor<I,S1,D>) -> Self::Output {
            ColumnVecRemAsDivisor{
                vec: vec,
                dividend: self.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<ColumnVecDivAsDivisor<I,S1,D>> for &'a Scalar<S2>
    where S1: Div<I::Item>, &'a S2: Rem<<S1 as Div<I::Item>>::Output> {
        type Output = ColumnVecRemAsDivisor<ColumnVecDivAsDivisor<I,S1,D>,&'a S2,D>;

        fn rem(self, vec: ColumnVecDivAsDivisor<I,S1,D>) -> Self::Output {
            ColumnVecRemAsDivisor{
                vec: vec,
                dividend: &self.0
            }
        }
    }



    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<Scalar<S2>> for ColumnVecRemAsDividend<I,S1,D> 
    where I::Item: Rem<S1>, <I::Item as Rem<S1>>::Output: Mul<S2> {
        type Output = ColumnVecMulAsFirstFactor<ColumnVecRemAsDividend<I,S1,D>,S2,D>;

        fn mul(self, factor: Scalar<S2>) -> Self::Output {
            ColumnVecMulAsFirstFactor{
                vec: self,
                factor: factor.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<&'a Scalar<S2>> for ColumnVecRemAsDividend<I,S1,D>
    where I::Item: Rem<S1>, <I::Item as Rem<S1>>::Output: Mul<&'a S2> {
        type Output = ColumnVecMulAsFirstFactor<ColumnVecRemAsDividend<I,S1,D>,&'a S2,D>;

        fn mul(self, factor: &'a Scalar<S2>) -> Self::Output {
            ColumnVecMulAsFirstFactor{
                vec: self,
                factor: &factor.0
            }
        }
    }

    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<ColumnVecRemAsDividend<I,S1,D>> for Scalar<S2>
    where I::Item: Rem<S1>, S2: Mul<<I::Item as Rem<S1>>::Output> {
        type Output = ColumnVecMulAsSecondFactor<ColumnVecRemAsDividend<I,S1,D>,S2,D>;

        fn mul(self, vec: ColumnVecRemAsDividend<I,S1,D>) -> Self::Output {
            ColumnVecMulAsSecondFactor{
                vec: vec,
                factor: self.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<ColumnVecRemAsDividend<I,S1,D>> for &'a Scalar<S2>
    where I::Item: Rem<S1>, &'a S2: Mul<<I::Item as Rem<S1>>::Output> {
        type Output = ColumnVecMulAsSecondFactor<ColumnVecRemAsDividend<I,S1,D>,&'a S2,D>;

        fn mul(self, vec: ColumnVecRemAsDividend<I,S1,D>) -> Self::Output {
            ColumnVecMulAsSecondFactor{
                vec: vec,
                factor: &self.0
            }
        }
    }


    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<Scalar<S2>> for ColumnVecRemAsDividend<I,S1,D> 
    where I::Item: Rem<S1>, <I::Item as Rem<S1>>::Output: Div<S2> {
        type Output = ColumnVecDivAsDividend<ColumnVecRemAsDividend<I,S1,D>,S2,D>;

        fn div(self, divisor: Scalar<S2>) -> Self::Output {
            ColumnVecDivAsDividend{
                vec: self,
                divisor: divisor.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<&'a Scalar<S2>> for ColumnVecRemAsDividend<I,S1,D>
    where I::Item: Rem<S1>, <I::Item as Rem<S1>>::Output: Div<&'a S2> {
        type Output = ColumnVecDivAsDividend<ColumnVecRemAsDividend<I,S1,D>,&'a S2,D>;

        fn div(self, divisor: &'a Scalar<S2>) -> Self::Output {
            ColumnVecDivAsDividend{
                vec: self,
                divisor: &divisor.0
            }
        }
    }

    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<ColumnVecRemAsDividend<I,S1,D>> for Scalar<S2>
    where I::Item: Rem<S1>, S2: Div<<I::Item as Rem<S1>>::Output> {
        type Output = ColumnVecDivAsDivisor<ColumnVecRemAsDividend<I,S1,D>,S2,D>;

        fn div(self, vec: ColumnVecRemAsDividend<I,S1,D>) -> Self::Output {
            ColumnVecDivAsDivisor{
                vec: vec,
                dividend: self.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<ColumnVecRemAsDividend<I,S1,D>> for &'a Scalar<S2>
    where I::Item: Rem<S1>, &'a S2: Div<<I::Item as Rem<S1>>::Output> {
        type Output = ColumnVecDivAsDivisor<ColumnVecRemAsDividend<I,S1,D>,&'a S2,D>;

        fn div(self, vec: ColumnVecRemAsDividend<I,S1,D>) -> Self::Output {
            ColumnVecDivAsDivisor{
                vec: vec,
                dividend: &self.0
            }
        }
    }


    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<Scalar<S2>> for ColumnVecRemAsDividend<I,S1,D> 
    where I::Item: Rem<S1>, <I::Item as Rem<S1>>::Output: Rem<S2> {
        type Output = ColumnVecRemAsDividend<ColumnVecRemAsDividend<I,S1,D>,S2,D>;

        fn rem(self, divisor: Scalar<S2>) -> Self::Output {
            ColumnVecRemAsDividend{
                vec: self,
                divisor: divisor.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<&'a Scalar<S2>> for ColumnVecRemAsDividend<I,S1,D>
    where I::Item: Rem<S1>, <I::Item as Rem<S1>>::Output: Rem<&'a S2> {
        type Output = ColumnVecRemAsDividend<ColumnVecRemAsDividend<I,S1,D>,&'a S2,D>;

        fn rem(self, divisor: &'a Scalar<S2>) -> Self::Output {
            ColumnVecRemAsDividend{
                vec: self,
                divisor: &divisor.0
            }
        }
    }

    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<ColumnVecRemAsDividend<I,S1,D>> for Scalar<S2>
    where I::Item: Rem<S1>, S2: Rem<<I::Item as Rem<S1>>::Output> {
        type Output = ColumnVecRemAsDivisor<ColumnVecRemAsDividend<I,S1,D>,S2,D>;

        fn rem(self, vec: ColumnVecRemAsDividend<I,S1,D>) -> Self::Output {
            ColumnVecRemAsDivisor{
                vec: vec,
                dividend: self.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<ColumnVecRemAsDividend<I,S1,D>> for &'a Scalar<S2>
    where I::Item: Rem<S1>, &'a S2: Rem<<I::Item as Rem<S1>>::Output> {
        type Output = ColumnVecRemAsDivisor<ColumnVecRemAsDividend<I,S1,D>,&'a S2,D>;

        fn rem(self, vec: ColumnVecRemAsDividend<I,S1,D>) -> Self::Output {
            ColumnVecRemAsDivisor{
                vec: vec,
                dividend: &self.0
            }
        }
    }



    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<Scalar<S2>> for ColumnVecRemAsDivisor<I,S1,D> 
    where S1: Rem<I::Item>, <S1 as Rem<I::Item>>::Output: Mul<S2> {
        type Output = ColumnVecMulAsFirstFactor<ColumnVecRemAsDivisor<I,S1,D>,S2,D>;

        fn mul(self, factor: Scalar<S2>) -> Self::Output {
            ColumnVecMulAsFirstFactor{
                vec: self,
                factor: factor.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<&'a Scalar<S2>> for ColumnVecRemAsDivisor<I,S1,D>
    where S1: Rem<I::Item>, <S1 as Rem<I::Item>>::Output: Mul<&'a S2> {
        type Output = ColumnVecMulAsFirstFactor<ColumnVecRemAsDivisor<I,S1,D>,&'a S2,D>;

        fn mul(self, factor: &'a Scalar<S2>) -> Self::Output {
            ColumnVecMulAsFirstFactor{
                vec: self,
                factor: &factor.0
            }
        }
    }

    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<ColumnVecRemAsDivisor<I,S1,D>> for Scalar<S2>
    where S1: Rem<I::Item>, S2: Mul<<S1 as Rem<I::Item>>::Output> {
        type Output = ColumnVecMulAsSecondFactor<ColumnVecRemAsDivisor<I,S1,D>,S2,D>;

        fn mul(self, vec: ColumnVecRemAsDivisor<I,S1,D>) -> Self::Output {
            ColumnVecMulAsSecondFactor{
                vec: vec,
                factor: self.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Mul<ColumnVecRemAsDivisor<I,S1,D>> for &'a Scalar<S2>
    where S1: Rem<I::Item>, &'a S2: Mul<<S1 as Rem<I::Item>>::Output> {
        type Output = ColumnVecMulAsSecondFactor<ColumnVecRemAsDivisor<I,S1,D>,&'a S2,D>;

        fn mul(self, vec: ColumnVecRemAsDivisor<I,S1,D>) -> Self::Output {
            ColumnVecMulAsSecondFactor{
                vec: vec,
                factor: &self.0
            }
        }
    }


    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<Scalar<S2>> for ColumnVecRemAsDivisor<I,S1,D> 
    where S1: Rem<I::Item>, <S1 as Rem<I::Item>>::Output: Div<S2> {
        type Output = ColumnVecDivAsDividend<ColumnVecRemAsDivisor<I,S1,D>,S2,D>;

        fn div(self, divisor: Scalar<S2>) -> Self::Output {
            ColumnVecDivAsDividend{
                vec: self,
                divisor: divisor.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<&'a Scalar<S2>> for ColumnVecRemAsDivisor<I,S1,D>
    where S1: Rem<I::Item>, <S1 as Rem<I::Item>>::Output: Div<&'a S2> {
        type Output = ColumnVecDivAsDividend<ColumnVecRemAsDivisor<I,S1,D>,&'a S2,D>;

        fn div(self, divisor: &'a Scalar<S2>) -> Self::Output {
            ColumnVecDivAsDividend{
                vec: self,
                divisor: &divisor.0
            }
        }
    }

    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<ColumnVecRemAsDivisor<I,S1,D>> for Scalar<S2>
    where S1: Rem<I::Item>, S2: Div<<S1 as Rem<I::Item>>::Output> {
        type Output = ColumnVecDivAsDivisor<ColumnVecRemAsDivisor<I,S1,D>,S2,D>;

        fn div(self, vec: ColumnVecRemAsDivisor<I,S1,D>) -> Self::Output {
            ColumnVecDivAsDivisor{
                vec: vec,
                dividend: self.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Div<ColumnVecRemAsDivisor<I,S1,D>> for &'a Scalar<S2>
    where S1: Rem<I::Item>, &'a S2: Div<<S1 as Rem<I::Item>>::Output> {
        type Output = ColumnVecDivAsDivisor<ColumnVecRemAsDivisor<I,S1,D>,&'a S2,D>;

        fn div(self, vec: ColumnVecRemAsDivisor<I,S1,D>) -> Self::Output {
            ColumnVecDivAsDivisor{
                vec: vec,
                dividend: &self.0
            }
        }
    }


    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<Scalar<S2>> for ColumnVecRemAsDivisor<I,S1,D> 
    where S1: Rem<I::Item>, <S1 as Rem<I::Item>>::Output: Rem<S2> {
        type Output = ColumnVecRemAsDividend<ColumnVecRemAsDivisor<I,S1,D>,S2,D>;

        fn rem(self, divisor: Scalar<S2>) -> Self::Output {
            ColumnVecRemAsDividend{
                vec: self,
                divisor: divisor.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<&'a Scalar<S2>> for ColumnVecRemAsDivisor<I,S1,D>
    where S1: Rem<I::Item>, <S1 as Rem<I::Item>>::Output: Rem<&'a S2> {
        type Output = ColumnVecRemAsDividend<ColumnVecRemAsDivisor<I,S1,D>,&'a S2,D>;

        fn rem(self, divisor: &'a Scalar<S2>) -> Self::Output {
            ColumnVecRemAsDividend{
                vec: self,
                divisor: &divisor.0
            }
        }
    }

    impl<I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<ColumnVecRemAsDivisor<I,S1,D>> for Scalar<S2>
    where S1: Rem<I::Item>, S2: Rem<<S1 as Rem<I::Item>>::Output> {
        type Output = ColumnVecRemAsDivisor<ColumnVecRemAsDivisor<I,S1,D>,S2,D>;

        fn rem(self, vec: ColumnVecRemAsDivisor<I,S1,D>) -> Self::Output {
            ColumnVecRemAsDivisor{
                vec: vec,
                dividend: self.0
            }
        }
    }

    impl<'a,I: Iterator,S1: Copy,S2: Copy,const D: usize> Rem<ColumnVecRemAsDivisor<I,S1,D>> for &'a Scalar<S2>
    where S1: Rem<I::Item>, &'a S2: Rem<<S1 as Rem<I::Item>>::Output> {
        type Output = ColumnVecRemAsDivisor<ColumnVecRemAsDivisor<I,S1,D>,&'a S2,D>;

        fn rem(self, vec: ColumnVecRemAsDivisor<I,S1,D>) -> Self::Output {
            ColumnVecRemAsDivisor{
                vec: vec,
                dividend: &self.0
            }
        }
    }

}

