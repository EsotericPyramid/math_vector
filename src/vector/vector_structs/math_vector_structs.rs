// NOTE: Most the math vector structs are actually defined in macroed_vector_structs.rs
//       only the complicated ones end up here

use std::{marker::PhantomData, ops::AddAssign};

use crate::{
    matrix::{
        mat_util_traits::MatrixLike, 
        matrix_structs::{MatRowVectorExprs, MatColVectorExprs, MatrixRow, MatrixColumn}
    }, trait_specialization_utils::*, util_structs::*, util_traits::HasOutput, vector::{
        vec_util_traits::*, VectorIter
    }
};


pub struct MatVecMul<M: MatrixLike, V: VectorLike + IsRepeatable, B: VectorBuilder, O>{pub(crate) mat: MatRowVectorExprs<M>, pub(crate) vec: V, pub(crate) inner_builder: B, pub(crate) phantom: PhantomData<O>}

unsafe impl<M: MatrixLike, V: VectorLike + IsRepeatable, B: VectorBuilder, O: AddAssign<<M::Item as std::ops::Mul<<V as Get>::Item>>::Output> + std::iter::Sum> Get for MatVecMul<M, V, B, O> where 
    M::Item: std::ops::Mul<<V as Get>::Item>,

    MatrixRow<M>: HasReuseBuf<BoundTypes = <MatrixRow<M> as Get>::BoundItems>,
{
    type GetBool = Y;
    type Inputs = (); //MatRowVectorExprs has none & V is repeatable
    type Item = O;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, _: usize) -> Self::Inputs {}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}

    #[inline]
    fn process(&mut self, index: usize, _: Self::Inputs) -> (Self::Item, Self::BoundItems) 
    {
        unsafe {
            let bound_item = self.vec.get(index).1;
            let mut sum = NoneIter::<O>::new().sum::<O>();
            let row = self.mat.get(index).0;
            for (index, row_elem) in VectorIter::new_from_parts(row, self.inner_builder.clone()).enumerate() {
                let col_elem =self.vec.get(index).0;
                sum += row_elem * col_elem;
            }
            (sum, bound_item) 
        }
    }
}

impl<M: MatrixLike, V: VectorLike + IsRepeatable, B: VectorBuilder, O> HasOutput for MatVecMul<M, V, B, O> where 
    (<MatRowVectorExprs<M> as HasOutput>::OutputBool, V::OutputBool): FilterPair,
{
    type OutputBool = <(<MatRowVectorExprs<M> as HasOutput>::OutputBool, V::OutputBool) as TyBoolPair>::Or;
    type Output = <(<MatRowVectorExprs<M> as HasOutput>::OutputBool, V::OutputBool) as FilterPair>::Filtered<<MatRowVectorExprs<M> as HasOutput>::Output, V::Output>;

    #[inline]
    unsafe fn output(&mut self) -> Self::Output {
        unsafe {
            <(<MatRowVectorExprs<M> as HasOutput>::OutputBool, V::OutputBool) as FilterPair>::filter(
                self.mat.output(), 
                self.vec.output()
            )
        }
    }

    #[inline]
    unsafe fn drop_output(&mut self) {
        unsafe{
            self.mat.drop_output();
            self.vec.drop_output();
        }
    }
}

impl<M: MatrixLike, V: VectorLike + IsRepeatable, B: VectorBuilder, O> HasReuseBuf for MatVecMul<M, V, B, O> {
    type FstHandleBool = V::FstHandleBool;
    type SndHandleBool = V::SndHandleBool;
    type BoundHandlesBool = V::BoundHandlesBool;
    type FstOwnedBufferBool = V::FstOwnedBufferBool;
    type SndOwnedBufferBool = V::SndOwnedBufferBool;
    type FstOwnedBuffer = V::FstOwnedBuffer;
    type SndOwnedBuffer = V::SndOwnedBuffer;
    type FstType = V::FstType;
    type SndType = V::SndType;
    type BoundTypes = V::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) { unsafe {self.vec.assign_1st_buf(index, val)}}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) { unsafe {self.vec.assign_2nd_buf(index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) { unsafe {self.vec.assign_bound_bufs(index, val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {self.vec.get_1st_buffer()}}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.vec.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buffer(&mut self) { unsafe {self.vec.drop_1st_buffer()}}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) { unsafe {self.vec.drop_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_1st_buf_index(index)}}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_2nd_buf_index(index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {self.vec.drop_bound_bufs_index(index)}}
}

pub struct VecMatMul<V: VectorLike + IsRepeatable, M: MatrixLike, B: VectorBuilder, O>{pub(crate) vec: V, pub(crate) mat: MatColVectorExprs<M>, pub(crate) inner_builder: B, pub(crate) phantom: PhantomData<O>}

unsafe impl<V: VectorLike + IsRepeatable, M: MatrixLike, B: VectorBuilder, O: AddAssign<<V::Item as std::ops::Mul<M::Item>>::Output> + std::iter::Sum> Get for VecMatMul<V, M, B, O> where 
    V::Item: std::ops::Mul<M::Item>,

    MatrixColumn<M>: HasReuseBuf<BoundTypes = <MatrixColumn<M> as Get>::BoundItems>,
{
    type GetBool = Y;
    type Inputs = (); //MatColVectorExprs has none & V is repeatable
    type Item = O;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, _: usize) -> Self::Inputs {}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}

    #[inline]
    fn process(&mut self, index: usize, _: Self::Inputs) -> (Self::Item, Self::BoundItems) 
    {
        unsafe {
            let bound_item = self.vec.get(index).1;
            let mut sum = NoneIter::<O>::new().sum::<O>();
            let col = self.mat.get(index).0;
            for (index, col_elem) in VectorIter::new_from_parts(col, self.inner_builder.clone()).enumerate() {
                let row_elem = self.vec.get(index).0;
                sum += row_elem * col_elem;
            }
            (sum, bound_item) 
        }
    }
}

impl<V: VectorLike + IsRepeatable, M: MatrixLike, B: VectorBuilder, O> HasOutput for VecMatMul<V, M, B, O> where 
    (V::OutputBool, <MatColVectorExprs<M> as HasOutput>::OutputBool): FilterPair,
{
    type OutputBool = <(V::OutputBool, <MatColVectorExprs<M> as HasOutput>::OutputBool) as TyBoolPair>::Or;
    type Output = <(V::OutputBool, <MatColVectorExprs<M> as HasOutput>::OutputBool) as FilterPair>::Filtered<V::Output, <MatColVectorExprs<M> as HasOutput>::Output>;

    #[inline]
    unsafe fn output(&mut self) -> Self::Output {
        unsafe {
            <(V::OutputBool, <MatColVectorExprs<M> as HasOutput>::OutputBool) as FilterPair>::filter(
                self.vec.output(),
                self.mat.output() 
            )
        }
    }

    #[inline]
    unsafe fn drop_output(&mut self) {
        unsafe{
            self.vec.drop_output();
            self.mat.drop_output();
        }
    }
}

impl<V: VectorLike + IsRepeatable, M: MatrixLike, B: VectorBuilder, O> HasReuseBuf for VecMatMul<V, M, B, O> {
    type FstHandleBool = V::FstHandleBool;
    type SndHandleBool = V::SndHandleBool;
    type BoundHandlesBool = V::BoundHandlesBool;
    type FstOwnedBufferBool = V::FstOwnedBufferBool;
    type SndOwnedBufferBool = V::SndOwnedBufferBool;
    type FstOwnedBuffer = V::FstOwnedBuffer;
    type SndOwnedBuffer = V::SndOwnedBuffer;
    type FstType = V::FstType;
    type SndType = V::SndType;
    type BoundTypes = V::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) { unsafe {self.vec.assign_1st_buf(index, val)}}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) { unsafe {self.vec.assign_2nd_buf(index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) { unsafe {self.vec.assign_bound_bufs(index, val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {self.vec.get_1st_buffer()}}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.vec.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buffer(&mut self) { unsafe {self.vec.drop_1st_buffer()}}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) { unsafe {self.vec.drop_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_1st_buf_index(index)}}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_2nd_buf_index(index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {self.vec.drop_bound_bufs_index(index)}}
}
