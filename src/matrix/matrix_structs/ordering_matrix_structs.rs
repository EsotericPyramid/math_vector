use crate::util_traits::*;
use crate::matrix::mat_util_traits::*;

pub struct MatColOffset<M: MatrixLike>{pub(crate) mat: M, pub(crate) offset: usize, pub(crate) num_columns: usize}

impl<M: MatrixLike> MatColOffset<M> {
    #[inline]
    fn offset_index(&self, index: usize) -> usize {
        let mut offset_index = index + self.offset;
        if offset_index >= index { 
            offset_index %= self.num_columns;
        } else { //index overflowed, LLVM should be able to elid this most of the time (hopefully)
            //if the index overflowed, (usize::MAX) + 1 was subtracted from it, add (usize::MAX)+1 mod self.num_columns to recover
            offset_index %= self.num_columns;
            offset_index += ((usize::MAX % self.num_columns) + 1) % self.num_columns; // 2 modulos to prevent overflow
            offset_index %= self.num_columns;
        }
        offset_index
    }
}

unsafe impl<M: MatrixLike> Get2D for MatColOffset<M> {
    type GetBool = M::GetBool;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs { unsafe {self.mat.get_inputs(self.offset_index(col_index), row_index)}}
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_inputs(self.offset_index(col_index), row_index)}}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.mat.process(inputs)}
}

impl<M: MatrixLike> HasOutput for MatColOffset<M> {
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.mat.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.mat.drop_output()}}
}

impl<M: MatrixLike> Has2DReuseBuf for MatColOffset<M> {
    type FstHandleBool = M::FstHandleBool;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = M::FstOwnedBufferBool;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type IsFstBufferTransposed = M::IsFstBufferTransposed;
    type IsSndBufferTransposed = M::IsSndBufferTransposed;
    type AreBoundBuffersTransposed = M::AreBoundBuffersTransposed;
    type FstOwnedBuffer = M::FstOwnedBuffer;
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = M::FstType;
    type SndType = M::SndType;
    type BoundTypes = M::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) { unsafe {self.mat.assign_1st_buf(self.offset_index(col_index), row_index, val)}}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) { unsafe {self.mat.assign_2nd_buf(self.offset_index(col_index), row_index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) { unsafe {self.mat.assign_bound_bufs(self.offset_index(col_index), row_index, val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {self.mat.get_1st_buffer()}}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.mat.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_1st_buf_index(self.offset_index(col_index), row_index)}}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_2nd_buf_index(self.offset_index(col_index), row_index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_bound_bufs_index(self.offset_index(col_index), row_index)}}
}


pub struct MatRowOffset<M: MatrixLike>{pub(crate) mat: M, pub(crate) offset: usize, pub(crate) num_rows: usize}

impl<M: MatrixLike> MatRowOffset<M> {
    #[inline]
    fn offset_index(&self, index: usize) -> usize {
        let mut offset_index = index + self.offset;
        if offset_index >= index { 
            offset_index %= self.num_rows;
        } else { //index overflowed, LLVM should be able to elid this most of the time
            //if the index overflowed, (usize::MAX) + 1 was subtracted from it, add (usize::MAX)+1 mod self.num_rows to recover
            offset_index %= self.num_rows;
            offset_index += ((usize::MAX % self.num_rows) + 1) % self.num_rows; // 2 modulos to prevent overflow
            offset_index %= self.num_rows;
        }
        offset_index
    }
}

unsafe impl<M: MatrixLike> Get2D for MatRowOffset<M> {
    type GetBool = M::GetBool;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs { unsafe {self.mat.get_inputs(col_index, self.offset_index(row_index))}}
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_inputs(col_index, self.offset_index(row_index))}}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.mat.process(inputs)}
}

impl<M: MatrixLike> HasOutput for MatRowOffset<M> {
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.mat.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.mat.drop_output()}}
}

impl<M: MatrixLike> Has2DReuseBuf for MatRowOffset<M> {
    type FstHandleBool = M::FstHandleBool;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = M::FstOwnedBufferBool;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type IsFstBufferTransposed = M::IsFstBufferTransposed;
    type IsSndBufferTransposed = M::IsSndBufferTransposed;
    type AreBoundBuffersTransposed = M::AreBoundBuffersTransposed;
    type FstOwnedBuffer = M::FstOwnedBuffer;
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = M::FstType;
    type SndType = M::SndType;
    type BoundTypes = M::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) { unsafe {self.mat.assign_1st_buf(col_index, self.offset_index(row_index), val)}}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) { unsafe {self.mat.assign_2nd_buf(col_index, self.offset_index(row_index), val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) { unsafe {self.mat.assign_bound_bufs(col_index, self.offset_index(row_index), val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {self.mat.get_1st_buffer()}}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.mat.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_1st_buf_index(col_index, self.offset_index(row_index))}}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_2nd_buf_index(col_index, self.offset_index(row_index))}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_bound_bufs_index(col_index, self.offset_index(row_index))}}
}