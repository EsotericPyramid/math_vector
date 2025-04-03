use crate::trait_specialization_utils::*;
use crate::util_traits::*;
use super::mat_util_traits::*;
use std::ops::*;
use std::mem::ManuallyDrop;

mod array_matrix_structs; 
mod binding_matrix_structs;
mod generator_matrix_structs;
mod macroed_matrix_structs;
mod vectorizing_matrix_structs;

pub use array_matrix_structs::*;
pub use binding_matrix_structs::*;
pub use generator_matrix_structs::*;
pub use macroed_matrix_structs::*;
pub use vectorizing_matrix_structs::*;


#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Owned2DArray<T, const D1: usize, const D2: usize>(pub(crate) ManuallyDrop<[[T; D1]; D2]>);

impl<T, const D1: usize, const D2: usize> Owned2DArray<T, D1, D2> {
    #[inline]
    pub fn unwrap(self) -> [[T; D1]; D2] {
        ManuallyDrop::into_inner(self.0)
    }
}

impl<T, const D1: usize, const D2: usize> Deref for Owned2DArray<T, D1, D2> {
    type Target = ManuallyDrop<[[T; D1]; D2]>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const D1: usize, const D2: usize> DerefMut for Owned2DArray<T, D1, D2> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

unsafe impl<T, const D1: usize, const D2: usize> Get2D for Owned2DArray<T, D1, D2> {
    type GetBool = Y;
    type AreInputsTransposed = N;
    type Inputs = T;
    type Item = T;
    type BoundItems = ();

    #[inline] 
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs { unsafe {
        std::ptr::read(self.0.get_unchecked(col_index).get_unchecked(row_index))
    }}
    #[inline]
    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
        (inputs, ())
    }

    #[inline]
    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) { unsafe {
        std::ptr::drop_in_place(self.0.get_unchecked_mut(col_index).get_unchecked_mut(row_index))
    }}
}

unsafe impl<T: Copy, const D1: usize, const D2: usize> IsRepeatable for Owned2DArray<T, D1, D2> {}

impl<T, const D1: usize, const D2: usize> HasOutput for Owned2DArray<T, D1, D2> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

impl<T, const D1: usize, const D2: usize> Has2DReuseBuf for Owned2DArray<T, D1, D2> {
    type FstHandleBool = N;
    type SndHandleBool = N;
    type BoundHandlesBool = N;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = N;
    type IsFstBufferTransposed = N;
    type IsSndBufferTransposed = N;
    type AreBoundBuffersTransposed = N;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = ();
    type FstType = ();
    type SndType = ();
    type BoundTypes = ();

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, _: usize, _: usize, _: Self::SndType) {}
    #[inline] unsafe fn assign_bound_bufs(&mut self, _: usize, _: usize, _: Self::BoundTypes) {}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, _: usize, _: usize) {}
}

pub struct Referring2DArray<'a, T: 'a, const D1: usize, const D2: usize>(pub(crate) [[T; D1]; D2], pub(crate) std::marker::PhantomData<&'a T>);

unsafe impl<'a, T: 'a, const D1: usize, const D2: usize> Get2D for Referring2DArray<'a, T, D1, D2> {
    type GetBool = Y;
    type AreInputsTransposed = N;
    type Inputs = &'a T;
    type Item = &'a T;
    type BoundItems = ();

    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {unsafe {&*(self.0.get_unchecked(col_index).get_unchecked(row_index) as *const T)}}
    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs, ())}
    unsafe fn drop_inputs(&mut self, _: usize, _: usize) {}
}

unsafe impl<'a, T: 'a, const D1: usize, const D2: usize> IsRepeatable for Referring2DArray<'a, T, D1, D2> {}

impl<'a, T: 'a, const D1: usize, const D2: usize> HasOutput for Referring2DArray<'a, T, D1, D2> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

impl<'a, T: 'a, const D1: usize, const D2: usize> Has2DReuseBuf for Referring2DArray<'a, T, D1, D2> {
    type FstHandleBool = N;
    type SndHandleBool = N;
    type BoundHandlesBool = N;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = N;
    type IsFstBufferTransposed = N;
    type IsSndBufferTransposed = N;
    type AreBoundBuffersTransposed = N;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = ();
    type FstType = ();
    type SndType = ();
    type BoundTypes = ();

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, _: usize, _: usize, _: Self::SndType) {}
    #[inline] unsafe fn assign_bound_bufs(&mut self, _: usize, _: usize, _: Self::BoundTypes) {}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, _: usize, _: usize) {}
}


//Note: these 2 technically impl HasOutput via vector_structs' impls on &[T; D], fine since none actually output anything
//Note: impls IsRepeatable through vector_structs' impl, still correct though
unsafe impl<'a, T, const D1: usize, const D2: usize> Get2D for &'a [[T; D1]; D2] {
    type GetBool = Y;
    type AreInputsTransposed = N;
    type Inputs = &'a T;
    type Item = &'a T;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs { unsafe {self.get_unchecked(col_index).get_unchecked(row_index)}}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs, ())}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize, _: usize) {}
}

impl<'a, T, const D1: usize, const D2: usize> Has2DReuseBuf for &'a [[T; D1]; D2] {
    type FstHandleBool = N;
    type SndHandleBool = N;
    type BoundHandlesBool = N;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = N;
    type IsFstBufferTransposed = N;
    type IsSndBufferTransposed = N;
    type AreBoundBuffersTransposed = N;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = ();
    type FstType = ();
    type SndType = ();
    type BoundTypes = ();

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, _: usize, _: usize, _: Self::SndType) {}
    #[inline] unsafe fn assign_bound_bufs(&mut self, _: usize, _: usize, _: Self::BoundTypes) {}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, _: usize, _: usize) {}
}


unsafe impl<'a, T, const D1: usize, const D2: usize> Get2D for &'a mut [[T; D1]; D2] {
    type GetBool = Y;
    type AreInputsTransposed = N;
    type Inputs = &'a mut T;
    type Item = &'a mut T;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs { unsafe {&mut*(self.get_unchecked_mut(col_index).get_unchecked_mut(row_index) as *mut T)}}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs, ())}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize, _: usize) {}
}

impl<'a, T, const D1: usize, const D2: usize> Has2DReuseBuf for &'a mut [[T; D1]; D2] {
    type FstHandleBool = N;
    type SndHandleBool = N;
    type BoundHandlesBool = N;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = N;
    type IsFstBufferTransposed = N;
    type IsSndBufferTransposed = N;
    type AreBoundBuffersTransposed = N;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = ();
    type FstType = ();
    type SndType = ();
    type BoundTypes = ();

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, _: usize, _: usize, _: Self::SndType) {}
    #[inline] unsafe fn assign_bound_bufs(&mut self, _: usize, _: usize, _: Self::BoundTypes) {}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, _: usize, _: usize) {}
}


#[inline] fn debox<T: Sized>(boxed: &mut Box<T>) -> &mut T {&mut *boxed}

unsafe impl<M: MatrixLike> Get2D for Box<M> {
    type GetBool = M::GetBool;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs { unsafe {(debox(self)).get_inputs(col_index, row_index)}}
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) { unsafe {(debox(self)).drop_inputs(col_index, row_index)}}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(debox(self)).process(inputs)}
}

impl<M: MatrixLike> Has2DReuseBuf for Box<M> {
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

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) { unsafe {(debox(self)).assign_1st_buf(col_index, row_index, val)}}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) { unsafe {(debox(self)).assign_2nd_buf(col_index, row_index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) { unsafe {(debox(self)).assign_bound_bufs(col_index, row_index, val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {(debox(self)).get_1st_buffer()}}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {(debox(self)).get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {(debox(self)).drop_1st_buf_index(col_index, row_index)}}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {(debox(self)).drop_2nd_buf_index(col_index, row_index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) { unsafe {(debox(self)).drop_bound_bufs_index(col_index, row_index)}}
}





pub struct MatBufSwap<M: MatrixLike>{pub(crate) mat: M} 

unsafe impl<M: MatrixLike> Get2D for MatBufSwap<M> {
    type GetBool = M::GetBool;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs { unsafe {self.mat.get_inputs(col_index, row_index)}}
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_inputs(col_index, row_index)}}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.mat.process(inputs)}
}

unsafe impl<M: IsRepeatable + MatrixLike> IsRepeatable for MatBufSwap<M> {}

impl<M: MatrixLike> HasOutput for MatBufSwap<M> {
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.mat.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.mat.drop_output()}}
}

impl<M: MatrixLike> Has2DReuseBuf for MatBufSwap<M> {
    type FstHandleBool = M::SndHandleBool;
    type SndHandleBool = M::FstHandleBool;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = M::SndOwnedBufferBool;
    type SndOwnedBufferBool = M::FstOwnedBufferBool;
    type IsFstBufferTransposed = M::IsSndBufferTransposed;
    type IsSndBufferTransposed = M::IsFstBufferTransposed;
    type AreBoundBuffersTransposed = M::AreBoundBuffersTransposed;
    type FstOwnedBuffer = M::SndOwnedBuffer;
    type SndOwnedBuffer = M::FstOwnedBuffer;
    type FstType = M::SndType;
    type SndType = M::FstType;
    type BoundTypes = M::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) { unsafe {self.mat.assign_2nd_buf(col_index, row_index, val)}}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) { unsafe {self.mat.assign_1st_buf(col_index, row_index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) { unsafe {self.mat.assign_bound_bufs(col_index, row_index, val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {self.mat.get_2nd_buffer()}}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.mat.get_1st_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_2nd_buf_index(col_index, row_index)}}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_1st_buf_index(col_index, row_index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_bound_bufs_index(col_index, row_index)}}
}


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


/// SAFETY: it is expected that the used_mat field is safe to output in addition to normal correct implementation
pub struct MatAttachUsedMat<M: MatrixLike, USEDM: MatrixLike>{pub(crate) mat: M, pub(crate) used_mat: USEDM}

unsafe impl<M: MatrixLike, USEDM: MatrixLike> Get2D for MatAttachUsedMat<M, USEDM> {
    type GetBool = M::GetBool;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs { unsafe {self.mat.get_inputs(col_index, row_index)}}
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_inputs(col_index, row_index)}}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.mat.process(inputs)}
}

unsafe impl<M: IsRepeatable + MatrixLike, USEDM: MatrixLike> IsRepeatable for MatAttachUsedMat<M, USEDM> {}

impl<M: MatrixLike, USEDM: MatrixLike> HasOutput for MatAttachUsedMat<M, USEDM> where (M::OutputBool, USEDM::OutputBool): FilterPair {
    type OutputBool = <(M::OutputBool, USEDM::OutputBool) as TyBoolPair>::Or;
    type Output = <(M::OutputBool, USEDM::OutputBool) as FilterPair>::Filtered<M::Output, USEDM::Output>;

    #[inline] 
    unsafe fn output(&mut self) -> Self::Output { unsafe {
        <(M::OutputBool, USEDM::OutputBool) as FilterPair>::filter(
            self.mat.output(),
            self.used_mat.output()
        )
    }}

    #[inline]
    unsafe fn drop_output(&mut self) { unsafe {
        self.mat.drop_output();
        self.used_mat.drop_output();
    }}
}

impl<M: MatrixLike, USEDM: MatrixLike> Has2DReuseBuf for MatAttachUsedMat<M, USEDM> 
where 
    (M::FstOwnedBufferBool, USEDM::FstOwnedBufferBool): SelectPair,
    (M::SndOwnedBufferBool, USEDM::SndOwnedBufferBool): SelectPair,
    (M::FstHandleBool, USEDM::FstHandleBool): SelectPair,
    (M::SndHandleBool, USEDM::SndHandleBool): SelectPair,
    (M::BoundHandlesBool, USEDM::BoundHandlesBool): FilterPair,
    (M::IsFstBufferTransposed, USEDM::IsFstBufferTransposed): TyBoolPair,
    (M::IsSndBufferTransposed, USEDM::IsSndBufferTransposed): TyBoolPair,    
    (M::AreBoundBuffersTransposed, USEDM::AreBoundBuffersTransposed): TyBoolPair
{
    type FstHandleBool = <(M::FstHandleBool, USEDM::FstHandleBool) as TyBoolPair>::Xor;
    type SndHandleBool = <(M::SndHandleBool, USEDM::SndHandleBool) as TyBoolPair>::Xor;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = <(M::FstOwnedBufferBool, USEDM::FstOwnedBufferBool) as TyBoolPair>::Xor; 
    type SndOwnedBufferBool = <(M::SndOwnedBufferBool, USEDM::SndOwnedBufferBool) as TyBoolPair>::Xor; 
    type IsFstBufferTransposed = <(M::IsFstBufferTransposed, USEDM::IsFstBufferTransposed) as TyBoolPair>::Or;
    type IsSndBufferTransposed = <(M::IsSndBufferTransposed, USEDM::IsSndBufferTransposed) as TyBoolPair>::Or;
    type AreBoundBuffersTransposed = <(M::AreBoundBuffersTransposed, USEDM::AreBoundBuffersTransposed) as TyBoolPair>::And;
    type FstOwnedBuffer = <(M::FstOwnedBufferBool, USEDM::FstOwnedBufferBool) as SelectPair>::Selected<M::FstOwnedBuffer, USEDM::FstOwnedBuffer>;
    type SndOwnedBuffer = <(M::SndOwnedBufferBool, USEDM::SndOwnedBufferBool) as SelectPair>::Selected<M::SndOwnedBuffer, USEDM::SndOwnedBuffer>;
    type FstType = <(M::FstHandleBool, USEDM::FstHandleBool) as SelectPair>::Selected<M::FstType, USEDM::FstType>;
    type SndType = <(M::SndHandleBool, USEDM::SndHandleBool) as SelectPair>::Selected<M::SndType, USEDM::SndType>;
    type BoundTypes = M::BoundTypes;

    #[inline] unsafe fn assign_1st_buf<'z>(&'z mut self, col_index: usize, row_index: usize, val: Self::FstType) { unsafe {
        let (l_val, r_val) = <(M::FstHandleBool, USEDM::FstHandleBool) as SelectPair>::deselect(val);
        self.mat.assign_1st_buf(col_index, row_index, l_val);
        self.used_mat.assign_1st_buf(col_index, row_index, r_val);
    }}
    #[inline] unsafe fn assign_2nd_buf<'z>(&'z mut self, col_index: usize, row_index: usize, val: Self::SndType) { unsafe {
        let (l_val, r_val) = <(M::SndHandleBool, USEDM::SndHandleBool) as SelectPair>::deselect(val);
        self.mat.assign_2nd_buf(col_index, row_index, l_val);
        self.used_mat.assign_2nd_buf(col_index, row_index, r_val);
    }}
    #[inline] unsafe fn assign_bound_bufs<'z>(&'z mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) { unsafe {
        self.mat.assign_bound_bufs(col_index, row_index, val);
    }}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {
        <(M::FstOwnedBufferBool, USEDM::FstOwnedBufferBool) as SelectPair>::select(self.mat.get_1st_buffer(), self.used_mat.get_1st_buffer())
    }}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {
        <(M::SndOwnedBufferBool, USEDM::SndOwnedBufferBool) as SelectPair>::select(self.mat.get_2nd_buffer(), self.used_mat.get_2nd_buffer())
    }}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {
        self.mat.drop_1st_buf_index(col_index, row_index);
        self.used_mat.drop_1st_buf_index(col_index, row_index);
    }}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {
        self.mat.drop_2nd_buf_index(col_index, row_index);
        self.used_mat.drop_2nd_buf_index(col_index, row_index);
    }}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) { unsafe {
        self.mat.drop_bound_bufs_index(col_index, row_index);
        self.used_mat.drop_bound_bufs_index(col_index, row_index);
    }}
}




pub struct FullMatMul<M1: MatrixLike + IsRepeatable, M2: MatrixLike + IsRepeatable>{pub(crate) l_mat: M1, pub(crate) r_mat: M2, pub(crate) shared_size: usize}

unsafe impl<M1: MatrixLike + IsRepeatable, M2: MatrixLike + IsRepeatable> Get2D for FullMatMul<M1, M2> where 
    M1::Item: Mul<M2::Item>,
    <M1::Item as Mul<M2::Item>>::Output: AddAssign,
    (M1::BoundHandlesBool, M2::BoundHandlesBool): FilterPair,
    (M1::AreInputsTransposed, M2::AreInputsTransposed): TyBoolPair,
{
    type GetBool = Y;
    type AreInputsTransposed = <(M1::AreInputsTransposed, M2::AreInputsTransposed) as TyBoolPair>::Or;
    type Inputs = (usize, usize);
    type Item = <M1::Item as Mul<M2::Item>>::Output;
    type BoundItems = <(M1::BoundHandlesBool, M2::BoundHandlesBool) as FilterPair>::Filtered<M1::BoundItems, M2::BoundItems>;

    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {(col_index, row_index)}
    unsafe fn drop_inputs(&mut self, _: usize, _: usize) {}
    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
        unsafe {
            let mut result = self.l_mat.get(0, inputs.1).0 * self.r_mat.get(inputs.0, 0).0;
            for i in 1..self.shared_size {
                result += self.l_mat.get(i, inputs.1).0 * self.r_mat.get(inputs.0, i).0;
            }
            let bound = <(M1::BoundHandlesBool, M2::BoundHandlesBool) as FilterPair>::filter(
                self.l_mat.get(inputs.0, inputs.1).1, 
                self.r_mat.get(inputs.0, inputs.1).1
            );
            (result, bound)
        }
    }
}

impl<M1: MatrixLike + IsRepeatable, M2: MatrixLike + IsRepeatable> HasOutput for FullMatMul<M1, M2> where (M1::OutputBool, M2::OutputBool): FilterPair {
    type OutputBool = <(M1::OutputBool, M2::OutputBool) as TyBoolPair>::Or;
    type Output = <(M1::OutputBool, M2::OutputBool) as FilterPair>::Filtered<M1::Output, M2::Output>;

    unsafe fn output(&mut self) -> Self::Output { unsafe {
        <(M1::OutputBool, M2::OutputBool) as FilterPair>::filter(
            self.l_mat.output(),
            self.r_mat.output()
        )
    }}
    unsafe fn drop_output(&mut self) { unsafe {
        self.l_mat.drop_output();
        self.r_mat.drop_output();
    }}
} 

impl<M1: MatrixLike + IsRepeatable, M2: MatrixLike + IsRepeatable> Has2DReuseBuf for FullMatMul<M1, M2>
where 
    (M1::FstOwnedBufferBool, M2::FstOwnedBufferBool): SelectPair,
    (M1::SndOwnedBufferBool, M2::SndOwnedBufferBool): SelectPair,
    (M1::FstHandleBool, M2::FstHandleBool): SelectPair,
    (M1::SndHandleBool, M2::SndHandleBool): SelectPair,
    (M1::BoundHandlesBool, M2::BoundHandlesBool): FilterPair,
    (M1::IsFstBufferTransposed, M2::IsFstBufferTransposed): TyBoolPair,
    (M1::IsSndBufferTransposed, M2::IsSndBufferTransposed): TyBoolPair,
    (M1::AreBoundBuffersTransposed, M2::AreBoundBuffersTransposed): TyBoolPair
{
    type FstHandleBool = <(M1::FstHandleBool, M2::FstHandleBool) as TyBoolPair>::Xor;
    type SndHandleBool = <(M1::SndHandleBool, M2::SndHandleBool) as TyBoolPair>::Xor;
    type BoundHandlesBool = <(M1::BoundHandlesBool, M2::BoundHandlesBool) as TyBoolPair>::Or;
    type FstOwnedBufferBool = <(M1::FstOwnedBufferBool, M2::FstOwnedBufferBool) as TyBoolPair>::Xor; 
    type SndOwnedBufferBool = <(M1::SndOwnedBufferBool, M2::SndOwnedBufferBool) as TyBoolPair>::Xor; 
    type IsFstBufferTransposed = <(M1::IsFstBufferTransposed, M2::IsFstBufferTransposed) as TyBoolPair>::Xor;
    type IsSndBufferTransposed = <(M1::IsSndBufferTransposed, M2::IsSndBufferTransposed) as TyBoolPair>::Xor;
    type AreBoundBuffersTransposed = <(M1::AreBoundBuffersTransposed, M2::AreBoundBuffersTransposed) as TyBoolPair>::Xor;
    type FstOwnedBuffer = <(M1::FstOwnedBufferBool, M2::FstOwnedBufferBool) as SelectPair>::Selected<M1::FstOwnedBuffer, M2::FstOwnedBuffer>;
    type SndOwnedBuffer = <(M1::SndOwnedBufferBool, M2::SndOwnedBufferBool) as SelectPair>::Selected<M1::SndOwnedBuffer, M2::SndOwnedBuffer>;
    type FstType = <(M1::FstHandleBool, M2::FstHandleBool) as SelectPair>::Selected<M1::FstType, M2::FstType>;
    type SndType = <(M1::SndHandleBool, M2::SndHandleBool) as SelectPair>::Selected<M1::SndType, M2::SndType>;
    type BoundTypes = <(M1::BoundHandlesBool, M2::BoundHandlesBool) as FilterPair>::Filtered<M1::BoundTypes, M2::BoundTypes>;

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) { unsafe {
        let (l_val, r_val) = <(M1::FstHandleBool, M2::FstHandleBool) as SelectPair>::deselect(val);
        self.l_mat.assign_1st_buf(col_index, row_index, l_val);
        self.r_mat.assign_1st_buf(col_index, row_index, r_val);
    }}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) { unsafe {
        let (l_val, r_val) = <(M1::SndHandleBool, M2::SndHandleBool) as SelectPair>::deselect(val);
        self.l_mat.assign_2nd_buf(col_index, row_index, l_val);
        self.r_mat.assign_2nd_buf(col_index, row_index, r_val);
    }}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) { unsafe {
        let (l_val, r_val) = <(M1::BoundHandlesBool, M2::BoundHandlesBool) as FilterPair>::defilter(val);
        self.l_mat.assign_bound_bufs(col_index, row_index, l_val);
        self.r_mat.assign_bound_bufs(col_index, row_index, r_val);
    }}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {
        <(M1::FstOwnedBufferBool, M2::FstOwnedBufferBool) as SelectPair>::select(self.l_mat.get_1st_buffer(), self.r_mat.get_1st_buffer())
    }}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {
        <(M1::SndOwnedBufferBool, M2::SndOwnedBufferBool) as SelectPair>::select(self.l_mat.get_2nd_buffer(), self.r_mat.get_2nd_buffer())
    }}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {
        self.l_mat.drop_1st_buf_index(col_index, row_index);
        self.r_mat.drop_1st_buf_index(col_index, row_index);
    }}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {
        self.l_mat.drop_2nd_buf_index(col_index, row_index);
        self.r_mat.drop_2nd_buf_index(col_index, row_index);
    }}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) { unsafe {
        self.l_mat.drop_bound_bufs_index(col_index, row_index);
        self.r_mat.drop_bound_bufs_index(col_index, row_index);
    }}
}
