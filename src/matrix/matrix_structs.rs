use crate::trait_specialization_utils::*;
use crate::util_traits::*;
use super::mat_util_traits::*;
use std::ops::*;
use std::mem::ManuallyDrop;

mod array_matrix_structs; 
mod binding_matrix_structs;
mod generator_matrix_structs;
mod macroed_matrix_structs;
mod math_matrix_structs;
mod misc_matrix_structs;
mod ordering_matrix_structs;
mod vectorizing_matrix_structs;

pub use array_matrix_structs::*;
pub use binding_matrix_structs::*;
pub use generator_matrix_structs::*;
pub use macroed_matrix_structs::*;
pub use math_matrix_structs::*;
pub use misc_matrix_structs::*;
pub use ordering_matrix_structs::*;
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


#[inline] fn debox<T: ?Sized>(boxed: &mut Box<T>) -> &mut T {&mut *boxed}

unsafe impl<M: MatrixLike + ?Sized> Get2D for Box<M> {
    type GetBool = M::GetBool;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs { unsafe {(debox(self)).get_inputs(col_index, row_index)}}
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) { unsafe {(debox(self)).drop_inputs(col_index, row_index)}}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(debox(self)).process(inputs)}
}

impl<M: MatrixLike + ?Sized> Has2DReuseBuf for Box<M> {
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
