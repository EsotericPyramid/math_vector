use crate::trait_specialization_utils::*;
use crate::util_traits::*;
use crate::matrix::mat_util_traits::*;

/// struct generating items using a FnMut closure with no inputs
pub struct MatGenerator<F: FnMut() -> O, O>(pub(crate) F);

unsafe impl<F: FnMut() -> O, O> Get2D for MatGenerator<F, O> {
    type GetBool = Y;
    type AreInputsTransposed = N;
    type Inputs = ();
    type Item = O;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, _: usize, _: usize) -> Self::Inputs {}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize, _: usize) {}
    #[inline] fn process(&mut self, _: usize, _: usize, _: Self::Inputs) -> (Self::Item, Self::BoundItems) {((self.0)(), ())}
}

impl<F: FnMut() -> O, O> HasOutput for MatGenerator<F, O> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

impl<F: FnMut() -> O, O> Has2DReuseBuf for MatGenerator<F, O> {
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


/// struct generating items using a FnMut closure given the column and row indices
pub struct MatIndexGenerator<F: FnMut(usize, usize) -> O, O>(pub(crate) F);

unsafe impl<F: FnMut(usize, usize) -> O, O> Get2D for MatIndexGenerator<F, O> {
    type GetBool = Y;
    type AreInputsTransposed = N;
    type Inputs = ();
    type Item = O;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, _: usize, _: usize) -> Self::Inputs {}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize, _: usize) {}
    #[inline] fn process(&mut self, col_index: usize, row_index: usize, _: Self::Inputs) -> (Self::Item, Self::BoundItems) {((self.0)(col_index, row_index), ())}
}

impl<F: FnMut(usize, usize) -> O, O> HasOutput for MatIndexGenerator<F, O> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

impl<F: FnMut(usize, usize) -> O, O> Has2DReuseBuf for MatIndexGenerator<F, O> {
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


pub struct MatIdentityGenerator<T: Copy>{pub(crate) zero: T, pub(crate) one: T}

unsafe impl<T: Copy> Get2D for MatIdentityGenerator<T> {
    type GetBool = Y;
    type AreInputsTransposed = N;
    type Inputs = ();
    type Item = T;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, _: usize, _: usize) -> Self::Inputs {}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize, _: usize) {}
    #[inline] fn process(&mut self, col_index: usize, row_index: usize, _: Self::Inputs) -> (Self::Item, Self::BoundItems) {(if col_index == row_index {self.one} else {self.zero}, ())}
}

impl<T: Copy> HasOutput for MatIdentityGenerator<T> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

impl<T: Copy> Has2DReuseBuf for MatIdentityGenerator<T> {
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
