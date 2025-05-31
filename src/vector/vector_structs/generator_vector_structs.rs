
use crate::trait_specialization_utils::*;
use crate::util_traits::*;
use crate::vector::vec_util_traits::*;

/// Struct generating a vector's items based on a closure (FnMut) with no inputs
pub struct VecGenerator<F: FnMut() -> O, O>(pub(crate) F);

unsafe impl<F: FnMut() -> O, O> Get for VecGenerator<F, O> {
    type GetBool = Y;
    type Inputs = ();
    type Item = O;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, _: usize) -> Self::Inputs {}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}
    #[inline] fn process(&mut self, _: usize, _: Self::Inputs) -> (Self::Item, Self::BoundItems) {((self.0)(), ())}
}

impl<F: FnMut() -> O, O> HasOutput for VecGenerator<F, O> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

impl<F: FnMut() -> O, O> HasReuseBuf for VecGenerator<F, O> {
    type FstHandleBool = N;
    type SndHandleBool = N;
    type BoundHandlesBool = N;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = N;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = ();
    type FstType = ();
    type SndType = ();
    type BoundTypes = ();

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, _: usize, _: Self::SndType) {}
    #[inline] unsafe fn assign_bound_bufs(&mut self, _: usize, _: Self::BoundTypes) {}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, _: usize) {}
}

/// Struct generating a vector's items based on a closure (FnMut) given the index
pub struct VecIndexGenerator<F: FnMut(usize) -> O, O>(pub(crate) F);

unsafe impl<F: FnMut(usize) -> O, O> Get for VecIndexGenerator<F, O> {
    type GetBool = Y;
    type Inputs = ();
    type Item = O;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, _: usize) -> Self::Inputs {}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}
    #[inline] fn process(&mut self, index: usize, _: Self::Inputs) -> (Self::Item, Self::BoundItems) {((self.0)(index), ())}
}

impl<F: FnMut(usize) -> O, O> HasOutput for VecIndexGenerator<F, O> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

impl<F: FnMut(usize) -> O, O> HasReuseBuf for VecIndexGenerator<F, O> {
    type FstHandleBool = N;
    type SndHandleBool = N;
    type BoundHandlesBool = N;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = N;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = ();
    type FstType = ();
    type SndType = ();
    type BoundTypes = ();

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, _: usize, _: Self::SndType) {}
    #[inline] unsafe fn assign_bound_bufs(&mut self, _: usize, _: Self::BoundTypes) {}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, _: usize) {}
}