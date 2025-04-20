use crate::util_traits::*;
use crate::vector::vec_util_traits::*;


pub struct VecBufSwap<T: VectorLike> {pub(crate) vec: T}

unsafe impl<T: VectorLike> Get for VecBufSwap<T> {
    type GetBool = T::GetBool;
    type Inputs = T::Inputs;
    type Item = T::Item;
    type BoundItems = T::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {self.vec.get_inputs(index)}}
    #[inline] unsafe fn drop_inputs(&mut self, index: usize) { unsafe {self.vec.drop_inputs(index)}}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(inputs)}
}

unsafe impl<T: VectorLike + IsRepeatable> IsRepeatable for VecBufSwap<T> {}

impl<T: VectorLike> HasOutput for VecBufSwap<T> {
    type OutputBool = T::OutputBool;
    type Output = T::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.vec.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.vec.drop_output()}}
}

impl<T: VectorLike> HasReuseBuf for VecBufSwap<T> {
    type FstHandleBool = T::SndHandleBool;
    type SndHandleBool = T::FstHandleBool;
    type BoundHandlesBool = T::BoundHandlesBool;
    type FstOwnedBufferBool = T::SndOwnedBufferBool;
    type SndOwnedBufferBool = T::FstOwnedBufferBool;
    type FstOwnedBuffer = T::SndOwnedBuffer;
    type SndOwnedBuffer = T::FstOwnedBuffer;
    type FstType = T::SndType;
    type SndType = T::FstType;
    type BoundTypes = T::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) { unsafe {self.vec.assign_2nd_buf(index, val)}}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) { unsafe {self.vec.assign_1st_buf(index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) { unsafe {self.vec.assign_bound_bufs(index, val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {self.vec.get_2nd_buffer()}}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.vec.get_1st_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_2nd_buf_index(index)}}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_1st_buf_index(index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {self.vec.drop_bound_bufs_index(index)}}
}



pub struct DynamicVectorLike<V: VectorLike>{pub(crate) vec: V, pub(crate) inputs: Option<V::Inputs>}

unsafe impl<V: VectorLike> Get for DynamicVectorLike<V> {
    type GetBool = V::GetBool;
    type Inputs = ();
    type Item = V::Item;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {self.inputs = Some(self.vec.get_inputs(index));}}
    #[inline] unsafe fn drop_inputs(&mut self, index: usize) { unsafe {self.vec.drop_inputs(index)}}
    #[inline] fn process(&mut self, _: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(self.inputs.take().unwrap())}
}

unsafe impl<V: IsRepeatable + VectorLike> IsRepeatable for DynamicVectorLike<V> {}

impl<V: VectorLike> HasOutput for DynamicVectorLike<V> {
    type OutputBool = V::OutputBool;
    type Output = V::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.vec.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.vec.drop_output()}}
}

impl<V: VectorLike> HasReuseBuf for DynamicVectorLike<V> {
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
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_1st_buf_index(index)}}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_2nd_buf_index(index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {self.vec.drop_bound_bufs_index(index)}}
}