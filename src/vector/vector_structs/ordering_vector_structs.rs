
use crate::util_traits::*;
use crate::vector::vec_util_traits::*;

/// Offsets (with rolling over) each element up by offset
pub struct VecOffset<T: VectorLike>{pub(crate) vec: T, pub(crate) offset: usize, pub(crate) size: usize}

impl<T: VectorLike> VecOffset<T> {
    #[inline]
    fn offset_index(&self, index: usize) -> usize {
        let mut offset_index = index + self.offset;
        if offset_index >= index { // FIXME: see if there is a better way to detect overflows
            offset_index %= self.size;
        } else { //index overflowed, LLVM should be able to elid this most of the time
            //if the index overflowed, (usize::MAX) + 1 was subtracted from it, add (usize::MAX)+1 mod self.size to recover
            offset_index %= self.size;
            offset_index += ((usize::MAX % self.size) + 1) % self.size; // 2 modulos to prevent overflow
            offset_index %= self.size;
        }
        offset_index
    }
}

unsafe impl<T: VectorLike> Get for VecOffset<T> {
    type GetBool = T::GetBool;
    type Inputs = T::Inputs;
    type Item = T::Item;
    type BoundItems = T::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {self.vec.get_inputs(self.offset_index(index))}}
    #[inline] unsafe fn drop_inputs(&mut self, index: usize) { unsafe {self.vec.drop_inputs(self.offset_index(index))}}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(inputs)}
}

impl<T: VectorLike> HasOutput for VecOffset<T> {
    type OutputBool = T::OutputBool;
    type Output = T::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.vec.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.vec.drop_output()}}
}

impl<T: VectorLike> HasReuseBuf for VecOffset<T> {
    type FstHandleBool = T::FstHandleBool;
    type SndHandleBool = T::SndHandleBool;
    type BoundHandlesBool = T::BoundHandlesBool;
    type FstOwnedBufferBool = T::FstOwnedBufferBool;
    type SndOwnedBufferBool = T::SndOwnedBufferBool;
    type FstOwnedBuffer = T::FstOwnedBuffer;
    type SndOwnedBuffer = T::SndOwnedBuffer;
    type FstType = T::FstType;
    type SndType = T::SndType;
    type BoundTypes = T::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) { unsafe {self.vec.assign_1st_buf(self.offset_index(index), val)}}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) { unsafe {self.vec.assign_2nd_buf(self.offset_index(index), val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) { unsafe {self.vec.assign_bound_bufs(self.offset_index(index), val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {self.vec.get_1st_buffer()}}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.vec.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_1st_buf_index(self.offset_index(index))}}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_2nd_buf_index(self.offset_index(index))}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {self.vec.drop_bound_bufs_index(self.offset_index(index))}}
}

/// Reverses the elements in the vector
pub struct VecReverse<V: VectorLike>{pub(crate) vec: V, pub(crate) max_index: usize}

impl<T: VectorLike> VecReverse<T> {
    #[inline]
    fn reverse_index(&self, index: usize) -> usize {
        self.max_index - index
    }
}

unsafe impl<T: VectorLike> Get for VecReverse<T> {
    type GetBool = T::GetBool;
    type Inputs = T::Inputs;
    type Item = T::Item;
    type BoundItems = T::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {self.vec.get_inputs(self.reverse_index(index))}}
    #[inline] unsafe fn drop_inputs(&mut self, index: usize) { unsafe {self.vec.drop_inputs(self.reverse_index(index))}}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(inputs)}
}

impl<T: VectorLike> HasOutput for VecReverse<T> {
    type OutputBool = T::OutputBool;
    type Output = T::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.vec.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.vec.drop_output()}}
}

impl<T: VectorLike> HasReuseBuf for VecReverse<T> {
    type FstHandleBool = T::FstHandleBool;
    type SndHandleBool = T::SndHandleBool;
    type BoundHandlesBool = T::BoundHandlesBool;
    type FstOwnedBufferBool = T::FstOwnedBufferBool;
    type SndOwnedBufferBool = T::SndOwnedBufferBool;
    type FstOwnedBuffer = T::FstOwnedBuffer;
    type SndOwnedBuffer = T::SndOwnedBuffer;
    type FstType = T::FstType;
    type SndType = T::SndType;
    type BoundTypes = T::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) { unsafe {self.vec.assign_1st_buf(self.reverse_index(index), val)}}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) { unsafe {self.vec.assign_2nd_buf(self.reverse_index(index), val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) { unsafe {self.vec.assign_bound_bufs(self.reverse_index(index), val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {self.vec.get_1st_buffer()}}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.vec.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_1st_buf_index(self.reverse_index(index))}}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_2nd_buf_index(self.reverse_index(index))}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {self.vec.drop_bound_bufs_index(self.reverse_index(index))}}
}
