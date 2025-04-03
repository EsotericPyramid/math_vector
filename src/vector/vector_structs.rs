use crate::trait_specialization_utils::*;
use crate::util_traits::*;
use super::vec_util_traits::*;
use std::ops::*;
use std::mem::ManuallyDrop;

mod macroed_vector_structs;
mod array_vector_structs;
mod binding_vector_structs;
mod generator_vector_structs;

pub use macroed_vector_structs::*;
pub use array_vector_structs::*;
pub use binding_vector_structs::*;
pub use generator_vector_structs::*;


#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct OwnedArray<T, const D: usize>(pub(crate) ManuallyDrop<[T; D]>);

impl<T, const D: usize> OwnedArray<T, D> {
    #[inline]
    pub fn unwrap(self) -> [T; D] {
        ManuallyDrop::into_inner(self.0)
    }
}

impl<T, const D: usize> Deref for OwnedArray<T, D> {
    type Target = ManuallyDrop<[T; D]>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const D: usize> DerefMut for OwnedArray<T, D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

unsafe impl<T, const D: usize> Get for OwnedArray<T, D> {
    type GetBool = Y;
    type Inputs = T;
    type Item = T;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {std::ptr::read(self.0.get_unchecked(index))}}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs, ())}
    #[inline] unsafe fn drop_inputs(&mut self, index: usize) { unsafe {std::ptr::drop_in_place(self.0.get_unchecked_mut(index))}}
}

//Safety: requires copy --> implies that items aren't invalidated after outputting --> Get can be repeated
unsafe impl<T: Copy, const D: usize> IsRepeatable for OwnedArray<T, D> {} 

impl<T, const D: usize> HasOutput for OwnedArray<T, D> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

impl<T, const D: usize> HasReuseBuf for OwnedArray<T, D> {
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


pub struct ReferringOwnedArray<'a, T: 'a, const D: usize>(pub(crate) [T; D], pub(crate) std::marker::PhantomData<&'a T>);

unsafe impl<'a, T: 'a, const D: usize> Get for ReferringOwnedArray<'a, T, D> {
    type GetBool = Y;
    type Inputs = &'a T;
    type Item = &'a T;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {&*(self.0.get_unchecked(index) as *const T)}}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs, ())}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}
}

unsafe impl<'a, T: 'a, const D: usize> IsRepeatable for ReferringOwnedArray<'a, T, D> {}

impl<'a, T: 'a, const D: usize> HasOutput for ReferringOwnedArray<'a, T, D> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

impl<'a, T: 'a, const D: usize> HasReuseBuf for ReferringOwnedArray<'a, T, D> {
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


unsafe impl<'a, T, const D: usize> Get for &'a [T; D] {
    type GetBool = Y;
    type Inputs = &'a T;
    type Item = &'a T;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {self.get_unchecked(index)}}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs, ())}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}
}

unsafe impl<'a, T, const D: usize> IsRepeatable for &'a [T; D] {}

impl<'a, T, const D: usize> HasOutput for &'a [T; D] {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

impl<'a, T, const D: usize> HasReuseBuf for &'a [T; D] {
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


unsafe impl<'a, T, const D: usize> Get for &'a mut [T; D] {
    type GetBool = Y;
    type Inputs = &'a mut T;
    type Item = &'a mut T;
    type BoundItems = ();

    //ptr shenanigans to change the lifetime
    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {&mut*(self.get_unchecked_mut(index) as *mut T)}}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs, ())}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}
}

impl<'a, T, const D: usize> HasOutput for &'a mut [T; D] {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

impl<'a, T, const D: usize> HasReuseBuf for &'a mut [T; D] {
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


unsafe impl<'a, T> Get for &'a [T] {
    type GetBool = Y;
    type Inputs = &'a T;
    type Item = &'a T;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {self.get_unchecked(index)}}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs, ())}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}
}

unsafe impl<'a, T> IsRepeatable for &'a [T] {}

impl<'a, T> HasOutput for &'a [T] {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

impl<'a, T> HasReuseBuf for &'a [T] {
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


unsafe impl<'a, T> Get for &'a mut [T] {
    type GetBool = Y;
    type Inputs = &'a mut T;
    type Item = &'a mut T;
    type BoundItems = ();

    //ptr shenanigans to change the lifetime
    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {&mut*(self.get_unchecked_mut(index) as *mut T)}}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs, ())}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}
}

impl<'a, T> HasOutput for &'a mut [T] {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

impl<'a, T> HasReuseBuf for &'a mut [T] {
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


#[inline] fn debox<T: ?Sized>(boxed: &mut Box<T>) -> &mut T {&mut *boxed}

unsafe impl<V: VectorLike + ?Sized> Get for Box<V> {
    type GetBool = V::GetBool;
    type Inputs = V::Inputs;
    type Item = V::Item;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {(debox(self)).get_inputs(index)}}
    #[inline] unsafe fn drop_inputs(&mut self, index: usize) { unsafe {(debox(self)).drop_inputs(index)}}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(debox(self)).process(inputs)}
}

unsafe impl<V: IsRepeatable + ?Sized> IsRepeatable for Box<V> {}

impl<V: VectorLike + ?Sized> HasReuseBuf for Box<V> {
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

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) { unsafe {(debox(self)).assign_1st_buf(index, val)}}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) { unsafe {(debox(self)).assign_2nd_buf(index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) { unsafe {(debox(self)).assign_bound_bufs(index, val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {(debox(self)).get_1st_buffer()}}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {(debox(self)).get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) { unsafe {(debox(self)).drop_1st_buf_index(index)}}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {(debox(self)).drop_2nd_buf_index(index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {(debox(self)).drop_bound_bufs_index(index)}}
}



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
    type GetBool = T::GetBool; // NOTE: N because offset_index adds a small amount of extra computation on get
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

/// SAFETY: it is expected that the used_vec field is safe to output in addition to normal correct implementation
pub struct VecAttachUsedVec<V: VectorLike, USEDV: VectorLike>{pub(crate) vec: V, pub(crate) used_vec: USEDV}

unsafe impl<V: VectorLike, USEDV: VectorLike> Get for VecAttachUsedVec<V, USEDV> {
    type GetBool = V::GetBool;
    type Inputs = V::Inputs;
    type Item = V::Item;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {self.vec.get_inputs(index)}}
    #[inline] unsafe fn drop_inputs(&mut self, index: usize) { unsafe {self.vec.drop_inputs(index)}}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(inputs)}
}

unsafe impl<V: IsRepeatable + VectorLike, USEDV: VectorLike> IsRepeatable for VecAttachUsedVec<V, USEDV> {}

impl<V: VectorLike, USEDV: VectorLike> HasOutput for VecAttachUsedVec<V, USEDV> where (V::OutputBool, USEDV::OutputBool): FilterPair {
    type OutputBool = <(V::OutputBool, USEDV::OutputBool) as TyBoolPair>::Or;
    type Output = <(V::OutputBool, USEDV::OutputBool) as FilterPair>::Filtered<V::Output, USEDV::Output>;

    #[inline] 
    unsafe fn output(&mut self) -> Self::Output { unsafe {
        <(V::OutputBool, USEDV::OutputBool) as FilterPair>::filter(
            self.vec.output(),
            self.used_vec.output()
        )
    }}

    #[inline]
    unsafe fn drop_output(&mut self) { unsafe {
        self.vec.drop_output();
        self.used_vec.drop_output();
    }}
}

impl<V: VectorLike, USEDV: VectorLike> HasReuseBuf for VecAttachUsedVec<V, USEDV> 
where 
    (V::FstOwnedBufferBool, USEDV::FstOwnedBufferBool): SelectPair,
    (V::SndOwnedBufferBool, USEDV::SndOwnedBufferBool): SelectPair,
    (V::FstHandleBool, USEDV::FstHandleBool): SelectPair,
    (V::SndHandleBool, USEDV::SndHandleBool): SelectPair,
    (V::BoundHandlesBool, USEDV::BoundHandlesBool): FilterPair
{
    type FstHandleBool = <(V::FstHandleBool, USEDV::FstHandleBool) as TyBoolPair>::Xor;
    type SndHandleBool = <(V::SndHandleBool, USEDV::SndHandleBool) as TyBoolPair>::Xor;
    type BoundHandlesBool = V::BoundHandlesBool;
    type FstOwnedBufferBool = <(V::FstOwnedBufferBool, USEDV::FstOwnedBufferBool) as TyBoolPair>::Xor; 
    type SndOwnedBufferBool = <(V::SndOwnedBufferBool, USEDV::SndOwnedBufferBool) as TyBoolPair>::Xor; 
    type FstOwnedBuffer = <(V::FstOwnedBufferBool, USEDV::FstOwnedBufferBool) as SelectPair>::Selected<V::FstOwnedBuffer, USEDV::FstOwnedBuffer>;
    type SndOwnedBuffer = <(V::SndOwnedBufferBool, USEDV::SndOwnedBufferBool) as SelectPair>::Selected<V::SndOwnedBuffer, USEDV::SndOwnedBuffer>;
    type FstType = <(V::FstHandleBool, USEDV::FstHandleBool) as SelectPair>::Selected<V::FstType, USEDV::FstType>;
    type SndType = <(V::SndHandleBool, USEDV::SndHandleBool) as SelectPair>::Selected<V::SndType, USEDV::SndType>;
    type BoundTypes = V::BoundTypes;

    #[inline] unsafe fn assign_1st_buf<'z>(&'z mut self, index: usize, val: Self::FstType) { unsafe {
        let (l_val, r_val) = <(V::FstHandleBool, USEDV::FstHandleBool) as SelectPair>::deselect(val);
        self.vec.assign_1st_buf(index, l_val);
        self.used_vec.assign_1st_buf(index, r_val);
    }}
    #[inline] unsafe fn assign_2nd_buf<'z>(&'z mut self, index: usize, val: Self::SndType) { unsafe {
        let (l_val, r_val) = <(V::SndHandleBool, USEDV::SndHandleBool) as SelectPair>::deselect(val);
        self.vec.assign_2nd_buf(index, l_val);
        self.used_vec.assign_2nd_buf(index, r_val);
    }}
    #[inline] unsafe fn assign_bound_bufs<'z>(&'z mut self, index: usize, val: Self::BoundTypes) { unsafe {
        self.vec.assign_bound_bufs(index, val);
    }}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {
        <(V::FstOwnedBufferBool, USEDV::FstOwnedBufferBool) as SelectPair>::select(self.vec.get_1st_buffer(), self.used_vec.get_1st_buffer())
    }}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {
        <(V::SndOwnedBufferBool, USEDV::SndOwnedBufferBool) as SelectPair>::select(self.vec.get_2nd_buffer(), self.used_vec.get_2nd_buffer())
    }}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) { unsafe {
        self.vec.drop_1st_buf_index(index);
        self.used_vec.drop_1st_buf_index(index);
    }}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {
        self.vec.drop_2nd_buf_index(index);
        self.used_vec.drop_2nd_buf_index(index);
    }}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {
        self.vec.drop_bound_bufs_index(index);
        self.used_vec.drop_bound_bufs_index(index);
    }}
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