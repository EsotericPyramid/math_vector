//! Module containing all of the various VectorLike structs

use crate::{
    trait_specialization_utils::*,
    util_traits::*,
};
use super::vec_util_traits::*;
use std::{
    mem::ManuallyDrop,
    ops::*,
    ptr,
};

// NOTE: vector_structs internally split across multiple files to keep them small and navigable
mod array_vector_structs;
mod binding_vector_structs;
mod generator_vector_structs;
mod macroed_vector_structs;
mod misc_vector_structs;
mod ordering_vector_structs;
mod slice_vector_structs;

pub use array_vector_structs::*;
pub use binding_vector_structs::*;
pub use generator_vector_structs::*;
pub use macroed_vector_structs::*;
pub use misc_vector_structs::*;
pub use ordering_vector_structs::*;
pub use slice_vector_structs::*;

/// an owned array rigged up to manually drop via the VectorLike traits
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

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {ptr::read(self.0.get_unchecked(index))}}
    #[inline] fn process(&mut self, _: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs, ())}
    #[inline] unsafe fn drop_inputs(&mut self, index: usize) { unsafe {ptr::drop_in_place(self.0.get_unchecked_mut(index))}}
}

// Safety: requires copy --> implies that items aren't invalidated after outputting --> Get can be repeated
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
    #[inline] unsafe fn drop_1st_buffer(&mut self) {}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) {}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, _: usize) {}
}

/// an owned array rigged up to repeatable return references to its elements via Get
pub struct ReferringOwnedArray<'a, T: 'a, const D: usize>(pub(crate) [T; D], pub(crate) std::marker::PhantomData<&'a T>);

unsafe impl<'a, T: 'a, const D: usize> Get for ReferringOwnedArray<'a, T, D> {
    type GetBool = Y;
    type Inputs = &'a T;
    type Item = &'a T;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {&*(self.0.get_unchecked(index) as *const T)}}
    #[inline] fn process(&mut self, _: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs, ())}
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
    #[inline] unsafe fn drop_1st_buffer(&mut self) {}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) {}
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
    #[inline] fn process(&mut self, _: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs, ())}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}
}

unsafe impl<'a, T, const D: usize> IsRepeatable for &'a [T; D] {}

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
    #[inline] unsafe fn drop_1st_buffer(&mut self) {}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) {}
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
    #[inline] fn process(&mut self, _: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs, ())}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}
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
    #[inline] unsafe fn drop_1st_buffer(&mut self) {}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) {}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, _: usize) {}
}


#[repr(transparent)]
pub struct OwnedSlice<T>(pub(crate) Box<ManuallyDrop<[T]>>);

impl<T> OwnedSlice<T> {
    #[inline]
    pub fn unwrap(self) -> Box<[T]> {
        unsafe {std::mem::transmute::<Box<ManuallyDrop<[T]>>, Box<[T]>>(self.0)}
    }
}

impl<T> Deref for OwnedSlice<T> {
    type Target = ManuallyDrop<[T]>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for OwnedSlice<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

unsafe impl<T> Get for OwnedSlice<T> {
    type GetBool = Y;
    type Inputs = T;
    type Item = T;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {ptr::read(self.0.get_unchecked(index))}}
    #[inline] fn process(&mut self, _: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs, ())}
    #[inline] unsafe fn drop_inputs(&mut self, index: usize) { unsafe {ptr::drop_in_place(self.0.get_unchecked_mut(index))}}
}

// Safety: requires copy --> implies that items aren't invalidated after outputting --> Get can be repeated
unsafe impl<T: Copy> IsRepeatable for OwnedSlice<T> {} 

impl<T> HasOutput for OwnedSlice<T> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

impl<T> HasReuseBuf for OwnedSlice<T> {
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
    #[inline] unsafe fn drop_1st_buffer(&mut self) {}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) {}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, _: usize) {}
}

/// an owned slice rigged up to repeatable return references to its elements via Get
pub struct ReferringOwnedSlice<'a, T: 'a>(pub(crate) Box<[T]>, pub(crate) std::marker::PhantomData<&'a T>);

unsafe impl<'a, T: 'a> Get for ReferringOwnedSlice<'a, T> {
    type GetBool = Y;
    type Inputs = &'a T;
    type Item = &'a T;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {&*(self.0.get_unchecked(index) as *const T)}}
    #[inline] fn process(&mut self, _: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs, ())}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}
}

unsafe impl<'a, T: 'a> IsRepeatable for ReferringOwnedSlice<'a, T> {}

impl<'a, T: 'a> HasOutput for ReferringOwnedSlice<'a, T> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

impl<'a, T: 'a> HasReuseBuf for ReferringOwnedSlice<'a, T> {
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
    #[inline] unsafe fn drop_1st_buffer(&mut self) {}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) {}
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
    #[inline] fn process(&mut self, _: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs, ())}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}
}

unsafe impl<'a, T> IsRepeatable for &'a [T] {}

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
    #[inline] unsafe fn drop_1st_buffer(&mut self) {}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) {}
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
    #[inline] fn process(&mut self, _: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs, ())}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}
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
    #[inline] unsafe fn drop_1st_buffer(&mut self) {}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) {}
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
    #[inline] fn process(&mut self, index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(debox(self)).process(index, inputs)}
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
    #[inline] unsafe fn drop_1st_buffer(&mut self) { unsafe {(debox(self)).drop_1st_buffer()}}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) { unsafe {(debox(self)).drop_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) { unsafe {(debox(self)).drop_1st_buf_index(index)}}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {(debox(self)).drop_2nd_buf_index(index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {(debox(self)).drop_bound_bufs_index(index)}}
}