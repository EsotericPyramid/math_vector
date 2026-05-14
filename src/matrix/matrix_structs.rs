//! Module containing all of the various MatrixLike structs

use super::mat_util_traits::*;
use crate::{trait_specialization_utils::*, util_traits::*};
use std::{mem::ManuallyDrop, ops::*, ptr};

mod array_matrix_structs;
mod slice_matrix_structs;
mod binding_matrix_structs;
mod generator_matrix_structs;
mod macroed_matrix_structs;
mod math_matrix_structs;
mod misc_matrix_structs;
mod ordering_matrix_structs;
mod vectorizing_matrix_structs;

pub use array_matrix_structs::*;
pub use slice_matrix_structs::*;
pub use binding_matrix_structs::*;
pub use generator_matrix_structs::*;
pub use macroed_matrix_structs::*;
pub use math_matrix_structs::*;
pub use misc_matrix_structs::*;
pub use ordering_matrix_structs::*;
pub use vectorizing_matrix_structs::*;

macro_rules! HasOutput_non_impl {
    (impl<{$($gen_def:tt)*}> $struct:ty) => {
        impl<$($gen_def)*> HasOutput for $struct {
            type OutputBool = N;
            type Output = ();
        
            #[inline]
            unsafe fn output(&mut self) -> Self::Output {}
            #[inline]
            unsafe fn drop_output(&mut self) {}
        }
    };
}

macro_rules! Has2DReuseBuf_non_impl {
    (impl<{$($gen_def:tt)*}> $struct:ty) => {
        impl<$($gen_def)*> Has2DReuseBuf for $struct {
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

            #[inline]
            unsafe fn assign_1st_buf(&mut self, _: usize, _: usize, _: Self::FstType) {}
            #[inline]
            unsafe fn assign_2nd_buf(&mut self, _: usize, _: usize, _: Self::SndType) {}
            #[inline]
            unsafe fn assign_bound_bufs(&mut self, _: usize, _: usize, _: Self::BoundTypes) {}
            #[inline]
            unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
            #[inline]
            unsafe fn get_2nd_buffer(&mut self) -> Self::FstOwnedBuffer {}
            #[inline]
            unsafe fn drop_1st_buffer(&mut self) {}
            #[inline]
            unsafe fn drop_2nd_buffer(&mut self) {}
            #[inline]
            unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
            #[inline]
            unsafe fn drop_2nd_buf_index(&mut self, _: usize, _: usize) {}
            #[inline]
            unsafe fn drop_bound_bufs_index(&mut self, _: usize, _: usize) {}
        }
    };
}

/// an owned 2d array rigged up to manually drop via the MatrixLike traits
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct MatrixArray<T, const D1: usize, const D2: usize>(
    pub(crate) ManuallyDrop<[[T; D1]; D2]>,
);

impl<T, const D1: usize, const D2: usize> MatrixArray<T, D1, D2> {
    #[inline]
    pub fn unwrap(self) -> [[T; D1]; D2] {
        ManuallyDrop::into_inner(self.0)
    }
}

impl<T, const D1: usize, const D2: usize> Deref for MatrixArray<T, D1, D2> {
    type Target = ManuallyDrop<[[T; D1]; D2]>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const D1: usize, const D2: usize> DerefMut for MatrixArray<T, D1, D2> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

unsafe impl<T, const D1: usize, const D2: usize> Get2D for MatrixArray<T, D1, D2> {
    type GetBool = Y;
    type AreInputsTransposed = N;
    type Inputs = T;
    type Item = T;
    type BoundItems = ();

    #[inline]
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {
        unsafe { ptr::read(self.0.get_unchecked(col_index).get_unchecked(row_index)) }
    }
    #[inline]
    fn process(
        &mut self,
        _: usize,
        _: usize,
        inputs: Self::Inputs,
    ) -> (Self::Item, Self::BoundItems) {
        (inputs, ())
    }

    #[inline]
    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {
        unsafe {
            ptr::drop_in_place(
                self.0
                    .get_unchecked_mut(col_index)
                    .get_unchecked_mut(row_index),
            )
        }
    }
}

unsafe impl<T: Copy, const D1: usize, const D2: usize> Is2DRepeatable for MatrixArray<T, D1, D2> {}

HasOutput_non_impl!(impl<{T, const D1: usize, const D2: usize}> MatrixArray<T, D1, D2>);
Has2DReuseBuf_non_impl!(impl<{T, const D1: usize, const D2: usize}> MatrixArray<T, D1, D2>);

/// an owned 2d array returning references to its items and which is rigged up to manually drop via the MatrixLike traits
pub struct ReferringMatrixArray<'a, T: 'a, const D1: usize, const D2: usize>(
    pub(crate) [[T; D1]; D2],
    pub(crate) std::marker::PhantomData<&'a T>,
);

unsafe impl<'a, T: 'a, const D1: usize, const D2: usize> Get2D for ReferringMatrixArray<'a, T, D1, D2> {
    type GetBool = Y;
    type AreInputsTransposed = N;
    type Inputs = &'a T;
    type Item = &'a T;
    type BoundItems = ();

    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {
        unsafe { &*(self.0.get_unchecked(col_index).get_unchecked(row_index) as *const T) }
    }
    fn process(
        &mut self,
        _: usize,
        _: usize,
        inputs: Self::Inputs,
    ) -> (Self::Item, Self::BoundItems) {
        (inputs, ())
    }
    unsafe fn drop_inputs(&mut self, _: usize, _: usize) {}
}

unsafe impl<'a, T: 'a, const D1: usize, const D2: usize> Is2DRepeatable
    for ReferringMatrixArray<'a, T, D1, D2>
{
}

HasOutput_non_impl!(impl<{'a, T: 'a, const D1: usize, const D2: usize}> ReferringMatrixArray<'a, T, D1, D2>);
Has2DReuseBuf_non_impl!(impl<{'a, T: 'a, const D1: usize, const D2: usize}> ReferringMatrixArray<'a, T, D1, D2>);

//Note: these 2 technically impl HasOutput via vector_structs' impls on &[T; D], fine since none actually output anything
unsafe impl<'a, T, const D1: usize, const D2: usize> Get2D for &'a [[T; D1]; D2] {
    type GetBool = Y;
    type AreInputsTransposed = N;
    type Inputs = &'a T;
    type Item = &'a T;
    type BoundItems = ();

    #[inline]
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {
        unsafe { self.get_unchecked(col_index).get_unchecked(row_index) }
    }
    #[inline]
    fn process(
        &mut self,
        _: usize,
        _: usize,
        inputs: Self::Inputs,
    ) -> (Self::Item, Self::BoundItems) {
        (inputs, ())
    }
    #[inline]
    unsafe fn drop_inputs(&mut self, _: usize, _: usize) {}
}

unsafe impl<T, const D1: usize, const D2: usize> Is2DRepeatable for &[[T; D1]; D2] {}

Has2DReuseBuf_non_impl!(impl<{T, const D1: usize, const D2: usize}> &[[T; D1]; D2]);

unsafe impl<'a, T, const D1: usize, const D2: usize> Get2D for &'a mut [[T; D1]; D2] {
    type GetBool = Y;
    type AreInputsTransposed = N;
    type Inputs = &'a mut T;
    type Item = &'a mut T;
    type BoundItems = ();

    #[inline]
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {
        unsafe {
            &mut *(self
                .get_unchecked_mut(col_index)
                .get_unchecked_mut(row_index) as *mut T)
        }
    }
    #[inline]
    fn process(
        &mut self,
        _: usize,
        _: usize,
        inputs: Self::Inputs,
    ) -> (Self::Item, Self::BoundItems) {
        (inputs, ())
    }
    #[inline]
    unsafe fn drop_inputs(&mut self, _: usize, _: usize) {}
}

unsafe impl<T, const D1: usize, const D2: usize> Is2DRepeatable for &mut [[T; D1]; D2] {}

Has2DReuseBuf_non_impl!(impl<{T, const D1: usize, const D2: usize}> &mut [[T; D1]; D2]);

#[repr(transparent)]
pub struct MatrixIliffeSlice<T>(pub(crate) Box<[Box<[ManuallyDrop<T>]>]>);

impl<T> MatrixIliffeSlice<T> {
    #[inline]
    pub fn unwrap(self) -> Box<[Box<[T]>]> {
        unsafe { std::mem::transmute::<Box<[Box<[ManuallyDrop<T>]>]>, Box<[Box<[T]>]>>(self.0)}
    }
}

impl<T> Deref for MatrixIliffeSlice<T> {
    type Target = [Box<[ManuallyDrop<T>]>];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for MatrixIliffeSlice<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

unsafe impl<T> Get2D for MatrixIliffeSlice<T> {
    type GetBool = Y;
    type AreInputsTransposed = N;
    type Inputs = T;
    type Item = T;
    type BoundItems = ();

    #[inline]
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {
        unsafe { ptr::read(&**self.0.get_unchecked(col_index).get_unchecked(row_index)) }
    }
    #[inline]
    fn process(
        &mut self,
        _: usize,
        _: usize,
        inputs: Self::Inputs,
    ) -> (Self::Item, Self::BoundItems) {
        (inputs, ())
    }

    #[inline]
    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {
        unsafe {
            ptr::drop_in_place(
                &mut**self.0
                    .get_unchecked_mut(col_index)
                    .get_unchecked_mut(row_index),
            )
        }
    }
}

unsafe impl<T: Copy> Is2DRepeatable for MatrixIliffeSlice<T> {}

HasOutput_non_impl!(impl<{T}> MatrixIliffeSlice<T>);
Has2DReuseBuf_non_impl!(impl<{T}> MatrixIliffeSlice<T>);

unsafe impl<'a, T: 'a, S: Deref<Target = [T]>> Get2D for &'a [S] {
    type GetBool = Y;
    type AreInputsTransposed = N;
    type Inputs = &'a T;
    type Item = &'a T;
    type BoundItems = ();

    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {
        unsafe { self.get_unchecked(col_index).get_unchecked(row_index) }
    }

    fn process(
        &mut self,
        _: usize,
        _: usize,
        inputs: Self::Inputs,
    ) -> (Self::Item, Self::BoundItems) {
        (inputs, ())
    }

    unsafe fn drop_inputs(&mut self, _: usize, _: usize) {}
}

unsafe impl<'a, T: 'a, S: Deref<Target = [T]>> Is2DRepeatable for &'a [S] {}

// note: has a HasOutput non_impl from `vector_structs.rs` (as part of `&'a [T]`)
Has2DReuseBuf_non_impl!(impl<{'a, T: 'a, S: Deref<Target = [T]>}> &'a [S]);

unsafe impl<'a, T: 'a, S: DerefMut<Target = [T]>> Get2D for &'a mut [S] {
    type GetBool = Y;
    type AreInputsTransposed = N;
    type Inputs = &'a mut T;
    type Item = &'a mut T;
    type BoundItems = ();

    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {
        //ptr shenanigans to change the lifetime
        unsafe {&mut *(self.get_unchecked_mut(col_index).get_unchecked_mut(row_index) as *mut _)}
    }

    fn process(
        &mut self,
        _: usize,
        _: usize,
        inputs: Self::Inputs,
    ) -> (Self::Item, Self::BoundItems) {
        (inputs, ())
    }

    unsafe fn drop_inputs(&mut self, _: usize, _: usize) {}
}

// note: has a HasOutput non_impl from `vector_structs.rs` (as part of `&'a mut [T]`)
Has2DReuseBuf_non_impl!(impl<{'a, T: 'a, S: DerefMut<Target = [T]>}> &'a mut [S]);


pub struct MatrixDopeSlice<T>{pub(crate) mat: Box<[ManuallyDrop<T>]>, pub(crate) height: usize}

unsafe impl<T> Get2D for MatrixDopeSlice<T> {
    type GetBool = Y;
    type AreInputsTransposed = N;
    type Inputs = T;
    type Item = T;
    type BoundItems = ();

    #[inline]
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {
        unsafe { ptr::read(&**self.mat.get_unchecked(row_index + col_index * self.height)) }
    }
    #[inline]
    fn process(
        &mut self,
        _: usize,
        _: usize,
        inputs: Self::Inputs,
    ) -> (Self::Item, Self::BoundItems) {
        (inputs, ())
    }
    #[inline]
    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {
        unsafe { ptr::drop_in_place(&mut **self.mat.get_unchecked_mut(row_index + col_index * self.height)); }
    }
}

unsafe impl<T: Copy> Is2DRepeatable for MatrixDopeSlice<T> {}

HasOutput_non_impl!(impl<{T}> MatrixDopeSlice<T>);
Has2DReuseBuf_non_impl!(impl<{T}> MatrixDopeSlice<T>);

pub struct RefMatrixDopeSlice<'a, T>{pub(crate) mat: &'a [T], pub(crate) height: usize}

unsafe impl<'a, T> Get2D for RefMatrixDopeSlice<'a, T> {
    type GetBool = Y;
    type AreInputsTransposed = N;
    type Inputs = &'a T;
    type Item = &'a T;
    type BoundItems = ();

    #[inline]
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {
        unsafe{ &self.mat.get_unchecked(row_index + col_index * self.height) }
    }
    #[inline]
    fn process(
        &mut self,
        _: usize,
        _: usize,
        inputs: Self::Inputs,
    ) -> (Self::Item, Self::BoundItems) {
        (inputs, ())
    }
    #[inline]
    unsafe fn drop_inputs(&mut self, _: usize, _: usize) {}
}

unsafe impl<'a, T> Is2DRepeatable for RefMatrixDopeSlice<'a, T> {}

HasOutput_non_impl!(impl<{'a, T}> RefMatrixDopeSlice<'a, T>);
Has2DReuseBuf_non_impl!(impl<{'a, T}> RefMatrixDopeSlice<'a, T>);

pub struct RefMutMatrixDopeSlice<'a, T>{pub(crate) mat: &'a mut [T], pub(crate) height: usize}

unsafe impl<'a, T> Get2D for RefMutMatrixDopeSlice<'a, T> {
    type GetBool = Y;
    type AreInputsTransposed = N;
    type Inputs = &'a mut T;
    type Item = &'a mut T;
    type BoundItems = ();

    #[inline]
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {
        unsafe{ &mut *(self.mat.get_unchecked_mut(row_index + col_index * self.height) as *mut _) }
    }
    #[inline]
    fn process(
        &mut self,
        _: usize,
        _: usize,
        inputs: Self::Inputs,
    ) -> (Self::Item, Self::BoundItems) {
        (inputs, ())
    }
    #[inline]
    unsafe fn drop_inputs(&mut self, _: usize, _: usize) {}
}

HasOutput_non_impl!(impl<{'a, T}> RefMutMatrixDopeSlice<'a, T>);
Has2DReuseBuf_non_impl!(impl<{'a, T}> RefMutMatrixDopeSlice<'a, T>);

#[inline]
fn debox<T: ?Sized>(boxed: &mut Box<T>) -> &mut T {
    &mut *boxed
}

unsafe impl<M: MatrixLike + ?Sized> Get2D for Box<M> {
    type GetBool = M::GetBool;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline]
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {
        unsafe { (debox(self)).get_inputs(col_index, row_index) }
    }
    #[inline]
    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {
        unsafe { (debox(self)).drop_inputs(col_index, row_index) }
    }
    #[inline]
    fn process(
        &mut self,
        col_index: usize,
        row_index: usize,
        inputs: Self::Inputs,
    ) -> (Self::Item, Self::BoundItems) {
        (debox(self)).process(col_index, row_index, inputs)
    }
}

unsafe impl<M: MatrixLike + ?Sized> Is2DRepeatable for Box<M> {}

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

    #[inline]
    unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) {
        unsafe { (debox(self)).assign_1st_buf(col_index, row_index, val) }
    }
    #[inline]
    unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) {
        unsafe { (debox(self)).assign_2nd_buf(col_index, row_index, val) }
    }
    #[inline]
    unsafe fn assign_bound_bufs(
        &mut self,
        col_index: usize,
        row_index: usize,
        val: Self::BoundTypes,
    ) {
        unsafe { (debox(self)).assign_bound_bufs(col_index, row_index, val) }
    }
    #[inline]
    unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
        unsafe { (debox(self)).get_1st_buffer() }
    }
    #[inline]
    unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {
        unsafe { (debox(self)).get_2nd_buffer() }
    }
    #[inline]
    unsafe fn drop_1st_buffer(&mut self) {
        unsafe { (debox(self)).drop_1st_buffer() }
    }
    #[inline]
    unsafe fn drop_2nd_buffer(&mut self) {
        unsafe { (debox(self)).drop_2nd_buffer() }
    }
    #[inline]
    unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize) {
        unsafe { (debox(self)).drop_1st_buf_index(col_index, row_index) }
    }
    #[inline]
    unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) {
        unsafe { (debox(self)).drop_2nd_buf_index(col_index, row_index) }
    }
    #[inline]
    unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) {
        unsafe { (debox(self)).drop_bound_bufs_index(col_index, row_index) }
    }
}
