use super::{
    MatrixIliffeSlice,
    MatrixDopeSlice,
};
use crate::{
    matrix::{mat_util_traits::*, matrix_structs::RefMutMatrixDopeSlice, RSMathDopeMatrix, RSMathIliffeMatrix, RSMatrixExpr},
    trait_specialization_utils::*,
    util_traits::*,
};
use std::{
    mem::{transmute_copy, ManuallyDrop, MaybeUninit}, 
    ops::DerefMut, 
    ptr
};

/// an owned 2d slice (Iliffe style) which acts as a buffer for Has2dReuseBuf (in first slot)
pub struct ReplaceMatrixIliffeSlice<T>(
    pub(crate) ManuallyDrop<Box<[Box<[ManuallyDrop<T>]>]>>,
);

unsafe impl<T> Get2D for ReplaceMatrixIliffeSlice<T> {
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
    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {
        unsafe {
            ptr::drop_in_place(
                self.0
                    .get_unchecked_mut(col_index)
                    .get_unchecked_mut(row_index),
            )
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
}

impl<T: Sized> HasOutput for ReplaceMatrixIliffeSlice<T> {
    type OutputBool = N;
    type Output = ();

    #[inline]
    unsafe fn output(&mut self) -> Self::Output {}

    #[inline]
    unsafe fn drop_output(&mut self) {} // dropped through reuse buf instead
}

impl<T> Has2DReuseBuf for ReplaceMatrixIliffeSlice<T> {
    type FstHandleBool = Y;
    type SndHandleBool = N;
    type BoundHandlesBool = N;
    type FstOwnedBufferBool = Y;
    type SndOwnedBufferBool = N;
    type IsFstBufferTransposed = N;
    type IsSndBufferTransposed = N;
    type AreBoundBuffersTransposed = N;
    type FstOwnedBuffer = RSMathIliffeMatrix<T>;
    type SndOwnedBuffer = ();
    type FstType = T;
    type SndType = ();
    type BoundTypes = ();

    #[inline]
    unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) {
        unsafe {
            ptr::write(
                self.0
                    .get_unchecked_mut(col_index)
                    .get_unchecked_mut(row_index),
                ManuallyDrop::new(val),
            )
        }
    }
    #[inline]
    unsafe fn assign_2nd_buf(&mut self, _: usize, _: usize, _: Self::SndType) {}
    #[inline]
    unsafe fn assign_bound_bufs(&mut self, _: usize, _: usize, _: Self::BoundTypes) {}
    #[inline]
    unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
        unsafe { RSMatrixExpr{
            mat: MatrixIliffeSlice(ptr::read(&*self.0)),
            num_rows: if self.0.len() > 0 {self.0[0].len()} else {0},
            num_cols: self.0.len()
        } }
    }
    #[inline]
    unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {}
    #[inline]
    unsafe fn drop_1st_buffer(&mut self) {
        unsafe {ManuallyDrop::drop(&mut self.0)}
    }
    #[inline]
    unsafe fn drop_2nd_buffer(&mut self) {}
    #[inline]
    unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize) {
        unsafe {
            ptr::drop_in_place(
                self.0
                    .get_unchecked_mut(col_index)
                    .get_unchecked_mut(row_index),
            )
        }
    }
    #[inline]
    unsafe fn drop_2nd_buf_index(&mut self, _: usize, _: usize) {}
    #[inline]
    unsafe fn drop_bound_bufs_index(&mut self, _: usize, _: usize) {}
}

/// struct attaching a &mut 2d slice (Iliffe style) / &mut RSMathIliffeMatrix as a buffer (in first slot)
pub struct MatAttachIliffeSlice<'a, M: MatrixLike<FstHandleBool = N>, T, S: DerefMut<Target = [T]>>
{
    pub(crate) mat: M,
    pub(crate) buf: &'a mut [S],
}

unsafe impl<'a, M: MatrixLike<FstHandleBool = N>, T, S: DerefMut<Target = [T]>> Get2D
    for MatAttachIliffeSlice<'a, M, T, S>
{
    type GetBool = M::GetBool;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline]
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {
        unsafe { self.mat.get_inputs(col_index, row_index) }
    }
    #[inline]
    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {
        unsafe { self.mat.drop_inputs(col_index, row_index) }
    }
    #[inline]
    fn process(
        &mut self,
        col_index: usize,
        row_index: usize,
        inputs: Self::Inputs,
    ) -> (Self::Item, Self::BoundItems) {
        self.mat.process(col_index, row_index, inputs)
    }
}

unsafe impl<
    'a,
    M: Is2DRepeatable + MatrixLike<FstHandleBool = N>,
    T,
    S: DerefMut<Target = [T]>
> Is2DRepeatable for MatAttachIliffeSlice<'a, M, T, S>
{
}

impl<'a, M: MatrixLike<FstHandleBool = N>, T, S: DerefMut<Target = [T]>> HasOutput
    for MatAttachIliffeSlice<'a, M, T, S>
{
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline]
    unsafe fn output(&mut self) -> Self::Output {
        unsafe { self.mat.output() }
    }
    #[inline]
    unsafe fn drop_output(&mut self) {
        unsafe { self.mat.drop_output() }
    }
}

impl<'a, M: MatrixLike<FstHandleBool = N>, T, S: DerefMut<Target = [T]>> Has2DReuseBuf
    for MatAttachIliffeSlice<'a, M, T, S>
{
    type FstHandleBool = Y;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type IsFstBufferTransposed = N;
    type IsSndBufferTransposed = M::IsSndBufferTransposed;
    type AreBoundBuffersTransposed = M::AreBoundBuffersTransposed;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = T;
    type SndType = M::SndType;
    type BoundTypes = M::BoundTypes;

    #[inline]
    unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) {
        unsafe {
            *self
                .buf
                .get_unchecked_mut(col_index)
                .get_unchecked_mut(row_index) = val
        }
    }
    #[inline]
    unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) {
        unsafe { self.mat.assign_2nd_buf(col_index, row_index, val) }
    }
    #[inline]
    unsafe fn assign_bound_bufs(
        &mut self,
        col_index: usize,
        row_index: usize,
        val: Self::BoundTypes,
    ) {
        unsafe { self.mat.assign_bound_bufs(col_index, row_index, val) }
    }
    #[inline]
    unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline]
    unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {
        unsafe { self.mat.get_2nd_buffer() }
    }
    #[inline]
    unsafe fn drop_1st_buffer(&mut self) {}
    #[inline]
    unsafe fn drop_2nd_buffer(&mut self) {
        unsafe { self.mat.drop_2nd_buffer() }
    }
    #[inline]
    unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline]
    unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) {
        unsafe { self.mat.drop_2nd_buf_index(col_index, row_index) }
    }
    #[inline]
    unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) {
        unsafe { self.mat.drop_bound_bufs_index(col_index, row_index) }
    }
}

/// struct creating a 2d slice (Iliffe style) / RSMathIliffeMatrix in the first slot
pub struct MatCreateIliffeSlice<M: MatrixLike<FstHandleBool = N>, T> {
    pub(crate) mat: M,
    pub(crate) buf: ManuallyDrop<Box<[Box<[MaybeUninit<T>]>]>>,
}

unsafe impl<M: MatrixLike<FstHandleBool = N>, T> Get2D
    for MatCreateIliffeSlice<M, T>
{
    type GetBool = M::GetBool;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline]
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {
        unsafe { self.mat.get_inputs(col_index, row_index) }
    }
    #[inline]
    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {
        unsafe { self.mat.drop_inputs(col_index, row_index) }
    }
    #[inline]
    fn process(
        &mut self,
        col_index: usize,
        row_index: usize,
        inputs: Self::Inputs,
    ) -> (Self::Item, Self::BoundItems) {
        self.mat.process(col_index, row_index, inputs)
    }
}

unsafe impl<M: Is2DRepeatable + MatrixLike<FstHandleBool = N>, T>
    Is2DRepeatable for MatCreateIliffeSlice<M, T>
{
}

impl<M: MatrixLike<FstHandleBool = N>, T> HasOutput
    for MatCreateIliffeSlice<M, T>
{
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline]
    unsafe fn output(&mut self) -> Self::Output {
        unsafe { self.mat.output() }
    }
    #[inline]
    unsafe fn drop_output(&mut self) {
        unsafe { self.mat.drop_output() }
    }
}

impl<M: MatrixLike<FstHandleBool = N>, T> Has2DReuseBuf
    for MatCreateIliffeSlice<M, T>
{
    type FstHandleBool = Y;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = Y;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type IsFstBufferTransposed = N;
    type IsSndBufferTransposed = M::IsSndBufferTransposed;
    type AreBoundBuffersTransposed = M::AreBoundBuffersTransposed;
    type FstOwnedBuffer = RSMathIliffeMatrix<T>;
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = T;
    type SndType = M::SndType;
    type BoundTypes = M::BoundTypes;

    #[inline]
    unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) {
        unsafe {
            ptr::write(
                self.buf
                    .get_unchecked_mut(col_index)
                    .get_unchecked_mut(row_index),
                MaybeUninit::new(val),
            )
        }
    }
    #[inline]
    unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) {
        unsafe { self.mat.assign_2nd_buf(col_index, row_index, val) }
    }
    #[inline]
    unsafe fn assign_bound_bufs(
        &mut self,
        col_index: usize,
        row_index: usize,
        val: Self::BoundTypes,
    ) {
        unsafe { self.mat.assign_bound_bufs(col_index, row_index, val) }
    }
    #[inline]
    unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
        unsafe {
            RSMatrixExpr{
                mat: MatrixIliffeSlice(
                    transmute_copy::<ManuallyDrop<Box<[Box<[MaybeUninit<T>]>]>>, Box<[Box<[ManuallyDrop<T>]>]>>(&self.buf)
                ),
                num_rows: if self.buf.len() > 0 {self.buf[0].len()} else {0},
                num_cols: self.buf.len(),
            }
        }
    }
    #[inline]
    unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {
        unsafe { self.mat.get_2nd_buffer() }
    }
    #[inline]
    unsafe fn drop_1st_buffer(&mut self) {
        unsafe {ManuallyDrop::drop(&mut self.buf)}
    }
    #[inline]
    unsafe fn drop_2nd_buffer(&mut self) {
        unsafe { self.mat.drop_2nd_buffer() }
    }
    #[inline]
    unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline]
    unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) {
        unsafe { self.mat.drop_2nd_buf_index(col_index, row_index) }
    }
    #[inline]
    unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) {
        unsafe { self.mat.drop_bound_bufs_index(col_index, row_index) }
    }
}

/// struct creating a 2d slice (Iliffe style) / RSMathIliffeMatrix in the first slot if there isn't already one there
pub struct MatMaybeCreateIliffeSlice<M: MatrixLike, T>
where
    <M::FstHandleBool as TyBool>::Neg: Filter,
{
    pub(crate) mat: M,
    pub(crate) buf: ManuallyDrop<Box<[Box<[MaybeUninit<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>]>]>>,
}

unsafe impl<M: MatrixLike, T> Get2D
    for MatMaybeCreateIliffeSlice<M, T>
where
    <M::FstHandleBool as TyBool>::Neg: Filter,
{
    type GetBool = M::GetBool;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline]
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {
        unsafe { self.mat.get_inputs(col_index, row_index) }
    }
    #[inline]
    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {
        unsafe { self.mat.drop_inputs(col_index, row_index) }
    }
    #[inline]
    fn process(
        &mut self,
        col_index: usize,
        row_index: usize,
        inputs: Self::Inputs,
    ) -> (Self::Item, Self::BoundItems) {
        self.mat.process(col_index, row_index, inputs)
    }
}

unsafe impl<M: Is2DRepeatable + MatrixLike, T> Is2DRepeatable
    for MatMaybeCreateIliffeSlice<M, T>
where
    <M::FstHandleBool as TyBool>::Neg: Filter,
{
}

impl<M: MatrixLike, T> HasOutput
    for MatMaybeCreateIliffeSlice<M, T>
where
    <M::FstHandleBool as TyBool>::Neg: Filter,
{
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline]
    unsafe fn output(&mut self) -> Self::Output {
        unsafe { self.mat.output() }
    }
    #[inline]
    unsafe fn drop_output(&mut self) {
        unsafe { self.mat.drop_output() }
    }
}

impl<M: MatrixLike, T> Has2DReuseBuf
    for MatMaybeCreateIliffeSlice<M, T>
where
    <M::FstHandleBool as TyBool>::Neg: Filter,
    (M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg): SelectPair,
    (M::FstOwnedBufferBool, <M::FstHandleBool as TyBool>::Neg): TyBoolPair,
{
    type FstHandleBool = Y;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool =
        <(M::FstOwnedBufferBool, <M::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type IsFstBufferTransposed = M::IsFstBufferTransposed; //only works cause default when FstHandleBool == N is N
    type IsSndBufferTransposed = M::IsSndBufferTransposed;
    type AreBoundBuffersTransposed = M::AreBoundBuffersTransposed;
    type FstOwnedBuffer =
        <(M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg) as SelectPair>::Selected<
            M::FstOwnedBuffer,
            RSMathIliffeMatrix<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>,
        >;
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = <(M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg) as SelectPair>::Selected<
        M::FstType,
        <<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>,
    >;
    type SndType = M::SndType;
    type BoundTypes = M::BoundTypes;

    #[inline]
    unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) {
        unsafe {
            let (init_val, attached_val) =
                <(M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg) as SelectPair>::deselect(
                    val,
                );
            self.mat.assign_1st_buf(col_index, row_index, init_val);
            ptr::write(
                self.buf
                    .get_unchecked_mut(col_index)
                    .get_unchecked_mut(row_index),
                MaybeUninit::new(attached_val),
            );
        }
    }
    #[inline]
    unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) {
        unsafe { self.mat.assign_2nd_buf(col_index, row_index, val) }
    }
    #[inline]
    unsafe fn assign_bound_bufs(
        &mut self,
        col_index: usize,
        row_index: usize,
        val: Self::BoundTypes,
    ) {
        unsafe { self.mat.assign_bound_bufs(col_index, row_index, val) }
    }
    #[inline]
    unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
        unsafe {
            <(M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg) as SelectPair>::select(
                self.mat.get_1st_buffer(),
                RSMatrixExpr { 
                    mat: MatrixIliffeSlice(
                        transmute_copy::<
                            ManuallyDrop<Box<[Box<[MaybeUninit<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>]>]>>, 
                            Box<[Box<[ManuallyDrop<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>]>]>,
                        >(&self.buf)
                    ), 
                    num_rows: if self.buf.len() > 0 {self.buf[0].len()} else {0}, 
                    num_cols: self.buf.len(),
                },
            )
        }
    }
    #[inline]
    unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {
        unsafe { self.mat.get_2nd_buffer() }
    }
    #[inline]
    unsafe fn drop_1st_buffer(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.buf);
            self.mat.drop_1st_buffer() 
        }
    }
    #[inline]
    unsafe fn drop_2nd_buffer(&mut self) {
        unsafe { self.mat.drop_2nd_buffer() }
    }
    #[inline]
    unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline]
    unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) {
        unsafe { self.mat.drop_2nd_buf_index(col_index, row_index) }
    }
    #[inline]
    unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) {
        unsafe { self.mat.drop_bound_bufs_index(col_index, row_index) }
    }
}



/// an owned 2d slice (Iliffe style) which acts as a buffer for Has2dReuseBuf (in first slot)
pub struct ReplaceMatrixDopeSlice<T>{
    pub(crate) mat: ManuallyDrop<Box<[ManuallyDrop<T>]>>,
    pub(crate) height: usize,
}

unsafe impl<T> Get2D for ReplaceMatrixDopeSlice<T> {
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
    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {
        unsafe {
            ptr::drop_in_place(
                self.mat.get_unchecked_mut(row_index + col_index * self.height)
            )
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
}

impl<T: Sized> HasOutput for ReplaceMatrixDopeSlice<T> {
    type OutputBool = N;
    type Output = ();

    #[inline]
    unsafe fn output(&mut self) -> Self::Output {}

    #[inline]
    unsafe fn drop_output(&mut self) {} // dropped through reuse buf instead
}

impl<T> Has2DReuseBuf for ReplaceMatrixDopeSlice<T> {
    type FstHandleBool = Y;
    type SndHandleBool = N;
    type BoundHandlesBool = N;
    type FstOwnedBufferBool = Y;
    type SndOwnedBufferBool = N;
    type IsFstBufferTransposed = N;
    type IsSndBufferTransposed = N;
    type AreBoundBuffersTransposed = N;
    type FstOwnedBuffer = RSMathDopeMatrix<T>;
    type SndOwnedBuffer = ();
    type FstType = T;
    type SndType = ();
    type BoundTypes = ();

    #[inline]
    unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) {
        unsafe {
            ptr::write(
                self.mat.get_unchecked_mut(row_index + col_index * self.height),
                ManuallyDrop::new(val),
            )
        }
    }
    #[inline]
    unsafe fn assign_2nd_buf(&mut self, _: usize, _: usize, _: Self::SndType) {}
    #[inline]
    unsafe fn assign_bound_bufs(&mut self, _: usize, _: usize, _: Self::BoundTypes) {}
    #[inline]
    unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
        unsafe { RSMatrixExpr{
            mat: MatrixDopeSlice {
                mat: ptr::read(&*self.mat),
                height: self.height,
            },
            num_rows: self.height,
            num_cols: if self.height != 0 {self.mat.len() / self.height} else {0},
        } }
    }
    #[inline]
    unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {}
    #[inline]
    unsafe fn drop_1st_buffer(&mut self) {
        unsafe {ManuallyDrop::drop(&mut self.mat)}
    }
    #[inline]
    unsafe fn drop_2nd_buffer(&mut self) {}
    #[inline]
    unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize) {
        unsafe {
            ptr::drop_in_place(
                self.mat.get_unchecked_mut(row_index + col_index * self.height),
            )
        }
    }
    #[inline]
    unsafe fn drop_2nd_buf_index(&mut self, _: usize, _: usize) {}
    #[inline]
    unsafe fn drop_bound_bufs_index(&mut self, _: usize, _: usize) {}
}

/// struct attaching a &mut 2d slice (Iliffe style) / &mut RSMathDopeMatrix as a buffer (in first slot)
pub struct MatAttachDopeSlice<'a, M: MatrixLike<FstHandleBool = N>, T>
{
    pub(crate) mat: M,
    pub(crate) buf: RefMutMatrixDopeSlice<'a, T>,
}

unsafe impl<'a, M: MatrixLike<FstHandleBool = N>, T> Get2D
    for MatAttachDopeSlice<'a, M, T>
{
    type GetBool = M::GetBool;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline]
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {
        unsafe { self.mat.get_inputs(col_index, row_index) }
    }
    #[inline]
    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {
        unsafe { self.mat.drop_inputs(col_index, row_index) }
    }
    #[inline]
    fn process(
        &mut self,
        col_index: usize,
        row_index: usize,
        inputs: Self::Inputs,
    ) -> (Self::Item, Self::BoundItems) {
        self.mat.process(col_index, row_index, inputs)
    }
}

unsafe impl<
    'a,
    M: Is2DRepeatable + MatrixLike<FstHandleBool = N>,
    T,
> Is2DRepeatable for MatAttachDopeSlice<'a, M, T>
{
}

impl<'a, M: MatrixLike<FstHandleBool = N>, T> HasOutput
    for MatAttachDopeSlice<'a, M, T>
{
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline]
    unsafe fn output(&mut self) -> Self::Output {
        unsafe { self.mat.output() }
    }
    #[inline]
    unsafe fn drop_output(&mut self) {
        unsafe { self.mat.drop_output() }
    }
}

impl<'a, M: MatrixLike<FstHandleBool = N>, T> Has2DReuseBuf
    for MatAttachDopeSlice<'a, M, T>
{
    type FstHandleBool = Y;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type IsFstBufferTransposed = N;
    type IsSndBufferTransposed = M::IsSndBufferTransposed;
    type AreBoundBuffersTransposed = M::AreBoundBuffersTransposed;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = T;
    type SndType = M::SndType;
    type BoundTypes = M::BoundTypes;

    #[inline]
    unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) {
        unsafe {
            *self.buf.mat.get_unchecked_mut(row_index + col_index * self.buf.height) = val
        }
    }
    #[inline]
    unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) {
        unsafe { self.mat.assign_2nd_buf(col_index, row_index, val) }
    }
    #[inline]
    unsafe fn assign_bound_bufs(
        &mut self,
        col_index: usize,
        row_index: usize,
        val: Self::BoundTypes,
    ) {
        unsafe { self.mat.assign_bound_bufs(col_index, row_index, val) }
    }
    #[inline]
    unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline]
    unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {
        unsafe { self.mat.get_2nd_buffer() }
    }
    #[inline]
    unsafe fn drop_1st_buffer(&mut self) {}
    #[inline]
    unsafe fn drop_2nd_buffer(&mut self) {
        unsafe { self.mat.drop_2nd_buffer() }
    }
    #[inline]
    unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline]
    unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) {
        unsafe { self.mat.drop_2nd_buf_index(col_index, row_index) }
    }
    #[inline]
    unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) {
        unsafe { self.mat.drop_bound_bufs_index(col_index, row_index) }
    }
}

/// struct creating a 2d slice (Iliffe style) / RSMathDopeMatrix in the first slot
pub struct MatCreateDopeSlice<M: MatrixLike<FstHandleBool = N>, T> {
    pub(crate) mat: M,
    pub(crate) buf: ManuallyDrop<Box<[MaybeUninit<T>]>>,
    pub(crate) height: usize,
}

unsafe impl<M: MatrixLike<FstHandleBool = N>, T> Get2D
    for MatCreateDopeSlice<M, T>
{
    type GetBool = M::GetBool;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline]
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {
        unsafe { self.mat.get_inputs(col_index, row_index) }
    }
    #[inline]
    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {
        unsafe { self.mat.drop_inputs(col_index, row_index) }
    }
    #[inline]
    fn process(
        &mut self,
        col_index: usize,
        row_index: usize,
        inputs: Self::Inputs,
    ) -> (Self::Item, Self::BoundItems) {
        self.mat.process(col_index, row_index, inputs)
    }
}

unsafe impl<M: Is2DRepeatable + MatrixLike<FstHandleBool = N>, T>
    Is2DRepeatable for MatCreateDopeSlice<M, T>
{
}

impl<M: MatrixLike<FstHandleBool = N>, T> HasOutput
    for MatCreateDopeSlice<M, T>
{
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline]
    unsafe fn output(&mut self) -> Self::Output {
        unsafe { self.mat.output() }
    }
    #[inline]
    unsafe fn drop_output(&mut self) {
        unsafe { self.mat.drop_output() }
    }
}

impl<M: MatrixLike<FstHandleBool = N>, T> Has2DReuseBuf
    for MatCreateDopeSlice<M, T>
{
    type FstHandleBool = Y;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = Y;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type IsFstBufferTransposed = N;
    type IsSndBufferTransposed = M::IsSndBufferTransposed;
    type AreBoundBuffersTransposed = M::AreBoundBuffersTransposed;
    type FstOwnedBuffer = RSMathDopeMatrix<T>;
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = T;
    type SndType = M::SndType;
    type BoundTypes = M::BoundTypes;

    #[inline]
    unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) {
        unsafe {
            ptr::write(
                self.buf.get_unchecked_mut(row_index + col_index * self.height),
                MaybeUninit::new(val),
            )
        }
    }
    #[inline]
    unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) {
        unsafe { self.mat.assign_2nd_buf(col_index, row_index, val) }
    }
    #[inline]
    unsafe fn assign_bound_bufs(
        &mut self,
        col_index: usize,
        row_index: usize,
        val: Self::BoundTypes,
    ) {
        unsafe { self.mat.assign_bound_bufs(col_index, row_index, val) }
    }
    #[inline]
    unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
        unsafe {
            RSMatrixExpr{
                mat: MatrixDopeSlice{
                    mat: transmute_copy::<ManuallyDrop<Box<[MaybeUninit<T>]>>, Box<[ManuallyDrop<T>]>>(&self.buf),
                    height: self.height
                },
                num_rows: self.height,
                num_cols: if self.height != 0 {self.buf.len() / self.height} else {0},
            }
        }
    }
    #[inline]
    unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {
        unsafe { self.mat.get_2nd_buffer() }
    }
    #[inline]
    unsafe fn drop_1st_buffer(&mut self) {
        unsafe {ManuallyDrop::drop(&mut self.buf)}
    }
    #[inline]
    unsafe fn drop_2nd_buffer(&mut self) {
        unsafe { self.mat.drop_2nd_buffer() }
    }
    #[inline]
    unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline]
    unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) {
        unsafe { self.mat.drop_2nd_buf_index(col_index, row_index) }
    }
    #[inline]
    unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) {
        unsafe { self.mat.drop_bound_bufs_index(col_index, row_index) }
    }
}

/// struct creating a 2d slice (Iliffe style) / RSMathDopeMatrix in the first slot if there isn't already one there
pub struct MatMaybeCreateDopeSlice<M: MatrixLike, T>
where
    <M::FstHandleBool as TyBool>::Neg: Filter,
{
    pub(crate) mat: M,
    pub(crate) buf: ManuallyDrop<Box<[MaybeUninit<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>]>>,
    pub(crate) height: usize,
}

unsafe impl<M: MatrixLike, T> Get2D
    for MatMaybeCreateDopeSlice<M, T>
where
    <M::FstHandleBool as TyBool>::Neg: Filter,
{
    type GetBool = M::GetBool;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline]
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {
        unsafe { self.mat.get_inputs(col_index, row_index) }
    }
    #[inline]
    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {
        unsafe { self.mat.drop_inputs(col_index, row_index) }
    }
    #[inline]
    fn process(
        &mut self,
        col_index: usize,
        row_index: usize,
        inputs: Self::Inputs,
    ) -> (Self::Item, Self::BoundItems) {
        self.mat.process(col_index, row_index, inputs)
    }
}

unsafe impl<M: Is2DRepeatable + MatrixLike, T> Is2DRepeatable
    for MatMaybeCreateDopeSlice<M, T>
where
    <M::FstHandleBool as TyBool>::Neg: Filter,
{
}

impl<M: MatrixLike, T> HasOutput
    for MatMaybeCreateDopeSlice<M, T>
where
    <M::FstHandleBool as TyBool>::Neg: Filter,
{
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline]
    unsafe fn output(&mut self) -> Self::Output {
        unsafe { self.mat.output() }
    }
    #[inline]
    unsafe fn drop_output(&mut self) {
        unsafe { self.mat.drop_output() }
    }
}

impl<M: MatrixLike, T> Has2DReuseBuf
    for MatMaybeCreateDopeSlice<M, T>
where
    <M::FstHandleBool as TyBool>::Neg: Filter,
    (M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg): SelectPair,
    (M::FstOwnedBufferBool, <M::FstHandleBool as TyBool>::Neg): TyBoolPair,
{
    type FstHandleBool = Y;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool =
        <(M::FstOwnedBufferBool, <M::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type IsFstBufferTransposed = M::IsFstBufferTransposed; //only works cause default when FstHandleBool == N is N
    type IsSndBufferTransposed = M::IsSndBufferTransposed;
    type AreBoundBuffersTransposed = M::AreBoundBuffersTransposed;
    type FstOwnedBuffer =
        <(M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg) as SelectPair>::Selected<
            M::FstOwnedBuffer,
            RSMathDopeMatrix<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>,
        >;
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = <(M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg) as SelectPair>::Selected<
        M::FstType,
        <<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>,
    >;
    type SndType = M::SndType;
    type BoundTypes = M::BoundTypes;

    #[inline]
    unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) {
        unsafe {
            let (init_val, attached_val) =
                <(M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg) as SelectPair>::deselect(
                    val,
                );
            self.mat.assign_1st_buf(col_index, row_index, init_val);
            ptr::write(
                self.buf.get_unchecked_mut(row_index + col_index * self.height),
                MaybeUninit::new(attached_val),
            );
        }
    }
    #[inline]
    unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) {
        unsafe { self.mat.assign_2nd_buf(col_index, row_index, val) }
    }
    #[inline]
    unsafe fn assign_bound_bufs(
        &mut self,
        col_index: usize,
        row_index: usize,
        val: Self::BoundTypes,
    ) {
        unsafe { self.mat.assign_bound_bufs(col_index, row_index, val) }
    }
    #[inline]
    unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
        unsafe {
            <(M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg) as SelectPair>::select(
                self.mat.get_1st_buffer(),
                RSMatrixExpr { 
                    mat: MatrixDopeSlice{
                        mat: transmute_copy::<
                            ManuallyDrop<Box<[MaybeUninit<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>]>>, 
                            Box<[ManuallyDrop<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>]>,
                        >(&self.buf),
                        height: self.height,
                    }, 
                    num_rows: self.height, 
                    num_cols: if self.height != 0 {self.buf.len() / self.height} else {0},
                },
            )
        }
    }
    #[inline]
    unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {
        unsafe { self.mat.get_2nd_buffer() }
    }
    #[inline]
    unsafe fn drop_1st_buffer(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.buf);
            self.mat.drop_1st_buffer() 
        }
    }
    #[inline]
    unsafe fn drop_2nd_buffer(&mut self) {
        unsafe { self.mat.drop_2nd_buffer() }
    }
    #[inline]
    unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline]
    unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) {
        unsafe { self.mat.drop_2nd_buf_index(col_index, row_index) }
    }
    #[inline]
    unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) {
        unsafe { self.mat.drop_bound_bufs_index(col_index, row_index) }
    }
}