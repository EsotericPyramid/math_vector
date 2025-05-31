use crate::trait_specialization_utils::*;
use crate::util_traits::*;
use crate::matrix::mat_util_traits::*;
use crate::matrix::{MathMatrix, MatrixExpr};
use super::Owned2DArray;
use std::mem::ManuallyDrop;

/// an owned 2d array which acts as a buffer for Has2dReuseBuf (in first slot)
pub struct Replace2DArray<T, const D1: usize, const D2: usize>(pub(crate) ManuallyDrop<[[T; D1]; D2]>);

unsafe impl<T, const D1: usize, const D2: usize> Get2D for Replace2DArray<T, D1, D2> {
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
    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) { unsafe {
        std::ptr::drop_in_place(self.0.get_unchecked_mut(col_index).get_unchecked_mut(row_index))
    }}

    #[inline]
    fn process(&mut self, _: usize, _: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs, ())}
}

impl<T: Sized, const D1: usize, const D2: usize> HasOutput for Replace2DArray<T, D1, D2> {
    type OutputBool = N;
    type Output = ();

    #[inline]
    unsafe fn output(&mut self) -> Self::Output {}

    #[inline]
    unsafe fn drop_output(&mut self) {} // dropped through reuse buf instead
}

impl<T, const D1: usize, const D2: usize> Has2DReuseBuf for Replace2DArray<T, D1, D2> {
    type FstHandleBool = Y;
    type SndHandleBool = N;
    type BoundHandlesBool = N;
    type FstOwnedBufferBool = Y;
    type SndOwnedBufferBool = N;
    type IsFstBufferTransposed = N;
    type IsSndBufferTransposed = N;
    type AreBoundBuffersTransposed = N;
    type FstOwnedBuffer = MathMatrix<T, D1, D2>;
    type SndOwnedBuffer = ();
    type FstType = T;
    type SndType = ();
    type BoundTypes = ();

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) { unsafe {
        std::ptr::write(self.0.get_unchecked_mut(col_index).get_unchecked_mut(row_index), val)
    }}
    #[inline] unsafe fn assign_2nd_buf(&mut self, _: usize, _: usize, _: Self::SndType) {}
    #[inline] unsafe fn assign_bound_bufs(&mut self, _: usize, _: usize, _: Self::BoundTypes) {}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {
        MatrixExpr(Owned2DArray(std::ptr::read(&self.0)))
    }}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {
        std::ptr::drop_in_place(self.0.get_unchecked_mut(col_index).get_unchecked_mut(row_index))
    }}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, _: usize, _: usize) {}
}

/// struct attaching a &mut 2d array / &mut MathMatrix as a buffer (in first slot)
pub struct MatAttach2DBuf<'a, M: MatrixLike<FstHandleBool = N>, T, const D1: usize, const D2: usize>{pub(crate) mat: M, pub(crate) buf: &'a mut [[T; D1]; D2]}

unsafe impl<'a, M: MatrixLike<FstHandleBool = N>, T, const D1: usize, const D2: usize> Get2D for MatAttach2DBuf<'a, M, T, D1, D2> {
    type GetBool = M::GetBool;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs { unsafe {self.mat.get_inputs(col_index, row_index)}}
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_inputs(col_index, row_index)}}
    #[inline] fn process(&mut self, col_index: usize, row_index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.mat.process(col_index, row_index,  inputs)}
}

unsafe impl<'a, M: Is2DRepeatable + MatrixLike<FstHandleBool = N>, T, const D1: usize, const D2: usize> Is2DRepeatable for MatAttach2DBuf<'a, M, T, D1, D2> {}

impl<'a, M: MatrixLike<FstHandleBool = N>, T, const D1: usize, const D2: usize> HasOutput for MatAttach2DBuf<'a, M, T, D1, D2> {
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.mat.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.mat.drop_output()}}
}

impl<'b, M: MatrixLike<FstHandleBool = N>, T, const D1: usize, const D2: usize> Has2DReuseBuf for MatAttach2DBuf<'b, M, T, D1, D2> {
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

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) { unsafe {*self.buf.get_unchecked_mut(col_index).get_unchecked_mut(row_index) = val}}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) { unsafe {self.mat.assign_2nd_buf(col_index, row_index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) { unsafe {self.mat.assign_bound_bufs(col_index, row_index, val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.mat.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_2nd_buf_index(col_index, row_index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_bound_bufs_index(col_index, row_index)}}
}

/// struct creating a buffer in the first slot
pub struct MatCreate2DBuf<M: MatrixLike<FstHandleBool = N>, T, const D1: usize, const D2: usize>{pub(crate) mat: M, pub(crate) buf: [[std::mem::MaybeUninit<T>; D1]; D2]}

unsafe impl<M: MatrixLike<FstHandleBool = N>, T, const D1: usize, const D2: usize> Get2D for MatCreate2DBuf<M, T, D1, D2> {
    type GetBool = M::GetBool;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs { unsafe {self.mat.get_inputs(col_index, row_index)}}
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_inputs(col_index, row_index)}}
    #[inline] fn process(&mut self, col_index: usize, row_index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.mat.process(col_index, row_index,  inputs)}
}

unsafe impl<M: Is2DRepeatable + MatrixLike<FstHandleBool = N>, T, const D1: usize, const D2: usize> Is2DRepeatable for MatCreate2DBuf<M, T, D1, D2> {}

impl<M: MatrixLike<FstHandleBool = N>, T, const D1: usize, const D2: usize> HasOutput for MatCreate2DBuf<M, T, D1, D2> {
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.mat.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.mat.drop_output()}}
}

impl<M: MatrixLike<FstHandleBool = N>, T, const D1: usize, const D2: usize> Has2DReuseBuf for MatCreate2DBuf<M, T, D1, D2> {
    type FstHandleBool = Y;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = Y;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type IsFstBufferTransposed = N;
    type IsSndBufferTransposed = M::IsSndBufferTransposed;
    type AreBoundBuffersTransposed = M::AreBoundBuffersTransposed;
    type FstOwnedBuffer = MathMatrix<T, D1, D2>;
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = T;
    type SndType = M::SndType;
    type BoundTypes = M::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) { unsafe {std::ptr::write(self.buf.get_unchecked_mut(col_index).get_unchecked_mut(row_index), std::mem::MaybeUninit::new(val))}}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) { unsafe {self.mat.assign_2nd_buf(col_index, row_index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) { unsafe {self.mat.assign_bound_bufs(col_index, row_index, val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {
        MatrixExpr(Owned2DArray(std::mem::transmute_copy::<[[std::mem::MaybeUninit<T>; D1]; D2], ManuallyDrop<[[T; D1]; D2]>>(&self.buf)))
    }}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.mat.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_2nd_buf_index(col_index, row_index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_bound_bufs_index(col_index, row_index)}}
}

/// struct creating a buffer on the heap in the first slot
pub struct MatCreate2DHeapBuf<M: MatrixLike<FstHandleBool = N>, T, const D1: usize, const D2: usize>{pub(crate) mat: M, pub(crate) buf: Box<[[std::mem::MaybeUninit<T>; D1]; D2]>}

unsafe impl<M: MatrixLike<FstHandleBool = N>, T, const D1: usize, const D2: usize> Get2D for MatCreate2DHeapBuf<M, T, D1, D2> {
    type GetBool = M::GetBool;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs { unsafe {self.mat.get_inputs(col_index, row_index)}}
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_inputs(col_index, row_index)}}
    #[inline] fn process(&mut self, col_index: usize, row_index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.mat.process(col_index, row_index,  inputs)}
}

unsafe impl<M: Is2DRepeatable + MatrixLike<FstHandleBool = N>, T, const D1: usize, const D2: usize> Is2DRepeatable for MatCreate2DHeapBuf<M, T, D1, D2> {}

impl<M: MatrixLike<FstHandleBool = N>, T, const D1: usize, const D2: usize> HasOutput for MatCreate2DHeapBuf<M, T, D1, D2> {
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.mat.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.mat.drop_output()}}
}

impl<M: MatrixLike<FstHandleBool = N>, T, const D1: usize, const D2: usize> Has2DReuseBuf for MatCreate2DHeapBuf<M, T, D1, D2> {
    type FstHandleBool = Y;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = Y;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type IsFstBufferTransposed = N;
    type IsSndBufferTransposed = M::IsSndBufferTransposed;
    type AreBoundBuffersTransposed = M::AreBoundBuffersTransposed;
    type FstOwnedBuffer = MathMatrix<T, D1, D2>;
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = T;
    type SndType = M::SndType;
    type BoundTypes = M::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) { unsafe {std::ptr::write(self.buf.get_unchecked_mut(col_index).get_unchecked_mut(row_index), std::mem::MaybeUninit::new(val))}}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) { unsafe {self.mat.assign_2nd_buf(col_index, row_index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) { unsafe {self.mat.assign_bound_bufs(col_index, row_index, val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {
        MatrixExpr(Owned2DArray(std::mem::transmute_copy::<[[std::mem::MaybeUninit<T>; D1]; D2], ManuallyDrop<[[T; D1]; D2]>>(&self.buf)))
    }}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.mat.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_2nd_buf_index(col_index, row_index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_bound_bufs_index(col_index, row_index)}}
}

/// struct creating a buffer in the first slot if there isn't already one there
pub struct MatMaybeCreate2DBuf<M: MatrixLike, T, const D1: usize, const D2: usize>  where <M::FstHandleBool as TyBool>::Neg: Filter {pub(crate) mat: M, pub(crate) buf: [[std::mem::MaybeUninit<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>; D1]; D2]}

unsafe impl<M: MatrixLike, T, const D1: usize, const D2: usize> Get2D for MatMaybeCreate2DBuf<M, T, D1, D2> where <M::FstHandleBool as TyBool>::Neg: Filter {
    type GetBool = M::GetBool;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs { unsafe {self.mat.get_inputs(col_index, row_index)}}
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_inputs(col_index, row_index)}}
    #[inline] fn process(&mut self, col_index: usize, row_index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.mat.process(col_index, row_index,  inputs)}
}

unsafe impl<M: Is2DRepeatable + MatrixLike, T, const D1: usize, const D2: usize> Is2DRepeatable for MatMaybeCreate2DBuf<M, T, D1, D2> where <M::FstHandleBool as TyBool>::Neg: Filter {}

impl<M: MatrixLike, T, const D1: usize, const D2: usize> HasOutput for MatMaybeCreate2DBuf<M, T, D1, D2> where <M::FstHandleBool as TyBool>::Neg: Filter {
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.mat.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.mat.drop_output()}}
}

impl<M: MatrixLike, T, const D1: usize, const D2: usize> Has2DReuseBuf for MatMaybeCreate2DBuf<M, T, D1, D2> 
where
    <M::FstHandleBool as TyBool>::Neg: Filter,
    (M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg): SelectPair,
    (M::FstOwnedBufferBool, <M::FstHandleBool as TyBool>::Neg): TyBoolPair
{
    type FstHandleBool = Y;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = <(M::FstOwnedBufferBool, <M::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type IsFstBufferTransposed = M::IsFstBufferTransposed; //only works cause default when FstHandleBool == N is N
    type IsSndBufferTransposed = M::IsSndBufferTransposed;
    type AreBoundBuffersTransposed = M::AreBoundBuffersTransposed;
    type FstOwnedBuffer = <(M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg) as SelectPair>::Selected<M::FstOwnedBuffer, MathMatrix<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>, D1, D2>>;
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = <(M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg) as SelectPair>::Selected<M::FstType, <<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>;
    type SndType = M::SndType;
    type BoundTypes = M::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) { unsafe {
        let (init_val, attached_val) = <(M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg) as SelectPair>::deselect(val);
        self.mat.assign_1st_buf(col_index, row_index, init_val);
        std::ptr::write(self.buf.get_unchecked_mut(col_index).get_unchecked_mut(row_index), std::mem::MaybeUninit::new(attached_val));
    }}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) { unsafe {self.mat.assign_2nd_buf(col_index, row_index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) { unsafe {self.mat.assign_bound_bufs(col_index, row_index, val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {
        <(M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg) as SelectPair>::select(
            self.mat.get_1st_buffer(),
            MatrixExpr(Owned2DArray(std::mem::transmute_copy::<[[std::mem::MaybeUninit<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>; D1]; D2], ManuallyDrop<[[<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>; D1]; D2]>>(&self.buf)))
        )
    }}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.mat.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_2nd_buf_index(col_index, row_index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_bound_bufs_index(col_index, row_index)}}
}

/// struct creating a buffer on the heap in the first slot if there isn't already one there
pub struct MatMaybeCreate2DHeapBuf<M: MatrixLike, T, const D1: usize, const D2: usize>  where <M::FstHandleBool as TyBool>::Neg: Filter {pub(crate) mat: M, pub(crate) buf: Box<[[std::mem::MaybeUninit<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>; D1]; D2]>}

unsafe impl<M: MatrixLike, T, const D1: usize, const D2: usize> Get2D for MatMaybeCreate2DHeapBuf<M, T, D1, D2> where <M::FstHandleBool as TyBool>::Neg: Filter {
    type GetBool = M::GetBool;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs { unsafe {self.mat.get_inputs(col_index, row_index)}}
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_inputs(col_index, row_index)}}
    #[inline] fn process(&mut self, col_index: usize, row_index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.mat.process(col_index, row_index,  inputs)}
}

unsafe impl<M: Is2DRepeatable + MatrixLike, T, const D1: usize, const D2: usize> Is2DRepeatable for MatMaybeCreate2DHeapBuf<M, T, D1, D2> where <M::FstHandleBool as TyBool>::Neg: Filter {}

impl<M: MatrixLike, T, const D1: usize, const D2: usize> HasOutput for MatMaybeCreate2DHeapBuf<M, T, D1, D2> where <M::FstHandleBool as TyBool>::Neg: Filter {
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.mat.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.mat.drop_output()}}
}

impl<M: MatrixLike, T, const D1: usize, const D2: usize> Has2DReuseBuf for MatMaybeCreate2DHeapBuf<M, T, D1, D2> 
where
    <M::FstHandleBool as TyBool>::Neg: Filter,
    (M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg): SelectPair,
    (M::FstOwnedBufferBool, <M::FstHandleBool as TyBool>::Neg): TyBoolPair
{
    type FstHandleBool = Y;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = <(M::FstOwnedBufferBool, <M::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type IsFstBufferTransposed = M::IsFstBufferTransposed; //only works cause default when FstHandleBool == N is N
    type IsSndBufferTransposed = M::IsSndBufferTransposed;
    type AreBoundBuffersTransposed = M::AreBoundBuffersTransposed;
    type FstOwnedBuffer = <(M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg) as SelectPair>::Selected<M::FstOwnedBuffer, Box<MathMatrix<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>, D1, D2>>>;
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = <(M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg) as SelectPair>::Selected<M::FstType, <<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>;
    type SndType = M::SndType;
    type BoundTypes = M::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) { unsafe {
        let (init_val, attached_val) = <(M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg) as SelectPair>::deselect(val);
        self.mat.assign_1st_buf(col_index, row_index, init_val);
        std::ptr::write(self.buf.get_unchecked_mut(col_index).get_unchecked_mut(row_index), std::mem::MaybeUninit::new(attached_val));
    }}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) { unsafe {self.mat.assign_2nd_buf(col_index, row_index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) { unsafe {self.mat.assign_bound_bufs(col_index, row_index, val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {
        <(M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg) as SelectPair>::select(
            self.mat.get_1st_buffer(),
            std::mem::transmute_copy::<Box<[[std::mem::MaybeUninit<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>; D1]; D2]>, Box<MathMatrix<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>, D1, D2>>>(&self.buf)
        )
    }}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.mat.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_2nd_buf_index(col_index, row_index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_bound_bufs_index(col_index, row_index)}}
}
