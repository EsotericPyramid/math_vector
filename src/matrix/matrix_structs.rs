use crate::trait_specialization_utils::*;
use crate::util_traits::*;
use super::mat_util_traits::*;
use super::MathMatrix;
use super::MatrixExpr;
use super::VectorizedMatrix;
use crate::vector::vec_util_traits::*;
use std::ops::*;
use std::mem::ManuallyDrop;

#[repr(transparent)]
#[derive(Clone,Copy)]
pub struct Owned2DArray<T,const D1: usize,const D2: usize>(pub(crate) ManuallyDrop<[[T; D1]; D2]>);

impl<T,const D1: usize,const D2: usize> Owned2DArray<T,D1,D2> {
    #[inline]
    pub fn unwrap(self) -> [[T; D1]; D2] {
        ManuallyDrop::into_inner(self.0)
    }
}

impl<T,const D1: usize,const D2: usize> Deref for Owned2DArray<T,D1,D2> {
    type Target = ManuallyDrop<[[T; D1]; D2]>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T,const D1: usize,const D2: usize> DerefMut for Owned2DArray<T,D1,D2> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

unsafe impl<T,const D1: usize,const D2: usize> Get2D for Owned2DArray<T,D1,D2> {
    type GetBool = Y;
    type IsRepeatable = N;
    type AreInputsTransposed = N;
    type Inputs = T;
    type Item = T;
    type BoundItems = ();

    #[inline] 
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {
        std::ptr::read(self.0.get_unchecked(col_index).get_unchecked(row_index))
    }
    #[inline]
    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
        (inputs,())
    }

    #[inline]
    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {
        std::ptr::drop_in_place(self.0.get_unchecked_mut(col_index).get_unchecked_mut(row_index))
    }
}

impl<T,const D1: usize,const D2: usize> HasOutput for Owned2DArray<T,D1,D2> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

impl<T,const D1: usize,const D2: usize> Has2DReuseBuf for Owned2DArray<T,D1,D2> {
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


unsafe impl<'a,T,const D1: usize,const D2: usize> Get2D for &'a [[T; D1]; D2] {
    type GetBool = Y;
    type IsRepeatable = Y;
    type AreInputsTransposed = N;
    type Inputs = &'a T;
    type Item = &'a T;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {self.get_unchecked(col_index).get_unchecked(row_index)}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs,())}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize, _: usize) {}
}

impl<'a,T,const D1: usize,const D2: usize> Has2DReuseBuf for &'a [[T; D1]; D2] {
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


unsafe impl<'a,T,const D1: usize,const D2: usize> Get2D for &'a mut [[T; D1]; D2] {
    type GetBool = Y;
    type IsRepeatable = Y;
    type AreInputsTransposed = N;
    type Inputs = &'a mut T;
    type Item = &'a mut T;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {&mut*(self.get_unchecked_mut(col_index).get_unchecked_mut(row_index) as *mut T)}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs,())}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize, _: usize) {}
}

impl<'a,T,const D1: usize,const D2: usize> Has2DReuseBuf for &'a mut [[T; D1]; D2] {
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


#[inline] fn debox<T: Sized>(boxed: &mut Box<T>) -> &mut T {&mut *boxed}

unsafe impl<M: MatrixLike> Get2D for Box<M> {
    type GetBool = M::GetBool;
    type IsRepeatable = M::IsRepeatable;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {(debox(self)).get_inputs(col_index,row_index)}
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {(debox(self)).drop_inputs(col_index,row_index)}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(debox(self)).process(inputs)}
}

impl<M: MatrixLike> Has2DReuseBuf for Box<M> {
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

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) {(debox(self)).assign_1st_buf(col_index,row_index,val)}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) {(debox(self)).assign_2nd_buf(col_index,row_index,val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) {(debox(self)).assign_bound_bufs(col_index,row_index,val)}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {(debox(self)).get_1st_buffer()}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {(debox(self)).get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize) {(debox(self)).drop_1st_buf_index(col_index,row_index)}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) {(debox(self)).drop_2nd_buf_index(col_index,row_index)}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) {(debox(self)).drop_bound_bufs_index(col_index,row_index)}
}


pub struct Replace2DArray<T,const D1: usize,const D2: usize>(pub(crate) ManuallyDrop<[[T; D1]; D2]>);

unsafe impl<T,const D1: usize,const D2: usize> Get2D for Replace2DArray<T,D1,D2> {
    type GetBool = Y;
    type IsRepeatable = N;
    type AreInputsTransposed = N;
    type Inputs = T;
    type Item = T;
    type BoundItems = ();


    #[inline]
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {
        std::ptr::read(self.0.get_unchecked(col_index).get_unchecked(row_index))
    }

    #[inline]
    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {
        std::ptr::drop_in_place(self.0.get_unchecked_mut(col_index).get_unchecked_mut(row_index))
    }

    #[inline]
    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs,())}
}

impl<T: Sized,const D1: usize,const D2: usize> HasOutput for Replace2DArray<T,D1,D2> {
    type OutputBool = N;
    type Output = ();

    #[inline]
    unsafe fn output(&mut self) -> Self::Output {}

    #[inline]
    unsafe fn drop_output(&mut self) {} // dropped through reuse buf instead
}

impl<T,const D1: usize,const D2: usize> Has2DReuseBuf for Replace2DArray<T,D1,D2> {
    type FstHandleBool = Y;
    type SndHandleBool = N;
    type BoundHandlesBool = N;
    type FstOwnedBufferBool = Y;
    type SndOwnedBufferBool = N;
    type IsFstBufferTransposed = N;
    type IsSndBufferTransposed = N;
    type AreBoundBuffersTransposed = N;
    type FstOwnedBuffer = MathMatrix<T,D1,D2>;
    type SndOwnedBuffer = ();
    type FstType = T;
    type SndType = ();
    type BoundTypes = ();

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) {
        std::ptr::write(self.0.get_unchecked_mut(col_index).get_unchecked_mut(row_index), val)
    }
    #[inline] unsafe fn assign_2nd_buf(&mut self, _: usize, _: usize, _: Self::SndType) {}
    #[inline] unsafe fn assign_bound_bufs(&mut self, _: usize, _: usize, _: Self::BoundTypes) {}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
        MatrixExpr(Owned2DArray(std::ptr::read(&self.0)))
    }
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize) {
        std::ptr::drop_in_place(self.0.get_unchecked_mut(col_index).get_unchecked_mut(row_index))
    }
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, _: usize, _: usize) {}
}


pub struct Replace2DHeapArray<T,const D1: usize,const D2: usize>(pub(crate) ManuallyDrop<Box<[[T; D1]; D2]>>);

unsafe impl<T,const D1: usize,const D2: usize> Get2D for Replace2DHeapArray<T,D1,D2> {
    type GetBool = Y;
    type IsRepeatable = N;
    type AreInputsTransposed = N;
    type Inputs = T;
    type Item = T;
    type BoundItems = ();


    #[inline]
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {
        std::ptr::read(self.0.get_unchecked(col_index).get_unchecked(row_index))
    }

    #[inline]
    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {
        std::ptr::drop_in_place(self.0.get_unchecked_mut(col_index).get_unchecked_mut(row_index))
    }

    #[inline]
    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs,())}
}

impl<T: Sized,const D1: usize,const D2: usize> HasOutput for Replace2DHeapArray<T,D1,D2> {
    type OutputBool = N;
    type Output = ();

    #[inline]
    unsafe fn output(&mut self) -> Self::Output {}

    #[inline]
    unsafe fn drop_output(&mut self) {} // dropped through reuse buf instead
}

impl<T,const D1: usize,const D2: usize> Has2DReuseBuf for Replace2DHeapArray<T,D1,D2> {
    type FstHandleBool = Y;
    type SndHandleBool = N;
    type BoundHandlesBool = N;
    type FstOwnedBufferBool = Y;
    type SndOwnedBufferBool = N;
    type IsFstBufferTransposed = N;
    type IsSndBufferTransposed = N;
    type AreBoundBuffersTransposed = N;
    type FstOwnedBuffer = Box<MathMatrix<T,D1,D2>>;
    type SndOwnedBuffer = ();
    type FstType = T;
    type SndType = ();
    type BoundTypes = ();

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) {
        std::ptr::write(self.0.get_unchecked_mut(col_index).get_unchecked_mut(row_index), val)
    }
    #[inline] unsafe fn assign_2nd_buf(&mut self, _: usize, _: usize, _: Self::SndType) {}
    #[inline] unsafe fn assign_bound_bufs(&mut self, _: usize, _: usize, _: Self::BoundTypes) {}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
        //  safety, equivilent types in order:
        //      ManuallyDrop<Box<[[T; D1]; D2]>>
        //      Box<[[T; D1]; D2]>
        //      Box<ManuallyDrop<[[T; D1]; D2]>>
        //      Box<Owned2DArray<T,D1,D2>>
        //      Box<MatrixExpr<Owned2DArray<T,D1,D2>,D1,D2>>
        //      Box<MathMatrix<T,D1,D2>>
        std::mem::transmute_copy::<ManuallyDrop<Box<[[T; D1]; D2]>>, Box<MathMatrix<T,D1,D2>>>(&self.0)
    }
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize) {
        std::ptr::drop_in_place(self.0.get_unchecked_mut(col_index).get_unchecked_mut(row_index))
    }
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, _: usize, _: usize) {}
}


pub struct MatGenerator<F: FnMut() -> O,O>(pub(crate) F);

unsafe impl<F: FnMut() -> O,O> Get2D for MatGenerator<F,O> {
    type GetBool = Y;
    type IsRepeatable = N;
    type AreInputsTransposed = N;
    type Inputs = ();
    type Item = O;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, _: usize, _: usize) -> Self::Inputs {}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize, _: usize) {}
    #[inline] fn process(&mut self, _: Self::Inputs) -> (Self::Item, Self::BoundItems) {((self.0)(),())}
}

impl<F: FnMut() -> O,O> HasOutput for MatGenerator<F,O> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

impl<F: FnMut() -> O,O> Has2DReuseBuf for MatGenerator<F,O> {
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


pub struct MatAttach2DBuf<'a,M: MatrixLike<FstHandleBool = N>,T,const D1: usize,const D2: usize>{pub(crate) mat: M,pub(crate) buf: &'a mut [[T; D1]; D2]}

unsafe impl<'a,M: MatrixLike<FstHandleBool = N>,T,const D1: usize,const D2: usize> Get2D for MatAttach2DBuf<'a,M,T,D1,D2> {
    type GetBool = M::GetBool;
    type IsRepeatable = M::IsRepeatable;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {self.mat.get_inputs(col_index,row_index)}
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {self.mat.drop_inputs(col_index,row_index)}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.mat.process(inputs)}
}

impl<'a,M: MatrixLike<FstHandleBool = N>,T,const D1: usize,const D2: usize> HasOutput for MatAttach2DBuf<'a,M,T,D1,D2> {
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output {self.mat.output()}
    #[inline] unsafe fn drop_output(&mut self) {self.mat.drop_output()}
}

impl<'b,M: MatrixLike<FstHandleBool = N>,T,const D1: usize,const D2: usize> Has2DReuseBuf for MatAttach2DBuf<'b,M,T,D1,D2> {
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

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) {*self.buf.get_unchecked_mut(col_index).get_unchecked_mut(row_index) = val}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) {self.mat.assign_2nd_buf(col_index,row_index,val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) {self.mat.assign_bound_bufs(col_index,row_index,val)}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.mat.get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) {self.mat.drop_2nd_buf_index(col_index,row_index)}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) {self.mat.drop_bound_bufs_index(col_index,row_index)}
}


pub struct MatCreate2DBuf<M: MatrixLike<FstHandleBool = N>,T,const D1: usize,const D2: usize>{pub(crate) mat: M,pub(crate) buf: [[std::mem::MaybeUninit<T>; D1]; D2]}

unsafe impl<M: MatrixLike<FstHandleBool = N>,T,const D1: usize,const D2: usize> Get2D for MatCreate2DBuf<M,T,D1,D2> {
    type GetBool = M::GetBool;
    type IsRepeatable = M::IsRepeatable;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {self.mat.get_inputs(col_index,row_index)}
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {self.mat.drop_inputs(col_index,row_index)}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.mat.process(inputs)}
}

impl<M: MatrixLike<FstHandleBool = N>,T,const D1: usize,const D2: usize> HasOutput for MatCreate2DBuf<M,T,D1,D2> {
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output {self.mat.output()}
    #[inline] unsafe fn drop_output(&mut self) {self.mat.drop_output()}
}

impl<M: MatrixLike<FstHandleBool = N>,T,const D1: usize,const D2: usize> Has2DReuseBuf for MatCreate2DBuf<M,T,D1,D2> {
    type FstHandleBool = Y;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = Y;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type IsFstBufferTransposed = N;
    type IsSndBufferTransposed = M::IsSndBufferTransposed;
    type AreBoundBuffersTransposed = M::AreBoundBuffersTransposed;
    type FstOwnedBuffer = MathMatrix<T,D1,D2>;
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = T;
    type SndType = M::SndType;
    type BoundTypes = M::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) {std::ptr::write(self.buf.get_unchecked_mut(col_index).get_unchecked_mut(row_index),std::mem::MaybeUninit::new(val))}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) {self.mat.assign_2nd_buf(col_index,row_index,val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) {self.mat.assign_bound_bufs(col_index,row_index,val)}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
        MatrixExpr(Owned2DArray(std::mem::transmute_copy::<[[std::mem::MaybeUninit<T>; D1]; D2],ManuallyDrop<[[T; D1]; D2]>>(&self.buf)))
    }
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.mat.get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) {self.mat.drop_2nd_buf_index(col_index,row_index)}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) {self.mat.drop_bound_bufs_index(col_index,row_index)}
}


pub struct MatCreate2DHeapBuf<M: MatrixLike<FstHandleBool = N>,T,const D1: usize,const D2: usize>{pub(crate) mat: M,pub(crate) buf: ManuallyDrop<Box<[[std::mem::MaybeUninit<T>; D1]; D2]>>}

unsafe impl<M: MatrixLike<FstHandleBool = N>,T,const D1: usize,const D2: usize> Get2D for MatCreate2DHeapBuf<M,T,D1,D2> {
    type GetBool = M::GetBool;
    type IsRepeatable = M::IsRepeatable;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {self.mat.get_inputs(col_index,row_index)}
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {self.mat.drop_inputs(col_index,row_index)}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.mat.process(inputs)}
}

impl<M: MatrixLike<FstHandleBool = N>,T,const D1: usize,const D2: usize> HasOutput for MatCreate2DHeapBuf<M,T,D1,D2> {
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output {self.mat.output()}
    #[inline] unsafe fn drop_output(&mut self) {self.mat.drop_output()}
}

impl<M: MatrixLike<FstHandleBool = N>,T,const D1: usize,const D2: usize> Has2DReuseBuf for MatCreate2DHeapBuf<M,T,D1,D2> {
    type FstHandleBool = Y;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = Y;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type IsFstBufferTransposed = N;
    type IsSndBufferTransposed = M::IsSndBufferTransposed;
    type AreBoundBuffersTransposed = M::AreBoundBuffersTransposed;
    type FstOwnedBuffer = MathMatrix<T,D1,D2>;
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = T;
    type SndType = M::SndType;
    type BoundTypes = M::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) {std::ptr::write(self.buf.get_unchecked_mut(col_index).get_unchecked_mut(row_index),std::mem::MaybeUninit::new(val))}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) {self.mat.assign_2nd_buf(col_index,row_index,val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) {self.mat.assign_bound_bufs(col_index,row_index,val)}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
        MatrixExpr(Owned2DArray(std::mem::transmute_copy::<[[std::mem::MaybeUninit<T>; D1]; D2],ManuallyDrop<[[T; D1]; D2]>>(&self.buf)))
    }
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.mat.get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) {self.mat.drop_2nd_buf_index(col_index,row_index)}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) {self.mat.drop_bound_bufs_index(col_index,row_index)}
}


pub struct MatMaybeCreate2DBuf<M: MatrixLike,T,const D1: usize,const D2: usize>  where <M::FstHandleBool as TyBool>::Neg: Filter {pub(crate) mat: M,pub(crate) buf: [[std::mem::MaybeUninit<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>; D1]; D2]}

unsafe impl<M: MatrixLike,T,const D1: usize,const D2: usize> Get2D for MatMaybeCreate2DBuf<M,T,D1,D2> where <M::FstHandleBool as TyBool>::Neg: Filter {
    type GetBool = M::GetBool;
    type IsRepeatable = M::IsRepeatable;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {self.mat.get_inputs(col_index,row_index)}
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {self.mat.drop_inputs(col_index,row_index)}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.mat.process(inputs)}
}

impl<M: MatrixLike,T,const D1: usize,const D2: usize> HasOutput for MatMaybeCreate2DBuf<M,T,D1,D2> where <M::FstHandleBool as TyBool>::Neg: Filter {
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output {self.mat.output()}
    #[inline] unsafe fn drop_output(&mut self) {self.mat.drop_output()}
}

impl<M: MatrixLike,T,const D1: usize,const D2: usize> Has2DReuseBuf for MatMaybeCreate2DBuf<M,T,D1,D2> 
where
    <M::FstHandleBool as TyBool>::Neg: Filter,
    (M::FstHandleBool,<M::FstHandleBool as TyBool>::Neg): SelectPair,
    (M::FstOwnedBufferBool,<M::FstHandleBool as TyBool>::Neg): TyBoolPair
{
    type FstHandleBool = Y;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = <(M::FstOwnedBufferBool,<M::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type IsFstBufferTransposed = M::IsFstBufferTransposed; //only works cause default when FstHandleBool == N is N
    type IsSndBufferTransposed = M::IsSndBufferTransposed;
    type AreBoundBuffersTransposed = M::AreBoundBuffersTransposed;
    type FstOwnedBuffer = <(M::FstHandleBool,<M::FstHandleBool as TyBool>::Neg) as SelectPair>::Selected<M::FstOwnedBuffer, MathMatrix<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>,D1,D2>>;
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = <(M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg) as SelectPair>::Selected<M::FstType,<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>;
    type SndType = M::SndType;
    type BoundTypes = M::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) {
        let (init_val,attached_val) = <(M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg) as SelectPair>::deselect(val);
        self.mat.assign_1st_buf(col_index,row_index,init_val);
        std::ptr::write(self.buf.get_unchecked_mut(col_index).get_unchecked_mut(row_index),std::mem::MaybeUninit::new(attached_val));
    }
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) {self.mat.assign_2nd_buf(col_index,row_index,val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) {self.mat.assign_bound_bufs(col_index,row_index,val)}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
        <(M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg) as SelectPair>::select(
            self.mat.get_1st_buffer(),
            MatrixExpr(Owned2DArray(std::mem::transmute_copy::<[[std::mem::MaybeUninit<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>; D1]; D2],ManuallyDrop<[[<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>; D1]; D2]>>(&self.buf)))
        )
    }
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.mat.get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) {self.mat.drop_2nd_buf_index(col_index,row_index)}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) {self.mat.drop_bound_bufs_index(col_index,row_index)}
}


pub struct MatMaybeCreate2DHeapBuf<M: MatrixLike,T,const D1: usize,const D2: usize>  where <M::FstHandleBool as TyBool>::Neg: Filter {pub(crate) mat: M,pub(crate) buf: ManuallyDrop<Box<[[std::mem::MaybeUninit<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>; D1]; D2]>>}

unsafe impl<M: MatrixLike,T,const D1: usize,const D2: usize> Get2D for MatMaybeCreate2DHeapBuf<M,T,D1,D2> where <M::FstHandleBool as TyBool>::Neg: Filter {
    type GetBool = M::GetBool;
    type IsRepeatable = M::IsRepeatable;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {self.mat.get_inputs(col_index,row_index)}
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {self.mat.drop_inputs(col_index,row_index)}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.mat.process(inputs)}
}

impl<M: MatrixLike,T,const D1: usize,const D2: usize> HasOutput for MatMaybeCreate2DHeapBuf<M,T,D1,D2> where <M::FstHandleBool as TyBool>::Neg: Filter {
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output {self.mat.output()}
    #[inline] unsafe fn drop_output(&mut self) {self.mat.drop_output()}
}

impl<M: MatrixLike,T,const D1: usize,const D2: usize> Has2DReuseBuf for MatMaybeCreate2DHeapBuf<M,T,D1,D2> 
where
    <M::FstHandleBool as TyBool>::Neg: Filter,
    (M::FstHandleBool,<M::FstHandleBool as TyBool>::Neg): SelectPair,
    (M::FstOwnedBufferBool,<M::FstHandleBool as TyBool>::Neg): TyBoolPair
{
    type FstHandleBool = Y;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = <(M::FstOwnedBufferBool,<M::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type IsFstBufferTransposed = M::IsFstBufferTransposed; //only works cause default when FstHandleBool == N is N
    type IsSndBufferTransposed = M::IsSndBufferTransposed;
    type AreBoundBuffersTransposed = M::AreBoundBuffersTransposed;
    type FstOwnedBuffer = <(M::FstHandleBool,<M::FstHandleBool as TyBool>::Neg) as SelectPair>::Selected<M::FstOwnedBuffer, Box<MathMatrix<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>,D1,D2>>>;
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = <(M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg) as SelectPair>::Selected<M::FstType,<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>;
    type SndType = M::SndType;
    type BoundTypes = M::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) {
        let (init_val,attached_val) = <(M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg) as SelectPair>::deselect(val);
        self.mat.assign_1st_buf(col_index,row_index,init_val);
        std::ptr::write(self.buf.get_unchecked_mut(col_index).get_unchecked_mut(row_index),std::mem::MaybeUninit::new(attached_val));
    }
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) {self.mat.assign_2nd_buf(col_index,row_index,val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) {self.mat.assign_bound_bufs(col_index,row_index,val)}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
        <(M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg) as SelectPair>::select(
            self.mat.get_1st_buffer(),
            std::mem::transmute_copy::<Box<[[std::mem::MaybeUninit<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>; D1]; D2]>,Box<MathMatrix<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>,D1,D2>>>(&self.buf)
        )
    }
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.mat.get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) {self.mat.drop_2nd_buf_index(col_index,row_index)}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) {self.mat.drop_bound_bufs_index(col_index,row_index)}
}


pub struct MatBind<M: MatrixLike<FstHandleBool = Y>>{pub(crate) mat: M}

unsafe impl<M: MatrixLike<FstHandleBool = Y>> Get2D for MatBind<M> where (M::BoundHandlesBool,Y): FilterPair {
    type GetBool = N;
    type IsRepeatable = N;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = ();
    type BoundItems = <(M::BoundHandlesBool,Y) as FilterPair>::Filtered<M::BoundItems,M::Item>;

    #[inline]
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {
        self.mat.get_inputs(col_index, row_index)
    }

    #[inline]
    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {
        self.mat.drop_inputs(col_index, row_index)
    }

    #[inline]
    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
        let (item,bound) = self.mat.process(inputs);
        ((),<(M::BoundHandlesBool,Y) as FilterPair>::filter(bound,item))
    }
}

impl<M: MatrixLike<FstHandleBool = Y>> HasOutput for MatBind<M> where (M::OutputBool,M::FstOwnedBufferBool): FilterPair {
    type OutputBool = <(M::OutputBool,M::FstOwnedBufferBool) as TyBoolPair>::Or;
    type Output = <(M::OutputBool,M::FstOwnedBufferBool) as FilterPair>::Filtered<M::Output,M::FstOwnedBuffer>;

    #[inline]
    unsafe fn output(&mut self) -> Self::Output {
        <(M::OutputBool,M::FstOwnedBufferBool) as FilterPair>::filter(self.mat.output(),self.mat.get_1st_buffer())
    }

    #[inline]
    unsafe fn drop_output(&mut self) {
        self.mat.drop_output();
    }
}

impl<M: MatrixLike<FstHandleBool = Y>> Has2DReuseBuf for MatBind<M> where (M::BoundHandlesBool,Y): FilterPair, (M::IsFstBufferTransposed,M::AreBoundBuffersTransposed): TyBoolPair {
    type FstHandleBool = N;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = <(M::BoundHandlesBool,Y) as TyBoolPair>::Or;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type IsFstBufferTransposed = N;
    type IsSndBufferTransposed = M::IsSndBufferTransposed;
    type AreBoundBuffersTransposed = <(M::IsFstBufferTransposed,M::AreBoundBuffersTransposed) as TyBoolPair>::And;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = ();
    type SndType = M::SndType;
    type BoundTypes = <(M::BoundHandlesBool,Y) as FilterPair>::Filtered<M::BoundTypes,M::FstType>;

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) {self.mat.assign_2nd_buf(col_index, row_index, val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) {
        let (bounded_vals,new_bound_val) = <(M::BoundHandlesBool,Y) as FilterPair>::defilter(val);
        self.mat.assign_bound_bufs(col_index, row_index, bounded_vals);
        self.mat.assign_1st_buf(col_index, row_index, new_bound_val);
    }
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.mat.get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) {self.mat.drop_2nd_buf_index(col_index, row_index)}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) {
        self.mat.drop_1st_buf_index(col_index, row_index);
        self.mat.drop_bound_bufs_index(col_index, row_index);
    }
}


pub struct MatMapBind<M: MatrixLike<FstHandleBool = Y>,F: FnMut(M::Item) -> (I,B),I,B>{pub(crate) mat: M,pub(crate) f: F}

unsafe impl<M: MatrixLike<FstHandleBool = Y>,F: FnMut(M::Item) -> (I,B),I,B> Get2D for MatMapBind<M,F,I,B> where (M::BoundHandlesBool,Y): FilterPair {
    type GetBool = Y;
    type IsRepeatable = N;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = I;
    type BoundItems = <(M::BoundHandlesBool,Y) as FilterPair>::Filtered<M::BoundItems,B>;

    #[inline]
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {
        self.mat.get_inputs(col_index, row_index)
    }

    #[inline]
    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {
        self.mat.drop_inputs(col_index, row_index)
    }

    #[inline]
    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
        let (item,bound) = self.mat.process(inputs);
        let (processed_item,processed_bound) = (self.f)(item);
        (processed_item,<(M::BoundHandlesBool,Y) as FilterPair>::filter(bound,processed_bound))
    }
}

impl<M: MatrixLike<FstHandleBool = Y>,F: FnMut(M::Item) -> (I,B),I,B> HasOutput for MatMapBind<M,F,I,B> where (M::OutputBool,M::FstOwnedBufferBool): FilterPair {
    type OutputBool = <(M::OutputBool,M::FstOwnedBufferBool) as TyBoolPair>::Or;
    type Output = <(M::OutputBool,M::FstOwnedBufferBool) as FilterPair>::Filtered<M::Output,M::FstOwnedBuffer>;

    #[inline]
    unsafe fn output(&mut self) -> Self::Output {
        <(M::OutputBool,M::FstOwnedBufferBool) as FilterPair>::filter(self.mat.output(),self.mat.get_1st_buffer())
    }

    #[inline]
    unsafe fn drop_output(&mut self) {
        self.mat.drop_output();
    }
}

impl<M: MatrixLike<FstHandleBool = Y>,F: FnMut(M::Item) -> (I,B),I,B> Has2DReuseBuf for MatMapBind<M,F,I,B> where (M::BoundHandlesBool,Y): FilterPair, (M::IsFstBufferTransposed,M::AreBoundBuffersTransposed): TyBoolPair {
    type FstHandleBool = N;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = <(M::BoundHandlesBool,Y) as TyBoolPair>::Or;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type IsFstBufferTransposed = N;
    type IsSndBufferTransposed = M::IsSndBufferTransposed;
    type AreBoundBuffersTransposed = <(M::IsFstBufferTransposed,M::AreBoundBuffersTransposed) as TyBoolPair>::And;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = ();
    type SndType = M::SndType;
    type BoundTypes = <(M::BoundHandlesBool,Y) as FilterPair>::Filtered<M::BoundTypes,M::FstType>;

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) {self.mat.assign_2nd_buf(col_index, row_index, val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) {
        let (bounded_vals,new_bound_val) = <(M::BoundHandlesBool,Y) as FilterPair>::defilter(val);
        self.mat.assign_bound_bufs(col_index, row_index, bounded_vals);
        self.mat.assign_1st_buf(col_index, row_index, new_bound_val);
    }
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.mat.get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) {self.mat.drop_2nd_buf_index(col_index, row_index)}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) {
        self.mat.drop_1st_buf_index(col_index, row_index);
        self.mat.drop_bound_bufs_index(col_index, row_index);
    }
}


pub struct MatBufSwap<M: MatrixLike>{pub(crate) mat: M} 

unsafe impl<M: MatrixLike> Get2D for MatBufSwap<M> {
    type GetBool = M::GetBool;
    type IsRepeatable = M::IsRepeatable;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {self.mat.get_inputs(col_index,row_index)}
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {self.mat.drop_inputs(col_index,row_index)}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.mat.process(inputs)}
}

impl<M: MatrixLike> HasOutput for MatBufSwap<M> {
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output {self.mat.output()}
    #[inline] unsafe fn drop_output(&mut self) {self.mat.drop_output()}
}

impl<M: MatrixLike> Has2DReuseBuf for MatBufSwap<M> {
    type FstHandleBool = M::SndHandleBool;
    type SndHandleBool = M::FstHandleBool;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = M::SndOwnedBufferBool;
    type SndOwnedBufferBool = M::FstOwnedBufferBool;
    type IsFstBufferTransposed = M::IsSndBufferTransposed;
    type IsSndBufferTransposed = M::IsFstBufferTransposed;
    type AreBoundBuffersTransposed = M::AreBoundBuffersTransposed;
    type FstOwnedBuffer = M::SndOwnedBuffer;
    type SndOwnedBuffer = M::FstOwnedBuffer;
    type FstType = M::SndType;
    type SndType = M::FstType;
    type BoundTypes = M::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) {self.mat.assign_2nd_buf(col_index,row_index,val)}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) {self.mat.assign_1st_buf(col_index,row_index,val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) {self.mat.assign_bound_bufs(col_index,row_index,val)}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {self.mat.get_2nd_buffer()}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.mat.get_1st_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize) {self.mat.drop_2nd_buf_index(col_index,row_index)}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) {self.mat.drop_1st_buf_index(col_index,row_index)}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) {self.mat.drop_bound_bufs_index(col_index,row_index)}
}

pub struct MatColOffset<M: MatrixLike>{pub(crate) mat: M, pub(crate) offset: usize, pub(crate) num_columns: usize}

impl<M: MatrixLike> MatColOffset<M> {
    #[inline]
    fn offset_index(&self,index: usize) -> usize {
        let mut offset_index = index + self.offset;
        if offset_index >= index { 
            offset_index %= self.num_columns;
        } else { //index overflowed, LLVM should be able to elid this most of the time (hopefully)
            //if the index overflowed, (usize::MAX) + 1 was subtracted from it, add (usize::MAX)+1 mod self.num_columns to recover
            offset_index %= self.num_columns;
            offset_index += ((usize::MAX % self.num_columns) + 1) % self.num_columns; // 2 modulos to prevent overflow
            offset_index %= self.num_columns;
        }
        offset_index
    }
}

unsafe impl<M: MatrixLike> Get2D for MatColOffset<M> {
    type GetBool = M::GetBool;
    type IsRepeatable = N;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {self.mat.get_inputs(self.offset_index(col_index),row_index)}
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {self.mat.drop_inputs(self.offset_index(col_index),row_index)}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.mat.process(inputs)}
}

impl<M: MatrixLike> HasOutput for MatColOffset<M> {
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output {self.mat.output()}
    #[inline] unsafe fn drop_output(&mut self) {self.mat.drop_output()}
}

impl<M: MatrixLike> Has2DReuseBuf for MatColOffset<M> {
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

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) {self.mat.assign_1st_buf(self.offset_index(col_index),row_index,val)}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) {self.mat.assign_2nd_buf(self.offset_index(col_index),row_index,val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) {self.mat.assign_bound_bufs(self.offset_index(col_index),row_index,val)}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {self.mat.get_1st_buffer()}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.mat.get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize) {self.mat.drop_1st_buf_index(self.offset_index(col_index),row_index)}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) {self.mat.drop_2nd_buf_index(self.offset_index(col_index),row_index)}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) {self.mat.drop_bound_bufs_index(self.offset_index(col_index),row_index)}
}

pub struct MatRowOffset<M: MatrixLike>{pub(crate) mat: M, pub(crate) offset: usize, pub(crate) num_rows: usize}

impl<M: MatrixLike> MatRowOffset<M> {
    #[inline]
    fn offset_index(&self,index: usize) -> usize {
        let mut offset_index = index + self.offset;
        if offset_index >= index { 
            offset_index %= self.num_rows;
        } else { //index overflowed, LLVM should be able to elid this most of the time
            //if the index overflowed, (usize::MAX) + 1 was subtracted from it, add (usize::MAX)+1 mod self.num_rows to recover
            offset_index %= self.num_rows;
            offset_index += ((usize::MAX % self.num_rows) + 1) % self.num_rows; // 2 modulos to prevent overflow
            offset_index %= self.num_rows;
        }
        offset_index
    }
}

unsafe impl<M: MatrixLike> Get2D for MatRowOffset<M> {
    type GetBool = M::GetBool;
    type IsRepeatable = N;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {self.mat.get_inputs(col_index,self.offset_index(row_index))}
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) {self.mat.drop_inputs(col_index,self.offset_index(row_index))}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.mat.process(inputs)}
}

impl<M: MatrixLike> HasOutput for MatRowOffset<M> {
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output {self.mat.output()}
    #[inline] unsafe fn drop_output(&mut self) {self.mat.drop_output()}
}

impl<M: MatrixLike> Has2DReuseBuf for MatRowOffset<M> {
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

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) {self.mat.assign_1st_buf(col_index,self.offset_index(row_index),val)}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) {self.mat.assign_2nd_buf(col_index,self.offset_index(row_index),val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) {self.mat.assign_bound_bufs(col_index,self.offset_index(row_index),val)}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {self.mat.get_1st_buffer()}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.mat.get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize) {self.mat.drop_1st_buf_index(col_index,self.offset_index(row_index))}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) {self.mat.drop_2nd_buf_index(col_index,self.offset_index(row_index))}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) {self.mat.drop_bound_bufs_index(col_index,self.offset_index(row_index))}
}



pub struct MatrixColumn<M: MatrixLike>{pub(crate) mat: *mut M, pub(crate) column_num: usize}

unsafe impl<M: MatrixLike> Get for MatrixColumn<M> {
    type GetBool = M::GetBool;
    type IsRepeatable = M::GetBool;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline]
    unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {unsafe { (*self.mat).get_inputs(self.column_num, index)} }

    #[inline]
    unsafe fn drop_inputs(&mut self, index: usize) {unsafe { (*self.mat).drop_inputs(self.column_num, index)} }
    
    #[inline]
    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {unsafe { (*self.mat).process(inputs)}}
}

impl<M: MatrixLike> HasOutput for MatrixColumn<M> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

impl<M: MatrixLike> HasReuseBuf for MatrixColumn<M> {
    type FstHandleBool = N;
    type SndHandleBool = N;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = N;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = ();
    type FstType = ();
    type SndType = ();
    type BoundTypes = M::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self,_: usize,_: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self,_: usize,_: Self::SndType) {}
    #[inline] unsafe fn assign_bound_bufs(&mut self,index: usize,val: Self::BoundTypes) {(*self.mat).assign_bound_bufs(self.column_num, index,val)}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {}
    #[inline] unsafe fn drop_1st_buf_index(&mut self,_: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self,_: usize) {}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self,index: usize) {(*self.mat).drop_bound_bufs_index(self.column_num, index)}
}


pub struct MatColVectorExprs<M: MatrixLike>{pub(crate) mat: M}

unsafe impl<M: MatrixLike> Get for MatColVectorExprs<M> {
    type GetBool = M::GetBool;
    type IsRepeatable = M::IsRepeatable;
    type Inputs = usize;
    type Item = MatrixColumn<M>;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {index}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(MatrixColumn{mat: &mut self.mat as *mut M, column_num: inputs},())}
}

impl<M: MatrixLike> HasOutput for MatColVectorExprs<M> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {self.mat.drop_output()}
}

impl<M: MatrixLike> HasReuseBuf for MatColVectorExprs<M> {
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

unsafe impl<M: MatrixLike> VectorizedMatrix for MatColVectorExprs<M> {}

pub struct MatrixRow<M: MatrixLike>{pub(crate) mat: *mut M, pub(crate) row_num: usize}

unsafe impl<M: MatrixLike> Get for MatrixRow<M> {
    type GetBool = M::GetBool;
    type IsRepeatable = M::GetBool;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline]
    unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {unsafe { (*self.mat).get_inputs(index, self.row_num)} }

    #[inline]
    unsafe fn drop_inputs(&mut self, index: usize) {unsafe { (*self.mat).drop_inputs(index, self.row_num)} }
    
    #[inline]
    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {unsafe { (*self.mat).process(inputs)}}
}

impl<M: MatrixLike> HasOutput for MatrixRow<M> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

impl<M: MatrixLike> HasReuseBuf for MatrixRow<M> {
    type FstHandleBool = N;
    type SndHandleBool = N;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = N;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = ();
    type FstType = ();
    type SndType = ();
    type BoundTypes = M::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self,_: usize,_: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self,_: usize,_: Self::SndType) {}
    #[inline] unsafe fn assign_bound_bufs(&mut self,index: usize,val: Self::BoundTypes) {(*self.mat).assign_bound_bufs(index, self.row_num,val)}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {}
    #[inline] unsafe fn drop_1st_buf_index(&mut self,_: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self,_: usize) {}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self,index: usize) {(*self.mat).drop_bound_bufs_index(index, self.row_num)}
}


pub struct MatRowVectorExprs<M: MatrixLike>{pub(crate) mat: M}

unsafe impl<M: MatrixLike> Get for MatRowVectorExprs<M> {
    type GetBool = M::GetBool;
    type IsRepeatable = M::IsRepeatable;
    type Inputs = usize;
    type Item = MatrixRow<M>;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {index}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(MatrixRow{mat: &mut self.mat as *mut M, row_num: inputs},())}
}

impl<M: MatrixLike> HasOutput for MatRowVectorExprs<M> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {self.mat.drop_output()}
}

impl<M: MatrixLike> HasReuseBuf for MatRowVectorExprs<M> {
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

unsafe impl<M: MatrixLike> VectorizedMatrix for MatRowVectorExprs<M> {}


pub struct FullMatMul<M1: MatrixLike<IsRepeatable = Y>, M2: MatrixLike<IsRepeatable = Y>>{pub(crate) l_mat: M1, pub(crate) r_mat: M2, pub(crate) shared_size: usize}

unsafe impl<M1: MatrixLike<IsRepeatable = Y>, M2: MatrixLike<IsRepeatable = Y>> Get2D for FullMatMul<M1,M2> where 
    M1::Item: Mul<M2::Item>,
    <M1::Item as Mul<M2::Item>>::Output: AddAssign,
    (M1::BoundHandlesBool,M2::BoundHandlesBool): FilterPair,
    (M1::AreInputsTransposed,M2::AreInputsTransposed): TyBoolPair,
{
    type GetBool = Y;
    type IsRepeatable = N;
    type AreInputsTransposed = <(M1::AreInputsTransposed, M2::AreInputsTransposed) as TyBoolPair>::Or;
    type Inputs = (usize,usize);
    type Item = <M1::Item as Mul<M2::Item>>::Output;
    type BoundItems = <(M1::BoundHandlesBool,M2::BoundHandlesBool) as FilterPair>::Filtered<M1::BoundItems, M2::BoundItems>;

    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {(col_index,row_index)}
    unsafe fn drop_inputs(&mut self, _: usize, _: usize) {}
    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
        unsafe {
            let mut result = self.l_mat.get(0, inputs.1).0 * self.r_mat.get(inputs.0, 0).0;
            for i in 1..self.shared_size {
                result += self.l_mat.get(i, inputs.1).0 * self.r_mat.get(inputs.0, i).0;
            }
            let bound = <(M1::BoundHandlesBool,M2::BoundHandlesBool) as FilterPair>::filter(
                self.l_mat.get(inputs.0,inputs.1).1, 
                self.r_mat.get(inputs.0,inputs.1).1
            );
            (result,bound)
        }
    }
}

impl<M1: MatrixLike<IsRepeatable = Y>,M2: MatrixLike<IsRepeatable = Y>> HasOutput for FullMatMul<M1,M2> where (M1::OutputBool, M2::OutputBool): FilterPair {
    type OutputBool = <(M1::OutputBool,M2::OutputBool) as TyBoolPair>::Or;
    type Output = <(M1::OutputBool,M2::OutputBool) as FilterPair>::Filtered<M1::Output,M2::Output>;

    unsafe fn output(&mut self) -> Self::Output {
        <(M1::OutputBool,M2::OutputBool) as FilterPair>::filter(
            self.l_mat.output(),
            self.r_mat.output()
        )
    }
    unsafe fn drop_output(&mut self) {
        self.l_mat.drop_output();
        self.r_mat.drop_output();
    }
} 

impl<M1: MatrixLike<IsRepeatable = Y>,M2: MatrixLike<IsRepeatable = Y>> Has2DReuseBuf for FullMatMul<M1,M2>
where 
    (M1::FstOwnedBufferBool, M2::FstOwnedBufferBool): SelectPair,
    (M1::SndOwnedBufferBool, M2::SndOwnedBufferBool): SelectPair,
    (M1::FstHandleBool, M2::FstHandleBool): SelectPair,
    (M1::SndHandleBool, M2::SndHandleBool): SelectPair,
    (M1::BoundHandlesBool, M2::BoundHandlesBool): FilterPair,
    (M1::IsFstBufferTransposed, M2::IsFstBufferTransposed): TyBoolPair,
    (M1::IsSndBufferTransposed, M2::IsSndBufferTransposed): TyBoolPair,
    (M1::AreBoundBuffersTransposed, M2::AreBoundBuffersTransposed): TyBoolPair
{
    type FstHandleBool = <(M1::FstHandleBool, M2::FstHandleBool) as TyBoolPair>::Xor;
    type SndHandleBool = <(M1::SndHandleBool, M2::SndHandleBool) as TyBoolPair>::Xor;
    type BoundHandlesBool = <(M1::BoundHandlesBool, M2::BoundHandlesBool) as TyBoolPair>::Or;
    type FstOwnedBufferBool = <(M1::FstOwnedBufferBool, M2::FstOwnedBufferBool) as TyBoolPair>::Xor; 
    type SndOwnedBufferBool = <(M1::SndOwnedBufferBool, M2::SndOwnedBufferBool) as TyBoolPair>::Xor; 
    type IsFstBufferTransposed = <(M1::IsFstBufferTransposed, M2::IsFstBufferTransposed) as TyBoolPair>::Xor;
    type IsSndBufferTransposed = <(M1::IsSndBufferTransposed, M2::IsSndBufferTransposed) as TyBoolPair>::Xor;
    type AreBoundBuffersTransposed = <(M1::AreBoundBuffersTransposed, M2::AreBoundBuffersTransposed) as TyBoolPair>::Xor;
    type FstOwnedBuffer = <(M1::FstOwnedBufferBool, M2::FstOwnedBufferBool) as SelectPair>::Selected<M1::FstOwnedBuffer,M2::FstOwnedBuffer>;
    type SndOwnedBuffer = <(M1::SndOwnedBufferBool, M2::SndOwnedBufferBool) as SelectPair>::Selected<M1::SndOwnedBuffer,M2::SndOwnedBuffer>;
    type FstType = <(M1::FstHandleBool, M2::FstHandleBool) as SelectPair>::Selected<M1::FstType,M2::FstType>;
    type SndType = <(M1::SndHandleBool, M2::SndHandleBool) as SelectPair>::Selected<M1::SndType,M2::SndType>;
    type BoundTypes = <(M1::BoundHandlesBool, M2::BoundHandlesBool) as FilterPair>::Filtered<M1::BoundTypes,M2::BoundTypes>;

    #[inline] unsafe fn assign_1st_buf(&mut self,col_index: usize,row_index: usize,val: Self::FstType) {
        let (l_val,r_val) = <(M1::FstHandleBool, M2::FstHandleBool) as SelectPair>::deselect(val);
        self.l_mat.assign_1st_buf(col_index,row_index,l_val);
        self.r_mat.assign_1st_buf(col_index,row_index,r_val);
    }
    #[inline] unsafe fn assign_2nd_buf(&mut self,col_index: usize,row_index: usize,val: Self::SndType) {
        let (l_val,r_val) = <(M1::SndHandleBool, M2::SndHandleBool) as SelectPair>::deselect(val);
        self.l_mat.assign_2nd_buf(col_index,row_index,l_val);
        self.r_mat.assign_2nd_buf(col_index,row_index,r_val);
    }
    #[inline] unsafe fn assign_bound_bufs(&mut self,col_index: usize,row_index: usize,val: Self::BoundTypes) {
        let (l_val,r_val) = <(M1::BoundHandlesBool, M2::BoundHandlesBool) as FilterPair>::defilter(val);
        self.l_mat.assign_bound_bufs(col_index,row_index,l_val);
        self.r_mat.assign_bound_bufs(col_index,row_index,r_val);
    }
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
        <(M1::FstOwnedBufferBool, M2::FstOwnedBufferBool) as SelectPair>::select(self.l_mat.get_1st_buffer(),self.r_mat.get_1st_buffer())
    }
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {
        <(M1::SndOwnedBufferBool, M2::SndOwnedBufferBool) as SelectPair>::select(self.l_mat.get_2nd_buffer(),self.r_mat.get_2nd_buffer())
    }
    #[inline] unsafe fn drop_1st_buf_index(&mut self,col_index: usize,row_index: usize) {
        self.l_mat.drop_1st_buf_index(col_index,row_index);
        self.r_mat.drop_1st_buf_index(col_index,row_index);
    }
    #[inline] unsafe fn drop_2nd_buf_index(&mut self,col_index: usize,row_index: usize) {
        self.l_mat.drop_2nd_buf_index(col_index,row_index);
        self.r_mat.drop_2nd_buf_index(col_index,row_index);
    }
    #[inline] unsafe fn drop_bound_bufs_index(&mut self,col_index: usize,row_index: usize) {
        self.l_mat.drop_bound_bufs_index(col_index,row_index);
        self.r_mat.drop_bound_bufs_index(col_index,row_index);
    }
}



macro_rules! is_unit {
    (()) => {
        N
    };
    ($ty:ty) => {
        Y
    };
}

macro_rules! is_present {
    ($tokens:tt) => {
        Y
    };
    () => {
        N
    }
}

macro_rules! optional_type {
    () => {
        ()
    };
    ($ty:ty) => {
        $ty
    }
}

macro_rules! optional_expr {
    () => {
        ()
    };
    ($expr:expr) => {
        $expr
    }
}

macro_rules! optimized_or {
    ($ty_bool:ty, $tokens:tt) => {
        Y
    };
    ($ty_bool:ty, ) => {
        $ty_bool
    }
}


macro_rules! mat_struct {
    ( // Get2D (+ non-lazy)* -> Get2D
        $struct:ident<$($($lifetime:lifetime),+,)? {$mat_generic:ident} $(,$($generic:ident $(: $($generic_lifetime:lifetime |)? $fst_generic_bound:path $(| $generic_bound:path)*)?),+)?>{$mat:ident $(,$($field:ident: $field_ty:ty),+)?}
        $(where $($bound_ty:ty: $fst_where_bound:path $(| $where_bound:path)*),+)?;
        $(output: $outputted_field:ident: $output_ty:ty,)?
        get2D: $item:ty, |$self:ident,$(($is_mut:tt))? $input:ident| $get_expr:expr
    ) => {
        pub struct $struct<$($($lifetime),+,)? $mat_generic: MatrixLike $(,$($generic $(: $($generic_lifetime +)? $fst_generic_bound $(+ $generic_bound)*)?),+)?> $(where $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {pub(crate) $mat: $mat_generic $(,$(pub(crate) $field: $field_ty),+)?}

        unsafe impl<$($($lifetime),+,)? $mat_generic: MatrixLike $(,$($generic $(: $fst_generic_bound $(+ $generic_bound)*)?),+)?> Get2D for $struct<$($($lifetime),+,)? $mat_generic $(,$($generic),+)?> $(where $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {
            type GetBool = is_unit!($item);
            type IsRepeatable = N;
            type AreInputsTransposed = <$mat_generic as Get2D>::AreInputsTransposed;
            type Inputs = <$mat_generic as Get2D>::Inputs;
            type Item = $item;
            type BoundItems = <$mat_generic as Get2D>::BoundItems;

            #[inline]
            unsafe fn get_inputs(&mut self,col_index: usize,row_index: usize) -> Self::Inputs {self.$mat.get_inputs(col_index,row_index)}

            #[inline]
            unsafe fn drop_inputs(&mut self,col_index: usize,row_index: usize) {self.$mat.drop_inputs(col_index,row_index)}

            #[inline]
            fn process($self: &mut Self,inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
                let ($($is_mut)? $input,bound_items) = $self.$mat.process(inputs);
                ($get_expr,bound_items)
            }
        }

        impl<$($($lifetime),+,)? $mat_generic: MatrixLike $(,$($generic $(: $fst_generic_bound $(+ $generic_bound)*)?),+)?> HasOutput for $struct<$($($lifetime),+,)? $mat_generic $(,$($generic),+)?> 
        where ($mat_generic::OutputBool,is_present!($($outputted_field)?)): FilterPair $(,$($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {
            type OutputBool = optimized_or!($mat_generic::OutputBool,$($outputted_field)?);
            type Output = <($mat_generic::OutputBool,is_present!($($outputted_field)?)) as FilterPair>::Filtered<$mat_generic::Output,optional_type!($($output_ty)?)>;

            #[inline]
            unsafe fn output(&mut self) -> Self::Output {
                <($mat_generic::OutputBool,is_present!($($outputted_field)?)) as FilterPair>::filter(self.$mat.output(),optional_expr!($(self.$outputted_field.output())?))
            }

            #[inline]
            unsafe fn drop_output(&mut self) {
                self.$mat.drop_output();
                $(self.$outputted_field.output();)?
            }
        }

        impl<$($($lifetime),+,)? $mat_generic: MatrixLike $(,$($generic $(: $fst_generic_bound $(+ $generic_bound)*)?),+)?> Has2DReuseBuf for $struct<$($($lifetime),+,)? $mat_generic $(,$($generic),+)?> 
        $(where $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {
            type FstHandleBool = <$mat_generic as Has2DReuseBuf>::FstHandleBool;
            type SndHandleBool = <$mat_generic as Has2DReuseBuf>::SndHandleBool;
            type BoundHandlesBool = <$mat_generic as Has2DReuseBuf>::BoundHandlesBool;
            type FstOwnedBufferBool = <$mat_generic as Has2DReuseBuf>::FstOwnedBufferBool;
            type SndOwnedBufferBool = <$mat_generic as Has2DReuseBuf>::SndOwnedBufferBool;
            type IsFstBufferTransposed = <$mat_generic as Has2DReuseBuf>::IsFstBufferTransposed;
            type IsSndBufferTransposed = <$mat_generic as Has2DReuseBuf>::IsSndBufferTransposed;
            type AreBoundBuffersTransposed = <$mat_generic as Has2DReuseBuf>::AreBoundBuffersTransposed;
            type FstOwnedBuffer = <$mat_generic as Has2DReuseBuf>::FstOwnedBuffer;
            type SndOwnedBuffer = <$mat_generic as Has2DReuseBuf>::SndOwnedBuffer;
            type FstType = <$mat_generic as Has2DReuseBuf>::FstType;
            type SndType = <$mat_generic as Has2DReuseBuf>::SndType;
            type BoundTypes = <$mat_generic as Has2DReuseBuf>::BoundTypes;

            #[inline] unsafe fn assign_1st_buf(&mut self,col_index: usize,row_index: usize, val: Self::FstType) {self.$mat.assign_1st_buf(col_index,row_index,val)}
            #[inline] unsafe fn assign_2nd_buf(&mut self,col_index: usize,row_index: usize, val: Self::SndType) {self.$mat.assign_2nd_buf(col_index,row_index,val)}
            #[inline] unsafe fn assign_bound_bufs(&mut self,col_index: usize,row_index: usize, val: Self::BoundTypes) {self.$mat.assign_bound_bufs(col_index,row_index,val)}
            #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {self.$mat.get_1st_buffer()}
            #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.$mat.get_2nd_buffer()}
            #[inline] unsafe fn drop_1st_buf_index(&mut self,col_index: usize,row_index: usize) {self.$mat.drop_1st_buf_index(col_index,row_index)}
            #[inline] unsafe fn drop_2nd_buf_index(&mut self,col_index: usize,row_index: usize) {self.$mat.drop_2nd_buf_index(col_index,row_index)}
            #[inline] unsafe fn drop_bound_bufs_index(&mut self,col_index: usize,row_index: usize) {self.$mat.drop_bound_bufs_index(col_index,row_index)}
        }
    };
    ( // Get2D + Get2D (+ non-lazy)* -> Get2D
        $struct:ident<$($($lifetime:lifetime),+,)? {$l_mat_generic:ident,$r_mat_generic:ident} $(,$($generic:ident $(: $($generic_lifetime:lifetime |)? $fst_generic_bound:path $(| $generic_bound:path)*)?),+)?>{$l_mat:ident, $r_mat:ident $(,$($field:ident: $field_ty:ty),+)?}
        $(where $($bound_ty:ty: $fst_where_bound:path $(| $where_bound:path)*),+)?;
        $(output: $outputted_field:ident: $output_ty:ty,)?
        get2D: $item:ty, |$self:ident,$(($l_is_mut:tt))? $l_input:ident,$(($r_is_mut:tt))? $r_input:ident| $get_expr:expr
    ) => {
        pub struct $struct<$($($lifetime),+,)? $l_mat_generic: MatrixLike, $r_mat_generic: MatrixLike $(,$($generic $(: $($generic_lifetime +)? $fst_generic_bound $(+ $generic_bound)*)?),+)?> $(where $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {pub(crate) $l_mat: $l_mat_generic, pub(crate) $r_mat: $r_mat_generic $(,$(pub(crate) $field: $field_ty),+)?}

        unsafe impl<$($($lifetime),+,)? $l_mat_generic: MatrixLike, $r_mat_generic: MatrixLike $(,$($generic $(: $($generic_lifetime +)? $fst_generic_bound $(+ $generic_bound)*)?),+)?> Get2D for $struct<$($($lifetime),+,)? $l_mat_generic, $r_mat_generic $(,$($generic),+)?> where ($l_mat_generic::BoundHandlesBool, $r_mat_generic::BoundHandlesBool): FilterPair, (<$l_mat_generic as Get2D>::AreInputsTransposed,<$r_mat_generic as Get2D>::AreInputsTransposed): TyBoolPair $(, $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {
            type GetBool = is_unit!($item);
            type IsRepeatable = N;
            type AreInputsTransposed = <(<$l_mat_generic as Get2D>::AreInputsTransposed,<$r_mat_generic as Get2D>::AreInputsTransposed) as TyBoolPair>::And;
            type Inputs = ($l_mat_generic::Inputs,$r_mat_generic::Inputs);
            type Item = $item;
            type BoundItems = <($l_mat_generic::BoundHandlesBool, $r_mat_generic::BoundHandlesBool) as FilterPair>::Filtered<$l_mat_generic::BoundItems,$r_mat_generic::BoundItems>;

            #[inline]
            unsafe fn get_inputs(&mut self,col_index: usize,row_index: usize) -> Self::Inputs {(self.$l_mat.get_inputs(col_index,row_index),self.$r_mat.get_inputs(col_index,row_index))}

            #[inline]
            unsafe fn drop_inputs(&mut self,col_index: usize,row_index: usize) {
                self.$l_mat.drop_inputs(col_index,row_index);
                self.$r_mat.drop_inputs(col_index,row_index);
            }

            #[inline]
            fn process($self: &mut Self,inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
                let ($($l_is_mut)? $l_input,l_bound_items) = $self.$l_mat.process(inputs.0);
                let ($($r_is_mut)? $r_input,r_bound_items) = $self.$r_mat.process(inputs.1);
                ($get_expr,<($l_mat_generic::BoundHandlesBool, $r_mat_generic::BoundHandlesBool) as FilterPair>::filter(l_bound_items,r_bound_items))
            }
        }
    
        impl<$($($lifetime),+,)? $l_mat_generic: MatrixLike, $r_mat_generic: MatrixLike $(,$($generic $(: $($generic_lifetime +)? $fst_generic_bound $(+ $generic_bound)*)?),+)?> HasOutput for $struct<$($($lifetime),+,)? $l_mat_generic, $r_mat_generic $(,$($generic),+)?> where ($l_mat_generic::OutputBool,$r_mat_generic::OutputBool): FilterPair, (<($l_mat_generic::OutputBool,$r_mat_generic::OutputBool) as TyBoolPair>::Or,is_present!($($outputted_field)?)): FilterPair $(, $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {
            type OutputBool = optimized_or!(<($l_mat_generic::OutputBool,$r_mat_generic::OutputBool) as TyBoolPair>::Or,$($outputted_field)?);
            type Output = <(<($l_mat_generic::OutputBool,$r_mat_generic::OutputBool) as TyBoolPair>::Or,is_present!($($outputted_field)?)) as FilterPair>::Filtered<<($l_mat_generic::OutputBool,$r_mat_generic::OutputBool) as FilterPair>::Filtered<$l_mat_generic::Output,$r_mat_generic::Output>,optional_type!($($output_ty)?)>;
        
            #[inline]
            unsafe fn output(&mut self) -> Self::Output {
                <(<($l_mat_generic::OutputBool,$r_mat_generic::OutputBool) as TyBoolPair>::Or,is_present!($($outputted_field)?)) as FilterPair>::filter(
                    <($l_mat_generic::OutputBool,$r_mat_generic::OutputBool) as FilterPair>::filter(self.$l_mat.output(),self.$r_mat.output()),
                    optional_expr!($(self.$outputted_field.output())?)
                )
            }

            #[inline]
            unsafe fn drop_output(&mut self) {
                self.$l_mat.drop_output();
                self.$r_mat.drop_output();
                $(self.$outputted_field.output();)?
            }
        }

        impl<$($($lifetime),+,)? $l_mat_generic: MatrixLike, $r_mat_generic: MatrixLike $(,$($generic $(: $($generic_lifetime +)? $fst_generic_bound $(+ $generic_bound)*)?),+)?> Has2DReuseBuf for $struct<$($($lifetime),+,)? $l_mat_generic, $r_mat_generic $(,$($generic),+)?> 
        where 
            (<$l_mat_generic as Has2DReuseBuf>::FstOwnedBufferBool, <$r_mat_generic as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
            (<$l_mat_generic as Has2DReuseBuf>::SndOwnedBufferBool, <$r_mat_generic as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
            (<$l_mat_generic as Has2DReuseBuf>::FstHandleBool, <$r_mat_generic as Has2DReuseBuf>::FstHandleBool): SelectPair,
            (<$l_mat_generic as Has2DReuseBuf>::SndHandleBool, <$r_mat_generic as Has2DReuseBuf>::SndHandleBool): SelectPair,
            (<$l_mat_generic as Has2DReuseBuf>::BoundHandlesBool, <$r_mat_generic as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
            (<$l_mat_generic as Has2DReuseBuf>::IsFstBufferTransposed, <$r_mat_generic as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
            (<$l_mat_generic as Has2DReuseBuf>::IsSndBufferTransposed, <$r_mat_generic as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
            (<$l_mat_generic as Has2DReuseBuf>::AreBoundBuffersTransposed, <$r_mat_generic as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair
            $(, $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)?
        {
            type FstHandleBool = <(<$l_mat_generic as Has2DReuseBuf>::FstHandleBool, <$r_mat_generic as Has2DReuseBuf>::FstHandleBool) as TyBoolPair>::Xor;
            type SndHandleBool = <(<$l_mat_generic as Has2DReuseBuf>::SndHandleBool, <$r_mat_generic as Has2DReuseBuf>::SndHandleBool) as TyBoolPair>::Xor;
            type BoundHandlesBool = <(<$l_mat_generic as Has2DReuseBuf>::BoundHandlesBool, <$r_mat_generic as Has2DReuseBuf>::BoundHandlesBool) as TyBoolPair>::Or;
            type FstOwnedBufferBool = <(<$l_mat_generic as Has2DReuseBuf>::FstOwnedBufferBool, <$r_mat_generic as Has2DReuseBuf>::FstOwnedBufferBool) as TyBoolPair>::Xor; 
            type SndOwnedBufferBool = <(<$l_mat_generic as Has2DReuseBuf>::SndOwnedBufferBool, <$r_mat_generic as Has2DReuseBuf>::SndOwnedBufferBool) as TyBoolPair>::Xor; 
            type IsFstBufferTransposed = <(<$l_mat_generic as Has2DReuseBuf>::IsFstBufferTransposed, <$r_mat_generic as Has2DReuseBuf>::IsFstBufferTransposed) as TyBoolPair>::Xor;
            type IsSndBufferTransposed = <(<$l_mat_generic as Has2DReuseBuf>::IsSndBufferTransposed, <$r_mat_generic as Has2DReuseBuf>::IsSndBufferTransposed) as TyBoolPair>::Xor;
            type AreBoundBuffersTransposed = <(<$l_mat_generic as Has2DReuseBuf>::AreBoundBuffersTransposed, <$r_mat_generic as Has2DReuseBuf>::AreBoundBuffersTransposed) as TyBoolPair>::Xor;
            type FstOwnedBuffer = <(<$l_mat_generic as Has2DReuseBuf>::FstOwnedBufferBool, <$r_mat_generic as Has2DReuseBuf>::FstOwnedBufferBool) as SelectPair>::Selected<<$l_mat_generic as Has2DReuseBuf>::FstOwnedBuffer,<$r_mat_generic as Has2DReuseBuf>::FstOwnedBuffer>;
            type SndOwnedBuffer = <(<$l_mat_generic as Has2DReuseBuf>::SndOwnedBufferBool, <$r_mat_generic as Has2DReuseBuf>::SndOwnedBufferBool) as SelectPair>::Selected<<$l_mat_generic as Has2DReuseBuf>::SndOwnedBuffer,<$r_mat_generic as Has2DReuseBuf>::SndOwnedBuffer>;
            type FstType = <(<$l_mat_generic as Has2DReuseBuf>::FstHandleBool, <$r_mat_generic as Has2DReuseBuf>::FstHandleBool) as SelectPair>::Selected<<$l_mat_generic as Has2DReuseBuf>::FstType,<$r_mat_generic as Has2DReuseBuf>::FstType>;
            type SndType = <(<$l_mat_generic as Has2DReuseBuf>::SndHandleBool, <$r_mat_generic as Has2DReuseBuf>::SndHandleBool) as SelectPair>::Selected<<$l_mat_generic as Has2DReuseBuf>::SndType,<$r_mat_generic as Has2DReuseBuf>::SndType>;
            type BoundTypes = <(<$l_mat_generic as Has2DReuseBuf>::BoundHandlesBool, <$r_mat_generic as Has2DReuseBuf>::BoundHandlesBool) as FilterPair>::Filtered<<$l_mat_generic as Has2DReuseBuf>::BoundTypes,<$r_mat_generic as Has2DReuseBuf>::BoundTypes>;
        
            #[inline] unsafe fn assign_1st_buf(&mut self,col_index: usize,row_index: usize,val: Self::FstType) {
                let (l_val,r_val) = <(<$l_mat_generic as Has2DReuseBuf>::FstHandleBool, <$r_mat_generic as Has2DReuseBuf>::FstHandleBool) as SelectPair>::deselect(val);
                self.$l_mat.assign_1st_buf(col_index,row_index,l_val);
                self.$r_mat.assign_1st_buf(col_index,row_index,r_val);
            }
            #[inline] unsafe fn assign_2nd_buf(&mut self,col_index: usize,row_index: usize,val: Self::SndType) {
                let (l_val,r_val) = <(<$l_mat_generic as Has2DReuseBuf>::SndHandleBool, <$r_mat_generic as Has2DReuseBuf>::SndHandleBool) as SelectPair>::deselect(val);
                self.$l_mat.assign_2nd_buf(col_index,row_index,l_val);
                self.$r_mat.assign_2nd_buf(col_index,row_index,r_val);
            }
            #[inline] unsafe fn assign_bound_bufs(&mut self,col_index: usize,row_index: usize,val: Self::BoundTypes) {
                let (l_val,r_val) = <(<$l_mat_generic as Has2DReuseBuf>::BoundHandlesBool, <$r_mat_generic as Has2DReuseBuf>::BoundHandlesBool) as FilterPair>::defilter(val);
                self.$l_mat.assign_bound_bufs(col_index,row_index,l_val);
                self.$r_mat.assign_bound_bufs(col_index,row_index,r_val);
            }
            #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
                <(<$l_mat_generic as Has2DReuseBuf>::FstOwnedBufferBool, <$r_mat_generic as Has2DReuseBuf>::FstOwnedBufferBool) as SelectPair>::select(self.$l_mat.get_1st_buffer(),self.$r_mat.get_1st_buffer())
            }
            #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {
                <(<$l_mat_generic as Has2DReuseBuf>::SndOwnedBufferBool, <$r_mat_generic as Has2DReuseBuf>::SndOwnedBufferBool) as SelectPair>::select(self.$l_mat.get_2nd_buffer(),self.$r_mat.get_2nd_buffer())
            }
            #[inline] unsafe fn drop_1st_buf_index(&mut self,col_index: usize,row_index: usize) {
                self.$l_mat.drop_1st_buf_index(col_index,row_index);
                self.$r_mat.drop_1st_buf_index(col_index,row_index);
            }
            #[inline] unsafe fn drop_2nd_buf_index(&mut self,col_index: usize,row_index: usize) {
                self.$l_mat.drop_2nd_buf_index(col_index,row_index);
                self.$r_mat.drop_2nd_buf_index(col_index,row_index);
            }
            #[inline] unsafe fn drop_bound_bufs_index(&mut self,col_index: usize,row_index: usize) {
                self.$l_mat.drop_bound_bufs_index(col_index,row_index);
                self.$r_mat.drop_bound_bufs_index(col_index,row_index);
            }
        }
    }
}

mat_struct!(MatEntryMap<{M},F: FnMut(M::Item) -> O,O>{mat, f: F}; get2D: O, |self,input| (self.f)(input));
mat_struct!(MatEntryFold<{M},F: FnMut(O,M::Item) -> O,O>{mat, f: F, cell: Option<O>}; output: cell: O, get2D: (), |self,input| self.cell = Some((self.f)(self.cell.take().unwrap(),input)));
mat_struct!(MatEntryFoldRef<{M},F: FnMut(&mut O,M::Item),O>{mat, f: F, cell: ManuallyDrop<O>}; output: cell: O, get2D: (), |self,input| (self.f)(&mut self.cell,input)); // note: use of this is preferred to MatEntryFold

mat_struct!(MatEntryCopiedFold<{M},F: FnMut(O,M::Item) -> O,O>{mat, f: F, cell: Option<O>} where M::Item: Copy; output: cell: O, get2D: M::Item, |self,input| {self.cell = Some((self.f)(self.cell.take().unwrap(),input)); input});
mat_struct!(MatEntryCopiedFoldRef<{M},F: FnMut(&mut O,M::Item),O>{mat, f: F, cell: ManuallyDrop<O>} where M::Item: Copy; output: cell: O, get2D: M::Item, |self,input| {(self.f)(&mut self.cell,input); input});

mat_struct!(MatCopy<'a,{M},I: 'a | Copy>{mat} where M: Get2D<Item = &'a I>; get2D: I, |self,input| *input);
mat_struct!(MatClone<'a,{M},I: 'a | Clone>{mat} where M: Get2D<Item = &'a I>; get2D: I, |self,input| input.clone());

mat_struct!(MatNeg<{M}>{mat} where M::Item: Neg; get2D: <M::Item as Neg>::Output, |self,input| -input);

mat_struct!(MatMulR<{M},S: Copy>{mat,scalar: S} where S: Mul<M::Item>; get2D: <S as Mul<M::Item>>::Output, |self,input| self.scalar * input);
mat_struct!(MatDivR<{M},S: Copy>{mat,scalar: S} where S: Div<M::Item>; get2D: <S as Div<M::Item>>::Output, |self,input| self.scalar / input);
mat_struct!(MatRemR<{M},S: Copy>{mat,scalar: S} where S: Rem<M::Item>; get2D: <S as Rem<M::Item>>::Output, |self,input| self.scalar % input);
mat_struct!(MatMulL<{M},S: Copy>{mat,scalar: S} where M::Item: Mul<S>; get2D: <M::Item as Mul<S>>::Output, |self,input| input * self.scalar);
mat_struct!(MatDivL<{M},S: Copy>{mat,scalar: S} where M::Item: Div<S>; get2D: <M::Item as Div<S>>::Output, |self,input| input / self.scalar);
mat_struct!(MatRemL<{M},S: Copy>{mat,scalar: S} where M::Item: Rem<S>; get2D: <M::Item as Rem<S>>::Output, |self,input| input % self.scalar);

mat_struct!(MatDivAssign<'a,{M},I: 'a | DivAssign<S>,S: Copy>{mat,scalar: S} where M: Get2D<Item = &'a mut I>; get2D: (), |self, input| *input /= self.scalar);
mat_struct!(MatMulAssign<'a,{M},I: 'a | MulAssign<S>,S: Copy>{mat,scalar: S} where M: Get2D<Item = &'a mut I>; get2D: (), |self, input| *input *= self.scalar);
mat_struct!(MatRemAssign<'a,{M},I: 'a | RemAssign<S>,S: Copy>{mat,scalar: S} where M: Get2D<Item = &'a mut I>; get2D: (), |self, input| *input %= self.scalar);

mat_struct!(MatEntrySum<{M},S>{mat,scalar: ManuallyDrop<S>} where S: AddAssign<M::Item>; output: scalar: S, get2D: (), |self, input| *self.scalar += input);
mat_struct!(MatEntryProd<{M},S>{mat,scalar: ManuallyDrop<S>} where S: MulAssign<M::Item>; output: scalar: S, get2D: (), |self, input| *self.scalar *= input);

mat_struct!(MatCopiedEntrySum<{M},S>{mat,scalar: ManuallyDrop<S>} where M::Item: Copy, S: AddAssign<M::Item>; output: scalar: S, get2D: M::Item, |self, input| {*self.scalar += input; input});
mat_struct!(MatCopiedEntryProd<{M},S>{mat,scalar: ManuallyDrop<S>} where M::Item: Copy, S: MulAssign<M::Item>; output: scalar: S, get2D: M::Item, |self, input| {*self.scalar *= input; input});


mat_struct!(MatZip<{M1,M2}>{l_mat,r_mat}; get2D: (M1::Item, M2::Item), |self,l_input,r_input| (l_input,r_input));

mat_struct!(MatAdd<{M1,M2}>{l_mat,r_mat} where M1::Item: Add<M2::Item>; get2D: <M1::Item as Add<M2::Item>>::Output, |self,l_input,r_input| l_input + r_input);
mat_struct!(MatSub<{M1,M2}>{l_mat,r_mat} where M1::Item: Sub<M2::Item>; get2D: <M1::Item as Sub<M2::Item>>::Output, |self,l_input,r_input| l_input - r_input);
mat_struct!(MatCompMul<{M1,M2}>{l_mat,r_mat} where M1::Item: Mul<M2::Item>; get2D: <M1::Item as Mul<M2::Item>>::Output, |self,l_input,r_input| l_input * r_input);
mat_struct!(MatCompDiv<{M1,M2}>{l_mat,r_mat} where M1::Item: Div<M2::Item>; get2D: <M1::Item as Div<M2::Item>>::Output, |self,l_input,r_input| l_input / r_input);
mat_struct!(MatCompRem<{M1,M2}>{l_mat,r_mat} where M1::Item: Rem<M2::Item>; get2D: <M1::Item as Rem<M2::Item>>::Output, |self,l_input,r_input| l_input % r_input);

mat_struct!(MatAddAssign<'a,{M1,M2},I: 'a | AddAssign<M2::Item>>{l_mat,r_mat} where M1: Get2D<Item = &'a mut I>; get2D: (), |self,l_input,r_input| *l_input += r_input);
mat_struct!(MatSubAssign<'a,{M1,M2},I: 'a | SubAssign<M2::Item>>{l_mat,r_mat} where M1: Get2D<Item = &'a mut I>; get2D: (), |self,l_input,r_input| *l_input -= r_input);
mat_struct!(MatCompMulAssign<'a,{M1,M2},I: 'a | MulAssign<M2::Item>>{l_mat,r_mat} where M1: Get2D<Item = &'a mut I>; get2D: (), |self,l_input,r_input| *l_input *= r_input);
mat_struct!(MatCompDivAssign<'a,{M1,M2},I: 'a | DivAssign<M2::Item>>{l_mat,r_mat} where M1: Get2D<Item = &'a mut I>; get2D: (), |self,l_input,r_input| *l_input /= r_input);
mat_struct!(MatCompRemAssign<'a,{M1,M2},I: 'a | RemAssign<M2::Item>>{l_mat,r_mat} where M1: Get2D<Item = &'a mut I>; get2D: (), |self,l_input,r_input| *l_input %= r_input);