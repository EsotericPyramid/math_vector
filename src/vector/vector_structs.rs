use crate::trait_specialization_utils::*;
use crate::util_traits::*;
use super::vec_util_traits::*;
use super::MathVector;
use super::VectorExpr;
use std::ops::*;
use std::mem::ManuallyDrop;

#[repr(transparent)]
#[derive(Clone,Copy)]
pub struct OwnedArray<T,const D: usize>(pub(crate) ManuallyDrop<[T; D]>);

impl<T,const D: usize> OwnedArray<T,D> {
    #[inline]
    pub fn unwrap(self) -> [T; D] {
        ManuallyDrop::into_inner(self.0)
    }
}

impl<T,const D: usize> Deref for OwnedArray<T,D> {
    type Target = ManuallyDrop<[T; D]>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T,const D: usize> DerefMut for OwnedArray<T,D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T,const D: usize> Get for OwnedArray<T,D> {
    type GetBool = Y;
    type IsRepeatable = N;
    type Inputs = T;
    type Item = T;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {std::ptr::read(self.0.get_unchecked(index))}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs,())}
    #[inline] unsafe fn drop_inputs(&mut self, index: usize) {std::ptr::drop_in_place(self.0.get_unchecked_mut(index))}
}

impl<T,const D: usize> HasOutput for OwnedArray<T,D> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

unsafe impl<T,const D: usize> HasReuseBuf for OwnedArray<T,D> {
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

pub struct ReferringOwnedArray<'a,T: 'a,const D: usize>(pub(crate) ManuallyDrop<[T; D]>, pub(crate) std::marker::PhantomData<&'a T>);

impl<'a,T: 'a,const D: usize> Get for ReferringOwnedArray<'a,T,D> {
    type GetBool = Y;
    type IsRepeatable = Y;
    type Inputs = &'a T;
    type Item = &'a T;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {&*(self.0.get_unchecked(index) as *const T)}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs,())}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}
}

impl<'a,T: 'a,const D: usize> HasOutput for ReferringOwnedArray<'a,T,D> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

unsafe impl<'a,T: 'a,const D: usize> HasReuseBuf for ReferringOwnedArray<'a,T,D> {
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


impl<'a,T,const D: usize> Get for &'a [T; D] {
    type GetBool = Y;
    type IsRepeatable = Y;
    type Inputs = &'a T;
    type Item = &'a T;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {self.get_unchecked(index)}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs,())}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}
}

impl<'a,T,const D: usize> HasOutput for &'a [T; D] {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

unsafe impl<'a,T,const D: usize> HasReuseBuf for &'a [T; D] {
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


impl<'a,T,const D: usize> Get for &'a mut [T; D] {
    type GetBool = Y;
    type IsRepeatable = N;
    type Inputs = &'a mut T;
    type Item = &'a mut T;
    type BoundItems = ();

    //ptr shenanigans to change the lifetime
    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {&mut*(self.get_unchecked_mut(index) as *mut T)}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs,())}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}
}

impl<'a,T,const D: usize> HasOutput for &'a mut [T; D] {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

unsafe impl<'a,T,const D: usize> HasReuseBuf for &'a mut [T; D] {
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


#[inline] fn debox<T: Sized>(boxed: &mut Box<T>) -> &mut T {&mut *boxed}

impl<V: VectorLike> Get for Box<V> {
    type GetBool = V::GetBool;
    type IsRepeatable = N;
    type Inputs = V::Inputs;
    type Item = V::Item;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {(debox(self)).get_inputs(index)}
    #[inline] unsafe fn drop_inputs(&mut self, index: usize) {(debox(self)).drop_inputs(index)}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(debox(self)).process(inputs)}
}

unsafe impl<V: VectorLike> HasReuseBuf for Box<V> {
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

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) {(debox(self)).assign_1st_buf(index,val)}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) {(debox(self)).assign_2nd_buf(index,val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) {(debox(self)).assign_bound_bufs(index,val)}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {(debox(self)).get_1st_buffer()}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {(debox(self)).get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) {(debox(self)).drop_1st_buf_index(index)}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) {(debox(self)).drop_2nd_buf_index(index)}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) {(debox(self)).drop_bound_bufs_index(index)}
}


pub struct ReplaceArray<T,const D: usize>(pub(crate) ManuallyDrop<[T; D]>);

impl<T,const D: usize> Get for ReplaceArray<T,D> {
    type GetBool = Y;
    type IsRepeatable = N;
    type Inputs = T;
    type Item = T;
    type BoundItems = ();


    #[inline]
    unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {
        std::ptr::read(self.0.get_unchecked(index))
    }

    #[inline]
    unsafe fn drop_inputs(&mut self, index: usize) {
        std::ptr::drop_in_place(self.0.get_unchecked_mut(index))
    }

    #[inline]
    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs,())}
}

impl<T: Sized,const D: usize> HasOutput for ReplaceArray<T,D> {
    type OutputBool = N;
    type Output = ();

    #[inline]
    unsafe fn output(&mut self) -> Self::Output {}

    #[inline]
    unsafe fn drop_output(&mut self) {} // dropped through reuse buf instead
}

unsafe impl<T,const D: usize> HasReuseBuf for ReplaceArray<T,D> {
    type FstHandleBool = Y;
    type SndHandleBool = N;
    type BoundHandlesBool = N;
    type FstOwnedBufferBool = Y;
    type SndOwnedBufferBool = N;
    type FstOwnedBuffer = MathVector<T,D>;
    type SndOwnedBuffer = ();
    type FstType = T;
    type SndType = ();
    type BoundTypes = ();

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) {std::ptr::write(self.0.get_unchecked_mut(index), val)}
    #[inline] unsafe fn assign_2nd_buf(&mut self, _: usize, _: Self::SndType) {}
    #[inline] unsafe fn assign_bound_bufs(&mut self, _: usize, _: Self::BoundTypes) {}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
        VectorExpr(OwnedArray(std::ptr::read(&self.0)))
    }
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) {std::ptr::drop_in_place(self.0.get_unchecked_mut(index))}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, _: usize) {}
}


pub struct ReplaceHeapArray<T,const D: usize>(pub(crate) ManuallyDrop<Box<[T; D]>>);

impl<T,const D: usize> Get for ReplaceHeapArray<T,D> {
    type GetBool = Y;
    type IsRepeatable = N;
    type Inputs = T;
    type Item = T;
    type BoundItems = ();


    #[inline]
    unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {
        std::ptr::read(self.0.get_unchecked(index))
    }

    #[inline]
    unsafe fn drop_inputs(&mut self, index: usize) {
        std::ptr::drop_in_place(self.0.get_unchecked_mut(index))
    }

    #[inline]
    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs,())}
}

impl<T: Sized,const D: usize> HasOutput for ReplaceHeapArray<T,D> {
    type OutputBool = N;
    type Output = ();

    #[inline]
    unsafe fn output(&mut self) -> Self::Output {}

    #[inline]
    unsafe fn drop_output(&mut self) {} // dropped through reuse buf instead
}

unsafe impl<T,const D: usize> HasReuseBuf for ReplaceHeapArray<T,D> {
    type FstHandleBool = Y;
    type SndHandleBool = N;
    type BoundHandlesBool = N;
    type FstOwnedBufferBool = Y;
    type SndOwnedBufferBool = N;
    type FstOwnedBuffer = Box<MathVector<T,D>>;
    type SndOwnedBuffer = ();
    type FstType = T;
    type SndType = ();
    type BoundTypes = ();

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) {std::ptr::write(self.0.get_unchecked_mut(index), val)}
    #[inline] unsafe fn assign_2nd_buf(&mut self, _: usize, _: Self::SndType) {}
    #[inline] unsafe fn assign_bound_bufs(&mut self, _: usize, _: Self::BoundTypes) {}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
        //  safety, equivilent types in order:
        //      ManuallyDrop<Box<[T; D]>>
        //      Box<[T; D]>
        //      Box<ManuallyDrop<[T; D]>>
        //      Box<OwnedArray<[T; D]>>
        //      Box<VectorExpr<OwnedArray<[T; D]>,D>>
        //      Box<MathVector<T,D>>
        std::mem::transmute_copy::<ManuallyDrop<Box<[T; D]>>,Box<MathVector<T,D>>>(&self.0)
    }
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) {std::ptr::drop_in_place(self.0.get_unchecked_mut(index))}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, _: usize) {}
}


pub struct VecGenerator<F: FnMut() -> O,O>(pub(crate) F);

impl<F: FnMut() -> O,O> Get for VecGenerator<F,O> {
    type GetBool = Y;
    type IsRepeatable = N;
    type Inputs = ();
    type Item = O;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, _: usize) -> Self::Inputs {}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}
    #[inline] fn process(&mut self, _: Self::Inputs) -> (Self::Item, Self::BoundItems) {((self.0)(),())}
}

impl<F: FnMut() -> O,O> HasOutput for VecGenerator<F,O> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

unsafe impl<F: FnMut() -> O,O> HasReuseBuf for VecGenerator<F,O> {
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


pub struct VecIndexGenerator<F: FnMut(usize) -> O,O>(pub(crate) F);

impl<F: FnMut(usize) -> O,O> Get for VecIndexGenerator<F,O> {
    type GetBool = Y;
    type IsRepeatable = N;
    type Inputs = usize;
    type Item = O;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {index}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {((self.0)(inputs),())}
}

impl<F: FnMut(usize) -> O,O> HasOutput for VecIndexGenerator<F,O> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

unsafe impl<F: FnMut(usize) -> O,O> HasReuseBuf for VecIndexGenerator<F,O> {
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


pub struct VecAttachBuf<'a,V: VectorLike<FstHandleBool = N>,T,const D: usize>{pub(crate) vec: V, pub(crate) buf: &'a mut [T; D]} 

impl<'a,V: VectorLike<FstHandleBool = N>,T,const D: usize> Get for VecAttachBuf<'a,V,T,D> {
    type GetBool = V::GetBool;
    type IsRepeatable = V::IsRepeatable;
    type Inputs = V::Inputs;
    type Item = V::Item;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {self.vec.get_inputs(index)}

    #[inline] unsafe fn drop_inputs(&mut self, index: usize) {self.vec.drop_inputs(index)}

    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(inputs)}
}

impl<'a,V: VectorLike<FstHandleBool = N>,T,const D: usize> HasOutput for VecAttachBuf<'a,V,T,D> {
    type OutputBool = V::OutputBool;
    type Output = V::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output {self.vec.output()}
    #[inline] unsafe fn drop_output(&mut self) {self.vec.drop_output()}
}

unsafe impl<'b,V: VectorLike<FstHandleBool = N>,T,const D: usize> HasReuseBuf for VecAttachBuf<'b,V,T,D> {
    type FstHandleBool = Y;
    type SndHandleBool = V::SndHandleBool;
    type BoundHandlesBool = V::BoundHandlesBool;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = V::SndOwnedBufferBool;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = V::SndOwnedBuffer;
    type FstType = T;
    type SndType = V::SndType;
    type BoundTypes = V::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) {*self.buf.get_unchecked_mut(index) = val}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) {self.vec.assign_2nd_buf(index,val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) {self.vec.assign_bound_bufs(index,val)}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.vec.get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) {self.vec.drop_2nd_buf_index(index)}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) {self.vec.drop_bound_bufs_index(index)}
}


pub struct VecCreateBuf<V: VectorLike<FstHandleBool = N>,T,const D: usize>{pub(crate) vec: V, pub(crate) buf: [std::mem::MaybeUninit<T>; D]}

impl<V: VectorLike<FstHandleBool = N>,T,const D: usize> Get for VecCreateBuf<V,T,D> {
    type GetBool = V::GetBool;
    type IsRepeatable = V::IsRepeatable;
    type Inputs = V::Inputs;
    type Item = V::Item;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {self.vec.get_inputs(index)}

    #[inline] unsafe fn drop_inputs(&mut self, index: usize) {self.vec.drop_inputs(index)}

    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(inputs)}
}

impl<V: VectorLike<FstHandleBool = N>,T,const D: usize> HasOutput for VecCreateBuf<V,T,D> {
    type OutputBool = V::OutputBool;
    type Output = V::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output {self.vec.output()}
    #[inline] unsafe fn drop_output(&mut self) {self.vec.drop_output()}
}

unsafe impl<V: VectorLike<FstHandleBool = N>,T,const D: usize> HasReuseBuf for VecCreateBuf<V,T,D> {
    type FstHandleBool = Y;
    type SndHandleBool = V::SndHandleBool;
    type BoundHandlesBool = V::BoundHandlesBool;
    type FstOwnedBufferBool = Y;
    type SndOwnedBufferBool = V::SndOwnedBufferBool;
    type FstOwnedBuffer = MathVector<T,D>;
    type SndOwnedBuffer = V::SndOwnedBuffer;
    type FstType = T;
    type SndType = V::SndType;
    type BoundTypes = V::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) {std::ptr::write(self.buf.get_unchecked_mut(index),std::mem::MaybeUninit::new(val))}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) {self.vec.assign_2nd_buf(index,val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) {self.vec.assign_bound_bufs(index,val)}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
        VectorExpr(OwnedArray(std::mem::transmute_copy::<[std::mem::MaybeUninit<T>; D],ManuallyDrop<[T; D]>>(&self.buf)))
    }
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.vec.get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) {self.buf.get_unchecked_mut(index).assume_init_drop()}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) {self.vec.drop_2nd_buf_index(index)}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) {self.vec.drop_bound_bufs_index(index)}
}


pub struct VecCreateHeapBuf<V: VectorLike<FstHandleBool = N>,T,const D: usize>{pub(crate) vec: V, pub(crate) buf: ManuallyDrop<Box<[std::mem::MaybeUninit<T>; D]>>}

impl<V: VectorLike<FstHandleBool = N>,T,const D: usize> Get for VecCreateHeapBuf<V,T,D> {
    type GetBool = V::GetBool;
    type IsRepeatable = V::IsRepeatable;
    type Inputs = V::Inputs;
    type Item = V::Item;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {self.vec.get_inputs(index)}

    #[inline] unsafe fn drop_inputs(&mut self, index: usize) {self.vec.drop_inputs(index)}

    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(inputs)}
}

impl<V: VectorLike<FstHandleBool = N>,T,const D: usize> HasOutput for VecCreateHeapBuf<V,T,D> {
    type OutputBool = V::OutputBool;
    type Output = V::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output {self.vec.output()}
    #[inline] unsafe fn drop_output(&mut self) {self.vec.drop_output()}
}

unsafe impl<V: VectorLike<FstHandleBool = N>,T,const D: usize> HasReuseBuf for VecCreateHeapBuf<V,T,D> {
    type FstHandleBool = Y;
    type SndHandleBool = V::SndHandleBool;
    type BoundHandlesBool = V::BoundHandlesBool;
    type FstOwnedBufferBool = Y;
    type SndOwnedBufferBool = V::SndOwnedBufferBool;
    type FstOwnedBuffer = Box<MathVector<T,D>>;
    type SndOwnedBuffer = V::SndOwnedBuffer;
    type FstType = T;
    type SndType = V::SndType;
    type BoundTypes = V::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) {std::ptr::write(self.buf.get_unchecked_mut(index),std::mem::MaybeUninit::new(val))}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) {self.vec.assign_2nd_buf(index,val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) {self.vec.assign_bound_bufs(index,val)}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
        std::mem::transmute_copy::<Box<[std::mem::MaybeUninit<T>; D]>,Box<MathVector<T,D>>>(&self.buf)
    }
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.vec.get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) {self.buf.get_unchecked_mut(index).assume_init_drop()}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) {self.vec.drop_2nd_buf_index(index)}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) {self.vec.drop_bound_bufs_index(index)}
}


pub struct VecMaybeCreateBuf<V: VectorLike,T,const D: usize> where <V::FstHandleBool as TyBool>::Neg: Filter {pub(crate) vec: V, pub(crate) buf: [std::mem::MaybeUninit<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>; D]}

impl<V: VectorLike,T,const D: usize> Get for VecMaybeCreateBuf<V,T,D> where <V::FstHandleBool as TyBool>::Neg: Filter {
    type GetBool = V::GetBool;
    type IsRepeatable = V::IsRepeatable;
    type Inputs = V::Inputs;
    type Item = V::Item;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {self.vec.get_inputs(index)}

    #[inline] unsafe fn drop_inputs(&mut self, index: usize) {self.vec.drop_inputs(index)}

    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(inputs)}
}

impl<V: VectorLike,T,const D: usize> HasOutput for VecMaybeCreateBuf<V,T,D> where <V::FstHandleBool as TyBool>::Neg: Filter {
    type OutputBool = V::OutputBool;
    type Output = V::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output {self.vec.output()}
    #[inline] unsafe fn drop_output(&mut self) {self.vec.drop_output()}
}

unsafe impl<V: VectorLike,T,const D: usize> HasReuseBuf for VecMaybeCreateBuf<V,T,D> 
where 
    <V::FstHandleBool as TyBool>::Neg: Filter, 
    (V::FstHandleBool, <V::FstHandleBool as TyBool>::Neg): SelectPair, 
    (V::FstOwnedBufferBool,<V::FstHandleBool as TyBool>::Neg): TyBoolPair
{
    type FstHandleBool = Y;
    type SndHandleBool = V::SndHandleBool;
    type BoundHandlesBool = V::BoundHandlesBool;
    type FstOwnedBufferBool = <(V::FstOwnedBufferBool,<V::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or;
    type SndOwnedBufferBool = V::SndOwnedBufferBool;
    type FstOwnedBuffer = <(V::FstHandleBool,<V::FstHandleBool as TyBool>::Neg) as SelectPair>::Selected<V::FstOwnedBuffer, MathVector<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>,D>>;
    type SndOwnedBuffer = V::SndOwnedBuffer;
    type FstType = <(V::FstHandleBool, <V::FstHandleBool as TyBool>::Neg) as SelectPair>::Selected<V::FstType,<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>;
    type SndType = V::SndType;
    type BoundTypes = V::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) {
        let (init_val,attached_val) = <(V::FstHandleBool, <V::FstHandleBool as TyBool>::Neg) as SelectPair>::deselect(val);
        self.vec.assign_1st_buf(index, init_val);
        std::ptr::write(self.buf.get_unchecked_mut(index),std::mem::MaybeUninit::new(attached_val));
    }
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) {self.vec.assign_2nd_buf(index,val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) {self.vec.assign_bound_bufs(index,val)}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
        <(V::FstHandleBool,<V::FstHandleBool as TyBool>::Neg) as SelectPair>::select(
            self.vec.get_1st_buffer(),
            VectorExpr(OwnedArray(ManuallyDrop::new(std::mem::transmute_copy::<[std::mem::MaybeUninit<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>; D], [<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>; D]>(&self.buf))))
        )
    }
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.vec.get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) {
        self.vec.drop_1st_buf_index(index);
        // assuming that its safe to drop () even if we never properlly "initiallized" them
        self.buf.get_unchecked_mut(index).assume_init_drop();
    }
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) {self.vec.drop_2nd_buf_index(index)}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) {self.vec.drop_bound_bufs_index(index)}
}


pub struct VecMaybeCreateHeapBuf<V: VectorLike,T,const D: usize> where <V::FstHandleBool as TyBool>::Neg: Filter {pub(crate) vec: V, pub(crate) buf: ManuallyDrop<Box<[std::mem::MaybeUninit<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>; D]>>}

impl<V: VectorLike,T,const D: usize> Get for VecMaybeCreateHeapBuf<V,T,D> where <V::FstHandleBool as TyBool>::Neg: Filter {
    type GetBool = V::GetBool;
    type IsRepeatable = V::IsRepeatable;
    type Inputs = V::Inputs;
    type Item = V::Item;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {self.vec.get_inputs(index)}

    #[inline] unsafe fn drop_inputs(&mut self, index: usize) {self.vec.drop_inputs(index)}

    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(inputs)}
}

impl<V: VectorLike,T,const D: usize> HasOutput for VecMaybeCreateHeapBuf<V,T,D> where <V::FstHandleBool as TyBool>::Neg: Filter {
    type OutputBool = V::OutputBool;
    type Output = V::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output {self.vec.output()}
    #[inline] unsafe fn drop_output(&mut self) {self.vec.drop_output()}
}

unsafe impl<V: VectorLike,T,const D: usize> HasReuseBuf for VecMaybeCreateHeapBuf<V,T,D> 
where 
    <V::FstHandleBool as TyBool>::Neg: Filter, 
    (V::FstHandleBool, <V::FstHandleBool as TyBool>::Neg): SelectPair,
    (V::FstOwnedBufferBool,<V::FstHandleBool as TyBool>::Neg): TyBoolPair
{
    type FstHandleBool = Y;
    type SndHandleBool = V::SndHandleBool;
    type BoundHandlesBool = V::BoundHandlesBool;
    type FstOwnedBufferBool = <(V::FstOwnedBufferBool,<V::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or;
    type SndOwnedBufferBool = V::SndOwnedBufferBool;
    type FstOwnedBuffer = <(V::FstHandleBool,<V::FstHandleBool as TyBool>::Neg) as SelectPair>::Selected<V::FstOwnedBuffer, Box<MathVector<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>,D>>>;
    type SndOwnedBuffer = V::SndOwnedBuffer;
    type FstType = <(V::FstHandleBool, <V::FstHandleBool as TyBool>::Neg) as SelectPair>::Selected<V::FstType,<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>;
    type SndType = V::SndType;
    type BoundTypes = V::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) {
        let (init_val,attached_val) = <(V::FstHandleBool, <V::FstHandleBool as TyBool>::Neg) as SelectPair>::deselect(val);
        self.vec.assign_1st_buf(index, init_val);
        std::ptr::write(self.buf.get_unchecked_mut(index),std::mem::MaybeUninit::new(attached_val));
    }
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) {self.vec.assign_2nd_buf(index,val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) {self.vec.assign_bound_bufs(index,val)}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
        <(V::FstHandleBool,<V::FstHandleBool as TyBool>::Neg) as SelectPair>::select(
            self.vec.get_1st_buffer(),
            std::mem::transmute_copy::<Box<[std::mem::MaybeUninit<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>; D]>, Box<MathVector<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>, D>>>(&self.buf)
        )
    }
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.vec.get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) {
        self.vec.drop_1st_buf_index(index);
        // assuming that its safe to drop () even if we never properlly "initiallized" them
        self.buf.get_unchecked_mut(index).assume_init_drop();
    }
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) {self.vec.drop_2nd_buf_index(index)}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) {self.vec.drop_bound_bufs_index(index)}
}


pub struct VecBind<T: VectorLike<FstHandleBool = Y>> {pub(crate) vec: T} 

impl<T: VectorLike<FstHandleBool = Y>> Get for VecBind<T> where (T::BoundHandlesBool,Y): FilterPair {
    type GetBool = N;
    type IsRepeatable = N;
    type Inputs = T::Inputs;
    type Item = ();
    type BoundItems = <(T::BoundHandlesBool,Y) as FilterPair>::Filtered<T::BoundItems,T::Item>;

    #[inline]
    unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {
        self.vec.get_inputs(index)
    }

    #[inline]
    unsafe fn drop_inputs(&mut self, index: usize) {
        self.vec.drop_inputs(index)
    }

    #[inline]
    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
        let (item,bound) = self.vec.process(inputs);
        ((),<(T::BoundHandlesBool,Y) as FilterPair>::filter(bound,item))
    }
}

impl<T: VectorLike<FstHandleBool = Y>> HasOutput for VecBind<T> where (T::OutputBool,T::FstOwnedBufferBool): FilterPair {
    type OutputBool = <(T::OutputBool,T::FstOwnedBufferBool) as TyBoolPair>::Or;
    type Output = <(T::OutputBool,T::FstOwnedBufferBool) as FilterPair>::Filtered<T::Output,T::FstOwnedBuffer>;
    
    #[inline]
    unsafe fn output(&mut self) -> Self::Output {
        <(T::OutputBool,T::FstOwnedBufferBool) as FilterPair>::filter(self.vec.output(),self.vec.get_1st_buffer())
    }

    #[inline]
    unsafe fn drop_output(&mut self) {
        self.vec.drop_output(); // buffer dropped through HasReuseBuf
    }
}

unsafe impl<T: VectorLike<FstHandleBool = Y>> HasReuseBuf for VecBind<T> where (T::BoundHandlesBool,Y): FilterPair {
    type FstHandleBool = N;
    type SndHandleBool = T::SndHandleBool;
    type BoundHandlesBool = <(T::BoundHandlesBool,Y) as TyBoolPair>::Or;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = T::SndOwnedBufferBool;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = T::SndOwnedBuffer;
    type FstType = ();
    type SndType = T::SndType;
    type BoundTypes = <(T::BoundHandlesBool,Y) as FilterPair>::Filtered<T::BoundTypes,T::FstType>;

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) {self.vec.assign_2nd_buf(index, val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) {
        let (bounded_vals,new_bound_val) = <(T::BoundHandlesBool,Y) as FilterPair>::defilter(val);
        self.vec.assign_bound_bufs(index, bounded_vals);
        self.vec.assign_1st_buf(index, new_bound_val);
    }
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.vec.get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) {self.vec.drop_2nd_buf_index(index)}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) {
        self.vec.drop_1st_buf_index(index);
        self.vec.drop_bound_bufs_index(index);
    }
}


pub struct VecMapBind<T: VectorLike<FstHandleBool = Y>,F: FnMut(T::Item) -> (I,B),I,B> {pub(crate) vec: T, pub(crate) f: F}

impl<T: VectorLike<FstHandleBool = Y>,F: FnMut(T::Item) -> (I,B),I,B> Get for VecMapBind<T,F,I,B> where (T::BoundHandlesBool,Y): FilterPair {
    type GetBool = Y;
    type IsRepeatable = N;
    type Inputs = T::Inputs;
    type Item = I;
    type BoundItems = <(T::BoundHandlesBool,Y) as FilterPair>::Filtered<T::BoundItems,B>;

    #[inline]
    unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {
        self.vec.get_inputs(index)
    }

    #[inline]
    unsafe fn drop_inputs(&mut self, index: usize) {
        self.vec.drop_inputs(index)
    }

    #[inline]
    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
        let (item,bound) = self.vec.process(inputs);
        let (processed_item,processed_bound) = (self.f)(item);
        (processed_item,<(T::BoundHandlesBool,Y) as FilterPair>::filter(bound,processed_bound))
    }
}

impl<T: VectorLike<FstHandleBool = Y>,F: FnMut(T::Item) -> (I,B),I,B> HasOutput for VecMapBind<T,F,I,B> where (T::OutputBool,T::FstOwnedBufferBool): FilterPair {
    type OutputBool = <(T::OutputBool,T::FstOwnedBufferBool) as TyBoolPair>::Or;
    type Output = <(T::OutputBool,T::FstOwnedBufferBool) as FilterPair>::Filtered<T::Output,T::FstOwnedBuffer>;
    
    #[inline]
    unsafe fn output(&mut self) -> Self::Output {
        <(T::OutputBool,T::FstOwnedBufferBool) as FilterPair>::filter(self.vec.output(),self.vec.get_1st_buffer())
    }

    #[inline]
    unsafe fn drop_output(&mut self) {
        self.vec.drop_output(); // buffer dropped through HasReuseBuf
    }
}

unsafe impl<T: VectorLike<FstHandleBool = Y>,F: FnMut(T::Item) -> (I,B),I,B> HasReuseBuf for VecMapBind<T,F,I,B> where (T::BoundHandlesBool,Y): FilterPair {
    type FstHandleBool = N;
    type SndHandleBool = T::SndHandleBool;
    type BoundHandlesBool = <(T::BoundHandlesBool,Y) as TyBoolPair>::Or;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = T::SndOwnedBufferBool;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = T::SndOwnedBuffer;
    type FstType = ();
    type SndType = T::SndType;
    type BoundTypes = <(T::BoundHandlesBool,Y) as FilterPair>::Filtered<T::BoundTypes,T::FstType>;

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) {self.vec.assign_2nd_buf(index, val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) {
        let (bounded_vals,new_bound_val) = <(T::BoundHandlesBool,Y) as FilterPair>::defilter(val);
        self.vec.assign_bound_bufs(index, bounded_vals);
        self.vec.assign_1st_buf(index, new_bound_val);
    }
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.vec.get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) {self.vec.drop_2nd_buf_index(index)}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) {
        self.vec.drop_1st_buf_index(index);
        self.vec.drop_bound_bufs_index(index);
    }
}

pub struct VecHalfBind<T: VectorLike<FstHandleBool = Y>> {pub(crate) vec: T} 

impl<T: VectorLike<FstHandleBool = Y>> VecHalfBind<T> {
    pub(crate) unsafe fn get_bound_buf(&mut self) -> T::FstOwnedBuffer {
        self.vec.get_1st_buffer()
    }
}

impl<T: VectorLike<FstHandleBool = Y>> Get for VecHalfBind<T> where (T::BoundHandlesBool,Y): FilterPair {
    type GetBool = N;
    type IsRepeatable = N;
    type Inputs = T::Inputs;
    type Item = ();
    type BoundItems = <(T::BoundHandlesBool,Y) as FilterPair>::Filtered<T::BoundItems,T::Item>;

    #[inline]
    unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {
        self.vec.get_inputs(index)
    }

    #[inline]
    unsafe fn drop_inputs(&mut self, index: usize) {
        self.vec.drop_inputs(index)
    }

    #[inline]
    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
        let (item,bound) = self.vec.process(inputs);
        ((),<(T::BoundHandlesBool,Y) as FilterPair>::filter(bound,item))
    }
}

impl<T: VectorLike<FstHandleBool = Y>> HasOutput for VecHalfBind<T> where (T::OutputBool,T::FstOwnedBufferBool): TyBoolPair {
    type OutputBool = <(T::OutputBool,T::FstOwnedBufferBool) as TyBoolPair>::Or;
    type Output = T::Output;
    
    #[inline]
    unsafe fn output(&mut self) -> Self::Output {
        self.vec.output()
    }

    #[inline]
    unsafe fn drop_output(&mut self) {
        self.vec.drop_output(); // buffer dropped through HasReuseBuf
    }
}

unsafe impl<T: VectorLike<FstHandleBool = Y>> HasReuseBuf for VecHalfBind<T> where (T::BoundHandlesBool,Y): FilterPair {
    type FstHandleBool = N;
    type SndHandleBool = T::SndHandleBool;
    type BoundHandlesBool = <(T::BoundHandlesBool,Y) as TyBoolPair>::Or;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = T::SndOwnedBufferBool;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = T::SndOwnedBuffer;
    type FstType = ();
    type SndType = T::SndType;
    type BoundTypes = <(T::BoundHandlesBool,Y) as FilterPair>::Filtered<T::BoundTypes,T::FstType>;

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) {self.vec.assign_2nd_buf(index, val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) {
        let (bounded_vals,new_bound_val) = <(T::BoundHandlesBool,Y) as FilterPair>::defilter(val);
        self.vec.assign_bound_bufs(index, bounded_vals);
        self.vec.assign_1st_buf(index, new_bound_val);
    }
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.vec.get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) {self.vec.drop_2nd_buf_index(index)}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) {
        self.vec.drop_1st_buf_index(index);
        self.vec.drop_bound_bufs_index(index);
    }
}


pub struct VecBufSwap<T: VectorLike> {pub(crate) vec: T}

impl<T: VectorLike> Get for VecBufSwap<T> {
    type GetBool = T::GetBool;
    type IsRepeatable = T::IsRepeatable;
    type Inputs = T::Inputs;
    type Item = T::Item;
    type BoundItems = T::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {self.vec.get_inputs(index)}
    #[inline] unsafe fn drop_inputs(&mut self, index: usize) {self.vec.drop_inputs(index)}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(inputs)}
}

impl<T: VectorLike> HasOutput for VecBufSwap<T> {
    type OutputBool = T::OutputBool;
    type Output = T::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output {self.vec.output()}
    #[inline] unsafe fn drop_output(&mut self) {self.vec.drop_output()}
}

unsafe impl<T: VectorLike> HasReuseBuf for VecBufSwap<T> {
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

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) {self.vec.assign_2nd_buf(index, val)}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) {self.vec.assign_1st_buf(index, val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) {self.vec.assign_bound_bufs(index, val)}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {self.vec.get_2nd_buffer()}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.vec.get_1st_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) {self.vec.drop_2nd_buf_index(index)}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) {self.vec.drop_1st_buf_index(index)}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) {self.vec.drop_bound_bufs_index(index)}
}

pub struct VecOffset<T: VectorLike,const D: usize>{pub(crate) vec: T, pub(crate) offset: usize}

impl<T: VectorLike,const D: usize> VecOffset<T,D> {
    #[inline]
    fn offset_index(&self,index: usize) -> usize {
        let mut offset_index = index + self.offset;
        if offset_index >= index { 
            offset_index %= D;
        } else { //index overflowed, LLVM should be able to elid this most of the time
            //if the index overflowed, (usize::MAX) + 1 was subtracted from it, add (usize::MAX)+1 mod D to recover
            offset_index %= D;
            offset_index += ((usize::MAX % D) + 1) % D; // 2 modulos to prevent overflow
            offset_index %= D;
        }
        offset_index
    }
}

impl<T: VectorLike,const D: usize> Get for VecOffset<T,D> {
    type GetBool = T::GetBool;
    type IsRepeatable = N; // NOTE: N because offset_index adds a small amount of extra computation on get
    type Inputs = T::Inputs;
    type Item = T::Item;
    type BoundItems = T::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {self.vec.get_inputs(self.offset_index(index))}
    #[inline] unsafe fn drop_inputs(&mut self, index: usize) {self.vec.drop_inputs(self.offset_index(index))}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(inputs)}
}

impl<T: VectorLike,const D: usize> HasOutput for VecOffset<T,D> {
    type OutputBool = T::OutputBool;
    type Output = T::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output {self.vec.output()}
    #[inline] unsafe fn drop_output(&mut self) {self.vec.drop_output()}
}

unsafe impl<T: VectorLike,const D: usize> HasReuseBuf for VecOffset<T,D> {
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

    #[inline] unsafe fn assign_1st_buf(&mut self,index: usize,val: Self::FstType) {self.vec.assign_1st_buf(self.offset_index(index),val)}
    #[inline] unsafe fn assign_2nd_buf(&mut self,index: usize,val: Self::SndType) {self.vec.assign_2nd_buf(self.offset_index(index),val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self,index: usize,val: Self::BoundTypes) {self.vec.assign_bound_bufs(self.offset_index(index),val)}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {self.vec.get_1st_buffer()}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.vec.get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self,index: usize) {self.vec.drop_1st_buf_index(self.offset_index(index))}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self,index: usize) {self.vec.drop_2nd_buf_index(self.offset_index(index))}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self,index: usize) {self.vec.drop_bound_bufs_index(self.offset_index(index))}
}

// TODO: add offset method for runtime vecs
pub struct RuntimeVecOffset<T: VectorLike>{pub(crate) vec: T, pub(crate) offset: usize, pub(crate) size: usize}

impl<T: VectorLike> RuntimeVecOffset<T> {
    #[inline]
    fn offset_index(&self,index: usize) -> usize {
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

impl<T: VectorLike> Get for RuntimeVecOffset<T> {
    type GetBool = T::GetBool;
    type IsRepeatable = N; // NOTE: N because offset_index adds a small amount of extra computation on get
    type Inputs = T::Inputs;
    type Item = T::Item;
    type BoundItems = T::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {self.vec.get_inputs(self.offset_index(index))}
    #[inline] unsafe fn drop_inputs(&mut self, index: usize) {self.vec.drop_inputs(self.offset_index(index))}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(inputs)}
}

impl<T: VectorLike> HasOutput for RuntimeVecOffset<T> {
    type OutputBool = T::OutputBool;
    type Output = T::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output {self.vec.output()}
    #[inline] unsafe fn drop_output(&mut self) {self.vec.drop_output()}
}

unsafe impl<T: VectorLike> HasReuseBuf for RuntimeVecOffset<T> {
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

    #[inline] unsafe fn assign_1st_buf(&mut self,index: usize,val: Self::FstType) {self.vec.assign_1st_buf(self.offset_index(index),val)}
    #[inline] unsafe fn assign_2nd_buf(&mut self,index: usize,val: Self::SndType) {self.vec.assign_2nd_buf(self.offset_index(index),val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self,index: usize,val: Self::BoundTypes) {self.vec.assign_bound_bufs(self.offset_index(index),val)}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {self.vec.get_1st_buffer()}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.vec.get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self,index: usize) {self.vec.drop_1st_buf_index(self.offset_index(index))}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self,index: usize) {self.vec.drop_2nd_buf_index(self.offset_index(index))}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self,index: usize) {self.vec.drop_bound_bufs_index(self.offset_index(index))}
}

/// SAFETY: it is expected that the used_vec field is safe to output in addition to normal correct implementation
pub struct VecAttachUsedVec<V: VectorLike,USEDV: VectorLike>{pub(crate) vec: V, pub(crate) used_vec: USEDV}

impl<V: VectorLike,USEDV: VectorLike> Get for VecAttachUsedVec<V,USEDV> {
    type GetBool = V::GetBool;
    type IsRepeatable = V::IsRepeatable;
    type Inputs = V::Inputs;
    type Item = V::Item;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {self.vec.get_inputs(index)}
    #[inline] unsafe fn drop_inputs(&mut self, index: usize) {self.vec.drop_inputs(index)}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(inputs)}
}

impl<V: VectorLike,USEDV: VectorLike> HasOutput for VecAttachUsedVec<V,USEDV> where (V::OutputBool,USEDV::OutputBool): FilterPair {
    type OutputBool = <(V::OutputBool,USEDV::OutputBool) as TyBoolPair>::Or;
    type Output = <(V::OutputBool,USEDV::OutputBool) as FilterPair>::Filtered<V::Output,USEDV::Output>;

    #[inline] 
    unsafe fn output(&mut self) -> Self::Output {
        <(V::OutputBool,USEDV::OutputBool) as FilterPair>::filter(
            self.vec.output(),
            self.used_vec.output()
        )
    }

    #[inline]
    unsafe fn drop_output(&mut self) {
        self.vec.drop_output();
        self.used_vec.drop_output();
    }
}

unsafe impl<V: VectorLike,USEDV: VectorLike> HasReuseBuf for VecAttachUsedVec<V,USEDV> 
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
    type FstOwnedBuffer = <(V::FstOwnedBufferBool, USEDV::FstOwnedBufferBool) as SelectPair>::Selected<V::FstOwnedBuffer,USEDV::FstOwnedBuffer>;
    type SndOwnedBuffer = <(V::SndOwnedBufferBool, USEDV::SndOwnedBufferBool) as SelectPair>::Selected<V::SndOwnedBuffer,USEDV::SndOwnedBuffer>;
    type FstType = <(V::FstHandleBool, USEDV::FstHandleBool) as SelectPair>::Selected<V::FstType,USEDV::FstType>;
    type SndType = <(V::SndHandleBool, USEDV::SndHandleBool) as SelectPair>::Selected<V::SndType,USEDV::SndType>;
    type BoundTypes = V::BoundTypes;

    #[inline] unsafe fn assign_1st_buf<'z>(&'z mut self,index: usize,val: Self::FstType) {
        let (l_val,r_val) = <(V::FstHandleBool, USEDV::FstHandleBool) as SelectPair>::deselect(val);
        self.vec.assign_1st_buf(index,l_val);
        self.used_vec.assign_1st_buf(index,r_val);
    }
    #[inline] unsafe fn assign_2nd_buf<'z>(&'z mut self,index: usize,val: Self::SndType) {
        let (l_val,r_val) = <(V::SndHandleBool, USEDV::SndHandleBool) as SelectPair>::deselect(val);
        self.vec.assign_2nd_buf(index,l_val);
        self.used_vec.assign_2nd_buf(index,r_val);
    }
    #[inline] unsafe fn assign_bound_bufs<'z>(&'z mut self,index: usize,val: Self::BoundTypes) {
        self.vec.assign_bound_bufs(index,val);
    }
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
        <(V::FstOwnedBufferBool, USEDV::FstOwnedBufferBool) as SelectPair>::select(self.vec.get_1st_buffer(),self.used_vec.get_1st_buffer())
    }
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {
        <(V::SndOwnedBufferBool, USEDV::SndOwnedBufferBool) as SelectPair>::select(self.vec.get_2nd_buffer(),self.used_vec.get_2nd_buffer())
    }
    #[inline] unsafe fn drop_1st_buf_index(&mut self,index: usize) {
        self.vec.drop_1st_buf_index(index);
        self.used_vec.drop_1st_buf_index(index);
    }
    #[inline] unsafe fn drop_2nd_buf_index(&mut self,index: usize) {
        self.vec.drop_2nd_buf_index(index);
        self.used_vec.drop_2nd_buf_index(index);
    }
    #[inline] unsafe fn drop_bound_bufs_index(&mut self,index: usize) {
        self.vec.drop_bound_bufs_index(index);
        self.used_vec.drop_bound_bufs_index(index);
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

macro_rules! vec_struct {
    (
        $struct:ident<$($($lifetime:lifetime),+,)? {$vec_generic:ident} $(,$($generic:ident $(: $($generic_lifetime:lifetime |)? $fst_generic_bound:path $(| $generic_bound:path)*)?),+)?>{$vec:ident $(,$($field:ident: $field_ty:ty),+)?}
        $(where $($bound_ty:ty: $fst_where_bound:path $(| $where_bound:path)*),+)?;
        $(output: $outputted_field:ident: $output_ty:ty,)?
        get: $item:ty, |$self:ident,$(($is_mut:tt))? $input:ident| $get_expr:expr $(, $is_repeatable:ty)?
    ) => {
        pub struct $struct<$($($lifetime),+,)? $vec_generic: VectorLike $(,$($generic $(: $($generic_lifetime +)? $fst_generic_bound $(+ $generic_bound)*)?),+)?> $(where $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {pub(crate) $vec: $vec_generic $(,$(pub(crate) $field: $field_ty),+)?}

        impl<$($($lifetime),+,)? $vec_generic: VectorLike $(,$($generic $(: $fst_generic_bound $(+ $generic_bound)*)?),+)?> Get for $struct<$($($lifetime),+,)? $vec_generic $(,$($generic),+)?> $(where $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {
            type GetBool = is_unit!($item);
            type IsRepeatable = is_present!($($is_repeatable)?);
            type Inputs = <$vec_generic as Get>::Inputs;
            type Item = $item;
            type BoundItems = <$vec_generic as Get>::BoundItems;

            #[inline]
            unsafe fn get_inputs(&mut self,index: usize) -> Self::Inputs {self.$vec.get_inputs(index)}

            #[inline]
            unsafe fn drop_inputs(&mut self,index: usize) {self.$vec.drop_inputs(index)}

            #[inline]
            fn process($self: &mut Self,inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
                let ($($is_mut)? $input,bound_items) = $self.$vec.process(inputs);
                ($get_expr,bound_items)
            }
        }

        impl<$($($lifetime),+,)? $vec_generic: VectorLike $(,$($generic $(: $fst_generic_bound $(+ $generic_bound)*)?),+)?> HasOutput for $struct<$($($lifetime),+,)? $vec_generic $(,$($generic),+)?> 
        where ($vec_generic::OutputBool,is_present!($($outputted_field)?)): FilterPair $(,$($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {
            type OutputBool = optimized_or!($vec_generic::OutputBool,$($outputted_field)?);
            type Output = <($vec_generic::OutputBool,is_present!($($outputted_field)?)) as FilterPair>::Filtered<$vec_generic::Output,optional_type!($($output_ty)?)>;

            #[inline]
            unsafe fn output(&mut self) -> Self::Output {
                <($vec_generic::OutputBool,is_present!($($outputted_field)?)) as FilterPair>::filter(self.$vec.output(),optional_expr!($(self.$outputted_field.output())?))
            }

            #[inline]
            unsafe fn drop_output(&mut self) {
                self.$vec.drop_output();
                $(self.$outputted_field.output();)?
            }
        }

        unsafe impl<$($($lifetime),+,)? $vec_generic: VectorLike $(,$($generic $(: $fst_generic_bound $(+ $generic_bound)*)?),+)?> HasReuseBuf for $struct<$($($lifetime),+,)? $vec_generic $(,$($generic),+)?> 
        $(where $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {
            type FstHandleBool = <$vec_generic as HasReuseBuf>::FstHandleBool;
            type SndHandleBool = <$vec_generic as HasReuseBuf>::SndHandleBool;
            type BoundHandlesBool = <$vec_generic as HasReuseBuf>::BoundHandlesBool;
            type FstOwnedBufferBool = <$vec_generic as HasReuseBuf>::FstOwnedBufferBool;
            type SndOwnedBufferBool = <$vec_generic as HasReuseBuf>::SndOwnedBufferBool;
            type FstOwnedBuffer = <$vec_generic as HasReuseBuf>::FstOwnedBuffer;
            type SndOwnedBuffer = <$vec_generic as HasReuseBuf>::SndOwnedBuffer;
            type FstType = <$vec_generic as HasReuseBuf>::FstType;
            type SndType = <$vec_generic as HasReuseBuf>::SndType;
            type BoundTypes = <$vec_generic as HasReuseBuf>::BoundTypes;

            #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) {self.$vec.assign_1st_buf(index,val)}
            #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) {self.$vec.assign_2nd_buf(index,val)}
            #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) {self.$vec.assign_bound_bufs(index,val)}
            #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {self.$vec.get_1st_buffer()}
            #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.$vec.get_2nd_buffer()}
            #[inline] unsafe fn drop_1st_buf_index(&mut self,index: usize) {self.$vec.drop_1st_buf_index(index)}
            #[inline] unsafe fn drop_2nd_buf_index(&mut self,index: usize) {self.$vec.drop_2nd_buf_index(index)}
            #[inline] unsafe fn drop_bound_bufs_index(&mut self,index: usize) {self.$vec.drop_bound_bufs_index(index)}
        }
    };
    (
        $struct:ident<$($($lifetime:lifetime),+,)? {$l_vec_generic:ident,$r_vec_generic:ident} $(,$($generic:ident $(: $($generic_lifetime:lifetime |)? $fst_generic_bound:path $(| $generic_bound:path)*)?),+)?>{$l_vec:ident, $r_vec:ident $(,$($field:ident: $field_ty:ty),+)?}
        $(where $($bound_ty:ty: $fst_where_bound:path $(| $where_bound:path)*),+)?;
        $(output: $outputted_field:ident: $output_ty:ty,)?
        get: $item:ty, |$self:ident,$(($l_is_mut:tt))? $l_input:ident,$(($r_is_mut:tt))? $r_input:ident| $get_expr:expr $(, $is_repeatable:ty)?
    ) => {
        pub struct $struct<$($($lifetime),+,)? $l_vec_generic: VectorLike, $r_vec_generic: VectorLike $(,$($generic $(: $($generic_lifetime +)? $fst_generic_bound $(+ $generic_bound)*)?),+)?> $(where $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {pub(crate) $l_vec: $l_vec_generic, pub(crate) $r_vec: $r_vec_generic $(,$(pub(crate) $field: $field_ty),+)?}

        impl<$($($lifetime),+,)? $l_vec_generic: VectorLike, $r_vec_generic: VectorLike $(,$($generic $(: $($generic_lifetime +)? $fst_generic_bound $(+ $generic_bound)*)?),+)?> Get for $struct<$($($lifetime),+,)? $l_vec_generic, $r_vec_generic $(,$($generic),+)?> where ($l_vec_generic::BoundHandlesBool, $r_vec_generic::BoundHandlesBool): FilterPair $(, $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {
            type GetBool = is_unit!($item);
            type IsRepeatable = is_present!($($is_repeatable)?); 
            type Inputs = ($l_vec_generic::Inputs,$r_vec_generic::Inputs);
            type Item = $item;
            type BoundItems = <($l_vec_generic::BoundHandlesBool, $r_vec_generic::BoundHandlesBool) as FilterPair>::Filtered<$l_vec_generic::BoundItems,$r_vec_generic::BoundItems>;

            #[inline]
            unsafe fn get_inputs(&mut self,index: usize) -> Self::Inputs {(self.$l_vec.get_inputs(index),self.$r_vec.get_inputs(index))}

            #[inline]
            unsafe fn drop_inputs(&mut self,index: usize) {
                self.$l_vec.drop_inputs(index);
                self.$r_vec.drop_inputs(index);
            }

            #[inline]
            fn process($self: &mut Self,inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
                let ($($l_is_mut)? $l_input,l_bound_items) = $self.$l_vec.process(inputs.0);
                let ($($r_is_mut)? $r_input,r_bound_items) = $self.$r_vec.process(inputs.1);
                ($get_expr,<($l_vec_generic::BoundHandlesBool, $r_vec_generic::BoundHandlesBool) as FilterPair>::filter(l_bound_items,r_bound_items))
            }
        }
    
        impl<$($($lifetime),+,)? $l_vec_generic: VectorLike, $r_vec_generic: VectorLike $(,$($generic $(: $($generic_lifetime +)? $fst_generic_bound $(+ $generic_bound)*)?),+)?> HasOutput for $struct<$($($lifetime),+,)? $l_vec_generic, $r_vec_generic $(,$($generic),+)?> where ($l_vec_generic::OutputBool,$r_vec_generic::OutputBool): FilterPair, (<($l_vec_generic::OutputBool,$r_vec_generic::OutputBool) as TyBoolPair>::Or,is_present!($($outputted_field)?)): FilterPair $(, $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {
            type OutputBool = optimized_or!(<($l_vec_generic::OutputBool,$r_vec_generic::OutputBool) as TyBoolPair>::Or,$($outputted_field)?);
            type Output = <(<($l_vec_generic::OutputBool,$r_vec_generic::OutputBool) as TyBoolPair>::Or,is_present!($($outputted_field)?)) as FilterPair>::Filtered<<($l_vec_generic::OutputBool,$r_vec_generic::OutputBool) as FilterPair>::Filtered<$l_vec_generic::Output,$r_vec_generic::Output>,optional_type!($($output_ty)?)>;
        
            #[inline]
            unsafe fn output(&mut self) -> Self::Output {
                <(<($l_vec_generic::OutputBool,$r_vec_generic::OutputBool) as TyBoolPair>::Or,is_present!($($outputted_field)?)) as FilterPair>::filter(
                    <($l_vec_generic::OutputBool,$r_vec_generic::OutputBool) as FilterPair>::filter(self.$l_vec.output(),self.$r_vec.output()),
                    optional_expr!($(self.$outputted_field.output())?)
                )
            }

            #[inline]
            unsafe fn drop_output(&mut self) {
                self.$l_vec.drop_output();
                self.$r_vec.drop_output();
                $(self.$outputted_field.output();)?
            }
        }

        unsafe impl<$($($lifetime),+,)? $l_vec_generic: VectorLike, $r_vec_generic: VectorLike $(,$($generic $(: $($generic_lifetime +)? $fst_generic_bound $(+ $generic_bound)*)?),+)?> HasReuseBuf for $struct<$($($lifetime),+,)? $l_vec_generic, $r_vec_generic $(,$($generic),+)?> 
        where 
            (<$l_vec_generic as HasReuseBuf>::FstOwnedBufferBool, <$r_vec_generic as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
            (<$l_vec_generic as HasReuseBuf>::SndOwnedBufferBool, <$r_vec_generic as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
            (<$l_vec_generic as HasReuseBuf>::FstHandleBool, <$r_vec_generic as HasReuseBuf>::FstHandleBool): SelectPair,
            (<$l_vec_generic as HasReuseBuf>::SndHandleBool, <$r_vec_generic as HasReuseBuf>::SndHandleBool): SelectPair,
            (<$l_vec_generic as HasReuseBuf>::BoundHandlesBool, <$r_vec_generic as HasReuseBuf>::BoundHandlesBool): FilterPair
            $(, $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)?
        {
            type FstHandleBool = <(<$l_vec_generic as HasReuseBuf>::FstHandleBool, <$r_vec_generic as HasReuseBuf>::FstHandleBool) as TyBoolPair>::Xor;
            type SndHandleBool = <(<$l_vec_generic as HasReuseBuf>::SndHandleBool, <$r_vec_generic as HasReuseBuf>::SndHandleBool) as TyBoolPair>::Xor;
            type BoundHandlesBool = <(<$l_vec_generic as HasReuseBuf>::BoundHandlesBool, <$r_vec_generic as HasReuseBuf>::BoundHandlesBool) as TyBoolPair>::Or;
            type FstOwnedBufferBool = <(<$l_vec_generic as HasReuseBuf>::FstOwnedBufferBool, <$r_vec_generic as HasReuseBuf>::FstOwnedBufferBool) as TyBoolPair>::Xor; 
            type SndOwnedBufferBool = <(<$l_vec_generic as HasReuseBuf>::SndOwnedBufferBool, <$r_vec_generic as HasReuseBuf>::SndOwnedBufferBool) as TyBoolPair>::Xor; 
            type FstOwnedBuffer = <(<$l_vec_generic as HasReuseBuf>::FstOwnedBufferBool, <$r_vec_generic as HasReuseBuf>::FstOwnedBufferBool) as SelectPair>::Selected<<$l_vec_generic as HasReuseBuf>::FstOwnedBuffer,<$r_vec_generic as HasReuseBuf>::FstOwnedBuffer>;
            type SndOwnedBuffer = <(<$l_vec_generic as HasReuseBuf>::SndOwnedBufferBool, <$r_vec_generic as HasReuseBuf>::SndOwnedBufferBool) as SelectPair>::Selected<<$l_vec_generic as HasReuseBuf>::SndOwnedBuffer,<$r_vec_generic as HasReuseBuf>::SndOwnedBuffer>;
            type FstType = <(<$l_vec_generic as HasReuseBuf>::FstHandleBool, <$r_vec_generic as HasReuseBuf>::FstHandleBool) as SelectPair>::Selected<<$l_vec_generic as HasReuseBuf>::FstType,<$r_vec_generic as HasReuseBuf>::FstType>;
            type SndType = <(<$l_vec_generic as HasReuseBuf>::SndHandleBool, <$r_vec_generic as HasReuseBuf>::SndHandleBool) as SelectPair>::Selected<<$l_vec_generic as HasReuseBuf>::SndType,<$r_vec_generic as HasReuseBuf>::SndType>;
            type BoundTypes = <(<$l_vec_generic as HasReuseBuf>::BoundHandlesBool, <$r_vec_generic as HasReuseBuf>::BoundHandlesBool) as FilterPair>::Filtered<<$l_vec_generic as HasReuseBuf>::BoundTypes,<$r_vec_generic as HasReuseBuf>::BoundTypes>;
        
            #[inline] unsafe fn assign_1st_buf<'z>(&'z mut self,index: usize,val: Self::FstType) {
                let (l_val,r_val) = <(<$l_vec_generic as HasReuseBuf>::FstHandleBool, <$r_vec_generic as HasReuseBuf>::FstHandleBool) as SelectPair>::deselect(val);
                self.$l_vec.assign_1st_buf(index,l_val);
                self.$r_vec.assign_1st_buf(index,r_val);
            }
            #[inline] unsafe fn assign_2nd_buf<'z>(&'z mut self,index: usize,val: Self::SndType) {
                let (l_val,r_val) = <(<$l_vec_generic as HasReuseBuf>::SndHandleBool, <$r_vec_generic as HasReuseBuf>::SndHandleBool) as SelectPair>::deselect(val);
                self.$l_vec.assign_2nd_buf(index,l_val);
                self.$r_vec.assign_2nd_buf(index,r_val);
            }
            #[inline] unsafe fn assign_bound_bufs<'z>(&'z mut self,index: usize,val: Self::BoundTypes) {
                let (l_val,r_val) = <(<$l_vec_generic as HasReuseBuf>::BoundHandlesBool, <$r_vec_generic as HasReuseBuf>::BoundHandlesBool) as FilterPair>::defilter(val);
                self.$l_vec.assign_bound_bufs(index,l_val);
                self.$r_vec.assign_bound_bufs(index,r_val);
            }
            #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
                <(<$l_vec_generic as HasReuseBuf>::FstOwnedBufferBool, <$r_vec_generic as HasReuseBuf>::FstOwnedBufferBool) as SelectPair>::select(self.$l_vec.get_1st_buffer(),self.$r_vec.get_1st_buffer())
            }
            #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {
                <(<$l_vec_generic as HasReuseBuf>::SndOwnedBufferBool, <$r_vec_generic as HasReuseBuf>::SndOwnedBufferBool) as SelectPair>::select(self.$l_vec.get_2nd_buffer(),self.$r_vec.get_2nd_buffer())
            }
            #[inline] unsafe fn drop_1st_buf_index(&mut self,index: usize) {
                self.$l_vec.drop_1st_buf_index(index);
                self.$r_vec.drop_1st_buf_index(index);
            }
            #[inline] unsafe fn drop_2nd_buf_index(&mut self,index: usize) {
                self.$l_vec.drop_2nd_buf_index(index);
                self.$r_vec.drop_2nd_buf_index(index);
            }
            #[inline] unsafe fn drop_bound_bufs_index(&mut self,index: usize) {
                self.$l_vec.drop_bound_bufs_index(index);
                self.$r_vec.drop_bound_bufs_index(index);
            }
        }
    }
}

vec_struct!(VecMap<{T},F: FnMut(T::Item) -> O,O>{vec, f: F}; get: O, |self,input| (self.f)(input));
vec_struct!(VecFold<{T},F: FnMut(O,T::Item) -> O,O>{vec, f: F, cell: Option<O>}; output: cell: O, get: (), |self,input| self.cell = Some((self.f)(self.cell.take().unwrap(),input)));
vec_struct!(VecFoldRef<{T},F: FnMut(&mut O,T::Item),O>{vec, f: F, cell: ManuallyDrop<O>}; output: cell: O, get: (), |self,input| (self.f)(&mut self.cell,input)); // note: use of this is preferred to VecFold

vec_struct!(VecCopiedFold<{T},F: FnMut(O,T::Item) -> O,O>{vec, f: F, cell: Option<O>} where T::Item: Copy; output: cell: O, get: T::Item, |self,input| {self.cell = Some((self.f)(self.cell.take().unwrap(),input)); input});
vec_struct!(VecCopiedFoldRef<{T},F: FnMut(&mut O,T::Item),O>{vec, f: F, cell: ManuallyDrop<O>} where T::Item: Copy; output: cell: O, get: T::Item, |self,input| {(self.f)(&mut self.cell,input); input}); // note: use of this is preferred to VecFold

vec_struct!(VecCopy<'a,{T},I: 'a | Copy>{vec} where T: Get<Item = &'a I>; get: I, |self,input| *input, Y);
vec_struct!(VecClone<'a,{T},I: 'a | Clone>{vec} where T: Get<Item = &'a I>; get: I, |self,input| input.clone());

vec_struct!(VecNeg<{T}>{vec} where T::Item: Neg; get: <T::Item as Neg>::Output, |self,input| -input);

vec_struct!(VecMulR<{T},S: Copy>{vec,scalar: S} where S: Mul<T::Item>; get: <S as Mul<T::Item>>::Output, |self,input| self.scalar * input);
vec_struct!(VecDivR<{T},S: Copy>{vec,scalar: S} where S: Div<T::Item>; get: <S as Div<T::Item>>::Output, |self,input| self.scalar / input);
vec_struct!(VecRemR<{T},S: Copy>{vec,scalar: S} where S: Rem<T::Item>; get: <S as Rem<T::Item>>::Output, |self,input| self.scalar % input);
vec_struct!(VecMulL<{T},S: Copy>{vec,scalar: S} where T::Item: Mul<S>; get: <T::Item as Mul<S>>::Output, |self,input| input * self.scalar);
vec_struct!(VecDivL<{T},S: Copy>{vec,scalar: S} where T::Item: Div<S>; get: <T::Item as Div<S>>::Output, |self,input| input / self.scalar);
vec_struct!(VecRemL<{T},S: Copy>{vec,scalar: S} where T::Item: Rem<S>; get: <T::Item as Rem<S>>::Output, |self,input| input % self.scalar);

vec_struct!(VecMulAssign<'a,{T},I: 'a | MulAssign<S>,S: Copy>{vec,scalar: S} where T: Get<Item = &'a mut I>; get: (), |self, input| *input *= self.scalar);
vec_struct!(VecDivAssign<'a,{T},I: 'a | DivAssign<S>,S: Copy>{vec,scalar: S} where T: Get<Item = &'a mut I>; get: (), |self, input| *input /= self.scalar);
vec_struct!(VecRemAssign<'a,{T},I: 'a | RemAssign<S>,S: Copy>{vec,scalar: S} where T: Get<Item = &'a mut I>; get: (), |self, input| *input %= self.scalar);

vec_struct!(VecSum<{T},S>{vec,scalar: ManuallyDrop<S>} where S: AddAssign<T::Item>; output: scalar: S, get: (), |self,input| *self.scalar += input);
vec_struct!(VecProduct<{T},S>{vec,scalar: ManuallyDrop<S>} where S: MulAssign<T::Item>; output: scalar: S, get: (), |self,input| *self.scalar *= input);
vec_struct!(VecSqrMag<{T},S>{vec,scalar: ManuallyDrop<S>} where T::Item: Copy | Mul, S: AddAssign<<T::Item as Mul>::Output>; output: scalar: S, get: (), |self,input| *self.scalar += input*input);

vec_struct!(VecCopiedSum<{T},S>{vec,scalar: ManuallyDrop<S>} where T::Item: Copy, S: AddAssign<T::Item>; output: scalar: S, get: T::Item, |self,input| {*self.scalar += input; input});
vec_struct!(VecCopiedProduct<{T},S>{vec,scalar: ManuallyDrop<S>} where T::Item: Copy, S: MulAssign<T::Item>; output: scalar: S, get: T::Item, |self,input| {*self.scalar *= input; input});
vec_struct!(VecCopiedSqrMag<{T},S>{vec,scalar: ManuallyDrop<S>} where T::Item: Copy | Mul, S: AddAssign<<T::Item as Mul>::Output>; output: scalar: S, get: T::Item, |self,input| {*self.scalar += input*input; input});


vec_struct!(VecZip<{T1,T2}>{l_vec,r_vec}; get: (T1::Item,T2::Item), |self,l_input,r_input| (l_input,r_input), Y);

vec_struct!(VecAdd<{T1,T2}>{l_vec,r_vec} where T1::Item: Add<T2::Item>; get: <T1::Item as Add<T2::Item>>::Output, |self,l_input,r_input| l_input + r_input);
vec_struct!(VecSub<{T1,T2}>{l_vec,r_vec} where T1::Item: Sub<T2::Item>; get: <T1::Item as Sub<T2::Item>>::Output, |self,l_input,r_input| l_input - r_input);
vec_struct!(VecCompMul<{T1,T2}>{l_vec,r_vec} where T1::Item: Mul<T2::Item>; get: <T1::Item as Mul<T2::Item>>::Output, |self,l_input,r_input| l_input * r_input);
vec_struct!(VecCompDiv<{T1,T2}>{l_vec,r_vec} where T1::Item: Div<T2::Item>; get: <T1::Item as Div<T2::Item>>::Output, |self,l_input,r_input| l_input / r_input);
vec_struct!(VecCompRem<{T1,T2}>{l_vec,r_vec} where T1::Item: Rem<T2::Item>; get: <T1::Item as Rem<T2::Item>>::Output, |self,l_input,r_input| l_input % r_input);

vec_struct!(VecAddAssign<'a,{T1,T2},I: 'a | AddAssign<T2::Item>>{l_vec,r_vec} where T1: Get<Item = &'a mut I>; get: (), |self,l_input,r_input| *l_input += r_input);
vec_struct!(VecSubAssign<'a,{T1,T2},I: 'a | SubAssign<T2::Item>>{l_vec,r_vec} where T1: Get<Item = &'a mut I>; get: (), |self,l_input,r_input| *l_input -= r_input);
vec_struct!(VecCompMulAssign<'a,{T1,T2},I: 'a | MulAssign<T2::Item>>{l_vec,r_vec} where T1: Get<Item = &'a mut I>; get: (), |self,l_input,r_input| *l_input *= r_input);
vec_struct!(VecCompDivAssign<'a,{T1,T2},I: 'a | DivAssign<T2::Item>>{l_vec,r_vec} where T1: Get<Item = &'a mut I>; get: (), |self,l_input,r_input| *l_input /= r_input);
vec_struct!(VecCompRemAssign<'a,{T1,T2},I: 'a | RemAssign<T2::Item>>{l_vec,r_vec} where T1: Get<Item = &'a mut I>; get: (), |self,l_input,r_input| *l_input %= r_input);

vec_struct!(VecDot<{T1,T2},S>{l_vec,r_vec,scalar: ManuallyDrop<S>} where T1::Item: Mul<T2::Item>, S: AddAssign<<T1::Item as Mul<T2::Item>>::Output>; output: scalar: S, get: (), |self,l_input,r_input| *self.scalar += l_input * r_input);

