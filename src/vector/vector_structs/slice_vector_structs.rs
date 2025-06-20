use crate::trait_specialization_utils::*;
use crate::util_traits::*;
use crate::vector::vec_util_traits::*;
use crate::vector::RSMathVector;
use crate::vector::RSVectorExpr;
use super::OwnedSlice;
use std::mem::{ManuallyDrop, MaybeUninit};

/// an owned slice which additionally acts as an buffer for HasReuseBuf (in the first slot)
#[repr(transparent)]
pub struct ReplaceSlice<T>(pub(crate) ManuallyDrop<Box<ManuallyDrop<[T]>>>);

unsafe impl<T> Get for ReplaceSlice<T> {
    type GetBool = Y;
    type Inputs = T;
    type Item = T;
    type BoundItems = ();


    #[inline]
    unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {
        std::ptr::read(self.0.get_unchecked(index))
    }}

    #[inline]
    unsafe fn drop_inputs(&mut self, index: usize) { unsafe {
        std::ptr::drop_in_place(self.0.get_unchecked_mut(index))
    }}

    #[inline]
    fn process(&mut self, _: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(inputs, ())}
}

impl<T: Sized> HasOutput for ReplaceSlice<T> {
    type OutputBool = N;
    type Output = ();

    #[inline]
    unsafe fn output(&mut self) -> Self::Output {}

    #[inline]
    unsafe fn drop_output(&mut self) {} // dropped through reuse buf instead
}

impl<T> HasReuseBuf for ReplaceSlice<T> {
    type FstHandleBool = Y;
    type SndHandleBool = N;
    type BoundHandlesBool = N;
    type FstOwnedBufferBool = Y;
    type SndOwnedBufferBool = N;
    type FstOwnedBuffer = RSMathVector<T>;
    type SndOwnedBuffer = ();
    type FstType = T;
    type SndType = ();
    type BoundTypes = ();

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) { unsafe {std::ptr::write(self.0.get_unchecked_mut(index), val)}}
    #[inline] unsafe fn assign_2nd_buf(&mut self, _: usize, _: Self::SndType) {}
    #[inline] unsafe fn assign_bound_bufs(&mut self, _: usize, _: Self::BoundTypes) {}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { 
        let size = self.0.len();
        unsafe { RSVectorExpr{vec: OwnedSlice(std::ptr::read(&*self.0)), size} }
    }
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {}
    #[inline] unsafe fn drop_1st_buffer(&mut self) {unsafe { ManuallyDrop::drop(&mut self.0) }}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) {}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) { unsafe {std::ptr::drop_in_place(self.0.get_unchecked_mut(index))}}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, _: usize) {}
}


/// struct attaching an slice to a VectorLike to be used as a HasReuseBuf buffer (first slot)
pub struct VecAttachSlice<'a, V: VectorLike<FstHandleBool = N>, T>{pub(crate) vec: V, pub(crate) buf: &'a mut [T]} 

unsafe impl<'a, V: VectorLike<FstHandleBool = N>, T> Get for VecAttachSlice<'a, V, T> {
    type GetBool = V::GetBool;
    type Inputs = V::Inputs;
    type Item = V::Item;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {self.vec.get_inputs(index)}}

    #[inline] unsafe fn drop_inputs(&mut self, index: usize) { unsafe {self.vec.drop_inputs(index)}}

    #[inline] fn process(&mut self, index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(index, inputs)}
}

unsafe impl<'a, V: IsRepeatable + VectorLike<FstHandleBool = N>, T> IsRepeatable for VecAttachSlice<'a, V, T> {}

impl<'a, V: VectorLike<FstHandleBool = N>, T> HasOutput for VecAttachSlice<'a, V, T> {
    type OutputBool = V::OutputBool;
    type Output = V::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.vec.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.vec.drop_output()}}
}

impl<'b, V: VectorLike<FstHandleBool = N>, T> HasReuseBuf for VecAttachSlice<'b, V, T> {
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

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) { unsafe {*self.buf.get_unchecked_mut(index) = val}}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) { unsafe {self.vec.assign_2nd_buf(index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) { unsafe {self.vec.assign_bound_bufs(index, val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.vec.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buffer(&mut self) {}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) { unsafe {self.vec.drop_2nd_buffer();}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_2nd_buf_index(index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {self.vec.drop_bound_bufs_index(index)}}
}


/// struct attaching an initially uninitiallized slice to a VectorLike to be used as a HasReuseBuf buffer (first slot)
pub struct VecCreateSlice<V: VectorLike<FstHandleBool = N>, T>{pub(crate) vec: V, pub(crate) buf: ManuallyDrop<Box<[MaybeUninit<T>]>>}

unsafe impl<V: VectorLike<FstHandleBool = N>, T> Get for VecCreateSlice<V, T> {
    type GetBool = V::GetBool;
    type Inputs = V::Inputs;
    type Item = V::Item;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {self.vec.get_inputs(index)}}

    #[inline] unsafe fn drop_inputs(&mut self, index: usize) { unsafe {self.vec.drop_inputs(index)}}

    #[inline] fn process(&mut self, index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(index, inputs)}
}

unsafe impl<V: IsRepeatable + VectorLike<FstHandleBool = N>, T> IsRepeatable for VecCreateSlice<V, T> {}

impl<V: VectorLike<FstHandleBool = N>, T> HasOutput for VecCreateSlice<V, T> {
    type OutputBool = V::OutputBool;
    type Output = V::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.vec.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.vec.drop_output()}}
}

impl<V: VectorLike<FstHandleBool = N>, T> HasReuseBuf for VecCreateSlice<V, T> {
    type FstHandleBool = Y;
    type SndHandleBool = V::SndHandleBool;
    type BoundHandlesBool = V::BoundHandlesBool;
    type FstOwnedBufferBool = Y;
    type SndOwnedBufferBool = V::SndOwnedBufferBool;
    type FstOwnedBuffer = RSMathVector<T>;
    type SndOwnedBuffer = V::SndOwnedBuffer;
    type FstType = T;
    type SndType = V::SndType;
    type BoundTypes = V::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) { unsafe {std::ptr::write(self.buf.get_unchecked_mut(index), std::mem::MaybeUninit::new(val))}}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) { unsafe {self.vec.assign_2nd_buf(index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) { unsafe {self.vec.assign_bound_bufs(index, val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {
        let size = self.buf.len();
        unsafe { RSVectorExpr{vec: std::mem::transmute_copy::<ManuallyDrop<Box<[MaybeUninit<T>]>>, OwnedSlice<T>>(&self.buf), size} }
    }
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.vec.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buffer(&mut self) { unsafe {ManuallyDrop::drop(&mut self.buf);}}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) { unsafe {self.vec.drop_2nd_buffer();}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) { unsafe {self.buf.get_unchecked_mut(index).assume_init_drop()}}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_2nd_buf_index(index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {self.vec.drop_bound_bufs_index(index)}}
}


/// struct attaching an initially uninitiallized array to a VectorLike to be used as a HasReuseBuf buffer if there isn't already a buffer (first slot)
pub struct VecMaybeCreateSlice<V: VectorLike, T> where <V::FstHandleBool as TyBool>::Neg: Filter {pub(crate) vec: V, pub(crate) buf: ManuallyDrop<Box<[MaybeUninit<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>]>>}

unsafe impl<V: VectorLike, T> Get for VecMaybeCreateSlice<V, T> where <V::FstHandleBool as TyBool>::Neg: Filter {
    type GetBool = V::GetBool;
    type Inputs = V::Inputs;
    type Item = V::Item;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {self.vec.get_inputs(index)}}

    #[inline] unsafe fn drop_inputs(&mut self, index: usize) { unsafe {self.vec.drop_inputs(index)}}

    #[inline] fn process(&mut self, index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(index, inputs)}
}

unsafe impl<V: IsRepeatable + VectorLike, T> IsRepeatable for VecMaybeCreateSlice<V, T> where <V::FstHandleBool as TyBool>::Neg: Filter {}

impl<V: VectorLike, T> HasOutput for VecMaybeCreateSlice<V, T> where <V::FstHandleBool as TyBool>::Neg: Filter {
    type OutputBool = V::OutputBool;
    type Output = V::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.vec.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.vec.drop_output()}}
}

impl<V: VectorLike, T> HasReuseBuf for VecMaybeCreateSlice<V, T> 
where 
    <V::FstHandleBool as TyBool>::Neg: Filter, 
    (V::FstHandleBool, <V::FstHandleBool as TyBool>::Neg): SelectPair,
    (V::FstOwnedBufferBool, <V::FstHandleBool as TyBool>::Neg): TyBoolPair
{
    type FstHandleBool = Y;
    type SndHandleBool = V::SndHandleBool;
    type BoundHandlesBool = V::BoundHandlesBool;
    type FstOwnedBufferBool = <(V::FstOwnedBufferBool, <V::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or;
    type SndOwnedBufferBool = V::SndOwnedBufferBool;
    type FstOwnedBuffer = <(V::FstHandleBool, <V::FstHandleBool as TyBool>::Neg) as SelectPair>::Selected<V::FstOwnedBuffer, RSMathVector<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>>;
    type SndOwnedBuffer = V::SndOwnedBuffer;
    type FstType = <(V::FstHandleBool, <V::FstHandleBool as TyBool>::Neg) as SelectPair>::Selected<V::FstType, <<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>;
    type SndType = V::SndType;
    type BoundTypes = V::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) { unsafe {
        let (init_val, attached_val) = <(V::FstHandleBool, <V::FstHandleBool as TyBool>::Neg) as SelectPair>::deselect(val);
        self.vec.assign_1st_buf(index, init_val);
        std::ptr::write(self.buf.get_unchecked_mut(index), std::mem::MaybeUninit::new(attached_val));
    }}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) { unsafe {self.vec.assign_2nd_buf(index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) { unsafe {self.vec.assign_bound_bufs(index, val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {
        let size = self.buf.len();
        <(V::FstHandleBool, <V::FstHandleBool as TyBool>::Neg) as SelectPair>::select(
            self.vec.get_1st_buffer(),
            
            RSVectorExpr{vec: std::mem::transmute_copy::<ManuallyDrop<Box<[MaybeUninit<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>]>>, OwnedSlice<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>>(&self.buf), size}
        )
    }}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.vec.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buffer(&mut self) { unsafe {ManuallyDrop::drop(&mut self.buf)}}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) { unsafe {self.vec.drop_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) { unsafe {
        self.vec.drop_1st_buf_index(index);
        // assuming that its safe to drop () even if we never properlly "initiallized" them
        self.buf.get_unchecked_mut(index).assume_init_drop();
    }}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_2nd_buf_index(index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {self.vec.drop_bound_bufs_index(index)}}
}
