use crate::trait_specialization_utils::*;
use crate::util_traits::*;
use crate::vector::vec_util_traits::*;
use crate::vector::MathVector;
use crate::vector::VectorExpr;
use super::OwnedArray;
use std::mem::ManuallyDrop;

/// an owned array which additionally acts as an buffer for HasReuseBuf (in the first slot)
#[repr(transparent)]
pub struct ReplaceArray<T, const D: usize>(pub(crate) ManuallyDrop<[T; D]>);

unsafe impl<T, const D: usize> Get for ReplaceArray<T, D> {
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

impl<T: Sized, const D: usize> HasOutput for ReplaceArray<T, D> {
    type OutputBool = N;
    type Output = ();

    #[inline]
    unsafe fn output(&mut self) -> Self::Output {}

    #[inline]
    unsafe fn drop_output(&mut self) {} // dropped through reuse buf instead
}

impl<T, const D: usize> HasReuseBuf for ReplaceArray<T, D> {
    type FstHandleBool = Y;
    type SndHandleBool = N;
    type BoundHandlesBool = N;
    type FstOwnedBufferBool = Y;
    type SndOwnedBufferBool = N;
    type FstOwnedBuffer = MathVector<T, D>;
    type SndOwnedBuffer = ();
    type FstType = T;
    type SndType = ();
    type BoundTypes = ();

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) { unsafe {std::ptr::write(self.0.get_unchecked_mut(index), val)}}
    #[inline] unsafe fn assign_2nd_buf(&mut self, _: usize, _: Self::SndType) {}
    #[inline] unsafe fn assign_bound_bufs(&mut self, _: usize, _: Self::BoundTypes) {}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {
        VectorExpr(OwnedArray(std::ptr::read(&self.0)))
    }}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {}
    #[inline] unsafe fn drop_1st_buffer(&mut self) {}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) {}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) { unsafe {std::ptr::drop_in_place(self.0.get_unchecked_mut(index))}}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, _: usize) {}
}


/// struct attaching an array to a VectorLike to be used as a HasReuseBuf buffer (first slot)
pub struct VecAttachArray<'a, V: VectorLike<FstHandleBool = N>, T, const D: usize>{pub(crate) vec: V, pub(crate) buf: &'a mut [T; D]} 

unsafe impl<'a, V: VectorLike<FstHandleBool = N>, T, const D: usize> Get for VecAttachArray<'a, V, T, D> {
    type GetBool = V::GetBool;
    type Inputs = V::Inputs;
    type Item = V::Item;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {self.vec.get_inputs(index)}}

    #[inline] unsafe fn drop_inputs(&mut self, index: usize) { unsafe {self.vec.drop_inputs(index)}}

    #[inline] fn process(&mut self, index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(index, inputs)}
}

unsafe impl<'a, V: IsRepeatable + VectorLike<FstHandleBool = N>, T, const D: usize> IsRepeatable for VecAttachArray<'a, V, T, D> {}

impl<'a, V: VectorLike<FstHandleBool = N>, T, const D: usize> HasOutput for VecAttachArray<'a, V, T, D> {
    type OutputBool = V::OutputBool;
    type Output = V::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.vec.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.vec.drop_output()}}
}

impl<'b, V: VectorLike<FstHandleBool = N>, T, const D: usize> HasReuseBuf for VecAttachArray<'b, V, T, D> {
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

/// struct attaching an initially uninitiallized array to a VectorLike to be used as a HasReuseBuf buffer (first slot)
pub struct VecCreateArray<V: VectorLike<FstHandleBool = N>, T, const D: usize>{pub(crate) vec: V, pub(crate) buf: [std::mem::MaybeUninit<T>; D]}

unsafe impl<V: VectorLike<FstHandleBool = N>, T, const D: usize> Get for VecCreateArray<V, T, D> {
    type GetBool = V::GetBool;
    type Inputs = V::Inputs;
    type Item = V::Item;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {self.vec.get_inputs(index)}}

    #[inline] unsafe fn drop_inputs(&mut self, index: usize) { unsafe {self.vec.drop_inputs(index)}}

    #[inline] fn process(&mut self, index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(index, inputs)}
}

unsafe impl<V: IsRepeatable + VectorLike<FstHandleBool = N>, T, const D: usize> IsRepeatable for VecCreateArray<V, T, D> {}

impl<V: VectorLike<FstHandleBool = N>, T, const D: usize> HasOutput for VecCreateArray<V, T, D> {
    type OutputBool = V::OutputBool;
    type Output = V::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.vec.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.vec.drop_output()}}
}

impl<V: VectorLike<FstHandleBool = N>, T, const D: usize> HasReuseBuf for VecCreateArray<V, T, D> {
    type FstHandleBool = Y;
    type SndHandleBool = V::SndHandleBool;
    type BoundHandlesBool = V::BoundHandlesBool;
    type FstOwnedBufferBool = Y;
    type SndOwnedBufferBool = V::SndOwnedBufferBool;
    type FstOwnedBuffer = MathVector<T, D>;
    type SndOwnedBuffer = V::SndOwnedBuffer;
    type FstType = T;
    type SndType = V::SndType;
    type BoundTypes = V::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) { unsafe {std::ptr::write(self.buf.get_unchecked_mut(index), std::mem::MaybeUninit::new(val))}}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) { unsafe {self.vec.assign_2nd_buf(index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) { unsafe {self.vec.assign_bound_bufs(index, val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {
        VectorExpr(OwnedArray(std::mem::transmute_copy::<[std::mem::MaybeUninit<T>; D], ManuallyDrop<[T; D]>>(&self.buf)))
    }}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.vec.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buffer(&mut self) {}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) { unsafe {self.vec.drop_2nd_buffer();}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) { unsafe {self.buf.get_unchecked_mut(index).assume_init_drop()}}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_2nd_buf_index(index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {self.vec.drop_bound_bufs_index(index)}}
}

/// struct attaching an initially uninitiallized array (on the heap) to a VectorLike to be used as a HasReuseBuf buffer (first slot)
pub struct VecCreateHeapArray<V: VectorLike<FstHandleBool = N>, T, const D: usize>{pub(crate) vec: V, pub(crate) buf: std::mem::ManuallyDrop<Box<[std::mem::MaybeUninit<T>; D]>>}

unsafe impl<V: VectorLike<FstHandleBool = N>, T, const D: usize> Get for VecCreateHeapArray<V, T, D> {
    type GetBool = V::GetBool;
    type Inputs = V::Inputs;
    type Item = V::Item;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {self.vec.get_inputs(index)}}

    #[inline] unsafe fn drop_inputs(&mut self, index: usize) { unsafe {self.vec.drop_inputs(index)}}

    #[inline] fn process(&mut self, index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(index, inputs)}
}

unsafe impl<V: IsRepeatable + VectorLike<FstHandleBool = N>, T, const D: usize> IsRepeatable for VecCreateHeapArray<V, T, D> {}

impl<V: VectorLike<FstHandleBool = N>, T, const D: usize> HasOutput for VecCreateHeapArray<V, T, D> {
    type OutputBool = V::OutputBool;
    type Output = V::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.vec.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.vec.drop_output()}}
}

impl<V: VectorLike<FstHandleBool = N>, T, const D: usize> HasReuseBuf for VecCreateHeapArray<V, T, D> {
    type FstHandleBool = Y;
    type SndHandleBool = V::SndHandleBool;
    type BoundHandlesBool = V::BoundHandlesBool;
    type FstOwnedBufferBool = Y;
    type SndOwnedBufferBool = V::SndOwnedBufferBool;
    type FstOwnedBuffer = Box<MathVector<T, D>>;
    type SndOwnedBuffer = V::SndOwnedBuffer;
    type FstType = T;
    type SndType = V::SndType;
    type BoundTypes = V::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) { unsafe {std::ptr::write(self.buf.get_unchecked_mut(index), std::mem::MaybeUninit::new(val))}}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) { unsafe {self.vec.assign_2nd_buf(index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) { unsafe {self.vec.assign_bound_bufs(index, val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {
        std::mem::transmute_copy::<Box<[std::mem::MaybeUninit<T>; D]>, Box<MathVector<T, D>>>(&self.buf)
    }}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.vec.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buffer(&mut self) { unsafe {ManuallyDrop::drop(&mut self.buf);}}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) { unsafe {self.vec.drop_2nd_buffer();}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) { unsafe {self.buf.get_unchecked_mut(index).assume_init_drop()}}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_2nd_buf_index(index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {self.vec.drop_bound_bufs_index(index)}}
}

/// struct attaching an initially uninitiallized array to a VectorLike to be used as a HasReuseBuf buffer if there isn't already a buffer (first slot)
pub struct VecMaybeCreateArray<V: VectorLike, T, const D: usize> where <V::FstHandleBool as TyBool>::Neg: Filter {pub(crate) vec: V, pub(crate) buf: [std::mem::MaybeUninit<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>; D]}

unsafe impl<V: VectorLike, T, const D: usize> Get for VecMaybeCreateArray<V, T, D> where <V::FstHandleBool as TyBool>::Neg: Filter {
    type GetBool = V::GetBool;
    type Inputs = V::Inputs;
    type Item = V::Item;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {self.vec.get_inputs(index)}}

    #[inline] unsafe fn drop_inputs(&mut self, index: usize) { unsafe {self.vec.drop_inputs(index)}}

    #[inline] fn process(&mut self, index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(index, inputs)}
}

unsafe impl<V: IsRepeatable + VectorLike, T, const D: usize> IsRepeatable for VecMaybeCreateArray<V, T, D> where <V::FstHandleBool as TyBool>::Neg: Filter {}

impl<V: VectorLike, T, const D: usize> HasOutput for VecMaybeCreateArray<V, T, D> where <V::FstHandleBool as TyBool>::Neg: Filter {
    type OutputBool = V::OutputBool;
    type Output = V::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.vec.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.vec.drop_output()}}
}

impl<V: VectorLike, T, const D: usize> HasReuseBuf for VecMaybeCreateArray<V, T, D> 
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
    type FstOwnedBuffer = <(V::FstHandleBool, <V::FstHandleBool as TyBool>::Neg) as SelectPair>::Selected<V::FstOwnedBuffer, MathVector<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>, D>>;
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
        <(V::FstHandleBool, <V::FstHandleBool as TyBool>::Neg) as SelectPair>::select(
            self.vec.get_1st_buffer(),
            VectorExpr(OwnedArray(ManuallyDrop::new(std::mem::transmute_copy::<[std::mem::MaybeUninit<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>; D], [<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>; D]>(&self.buf))))
        )
    }}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.vec.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buffer(&mut self) {}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) { unsafe {self.vec.drop_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) { unsafe {
        self.vec.drop_1st_buf_index(index);
        // assuming that its safe to drop () even if we never properlly "initiallized" them
        self.buf.get_unchecked_mut(index).assume_init_drop();
    }}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_2nd_buf_index(index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {self.vec.drop_bound_bufs_index(index)}}
}

/// struct attaching an initially uninitiallized array (on the heap) to a VectorLike to be used as a HasReuseBuf buffer if there isn't already a buffer (first slot)
pub struct VecMaybeCreateHeapArray<V: VectorLike, T, const D: usize> where <V::FstHandleBool as TyBool>::Neg: Filter {pub(crate) vec: V, pub(crate) buf: std::mem::ManuallyDrop<Box<[std::mem::MaybeUninit<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>; D]>>}

unsafe impl<V: VectorLike, T, const D: usize> Get for VecMaybeCreateHeapArray<V, T, D> where <V::FstHandleBool as TyBool>::Neg: Filter {
    type GetBool = V::GetBool;
    type Inputs = V::Inputs;
    type Item = V::Item;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {self.vec.get_inputs(index)}}

    #[inline] unsafe fn drop_inputs(&mut self, index: usize) { unsafe {self.vec.drop_inputs(index)}}

    #[inline] fn process(&mut self, index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(index, inputs)}
}

unsafe impl<V: IsRepeatable + VectorLike, T, const D: usize> IsRepeatable for VecMaybeCreateHeapArray<V, T, D> where <V::FstHandleBool as TyBool>::Neg: Filter {}

impl<V: VectorLike, T, const D: usize> HasOutput for VecMaybeCreateHeapArray<V, T, D> where <V::FstHandleBool as TyBool>::Neg: Filter {
    type OutputBool = V::OutputBool;
    type Output = V::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.vec.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.vec.drop_output()}}
}

impl<V: VectorLike, T, const D: usize> HasReuseBuf for VecMaybeCreateHeapArray<V, T, D> 
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
    type FstOwnedBuffer = <(V::FstHandleBool, <V::FstHandleBool as TyBool>::Neg) as SelectPair>::Selected<V::FstOwnedBuffer, Box<MathVector<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>, D>>>;
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
        <(V::FstHandleBool, <V::FstHandleBool as TyBool>::Neg) as SelectPair>::select(
            self.vec.get_1st_buffer(),
            std::mem::transmute_copy::<Box<[std::mem::MaybeUninit<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>>; D]>, Box<MathVector<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<T>, D>>>(&self.buf)
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
