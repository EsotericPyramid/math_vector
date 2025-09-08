use crate::{
    util_traits::*,
    vector::vec_util_traits::*,
    trait_specialization_utils::*,
};

/// Struct swapping the buffers (or lack there of) in the 2 slots
pub struct VecBufSwap<V: VectorLike> {pub(crate) vec: V}

unsafe impl<V: VectorLike> Get for VecBufSwap<V> {
    type GetBool = V::GetBool;
    type Inputs = V::Inputs;
    type Item = V::Item;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {self.vec.get_inputs(index)}}
    #[inline] unsafe fn drop_inputs(&mut self, index: usize) { unsafe {self.vec.drop_inputs(index)}}
    #[inline] fn process(&mut self, index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(index, inputs)}
}

unsafe impl<V: VectorLike + IsRepeatable> IsRepeatable for VecBufSwap<V> {}

impl<V: VectorLike> HasOutput for VecBufSwap<V> {
    type OutputBool = V::OutputBool;
    type Output = V::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.vec.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.vec.drop_output()}}
}

impl<V: VectorLike> HasReuseBuf for VecBufSwap<V> {
    type FstHandleBool = V::SndHandleBool;
    type SndHandleBool = V::FstHandleBool;
    type BoundHandlesBool = V::BoundHandlesBool;
    type FstOwnedBufferBool = V::SndOwnedBufferBool;
    type SndOwnedBufferBool = V::FstOwnedBufferBool;
    type FstOwnedBuffer = V::SndOwnedBuffer;
    type SndOwnedBuffer = V::FstOwnedBuffer;
    type FstType = V::SndType;
    type SndType = V::FstType;
    type BoundTypes = V::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) { unsafe {self.vec.assign_2nd_buf(index, val)}}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) { unsafe {self.vec.assign_1st_buf(index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) { unsafe {self.vec.assign_bound_bufs(index, val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {self.vec.get_2nd_buffer()}}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.vec.get_1st_buffer()}}
    #[inline] unsafe fn drop_1st_buffer(&mut self) { unsafe {self.vec.drop_2nd_buffer()}}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) { unsafe {self.vec.drop_1st_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_2nd_buf_index(index)}}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_1st_buf_index(index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {self.vec.drop_bound_bufs_index(index)}}
}

/// Struct attaching a used vector's output and buffers to another vector
/// SAFETY: it is expected that the used_vec field is safe to output in addition to normal correct implementation
pub struct VecAttachUsedVec<V: VectorLike, USEDV: VectorLike>{pub(crate) vec: V, pub(crate) used_vec: USEDV, pub(crate) size: usize}

unsafe impl<V: VectorLike, USEDV: VectorLike> Get for VecAttachUsedVec<V, USEDV> {
    type GetBool = V::GetBool;
    type Inputs = V::Inputs;
    type Item = V::Item;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {self.vec.get_inputs(index)}}
    #[inline] unsafe fn drop_inputs(&mut self, index: usize) { unsafe {self.vec.drop_inputs(index)}}
    #[inline] fn process(&mut self, index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(index, inputs)}
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
        for i in 0..self.size {
            self.used_vec.drop_bound_bufs_index(i);
        }
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
    #[inline] unsafe fn drop_1st_buffer(&mut self) { unsafe {
        self.vec.drop_1st_buffer();
        self.used_vec.drop_1st_buffer();
    }}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) { unsafe {
        self.vec.drop_2nd_buffer();
        self.used_vec.drop_2nd_buffer();
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
    }}
}

/// Struct stabilizing a vector so that it can be made dynamic
/// specifically, stores retrieved inputs internally so that externally input is always ()
pub struct DynamicVectorLike<V: VectorLike>{pub(crate) vec: V, pub(crate) inputs: Option<V::Inputs>}

unsafe impl<V: VectorLike> Get for DynamicVectorLike<V> {
    type GetBool = V::GetBool;
    type Inputs = ();
    type Item = V::Item;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {self.inputs = Some(self.vec.get_inputs(index));}}
    #[inline] unsafe fn drop_inputs(&mut self, index: usize) { unsafe {self.vec.drop_inputs(index)}}
    #[inline] fn process(&mut self, index: usize, _: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(index, self.inputs.take().unwrap())}
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
    #[inline] unsafe fn drop_1st_buffer(&mut self) { unsafe {self.vec.drop_1st_buffer()}}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) { unsafe {self.vec.drop_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_1st_buf_index(index)}}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_2nd_buf_index(index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {self.vec.drop_bound_bufs_index(index)}}
}


pub struct RepeatedVec<'a, V: VectorLike + IsRepeatable>{pub(crate) vec: &'a mut V}

unsafe impl<'a, V: VectorLike + IsRepeatable> Get for RepeatedVec<'a, V> {
    type GetBool = V::GetBool;
    type Inputs = V::Inputs;
    type Item = V::Item;
    type BoundItems = V::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {unsafe {self.vec.get_inputs(index)}}
    #[inline] unsafe fn drop_inputs(&mut self, index: usize) {unsafe {self.vec.drop_inputs(index)}}
    #[inline] fn process(&mut self, index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.vec.process(index, inputs)}
}

unsafe impl<'a, V: VectorLike + IsRepeatable> IsRepeatable for RepeatedVec<'a, V> {}

impl<'a, V: VectorLike + IsRepeatable> HasOutput for RepeatedVec<'a, V> {
    type OutputBool = V::OutputBool;
    type Output = V::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output {unsafe {self.vec.output()}}
    #[inline] unsafe fn drop_output(&mut self) {unsafe {self.vec.drop_output()}}
}

impl<'a, V: VectorLike + IsRepeatable> HasReuseBuf for RepeatedVec<'a, V> {
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
    #[inline] unsafe fn drop_1st_buffer(&mut self) { unsafe {self.vec.drop_1st_buffer()}}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) { unsafe {self.vec.drop_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_1st_buf_index(index)}}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_2nd_buf_index(index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {self.vec.drop_bound_bufs_index(index)}}
}