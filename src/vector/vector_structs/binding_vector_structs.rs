use crate::trait_specialization_utils::*;
use crate::util_traits::*;
use crate::vector::vec_util_traits::*;
use std::ops::*;

/// struct binding the vector's item to the buffer in the first slot which is linked to output
pub struct VecBind<V: VectorLike<FstHandleBool = Y>> {pub(crate) vec: V} 

unsafe impl<V: VectorLike<FstHandleBool = Y>> Get for VecBind<V> where (V::BoundHandlesBool, Y): FilterPair {
    type GetBool = N;
    type Inputs = V::Inputs;
    type Item = ();
    type BoundItems = <(V::BoundHandlesBool, Y) as FilterPair>::Filtered<V::BoundItems, V::Item>;

    #[inline]
    unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {
        self.vec.get_inputs(index)
    }}

    #[inline]
    unsafe fn drop_inputs(&mut self, index: usize) { unsafe {
        self.vec.drop_inputs(index)
    }}

    #[inline]
    fn process(&mut self, index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
        let (item, bound) = self.vec.process(index, inputs);
        ((), <(V::BoundHandlesBool, Y) as FilterPair>::filter(bound, item))
    }
}

impl<V: VectorLike<FstHandleBool = Y>> HasOutput for VecBind<V> where (V::OutputBool, V::FstOwnedBufferBool): FilterPair {
    type OutputBool = <(V::OutputBool, V::FstOwnedBufferBool) as TyBoolPair>::Or;
    type Output = <(V::OutputBool, V::FstOwnedBufferBool) as FilterPair>::Filtered<V::Output, V::FstOwnedBuffer>;
    
    #[inline]
    unsafe fn output(&mut self) -> Self::Output { unsafe {
        <(V::OutputBool, V::FstOwnedBufferBool) as FilterPair>::filter(self.vec.output(), self.vec.get_1st_buffer())
    }}

    #[inline]
    unsafe fn drop_output(&mut self) { unsafe {
        self.vec.drop_output(); // buffer dropped through HasReuseBuf
        self.vec.drop_1st_buffer();
    }}
}

impl<V: VectorLike<FstHandleBool = Y>> HasReuseBuf for VecBind<V> where (V::BoundHandlesBool, Y): FilterPair {
    type FstHandleBool = N;
    type SndHandleBool = V::SndHandleBool;
    type BoundHandlesBool = <(V::BoundHandlesBool, Y) as TyBoolPair>::Or;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = V::SndOwnedBufferBool;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = V::SndOwnedBuffer;
    type FstType = ();
    type SndType = V::SndType;
    type BoundTypes = <(V::BoundHandlesBool, Y) as FilterPair>::Filtered<V::BoundTypes, V::FstType>;

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) { unsafe {self.vec.assign_2nd_buf(index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) { unsafe {
        let (bounded_vals, new_bound_val) = <(V::BoundHandlesBool, Y) as FilterPair>::defilter(val);
        self.vec.assign_bound_bufs(index, bounded_vals);
        self.vec.assign_1st_buf(index, new_bound_val);
    }}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.vec.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buffer(&mut self) {}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) { unsafe {self.vec.drop_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_2nd_buf_index(index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {
        self.vec.drop_1st_buf_index(index);
        self.vec.drop_bound_bufs_index(index);
    }}
}

/// struct maping the vector's item and binding part of the output to the 
/// buffer in the first slot which is linked to output and the other part as the new item
pub struct VecMapBind<V: VectorLike<FstHandleBool = Y>, F: FnMut(V::Item) -> (I, B), I, B> {pub(crate) vec: V, pub(crate) f: F}

unsafe impl<V: VectorLike<FstHandleBool = Y>, F: FnMut(V::Item) -> (I, B), I, B> Get for VecMapBind<V, F, I, B> where (V::BoundHandlesBool, Y): FilterPair {
    type GetBool = Y;
    type Inputs = V::Inputs;
    type Item = I;
    type BoundItems = <(V::BoundHandlesBool, Y) as FilterPair>::Filtered<V::BoundItems, B>;

    #[inline]
    unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {
        self.vec.get_inputs(index)
    }}

    #[inline]
    unsafe fn drop_inputs(&mut self, index: usize) { unsafe {
        self.vec.drop_inputs(index)
    }}

    #[inline]
    fn process(&mut self, index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
        let (item, bound) = self.vec.process(index, inputs);
        let (processed_item, processed_bound) = (self.f)(item);
        (processed_item, <(V::BoundHandlesBool, Y) as FilterPair>::filter(bound, processed_bound))
    }
}

impl<V: VectorLike<FstHandleBool = Y>, F: FnMut(V::Item) -> (I, B), I, B> HasOutput for VecMapBind<V, F, I, B> where (V::OutputBool, V::FstOwnedBufferBool): FilterPair {
    type OutputBool = <(V::OutputBool, V::FstOwnedBufferBool) as TyBoolPair>::Or;
    type Output = <(V::OutputBool, V::FstOwnedBufferBool) as FilterPair>::Filtered<V::Output, V::FstOwnedBuffer>;
    
    #[inline]
    unsafe fn output(&mut self) -> Self::Output { unsafe {
        <(V::OutputBool, V::FstOwnedBufferBool) as FilterPair>::filter(self.vec.output(), self.vec.get_1st_buffer())
    }}

    #[inline]
    unsafe fn drop_output(&mut self) { unsafe {
        self.vec.drop_output(); // buffer dropped through HasReuseBuf
        self.vec.drop_1st_buffer();
    }}
}

impl<V: VectorLike<FstHandleBool = Y>, F: FnMut(V::Item) -> (I, B), I, B> HasReuseBuf for VecMapBind<V, F, I, B> where (V::BoundHandlesBool, Y): FilterPair {
    type FstHandleBool = N;
    type SndHandleBool = V::SndHandleBool;
    type BoundHandlesBool = <(V::BoundHandlesBool, Y) as TyBoolPair>::Or;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = V::SndOwnedBufferBool;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = V::SndOwnedBuffer;
    type FstType = ();
    type SndType = V::SndType;
    type BoundTypes = <(V::BoundHandlesBool, Y) as FilterPair>::Filtered<V::BoundTypes, V::FstType>;

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) { unsafe {self.vec.assign_2nd_buf(index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) { unsafe {
        let (bounded_vals, new_bound_val) = <(V::BoundHandlesBool, Y) as FilterPair>::defilter(val);
        self.vec.assign_bound_bufs(index, bounded_vals);
        self.vec.assign_1st_buf(index, new_bound_val);
    }}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.vec.get_2nd_buffer()}} 
    #[inline] unsafe fn drop_1st_buffer(&mut self) {}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) { unsafe {self.vec.drop_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_2nd_buf_index(index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {
        self.vec.drop_1st_buf_index(index);
        self.vec.drop_bound_bufs_index(index);
    }}
}

/// struct binding the vector's item to the buffer in the first slot, linking it to an internal output
pub struct VecHalfBind<V: VectorLike<FstHandleBool = Y>> {pub(crate) vec: V} 

impl<V: VectorLike<FstHandleBool = Y>> VecHalfBind<V> {
    pub(crate) unsafe fn get_bound_buf(&mut self) -> V::FstOwnedBuffer { unsafe {
        self.vec.get_1st_buffer()
    }}
}

unsafe impl<V: VectorLike<FstHandleBool = Y>> Get for VecHalfBind<V> where (V::BoundHandlesBool, Y): FilterPair {
    type GetBool = N;
    type Inputs = V::Inputs;
    type Item = ();
    type BoundItems = <(V::BoundHandlesBool, Y) as FilterPair>::Filtered<V::BoundItems, V::Item>;

    #[inline]
    unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {
        self.vec.get_inputs(index)
    }}

    #[inline]
    unsafe fn drop_inputs(&mut self, index: usize) { unsafe {
        self.vec.drop_inputs(index)
    }}

    #[inline]
    fn process(&mut self, index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
        let (item, bound) = self.vec.process(index, inputs);
        ((), <(V::BoundHandlesBool, Y) as FilterPair>::filter(bound, item))
    }
}

impl<V: VectorLike<FstHandleBool = Y>> HasOutput for VecHalfBind<V> where (V::OutputBool, V::FstOwnedBufferBool): TyBoolPair {
    type OutputBool = <(V::OutputBool, V::FstOwnedBufferBool) as TyBoolPair>::Or;
    type Output = V::Output;
    
    #[inline]
    unsafe fn output(&mut self) -> Self::Output { unsafe {
        self.vec.output()
    }}

    #[inline]
    unsafe fn drop_output(&mut self) { unsafe {
        self.vec.drop_output(); // buffer dropped through HasReuseBuf
        self.vec.drop_1st_buffer();
    }}
}

impl<V: VectorLike<FstHandleBool = Y>> HasReuseBuf for VecHalfBind<V> where (V::BoundHandlesBool, Y): FilterPair {
    type FstHandleBool = N;
    type SndHandleBool = V::SndHandleBool;
    type BoundHandlesBool = <(V::BoundHandlesBool, Y) as TyBoolPair>::Or;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = V::SndOwnedBufferBool;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = V::SndOwnedBuffer;
    type FstType = ();
    type SndType = V::SndType;
    type BoundTypes = <(V::BoundHandlesBool, Y) as FilterPair>::Filtered<V::BoundTypes, V::FstType>;

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) { unsafe {self.vec.assign_2nd_buf(index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) { unsafe {
        let (bounded_vals, new_bound_val) = <(V::BoundHandlesBool, Y) as FilterPair>::defilter(val);
        self.vec.assign_bound_bufs(index, bounded_vals);
        self.vec.assign_1st_buf(index, new_bound_val);
    }}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.vec.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buffer(&mut self) {}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) { unsafe {self.vec.drop_2nd_buffer();}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_2nd_buf_index(index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {
        self.vec.drop_1st_buf_index(index);
        self.vec.drop_bound_bufs_index(index);
    }}
}