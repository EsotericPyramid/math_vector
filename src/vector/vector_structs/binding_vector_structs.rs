use crate::trait_specialization_utils::*;
use crate::util_traits::*;
use crate::vector::vec_util_traits::*;
use std::ops::*;

pub struct VecBind<T: VectorLike<FstHandleBool = Y>> {pub(crate) vec: T} 

unsafe impl<T: VectorLike<FstHandleBool = Y>> Get for VecBind<T> where (T::BoundHandlesBool, Y): FilterPair {
    type GetBool = N;
    type Inputs = T::Inputs;
    type Item = ();
    type BoundItems = <(T::BoundHandlesBool, Y) as FilterPair>::Filtered<T::BoundItems, T::Item>;

    #[inline]
    unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {
        self.vec.get_inputs(index)
    }}

    #[inline]
    unsafe fn drop_inputs(&mut self, index: usize) { unsafe {
        self.vec.drop_inputs(index)
    }}

    #[inline]
    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
        let (item, bound) = self.vec.process(inputs);
        ((), <(T::BoundHandlesBool, Y) as FilterPair>::filter(bound, item))
    }
}

impl<T: VectorLike<FstHandleBool = Y>> HasOutput for VecBind<T> where (T::OutputBool, T::FstOwnedBufferBool): FilterPair {
    type OutputBool = <(T::OutputBool, T::FstOwnedBufferBool) as TyBoolPair>::Or;
    type Output = <(T::OutputBool, T::FstOwnedBufferBool) as FilterPair>::Filtered<T::Output, T::FstOwnedBuffer>;
    
    #[inline]
    unsafe fn output(&mut self) -> Self::Output { unsafe {
        <(T::OutputBool, T::FstOwnedBufferBool) as FilterPair>::filter(self.vec.output(), self.vec.get_1st_buffer())
    }}

    #[inline]
    unsafe fn drop_output(&mut self) { unsafe {
        self.vec.drop_output(); // buffer dropped through HasReuseBuf
    }}
}

impl<T: VectorLike<FstHandleBool = Y>> HasReuseBuf for VecBind<T> where (T::BoundHandlesBool, Y): FilterPair {
    type FstHandleBool = N;
    type SndHandleBool = T::SndHandleBool;
    type BoundHandlesBool = <(T::BoundHandlesBool, Y) as TyBoolPair>::Or;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = T::SndOwnedBufferBool;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = T::SndOwnedBuffer;
    type FstType = ();
    type SndType = T::SndType;
    type BoundTypes = <(T::BoundHandlesBool, Y) as FilterPair>::Filtered<T::BoundTypes, T::FstType>;

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) { unsafe {self.vec.assign_2nd_buf(index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) { unsafe {
        let (bounded_vals, new_bound_val) = <(T::BoundHandlesBool, Y) as FilterPair>::defilter(val);
        self.vec.assign_bound_bufs(index, bounded_vals);
        self.vec.assign_1st_buf(index, new_bound_val);
    }}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.vec.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_2nd_buf_index(index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {
        self.vec.drop_1st_buf_index(index);
        self.vec.drop_bound_bufs_index(index);
    }}
}


pub struct VecMapBind<T: VectorLike<FstHandleBool = Y>, F: FnMut(T::Item) -> (I, B), I, B> {pub(crate) vec: T, pub(crate) f: F}

unsafe impl<T: VectorLike<FstHandleBool = Y>, F: FnMut(T::Item) -> (I, B), I, B> Get for VecMapBind<T, F, I, B> where (T::BoundHandlesBool, Y): FilterPair {
    type GetBool = Y;
    type Inputs = T::Inputs;
    type Item = I;
    type BoundItems = <(T::BoundHandlesBool, Y) as FilterPair>::Filtered<T::BoundItems, B>;

    #[inline]
    unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {
        self.vec.get_inputs(index)
    }}

    #[inline]
    unsafe fn drop_inputs(&mut self, index: usize) { unsafe {
        self.vec.drop_inputs(index)
    }}

    #[inline]
    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
        let (item, bound) = self.vec.process(inputs);
        let (processed_item, processed_bound) = (self.f)(item);
        (processed_item, <(T::BoundHandlesBool, Y) as FilterPair>::filter(bound, processed_bound))
    }
}

impl<T: VectorLike<FstHandleBool = Y>, F: FnMut(T::Item) -> (I, B), I, B> HasOutput for VecMapBind<T, F, I, B> where (T::OutputBool, T::FstOwnedBufferBool): FilterPair {
    type OutputBool = <(T::OutputBool, T::FstOwnedBufferBool) as TyBoolPair>::Or;
    type Output = <(T::OutputBool, T::FstOwnedBufferBool) as FilterPair>::Filtered<T::Output, T::FstOwnedBuffer>;
    
    #[inline]
    unsafe fn output(&mut self) -> Self::Output { unsafe {
        <(T::OutputBool, T::FstOwnedBufferBool) as FilterPair>::filter(self.vec.output(), self.vec.get_1st_buffer())
    }}

    #[inline]
    unsafe fn drop_output(&mut self) { unsafe {
        self.vec.drop_output(); // buffer dropped through HasReuseBuf
    }}
}

impl<T: VectorLike<FstHandleBool = Y>, F: FnMut(T::Item) -> (I, B), I, B> HasReuseBuf for VecMapBind<T, F, I, B> where (T::BoundHandlesBool, Y): FilterPair {
    type FstHandleBool = N;
    type SndHandleBool = T::SndHandleBool;
    type BoundHandlesBool = <(T::BoundHandlesBool, Y) as TyBoolPair>::Or;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = T::SndOwnedBufferBool;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = T::SndOwnedBuffer;
    type FstType = ();
    type SndType = T::SndType;
    type BoundTypes = <(T::BoundHandlesBool, Y) as FilterPair>::Filtered<T::BoundTypes, T::FstType>;

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) { unsafe {self.vec.assign_2nd_buf(index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) { unsafe {
        let (bounded_vals, new_bound_val) = <(T::BoundHandlesBool, Y) as FilterPair>::defilter(val);
        self.vec.assign_bound_bufs(index, bounded_vals);
        self.vec.assign_1st_buf(index, new_bound_val);
    }}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.vec.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_2nd_buf_index(index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {
        self.vec.drop_1st_buf_index(index);
        self.vec.drop_bound_bufs_index(index);
    }}
}


pub struct VecHalfBind<T: VectorLike<FstHandleBool = Y>> {pub(crate) vec: T} 

impl<T: VectorLike<FstHandleBool = Y>> VecHalfBind<T> {
    pub(crate) unsafe fn get_bound_buf(&mut self) -> T::FstOwnedBuffer { unsafe {
        self.vec.get_1st_buffer()
    }}
}

unsafe impl<T: VectorLike<FstHandleBool = Y>> Get for VecHalfBind<T> where (T::BoundHandlesBool, Y): FilterPair {
    type GetBool = N;
    type Inputs = T::Inputs;
    type Item = ();
    type BoundItems = <(T::BoundHandlesBool, Y) as FilterPair>::Filtered<T::BoundItems, T::Item>;

    #[inline]
    unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {
        self.vec.get_inputs(index)
    }}

    #[inline]
    unsafe fn drop_inputs(&mut self, index: usize) { unsafe {
        self.vec.drop_inputs(index)
    }}

    #[inline]
    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
        let (item, bound) = self.vec.process(inputs);
        ((), <(T::BoundHandlesBool, Y) as FilterPair>::filter(bound, item))
    }
}

impl<T: VectorLike<FstHandleBool = Y>> HasOutput for VecHalfBind<T> where (T::OutputBool, T::FstOwnedBufferBool): TyBoolPair {
    type OutputBool = <(T::OutputBool, T::FstOwnedBufferBool) as TyBoolPair>::Or;
    type Output = T::Output;
    
    #[inline]
    unsafe fn output(&mut self) -> Self::Output { unsafe {
        self.vec.output()
    }}

    #[inline]
    unsafe fn drop_output(&mut self) { unsafe {
        self.vec.drop_output(); // buffer dropped through HasReuseBuf
    }}
}

impl<T: VectorLike<FstHandleBool = Y>> HasReuseBuf for VecHalfBind<T> where (T::BoundHandlesBool, Y): FilterPair {
    type FstHandleBool = N;
    type SndHandleBool = T::SndHandleBool;
    type BoundHandlesBool = <(T::BoundHandlesBool, Y) as TyBoolPair>::Or;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = T::SndOwnedBufferBool;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = T::SndOwnedBuffer;
    type FstType = ();
    type SndType = T::SndType;
    type BoundTypes = <(T::BoundHandlesBool, Y) as FilterPair>::Filtered<T::BoundTypes, T::FstType>;

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) { unsafe {self.vec.assign_2nd_buf(index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) { unsafe {
        let (bounded_vals, new_bound_val) = <(T::BoundHandlesBool, Y) as FilterPair>::defilter(val);
        self.vec.assign_bound_bufs(index, bounded_vals);
        self.vec.assign_1st_buf(index, new_bound_val);
    }}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.vec.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.vec.drop_2nd_buf_index(index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {
        self.vec.drop_1st_buf_index(index);
        self.vec.drop_bound_bufs_index(index);
    }}
}