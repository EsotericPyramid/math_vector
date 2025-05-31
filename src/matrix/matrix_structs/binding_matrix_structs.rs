use crate::trait_specialization_utils::*;
use crate::util_traits::*;
use crate::matrix::mat_util_traits::*;

/// struct binding the item to the buffer in first slot (and adding it to output if owned)
pub struct MatBind<M: MatrixLike<FstHandleBool = Y>>{pub(crate) mat: M}

unsafe impl<M: MatrixLike<FstHandleBool = Y>> Get2D for MatBind<M> where (M::BoundHandlesBool, Y): FilterPair {
    type GetBool = N;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = ();
    type BoundItems = <(M::BoundHandlesBool, Y) as FilterPair>::Filtered<M::BoundItems, M::Item>;

    #[inline]
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs { unsafe {
        self.mat.get_inputs(col_index, row_index)
    }}

    #[inline]
    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) { unsafe {
        self.mat.drop_inputs(col_index, row_index)
    }}

    #[inline]
    fn process(&mut self, col_index: usize, row_index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
        let (item, bound) = self.mat.process(col_index, row_index,  inputs);
        ((), <(M::BoundHandlesBool, Y) as FilterPair>::filter(bound, item))
    }
}

impl<M: MatrixLike<FstHandleBool = Y>> HasOutput for MatBind<M> where (M::OutputBool, M::FstOwnedBufferBool): FilterPair {
    type OutputBool = <(M::OutputBool, M::FstOwnedBufferBool) as TyBoolPair>::Or;
    type Output = <(M::OutputBool, M::FstOwnedBufferBool) as FilterPair>::Filtered<M::Output, M::FstOwnedBuffer>;

    #[inline]
    unsafe fn output(&mut self) -> Self::Output { unsafe {
        <(M::OutputBool, M::FstOwnedBufferBool) as FilterPair>::filter(self.mat.output(), self.mat.get_1st_buffer())
    }}

    #[inline]
    unsafe fn drop_output(&mut self) { unsafe {
        self.mat.drop_output();
    }}
}

impl<M: MatrixLike<FstHandleBool = Y>> Has2DReuseBuf for MatBind<M> where (M::BoundHandlesBool, Y): FilterPair, (M::IsFstBufferTransposed, M::AreBoundBuffersTransposed): TyBoolPair {
    type FstHandleBool = N;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = <(M::BoundHandlesBool, Y) as TyBoolPair>::Or;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type IsFstBufferTransposed = N;
    type IsSndBufferTransposed = M::IsSndBufferTransposed;
    type AreBoundBuffersTransposed = <(M::IsFstBufferTransposed, M::AreBoundBuffersTransposed) as TyBoolPair>::And;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = ();
    type SndType = M::SndType;
    type BoundTypes = <(M::BoundHandlesBool, Y) as FilterPair>::Filtered<M::BoundTypes, M::FstType>;

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) { unsafe {self.mat.assign_2nd_buf(col_index, row_index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) { unsafe {
        let (bounded_vals, new_bound_val) = <(M::BoundHandlesBool, Y) as FilterPair>::defilter(val);
        self.mat.assign_bound_bufs(col_index, row_index, bounded_vals);
        self.mat.assign_1st_buf(col_index, row_index, new_bound_val);
    }}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.mat.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_2nd_buf_index(col_index, row_index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) { unsafe {
        self.mat.drop_1st_buf_index(col_index, row_index);
        self.mat.drop_bound_bufs_index(col_index, row_index);
    }}
}

/// struct mapping the item (via FnMut closure) with one output as item and other bound to buffer in first slot (and adding it to output if owned)
pub struct MatMapBind<M: MatrixLike<FstHandleBool = Y>, F: FnMut(M::Item) -> (I, B), I, B>{pub(crate) mat: M, pub(crate) f: F}

unsafe impl<M: MatrixLike<FstHandleBool = Y>, F: FnMut(M::Item) -> (I, B), I, B> Get2D for MatMapBind<M, F, I, B> where (M::BoundHandlesBool, Y): FilterPair {
    type GetBool = Y;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = I;
    type BoundItems = <(M::BoundHandlesBool, Y) as FilterPair>::Filtered<M::BoundItems, B>;

    #[inline]
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs { unsafe {
        self.mat.get_inputs(col_index, row_index)
    }}

    #[inline]
    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) { unsafe {
        self.mat.drop_inputs(col_index, row_index)
    }}

    #[inline]
    fn process(&mut self, col_index: usize, row_index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
        let (item, bound) = self.mat.process(col_index, row_index,  inputs);
        let (processed_item, processed_bound) = (self.f)(item);
        (processed_item, <(M::BoundHandlesBool, Y) as FilterPair>::filter(bound, processed_bound))
    }
}

impl<M: MatrixLike<FstHandleBool = Y>, F: FnMut(M::Item) -> (I, B), I, B> HasOutput for MatMapBind<M, F, I, B> where (M::OutputBool, M::FstOwnedBufferBool): FilterPair {
    type OutputBool = <(M::OutputBool, M::FstOwnedBufferBool) as TyBoolPair>::Or;
    type Output = <(M::OutputBool, M::FstOwnedBufferBool) as FilterPair>::Filtered<M::Output, M::FstOwnedBuffer>;

    #[inline]
    unsafe fn output(&mut self) -> Self::Output { unsafe {
        <(M::OutputBool, M::FstOwnedBufferBool) as FilterPair>::filter(self.mat.output(), self.mat.get_1st_buffer())
    }}

    #[inline]
    unsafe fn drop_output(&mut self) { unsafe {
        self.mat.drop_output();
    }}
}

impl<M: MatrixLike<FstHandleBool = Y>, F: FnMut(M::Item) -> (I, B), I, B> Has2DReuseBuf for MatMapBind<M, F, I, B> where (M::BoundHandlesBool, Y): FilterPair, (M::IsFstBufferTransposed, M::AreBoundBuffersTransposed): TyBoolPair {
    type FstHandleBool = N;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = <(M::BoundHandlesBool, Y) as TyBoolPair>::Or;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type IsFstBufferTransposed = N;
    type IsSndBufferTransposed = M::IsSndBufferTransposed;
    type AreBoundBuffersTransposed = <(M::IsFstBufferTransposed, M::AreBoundBuffersTransposed) as TyBoolPair>::And;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = ();
    type SndType = M::SndType;
    type BoundTypes = <(M::BoundHandlesBool, Y) as FilterPair>::Filtered<M::BoundTypes, M::FstType>;

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) { unsafe {self.mat.assign_2nd_buf(col_index, row_index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) { unsafe {
        let (bounded_vals, new_bound_val) = <(M::BoundHandlesBool, Y) as FilterPair>::defilter(val);
        self.mat.assign_bound_bufs(col_index, row_index, bounded_vals);
        self.mat.assign_1st_buf(col_index, row_index, new_bound_val);
    }}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.mat.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_2nd_buf_index(col_index, row_index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) { unsafe {
        self.mat.drop_1st_buf_index(col_index, row_index);
        self.mat.drop_bound_bufs_index(col_index, row_index);
    }}
}

/// struct mapping the item to the buffer in first slot and adding it to an *internal* output
pub struct MatHalfBind<M: MatrixLike<FstHandleBool = Y>> {pub(crate) mat: M} 

impl<M: MatrixLike<FstHandleBool = Y>> MatHalfBind<M> {
    pub(crate) unsafe fn get_bound_buf(&mut self) -> M::FstOwnedBuffer { unsafe {
        self.mat.get_1st_buffer()
    }}
}

unsafe impl<M: MatrixLike<FstHandleBool = Y>> Get2D for MatHalfBind<M> where (M::BoundHandlesBool, Y): FilterPair {
    type GetBool = N;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = ();
    type BoundItems = <(M::BoundHandlesBool, Y) as FilterPair>::Filtered<M::BoundItems, M::Item>;

    #[inline]
    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs { unsafe {
        self.mat.get_inputs(col_index, row_index)
    }}

    #[inline]
    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) { unsafe {
        self.mat.drop_inputs(col_index, row_index)
    }}

    #[inline]
    fn process(&mut self, col_index: usize, row_index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
        let (item, bound) = self.mat.process(col_index, row_index,  inputs);
        ((), <(M::BoundHandlesBool, Y) as FilterPair>::filter(bound, item))
    }
}

impl<M: MatrixLike<FstHandleBool = Y>> HasOutput for MatHalfBind<M> where (M::OutputBool, M::FstOwnedBufferBool): TyBoolPair {
    type OutputBool = <(M::OutputBool, M::FstOwnedBufferBool) as TyBoolPair>::Or;
    type Output = M::Output;
    
    #[inline]
    unsafe fn output(&mut self) -> Self::Output { unsafe {
        self.mat.output()
    }}

    #[inline]
    unsafe fn drop_output(&mut self) { unsafe {
        self.mat.drop_output(); // buffer dropped through HasReuseBuf
    }}
}

impl<M: MatrixLike<FstHandleBool = Y>> Has2DReuseBuf for MatHalfBind<M> where (M::BoundHandlesBool, Y): FilterPair, (M::IsFstBufferTransposed, M::AreBoundBuffersTransposed): TyBoolPair {
    type FstHandleBool = N;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = <(M::BoundHandlesBool, Y) as TyBoolPair>::Or;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type IsFstBufferTransposed = N;
    type IsSndBufferTransposed = M::IsSndBufferTransposed;
    type AreBoundBuffersTransposed = <(M::IsFstBufferTransposed, M::AreBoundBuffersTransposed) as TyBoolPair>::And;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = ();
    type SndType = M::SndType;
    type BoundTypes = <(M::BoundHandlesBool, Y) as FilterPair>::Filtered<M::BoundTypes, M::FstType>;

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) { unsafe {self.mat.assign_2nd_buf(col_index, row_index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) { unsafe {
        let (bounded_vals, new_bound_val) = <(M::BoundHandlesBool, Y) as FilterPair>::defilter(val);
        self.mat.assign_bound_bufs(col_index, row_index, bounded_vals);
        self.mat.assign_1st_buf(col_index, row_index, new_bound_val);
    }}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.mat.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_2nd_buf_index(col_index, row_index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) { unsafe {
        self.mat.drop_1st_buf_index(col_index, row_index);
        self.mat.drop_bound_bufs_index(col_index, row_index);
    }}
}