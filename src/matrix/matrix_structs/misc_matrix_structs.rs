use crate::{
    trait_specialization_utils::*,
    util_traits::*,
    matrix::mat_util_traits::*,
};

/// struct swapping the buffers (or lack thereof) in the first and second slots
pub struct MatBufSwap<M: MatrixLike>{pub(crate) mat: M} 

unsafe impl<M: MatrixLike> Get2D for MatBufSwap<M> {
    type GetBool = M::GetBool;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs { unsafe {self.mat.get_inputs(col_index, row_index)}}
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_inputs(col_index, row_index)}}
    #[inline] fn process(&mut self, col_index: usize, row_index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.mat.process(col_index, row_index,  inputs)}
}

unsafe impl<M: Is2DRepeatable + MatrixLike> Is2DRepeatable for MatBufSwap<M> {}

impl<M: MatrixLike> HasOutput for MatBufSwap<M> {
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.mat.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.mat.drop_output()}}
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

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) { unsafe {self.mat.assign_2nd_buf(col_index, row_index, val)}}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) { unsafe {self.mat.assign_1st_buf(col_index, row_index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) { unsafe {self.mat.assign_bound_bufs(col_index, row_index, val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {self.mat.get_2nd_buffer()}}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.mat.get_1st_buffer()}}
    #[inline] unsafe fn drop_1st_buffer(&mut self) { unsafe {self.mat.drop_2nd_buffer()}}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) { unsafe {self.mat.drop_1st_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_2nd_buf_index(col_index, row_index)}}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_1st_buf_index(col_index, row_index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_bound_bufs_index(col_index, row_index)}}
}


/// struct attaching a *used* matrix's output and buffers to another matrix
/// SAFETY: it is expected that the used_mat field is safe to output in addition to normal correct implementation
pub struct MatAttachUsedMat<M: MatrixLike, USEDM: MatrixLike>{pub(crate) mat: M, pub(crate) used_mat: USEDM}

unsafe impl<M: MatrixLike, USEDM: MatrixLike> Get2D for MatAttachUsedMat<M, USEDM> {
    type GetBool = M::GetBool;
    type AreInputsTransposed = M::AreInputsTransposed;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs { unsafe {self.mat.get_inputs(col_index, row_index)}}
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_inputs(col_index, row_index)}}
    #[inline] fn process(&mut self, col_index: usize, row_index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.mat.process(col_index, row_index,  inputs)}
}

unsafe impl<M: Is2DRepeatable + MatrixLike, USEDM: MatrixLike> Is2DRepeatable for MatAttachUsedMat<M, USEDM> {}

impl<M: MatrixLike, USEDM: MatrixLike> HasOutput for MatAttachUsedMat<M, USEDM> where (M::OutputBool, USEDM::OutputBool): FilterPair {
    type OutputBool = <(M::OutputBool, USEDM::OutputBool) as TyBoolPair>::Or;
    type Output = <(M::OutputBool, USEDM::OutputBool) as FilterPair>::Filtered<M::Output, USEDM::Output>;

    #[inline] 
    unsafe fn output(&mut self) -> Self::Output { unsafe {
        <(M::OutputBool, USEDM::OutputBool) as FilterPair>::filter(
            self.mat.output(),
            self.used_mat.output()
        )
    }}

    #[inline]
    unsafe fn drop_output(&mut self) { unsafe {
        self.mat.drop_output();
        self.used_mat.drop_output();
    }}
}

impl<M: MatrixLike, USEDM: MatrixLike> Has2DReuseBuf for MatAttachUsedMat<M, USEDM> 
where 
    (M::FstOwnedBufferBool, USEDM::FstOwnedBufferBool): SelectPair,
    (M::SndOwnedBufferBool, USEDM::SndOwnedBufferBool): SelectPair,
    (M::FstHandleBool, USEDM::FstHandleBool): SelectPair,
    (M::SndHandleBool, USEDM::SndHandleBool): SelectPair,
    (M::BoundHandlesBool, USEDM::BoundHandlesBool): FilterPair,
    (M::IsFstBufferTransposed, USEDM::IsFstBufferTransposed): TyBoolPair,
    (M::IsSndBufferTransposed, USEDM::IsSndBufferTransposed): TyBoolPair,    
    (M::AreBoundBuffersTransposed, USEDM::AreBoundBuffersTransposed): TyBoolPair
{
    type FstHandleBool = <(M::FstHandleBool, USEDM::FstHandleBool) as TyBoolPair>::Xor;
    type SndHandleBool = <(M::SndHandleBool, USEDM::SndHandleBool) as TyBoolPair>::Xor;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = <(M::FstOwnedBufferBool, USEDM::FstOwnedBufferBool) as TyBoolPair>::Xor; 
    type SndOwnedBufferBool = <(M::SndOwnedBufferBool, USEDM::SndOwnedBufferBool) as TyBoolPair>::Xor; 
    type IsFstBufferTransposed = <(M::IsFstBufferTransposed, USEDM::IsFstBufferTransposed) as TyBoolPair>::Or;
    type IsSndBufferTransposed = <(M::IsSndBufferTransposed, USEDM::IsSndBufferTransposed) as TyBoolPair>::Or;
    type AreBoundBuffersTransposed = <(M::AreBoundBuffersTransposed, USEDM::AreBoundBuffersTransposed) as TyBoolPair>::And;
    type FstOwnedBuffer = <(M::FstOwnedBufferBool, USEDM::FstOwnedBufferBool) as SelectPair>::Selected<M::FstOwnedBuffer, USEDM::FstOwnedBuffer>;
    type SndOwnedBuffer = <(M::SndOwnedBufferBool, USEDM::SndOwnedBufferBool) as SelectPair>::Selected<M::SndOwnedBuffer, USEDM::SndOwnedBuffer>;
    type FstType = <(M::FstHandleBool, USEDM::FstHandleBool) as SelectPair>::Selected<M::FstType, USEDM::FstType>;
    type SndType = <(M::SndHandleBool, USEDM::SndHandleBool) as SelectPair>::Selected<M::SndType, USEDM::SndType>;
    type BoundTypes = M::BoundTypes;

    #[inline] unsafe fn assign_1st_buf<'z>(&'z mut self, col_index: usize, row_index: usize, val: Self::FstType) { unsafe {
        let (l_val, r_val) = <(M::FstHandleBool, USEDM::FstHandleBool) as SelectPair>::deselect(val);
        self.mat.assign_1st_buf(col_index, row_index, l_val);
        self.used_mat.assign_1st_buf(col_index, row_index, r_val);
    }}
    #[inline] unsafe fn assign_2nd_buf<'z>(&'z mut self, col_index: usize, row_index: usize, val: Self::SndType) { unsafe {
        let (l_val, r_val) = <(M::SndHandleBool, USEDM::SndHandleBool) as SelectPair>::deselect(val);
        self.mat.assign_2nd_buf(col_index, row_index, l_val);
        self.used_mat.assign_2nd_buf(col_index, row_index, r_val);
    }}
    #[inline] unsafe fn assign_bound_bufs<'z>(&'z mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) { unsafe {
        self.mat.assign_bound_bufs(col_index, row_index, val);
    }}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {
        <(M::FstOwnedBufferBool, USEDM::FstOwnedBufferBool) as SelectPair>::select(self.mat.get_1st_buffer(), self.used_mat.get_1st_buffer())
    }}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {
        <(M::SndOwnedBufferBool, USEDM::SndOwnedBufferBool) as SelectPair>::select(self.mat.get_2nd_buffer(), self.used_mat.get_2nd_buffer())
    }}
    #[inline] unsafe fn drop_1st_buffer(&mut self) { unsafe {
        self.mat.drop_1st_buffer();
        self.used_mat.drop_1st_buffer();
    }}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) { unsafe {
        self.mat.drop_2nd_buffer();
        self.used_mat.drop_2nd_buffer();
    }}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {
        self.mat.drop_1st_buf_index(col_index, row_index);
        self.used_mat.drop_1st_buf_index(col_index, row_index);
    }}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {
        self.mat.drop_2nd_buf_index(col_index, row_index);
        self.used_mat.drop_2nd_buf_index(col_index, row_index);
    }}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) { unsafe {
        self.mat.drop_bound_bufs_index(col_index, row_index);
        self.used_mat.drop_bound_bufs_index(col_index, row_index);
    }}
}

/// struct stabilizing a matrix's type so that it can be made dynamic
pub struct DynamicMatrixLike<M: MatrixLike>{pub(crate) mat: M, pub(crate) inputs: Option<M::Inputs>}

unsafe impl<M: MatrixLike> Get2D for DynamicMatrixLike<M> {
    type GetBool = M::GetBool;
    type AreInputsTransposed = N;
    type Inputs = ();
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs { self.inputs = Some(unsafe {self.mat.get_inputs(col_index, row_index)}) }
    #[inline] unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_inputs(col_index, row_index)}}
    #[inline] fn process(&mut self, col_index: usize, row_index: usize, _: Self::Inputs) -> (Self::Item, Self::BoundItems) {self.mat.process(col_index, row_index,  self.inputs.take().unwrap())}
}

impl<M: MatrixLike> HasOutput for DynamicMatrixLike<M> {
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {self.mat.output()}}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.mat.drop_output()}}
}

impl<M: MatrixLike> Has2DReuseBuf for DynamicMatrixLike<M> {
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

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) { unsafe {self.mat.assign_1st_buf(col_index, row_index, val)}}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) { unsafe {self.mat.assign_2nd_buf(col_index, row_index, val)}}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) { unsafe {self.mat.assign_bound_bufs(col_index, row_index, val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {self.mat.get_1st_buffer()}}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.mat.get_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buffer(&mut self) { unsafe {self.mat.drop_1st_buffer()}}
    #[inline] unsafe fn drop_2nd_buffer(&mut self) { unsafe {self.mat.drop_2nd_buffer()}}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_1st_buf_index(col_index, row_index)}}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_2nd_buf_index(col_index, row_index)}}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.mat.drop_bound_bufs_index(col_index, row_index)}}
}
