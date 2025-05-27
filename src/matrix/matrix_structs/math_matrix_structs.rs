// NOTE: Most the math matrix structs are actually defined in macroed_matrix_structs.rs
//       only the complicated ones end up here
use crate::trait_specialization_utils::*;
use crate::util_traits::*;
use crate::matrix::mat_util_traits::*;
use std::ops::*;

/// struct multiplying 2 repeatable matrices
pub struct FullMatMul<M1: MatrixLike + Is2DRepeatable, M2: MatrixLike + Is2DRepeatable>{pub(crate) l_mat: M1, pub(crate) r_mat: M2, pub(crate) shared_size: usize}

unsafe impl<M1: MatrixLike + Is2DRepeatable, M2: MatrixLike + Is2DRepeatable> Get2D for FullMatMul<M1, M2> where 
    M1::Item: Mul<M2::Item>,
    <M1::Item as Mul<M2::Item>>::Output: AddAssign,
    (M1::BoundHandlesBool, M2::BoundHandlesBool): FilterPair,
    (M1::AreInputsTransposed, M2::AreInputsTransposed): TyBoolPair,
{
    type GetBool = Y;
    type AreInputsTransposed = <(M1::AreInputsTransposed, M2::AreInputsTransposed) as TyBoolPair>::Or;
    type Inputs = (usize, usize);
    type Item = <M1::Item as Mul<M2::Item>>::Output;
    type BoundItems = <(M1::BoundHandlesBool, M2::BoundHandlesBool) as FilterPair>::Filtered<M1::BoundItems, M2::BoundItems>;

    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs {(col_index, row_index)}
    unsafe fn drop_inputs(&mut self, _: usize, _: usize) {}
    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
        unsafe {
            let mut result = self.l_mat.get(0, inputs.1).0 * self.r_mat.get(inputs.0, 0).0;
            for i in 1..self.shared_size {
                result += self.l_mat.get(i, inputs.1).0 * self.r_mat.get(inputs.0, i).0;
            }
            let bound = <(M1::BoundHandlesBool, M2::BoundHandlesBool) as FilterPair>::filter(
                self.l_mat.get(inputs.0, inputs.1).1, 
                self.r_mat.get(inputs.0, inputs.1).1
            );
            (result, bound)
        }
    }
}

impl<M1: MatrixLike + Is2DRepeatable, M2: MatrixLike + Is2DRepeatable> HasOutput for FullMatMul<M1, M2> where (M1::OutputBool, M2::OutputBool): FilterPair {
    type OutputBool = <(M1::OutputBool, M2::OutputBool) as TyBoolPair>::Or;
    type Output = <(M1::OutputBool, M2::OutputBool) as FilterPair>::Filtered<M1::Output, M2::Output>;

    unsafe fn output(&mut self) -> Self::Output { unsafe {
        <(M1::OutputBool, M2::OutputBool) as FilterPair>::filter(
            self.l_mat.output(),
            self.r_mat.output()
        )
    }}
    unsafe fn drop_output(&mut self) { unsafe {
        self.l_mat.drop_output();
        self.r_mat.drop_output();
    }}
} 

impl<M1: MatrixLike + Is2DRepeatable, M2: MatrixLike + Is2DRepeatable> Has2DReuseBuf for FullMatMul<M1, M2>
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
    type FstOwnedBuffer = <(M1::FstOwnedBufferBool, M2::FstOwnedBufferBool) as SelectPair>::Selected<M1::FstOwnedBuffer, M2::FstOwnedBuffer>;
    type SndOwnedBuffer = <(M1::SndOwnedBufferBool, M2::SndOwnedBufferBool) as SelectPair>::Selected<M1::SndOwnedBuffer, M2::SndOwnedBuffer>;
    type FstType = <(M1::FstHandleBool, M2::FstHandleBool) as SelectPair>::Selected<M1::FstType, M2::FstType>;
    type SndType = <(M1::SndHandleBool, M2::SndHandleBool) as SelectPair>::Selected<M1::SndType, M2::SndType>;
    type BoundTypes = <(M1::BoundHandlesBool, M2::BoundHandlesBool) as FilterPair>::Filtered<M1::BoundTypes, M2::BoundTypes>;

    #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) { unsafe {
        let (l_val, r_val) = <(M1::FstHandleBool, M2::FstHandleBool) as SelectPair>::deselect(val);
        self.l_mat.assign_1st_buf(col_index, row_index, l_val);
        self.r_mat.assign_1st_buf(col_index, row_index, r_val);
    }}
    #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) { unsafe {
        let (l_val, r_val) = <(M1::SndHandleBool, M2::SndHandleBool) as SelectPair>::deselect(val);
        self.l_mat.assign_2nd_buf(col_index, row_index, l_val);
        self.r_mat.assign_2nd_buf(col_index, row_index, r_val);
    }}
    #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) { unsafe {
        let (l_val, r_val) = <(M1::BoundHandlesBool, M2::BoundHandlesBool) as FilterPair>::defilter(val);
        self.l_mat.assign_bound_bufs(col_index, row_index, l_val);
        self.r_mat.assign_bound_bufs(col_index, row_index, r_val);
    }}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {
        <(M1::FstOwnedBufferBool, M2::FstOwnedBufferBool) as SelectPair>::select(self.l_mat.get_1st_buffer(), self.r_mat.get_1st_buffer())
    }}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {
        <(M1::SndOwnedBufferBool, M2::SndOwnedBufferBool) as SelectPair>::select(self.l_mat.get_2nd_buffer(), self.r_mat.get_2nd_buffer())
    }}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {
        self.l_mat.drop_1st_buf_index(col_index, row_index);
        self.r_mat.drop_1st_buf_index(col_index, row_index);
    }}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {
        self.l_mat.drop_2nd_buf_index(col_index, row_index);
        self.r_mat.drop_2nd_buf_index(col_index, row_index);
    }}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) { unsafe {
        self.l_mat.drop_bound_bufs_index(col_index, row_index);
        self.r_mat.drop_bound_bufs_index(col_index, row_index);
    }}
}
