
use crate::vector::vec_util_traits::{VectorBuilder, VectorLike};

// Note: traits here aren't meant to be used by end users
use crate::trait_specialization_utils::TyBool;
use crate::util_traits::HasOutput;

pub unsafe trait Get2D {
    type GetBool: TyBool;
    type AreInputsTransposed: TyBool; // used to optimize access order
    type Inputs;
    type Item;
    type BoundItems;

    unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs; 

    unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize);

    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems);

    #[inline]
    unsafe fn get(&mut self, col_index: usize, row_index: usize) -> (Self::Item, Self::BoundItems) { unsafe {
        let inputs = self.get_inputs(col_index, row_index);
        self.process(inputs)
    }}
}

pub trait Has2DReuseBuf {
    type FstHandleBool: TyBool;
    type SndHandleBool: TyBool;
    type BoundHandlesBool: TyBool;
    type FstOwnedBufferBool: TyBool;
    type SndOwnedBufferBool: TyBool;
    type IsFstBufferTransposed: TyBool;
    type IsSndBufferTransposed: TyBool;
    type AreBoundBuffersTransposed: TyBool;
    type FstOwnedBuffer;
    type SndOwnedBuffer;
    type FstType;
    type SndType;
    type BoundTypes;

    unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType); 
    unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType);
    unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes);
    unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize);
    unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize);
    unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize);
    unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer;
    unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer;
}

///really just a shorthand for the individual traits
pub trait MatrixLike: Get2D + HasOutput + Has2DReuseBuf {}

impl<T: Get2D + HasOutput + Has2DReuseBuf> MatrixLike for T {}

pub trait MatrixBuilder: Clone {
    type MatrixWrapped<T: MatrixLike>;
    type TransposedMatrixWrapped<T: MatrixLike>;
    type ColWrapped<T: VectorLike>;
    type RowWrapped<T: VectorLike>;

    //FIXME (HRTBs): for<T: VectorLike> Self::ColBuilder::Wrapped<T> == Self::ColWrapped
    type ColBuilder: VectorBuilder;
    type RowBuilder: VectorBuilder;


    unsafe fn wrap_mat<T: MatrixLike>(&self, mat: T) -> Self::MatrixWrapped<T>;
    unsafe fn wrap_trans_mat<T: MatrixLike>(&self, mat: T) -> Self::TransposedMatrixWrapped<T>;
    unsafe fn wrap_col_vec<T: VectorLike>(&self, vec: T) -> Self::ColWrapped<T>;
    unsafe fn wrap_row_vec<T: VectorLike>(&self, vec: T) -> Self::RowWrapped<T>;        

    //FIXME (above is source of issue): currently requires correct implementation even though trait is not unsafe
    fn decompose(self) -> (Self::ColBuilder, Self::RowBuilder);
    fn compose(col: Self::ColBuilder, row: Self::RowBuilder) -> Self;
}

pub trait MatrixBuilderUnion<T: MatrixBuilder>: MatrixBuilder {
    type Union: MatrixBuilder;

    fn union(self, other: T) -> Self::Union;
}