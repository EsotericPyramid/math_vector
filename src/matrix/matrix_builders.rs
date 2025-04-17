use crate::{
    matrix::{mat_util_traits::{MatrixBuilder, MatrixLike}, MatrixExpr},
    vector::{vec_util_traits::VectorLike, VectorExpr, vector_builders::VectorExprBuilder}
};

use super::mat_util_traits::MatrixBuilderUnion;

#[derive(Clone)]
pub struct MatrixExprBuilder<const D1: usize, const D2: usize>;

impl<const D1: usize, const D2: usize> MatrixBuilder for MatrixExprBuilder<D1, D2> {
    type MatrixWrapped<T: MatrixLike> = MatrixExpr<T, D1, D2>;
    type TransposedMatrixWrapped<T: MatrixLike> = MatrixExpr<T, D2, D1>;
    type ColWrapped<T: VectorLike> = VectorExpr<T, D1>;
    type RowWrapped<T: VectorLike> = VectorExpr<T, D2>;

    type ColBuilder = VectorExprBuilder<D1>;
    type RowBuilder = VectorExprBuilder<D2>;

    #[inline] unsafe fn wrap_mat<T: MatrixLike>(&self, mat: T) -> Self::MatrixWrapped<T> {MatrixExpr(mat)}
    #[inline] unsafe fn wrap_trans_mat<T: MatrixLike>(&self, mat: T) -> Self::TransposedMatrixWrapped<T> {MatrixExpr(mat)}
    #[inline] unsafe fn wrap_col_vec<T: VectorLike>(&self, vec: T) -> Self::ColWrapped<T> {VectorExpr(vec)}
    #[inline] unsafe fn wrap_row_vec<T: VectorLike>(&self, vec: T) -> Self::RowWrapped<T> {VectorExpr(vec)}

    #[inline] fn decompose(self) -> (Self::ColBuilder, Self::RowBuilder) {(VectorExprBuilder, VectorExprBuilder)}
    #[inline] fn compose(_: Self::ColBuilder, _: Self::RowBuilder) -> Self {Self}
}

impl<const D1: usize, const D2: usize> MatrixBuilderUnion<MatrixExprBuilder<D1,D2>> for MatrixExprBuilder<D1,D2> {
    type Union = Self;

    fn union(self, _: MatrixExprBuilder<D1,D2>) -> Self::Union {self}
}

