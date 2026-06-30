//! Structs implementing MatrixBuilder to wrap MatrixLikes with sizing information

use crate::{
    matrix::{
        MatrixExpr,
        RSMatrixExpr,
        mat_util_traits::{MatrixBuilder, MatrixBuilderCompose, MatrixBuilderUnion, MatrixLike},
    },
    vector::{
        vector_exprs::{
            VectorExpr, 
            RSVectorExpr, 
        },
        vec_util_traits::VectorLike, 
        vector_builders::*
    },
};

/// a simple const sized MatrixBuilder
#[derive(Clone)]
pub struct MatrixExprBuilder<const D1: usize, const D2: usize>;

impl<const D1: usize, const D2: usize> MatrixBuilder for MatrixExprBuilder<D1, D2> {
    type MatrixWrapped<T: MatrixLike> = MatrixExpr<T, D1, D2>;
    type TransposedMatrixWrapped<T: MatrixLike> = MatrixExpr<T, D2, D1>;
    type ColWrapped<T: VectorLike> = VectorExpr<T, D1>;
    type RowWrapped<T: VectorLike> = VectorExpr<T, D2>;

    type ColBuilder = VectorExprBuilder<D1>;
    type RowBuilder = VectorExprBuilder<D2>;

    #[inline]
    unsafe fn wrap_mat<T: MatrixLike>(&self, mat: T) -> Self::MatrixWrapped<T> {
        MatrixExpr(mat)
    }
    #[inline]
    unsafe fn wrap_trans_mat<T: MatrixLike>(&self, mat: T) -> Self::TransposedMatrixWrapped<T> {
        MatrixExpr(mat)
    }
    #[inline]
    unsafe fn wrap_col_vec<T: VectorLike>(&self, vec: T) -> Self::ColWrapped<T> {
        VectorExpr(vec)
    }
    #[inline]
    unsafe fn wrap_row_vec<T: VectorLike>(&self, vec: T) -> Self::RowWrapped<T> {
        VectorExpr(vec)
    }

    #[inline]
    fn decompose(self) -> (Self::ColBuilder, Self::RowBuilder) {
        (VectorExprBuilder, VectorExprBuilder)
    }

    #[inline]
    fn dimensions(&self) -> (usize, usize) {
        (D1, D2)
    }
}

impl<const D1: usize, const D2: usize> MatrixBuilderCompose<VectorExprBuilder<D2>>
    for VectorExprBuilder<D1>
{
    type Composition = MatrixExprBuilder<D1, D2>;

    fn compose(self, _: VectorExprBuilder<D2>) -> Self::Composition {
        MatrixExprBuilder
    }
}

#[derive(Clone)]
pub struct RSMatrixExprBuilder{
    pub(crate) num_rows: usize,
    pub(crate) num_cols: usize,
}

impl MatrixBuilder for RSMatrixExprBuilder {
    type MatrixWrapped<T: MatrixLike> = RSMatrixExpr<T>;
    type TransposedMatrixWrapped<T: MatrixLike> = RSMatrixExpr<T>;
    type ColWrapped<T: VectorLike> = RSVectorExpr<T>;
    type RowWrapped<T: VectorLike> = RSVectorExpr<T>;

    type ColBuilder = RSVectorExprBuilder;
    type RowBuilder = RSVectorExprBuilder;

    #[inline]
    unsafe fn wrap_mat<T: MatrixLike>(&self, mat: T) -> Self::MatrixWrapped<T> {
        RSMatrixExpr{
            mat,
            num_rows: self.num_rows,
            num_cols: self.num_cols,
        }
    }
    #[inline]
    unsafe fn wrap_trans_mat<T: MatrixLike>(&self, mat: T) -> Self::TransposedMatrixWrapped<T> {
        RSMatrixExpr{
            mat,
            num_rows: self.num_cols,
            num_cols: self.num_rows
        }
    }
    #[inline]
    unsafe fn wrap_col_vec<T: VectorLike>(&self, vec: T) -> Self::ColWrapped<T> {
        RSVectorExpr{
            vec,
            size: self.num_rows
        }
    }
    #[inline]
    unsafe fn wrap_row_vec<T: VectorLike>(&self, vec: T) -> Self::RowWrapped<T> {
        RSVectorExpr{
            vec,
            size: self.num_cols
        }
    }

    #[inline]
    fn decompose(self) -> (Self::ColBuilder, Self::RowBuilder) {
        (RSVectorExprBuilder{size: self.num_rows}, RSVectorExprBuilder{size: self.num_cols})
    }

    #[inline]
    fn dimensions(&self) -> (usize, usize) {
        (self.num_rows, self.num_cols)
    }
}

impl MatrixBuilderCompose<RSVectorExprBuilder> for RSVectorExprBuilder {
    type Composition = RSMatrixExprBuilder;

    #[inline]
    fn compose(self, other: RSVectorExprBuilder) -> Self::Composition {
        RSMatrixExprBuilder{
            num_rows: self.size,
            num_cols: other.size,
        }
    }
}



impl<const D1: usize, const D2: usize> MatrixBuilderUnion<MatrixExprBuilder<D1, D2>>
    for MatrixExprBuilder<D1, D2>
{
    type Union = Self;

    fn union(self, _: MatrixExprBuilder<D1, D2>) -> Self::Union {
        self
    }
}

impl<const D1: usize, const D2: usize> MatrixBuilderUnion<MatrixExprBuilder<D1, D2>>
    for RSMatrixExprBuilder
{
    type Union = MatrixExprBuilder<D1, D2>;

    fn union(self, other: MatrixExprBuilder<D1, D2>) -> Self::Union {
        assert_eq!(self.dimensions(), other.dimensions(), "math_vector error: cannot combine 2 matrixes of different size");
        MatrixExprBuilder
    }
}

impl<const D1: usize, const D2: usize> MatrixBuilderUnion<RSMatrixExprBuilder>
    for MatrixExprBuilder<D1, D2>
{
    type Union = MatrixExprBuilder<D1, D2>;

    fn union(self, other: RSMatrixExprBuilder) -> Self::Union {
        assert_eq!(self.dimensions(), other.dimensions(), "math_vector error: cannot combine 2 matrixes of different size");
        MatrixExprBuilder
    }
}

impl MatrixBuilderUnion<RSMatrixExprBuilder> for RSMatrixExprBuilder {
    type Union = Self;

    fn union(self, other: RSMatrixExprBuilder) -> Self::Union {
        assert_eq!(self.dimensions(), other.dimensions(), "math_vector error: cannot combine 2 matrixes of different size");
        self
    }
}