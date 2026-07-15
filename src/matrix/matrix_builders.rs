//! Structs implementing MatrixBuilder to wrap MatrixLikes with sizing information

use crate::{
    matrix::{
        MatrixOps, mat_util_traits::MatrixLike, matrix_exprs::{
            MatrixExpr,
            RSMatrixExpr,
        }, matrix_structs::{MatGenerator, MatIdentityGenerator, MatIndexGenerator},
    }, vector::{
        VectorOps, vec_util_traits::VectorLike, vector_builders::*, vector_exprs::{
            RSVectorExpr, VectorExpr, 
        },
    },
};


/// A way for a type to "build" wrappers around MatrixLikes which encode sizing information
/// or in other words, implementors carry minimal sizing information which can be applied to MatrixLikes
pub trait MatrixBuilder: Clone {
    /// wrapper directly indicated by this builder
    type MatrixWrapped<T: MatrixLike>: MatrixOps;
    /// transposition of the wrapper indicated
    type TransposedMatrixWrapped<T: MatrixLike>: MatrixOps;
    /// wrapper for an indicated column
    type ColWrapped<T: VectorLike>: VectorOps;
    /// wrapper for an indicated row
    type RowWrapped<T: VectorLike>: VectorOps;

    //FIXME (HRTBs): for<T: VectorLike> Self::ColBuilder::Wrapped<T> == Self::ColWrapped
    /// a builder wrapping columns like this builder
    type ColBuilder: VectorBuilder;
    /// a builder wrapping rows like this builder
    type RowBuilder: VectorBuilder;

    /// creates wrapper directly indicated by this builder
    unsafe fn wrap_mat<T: MatrixLike>(&self, mat: T) -> Self::MatrixWrapped<T>;
    /// creates transposition of the wrapper indicated
    unsafe fn wrap_trans_mat<T: MatrixLike>(&self, mat: T) -> Self::TransposedMatrixWrapped<T>;
    /// creates wrapper for an indicated column
    unsafe fn wrap_col_vec<T: VectorLike>(&self, vec: T) -> Self::ColWrapped<T>;
    /// creates wrapper for an indicated row
    unsafe fn wrap_row_vec<T: VectorLike>(&self, vec: T) -> Self::RowWrapped<T>;

    //FIXME (above is source of issue): currently requires correct implementation even though trait is not unsafe
    /// decomposes this matrix builder into a column and row vector builders
    fn decompose(self) -> (Self::ColBuilder, Self::RowBuilder);

    /// get the dimensions of this builder in `(num_rows, num_cols)` format
    fn dimensions(&self) -> (usize, usize);


    /// generates a Matrix with this builder using the given closure (FnMut) with no inputs
    fn generate<F: FnMut() -> O, O>(&self, f: F) -> Self::MatrixWrapped<MatGenerator<F, O>> {
        unsafe { self.wrap_mat(MatGenerator(f)) }
    }

    /// generates a Matrix with this builder using the given closure (FnMut) given the column and row indices as input
    fn index_generate<F: FnMut(usize, usize) -> O, O>(&self, f: F) -> Self::MatrixWrapped<MatIndexGenerator<F, O>> {
        unsafe { self.wrap_mat(MatIndexGenerator(f)) }
    }

    /// generates a Identity matrix with this builder
    fn generate_identity<T: Copy + num_traits::One + num_traits::Zero>(&self) -> Self::MatrixWrapped<MatIdentityGenerator<T>> {
        let (rows, columns) = self.dimensions();
        assert_eq!(rows, columns, "math_vector error: Cannot make a non-square identity matrix");
        
        unsafe { self.wrap_mat(MatIdentityGenerator {
            zero: T::zero(),
            one: T::one(),
        })}
    }
}

/// Enables an union operation between 2 MatrixBuilders into a single MatrixBuilder
pub trait MatrixBuilderUnion<T: MatrixBuilder>: MatrixBuilder {
    /// the resulting type of the Union
    type Union: MatrixBuilder;

    /// union 2 MatrixBuilders into a single MatrixBuilder
    /// additionally checks that the sizing information of each MatrixBuilder is equal
    fn union(self, other: T) -> Self::Union;
}

/// Enables 2 vector builders to construct a matrix builder
///
/// syntax: `ColBuilder: MatrixBuilderCompose<RowBuilder>`
pub trait MatrixBuilderCompose<T: VectorBuilder>: VectorBuilder {
    //FIXME (HRTBs): for<T: VectorLike> Self::Composition::ColBuilder::Wrapped<T> == Self::Wrapped
    /// the resulting type of the composition
    type Composition: MatrixBuilder;

    /// composes the 2 VectorBuilders into 1 MatrixBuilder
    ///
    /// self is the column builder, other is the row builder
    fn compose(self, other: T) -> Self::Composition;
}


/// a simple const sized MatrixBuilder
#[derive(Clone)]
pub struct MatrixExprBuilder<const D1: usize, const D2: usize>;

impl<const D1: usize, const D2: usize> MatrixExprBuilder<D1, D2> {
    pub fn new() -> Self {MatrixExprBuilder}
}

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
    pub num_rows: usize,
    pub num_cols: usize,
}

impl RSMatrixExprBuilder {
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        RSMatrixExprBuilder { num_rows, num_cols }
    }
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