//! Structs implementing VectorBuilder to wrap VectorLikes with sizing information

use super::vec_util_traits::{
    VectorBuilder, 
    VectorBuilderUnion, 
    VectorLike,
};
use super::{
    VectorExpr, 
    RSVectorExpr,
};

/// a simple const sized VectorBuilder
#[derive(Clone, Copy)]
pub struct VectorExprBuilder<const D: usize>;

impl<const D: usize> VectorBuilder for VectorExprBuilder<D> {
    type Wrapped<T: VectorLike> = VectorExpr<T, D>;
    
    unsafe fn wrap<T: VectorLike>(&self, vec: T) -> Self::Wrapped<T> {VectorExpr(vec)}
}

/// a simple runtime sized VectorBuilder
#[derive(Clone, Copy)]
pub struct RSVectorExprBuilder{pub(crate) size: usize}

impl VectorBuilder for RSVectorExprBuilder {
    type Wrapped<T: VectorLike> = RSVectorExpr<T>;

    unsafe fn wrap<T: VectorLike>(&self, vec: T) -> Self::Wrapped<T> {
        RSVectorExpr{vec, size: self.size}
    }
}



impl<const D: usize> VectorBuilderUnion<VectorExprBuilder<D>> for VectorExprBuilder<D> {
    type Union = Self;

    fn union(self, _: VectorExprBuilder<D>) -> Self::Union {self}
}

impl VectorBuilderUnion<RSVectorExprBuilder> for RSVectorExprBuilder {
    type Union = Self;

    fn union(self, other: RSVectorExprBuilder) -> Self::Union {
        assert!(self.size == other.size, "math_vector error: cannot combine 2 vectors of different size"); //FIXME: scuff error message
        self
    }
}


impl<const D: usize> VectorBuilderUnion<VectorExprBuilder<D>> for RSVectorExprBuilder {
    type Union = VectorExprBuilder<D>;

    fn union(self, other: VectorExprBuilder<D>) -> Self::Union {
        assert!(self.size == D, "math_vector error: cannot combine 2 vectors of different size"); //FIXME: scuff error message
        other
    }
}

impl<const D: usize> VectorBuilderUnion<RSVectorExprBuilder> for VectorExprBuilder<D> {
    type Union = VectorExprBuilder<D>;

    fn union(self, other: RSVectorExprBuilder) -> Self::Union {
        assert!(other.size == D, "math_vector error: cannot combine 2 vectors of different size"); //FIXME: scuff error message
        self
    }
}