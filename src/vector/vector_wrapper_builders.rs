use super::vec_util_traits::{VectorWrapperBuilder, CombinableVectorWrapperBuilder, VectorLike};
use super::{VectorExpr, RSVectorExpr};

#[derive(Clone)]
pub struct VectorExprBuilder<const D: usize>;

impl<const D: usize> VectorWrapperBuilder for VectorExprBuilder<D> {
    type Wrapped<T: VectorLike> = VectorExpr<T, D>;
    
    unsafe fn wrap<T: VectorLike>(&self, vec: T) -> Self::Wrapped<T> {VectorExpr(vec)}
}


#[derive(Clone)]
pub struct RSVectorExprBuilder{size: usize}

impl VectorWrapperBuilder for RSVectorExprBuilder {
    type Wrapped<T: VectorLike> = RSVectorExpr<T>;

    unsafe fn wrap<T: VectorLike>(&self, vec: T) -> Self::Wrapped<T> {
        RSVectorExpr{vec, size: self.size}
    }
}



impl<const D: usize> CombinableVectorWrapperBuilder<VectorExprBuilder<D>> for VectorExprBuilder<D> {
    type Union = Self;

    fn union(self, _: VectorExprBuilder<D>) -> Self::Union {self}
}

impl CombinableVectorWrapperBuilder<RSVectorExprBuilder> for RSVectorExprBuilder {
    type Union = Self;

    fn union(self, other: RSVectorExprBuilder) -> Self::Union {
        assert!(self.size == other.size, "math_vector error: cannot combine 2 vectors of different size"); //FIXME: scuff error message
        self
    }
}

impl<const D: usize> CombinableVectorWrapperBuilder<VectorExprBuilder<D>> for RSVectorExprBuilder {
    type Union = VectorExprBuilder<D>;

    fn union(self, other: VectorExprBuilder<D>) -> Self::Union {
        assert!(self.size == D, "math_vector error: cannot combine 2 vectors of different size"); //FIXME: scuff error message
        other
    }
}

impl<const D: usize> CombinableVectorWrapperBuilder<RSVectorExprBuilder> for VectorExprBuilder<D> {
    type Union = VectorExprBuilder<D>;

    fn union(self, other: RSVectorExprBuilder) -> Self::Union {
        assert!(other.size == D, "math_vector error: cannot combine 2 vectors of different size"); //FIXME: scuff error message
        self
    }
}