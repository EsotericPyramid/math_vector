//! Structs implementing VectorBuilder to wrap VectorLikes with sizing information (and potentially more like inner products)

use crate::vector::vector_math::{GenericInnerProduct, VectorInnerProdExpr};

use super::vec_util_traits::{VectorBuilder, VectorBuilderUnion, VectorLike};
use super::{RSVectorExpr, VectorExpr};

/// a simple const sized VectorBuilder
#[derive(Clone, Copy)]
pub struct VectorExprBuilder<const D: usize>;

impl<const D: usize> VectorBuilder for VectorExprBuilder<D> {
    type Wrapped<T: VectorLike> = VectorExpr<T, D>;

    unsafe fn wrap<T: VectorLike>(&self, vec: T) -> Self::Wrapped<T> {
        VectorExpr(vec)
    }
    fn size(&self) -> usize {
        D
    }
}

/// a simple runtime sized VectorBuilder
#[derive(Clone, Copy)]
pub struct RSVectorExprBuilder {
    pub(crate) size: usize,
}

impl VectorBuilder for RSVectorExprBuilder {
    type Wrapped<T: VectorLike> = RSVectorExpr<T>;

    unsafe fn wrap<T: VectorLike>(&self, vec: T) -> Self::Wrapped<T> {
        RSVectorExpr {
            vec,
            size: self.size,
        }
    }
    fn size(&self) -> usize {
        self.size
    }
}

#[derive(Clone, Copy)]
pub struct VectorInnerProdExprBuilder<B: VectorBuilder, IP: GenericInnerProduct> {
    pub(crate) builder: B,
    pub(crate) inner_prod: IP,
}

impl<B: VectorBuilder, IP: GenericInnerProduct> VectorBuilder for VectorInnerProdExprBuilder<B, IP> {
    type Wrapped<T: VectorLike> = VectorInnerProdExpr<B::Wrapped<T>, IP>;

    fn size(&self) -> usize {
        self.builder.size()
    }

    unsafe fn wrap<T: VectorLike>(&self, vec: T) -> Self::Wrapped<T> {
        VectorInnerProdExpr{
            vec: unsafe {
                self.builder.wrap(vec)
            },
            inner_prod: self.inner_prod,
        }
    }
}

// for now, this is the only impl of Union on it since its the only one which is obvious
impl<B1: VectorBuilderUnion<B2>, B2: VectorBuilder, IP: GenericInnerProduct> VectorBuilderUnion<VectorInnerProdExprBuilder<B2, IP>> for VectorInnerProdExprBuilder<B1, IP> {
    type Union = VectorInnerProdExprBuilder<<B1 as VectorBuilderUnion<B2>>::Union, IP>;

    fn union(self, other: VectorInnerProdExprBuilder<B2, IP>) -> Self::Union {
        VectorInnerProdExprBuilder {
            builder: self.builder.union(other.builder),
            inner_prod: self.inner_prod
        }
    }
}


impl<const D: usize> VectorBuilderUnion<VectorExprBuilder<D>> for VectorExprBuilder<D> {
    type Union = Self;

    fn union(self, _: VectorExprBuilder<D>) -> Self::Union {
        self
    }
}

impl<const D: usize> VectorBuilderUnion<VectorExprBuilder<D>> for RSVectorExprBuilder {
    type Union = VectorExprBuilder<D>;

    fn union(self, other: VectorExprBuilder<D>) -> Self::Union {
        assert!(
            self.size == D,
            "math_vector error: cannot combine 2 vectors of different size"
        ); //FIXME: scuff error message
        other
    }
}

impl<const D: usize> VectorBuilderUnion<RSVectorExprBuilder> for VectorExprBuilder<D> {
    type Union = VectorExprBuilder<D>;

    fn union(self, other: RSVectorExprBuilder) -> Self::Union {
        assert!(
            other.size == D,
            "math_vector error: cannot combine 2 vectors of different size"
        ); //FIXME: scuff error message
        self
    }
}

impl VectorBuilderUnion<RSVectorExprBuilder> for RSVectorExprBuilder {
    type Union = Self;

    fn union(self, other: RSVectorExprBuilder) -> Self::Union {
        assert!(
            self.size == other.size,
            "math_vector error: cannot combine 2 vectors of different size"
        ); //FIXME: scuff error message
        self
    }
}

