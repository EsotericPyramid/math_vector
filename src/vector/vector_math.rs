use crate::{
    trait_specialization_utils::*, 
    vector::{
        self, 
        VectorOps, 
        vec_util_traits::*,
        vector_structs::*,
    },
    util_traits::HasOutput,
};
use std::ops::{Index, IndexMut};
use alga::general::{ComplexField, RealField};

pub trait ConcreteVectorExpr: VectorOps + Index<usize> + IndexMut<usize> where 
    Self: From<Vec<<Self as Index<usize>>::Output>>,
    Vec<<Self as Index<usize>>::Output>: From<Self>,
    <Self as VectorOps>::Unwrapped: Get<Item = <Self as Index<usize>>::Output>,

    <Self as Index<usize>>::Output: Sized,
{
    type Referenced<'a>: VectorOps + Index<usize, Output = <Self as Index<usize>>::Output> 
    where 
        <Self::Referenced<'a> as VectorOps>::Unwrapped: Get<Item = &'a <Self::Referenced<'a> as Index<usize>>::Output>,
        <Self as Index<usize>>::Output: 'a,
        Self: 'a,
    ;
    type ReferencedMut<'a>: VectorOps + Index<usize, Output = <Self as Index<usize>>::Output> + IndexMut<usize>
    where 
        <Self::ReferencedMut<'a> as VectorOps>::Unwrapped: Get<Item = &'a mut <Self::ReferencedMut<'a> as Index<usize>>::Output>,
        <Self as Index<usize>>::Output: 'a,
        Self: 'a,
    ;

    
    fn borrow<'a>(&'a self) -> Self::Referenced<'a> where 
        <Self::Referenced<'a> as VectorOps>::Unwrapped: Get<Item = &'a <Self::Referenced<'a> as Index<usize>>::Output>,
        <Self as Index<usize>>::Output: 'a,
        Self: 'a
    ;

    fn borrow_mut<'a>(&'a mut self) -> Self::ReferencedMut<'a> where 
        <Self::ReferencedMut<'a> as VectorOps>::Unwrapped: Get<Item = &'a mut <Self::ReferencedMut<'a> as Index<usize>>::Output>,
        <Self as Index<usize>>::Output: 'a,
        Self: 'a,
    ;
}

pub trait InnerProduct<V: VectorOps, F: ComplexField> where V::Unwrapped: Get<Item = F> {
    fn inner_prod(lhs_vector: V, rhs_vector: V) -> F  where V::Unwrapped: Get<Item = F>;
}

pub struct DotProduct;

/// NOTE: although the dot product may often be thought of as an inner product, thats only true for Real Fields (where conjugate(x) = x), thus the RealField trait bound
impl<V: VectorOps, F: ComplexField + RealField> InnerProduct<V, F> for DotProduct where 
    V::Unwrapped: Get<Item = F>,
    V::Builder: VectorBuilderUnion<V::Builder>,
    (<V::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
    (<V::Unwrapped as HasReuseBuf>::FstHandleBool, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
    (<V::Unwrapped as HasReuseBuf>::SndHandleBool, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
    (<V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
    (<V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
    (<V::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
    (<(<V::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y): FilterPair,
    <<V::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<vector::vector_structs::VecDot<V::Unwrapped, V::Unwrapped, F>>: VectorOps,
    <<<V::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<vector::vector_structs::VecDot<V::Unwrapped, V::Unwrapped, F>> as VectorOps>::Unwrapped: HasOutput<Output = F>,
    <<<V::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<VecDot<V::Unwrapped, V::Unwrapped, F>> as VectorOps>::Unwrapped: HasReuseBuf<
        BoundTypes = <<<<V::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<VecDot<V::Unwrapped, V::Unwrapped, F>> as VectorOps>::Unwrapped as Get>::BoundItems
    >,
{
    fn inner_prod(lhs_vector: V, rhs_vector: V) -> F  where V::Unwrapped: Get<Item = F> {
        lhs_vector.dot(rhs_vector).consume()
    }
}

pub struct EuclideanInnerProduct;

impl<V: VectorOps, F: ComplexField> InnerProduct<V, F> for EuclideanInnerProduct where 
    V::Unwrapped: Get<Item = F>,
    V::Builder: VectorBuilderUnion<V::Builder>,
    (<V::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
    (<V::Unwrapped as HasReuseBuf>::FstHandleBool, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
    (<V::Unwrapped as HasReuseBuf>::SndHandleBool, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
    (<V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
    (<V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
    (<V::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
    (<(<V::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y): FilterPair,
    <<V::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<vector::vector_structs::VecEuclidInnerProd<V::Unwrapped, V::Unwrapped, F>>: VectorOps,
    <<<V::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<vector::vector_structs::VecEuclidInnerProd<V::Unwrapped, V::Unwrapped, F>> as VectorOps>::Unwrapped: HasOutput<Output = F>,
    <<<V::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<VecEuclidInnerProd<V::Unwrapped, V::Unwrapped, F>> as VectorOps>::Unwrapped: HasReuseBuf<
        BoundTypes = <<<<V::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<VecEuclidInnerProd<V::Unwrapped, V::Unwrapped, F>> as VectorOps>::Unwrapped as Get>::BoundItems
    >,
{
    fn inner_prod(lhs_vector: V, rhs_vector: V) -> F  where V::Unwrapped: Get<Item = F> {
        lhs_vector.euclidean_inner_prod(rhs_vector).consume()
    }
}