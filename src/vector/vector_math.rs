use crate::{
    trait_specialization_utils::*, 
    vector::{
        VectorOps, 
        vec_util_traits::*,
        vector_structs::*,
    },
    util_traits::HasOutput,
};
use std::ops::{Index, IndexMut};
use alga::general::{ComplexField, RealField};

pub trait ConcreteVectorExpr: VectorOps + Index<usize> + IndexMut<usize> where 
    //Self: From<Vec<<Self as Index<usize>>::Output>>,
    //Vec<<Self as Index<usize>>::Output>: From<Self>,
    <Self as VectorOps>::Unwrapped: Get<Item = <Self as Index<usize>>::Output>,

    <Self as Index<usize>>::Output: Sized,
{
    type ReferencedInner<'a>: VectorLike<Item = &'a <Self as Index<usize>>::Output> 
    where 
        <Self as Index<usize>>::Output: 'a,
        Self: 'a,
    ;
    type Referenced<'a>: VectorOps<Unwrapped = Self::ReferencedInner<'a>> + Index<usize, Output = <Self as Index<usize>>::Output> 
    where
        <Self as Index<usize>>::Output: 'a,
        Self: 'a,
    ;
    type ReferencedMutInner<'a>: VectorLike<Item = &'a mut <Self as Index<usize>>::Output> 
    where 
        <Self as Index<usize>>::Output: 'a,
        Self: 'a,
    ;
    type ReferencedMut<'a>: VectorOps<Unwrapped = Self::ReferencedMutInner<'a>> + Index<usize, Output = <Self as Index<usize>>::Output> + IndexMut<usize>
    where 
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

pub trait InnerProduct<V1: VectorOps, V2: VectorOps, F: ComplexField> 
where 
    V1::Unwrapped: Get<Item = F>,
    V2::Unwrapped: Get<Item = F>,
{
    fn inner_prod(lhs_vector: V1, rhs_vector: V2) -> <
        (<(<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y) as FilterPair
    >::Filtered<
        <
            (<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as FilterPair
        >::Filtered<
            <V1::Unwrapped as HasOutput>::Output, <V2::Unwrapped as HasOutput>::Output
        >, 
        F
    >
    where 
        V1::Unwrapped: Get<Item = F>,
        V2::Unwrapped: Get<Item = F>,
        (<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y): FilterPair,
    ;
    fn raw_inner_prod(lhs_vector: V1, rhs_vector: V2) -> (<
        (<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as FilterPair
    >::Filtered<
        <V1::Unwrapped as HasOutput>::Output, <V2::Unwrapped as HasOutput>::Output
    >, F) 
    where 
        V1::Unwrapped: Get<Item = F>,
        V2::Unwrapped: Get<Item = F>,
        (<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y): FilterPair,
    {
        unsafe {<(<(<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y)>::defilter(Self::inner_prod(lhs_vector, rhs_vector))}
    }
}

pub struct DotProduct;

/// NOTE: although the dot product may often be thought of as an inner product, thats only true for Real Fields (where conjugate(x) = x), thus the RealField trait bound
impl<V1: VectorOps, V2: VectorOps, F: ComplexField + RealField> InnerProduct<V1, V2, F> for DotProduct where 
    V1::Unwrapped: Get<Item = F>,
    V2::Unwrapped: Get<Item = F>,
    V1::Builder: VectorBuilderUnion<V2::Builder>,
    (<V1::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V2::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
    (<V1::Unwrapped as HasReuseBuf>::FstHandleBool, <V2::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
    (<V1::Unwrapped as HasReuseBuf>::SndHandleBool, <V2::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
    (<V1::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V2::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
    (<V1::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V2::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
    (<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool): FilterPair,
    (<(<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y): FilterPair,
    <<V1::Builder as VectorBuilderUnion<V2::Builder>>::Union as VectorBuilder>::Wrapped<VecDot<V1::Unwrapped, V2::Unwrapped, F>>: VectorOps,
    <<<V1::Builder as VectorBuilderUnion<V2::Builder>>::Union as VectorBuilder>::Wrapped<VecDot<V1::Unwrapped, V2::Unwrapped, F>> as VectorOps>::Unwrapped: HasOutput<Output = <
        (<(<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y) as FilterPair
    >::Filtered<
        <
            (<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as FilterPair
        >::Filtered<
            <V1::Unwrapped as HasOutput>::Output, <V2::Unwrapped as HasOutput>::Output
        >, 
        F
    >>,
    <<<V1::Builder as VectorBuilderUnion<V2::Builder>>::Union as VectorBuilder>::Wrapped<VecDot<V1::Unwrapped, V2::Unwrapped, F>> as VectorOps>::Unwrapped: HasReuseBuf<
        BoundTypes = <<<<V1::Builder as VectorBuilderUnion<V2::Builder>>::Union as VectorBuilder>::Wrapped<VecDot<V1::Unwrapped, V2::Unwrapped, F>> as VectorOps>::Unwrapped as Get>::BoundItems
    >,
{
    fn inner_prod(lhs_vector: V1, rhs_vector: V2) -> <
        (<(<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y) as FilterPair
    >::Filtered<
        <
            (<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as FilterPair
        >::Filtered<
            <V1::Unwrapped as HasOutput>::Output, <V2::Unwrapped as HasOutput>::Output
        >, 
        F
    > where 
        V1::Unwrapped: Get<Item = F>,
        V2::Unwrapped: Get<Item = F>,
        (<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y): FilterPair,
    {
        lhs_vector.dot(rhs_vector).consume()
    }
}

pub struct EuclideanInnerProduct;

impl<V1: VectorOps, V2: VectorOps, F: ComplexField> InnerProduct<V1, V2, F> for EuclideanInnerProduct where 
    V1::Unwrapped: Get<Item = F>,
    V2::Unwrapped: Get<Item = F>,
    V1::Builder: VectorBuilderUnion<V2::Builder>,
    (<V1::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V2::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
    (<V1::Unwrapped as HasReuseBuf>::FstHandleBool, <V2::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
    (<V1::Unwrapped as HasReuseBuf>::SndHandleBool, <V2::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
    (<V1::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V2::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
    (<V1::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V2::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
    (<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool): FilterPair,
    (<(<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y): FilterPair,
    <<V1::Builder as VectorBuilderUnion<V2::Builder>>::Union as VectorBuilder>::Wrapped<VecEuclidInnerProd<V1::Unwrapped, V2::Unwrapped, F>>: VectorOps,
    <<<V1::Builder as VectorBuilderUnion<V2::Builder>>::Union as VectorBuilder>::Wrapped<VecEuclidInnerProd<V1::Unwrapped, V2::Unwrapped, F>> as VectorOps>::Unwrapped: HasOutput<Output = <
        (<(<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y) as FilterPair
    >::Filtered<
        <
            (<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as FilterPair
        >::Filtered<
            <V1::Unwrapped as HasOutput>::Output, <V2::Unwrapped as HasOutput>::Output
        >, 
        F
    >>,
    <<<V1::Builder as VectorBuilderUnion<V2::Builder>>::Union as VectorBuilder>::Wrapped<VecEuclidInnerProd<V1::Unwrapped, V2::Unwrapped, F>> as VectorOps>::Unwrapped: HasReuseBuf<
        BoundTypes = <<<<V1::Builder as VectorBuilderUnion<V2::Builder>>::Union as VectorBuilder>::Wrapped<VecEuclidInnerProd<V1::Unwrapped, V2::Unwrapped, F>> as VectorOps>::Unwrapped as Get>::BoundItems
    >,
{
    fn inner_prod(lhs_vector: V1, rhs_vector: V2) -> <
        (<(<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y) as FilterPair
    >::Filtered<
        <
            (<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as FilterPair
        >::Filtered<
            <V1::Unwrapped as HasOutput>::Output, <V2::Unwrapped as HasOutput>::Output
        >, 
        F
    >  
    where 
        V1::Unwrapped: Get<Item = F>,
        V2::Unwrapped: Get<Item = F>,
        (<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y): FilterPair,
    {
        lhs_vector.euclidean_inner_prod(rhs_vector).consume()
    }
}

pub trait Norm<V: VectorOps, F: ComplexField>: InnerProduct<V, V, F> where V::Unwrapped: Get<Item = F> {
    fn sqr_norm(vector: V) -> <(<V::Unwrapped as HasOutput>::OutputBool, Y) as FilterPair>::Filtered<<V::Unwrapped as HasOutput>::Output, F> where 
        (<V::Unwrapped as HasOutput>::OutputBool, Y): FilterPair;
    fn raw_sqr_norm(vector: V) -> (<V::Unwrapped as HasOutput>::Output, F) where 
        (<V::Unwrapped as HasOutput>::OutputBool, Y): FilterPair 
    {
        unsafe {<(<V::Unwrapped as HasOutput>::OutputBool, Y)>::defilter(Self::sqr_norm(vector))}
    }
    fn norm(vector: V) -> <(<V::Unwrapped as HasOutput>::OutputBool, Y) as FilterPair>::Filtered<<V::Unwrapped as HasOutput>::Output, F> where 
        (<V::Unwrapped as HasOutput>::OutputBool, Y): FilterPair
    {
        let (output, norm) = Self::raw_norm(vector);
        <(<V::Unwrapped as HasOutput>::OutputBool, Y)>::filter(output, norm)
    }
    fn raw_norm(vector: V) -> (<V::Unwrapped as HasOutput>::Output, F) where 
        (<V::Unwrapped as HasOutput>::OutputBool, Y): FilterPair 
    {
        let (output, sqr_norm) = Self::raw_sqr_norm(vector);
        (output, sqr_norm.sqrt())
    }
}

impl<V: VectorOps, F: ComplexField + RealField> Norm<V, F> for DotProduct where 
    V::Unwrapped: Get<Item = F>, 
    Self: InnerProduct<V, V, F>,
    (<V::Unwrapped as HasOutput>::OutputBool, Y): FilterPair,
    <V::Builder as VectorBuilder>::Wrapped<VecSqrMag<V::Unwrapped, F>>: VectorOps,
    <<V::Builder as VectorBuilder>::Wrapped<VecSqrMag<V::Unwrapped, F>> as VectorOps>::Unwrapped: HasOutput<Output = 
        <(<V::Unwrapped as HasOutput>::OutputBool, Y) as FilterPair>::Filtered<<V::Unwrapped as HasOutput>::Output, F>
    >,
    <<V::Builder as VectorBuilder>::Wrapped<VecSqrMag<V::Unwrapped, F>> as VectorOps>::Unwrapped: HasReuseBuf<
        BoundTypes = <<<V::Builder as VectorBuilder>::Wrapped<VecSqrMag<V::Unwrapped, F>> as VectorOps>::Unwrapped as Get>::BoundItems,
    >
{
    fn sqr_norm(vector: V) -> <(<V::Unwrapped as HasOutput>::OutputBool, Y) as FilterPair>::Filtered<<V::Unwrapped as HasOutput>::Output, F> where 
        (<V::Unwrapped as HasOutput>::OutputBool, Y): FilterPair
    {
        vector.sqr_mag().consume()
    }
}

impl<V: VectorOps, F: ComplexField> Norm<V, F> for EuclideanInnerProduct where 
    V::Unwrapped: Get<Item = F>, 
    Self: InnerProduct<V, V, F>,
    (<V::Unwrapped as HasOutput>::OutputBool, Y): FilterPair,
    <V::Builder as VectorBuilder>::Wrapped<VecSqrEuclidMag<V::Unwrapped, F>>: VectorOps,
    <<V::Builder as VectorBuilder>::Wrapped<VecSqrEuclidMag<V::Unwrapped, F>> as VectorOps>::Unwrapped: HasOutput<Output = 
        <(<V::Unwrapped as HasOutput>::OutputBool, Y) as FilterPair>::Filtered<<V::Unwrapped as HasOutput>::Output, F>
    >,
    <<V::Builder as VectorBuilder>::Wrapped<VecSqrEuclidMag<V::Unwrapped, F>> as VectorOps>::Unwrapped: HasReuseBuf<
        BoundTypes = <<<V::Builder as VectorBuilder>::Wrapped<VecSqrEuclidMag<V::Unwrapped, F>> as VectorOps>::Unwrapped as Get>::BoundItems,
    >
{
    fn sqr_norm(vector: V) -> <(<V::Unwrapped as HasOutput>::OutputBool, Y) as FilterPair>::Filtered<<V::Unwrapped as HasOutput>::Output, F> where 
        (<V::Unwrapped as HasOutput>::OutputBool, Y): FilterPair
    {
        vector.sqr_euclid_mag().consume()
    }
}

