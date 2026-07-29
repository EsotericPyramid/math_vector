//! A collection of more in depth mathematical methods
//! 
//! These methods are relegated to here either because they aren't commonly needed
//! or because they aren't implemented fully lazily

use crate::{
    trait_specialization_utils::*, 
    util_traits::HasOutput, 
    vector::{
        VectorOps, 
        vec_util_traits::*, 
        vector_builders::{
            VectorBuilder,
            VectorBuilderUnion,
        }, 
        vector_structs::*,
        vector_exprs::*,
    }
};
use std::ops::Index;
use alga::general::{ComplexField, RealField};

/// a helper trait that implies that the implementor implements InnerProduct<F> for some F
pub trait GenericInnerProduct: Copy {}

/// A trait defining a [inner product](https://en.wikipedia.org/wiki/Inner_product_space) between 2 vectors
/// 
/// note: many of these methods involve evaluation
pub trait InnerProduct<F: ComplexField>: GenericInnerProduct
where 
{
    /// the inner [`VectorLike`] which preforms the inner product between `V1` and `V2`
    type InnerProductInner<V1: VectorOps, V2: VectorOps>: VectorLike where 
        V1::Unwrapped: Get<Item = F>,
        V2::Unwrapped: Get<Item = F>,
        (<V1::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V2::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<V1::Unwrapped as HasReuseBuf>::FstHandleBool, <V2::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
        (<V1::Unwrapped as HasReuseBuf>::SndHandleBool, <V2::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
        (<V1::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V2::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<V1::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V2::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y): FilterPair,
    ;

    /// calculate the inner product bewteen `lhs_vector` and `rhs_vector`
    fn inner_prod<V1: VectorOps, V2: VectorOps>(lhs_vector: V1, rhs_vector: V2) -> 
        <<V1::Builder as VectorBuilderUnion<V2::Builder>>::Union as VectorBuilder>::Wrapped<Self::InnerProductInner<V1, V2>>
    where
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
    ;

    /// eagerly calculate the inner product bewteen `lhs_vector` and `rhs_vector`
    #[inline]
    fn eager_inner_prod<V1: VectorOps, V2: VectorOps>(lhs_vector: V1, rhs_vector: V2) -> <
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
        V1::Builder: VectorBuilderUnion<V2::Builder>,
        (<V1::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V2::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<V1::Unwrapped as HasReuseBuf>::FstHandleBool, <V2::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
        (<V1::Unwrapped as HasReuseBuf>::SndHandleBool, <V2::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
        (<V1::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V2::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<V1::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V2::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y): FilterPair,
        <<<V1::Builder as VectorBuilderUnion<V2::Builder>>::Union as VectorBuilder>::Wrapped<Self::InnerProductInner<V1, V2>> as VectorOps>::Unwrapped: HasOutput<Output = <
            (<(<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y) as FilterPair
        >::Filtered<
            <
                (<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as FilterPair
            >::Filtered<
                <V1::Unwrapped as HasOutput>::Output, <V2::Unwrapped as HasOutput>::Output
            >, 
            F
        >>,
        <<<V1::Builder as VectorBuilderUnion<V2::Builder>>::Union as VectorBuilder>::Wrapped<Self::InnerProductInner<V1, V2>> as VectorOps>::Unwrapped: HasReuseBuf<
            BoundTypes = <<<<V1::Builder as VectorBuilderUnion<V2::Builder>>::Union as VectorBuilder>::Wrapped<Self::InnerProductInner<V1, V2>> as VectorOps>::Unwrapped as Get>::BoundItems
        >,
    {
        Self::inner_prod(lhs_vector, rhs_vector).consume()
    }

    /// eagerly calculate the inner product bewteen `lhs_vector` and `rhs_vector` while cleanly keeping the outputs and result of the inner product separate
    /// 
    /// That makes it much better suited for parametrized situations
    #[inline]
    fn raw_eager_inner_prod<V1: VectorOps, V2: VectorOps>(lhs_vector: V1, rhs_vector: V2) -> (<
        (<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as FilterPair
    >::Filtered<
        <V1::Unwrapped as HasOutput>::Output, <V2::Unwrapped as HasOutput>::Output
    >, F) 
    where 
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
        <<<V1::Builder as VectorBuilderUnion<V2::Builder>>::Union as VectorBuilder>::Wrapped<Self::InnerProductInner<V1, V2>> as VectorOps>::Unwrapped: HasOutput<Output = <
            (<(<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y) as FilterPair
        >::Filtered<
            <
                (<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as FilterPair
            >::Filtered<
                <V1::Unwrapped as HasOutput>::Output, <V2::Unwrapped as HasOutput>::Output
            >, 
            F
        >>,
        <<<V1::Builder as VectorBuilderUnion<V2::Builder>>::Union as VectorBuilder>::Wrapped<Self::InnerProductInner<V1, V2>> as VectorOps>::Unwrapped: HasReuseBuf<
            BoundTypes = <<<<V1::Builder as VectorBuilderUnion<V2::Builder>>::Union as VectorBuilder>::Wrapped<Self::InnerProductInner<V1, V2>> as VectorOps>::Unwrapped as Get>::BoundItems
        >,
    {
        unsafe {<(<(<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y)>::defilter(Self::eager_inner_prod(lhs_vector, rhs_vector))}
    }    
}

/// an struct implementing [`InnerProduct`] for the typical dot product
#[derive(Clone, Copy)]
pub struct DotProduct;

impl GenericInnerProduct for DotProduct {}

impl<F: ComplexField + RealField> InnerProduct<F> for DotProduct {
    type InnerProductInner<V1: VectorOps, V2: VectorOps> = VecDot<V1::Unwrapped, V2::Unwrapped, F> where 
        V1::Unwrapped: Get<Item = F>,
        V2::Unwrapped: Get<Item = F>,
        (<V1::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V2::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<V1::Unwrapped as HasReuseBuf>::FstHandleBool, <V2::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
        (<V1::Unwrapped as HasReuseBuf>::SndHandleBool, <V2::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
        (<V1::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V2::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<V1::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V2::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y): FilterPair,
    ;

    #[inline]
    fn inner_prod<V1: VectorOps, V2: VectorOps>(lhs_vector: V1, rhs_vector: V2) -> 
        <<V1::Builder as VectorBuilderUnion<V2::Builder>>::Union as VectorBuilder>::Wrapped<Self::InnerProductInner<V1, V2>>
    where
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
    {
        lhs_vector.dot(rhs_vector)
    }
}

/// an struct implementing [`InnerProduct`] in the typical way for [complex vectors](https://en.wikipedia.org/wiki/Dot_product#Complex_vectors)
#[derive(Clone, Copy)]
pub struct EuclideanInnerProduct;

impl GenericInnerProduct for EuclideanInnerProduct {}

impl<F: ComplexField> InnerProduct<F> for EuclideanInnerProduct {
    type InnerProductInner<V1: VectorOps, V2: VectorOps> = VecEuclidInnerProd<V1::Unwrapped, V2::Unwrapped, F> where 
        V1::Unwrapped: Get<Item = F>,
        V2::Unwrapped: Get<Item = F>,
        (<V1::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V2::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<V1::Unwrapped as HasReuseBuf>::FstHandleBool, <V2::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
        (<V1::Unwrapped as HasReuseBuf>::SndHandleBool, <V2::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
        (<V1::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V2::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<V1::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V2::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y): FilterPair,
    ;

    #[inline]
    fn inner_prod<V1: VectorOps, V2: VectorOps>(lhs_vector: V1, rhs_vector: V2) -> 
        <<V1::Builder as VectorBuilderUnion<V2::Builder>>::Union as VectorBuilder>::Wrapped<Self::InnerProductInner<V1, V2>>
    where
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
    {
        lhs_vector.euclidean_inner_prod(rhs_vector)
    }
}

/// an extension of [`VectorOps`] for vector exprs which have inner products
pub trait VectorInnerProdOps: VectorOps {
    /// the inner product type of the implementor
    type InnerProd: GenericInnerProduct;

    /// perform the inner product between `self` and another vector
    #[inline]
    fn inner_prod<V: VectorInnerProdOps<InnerProd = Self::InnerProd>, F: ComplexField>(self, other: V) -> 
        <<Self::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<<Self::InnerProd as InnerProduct<F>>::InnerProductInner<Self, V>>
    where 
        Self::InnerProd: InnerProduct<F>,
        Self::Unwrapped: Get<Item = F>,
        V::Unwrapped: Get<Item = F>,
        Self::Builder: VectorBuilderUnion<V::Builder>,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndHandleBool, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y): FilterPair,
    {
        <Self::InnerProd as InnerProduct<F>>::inner_prod(self, other)
    } 

    /// eagerly performs the inner product between `self` and another vector
    #[inline]
    fn eager_inner_prod<V: VectorInnerProdOps<InnerProd = Self::InnerProd>, F: ComplexField>(self, other: V) -> <
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y) as FilterPair
    >::Filtered<
        <
            (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as FilterPair
        >::Filtered<
            <Self::Unwrapped as HasOutput>::Output, <V::Unwrapped as HasOutput>::Output
        >, 
        F
    >
    where 
        Self::InnerProd: InnerProduct<F>,
        Self::Unwrapped: Get<Item = F>,
        V::Unwrapped: Get<Item = F>,
        Self::Builder: VectorBuilderUnion<V::Builder>,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndHandleBool, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y): FilterPair,
        <<<Self::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<<Self::InnerProd as InnerProduct<F>>::InnerProductInner<Self, V>> as VectorOps>::Unwrapped: HasOutput<Output = <
            (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y) as FilterPair
        >::Filtered<
            <
                (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as FilterPair
            >::Filtered<
                <Self::Unwrapped as HasOutput>::Output, <V::Unwrapped as HasOutput>::Output
            >, 
            F
        >>,
        <<<Self::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<<Self::InnerProd as InnerProduct<F>>::InnerProductInner<Self, V>> as VectorOps>::Unwrapped: HasReuseBuf<
            BoundTypes = <<<<Self::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<<Self::InnerProd as InnerProduct<F>>::InnerProductInner<Self, V>> as VectorOps>::Unwrapped as Get>::BoundItems
        >,
    {
        <Self::InnerProd as InnerProduct<F>>::eager_inner_prod(self, other)
    } 

    /// eagerly performs the inner product between `self` and another vector while cleanly keeping the outputs and result of the inner product separate
    /// 
    /// That makes it much better suited for parametrized situations
    #[inline]
    fn raw_eager_inner_prod<V: VectorInnerProdOps<InnerProd = Self::InnerProd>, F: ComplexField>(self, other: V) -> (<
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as FilterPair
    >::Filtered<
        <Self::Unwrapped as HasOutput>::Output, <V::Unwrapped as HasOutput>::Output
    >, F) 
    where 
        Self::InnerProd: InnerProduct<F>,
        Self::Unwrapped: Get<Item = F>,
        V::Unwrapped: Get<Item = F>,
        Self::Builder: VectorBuilderUnion<V::Builder>,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndHandleBool, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y): FilterPair,
        <<<Self::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<<Self::InnerProd as InnerProduct<F>>::InnerProductInner<Self, V>> as VectorOps>::Unwrapped: HasOutput<Output = <
            (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y) as FilterPair
        >::Filtered<
            <
                (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as FilterPair
            >::Filtered<
                <Self::Unwrapped as HasOutput>::Output, <V::Unwrapped as HasOutput>::Output
            >, 
            F
        >>,
        <<<Self::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<<Self::InnerProd as InnerProduct<F>>::InnerProductInner<Self, V>> as VectorOps>::Unwrapped: HasReuseBuf<
            BoundTypes = <<<<Self::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<<Self::InnerProd as InnerProduct<F>>::InnerProductInner<Self, V>> as VectorOps>::Unwrapped as Get>::BoundItems
        >,
    {
        <Self::InnerProd as InnerProduct<F>>::raw_eager_inner_prod(self, other)
    }
     
    /// projects this vector onto `onto`
    /// 
    /// note: this eagerly evaluates this vector
    #[inline]
    fn proj<'a, V: VectorInnerProdOps<InnerProd = Self::InnerProd> + ConcreteVectorExpr + 'a, F: ComplexField + 'a>(self, onto: &'a V) -> <
        <V::Copied<'a> as VectorOps>::Builder as VectorBuilder
    >::Wrapped<
        VecAttachOutput<
            VecMulR<
                VecCopy<
                    'a, 
                    V::ReferencedInner<'a>, 
                    F
                >, 
                F
            >, 
            <
                (<Self::Unwrapped as HasOutput>::OutputBool, <V::ReferencedInner<'a> as HasOutput>::OutputBool) as FilterPair
            >::Filtered<
                <Self::Unwrapped as HasOutput>::Output, <V::ReferencedInner<'a> as HasOutput>::Output
            >,
            <(<Self::Unwrapped as HasOutput>::OutputBool, <V::ReferencedInner<'a> as HasOutput>::OutputBool) as TyBoolPair>::Or,
        >
    >
    where 
        Self::InnerProd: InnerProduct<F>,
        Self::Unwrapped: Get<Item = F>,
        V::Unwrapped: Get<Item = V::Output>,
        V: Index<usize, Output = F>,
        V::Copied<'a>: VectorInnerProdOps<InnerProd = Self::InnerProd>,
        Self::Builder: VectorBuilderUnion<<V::Copied<'a> as VectorOps>::Builder>,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V::ReferencedInner<'a> as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <V::ReferencedInner<'a> as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndHandleBool, <V::ReferencedInner<'a> as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V::ReferencedInner<'a> as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V::ReferencedInner<'a> as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::ReferencedInner<'a> as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::ReferencedInner<'a> as HasOutput>::OutputBool) as TyBoolPair>::Or, Y): FilterPair,
        <<<Self::Builder as VectorBuilderUnion<<V::Copied<'a> as VectorOps>::Builder>>::Union as VectorBuilder>::Wrapped<<Self::InnerProd as InnerProduct<F>>::InnerProductInner<Self, V::Copied<'a>>> as VectorOps>::Unwrapped: HasOutput<Output = <
            (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::ReferencedInner<'a> as HasOutput>::OutputBool) as TyBoolPair>::Or, Y) as FilterPair
        >::Filtered<
            <
                (<Self::Unwrapped as HasOutput>::OutputBool, <V::ReferencedInner<'a> as HasOutput>::OutputBool) as FilterPair
            >::Filtered<
                <Self::Unwrapped as HasOutput>::Output, <V::ReferencedInner<'a> as HasOutput>::Output
            >, 
            F
        >>,
        <<<Self::Builder as VectorBuilderUnion<<V::Copied<'a> as VectorOps>::Builder>>::Union as VectorBuilder>::Wrapped<<Self::InnerProd as InnerProduct<F>>::InnerProductInner<Self, V::Copied<'a>>> as VectorOps>::Unwrapped: HasReuseBuf<
            BoundTypes = <<<<Self::Builder as VectorBuilderUnion<<V::Copied<'a> as VectorOps>::Builder>>::Union as VectorBuilder>::Wrapped<<Self::InnerProd as InnerProduct<F>>::InnerProductInner<Self, V::Copied<'a>>> as VectorOps>::Unwrapped as Get>::BoundItems
        >,
        (
            <V::ReferencedInner<'a> as HasOutput>::OutputBool,
            <(<Self::Unwrapped as HasOutput>::OutputBool, <V::ReferencedInner<'a> as HasOutput>::OutputBool) as TyBoolPair>::Or
        ): FilterPair
    {
        let (output, proj_mag) = self.raw_eager_inner_prod(onto.copy());
        onto.copy().mul_r(proj_mag).maybe_attach_output(output)
    }

    /// gets the orthogonal part of this vector with `with`
    /// 
    /// note: this eagerly evaluates this vector
    #[inline]
    fn orthogonal_part<'a, V: VectorInnerProdOps<InnerProd = Self::InnerProd> + ConcreteVectorExpr + 'a, F: ComplexField + 'a>(&'a self, with: &'a V) -> <
        <<Self::Copied<'a> as VectorOps>::Builder as VectorBuilderUnion<<V::Copied<'a> as VectorOps>::Builder>>::Union as VectorBuilder
    >::Wrapped<
        VecAttachOutput<
            VecSub<
                <Self::Copied<'a> as VectorOps>::Unwrapped,
                VecMulR<
                    VecCopy<
                        'a, 
                        V::ReferencedInner<'a>, 
                        F
                    >, 
                    F
                >, 
            >,
            <
                (<Self::ReferencedInner<'a> as HasOutput>::OutputBool, <V::ReferencedInner<'a> as HasOutput>::OutputBool) as FilterPair
            >::Filtered<
                <Self::ReferencedInner<'a> as HasOutput>::Output, <V::ReferencedInner<'a> as HasOutput>::Output
            >,
            <(<Self::ReferencedInner<'a> as HasOutput>::OutputBool, <V::ReferencedInner<'a> as HasOutput>::OutputBool) as TyBoolPair>::Or,
        >
    >
    where 
        Self: ConcreteVectorExpr + 'a,
        Self::InnerProd: InnerProduct<F>,
        Self::Unwrapped: Get<Item = F> + IsRepeatable,
        Self: Index<usize, Output = F>,
        Self::Copied<'a>: VectorInnerProdOps<InnerProd = Self::InnerProd>,
        V::Unwrapped: Get<Item = V::Output>,
        V: Index<usize, Output = F>,
        V::Copied<'a>: VectorInnerProdOps<InnerProd = Self::InnerProd>,
        <Self::Copied<'a> as VectorOps>::Builder: VectorBuilderUnion<<V::Copied<'a> as VectorOps>::Builder>,
        (<Self::ReferencedInner<'a> as HasReuseBuf>::BoundHandlesBool, <V::ReferencedInner<'a> as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::ReferencedInner<'a> as HasReuseBuf>::FstHandleBool, <V::ReferencedInner<'a> as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::ReferencedInner<'a> as HasReuseBuf>::SndHandleBool, <V::ReferencedInner<'a> as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::ReferencedInner<'a> as HasReuseBuf>::FstOwnedBufferBool, <V::ReferencedInner<'a> as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::ReferencedInner<'a> as HasReuseBuf>::SndOwnedBufferBool, <V::ReferencedInner<'a> as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::ReferencedInner<'a> as HasOutput>::OutputBool, <V::ReferencedInner<'a> as HasOutput>::OutputBool): FilterPair,
        (<(<Self::ReferencedInner<'a> as HasOutput>::OutputBool, <V::ReferencedInner<'a> as HasOutput>::OutputBool) as TyBoolPair>::Or, Y): FilterPair,
        <<<<Self::Copied<'a> as VectorOps>::Builder as VectorBuilderUnion<<V::Copied<'a> as VectorOps>::Builder>>::Union as VectorBuilder>::Wrapped<<<Self::Copied<'a> as VectorInnerProdOps>::InnerProd as InnerProduct<F>>::InnerProductInner<Self::Copied<'a>, V::Copied<'a>>> as VectorOps>::Unwrapped: HasOutput<Output = <
            (<(<Self::ReferencedInner<'a> as HasOutput>::OutputBool, <V::ReferencedInner<'a> as HasOutput>::OutputBool) as TyBoolPair>::Or, Y) as FilterPair
        >::Filtered<
            <
                (<Self::ReferencedInner<'a> as HasOutput>::OutputBool, <V::ReferencedInner<'a> as HasOutput>::OutputBool) as FilterPair
            >::Filtered<
                <Self::ReferencedInner<'a> as HasOutput>::Output, <V::ReferencedInner<'a> as HasOutput>::Output
            >, 
            F
        >>,
        <<<<Self::Copied<'a> as VectorOps>::Builder as VectorBuilderUnion<<V::Copied<'a> as VectorOps>::Builder>>::Union as VectorBuilder>::Wrapped<<<Self::Copied<'a> as VectorInnerProdOps>::InnerProd as InnerProduct<F>>::InnerProductInner<Self::Copied<'a>, V::Copied<'a>>> as VectorOps>::Unwrapped: HasReuseBuf<
            BoundTypes = <<<<<Self::Copied<'a> as VectorOps>::Builder as VectorBuilderUnion<<V::Copied<'a> as VectorOps>::Builder>>::Union as VectorBuilder>::Wrapped<<<Self::Copied<'a> as VectorInnerProdOps>::InnerProd as InnerProduct<F>>::InnerProductInner<Self::Copied<'a>, V::Copied<'a>>> as VectorOps>::Unwrapped as Get>::BoundItems
        >,
        (
            <Self::Unwrapped as HasOutput>::OutputBool, 
            <V::ReferencedInner<'a> as HasOutput>::OutputBool
        ): FilterPair,
        (
            <(
                <Self::ReferencedInner<'a> as HasOutput>::OutputBool, 
                <V::ReferencedInner<'a> as HasOutput>::OutputBool
            ) as TyBoolPair>::Or, 
            <(
                <Self::ReferencedInner<'a> as HasOutput>::OutputBool, 
                <V::ReferencedInner<'a> as HasOutput>::OutputBool
            ) as TyBoolPair>::Or,
        ): FilterPair
    {
        let (output, proj_mag) = self.copy().raw_eager_inner_prod(with.copy());
        self.copy().sub(with.copy().mul_r(proj_mag)).maybe_attach_output(output)
    }
}