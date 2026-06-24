use crate::{
    trait_specialization_utils::*, util_traits::HasOutput, vector::{
        vec_util_traits::*, vector_structs::*, VectorOps
    }
};
use std::ops::{Index, IndexMut};
use alga::general::{ComplexField, RealField};

pub trait ConcreteVectorExpr: VectorOps + Index<usize> + IndexMut<usize> where 
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

pub trait InnerProduct<F: ComplexField>
where 
{
    type InnerProductInner<V1: VectorOps, V2: VectorOps>: VectorLike where 
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

    fn inner_prod<V1: VectorOps, V2: VectorOps>(lhs_vector: V1, rhs_vector: V2) -> <
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
    ;

    fn raw_inner_prod<V1: VectorOps, V2: VectorOps>(lhs_vector: V1, rhs_vector: V2) -> (<
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
        unsafe {<(<(<V1::Unwrapped as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y)>::defilter(Self::inner_prod(lhs_vector, rhs_vector))}
    }    
}

pub struct DotProduct;

impl<F: ComplexField + RealField> InnerProduct<F> for DotProduct {
    type InnerProductInner<V1: VectorOps, V2: VectorOps> = VecDot<V1::Unwrapped, V2::Unwrapped, F> where 
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

    fn inner_prod<V1: VectorOps, V2: VectorOps>(lhs_vector: V1, rhs_vector: V2) -> <
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
        lhs_vector.dot(rhs_vector).consume()
    }
}

pub struct EuclideanInnerProduct;

impl<F: ComplexField> InnerProduct<F> for EuclideanInnerProduct {
    type InnerProductInner<V1: VectorOps, V2: VectorOps> = VecEuclidInnerProd<V1::Unwrapped, V2::Unwrapped, F> where 
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

    fn inner_prod<V1: VectorOps, V2: VectorOps>(lhs_vector: V1, rhs_vector: V2) -> <
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
        lhs_vector.euclidean_inner_prod(rhs_vector).consume()
    }
}