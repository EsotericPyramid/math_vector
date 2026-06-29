use crate::{
    trait_specialization_utils::*, util_traits::HasOutput, vector::{
        ArrayVectorOps, VectorEvalOps, VectorOps, vec_util_traits::*, vector_builders::VectorInnerProdExprBuilder, vector_structs::*
    }
};
use std::{mem::ManuallyDrop, ops::{Index, IndexMut}};
use alga::general::{ComplexField, RealField};

pub trait ConcreteVectorExpr: VectorOps + IndexMut<usize> where 
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
    type Copied<'a>: VectorOps<Unwrapped = VecCopy<'a, Self::ReferencedInner<'a>, Self::Output>>
    where
        Self::Output: Copy,
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

    fn copy<'a>(&'a self) -> Self::Copied<'a> where
        Self::Output: Copy,
        <Self as Index<usize>>::Output: 'a,
        Self: 'a
    ;
}

/// implies that the implementor implements InnerProduct<F> for some F
pub trait GenericInnerProduct: Copy {}

/// really implementors of this should be ZSTs
pub trait InnerProduct<F: ComplexField>: GenericInnerProduct
where 
{
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

// monad lol
pub struct VectorInnerProdExpr<V: VectorOps, IP: GenericInnerProduct> {
    pub(crate) vec: V,
    pub(crate) inner_prod: IP,
}

impl<V: VectorOps, IP: GenericInnerProduct> std::ops::Deref for VectorInnerProdExpr<V, IP> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        &self.vec
    }
}

impl<V: VectorOps, IP: GenericInnerProduct> std::ops::DerefMut for VectorInnerProdExpr<V, IP> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vec
    }
}

impl<V: VectorOps, IP: GenericInnerProduct> AsRef<V> for VectorInnerProdExpr<V, IP> {
    fn as_ref(&self) -> &V {
        &self.vec
    }
}

impl<V: VectorOps, IP: GenericInnerProduct> AsMut<V> for VectorInnerProdExpr<V, IP> {
    fn as_mut(&mut self) -> &mut V {
        &mut self.vec
    }
}

impl<V: VectorOps + Index<Idx>, IP: GenericInnerProduct, Idx> Index<Idx> for VectorInnerProdExpr<V, IP> {
    type Output = V::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.vec[index]
    }
}

impl<V: VectorOps + IndexMut<Idx>, IP: GenericInnerProduct, Idx> IndexMut<Idx> for VectorInnerProdExpr<V, IP> {
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        &mut self.vec[index]
    }
}

unsafe impl<V: VectorOps, IP: GenericInnerProduct> VectorOps for VectorInnerProdExpr<V, IP> {
    type Builder = VectorInnerProdExprBuilder<V::Builder, IP>;
    type Unwrapped = V::Unwrapped;

    fn size(&self) -> usize {
        self.vec.size()
    }

    fn get_builder(&self) -> Self::Builder {
        VectorInnerProdExprBuilder {
            builder: self.vec.get_builder(),
            inner_prod: self.inner_prod,
        }
    }

    fn unwrap(self) -> Self::Unwrapped {
        self.vec.unwrap()
    }
}

impl<V: ArrayVectorOps<D>, IP: GenericInnerProduct, const D: usize> ArrayVectorOps<D> for VectorInnerProdExpr<V, IP> {}

impl<V: VectorEvalOps, IP: GenericInnerProduct> VectorEvalOps for VectorInnerProdExpr<V, IP> {
    type MaybeCreateBuffer<T: VectorLike> = V::MaybeCreateBuffer<T>
        where
            <<T as HasReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
            (
                <T as HasReuseBuf>::FstHandleBool,
                <<T as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
            ): SelectPair,
            (
                <T as HasReuseBuf>::FstOwnedBufferBool,
                <<T as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
            ): TyBoolPair;

    fn maybe_create_buffer(self) -> Self::MaybeCreateBuffer<Self::Unwrapped>
    where
        <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
        (
            <Self::Unwrapped as HasReuseBuf>::FstHandleBool,
            <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
        ): SelectPair,
        (
            <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool,
            <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
        ): TyBoolPair,
        Self: Sized
    {
        self.vec.maybe_create_buffer()
    }
}

impl<V: ConcreteVectorExpr, IP: GenericInnerProduct> ConcreteVectorExpr for VectorInnerProdExpr<V, IP> 
where 
    <V as VectorOps>::Unwrapped: Get<Item = <V as Index<usize>>::Output>,
    <V as Index<usize>>::Output: Sized,
{
    type ReferencedInner<'a> = V::ReferencedInner<'a>
        where 
            <Self as Index<usize>>::Output: 'a,
            Self: 'a,;
    type Referenced<'a> = VectorInnerProdExpr<V::Referenced<'a>, IP>
        where
            <Self as Index<usize>>::Output: 'a,
            Self: 'a,;
    type Copied<'a> = VectorInnerProdExpr<V::Copied<'a>, IP>
        where
            Self::Output: Copy,
            <Self as Index<usize>>::Output: 'a,
            Self: 'a,;
    type ReferencedMutInner<'a> = V::ReferencedMutInner<'a> 
        where 
            <Self as Index<usize>>::Output: 'a,
            Self: 'a,;
    type ReferencedMut<'a> = VectorInnerProdExpr<V::ReferencedMut<'a>, IP>
        where 
            <Self as Index<usize>>::Output: 'a,
            Self: 'a,;

    fn borrow<'a>(&'a self) -> Self::Referenced<'a> where 
        <Self::Referenced<'a> as VectorOps>::Unwrapped: Get<Item = &'a <Self::Referenced<'a> as Index<usize>>::Output>,
        <Self as Index<usize>>::Output: 'a,
        Self: 'a
    {
        VectorInnerProdExpr {
            vec: self.vec.borrow(),
            inner_prod: self.inner_prod
        }
    }

    fn borrow_mut<'a>(&'a mut self) -> Self::ReferencedMut<'a> where 
        <Self::Referenced<'a> as VectorOps>::Unwrapped: Get<Item = &'a <Self::Referenced<'a> as Index<usize>>::Output>,
        <Self as Index<usize>>::Output: 'a,
        Self: 'a
    {
        VectorInnerProdExpr {
            vec: self.vec.borrow_mut(),
            inner_prod: self.inner_prod
        }
    }

    fn copy<'a>(&'a self) -> Self::Copied<'a> where
        Self::Output: Copy,
        <Self as Index<usize>>::Output: 'a,
        Self: 'a
    {
        VectorInnerProdExpr {
            vec: self.vec.copy(),
            inner_prod: self.inner_prod
        }
    }
}

impl<V: VectorOps, IP: GenericInnerProduct> VectorInnerProdOps for VectorInnerProdExpr<V, IP> {
    type InnerProd = IP;
}

pub trait VectorInnerProdOps: VectorOps {
    type InnerProd: GenericInnerProduct;

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
            <VecMulR<VecCopy<'a, V::ReferencedInner<'a>, F>, F> as HasOutput>::OutputBool,
            <(<Self::Unwrapped as HasOutput>::OutputBool, <V::ReferencedInner<'a> as HasOutput>::OutputBool) as TyBoolPair>::Or
        ): FilterPair
    {
        let (output, proj_mag) = self.raw_eager_inner_prod(onto.copy());
        let onto_copy = onto.copy();
        let onto_builder = onto_copy.get_builder();
        unsafe { onto_builder.wrap(VecAttachOutput {
            vec: VecMulR {
                vec: onto_copy.unwrap(),
                scalar: proj_mag
            },
            output: ManuallyDrop::new(output),
            marker: std::marker::PhantomData,
        }) }
    }
    
}