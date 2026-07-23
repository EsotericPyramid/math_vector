//! Structs implementing VectorBuilder to wrap VectorLikes with sizing information

use super::vector_exprs::*;
use super::vector_math::GenericInnerProduct;
use super::vector_structs::{VectorArray, VectorSlice, VecGenerator, VecIndexGenerator, VecFilled};

use super::vec_util_traits::VectorLike;
use super::{RSVectorExpr, VectorExpr, VectorInnerProdExpr, VectorOps};
use std::mem::ManuallyDrop;

/// A way for a type to "build" wrappers around VectorLikes which encode sizing information
/// 
/// or in other words, implementors carry minimal sizing information which can be applied to VectorLikes
pub trait VectorBuilder: Copy {
    /// The parametrized wrapper generated around VectorLikes which this builder generates
    type Wrapped<T: VectorLike>: VectorOps<Unwrapped = T, Builder = Self>;

    /// Wrap the given [`VectorLike`] in this builder's corresponding wrapper. 
    /// 
    /// Safety:
    /// - the [`VectorLike`] is unused
    /// - the [`VectorLike`] can be taken as this builder's size (most will have exactly one correct size but some like [`super::vector_structs::VecMap`] do not)
    unsafe fn wrap<T: VectorLike>(&self, vec: T) -> Self::Wrapped<T>;

    /// get the size indicated by this builder
    fn size(&self) -> usize;

    /// *generate* a Vector of this builder's size using the given closure (FnMut) with no inputs
    /// 
    /// this is only a generator, nothing is calculated or stored on this call.
    #[inline]
    fn generate<F: FnMut() -> O, O>(&self, f: F) -> Self::Wrapped<VecGenerator<F, O>> {
        unsafe { self.wrap(VecGenerator(f)) }
    }

    /// *generate* a Vector of this builder's size using the given closure (FnMut) with an input of the current index
    /// 
    /// this is only a generator, nothing is calculated or stored on this call.
    #[inline]
    fn index_generate<F: FnMut(usize) -> O, O>(&self, f: F) -> Self::Wrapped<VecIndexGenerator<F, O>> {
        unsafe { self.wrap(VecIndexGenerator(f)) }
    }

    /// *generate* a Vector of this builder's size by filling it with copies of the given `filler`
    /// 
    /// this is only a generator, nothing is calculated or stored on this call.
    /// see [`InitializableVectorBuilder::new_filled`] for a version with does *immediately* allocate memory (+ a more convienent type over evaluating)
    #[inline]
    fn gen_filled<T: Copy>(&self, filler: T) -> Self::Wrapped<VecFilled<T>> {
        unsafe { self.wrap(VecFilled(filler)) }
    }

    /// *generate* a Vector of this builder's size by filling with 0's
    /// 
    /// this is only a generator, nothing is calculated or stored on this call.
    /// see [`InitializableVectorBuilder::new_zeroed`] for a version with does *immediately* allocate memory (+ a more convienent type over evaluating)
    #[inline]
    fn gen_zeroed<T: num_traits::Zero + Copy>(&self) -> Self::Wrapped<VecFilled<T>> {
        self.gen_filled(T::zero())
    }

    /// *generate* a Vector of this builder's size by filling with 1's
    /// 
    /// this is only a generator, nothing is calculated or stored on this call.
    /// see [`InitializableVectorBuilder::new_oned`] for a version with does *immediately* allocate memory (+ a more convienent type over evaluating)
    #[inline]
    fn gen_oned<T: num_traits::One + Copy>(&self) -> Self::Wrapped<VecFilled<T>> {
        self.gen_filled(T::one())
    }
}

pub trait InitializableVectorBuilder: VectorBuilder {
    // source code reader note: wonder why ConcreteInner is needed?, so do I.
    //.     Good luck trying to write this w/o it, every time I tried it results in E0275
    type ConcreteInner<T: Sized>: VectorLike<Item = T>;
    type Concrete<T: Sized>: ConcreteVectorExpr<Unwrapped = Self::ConcreteInner<T>, Builder = Self, Output = T>;

    /// *allocate* a Vector of this builder's size filled with copies of the given `filler`
    /// 
    /// there is also a generator variant of this method which avoids allocating memory unecessarily: [`VectorBuilder::gen_filled`].
    /// This may be prefered over that to get a `ConcreteVectorExpr` with a cleaner typing that simply eval-ing in generic implementations
    fn new_filled<T: Copy>(&self, filler: T) -> Self::Concrete<T>;
    
    /// *allocate* a Vector of this builder's size filled with 0's
    /// 
    /// there is also a generator variant of this method which avoids allocating memory unecessarily: [`VectorBuilder::gen_zeroed`].
    /// This may be prefered over that to get a `ConcreteVectorExpr` with a cleaner typing that simply eval-ing in generic implementations
    fn new_zeroed<T: num_traits::Zero + Copy>(&self) -> Self::Concrete<T>
    {
        self.new_filled(T::zero())
    }

    /// *allocate* a Vector of this builder's size filled with 1's
    /// 
    /// there is also a generator variant of this method which avoids allocating memory unecessarily: [`VectorBuilder::gen_oned`].
    /// This may be prefered over that to get a `ConcreteVectorExpr` with a cleaner typing that simply eval-ing in generic implementations
    fn new_oned<T: num_traits::One + Copy>(&self) -> Self::Concrete<T>
    {
        self.new_filled(T::one())
    }
}

/// Enables an union operation between 2 VectorBuilders into a single [`VectorBuilder`]
pub trait VectorBuilderUnion<T: VectorBuilder>: VectorBuilder {
    /// the resulting type of the Union
    type Union: VectorBuilder;

    /// union 2 VectorBuilders into a single VectorBuilder
    /// 
    /// additionally checks that the sizing information of each VectorBuilder is equal
    fn union(self, other: T) -> Self::Union;
}

/// a simple const sized [`VectorBuilder`]
/// 
/// this is the builder equivalent of [`VectorExpr`]
#[derive(Clone, Copy)]
pub struct VectorExprBuilder<const D: usize>;

impl<const D: usize> VectorExprBuilder<D> {
    /// Create a new [`VectorExprBuilder`]
    /// 
    /// the sizing isn't expressed in this function but rather in the type calling it or in the type returned.
    /// so, to generate a dimension 10 builder, use `VectorExprBuilder::<10>::new()`
    /// 
    /// note: since this struct has no fields, it is equivalent to simply use `VectorExprBuilder::<10>`
    pub fn new() -> Self {VectorExprBuilder}
}

impl<const D: usize> VectorBuilder for VectorExprBuilder<D> {
    type Wrapped<T: VectorLike> = VectorExpr<T, D>;

    unsafe fn wrap<T: VectorLike>(&self, vec: T) -> Self::Wrapped<T> {
        VectorExpr(vec)
    }
    fn size(&self) -> usize {
        D
    }
}

impl<const D: usize> InitializableVectorBuilder for VectorExprBuilder<D> {
    type ConcreteInner<T: Sized> = VectorArray<T, D>;
    type Concrete<T> = MathVector<T, D>;

    #[inline]
    fn new_filled<T: Copy>(&self, filler: T) -> Self::Concrete<T> {
        unsafe { self.wrap(VectorArray(ManuallyDrop::new([filler; D]))) }
    }
}

/// a simple const sized [`VectorBuilder`]
/// 
/// this is the builder equivalent of [`VectorExpr`]
#[derive(Clone, Copy)]
pub struct HeapedVectorExprBuilder<const D: usize>;

impl<const D: usize> HeapedVectorExprBuilder<D> {
    /// Create a new [`HeapedVectorExprBuilder`]
    /// 
    /// the sizing isn't expressed in this function but rather in the type calling it or in the type returned.
    /// so, to generate a dimension 10 builder, use `HeapedVectorExprBuilder::<10>::new()`
    /// 
    /// note: since this struct has no fields, it is equivalent to simply use `HeapedVectorExprBuilder::<10>`
    pub fn new() -> Self {HeapedVectorExprBuilder}
}

impl<const D: usize> VectorBuilder for HeapedVectorExprBuilder<D> {
    type Wrapped<T: VectorLike> = HeapedVectorExpr<T, D>;

    unsafe fn wrap<T: VectorLike>(&self, vec: T) -> Self::Wrapped<T> {
        HeapedVectorExpr(VectorExpr(vec))
    }
    fn size(&self) -> usize {
        D
    }
}

impl<const D: usize> InitializableVectorBuilder for HeapedVectorExprBuilder<D> {
    type ConcreteInner<T: Sized> = Box<VectorArray<T, D>>;
    type Concrete<T> = HeapedMathVector<T, D>;

    #[inline]
    fn new_filled<T: Copy>(&self, filler: T) -> Self::Concrete<T> {
        unsafe { self.wrap(Box::new(VectorArray(ManuallyDrop::new([filler; D])))) }
    }
}


/// a simple runtime sized [`VectorBuilder`]
/// 
/// this is the builder equivalent of [`RSVectorExpr`]
#[derive(Clone, Copy)]
pub struct RSVectorExprBuilder {
    pub size: usize,
}

impl RSVectorExprBuilder {
    /// Create a new [`RSVectorExprBuilder`] with the given size
    pub fn new(size: usize) -> Self {
        RSVectorExprBuilder { size }
    }
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

impl InitializableVectorBuilder for RSVectorExprBuilder {
    type ConcreteInner<T: Sized> = VectorSlice<T>;
    type Concrete<T: Sized> = RSMathVector<T>;

    fn new_filled<T: Copy>(&self, filler: T) -> Self::Concrete<T> {
        unsafe { self.wrap(
            VectorSlice(std::mem::transmute::<
                Box<[T]>, Box<ManuallyDrop<[T]>>
            >(vec![filler; self.size].into_boxed_slice()))
        ) }
    }
}

/// a [`VectorBuilder`] monad which additionally adds an [inner product](https://en.wikipedia.org/wiki/Inner_product_space) to its wrapped [`VectorLike`] 
/// 
/// this is the builder equivalent of [`VectorInnerProdExpr`]
#[derive(Clone, Copy)]
pub struct VectorInnerProdExprBuilder<B: VectorBuilder, IP: GenericInnerProduct> {
    /// the underlying builder which provides sizing information
    pub builder: B,
    /// the attached innerproduct
    /// 
    /// although it is only required to impl GenericInnerProduct, it is expected
    /// to also impl `InnerProduct<F>` for some F: [`alga::general::ComplexField`] to be useful
    pub inner_prod: IP,
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

/// for now, this is the only impl of Union on [`VectorInnerProdExprBuilder`] since its the only one which is obvious.
/// In the future, some may be added for between it and other builders.
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

impl<const D: usize> VectorBuilderUnion<VectorExprBuilder<D>> for HeapedVectorExprBuilder<D> {
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


impl<const D: usize> VectorBuilderUnion<HeapedVectorExprBuilder<D>> for VectorExprBuilder<D> {
    type Union = HeapedVectorExprBuilder<D>;

    fn union(self, other: HeapedVectorExprBuilder<D>) -> Self::Union {
        other
    }
}

impl<const D: usize> VectorBuilderUnion<HeapedVectorExprBuilder<D>> for HeapedVectorExprBuilder<D> {
    type Union = HeapedVectorExprBuilder<D>;

    fn union(self, other: HeapedVectorExprBuilder<D>) -> Self::Union {
        other
    }
}

impl<const D: usize> VectorBuilderUnion<HeapedVectorExprBuilder<D>> for RSVectorExprBuilder {
    type Union = HeapedVectorExprBuilder<D>;

    fn union(self, other: HeapedVectorExprBuilder<D>) -> Self::Union {
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

impl<const D: usize> VectorBuilderUnion<RSVectorExprBuilder> for HeapedVectorExprBuilder<D> {
    type Union = HeapedVectorExprBuilder<D>;

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

