//! Module containing all to do with Vectors and basic operations to do on them

use crate::{
    matrix::{
        mat_util_traits::{Get2D, MatrixBuilder}, matrix_structs::{MatColVectorExprs, MatrixColumn}, MatrixOps
    },
    trait_specialization_utils::*,
    util_traits::HasOutput, vector::vector_math::ConcreteVectorExpr,
};
use std::{
    iter::Sum,
    marker::PhantomData,
    mem::{self, ManuallyDrop, MaybeUninit},
    ops::*,
    ptr,
    slice::SliceIndex,
};

pub mod vec_util_traits;
pub mod vector_builders;
pub mod vector_structs;
pub mod vector_math;
pub mod vector_initialization;

use alga::general::ComplexField;
use vec_util_traits::*;
use vector_builders::*;
use vector_structs::*;

/// A const sized vector wrapper
/// T: the underlying VectorLike type, generally inferred or generic
/// D: the size of the vector
#[repr(transparent)]
pub struct VectorExpr<V: VectorLike, const D: usize>(pub(crate) V); // note: VectorExpr only holds fully unused VectorLike objects

impl<V: VectorLike, const D: usize> VectorExpr<V, D> {
    /// evaluates the VectorExpr and returns the resulting vector (on the heap) alongside its output (if present)
    /// if the VectorExpr has no item (& thus results in a vector w/ ZST elements), see consume to not return that vector
    /// will try to use the first buffer if available (fails if the provided buffer is not bindable to the output)
    ///
    /// Warning:
    /// this method may cause a stack overflow if not compiled with `--release`
    ///
    /// Note:
    /// methods like sum, product, or fold can place build up outputs
    ///
    /// output is generally nested 2 element tuples
    /// newer values to the right
    /// binary operators merge the output of the 2 vectors
    #[inline]
    pub fn heap_eval(self) -> <VecBind<VecMaybeCreateHeapArray<V, V::Item, D>> as HasOutput>::Output
    where
        <V::FstHandleBool as TyBool>::Neg: Filter,
        (V::FstHandleBool, <V::FstHandleBool as TyBool>::Neg): SelectPair,
        (V::FstOwnedBufferBool, <V::FstHandleBool as TyBool>::Neg): TyBoolPair,
        (
            V::OutputBool,
            <(V::FstOwnedBufferBool, <V::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or,
        ): FilterPair,
        (V::BoundHandlesBool, Y): FilterPair,
        VecBind<VecMaybeCreateHeapArray<V, V::Item, D>>: HasReuseBuf<
            BoundTypes = <VecBind<VecMaybeCreateHeapArray<V, V::Item, D>> as Get>::BoundItems,
        >,
    {
        self.maybe_create_heap_array().bind().consume()
    }
}

impl<V: VectorLike + IsRepeatable, const D: usize> VectorExpr<V, D> {
    /// Retrieves an arbitrary value from a repeatable VectorExpr
    pub fn get(&mut self, index: usize) -> V::Item {
        // the nature of IsRepeatable means that any index can be called any number of times so this is fine
        if index >= D {
            panic!("math_vector Error: index access out of bound")
        }
        unsafe {
            let inputs = self.0.get_inputs(index);
            let (item, _) = self.0.process(index, inputs);
            item
        }
    }
}

impl<V: VectorLike, const D: usize> Drop for VectorExpr<V, D> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            for i in 0..D {
                self.0.drop_inputs(i);
            }
            self.0.drop_output();
            self.0.drop_1st_buffer();
            self.0.drop_2nd_buffer();
        }
    }
}

impl<V: VectorLike, const D: usize> IntoIterator for VectorExpr<V, D>
where
    V: HasReuseBuf<BoundTypes = V::BoundItems>,
{
    type IntoIter = VectorIter<V>;
    type Item = <V as Get>::Item;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.into_vec_iter()
    }
}

/// a simple type alias for a VectorExpr created from an array of type [T; D]
pub type MathVector<T, const D: usize> = VectorExpr<VectorArray<T, D>, D>;

impl<T, const D: usize> MathVector<T, D> {
    /// marks this MathVector to have its buffer reused
    /// buffer placed on fst buffer
    #[inline]
    pub fn reuse(self) -> VectorExpr<ReplaceArray<T, D>, D> {
        VectorExpr(ReplaceArray(self.unwrap().0))
    }
    /// marks this MathVector to have its buffer reused while keeping it on the heap
    /// buffer placed on fst buffer
    #[inline]
    pub fn heap_reuse(self: Box<Self>) -> VectorExpr<Box<ReplaceArray<T, D>>, D> {
        // Safety, series of equivilent types:
        // Box<MathVector<T,D>>
        // Box<VectorExpr<VectorArray<T, D>, D>>, de-alias MathVector
        // Box<ManuallyDrop<[T; D]>>, VectorExpr & VectorArray are transparent
        // VectorExpr<Box<ReplaceArray<T, D>>, D>, VectorExpr & ReplaceArray are transparent
        unsafe { mem::transmute::<Box<Self>, VectorExpr<Box<ReplaceArray<T, D>>, D>>(self) }
    }

    /// converts this MathVector to a repeatable VectorExpr w/ Item = &'a T
    #[inline]
    pub fn referred<'a>(self) -> VectorExpr<ReferringVectorArray<'a, T, D>, D>
    where
        T: 'a,
    {
        VectorExpr(ReferringVectorArray(
            unsafe { mem::transmute_copy::<ManuallyDrop<[T; D]>, [T; D]>(&self.unwrap().0) },
            std::marker::PhantomData,
        )) //FIXME: unecessary transmute copy to get the compiler to not complain
    }

    /// references the element at index without checking bounds
    /// safety: index is in bounds
    #[inline]
    pub unsafe fn get_unchecked<I: SliceIndex<[T]>>(&self, index: I) -> &I::Output {
        unsafe { self.0.0.get_unchecked(index) }
    }

    /// mutably references the element at index without checking bounds
    /// safety: index is in bounds
    #[inline]
    pub unsafe fn get_unchecked_mut<I: SliceIndex<[T]>>(&mut self, index: I) -> &mut I::Output {
        unsafe { self.0.0.get_unchecked_mut(index) }
    }
}

impl<T: Clone, const D: usize> Clone for MathVector<T, D> {
    #[inline]
    fn clone(&self) -> Self {
        VectorExpr(self.0.clone())
    }
}

impl<T, const D: usize> Deref for MathVector<T, D> {
    type Target = [T; D];

    fn deref(&self) -> &Self::Target {
        &self.0.0
    }
}

impl<T, const D: usize> DerefMut for MathVector<T, D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0.0
    }
}

impl<T, const D: usize> From<[T; D]> for MathVector<T, D> {
    #[inline]
    fn from(value: [T; D]) -> Self {
        VectorExpr(VectorArray(ManuallyDrop::new(value)))
    }
}

impl<T, const D: usize> From<MathVector<T, D>> for [T; D] {
    #[inline]
    fn from(value: MathVector<T, D>) -> Self {
        value.unwrap().unwrap()
    }
}

impl<'a, T, const D: usize> From<&'a [T; D]> for &'a MathVector<T, D> {
    #[inline]
    fn from(value: &'a [T; D]) -> Self {
        unsafe { mem::transmute::<&'a [T; D], &'a MathVector<T, D>>(value) }
    }
}

impl<'a, T, const D: usize> From<&'a MathVector<T, D>> for &'a [T; D] {
    #[inline]
    fn from(value: &'a MathVector<T, D>) -> Self {
        unsafe { mem::transmute::<&'a MathVector<T, D>, &'a [T; D]>(value) }
    }
}

impl<'a, T, const D: usize> From<&'a mut [T; D]> for &'a mut MathVector<T, D> {
    #[inline]
    fn from(value: &'a mut [T; D]) -> Self {
        unsafe { mem::transmute::<&'a mut [T; D], &'a mut MathVector<T, D>>(value) }
    }
}

impl<'a, T, const D: usize> From<&'a mut MathVector<T, D>> for &'a mut [T; D] {
    #[inline]
    fn from(value: &'a mut MathVector<T, D>) -> Self {
        unsafe { mem::transmute::<&'a mut MathVector<T, D>, &'a mut [T; D]>(value) }
    }
}

impl<T, const D: usize> From<Box<[T; D]>> for Box<MathVector<T, D>> {
    #[inline]
    fn from(value: Box<[T; D]>) -> Self {
        unsafe { mem::transmute::<Box<[T; D]>, Box<MathVector<T, D>>>(value) }
    }
}

impl<T, const D: usize> From<Box<MathVector<T, D>>> for Box<[T; D]> {
    #[inline]
    fn from(value: Box<MathVector<T, D>>) -> Self {
        unsafe { mem::transmute::<Box<MathVector<T, D>>, Box<[T; D]>>(value) }
    }
}

impl<T, I, const D: usize> Index<I> for MathVector<T, D>
where
    [T; D]: Index<I>,
{
    type Output = <[T; D] as Index<I>>::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        &self.0.0[index]
    }
}

impl<T, I, const D: usize> Index<I> for &MathVector<T, D>
where
    [T; D]: Index<I>,
{
    type Output = <[T; D] as Index<I>>::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        &self.0.0[index]
    }
}

impl<T, I, const D: usize> Index<I> for VectorExpr<&[T; D], D>
where
    [T; D]: Index<I>,
{
    type Output = <[T; D] as Index<I>>::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, I, const D: usize> Index<I> for &mut MathVector<T, D>
where
    [T; D]: Index<I>,
{
    type Output = <[T; D] as Index<I>>::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        &self.0.0[index]
    }
}

impl<T, I, const D: usize> Index<I> for VectorExpr<&mut [T; D], D>
where
    [T; D]: Index<I>,
{
    type Output = <[T; D] as Index<I>>::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, I, const D: usize> IndexMut<I> for MathVector<T, D>
where
    [T; D]: IndexMut<I>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0.0[index]
    }
}

impl<T, I, const D: usize> IndexMut<I> for &mut MathVector<T, D>
where
    [T; D]: IndexMut<I>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0.0[index]
    }
}

impl<T, I, const D: usize> IndexMut<I> for VectorExpr<&mut [T; D], D>
where
    [T; D]: IndexMut<I>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T, const D: usize> ConcreteVectorExpr for MathVector<T, D> {
    type ReferencedInner<'a> = &'a [T; D] 
    where 
        T: 'a,
        Self: 'a,
    ;
    type Referenced<'a> = &'a MathVector<T, D>
    where
        T: 'a,
        Self: 'a,
    ;
    type Copied<'a> = VectorExpr<VecCopy<'a, Self::ReferencedInner<'a>, T>, D>
        where
            Self::Output: Copy,
            Self::Output: 'a,
            Self: 'a,;
    type ReferencedMutInner<'a> = &'a mut [T; D] 
    where 
        T: 'a,
        Self: 'a,
    ;
    type ReferencedMut<'a> = &'a mut MathVector<T, D>
    where
        T: 'a,
        Self: 'a
    ;

    fn borrow<'a>(&'a self) -> Self::Referenced<'a> where 
        <Self::Referenced<'a> as VectorOps>::Unwrapped: Get<Item = &'a <Self::Referenced<'a> as Index<usize>>::Output>,
        Self::Output: 'a,
        Self: 'a 
    {
        self
    }

    fn borrow_mut<'a>(&'a mut self) -> Self::ReferencedMut<'a> where 
        <Self::ReferencedMut<'a> as VectorOps>::Unwrapped: Get<Item = &'a mut <Self::ReferencedMut<'a> as Index<usize>>::Output>,
        Self::Output: 'a,
        Self: 'a, 
    {
        self
    }

    fn copy<'a>(&'a self) -> Self::Copied<'a> where
        Self::Output: Copy,
        Self::Output: 'a,
        Self: 'a
    {
        VectorExpr(VecCopy{ vec: self.unwrap() })
    }
}

impl<T, I, USEDV: VectorLike, const D: usize> Index<I> for VectorExpr<VecAttachUsedVec<VectorArray<T, D>, USEDV>, D>
where
    [T; D]: Index<I>,
    (N, USEDV::OutputBool): FilterPair,
    (N, USEDV::FstOwnedBufferBool): SelectPair,
    (N, USEDV::SndOwnedBufferBool): SelectPair,
    (N, USEDV::FstHandleBool): SelectPair,
    (N, USEDV::SndHandleBool): SelectPair,
    (N, USEDV::BoundHandlesBool): FilterPair,
{
    type Output = <[T; D] as Index<I>>::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        &self.0.vec.0[index]
    }
}

impl<T, I, USEDV: VectorLike, const D: usize> IndexMut<I> for VectorExpr<VecAttachUsedVec<VectorArray<T, D>, USEDV>, D>
where
    [T; D]: IndexMut<I>,
    (N, USEDV::OutputBool): FilterPair,
    (N, USEDV::FstOwnedBufferBool): SelectPair,
    (N, USEDV::SndOwnedBufferBool): SelectPair,
    (N, USEDV::FstHandleBool): SelectPair,
    (N, USEDV::SndHandleBool): SelectPair,
    (N, USEDV::BoundHandlesBool): FilterPair,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0.vec.0[index]
    }
}

impl<T, I, USEDV: VectorLike, const D: usize> Index<I> for VectorExpr<VecAttachUsedVec<Box<VectorArray<T, D>>, USEDV>, D>
where
    [T; D]: Index<I>,
    (N, USEDV::OutputBool): FilterPair,
    (N, USEDV::FstOwnedBufferBool): SelectPair,
    (N, USEDV::SndOwnedBufferBool): SelectPair,
    (N, USEDV::FstHandleBool): SelectPair,
    (N, USEDV::SndHandleBool): SelectPair,
    (N, USEDV::BoundHandlesBool): FilterPair,
{
    type Output = <[T; D] as Index<I>>::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        &self.0.vec.0[index]
    }
}

impl<T, I, USEDV: VectorLike, const D: usize> IndexMut<I> for VectorExpr<VecAttachUsedVec<Box<VectorArray<T, D>>, USEDV>, D>
where
    [T; D]: IndexMut<I>,
    (N, USEDV::OutputBool): FilterPair,
    (N, USEDV::FstOwnedBufferBool): SelectPair,
    (N, USEDV::SndOwnedBufferBool): SelectPair,
    (N, USEDV::FstHandleBool): SelectPair,
    (N, USEDV::SndHandleBool): SelectPair,
    (N, USEDV::BoundHandlesBool): FilterPair,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0.vec.0[index]
    }
}

impl<T, USEDV: VectorLike, const D: usize> ConcreteVectorExpr for VectorExpr<VecAttachUsedVec<VectorArray<T, D>, USEDV>, D> where
    (N, USEDV::OutputBool): FilterPair,
    (N, USEDV::FstOwnedBufferBool): SelectPair,
    (N, USEDV::SndOwnedBufferBool): SelectPair,
    (N, USEDV::FstHandleBool): SelectPair,
    (N, USEDV::SndHandleBool): SelectPair,
    (N, USEDV::BoundHandlesBool): FilterPair,
{
    type ReferencedInner<'a> = &'a [T; D]
        where 
            Self::Output: 'a,
            Self: 'a,;
    type Referenced<'a> = VectorExpr<&'a [T; D], D>
        where
            Self::Output: 'a,
            Self: 'a,;
    type Copied<'a> = VectorExpr<VecCopy<'a, Self::ReferencedInner<'a>, T>, D>
        where
            Self::Output: Copy,
            Self::Output: 'a,
            Self: 'a,;
    type ReferencedMutInner<'a> =  &'a mut [T; D]
        where 
            Self::Output: 'a,
            Self: 'a,;
    type ReferencedMut<'a> = VectorExpr<&'a mut [T; D], D>
        where 
            Self::Output: 'a,
            Self: 'a,;

    fn borrow<'a>(&'a self) -> Self::Referenced<'a> where 
            <Self::Referenced<'a> as VectorOps>::Unwrapped: Get<Item = &'a <Self::Referenced<'a> as Index<usize>>::Output>,
            Self::Output: 'a,
            Self: 'a 
    {
        VectorExpr(&*self.0.vec.0)
    }

    fn borrow_mut<'a>(&'a mut self) -> Self::ReferencedMut<'a> where 
            <Self::ReferencedMut<'a> as VectorOps>::Unwrapped: Get<Item = &'a mut <Self::ReferencedMut<'a> as Index<usize>>::Output>,
            Self::Output: 'a,
            Self: 'a, {
        VectorExpr(&mut *self.0.vec.0)
    }

    fn copy<'a>(&'a self) -> Self::Copied<'a> where
        Self::Output: Copy,
        Self::Output: 'a,
        Self: 'a
    {
        self.borrow().copied()
    }
}


impl<T: std::fmt::Display, const D: usize> std::fmt::Display for MathVector<T, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut strings = Vec::with_capacity(D);
        let mut max_length = 0;
        for v in <&[_; _]>::from(self) {
            strings.push(v.to_string());
            max_length = std::cmp::max(max_length, strings[strings.len() - 1].len());
        }
        if strings.len() > 1 {
            write!(f, "\n┌ {:max_length$} ┐\n", strings[0])?;
            for string in &strings[1..strings.len() - 1] {
                writeln!(f, "│ {:max_length$} │", string)?;
            }
            write!(f, "└ {:max_length$} ┘", strings[strings.len() - 1])?;
        } else if strings.len() == 1 {
            write!(f, "\n[ {} ]", strings[0])?;
        } else {
            write!(f, "\n[]")?;
        }
        Ok(())
    }
}

//impl<T1: AddAssign<T2>, T2, const D: usize> AddAssign<MathVector<T2, D>> for MathVector<T1, D>
//{
//    #[inline]
//    fn add_assign(&mut self, rhs: MathVector<T2, D>) {
//        VectorExpr::<_, D>::consume(VectorOps::add_assign(self, rhs));
//    }
//}

impl<T: MulAssign<S>, S: Copy, const D: usize> MulAssign<S> for MathVector<T, D> {
    #[inline]
    fn mul_assign(&mut self, rhs: S) {
        <&mut Self as VectorOps>::mul_assign(self, rhs).consume();
    }
}

impl<T: DivAssign<S>, S: Copy, const D: usize> DivAssign<S> for MathVector<T, D> {
    #[inline]
    fn div_assign(&mut self, rhs: S) {
        <&mut Self as VectorOps>::div_assign(self, rhs).consume();
    }
}

impl<T: RemAssign<S>, S: Copy, const D: usize> RemAssign<S> for MathVector<T, D> {
    #[inline]
    fn rem_assign(&mut self, rhs: S) {
        <&mut Self as VectorOps>::rem_assign(self, rhs).consume();
    }
}

impl<T1: num_traits::Zero + AddAssign<T2>, T2, const D: usize> Sum<MathVector<T2, D>> for MathVector<T1, D> {
    #[inline]
    fn sum<I: Iterator<Item = MathVector<T2, D>>>(iter: I) -> Self {
        let mut sum = vector_gen(|| T1::zero())
            .create_array()
            .bind()
            .consume();
        for vec in iter {
            sum += vec;
        }
        sum
    }
}

/// generates a Vector of size D using the given closure (FnMut) with no inputs
#[inline]
pub fn vector_gen<F: FnMut() -> O, O, const D: usize>(f: F) -> VectorExpr<VecGenerator<F, O>, D> {
    VectorExpr(VecGenerator(f))
}

/// generates a Vector of size D using the given closure (FnMut) with an input of the current index
#[inline]
pub fn vector_index_gen<F: FnMut(usize) -> O, O, const D: usize>(
    f: F,
) -> VectorExpr<VecIndexGenerator<F, O>, D> {
    VectorExpr(VecIndexGenerator(f))
}

// TODO: finish implementing RSVectorExpr
#[derive(Clone)]
pub struct RSVectorExpr<V: VectorLike> {
    pub(crate) vec: V,
    pub(crate) size: usize,
}

impl<V: VectorLike> RSVectorExpr<V> {
    /// converts this runtime sized vector into a const sized one
    ///
    /// panics if this vector does not have size D
    #[inline]
    pub fn const_sized<const D: usize>(self) -> VectorExpr<V, D> {
        if self.size != D {
            panic!(
                "math_vector error: cannot convert a RS vector with size {} into a const sized vector with size {}",
                self.size, D
            )
        }
        unsafe { mem::transmute_copy::<V, VectorExpr<V, D>>(&ManuallyDrop::new(self).vec) }
    }
}

impl<V: VectorLike + IsRepeatable> RSVectorExpr<V> {
    #[inline]
    pub fn get(&mut self, index: usize) -> V::Item {
        if index >= self.size {
            panic!("math_vector Error: index access out of bounds")
        }
        unsafe {
            let inputs = self.vec.get_inputs(index);
            let (item, _) = self.vec.process(index, inputs);
            item
        }
    }
}

impl<V: VectorLike> Drop for RSVectorExpr<V> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            for i in 0..self.size {
                self.vec.drop_inputs(i);
            }
            self.vec.drop_output();
            self.vec.drop_1st_buffer();
            self.vec.drop_2nd_buffer();
        }
    }
}

impl<V: VectorLike> IntoIterator for RSVectorExpr<V>
where
    V: HasReuseBuf<BoundTypes = V::BoundItems>,
{
    type IntoIter = VectorIter<V>;
    type Item = <V as Get>::Item;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.into_vec_iter()
    }
}

pub type RSMathVector<T> = RSVectorExpr<VectorSlice<T>>;

impl<T> RSMathVector<T> {
    #[inline]
    pub fn reuse(self) -> RSVectorExpr<ReplaceSlice<T>> {
        let size = self.size;
        RSVectorExpr {
            vec: ReplaceSlice(ManuallyDrop::new(self.unwrap().0)),
            size,
        }
    }

    /// converts this RSMathVector to a repeatable VectorExpr w/ Item = &'a T
    #[inline]
    pub fn referred<'a>(self) -> RSVectorExpr<ReferringVectorSlice<'a, T>>
    where
        T: 'a,
    {
        let size = self.size;
        RSVectorExpr {
            vec: ReferringVectorSlice(
                unsafe {
                    mem::transmute_copy::<Box<ManuallyDrop<[T]>>, Box<[T]>>(&self.unwrap().0)
                },
                std::marker::PhantomData,
            ),
            size,
        } //FIXME: unecessary transmute copy to get the compiler to not complain
    }

    #[inline]
    pub unsafe fn get_unchecked<I: SliceIndex<[T]>>(&self, index: I) -> &I::Output {
        unsafe { self.vec.0.get_unchecked(index) }
    }

    #[inline]
    pub unsafe fn get_unchecked_mut<I: SliceIndex<[T]>>(&mut self, index: I) -> &mut I::Output {
        unsafe { self.vec.0.get_unchecked_mut(index) }
    }
}

impl<T> Deref for RSMathVector<T> {
    type Target = Box<[T]>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { mem::transmute::<&Box<ManuallyDrop<[T]>>, &Box<[T]>>(&self.vec.0) }
    }
}

impl<T> DerefMut for RSMathVector<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { mem::transmute::<&mut Box<ManuallyDrop<[T]>>, &mut Box<[T]>>(&mut self.vec.0) }
    }
}

impl<T> From<Vec<T>> for RSMathVector<T> {
    #[inline]
    fn from(value: Vec<T>) -> Self {
        Self::from(<Box<[T]>>::from(value))
    }
}

impl<T> From<Box<[T]>> for RSMathVector<T> {
    #[inline]
    fn from(value: Box<[T]>) -> Self {
        let size = value.len();
        unsafe {
            RSVectorExpr {
                vec: mem::transmute::<Box<[T]>, VectorSlice<T>>(value),
                size,
            }
        }
    }
}

impl<T> From<RSMathVector<T>> for Vec<T> {
    #[inline]
    fn from(value: RSMathVector<T>) -> Self {
        <Vec<T>>::from(<Box<[T]>>::from(value))
    }
}

impl<T> From<RSMathVector<T>> for Box<[T]> {
    #[inline]
    fn from(value: RSMathVector<T>) -> Self {
        unsafe { mem::transmute_copy::<VectorSlice<T>, Box<[T]>>(&ManuallyDrop::new(value).vec) }
    }
}

impl<T, I> Index<I> for RSMathVector<T>
where
    [T]: Index<I>,
{
    type Output = <[T] as Index<I>>::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        &self.vec.0[index]
    }
}

impl<T, I> IndexMut<I> for RSMathVector<T>
where
    [T]: IndexMut<I>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.vec.0[index]
    }
}

impl<T> ConcreteVectorExpr for RSMathVector<T> {
    type ReferencedInner<'a> = &'a [T] 
        where 
            Self::Output: 'a,
            Self: 'a,;
    type Referenced<'a> = RefRSMathVector<'a, T> 
        where
            Self::Output: 'a,
            Self: 'a,;
    type Copied<'a> = RSVectorExpr<VecCopy<'a, &'a [T], T>>
        where
            Self::Output: Copy,
            Self::Output: 'a,
            Self: 'a,;
    type ReferencedMutInner<'a> = &'a mut [T] 
        where 
            Self::Output: 'a,
            Self: 'a,;
    type ReferencedMut<'a> = RefMutRSMathVector<'a, T>
        where 
            Self::Output: 'a,
            Self: 'a,;

    #[inline]
    fn borrow<'a>(&'a self) -> Self::Referenced<'a> where 
        <Self::Referenced<'a> as VectorOps>::Unwrapped: Get<Item = &'a <Self::Referenced<'a> as Index<usize>>::Output>,
        Self::Output: 'a,
        Self: 'a
    {
        let size = self.size;
        RSVectorExpr { vec: &**self, size }
    }

    #[inline]
    fn borrow_mut<'a>(&'a mut self) -> Self::ReferencedMut<'a> where 
        <Self::ReferencedMut<'a> as VectorOps>::Unwrapped: Get<Item = &'a mut <Self::ReferencedMut<'a> as Index<usize>>::Output>,
        Self::Output: 'a,
        Self: 'a,
    {
        let size = self.size;
        RSVectorExpr { vec: &mut **self, size }
    }

    #[inline]
    fn copy<'a>(&'a self) -> Self::Copied<'a> where
        Self::Output: Copy,
        Self::Output: 'a,
        Self: 'a
    {
        self.borrow().copied()
    }
}

impl<T: std::fmt::Display> std::fmt::Display for RSMathVector<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.borrow().fmt(f)
    }
}

pub type RefRSMathVector<'a, T> = RSVectorExpr<&'a [T]>;

impl<'a, T> From<&'a [T]> for RefRSMathVector<'a, T> {
    #[inline]
    fn from(value: &'a [T]) -> Self {
        let size = value.len();
        RSVectorExpr {
            vec: value,
            size,
        }
    }
}

impl<'a, T> From<RefRSMathVector<'a, T>> for &'a [T] {
    #[inline]
    fn from(value: RefRSMathVector<'a, T>) -> &'a [T] {
        value.vec
    }
}

impl<'a, T, I> Index<I> for RefRSMathVector<'a, T>
where
    [T]: Index<I>,
{
    type Output = <[T] as Index<I>>::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        &self.vec[index]
    }
}

impl<'a, T: std::fmt::Display> std::fmt::Display for RefRSMathVector<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut strings = Vec::with_capacity(self.size());
        let mut max_length = 0;
        for v in <&[_]>::from(self.clone()) {
            strings.push(v.to_string());
            max_length = std::cmp::max(max_length, strings[strings.len() - 1].len());
        }
        if strings.len() > 1 {
            write!(f, "\n┌ {:max_length$} ┐\n", strings[0])?;
            for string in &strings[1..strings.len() - 1] {
                writeln!(f, "│ {:max_length$} │", string)?;
            }
            write!(f, "└ {:max_length$} ┘", strings[strings.len() - 1])?;
        } else if strings.len() == 1 {
            write!(f, "\n[ {} ]", strings[0])?;
        } else {
            write!(f, "\n[]")?;
        }
        Ok(())
    }
}

pub type RefMutRSMathVector<'a, T> = RSVectorExpr<&'a mut [T]>;

impl<'a, T> RefMutRSMathVector<'a, T> {
    pub fn deref<'b: 'a>(&'b self) -> RefRSMathVector<'a, T> {
        RSVectorExpr {
            vec: self.vec,
            size: self.size,
        }
    }
}

impl<'a, T> From<&'a mut [T]> for RefMutRSMathVector<'a, T> {
    #[inline]
    fn from(value: &'a mut [T]) -> Self {
        let size = value.len();
        RSVectorExpr {
            vec: value,
            size,
        }
    }
}

impl<'a, T> From<RefMutRSMathVector<'a, T>> for &'a mut [T] {
    #[inline]
    fn from(value: RefMutRSMathVector<'a, T>) -> &'a mut [T] {
        value.unwrap()
    }
}

impl<'a, T, I> Index<I> for RefMutRSMathVector<'a, T>
where
    [T]: Index<I>,
{
    type Output = <[T] as Index<I>>::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        &self.vec[index]
    }
}

impl<'a, T, I> IndexMut<I> for RefMutRSMathVector<'a, T>
where
    [T]: IndexMut<I>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.vec[index]
    }
}

impl<T, I, USEDV: VectorLike> Index<I> for RSVectorExpr<VecAttachUsedVec<VectorSlice<T>, USEDV>> where 
    [T]: Index<I>,
    (N, USEDV::OutputBool): FilterPair,
    (N, USEDV::FstOwnedBufferBool): SelectPair,
    (N, USEDV::SndOwnedBufferBool): SelectPair,
    (N, USEDV::FstHandleBool): SelectPair,
    (N, USEDV::SndHandleBool): SelectPair,
    (N, USEDV::BoundHandlesBool): FilterPair,
{
    type Output = <[T] as Index<I>>::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.vec.vec[index]
    }
}

impl<T, I, USEDV: VectorLike> IndexMut<I> for RSVectorExpr<VecAttachUsedVec<VectorSlice<T>, USEDV>> where 
    [T]: IndexMut<I>,
    (N, USEDV::OutputBool): FilterPair,
    (N, USEDV::FstOwnedBufferBool): SelectPair,
    (N, USEDV::SndOwnedBufferBool): SelectPair,
    (N, USEDV::FstHandleBool): SelectPair,
    (N, USEDV::SndHandleBool): SelectPair,
    (N, USEDV::BoundHandlesBool): FilterPair,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.vec.vec[index]
    }
}

impl<T, USEDV: VectorLike> ConcreteVectorExpr for RSVectorExpr<VecAttachUsedVec<VectorSlice<T>, USEDV>> where 
    (N, USEDV::OutputBool): FilterPair,
    (N, USEDV::FstOwnedBufferBool): SelectPair,
    (N, USEDV::SndOwnedBufferBool): SelectPair,
    (N, USEDV::FstHandleBool): SelectPair,
    (N, USEDV::SndHandleBool): SelectPair,
    (N, USEDV::BoundHandlesBool): FilterPair,
{
    type ReferencedInner<'a> = &'a [T] 
        where 
            Self::Output: 'a,
            Self: 'a,;
    type Referenced<'a> = RefRSMathVector<'a, T> 
        where
            Self::Output: 'a,
            Self: 'a,;
    type Copied<'a> = RSVectorExpr<VecCopy<'a, &'a [T], T>>
        where
            Self::Output: Copy,
            Self::Output: 'a,
            Self: 'a,;
    type ReferencedMutInner<'a> = &'a mut [T] 
        where 
            Self::Output: 'a,
            Self: 'a,;
    type ReferencedMut<'a> = RefMutRSMathVector<'a, T>
        where 
            Self::Output: 'a,
            Self: 'a,;

    fn borrow<'a>(&'a self) -> Self::Referenced<'a> where 
        <Self::Referenced<'a> as VectorOps>::Unwrapped: Get<Item = &'a <Self::Referenced<'a> as Index<usize>>::Output>,
        Self::Output: 'a,
        Self: 'a
    {
        let size = self.size;
        RSVectorExpr{ vec: &**self.vec.vec.0, size }
    }
    fn borrow_mut<'a>(&'a mut self) -> Self::ReferencedMut<'a> where 
        <Self::ReferencedMut<'a> as VectorOps>::Unwrapped: Get<Item = &'a mut <Self::ReferencedMut<'a> as Index<usize>>::Output>,
        Self::Output: 'a,
        Self: 'a,
    {
        let size = self.size;
        RSVectorExpr{ vec: &mut **self.vec.vec.0, size }
    }
    fn copy<'a>(&'a self) -> Self::Copied<'a> where
        Self::Output: Copy,
        Self::Output: 'a,
        Self: 'a
    {
        self.borrow().copied()
    }
}

impl<'a, T: std::fmt::Display> std::fmt::Display for RefMutRSMathVector<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.deref().fmt(f)
    }
}

/// a VectorExpr iterator
pub struct VectorIter<V: VectorLike> {
    vec: V,
    live_input_start: usize,
    dead_output_start: usize,
    size: usize,
} // note: ranges are start inclusive, end exclusive

impl<V: VectorLike> VectorIter<V> {
    #[inline]
    pub unsafe fn new_from_parts<B: VectorBuilder>(vec: V, builder: B) -> Self {
        VectorIter {
            vec,
            live_input_start: 0,
            dead_output_start: 0,
            size: builder.size(),
        }
    }

    /// retrieves the next item without checking
    /// Safety: there must be another item to return
    #[inline]
    pub unsafe fn next_unchecked(&mut self) -> V::Item
    where
        V: HasReuseBuf<BoundTypes = V::BoundItems>,
    {
        unsafe {
            let index = self.live_input_start;
            self.live_input_start += 1;
            let inputs = self.vec.get_inputs(index);
            let (item, bound_items) = self.vec.process(index, inputs);
            self.vec.assign_bound_bufs(index, bound_items);
            self.dead_output_start += 1;
            item
        }
    }

    /// retrieves the VectorIter's output without checking consumption
    /// Safety: the VectorLike must be fully consumed
    #[inline]
    pub unsafe fn unchecked_output(self) -> V::Output {
        // NOTE: manual drop shenanigans to prevent VectorIter from being dropped normally
        //       doing so would incorrectly drop HasReuseBuf & output
        let mut man_drop_self = ManuallyDrop::new(self);
        let output;
        unsafe {
            output = man_drop_self.vec.output();
            ptr::drop_in_place(&mut man_drop_self.vec);
        }
        output
    }

    /// retrieves the VectorIter's output
    /// the VectorIter must be fully consumed or this function will panic
    #[inline]
    pub fn output(self) -> V::Output {
        assert!(
            self.live_input_start == self.size,
            "math_vector error: A VectorIter must be fully used before outputting"
        );
        debug_assert!(
            self.dead_output_start == self.size,
            "math_vector internal error: A VectorIter's output buffers (somehow) weren't fully filled despite the inputs being fully used, likely an internal issue"
        );
        unsafe { self.unchecked_output() }
    }

    /// fully consumes the VectorIter and then returns its output
    #[inline]
    pub fn consume(mut self) -> V::Output
    where
        V: HasReuseBuf<BoundTypes = V::BoundItems>,
    {
        self.no_output_consume();
        unsafe { self.unchecked_output() } // safety: VectorIter was fully used
    }

    /// fully consumes the VectorIter without returning its output
    #[inline]
    pub fn no_output_consume(&mut self)
    where
        V: HasReuseBuf<BoundTypes = V::BoundItems>,
    {
        while self.live_input_start < self.size {
            unsafe {
                let _ = self.next_unchecked();
            }
        }
    }
}

impl<V: VectorLike> Drop for VectorIter<V> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            for i in 0..self.dead_output_start {
                //up to the start of the dead area in output
                self.vec.drop_bound_bufs_index(i);
            }
            for i in self.live_input_start..self.size {
                self.vec.drop_inputs(i);
            }
            // note: when VectorIter outputs, it is forgotten, so we can assume output hasn't been called
            self.vec.drop_output();
            self.vec.drop_1st_buffer();
            self.vec.drop_2nd_buffer();
        }
    }
}

impl<V: VectorLike> Iterator for VectorIter<V>
where
    V: HasReuseBuf<BoundTypes = V::BoundItems>,
{
    type Item = V::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.live_input_start < self.size {
            // != instead of < as it is known that start is always <= end so their equivilent
            unsafe { Some(self.next_unchecked()) }
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.size - self.live_input_start;
        (size, Some(size))
    }
}

impl<V: VectorLike> ExactSizeIterator for VectorIter<V> where
    V: HasReuseBuf<BoundTypes = V::BoundItems>
{
}

impl<V: VectorLike> std::iter::FusedIterator for VectorIter<V> where
    V: HasReuseBuf<BoundTypes = V::BoundItems>
{
}

macro_rules! select {
    ($bool:tt {$($true:tt)*} {$($false:tt)*}) => {
        $($true)*
    };
    ({$($true:tt)*} {$($false:tt)*}) => {
        $($false)*
    }
}

macro_rules! vec_op {
    ( // vec ops
        $(
            $(#[doc = $doc:literal])+
            fn $op_name:ident$(<
                $(
                    $lifetime:lifetime,
                )* 
                $(
                    $generic:ident $(: $($generic_lifetime:lifetime|)? $generic_bound_head:path $(| $generic_bound:path)*)? $(,)?
                ),*
            >)?($self:ident $(: $self_ty:ty)? $(, $val:ident: $ty:ty)* $(,)?) -> $({$has_output:tt})? $unwrapped_output:ty $(, output: $output:ty)? $(,)?
            $(where $(
                $where_bounded:ty: $($where_bound_lifetime:lifetime $(|)?)? $where_bound_head:path $(| $where_bound:path)*
            ),+ $(,)?)?
            {$($expr:tt)*}
        )*
    ) => {
        $(
            // note: there might be a cleaner way to write this...
            select!($($has_output)? {
                $(#[doc=$doc])+
                #[inline]
                fn $op_name$(<
                    $($lifetime,)* 
                    $($generic $(: $($generic_lifetime+)? $generic_bound_head $(+ $generic_bound)*)?),*
                >)?($self $(: $self_ty)? $(, $val: $ty)*) -> <Self::Builder as VectorBuilder>::Wrapped<$unwrapped_output>
                where
                    (<<Self as VectorOps>::Unwrapped as HasOutput>::OutputBool, Y): FilterPair,
                    $($(
                        $where_bounded: $($where_bound_lifetime +)? $where_bound_head $(+ $where_bound)*,
                    )+)?
                {
                    $($expr)*
                }
            } {
                $(#[doc=$doc])*
                #[inline]
                fn $op_name$(<
                    $($lifetime,)* 
                    $($generic $(: $($generic_lifetime+)? $generic_bound_head $(+ $generic_bound)*)?),*
                >)?($self $(: $self_ty)? $(, $val: $ty)*) -> <Self::Builder as VectorBuilder>::Wrapped<$unwrapped_output>
                where
                    $($(
                        $where_bounded: $($where_bound_lifetime +)? $where_bound_head $(+ $where_bound)*,
                    )+)?
                {
                    $($expr)*
                }
            });
        )*
    };
    ( // vec x vec ops
        $(
            $(#[doc = $doc:literal])+
            fn $op_name:ident<
                $(
                    $lifetime:lifetime,
                )* 
                {$v:ident $(: $($v_lifetime:lifetime|)? $($v_bound:path)|+)?} 
                $(
                    ,$generic:ident $(: $($generic_lifetime:lifetime|)? $generic_bound_head:path $(| $generic_bound:path)*)?
                )*
            >($self:ident $(: $self_ty:ty)? $(, $val:ident: $ty:ty)* $(,)?) -> $({$has_output:tt})? $unwrapped_output:ty $(, output: $output:ty)?
            $(where $(
                $where_bounded:ty: $($where_bound_lifetime:lifetime |)? $where_bound_head:path $(| $where_bound:path)*
            ),+ $(,)?)?
            {$($expr:tt)*}
        )*
    ) => {
        $(
            select!($($has_output)? {
                $(#[doc=$doc])+
                #[inline]
                fn $op_name<
                    $($lifetime,)* 
                    $v : $($($v_lifetime +)?)? VectorOps $($(+ $v_bound)+)?
                    $(,$generic $(: $($generic_lifetime+)? $generic_bound_head $(+ $generic_bound)*)?)*
                >($self $(: $self_ty:ty)? $(, $val: $ty)*) -> <<
                    <Self as VectorOps>::Builder as VectorBuilderUnion<<$v as VectorOps>::Builder>
                >::Union as VectorBuilder>::Wrapped<$unwrapped_output> 
                where
                    <Self as VectorOps>::Builder: VectorBuilderUnion<<V as VectorOps>::Builder>,
                    (
                        <<Self as VectorOps>::Unwrapped as HasOutput>::OutputBool,
                        <<V as VectorOps>::Unwrapped as HasOutput>::OutputBool,
                    ): FilterPair,
                    (
                        <(
                            <<Self as VectorOps>::Unwrapped as HasOutput>::OutputBool,
                            <<V as VectorOps>::Unwrapped as HasOutput>::OutputBool,
                        ) as TyBoolPair>::Or,
                        Y,
                    ): FilterPair,
                    (
                        <<Self as VectorOps>::Unwrapped as HasReuseBuf>::BoundHandlesBool,
                        <<V as VectorOps>::Unwrapped as HasReuseBuf>::BoundHandlesBool,
                    ): FilterPair,
                    (
                        <<Self as VectorOps>::Unwrapped as HasReuseBuf>::FstHandleBool,
                        <<V as VectorOps>::Unwrapped as HasReuseBuf>::FstHandleBool,
                    ): SelectPair,
                    (
                        <<Self as VectorOps>::Unwrapped as HasReuseBuf>::SndHandleBool,
                        <<V as VectorOps>::Unwrapped as HasReuseBuf>::SndHandleBool,
                    ): SelectPair,
                    (
                        <<Self as VectorOps>::Unwrapped as HasReuseBuf>::FstOwnedBufferBool,
                        <<V as VectorOps>::Unwrapped as HasReuseBuf>::FstOwnedBufferBool,
                    ): SelectPair,
                    (
                        <<Self as VectorOps>::Unwrapped as HasReuseBuf>::SndOwnedBufferBool,
                        <<V as VectorOps>::Unwrapped as HasReuseBuf>::SndOwnedBufferBool,
                    ): SelectPair,
                    $($(
                        $where_bounded: $($where_bound_lifetime +)? $where_bound_head $(+ $where_bound)*,
                    )+)?
                {$($expr)*}
            } {
                $(#[doc=$doc])+
                #[inline]
                fn $op_name<
                    $($lifetime,)* 
                    $v : $($($v_lifetime +)?)? VectorOps $($(+ $v_bound)+)?
                    $(,$generic $(: $($generic_lifetime+)? $generic_bound_head $(+ $generic_bound)*)?)*
                >($self $(: $self_ty:ty)? $(, $val: $ty)*) -> <<
                    <Self as VectorOps>::Builder as VectorBuilderUnion<<$v as VectorOps>::Builder>
                >::Union as VectorBuilder>::Wrapped<$unwrapped_output> 
                where
                    <Self as VectorOps>::Builder: VectorBuilderUnion<<V as VectorOps>::Builder>,
                    (
                        <<Self as VectorOps>::Unwrapped as HasOutput>::OutputBool,
                        <<V as VectorOps>::Unwrapped as HasOutput>::OutputBool,
                    ): FilterPair,
                    (
                        <<Self as VectorOps>::Unwrapped as HasReuseBuf>::BoundHandlesBool,
                        <<V as VectorOps>::Unwrapped as HasReuseBuf>::BoundHandlesBool,
                    ): FilterPair,
                    (
                        <<Self as VectorOps>::Unwrapped as HasReuseBuf>::FstHandleBool,
                        <<V as VectorOps>::Unwrapped as HasReuseBuf>::FstHandleBool,
                    ): SelectPair,
                    (
                        <<Self as VectorOps>::Unwrapped as HasReuseBuf>::SndHandleBool,
                        <<V as VectorOps>::Unwrapped as HasReuseBuf>::SndHandleBool,
                    ): SelectPair,
                    (
                        <<Self as VectorOps>::Unwrapped as HasReuseBuf>::FstOwnedBufferBool,
                        <<V as VectorOps>::Unwrapped as HasReuseBuf>::FstOwnedBufferBool,
                    ): SelectPair,
                    (
                        <<Self as VectorOps>::Unwrapped as HasReuseBuf>::SndOwnedBufferBool,
                        <<V as VectorOps>::Unwrapped as HasReuseBuf>::SndOwnedBufferBool,
                    ): SelectPair,
                    $($(
                        $where_bounded: $($where_bound_lifetime +)? $where_bound_head $(+ $where_bound)*,
                    )+)?
                {$($expr)*}
            });
        )*
    };
}

/// a trait with various vector operations
// triggers on the output types of the fn's even though they can't be shortened
pub unsafe trait VectorOps: Sized {
    /// the underlying VectorLike contained in Self
    type Unwrapped: VectorLike;
    /// type which builds the Wrapper around the VectorLike
    type Builder: VectorBuilder;

    /// get the underlying VectorLike
    fn unwrap(self) -> Self::Unwrapped;
    /// get the builder for this vector's wrapper
    fn get_builder(&self) -> Self::Builder;
    /// get the size of this vector
    fn size(&self) -> usize;

    /// converts the vector into a iterator
    #[inline]
    fn into_vec_iter(self) -> VectorIter<Self::Unwrapped>
    where
        Self::Unwrapped: HasReuseBuf<BoundTypes = <Self::Unwrapped as Get>::BoundItems>,
        Self: Sized,
    {
        let size = self.size();
        VectorIter {
            vec: self.unwrap(),
            live_input_start: 0,
            dead_output_start: 0,
            size,
        }
    }

    /// consumes the vector and returns the built up output
    ///
    /// Note:
    /// methods like sum, product, or fold can place build up outputs
    ///
    /// output is generally nested 2 element tuples
    /// newer values to the right
    /// binary operators merge the output of the 2 vectors
    #[inline]
    fn consume(self) -> <Self::Unwrapped as HasOutput>::Output
    where
        Self::Unwrapped: HasReuseBuf<BoundTypes = <Self::Unwrapped as Get>::BoundItems>,
        Self: Sized,
    {
        self.into_vec_iter().consume()
    }

    // single vector ops
    vec_op!(
        /// binds the vector's item to its fst buffer, adding the buffer to Output if owned by the vector
        fn bind(self) -> VecBind<Self::Unwrapped>
        where
            Self::Unwrapped: VectorLike<FstHandleBool = Y>,
            (
                <Self::Unwrapped as HasOutput>::OutputBool,
                <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool,
            ): FilterPair,
            (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, Y): FilterPair,
            VecBind<Self::Unwrapped>:
                HasReuseBuf<BoundTypes = <VecBind<Self::Unwrapped> as Get>::BoundItems>,
        {
            let builder = self.get_builder();
            unsafe { builder.wrap(VecBind { vec: self.unwrap() }) }
        }
    
        /// binds the vector's item to its fst buffer, adding the buffer to Output if owned by the vector without cleanly filtering the output
        fn raw_bind(self) -> VecRawBind<Self::Unwrapped>
        where
            Self::Unwrapped: VectorLike<FstHandleBool = Y>,
            (
                <Self::Unwrapped as HasOutput>::OutputBool,
                <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool,
            ): TyBoolPair,
            (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, Y): FilterPair,
            VecBind<Self::Unwrapped>:
                HasReuseBuf<BoundTypes = <VecBind<Self::Unwrapped> as Get>::BoundItems>,
        {
            let builder = self.get_builder();
            unsafe { builder.wrap(VecRawBind { vec: self.unwrap() }) }
        }
    
        /// maps the vector w/ the provided closure taking the vector's item and outputing the new item and a value to bind
        /// binds that value to the fst buffer, adding the buffer to Output if owned by the vector
        fn map_bind<F: FnMut(<Self::Unwrapped as Get>::Item) -> (I, B), I, B>(self, f: F) -> VecMapBind<Self::Unwrapped, F, I, B>
        where
            Self::Unwrapped: VectorLike<FstHandleBool = Y>,
            (
                <Self::Unwrapped as HasOutput>::OutputBool,
                <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool,
            ): FilterPair,
            (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, Y): FilterPair,
            VecMapBind<Self::Unwrapped, F, I, B>:
                HasReuseBuf<BoundTypes = <VecMapBind<Self::Unwrapped, F, I, B> as Get>::BoundItems>,
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecMapBind {
                    vec: self.unwrap(),
                    f,
                })
            }
        }
    
        /// binds the vector's item to its fst buffer, adding the buffer to an internal output if owned by the vector
        ///
        /// Note:
        /// this internal output is not readily accessible and doesn't add much over bind
        /// As such, end users should generally just use bind
        fn half_bind(self) -> VecHalfBind<Self::Unwrapped>
        where
            Self::Unwrapped: VectorLike<FstHandleBool = Y>,
            (
                <Self::Unwrapped as HasOutput>::OutputBool,
                <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool,
            ): FilterPair,
            (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, Y): FilterPair,
            VecHalfBind<Self::Unwrapped>:
                HasReuseBuf<BoundTypes = <VecBind<Self::Unwrapped> as Get>::BoundItems>,
        {
            let builder = self.get_builder();
            unsafe { builder.wrap(VecHalfBind { vec: self.unwrap() }) }
        }
    
        /// swaps the vector's first and second buffers
        fn buf_swap(self) -> VecBufSwap<Self::Unwrapped> {
            let builder = self.get_builder();
            unsafe { builder.wrap(VecBufSwap { vec: self.unwrap() }) }
        }
        
        /// returns a vector identical to a Repeatable Vector (DEV NOTE: honestly not sure when this is necessary since it uses a &mut)
        fn repeated(self: &mut Self) -> RepeatedVec<'_, Self::Unwrapped>
        where
            Self::Unwrapped: IsRepeatable,
            Self: AsMut<Self::Unwrapped>,
        {
            let builder = self.get_builder();
            unsafe { builder.wrap(RepeatedVec { vec: self.as_mut() }) }
        }
        
        /// converts the underlying VectorLike to a dynamic object
        /// stabilizes the overall type to a consitent one
        /// ex:
        /// ```
        ///    use math_vector::{vector::*};
        ///
        ///    let mut vec = VectorExpr::from([1; 100]).make_dynamic();
        ///    for _ in 0..10 {
        ///        vec = vec.add(VectorExpr::from([1; 100])).make_dynamic();
        ///    }
        ///    let output = vec.eval();
        /// ```
        fn make_dynamic(self) -> Box<
            dyn VectorLike<
                GetBool = <Self::Unwrapped as Get>::GetBool,
                Inputs = (),
                Item = <Self::Unwrapped as Get>::Item,
                BoundItems = <Self::Unwrapped as Get>::BoundItems,
                OutputBool = <Self::Unwrapped as HasOutput>::OutputBool,
                Output = <Self::Unwrapped as HasOutput>::Output,
                FstHandleBool = <Self::Unwrapped as HasReuseBuf>::FstHandleBool,
                SndHandleBool = <Self::Unwrapped as HasReuseBuf>::SndHandleBool,
                BoundHandlesBool = <Self::Unwrapped as HasReuseBuf>::BoundHandlesBool,
                FstOwnedBufferBool = <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool,
                SndOwnedBufferBool = <Self::Unwrapped as HasReuseBuf>::SndOwnedBufferBool,
                FstOwnedBuffer = <Self::Unwrapped as HasReuseBuf>::FstOwnedBuffer,
                SndOwnedBuffer = <Self::Unwrapped as HasReuseBuf>::SndOwnedBuffer,
                FstType = <Self::Unwrapped as HasReuseBuf>::FstType,
                SndType = <Self::Unwrapped as HasReuseBuf>::SndType,
                BoundTypes = <Self::Unwrapped as HasReuseBuf>::BoundTypes,
            >,
        >,
        where
            Self::Unwrapped: 'static | std::any::Any, //jank 
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(Box::new(DynamicVectorLike{vec: self.unwrap(), inputs: None}) as Box<dyn VectorLike<
                GetBool = <Self::Unwrapped as Get>::GetBool,
                Inputs = (),
                Item = <Self::Unwrapped as Get>::Item,
                BoundItems = <Self::Unwrapped as Get>::BoundItems,
        
                OutputBool = <Self::Unwrapped as HasOutput>::OutputBool,
                Output = <Self::Unwrapped as HasOutput>::Output,
        
                FstHandleBool = <Self::Unwrapped as HasReuseBuf>::FstHandleBool,
                SndHandleBool = <Self::Unwrapped as HasReuseBuf>::SndHandleBool,
                BoundHandlesBool = <Self::Unwrapped as HasReuseBuf>::BoundHandlesBool,
                FstOwnedBufferBool = <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool,
                SndOwnedBufferBool = <Self::Unwrapped as HasReuseBuf>::SndOwnedBufferBool,
                FstOwnedBuffer = <Self::Unwrapped as HasReuseBuf>::FstOwnedBuffer,
                SndOwnedBuffer = <Self::Unwrapped as HasReuseBuf>::SndOwnedBuffer,
                FstType = <Self::Unwrapped as HasReuseBuf>::FstType,
                SndType = <Self::Unwrapped as HasReuseBuf>::SndType,
                BoundTypes = <Self::Unwrapped as HasReuseBuf>::BoundTypes,
            >>)
            }
        }
    
        /// attaches arbitrary to a vector's output
        fn attach_output<O>(self, output: O) -> {has_output} VecAttachOutput<Self::Unwrapped, O, Y> {
            let builder = self.get_builder();
            unsafe { builder.wrap(VecAttachOutput { 
                vec: self.unwrap(),
                output: ManuallyDrop::new(output),
                marker: PhantomData,
            }) }
        }

        /// maybe attaches arbitrary to a vector's output
        fn maybe_attach_output<O, OB>(self, output: O) -> VecAttachOutput<Self::Unwrapped, O, OB> where
            (<Self::Unwrapped as HasOutput>::OutputBool, OB): FilterPair
        {
            let builder = self.get_builder();
            unsafe { builder.wrap(VecAttachOutput { 
                vec: self.unwrap(),
                output: ManuallyDrop::new(output),
                marker: PhantomData,
            }) }
        }

        /// offsets (with rolling over) each element of the vector up by the given offset
        fn offset_up(self, offset: usize) -> VecOffset<Self::Unwrapped> {
            let size = self.size();
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecOffset {
                    vec: self.unwrap(),
                    offset: offset % size,
                    size,
                })
            }
        }
    
        /// offsets (with rolling over) each element of the vector down by the given offset
        fn offset_down(self, offset: usize) ->  VecOffset<Self::Unwrapped> {
            let size = self.size();
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecOffset {
                    vec: self.unwrap(),
                    offset: size - (offset % size),
                    size,
                })
            }
        }
    
        /// reverses the vector
        fn reverse(self) -> {has_output} VecReverse<Self::Unwrapped> {
            let max_index = self.size() - 1;
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecReverse {
                    vec: self.unwrap(),
                    max_index,
                })
            }
        }
    
        /// maps the vector's items with the provided closure
        fn map<F: FnMut(<Self::Unwrapped as Get>::Item) -> O, O>(self, f: F) -> VecMap<Self::Unwrapped, F, O> {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecMap {
                    vec: self.unwrap(),
                    f,
                })
            }
        }
    
        /// folds the vector's items into a single value using the provided closure
        /// note: fold_ref should be used whenever possible due to implementation
        fn fold<F: FnMut(O, <Self::Unwrapped as Get>::Item) -> O, O>(self, f: F, init: O) -> {has_output} VecFold<Self::Unwrapped, F, O>, output: O {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecFold {
                    vec: self.unwrap(),
                    f,
                    cell: Some(init),
                })
            }
        }
    
        /// folds the vector's items into a single value using the provided closure while preserving the items
        ///
        /// note: fold_ref should be used whenever possible due to implementation
        fn copied_fold<F: FnMut(O, <Self::Unwrapped as Get>::Item) -> O, O>(self, f: F, init: O) -> {has_output} VecCopiedFold<Self::Unwrapped, F, O>, output: O
        where
            <Self::Unwrapped as Get>::Item: Copy,
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecCopiedFold {
                    vec: self.unwrap(),
                    f,
                    cell: Some(init),
                })
            }
        }
    
        /// folds the vector's items into a single value using the provided closure
        fn fold_ref<F: FnMut(&mut O, <Self::Unwrapped as Get>::Item), O>(self, f: F, init: O) -> {has_output} VecFoldRef<Self::Unwrapped, F, O>, output: O {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecFoldRef {
                    vec: self.unwrap(),
                    f,
                    cell: ManuallyDrop::new(init),
                })
            }
        }
    
        /// folds the vector's items into a single value using the provided closure while preserving the items
        fn copied_fold_ref<F: FnMut(&mut O, <Self::Unwrapped as Get>::Item), O>(self, f: F, init: O) -> {has_output} VecCopiedFoldRef<Self::Unwrapped, F, O>, output: O
        where
            <Self::Unwrapped as Get>::Item: Copy,
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecCopiedFoldRef {
                    vec: self.unwrap(),
                    f,
                    cell: ManuallyDrop::new(init),
                })
            }
        }
    
        /// copies each of the vector's items, useful for turning &T -> T
        fn copied<'a, I: Copy>(self) -> VecCopy<'a, Self::Unwrapped, I>
        where
            Self::Unwrapped: Get<Item = &'a I>,
        {
            let builder = self.get_builder();
            unsafe { builder.wrap(VecCopy { vec: self.unwrap() }) }
        }
    
        /// clones each of the vector's items, useful for turning &T -> T
        fn cloned<'a, I: Clone>(self) -> VecClone<'a, Self::Unwrapped, I>
        where
            Self::Unwrapped: Get<Item = &'a I>,
        {
            let builder = self.get_builder();
            unsafe { builder.wrap(VecClone { vec: self.unwrap() }) }
        }
    
        /// negates each of the vector's items
        fn neg(self) -> VecNeg<Self::Unwrapped>
        where
            <Self::Unwrapped as Get>::Item: Neg,
        {
            let builder = self.get_builder();
            unsafe { builder.wrap(VecNeg { vec: self.unwrap() }) }
        }
    
        /// multiples a scalar with the vector (vector items are rhs) (*may* be identitical to mul_l)
        fn mul_r<S: Mul<<Self::Unwrapped as Get>::Item> | Copy>(self, scalar: S) -> VecMulR<Self::Unwrapped, S> {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecMulR {
                    vec: self.unwrap(),
                    scalar,
                })
            }
        }
    
        /// divides a scalar with the vector (vector items are rhs)
        fn div_r<S: Div<<Self::Unwrapped as Get>::Item> | Copy>(self, scalar: S) -> VecDivR<Self::Unwrapped, S> {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecDivR {
                    vec: self.unwrap(),
                    scalar,
                })
            }
        }
    
        /// gets the remainder (ie. %) of a scalar with the vector (vector items are rhs)
        fn rem_r<S: Rem<<Self::Unwrapped as Get>::Item> | Copy>(self, scalar: S,) -> VecRemR<Self::Unwrapped, S> {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecRemR {
                    vec: self.unwrap(),
                    scalar,
                })
            }
        }
    
        /// multiplies the vector with a scalar (vector items are lhs) (*may* be identitcal to mul_r)
        fn mul_l<S: Copy>(self, scalar: S) -> VecMulL<Self::Unwrapped, S>
        where
            <Self::Unwrapped as Get>::Item: Mul<S>,
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecMulL {
                    vec: self.unwrap(),
                    scalar,
                })
            }
        }
    
        /// divides the vector with a scalar (vector items are lhs)
        fn div_l<S: Copy>(self, scalar: S) -> VecDivL<Self::Unwrapped, S>
        where
            <Self::Unwrapped as Get>::Item: Div<S>,
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecDivL {
                    vec: self.unwrap(),
                    scalar,
                })
            }
        }
    
        /// gets the remainder (ie. %) of the vector with a scalar (vector items are lhs)
        fn rem_l<S: Copy>(self, scalar: S) -> VecRemL<Self::Unwrapped, S>
        where
            <Self::Unwrapped as Get>::Item: Rem<S>,
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecRemL {
                    vec: self.unwrap(),
                    scalar,
                })
            }
        }
    
        /// mul assigns (*=) the vector's items (&mut T) with a scalar
        fn mul_assign<'a, I: 'a | MulAssign<S>, S: Copy>(self, scalar: S) -> VecMulAssign<'a, Self::Unwrapped, I, S>
        where
            Self::Unwrapped: Get<Item = &'a mut I>,
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecMulAssign {
                    vec: self.unwrap(),
                    scalar,
                })
            }
        }
    
        /// div assigns (/=) the vector's items (&mut T) with a scalar
        fn div_assign<'a, I: 'a | DivAssign<S>, S: Copy>(self, scalar: S) -> VecDivAssign<'a, Self::Unwrapped, I, S>
        where
            Self::Unwrapped: Get<Item = &'a mut I>,
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecDivAssign {
                    vec: self.unwrap(),
                    scalar,
                })
            }
        }
    
        /// rem assigns (%=) the vector's items (&mut T) with a scalar
        fn rem_assign<'a, I: 'a | RemAssign<S>, S: Copy>(self, scalar: S) -> VecRemAssign<'a, Self::Unwrapped, I, S>
        where
            Self::Unwrapped: Get<Item = &'a mut I>,
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecRemAssign {
                    vec: self.unwrap(),
                    scalar,
                })
            }
        }
    
        /// calculates the sum of the vector's elements and adds it to the output
        fn sum<S: num_traits::Zero | AddAssign<<Self::Unwrapped as Get>::Item>>(self) -> {has_output} VecSum<Self::Unwrapped, S>, output: S {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecSum {
                    vec: self.unwrap(),
                    scalar: ManuallyDrop::new(S::zero()),
                })
            }
        }
    
        /// calculates the sum of the vector's elements, initialized at given value (not necessarily 0), and adds it to the output
        fn initialized_sum<S: AddAssign<<Self::Unwrapped as Get>::Item>>(self, init: S) -> {has_output} VecSum<Self::Unwrapped, S>, output: S {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecSum {
                    vec: self.unwrap(),
                    scalar: ManuallyDrop::new(init),
                })
            }
        }
    
        /// calculates the sum of the vector's elements, adding it to the output, while maintaining the vector's items
        fn copied_sum<S: num_traits::Zero | AddAssign<<Self::Unwrapped as Get>::Item>>(self) -> {has_output} VecCopiedSum<Self::Unwrapped, S>, output: S
        where
            <Self::Unwrapped as Get>::Item: Copy,
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecCopiedSum {
                    vec: self.unwrap(),
                    scalar: ManuallyDrop::new(S::zero()),
                })
            }
        }
    
        /// calculates the sum of the vector's elements, initialized at given value (not necessarily 0), adding it to the output while maintaing the vector's items
        fn initialized_copied_sum<S: AddAssign<<Self::Unwrapped as Get>::Item>>(self, init: S) -> {has_output} VecCopiedSum<Self::Unwrapped, S>, output: S
        where
            <Self::Unwrapped as Get>::Item: Copy,
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecCopiedSum {
                    vec: self.unwrap(),
                    scalar: ManuallyDrop::new(init),
                })
            }
        }
    
        /// calculates the product of the vector's elements and adds it to the output
        fn product<S: num_traits::One | MulAssign<<Self::Unwrapped as Get>::Item>>(self) -> {has_output} VecProduct<Self::Unwrapped, S>, output: S {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecProduct {
                    vec: self.unwrap(),
                    scalar: ManuallyDrop::new(S::one()),
                })
            }
        }
    
        /// calculates the product of the vector's elements, initialized at given value (not necessarily 1), and adds it to the output
        fn initialized_product<S: MulAssign<<Self::Unwrapped as Get>::Item>>(self, init: S) -> {has_output} VecProduct<Self::Unwrapped, S>, output: S {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecProduct {
                    vec: self.unwrap(),
                    scalar: ManuallyDrop::new(init),
                })
            }
        }
    
        /// calculates the product of the vector's elements, adding it to the output, while maintaining the vector's items
        fn copied_product<S: num_traits::One | MulAssign<<Self::Unwrapped as Get>::Item>>(self) -> {has_output} VecCopiedProduct<Self::Unwrapped, S>, output: S
        where
            <Self::Unwrapped as Get>::Item: Copy,
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecCopiedProduct {
                    vec: self.unwrap(),
                    scalar: ManuallyDrop::new(S::one()),
                })
            }
        }
    
        /// calculates the products of the vector's elements, initialized at given value (not necessarily 0), adding it to the output while maintaing the vector's items
        fn initialized_copied_product<S: MulAssign<<Self::Unwrapped as Get>::Item>>(self, init: S) -> {has_output} VecCopiedProduct<Self::Unwrapped, S>, output: S
        where
            <Self::Unwrapped as Get>::Item: Copy,
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecCopiedProduct {
                    vec: self.unwrap(),
                    scalar: ManuallyDrop::new(init),
                })
            }
        }
    
        /// calculates the square of the vector's magnitude based on the dot product (ie. sum of each element's square) and adds it to the output
        /// note: this isn't actually correct if working with complex numbers. In that case, use `sqr_euclid_mag`
        fn sqr_mag<S: num_traits::One | AddAssign<<<Self::Unwrapped as Get>::Item as Mul>::Output>>(self) -> {has_output} VecSqrMag<Self::Unwrapped, S>, output: S
        where
            <Self::Unwrapped as Get>::Item: Copy | Mul,
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecSqrMag {
                    vec: self.unwrap(),
                    scalar: ManuallyDrop::new(S::one()),
                })
            }
        }
    
        /// calculates the square of the vector's magnitude based on the dot product (ie. sum of each element's square) and adds it to the output
        /// note: unless actually working with complex numbers, this is the same as `sqr_mag`
        fn sqr_euclid_mag<F: Copy | ComplexField>(self) -> {has_output} VecSqrEuclidMag<Self::Unwrapped, F>, output: F
        where
            Self::Unwrapped: Get<Item = F>,
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecSqrEuclidMag {
                    vec: self.unwrap(),
                    scalar: ManuallyDrop::new(F::one()),
                })
            }
        }
    
        /// calculates the square of the vector's magnitude (ie. sum of each element's square), adding it to the output, while maintaining the vector's items
        /// note: this isn't actually correct if working with complex numbers. In that case, use copied_sqr_euclid_mag or copied_euclid_mag
        fn copied_sqr_mag<S: num_traits::Zero | AddAssign<<<Self::Unwrapped as Get>::Item as Mul>::Output>>(self) -> {has_output} VecCopiedSqrMag<Self::Unwrapped, S>, output: S
        where
            <Self::Unwrapped as Get>::Item: Copy | Mul,
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecCopiedSqrMag {
                    vec: self.unwrap(),
                    scalar: ManuallyDrop::new(S::zero()),
                })
            }
        }
    
        /// calculates the square of the vector's magnitude based on the dot product (ie. sum of each element's square) and adds it to the output
        /// note: unless actually working with complex numbers, this is the same as `sqr_mag`
        fn copied_sqr_euclid_mag<F: Copy | ComplexField>(self) -> {has_output} VecSqrEuclidMag<Self::Unwrapped, F>, output: F
        where
            Self::Unwrapped: Get<Item = F>,
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecSqrEuclidMag {
                    vec: self.unwrap(),
                    scalar: ManuallyDrop::new(F::one()),
                })
            }
        }
    
        /// conjugates each entry of the vector
        fn conjugate(self) -> VecConjugate<Self::Unwrapped> where <Self::Unwrapped as Get>::Item: ComplexField {
            let builder = self.get_builder();
            unsafe { builder.wrap(VecConjugate { vec: self.unwrap() }) }
        }


        /// attaches a &mut RSMathVector to the first buffer
        /// note: due to current borrow checker limitations surrounding for<'a>, this isn't very useful in reality
        fn attach_slice<'a, T>(self, buf: &'a mut RSMathVector<T>) -> VecAttachSlice<'a, Self::Unwrapped, T>
        where 
            Self::Unwrapped: HasReuseBuf<FstHandleBool = N>,
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecAttachSlice {
                    vec: self.unwrap(),
                    buf,
                })
            }
        }

        /// creates a slice in the first buffer
        fn create_slice<T>(self) -> VecCreateSlice<Self::Unwrapped, T>
        where
            Self::Unwrapped: HasReuseBuf<FstHandleBool = N>,
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecCreateSlice {
                    vec: self.unwrap(),
                    buf: ManuallyDrop::new(Box::new_uninit_slice(builder.size())),
                })
            }
        }

        /// creates a array in the first buffer if there isn't already one there
        fn maybe_create_slice<T>(self) -> VecMaybeCreateSlice<Self::Unwrapped, T>
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
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecMaybeCreateSlice {
                    vec: self.unwrap(),
                    buf: ManuallyDrop::new(Box::new_uninit_slice(builder.size())),
                })
            }
        }
    );
    
    // vec x vec ops
    vec_op!(
        /// zips together the items of 2 vectors into 2 element tuples
        fn zip<{V}>(self, other: V) -> VecZip<Self::Unwrapped, V::Unwrapped> {
            let builder = self.get_builder().union(other.get_builder());
            unsafe {
                builder.wrap(VecZip {
                    l_vec: self.unwrap(),
                    r_vec: other.unwrap(),
                })
            }
        }

        /// adds 2 vectors
        fn add<{V}>(self, other: V) -> VecAdd<Self::Unwrapped, V::Unwrapped> where <Self::Unwrapped as Get>::Item: Add<<V::Unwrapped as Get>::Item> {
            let builder = self.get_builder().union(other.get_builder());
            unsafe {
                builder.wrap(VecAdd {
                    l_vec: self.unwrap(),
                    r_vec: other.unwrap(),
                })
            }
        }

        /// substracts the other vector from self
        fn sub<{V}>(self, other: V) -> VecSub<Self::Unwrapped, V::Unwrapped> where <Self::Unwrapped as Get>::Item: Sub<<V::Unwrapped as Get>::Item> {
            let builder = self.get_builder().union(other.get_builder());
            unsafe {
                builder.wrap(VecSub {
                    l_vec: self.unwrap(),
                    r_vec: other.unwrap(),
                })
            }
        }

        /// component-wise multiplies 2 vectors
        fn comp_mul<{V}>(self, other: V) -> VecCompMul<Self::Unwrapped, V::Unwrapped> where <Self::Unwrapped as Get>::Item: Mul<<V::Unwrapped as Get>::Item> {
            let builder = self.get_builder().union(other.get_builder());
            unsafe {
                builder.wrap(VecCompMul {
                    l_vec: self.unwrap(),
                    r_vec: other.unwrap(),
                })
            }
        }

        /// component-wise divides self by other
        fn comp_div<{V}>(self, other: V) -> VecCompDiv<Self::Unwrapped, V::Unwrapped> where <Self::Unwrapped as Get>::Item: Div<<V::Unwrapped as Get>::Item> {
            let builder = self.get_builder().union(other.get_builder());
            unsafe {
                builder.wrap(VecCompDiv {
                    l_vec: self.unwrap(),
                    r_vec: other.unwrap(),
                })
            }
        }

        /// component-wise get remainder (%) of self by other
        fn comp_rem<{V}>(self, other: V) -> VecCompRem<Self::Unwrapped, V::Unwrapped> where <Self::Unwrapped as Get>::Item: Rem<<V::Unwrapped as Get>::Item> {
            let builder = self.get_builder().union(other.get_builder());
            unsafe {
                builder.wrap(VecCompRem {
                    l_vec: self.unwrap(),
                    r_vec: other.unwrap(),
                })
            }
        }
        
        /// add assigns (+=) the self's items (&mut T) with other
        fn add_assign<'a, {V}, I: 'a | AddAssign<<V::Unwrapped as Get>::Item>>(self, other: V) -> VecAddAssign<'a, Self::Unwrapped, V::Unwrapped, I>
        where Self::Unwrapped: Get<Item = &'a mut I>,
        {
            let builder = self.get_builder().union(other.get_builder());
            unsafe {
                builder.wrap(VecAddAssign {
                    l_vec: self.unwrap(),
                    r_vec: other.unwrap(),
                })
            }
        }
    
        /// sub assigns (-=) the self's items (&mut T) with other
        fn sub_assign<'a, {V}, I: 'a | SubAssign<<V::Unwrapped as Get>::Item>>(self, other: V) -> VecSubAssign<'a, Self::Unwrapped, V::Unwrapped, I>
        where Self::Unwrapped: Get<Item = &'a mut I>,
        {
            let builder = self.get_builder().union(other.get_builder());
            unsafe {
                builder.wrap(VecSubAssign {
                    l_vec: self.unwrap(),
                    r_vec: other.unwrap(),
                })
            }
        }
    
        /// mul assigns (*=) the self's items (&mut T) with other
        fn comp_mul_assign<'a, {V}, I: 'a | MulAssign<<V::Unwrapped as Get>::Item>>(self, other: V) -> VecCompMulAssign<'a, Self::Unwrapped, V::Unwrapped, I>
        where Self::Unwrapped: Get<Item = &'a mut I>,
        {
            let builder = self.get_builder().union(other.get_builder());
            unsafe {
                builder.wrap(VecCompMulAssign {
                    l_vec: self.unwrap(),
                    r_vec: other.unwrap(),
                })
            }
        }
    
        /// div assigns (/=) the self's items (&mut T) with other
        fn comp_div_assign<'a, {V}, I: 'a | DivAssign<<V::Unwrapped as Get>::Item>>(self, other: V) -> VecCompDivAssign<'a, Self::Unwrapped, V::Unwrapped, I>
        where Self::Unwrapped: Get<Item = &'a mut I>,
        {
            let builder = self.get_builder().union(other.get_builder());
            unsafe {
                builder.wrap(VecCompDivAssign {
                    l_vec: self.unwrap(),
                    r_vec: other.unwrap(),
                })
            }
        }
    
        /// rem assigns (%=) the self's items (&mut T) with other
        fn comp_rem_assign<'a, {V}, I: 'a | RemAssign<<V::Unwrapped as Get>::Item>>(self, other: V) -> VecCompRemAssign<'a, Self::Unwrapped, V::Unwrapped, I>
        where Self::Unwrapped: Get<Item = &'a mut I>,
        {
            let builder = self.get_builder().union(other.get_builder());
            unsafe {
                builder.wrap(VecCompRemAssign {
                    l_vec: self.unwrap(),
                    r_vec: other.unwrap(),
                })
            }
        }
        /// calculates the dot product of 2 vectors and adds it to the output
        fn dot<{V}, S: num_traits::Zero | AddAssign<<<Self::Unwrapped as Get>::Item as Mul<<V::Unwrapped as Get>::Item>>::Output>>(self, other: V) -> {has_output} VecDot<Self::Unwrapped, V::Unwrapped, S>, output: S
        where <Self::Unwrapped as Get>::Item: Mul<<V::Unwrapped as Get>::Item> 
        {
            let builder = self.get_builder().union(other.get_builder());
            unsafe {
                builder.wrap(VecDot {
                    l_vec: self.unwrap(),
                    r_vec: other.unwrap(),
                    scalar: ManuallyDrop::new(S::zero()),
                })
            }
        }
        
        /// calculates the euclidean inner product of 2 vectors and adds it to the output
        fn euclidean_inner_prod<{V}, S: num_traits::Zero | AddAssign<<<Self::Unwrapped as Get>::Item as Mul<<V::Unwrapped as Get>::Item>>::Output>>(self, other: V) -> {has_output} VecEuclidInnerProd<Self::Unwrapped, V::Unwrapped, S>, output: S
        where
            <Self::Unwrapped as Get>::Item: ComplexField | Mul<<V::Unwrapped as Get>::Item>,
            <V::Unwrapped as Get>::Item: ComplexField,
        {
            let builder = self.get_builder().union(other.get_builder());
            unsafe {
                builder.wrap(VecEuclidInnerProd {
                    l_vec: self.unwrap(),
                    r_vec: other.unwrap(),
                    scalar: ManuallyDrop::new(S::zero()),
                })
            }
        }
    
    
        /// calculates the dot product of 2 vectors and adds it to the output
        fn copied_dot<{V}, S: num_traits::Zero | AddAssign<<<Self::Unwrapped as Get>::Item as Mul<<V::Unwrapped as Get>::Item>>::Output>>(self, other: V) -> {has_output} VecCopiedDot<Self::Unwrapped, V::Unwrapped, S>, output: S
        where
            <Self::Unwrapped as Get>::Item: Copy | Mul<<V::Unwrapped as Get>::Item>,
            <V::Unwrapped as Get>::Item: Copy,
        {
            let builder = self.get_builder().union(other.get_builder());
            unsafe {
                builder.wrap(VecCopiedDot {
                    l_vec: self.unwrap(),
                    r_vec: other.unwrap(),
                    scalar: ManuallyDrop::new(S::zero()),
                })
            }
        }
    );

    /// multiplies this vector with a matrix (V * M)
    fn vec_mat_mul<
        M: MatrixOps,
        O: num_traits::Zero + AddAssign<
            <<Self::Unwrapped as Get>::Item as Mul<<M::Unwrapped as Get2D>::Item>>::Output,
        >,
    >(
        self,
        mat: M,
    ) -> <<M::Builder as MatrixBuilder>::RowBuilder as VectorBuilder>::Wrapped<
        VecMatMul<
            Self::Unwrapped,
            M::Unwrapped,
            <Self::Builder as VectorBuilderUnion<<M::Builder as MatrixBuilder>::ColBuilder>>::Union,
            O,
        >,
    >
    where
        Self::Unwrapped: IsRepeatable,
        <Self::Unwrapped as Get>::Item: Mul<<M::Unwrapped as Get2D>::Item>,
        MatrixColumn<M::Unwrapped>:
            HasReuseBuf<BoundTypes = <MatrixColumn<M::Unwrapped> as Get>::BoundItems>,
        (
            <Self::Unwrapped as HasOutput>::OutputBool,
            <MatColVectorExprs<M::Unwrapped> as HasOutput>::OutputBool,
        ): FilterPair,
        Self::Builder: VectorBuilderUnion<<M::Builder as MatrixBuilder>::ColBuilder>,
    {
        unsafe {
            let (mat_col_builder, mat_row_builder) = mat.get_builder().decompose();
            let vec_builder = self.get_builder();
            mat_row_builder.wrap(VecMatMul {
                mat: MatColVectorExprs { mat: mat.unwrap() },
                vec: self.unwrap(),
                inner_builder: vec_builder.union(mat_col_builder),
                phantom: PhantomData::<O>,
            })
        }
    }
}

/// a trait with various vector operations for const sized vectors
// TODO: rename?
pub trait ArrayVectorOps<const D: usize>: VectorOps {
    vec_op!(
        /// attaches a &mut MathVector to the first buffer
        /// note: due to current borrow checker limitations surrounding for<'a>, this isn't very useful in reality
        fn attach_array<'a, T>(self, buf: &'a mut MathVector<T, D>) -> VecAttachArray<'a, Self::Unwrapped, T, D>
        where
            Self::Unwrapped: HasReuseBuf<FstHandleBool = N>,
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecAttachArray {
                    vec: self.unwrap(),
                    buf,
                })
            }
        }
    
        /// creates a array in the first buffer
        fn create_array<T>(self) -> VecCreateArray<Self::Unwrapped, T, D>
        where
            Self::Unwrapped: HasReuseBuf<FstHandleBool = N>,
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecCreateArray {
                    vec: self.unwrap(),
                    buf: MaybeUninit::uninit().assume_init(),
                })
            }
        }
    
        /// creates a array on the heap in the first buffer
        /// note: a pre-existing buffer may or may not be owned by the vector
        fn create_heap_array<T>(self) -> VecCreateHeapArray<Self::Unwrapped, T, D>
        where
            Self::Unwrapped: HasReuseBuf<FstHandleBool = N>,
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecCreateHeapArray {
                    vec: self.unwrap(),
                    buf: ManuallyDrop::new(Box::new(MaybeUninit::uninit().assume_init())),
                })
            }
        }
    
        /// creates a array in the first buffer if there isn't already one there
        fn maybe_create_array<T>(self) -> VecMaybeCreateArray<Self::Unwrapped, T, D>
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
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecMaybeCreateArray {
                    vec: self.unwrap(),
                    buf: MaybeUninit::uninit().assume_init(),
                })
            }
        }
    
        /// creates a array on the heap in the first buffer if there isn't already one there
        /// note: a pre-existing buffer may or may not be on the heap or owned by the vector
        fn maybe_create_heap_array<T>(self) -> VecMaybeCreateHeapArray<Self::Unwrapped, T, D>
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
        {
            let builder = self.get_builder();
            unsafe {
                builder.wrap(VecMaybeCreateHeapArray {
                    vec: self.unwrap(),
                    buf: ManuallyDrop::new(Box::new(MaybeUninit::uninit().assume_init())),
                })
            }
        }  
    );
}

/// a trait enabling a vector to be evaluated inplace (without offloading its outputs and buffers)
pub trait VectorInPlaceEvalOps: VectorOps {
    type ConcreteVectorLike: VectorLike;
    type UsedVector: VectorLike;

    /// turns the vector into a concrete one by evaluating and storing it
    fn eval_in_place(self) -> <Self::Builder as VectorBuilder>::Wrapped<
        VecAttachUsedVec<Self::ConcreteVectorLike, Self::UsedVector>,
    > where 
        <Self::Builder as VectorBuilder>::Wrapped<
            VecAttachUsedVec<Self::ConcreteVectorLike, Self::UsedVector>,
        >: ConcreteVectorExpr,

        <<Self::Builder as VectorBuilder>::Wrapped<
            VecAttachUsedVec<Self::ConcreteVectorLike, Self::UsedVector>,
        > as VectorOps>::Unwrapped: Get<Item = 
            <
                <Self::Builder as VectorBuilder>::Wrapped<
                    VecAttachUsedVec<Self::ConcreteVectorLike, Self::UsedVector>,
                > as Index<usize>
            >::Output
        >,

        <
            <Self::Builder as VectorBuilder>::Wrapped<
                VecAttachUsedVec<Self::ConcreteVectorLike, Self::UsedVector>,
            > as Index<usize>
        >::Output: Sized,

        (
            <Self::ConcreteVectorLike as HasOutput>::OutputBool,
            <Self::UsedVector as HasOutput>::OutputBool,
        ): FilterPair,
        (
            <Self::ConcreteVectorLike as HasReuseBuf>::FstHandleBool,
            <Self::UsedVector as HasReuseBuf>::FstHandleBool,
        ): SelectPair,
        (
            <Self::ConcreteVectorLike as HasReuseBuf>::SndHandleBool,
            <Self::UsedVector as HasReuseBuf>::SndHandleBool,
        ): SelectPair,
        (
            <Self::ConcreteVectorLike as HasReuseBuf>::BoundHandlesBool,
            <Self::UsedVector as HasReuseBuf>::BoundHandlesBool,
        ): FilterPair,
        (
            <Self::ConcreteVectorLike as HasReuseBuf>::FstOwnedBufferBool,
            <Self::UsedVector as HasReuseBuf>::FstOwnedBufferBool,
        ): SelectPair,
        (
            <Self::ConcreteVectorLike as HasReuseBuf>::SndOwnedBufferBool,
            <Self::UsedVector as HasReuseBuf>::SndOwnedBufferBool,
        ): SelectPair,
        (
            <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
            <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool,
        ): TyBoolPair,
        <(
            <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
            <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool,
        ) as TyBoolPair>::Or: IsTrue,
        Self: Sized,
    ;
}

pub trait VectorEvalOps: VectorOps {
    type MaybeCreateBuffer<V: VectorLike>: VectorLike<FstHandleBool = Y>
    where
        <<V as HasReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
        (
            <V as HasReuseBuf>::FstHandleBool,
            <<V as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
        ): SelectPair,
        (
            <V as HasReuseBuf>::FstOwnedBufferBool,
            <<V as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
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
        Self: Sized;

    /// evaluates the VectorExpr and returns the resulting vector alongside its output (if present) without cleanly filtering it with the output
    /// if the VectorExpr has no item (& thus results in a vector w/ ZST elements) or the item is irrelevent, see consume to not return that vector
    /// will try to use the first buffer if available (fails if the provided buffer is not bindable to the output)
    ///
    /// Warning:
    /// if this method is trying the evaluate the vector *onto the stack*, it is very possible to overflow the stack with larger vectors
    /// use heap_eval if this is a concern
    ///
    /// Note:
    /// methods like sum, product, or fold can place build up outputs
    ///
    /// output is generally nested 2 element tuples
    /// newer values to the right
    /// binary operators merge the output of the 2 vectors
    #[inline]
    fn raw_eval(
        self,
    ) -> (
        <Self::MaybeCreateBuffer<Self::Unwrapped> as HasOutput>::Output,
        <Self::MaybeCreateBuffer<Self::Unwrapped> as HasReuseBuf>::FstOwnedBuffer,
    )
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
        (
            <Self::MaybeCreateBuffer<Self::Unwrapped> as HasReuseBuf>::BoundHandlesBool,
            Y,
        ): FilterPair,
        (
            <Self::MaybeCreateBuffer<Self::Unwrapped> as HasOutput>::OutputBool,
            <Self::MaybeCreateBuffer<Self::Unwrapped> as HasReuseBuf>::FstOwnedBufferBool,
        ): TyBoolPair,
        VecRawBind<Self::MaybeCreateBuffer<Self::Unwrapped>>: HasReuseBuf<
            BoundTypes = <VecRawBind<Self::MaybeCreateBuffer<Self::Unwrapped>> as Get>::BoundItems,
        >,
        Self: Sized,
    {
        let builder = self.get_builder();
        unsafe {
            VectorIter::new_from_parts(
                VecRawBind {
                    vec: self.maybe_create_buffer(),
                },
                builder,
            )
            .consume()
        }
    }

    /// evaluates the VectorExpr and returns the resulting vector alongside its output (if present)
    /// if the VectorExpr has no item (& thus results in a vector w/ ZST elements) or the item is irrelevent, see consume to not return that vector
    /// will try to use the first buffer if available (fails if the provided buffer is not bindable to the output)
    ///
    /// Warning:
    /// if this method is trying the evaluate the vector *onto the stack*, it is very possible to overflow the stack with larger vectors
    /// use heap_eval if this is a concern
    ///
    /// Note:
    /// methods like sum, product, or fold can place build up outputs
    ///
    /// output is generally nested 2 element tuples
    /// newer values to the right
    /// binary operators merge the output of the 2 vectors
    #[inline]
    fn eval(self) -> <VecBind<Self::MaybeCreateBuffer<Self::Unwrapped>> as HasOutput>::Output
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
        (
            <Self::MaybeCreateBuffer<Self::Unwrapped> as HasReuseBuf>::BoundHandlesBool,
            Y,
        ): FilterPair,
        (
            <Self::MaybeCreateBuffer<Self::Unwrapped> as HasOutput>::OutputBool,
            <Self::MaybeCreateBuffer<Self::Unwrapped> as HasReuseBuf>::FstOwnedBufferBool,
        ): FilterPair,
        VecBind<Self::MaybeCreateBuffer<Self::Unwrapped>>: HasReuseBuf<
            BoundTypes = <VecBind<Self::MaybeCreateBuffer<Self::Unwrapped>> as Get>::BoundItems,
        >,
        Self: Sized,
    {
        let builder = self.get_builder();
        unsafe {
            VectorIter::new_from_parts(
                VecBind {
                    vec: self.maybe_create_buffer(),
                },
                builder,
            )
            .consume()
        }
    }
}

// // not currently possible since nothing *technically* changes when D does, possible when ArrayVectorOps later uses assoc const
//impl<V, const D: usize> VectorEvalOps for V where V: ArrayVectorOps<D> {
//    const dummy: usize = D;
//
//    type MaybeCreateBuffer<T: VectorLike> = VecMaybeCreateArray<T, <T as Get>::Item, D> where
//        <<T as HasReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
//        (<T as HasReuseBuf>::FstHandleBool, <<T as HasReuseBuf>::FstHandleBool as TyBool>::Neg): SelectPair,
//        (<T as HasReuseBuf>::FstOwnedBufferBool, <<T as HasReuseBuf>::FstHandleBool as TyBool>::Neg): TyBoolPair,
//    ;
//
//    fn maybe_create_buffer(self) -> <Self::Builder as VectorBuilder>::Wrapped<Self::MaybeCreateBuffer<Self::Unwrapped>> where
//        <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
//        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg): SelectPair,
//        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg): TyBoolPair,
//        Self: Sized,
//    {
//        self.maybe_create_array()
//    }
//}

unsafe impl<V: VectorLike, const D: usize> VectorOps for VectorExpr<V, D> {
    type Unwrapped = V;
    type Builder = VectorExprBuilder<D>;

    #[inline]
    fn unwrap(self) -> Self::Unwrapped {
        // safe because this is just move done manually as VectorExpr impls Drop
        // normally a problem as this leaves the fields of the struct at potentially
        // invalid states which are assumed valid by the drop impl, however we just
        // disable dropping temporarily so this isn't a concern
        // does lead to leaking however, but it is ultimately fixed by wrap and the interim
        // (should) be non-panicking so leaking shouldn't happen
        unsafe { ptr::read(&ManuallyDrop::new(self).0) }
    }
    #[inline]
    fn get_builder(&self) -> Self::Builder {
        VectorExprBuilder
    }
    #[inline]
    fn size(&self) -> usize {
        D
    }
}
impl<V: VectorLike, const D: usize> ArrayVectorOps<D> for VectorExpr<V, D> {}


impl<V: VectorLike, const D: usize> VectorInPlaceEvalOps for VectorExpr<V, D> 
where 
    <V::FstHandleBool as TyBool>::Neg: Filter,
    (V::BoundHandlesBool, Y): FilterPair,
    (V::FstHandleBool, <V::FstHandleBool as TyBool>::Neg): SelectPair<
    Selected<V::FstOwnedBuffer, MathVector<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<<V as Get>::Item>, D>> = MathVector<V::Item, D>>,
    (V::FstOwnedBufferBool, <V::FstHandleBool as TyBool>::Neg): TyBoolPair,
    (<V::FstHandleBool as TyBool>::Neg, V::FstOwnedBufferBool): TyBoolPair,
    (V::OutputBool, <(V::FstOwnedBufferBool, <V::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or): FilterPair,
    VecHalfBind<VecMaybeCreateArray<V, V::Item, D>>: HasReuseBuf<BoundTypes = <(V::BoundHandlesBool, Y) as FilterPair>::Filtered<V::BoundItems, V::Item>>,
    (N, <(V::OutputBool, <(V::FstOwnedBufferBool, <V::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or) as TyBoolPair>::Or): FilterPair,
    (N, <VecHalfBind<VecMaybeCreateArray<V, V::Item, D>> as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
    (N, <VecHalfBind<VecMaybeCreateArray<V, V::Item, D>> as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
    (N, <VecHalfBind<VecMaybeCreateArray<V, V::Item, D>> as HasReuseBuf>::FstHandleBool): SelectPair,
    (N, <VecHalfBind<VecMaybeCreateArray<V, V::Item, D>> as HasReuseBuf>::SndHandleBool): SelectPair,
    (N, <VecHalfBind<VecMaybeCreateArray<V, V::Item, D>> as HasReuseBuf>::BoundHandlesBool): FilterPair,
{
    type ConcreteVectorLike = VectorArray<V::Item, D>;
    type UsedVector = VecHalfBind<VecMaybeCreateArray<V, V::Item, D>>;

    fn eval_in_place(self) -> <Self::Builder as VectorBuilder>::Wrapped<
            VecAttachUsedVec<Self::ConcreteVectorLike, Self::UsedVector>,
        > where 
            (
                <Self::ConcreteVectorLike as HasOutput>::OutputBool,
                <Self::UsedVector as HasOutput>::OutputBool,
            ): FilterPair,
            (
                <Self::ConcreteVectorLike as HasReuseBuf>::FstHandleBool,
                <Self::UsedVector as HasReuseBuf>::FstHandleBool,
            ): SelectPair,
            (
                <Self::ConcreteVectorLike as HasReuseBuf>::SndHandleBool,
                <Self::UsedVector as HasReuseBuf>::SndHandleBool,
            ): SelectPair,
            (
                <Self::ConcreteVectorLike as HasReuseBuf>::BoundHandlesBool,
                <Self::UsedVector as HasReuseBuf>::BoundHandlesBool,
            ): FilterPair,
            (
                <Self::ConcreteVectorLike as HasReuseBuf>::FstOwnedBufferBool,
                <Self::UsedVector as HasReuseBuf>::FstOwnedBufferBool,
            ): SelectPair,
            (
                <Self::ConcreteVectorLike as HasReuseBuf>::SndOwnedBufferBool,
                <Self::UsedVector as HasReuseBuf>::SndOwnedBufferBool,
            ): SelectPair,
            (
                <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
                <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool,
            ): TyBoolPair,
            <(
                <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
                <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool,
            ) as TyBoolPair>::Or: IsTrue,
            
            Self: Sized, 
    {
        let builder = self.get_builder();
        let mut vec_iter = self.maybe_create_array().half_bind().into_iter();
        unsafe {
            vec_iter.no_output_consume();
            builder.wrap(VecAttachUsedVec{vec: vec_iter.vec.get_bound_buf().unwrap(), used_vec: ptr::read(&vec_iter.vec), size: builder.size()})
        }
    }
}

impl<V: VectorLike, const D: usize> VectorEvalOps for VectorExpr<V, D> {
    type MaybeCreateBuffer<T: VectorLike>
        = VecMaybeCreateArray<T, <T as Get>::Item, D>
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
        Self: Sized,
    {
        self.maybe_create_array().unwrap()
    }
}

unsafe impl<V: VectorLike, const D: usize> VectorOps for Box<VectorExpr<V, D>> {
    type Unwrapped = Box<V>;
    type Builder = VectorExprBuilder<D>;

    #[inline]
    fn unwrap(self) -> Self::Unwrapped {
        // safe because this is just move done manually as VectorExpr impls Drop
        // normally a problem as this leaves the fields of the struct at potentially
        // invalid states which are assumed valid by the drop impl, however we just
        // disable dropping temporarily so this isn't a concern
        // does lead to leaking however, but it is ultimately fixed by wrap and the interim
        // (should) be non-panicking so leaking shouldn't happen
        Box::new(unsafe { ptr::read(&ManuallyDrop::new(self).0) })
    }
    #[inline]
    fn get_builder(&self) -> Self::Builder {
        VectorExprBuilder
    }
    #[inline]
    fn size(&self) -> usize {
        D
    }
}
impl<V: VectorLike, const D: usize> ArrayVectorOps<D> for Box<VectorExpr<V, D>> {}

impl<V: VectorLike, const D: usize> VectorInPlaceEvalOps for Box<VectorExpr<V, D>>
where 
    <V::FstHandleBool as TyBool>::Neg: Filter,
    (V::BoundHandlesBool, Y): FilterPair,
    (V::FstHandleBool, <V::FstHandleBool as TyBool>::Neg): SelectPair<
        Selected<V::FstOwnedBuffer, Box<MathVector<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<<V as Get>::Item>, D>>> = Box<MathVector<V::Item, D>>
    >,
    (V::FstOwnedBufferBool, <V::FstHandleBool as TyBool>::Neg): TyBoolPair,
    (<V::FstHandleBool as TyBool>::Neg, V::FstOwnedBufferBool): TyBoolPair,
    (V::OutputBool, <(V::FstOwnedBufferBool, <V::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or): FilterPair,
    VecHalfBind<VecMaybeCreateHeapArray<Box<V>, V::Item, D>>: HasReuseBuf<BoundTypes = <(V::BoundHandlesBool, Y) as FilterPair>::Filtered<V::BoundItems, V::Item>>,
    (N, <(V::OutputBool, <(V::FstOwnedBufferBool, <V::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or) as TyBoolPair>::Or): FilterPair,
    (N, <VecHalfBind<VecMaybeCreateHeapArray<Box<V>, V::Item, D>> as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
    (N, <VecHalfBind<VecMaybeCreateHeapArray<Box<V>, V::Item, D>> as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
    (N, <VecHalfBind<VecMaybeCreateHeapArray<Box<V>, V::Item, D>> as HasReuseBuf>::FstHandleBool): SelectPair,
    (N, <VecHalfBind<VecMaybeCreateHeapArray<Box<V>, V::Item, D>> as HasReuseBuf>::SndHandleBool): SelectPair,
    (N, <VecHalfBind<VecMaybeCreateHeapArray<Box<V>, V::Item, D>> as HasReuseBuf>::BoundHandlesBool): FilterPair,
{
    type ConcreteVectorLike = Box<VectorArray<V::Item, D>>;
    type UsedVector = VecHalfBind<VecMaybeCreateHeapArray<Box<V>, V::Item, D>>;

    fn eval_in_place(self) -> <Self::Builder as VectorBuilder>::Wrapped<
            VecAttachUsedVec<Self::ConcreteVectorLike, Self::UsedVector>,
        > where 
            (
                <Self::ConcreteVectorLike as HasOutput>::OutputBool,
                <Self::UsedVector as HasOutput>::OutputBool,
            ): FilterPair,
            (
                <Self::ConcreteVectorLike as HasReuseBuf>::FstHandleBool,
                <Self::UsedVector as HasReuseBuf>::FstHandleBool,
            ): SelectPair,
            (
                <Self::ConcreteVectorLike as HasReuseBuf>::SndHandleBool,
                <Self::UsedVector as HasReuseBuf>::SndHandleBool,
            ): SelectPair,
            (
                <Self::ConcreteVectorLike as HasReuseBuf>::BoundHandlesBool,
                <Self::UsedVector as HasReuseBuf>::BoundHandlesBool,
            ): FilterPair,
            (
                <Self::ConcreteVectorLike as HasReuseBuf>::FstOwnedBufferBool,
                <Self::UsedVector as HasReuseBuf>::FstOwnedBufferBool,
            ): SelectPair,
            (
                <Self::ConcreteVectorLike as HasReuseBuf>::SndOwnedBufferBool,
                <Self::UsedVector as HasReuseBuf>::SndOwnedBufferBool,
            ): SelectPair,
            (
                <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
                <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool,
            ): TyBoolPair,
            <(
                <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
                <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool,
            ) as TyBoolPair>::Or: IsTrue,
            
            Self: Sized, 
    {
        let builder = self.get_builder();
        let mut vec_iter = self.maybe_create_heap_array().half_bind().into_iter();
        unsafe {
            vec_iter.no_output_consume();
            builder.wrap(VecAttachUsedVec{vec: vec_iter.vec.get_bound_buf().unwrap(), used_vec: ptr::read(&vec_iter.vec), size: builder.size()})
        }
    }
}

impl<V: VectorLike, const D: usize> VectorEvalOps for Box<VectorExpr<V, D>> {
    type MaybeCreateBuffer<T: VectorLike>
        = VecMaybeCreateHeapArray<T, <T as Get>::Item, D>
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
        Self: Sized,
    {
        self.maybe_create_heap_array().unwrap()
    }
}

//already repeatable / can't truly be made repeatable so not implemented
unsafe impl<'a, T, const D: usize> VectorOps for &'a MathVector<T, D> {
    type Unwrapped = &'a [T; D];
    type Builder = VectorExprBuilder<D>;

    #[inline]
    fn unwrap(self) -> Self::Unwrapped {
        &self.0
    }
    #[inline]
    fn get_builder(&self) -> Self::Builder {
        VectorExprBuilder
    }
    #[inline]
    fn size(&self) -> usize {
        D
    }
}
impl<T, const D: usize> ArrayVectorOps<D> for &MathVector<T, D> {}

impl<T, const D: usize> VectorEvalOps for &MathVector<T, D> {
    type MaybeCreateBuffer<V: VectorLike>
        = VecMaybeCreateArray<V, <V as Get>::Item, D>
    where
        <<V as HasReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
        (
            <V as HasReuseBuf>::FstHandleBool,
            <<V as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
        ): SelectPair,
        (
            <V as HasReuseBuf>::FstOwnedBufferBool,
            <<V as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
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
        Self: Sized,
    {
        self.maybe_create_array().unwrap()
    }
}

unsafe impl<'a, T, const D: usize> VectorOps for &'a mut MathVector<T, D> {
    type Unwrapped = &'a mut [T; D];
    type Builder = VectorExprBuilder<D>;

    #[inline]
    fn unwrap(self) -> Self::Unwrapped {
        &mut self.0
    }
    #[inline]
    fn get_builder(&self) -> Self::Builder {
        VectorExprBuilder
    }
    #[inline]
    fn size(&self) -> usize {
        D
    }
}
impl<T, const D: usize> ArrayVectorOps<D> for &mut MathVector<T, D> {}

impl<T, const D: usize> VectorEvalOps for &mut MathVector<T, D> {
    type MaybeCreateBuffer<V: VectorLike>
        = VecMaybeCreateArray<V, <V as Get>::Item, D>
    where
        <<V as HasReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
        (
            <V as HasReuseBuf>::FstHandleBool,
            <<V as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
        ): SelectPair,
        (
            <V as HasReuseBuf>::FstOwnedBufferBool,
            <<V as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
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
        Self: Sized,
    {
        self.maybe_create_array().unwrap()
    }
}

unsafe impl<'a, T, const D: usize> VectorOps for &'a Box<MathVector<T, D>> {
    type Unwrapped = &'a [T; D];
    type Builder = VectorExprBuilder<D>;

    #[inline]
    fn unwrap(self) -> Self::Unwrapped {
        &self.0
    }
    #[inline]
    fn get_builder(&self) -> Self::Builder {
        VectorExprBuilder
    }
    #[inline]
    fn size(&self) -> usize {
        D
    }
}
impl<T, const D: usize> ArrayVectorOps<D> for &Box<MathVector<T, D>> {}

impl<T, const D: usize> VectorEvalOps for &Box<MathVector<T, D>> {
    type MaybeCreateBuffer<V: VectorLike>
        = VecMaybeCreateHeapArray<V, <V as Get>::Item, D>
    where
        <<V as HasReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
        (
            <V as HasReuseBuf>::FstHandleBool,
            <<V as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
        ): SelectPair,
        (
            <V as HasReuseBuf>::FstOwnedBufferBool,
            <<V as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
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
        Self: Sized,
    {
        self.maybe_create_heap_array().unwrap()
    }
}

unsafe impl<'a, T, const D: usize> VectorOps for &'a mut Box<MathVector<T, D>> {
    type Unwrapped = &'a mut [T; D];
    type Builder = VectorExprBuilder<D>;

    #[inline]
    fn unwrap(self) -> Self::Unwrapped {
        &mut self.0
    }
    #[inline]
    fn get_builder(&self) -> Self::Builder {
        VectorExprBuilder
    }
    #[inline]
    fn size(&self) -> usize {
        D
    }
}
impl<T, const D: usize> ArrayVectorOps<D> for &mut Box<MathVector<T, D>> {}

impl<T, const D: usize> VectorEvalOps for &mut Box<MathVector<T, D>> {
    type MaybeCreateBuffer<V: VectorLike>
        = VecMaybeCreateHeapArray<V, <V as Get>::Item, D>
    where
        <<V as HasReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
        (
            <V as HasReuseBuf>::FstHandleBool,
            <<V as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
        ): SelectPair,
        (
            <V as HasReuseBuf>::FstOwnedBufferBool,
            <<V as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
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
        Self: Sized,
    {
        self.maybe_create_heap_array().unwrap()
    }
}

unsafe impl<V: VectorLike> VectorOps for RSVectorExpr<V> {
    type Unwrapped = V;
    type Builder = RSVectorExprBuilder;

    #[inline]
    fn unwrap(self) -> Self::Unwrapped {
        // safe because this is just move done manually as VectorExpr impls Drop
        // normally a problem as this leaves the fields of the struct at potentially
        // invalid states which are assumed valid by the drop impl, however we just
        // disable dropping temporarily so this isn't a concern
        // does lead to leaking however, but it is ultimately fixed by wrap and the interim
        // (should) be non-panicking so leaking shouldn't happen
        unsafe { ptr::read(&ManuallyDrop::new(self).vec) }
    }
    #[inline]
    fn get_builder(&self) -> Self::Builder {
        RSVectorExprBuilder { size: self.size }
    }
    #[inline]
    fn size(&self) -> usize {
        self.size
    }
}


impl<V: VectorLike> VectorInPlaceEvalOps for RSVectorExpr<V> 
where
    <V::FstHandleBool as TyBool>::Neg: Filter,
    (V::BoundHandlesBool, Y): FilterPair,
    (V::FstHandleBool, <V::FstHandleBool as TyBool>::Neg): SelectPair<
        Selected<V::FstOwnedBuffer, RSMathVector<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<<V as Get>::Item>>> = RSMathVector<V::Item>>,
    (V::FstOwnedBufferBool, <V::FstHandleBool as TyBool>::Neg): TyBoolPair,
    (<V::FstHandleBool as TyBool>::Neg, V::FstOwnedBufferBool): TyBoolPair,
    (V::OutputBool, <(V::FstOwnedBufferBool, <V::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or): FilterPair,
    VecHalfBind<VecMaybeCreateSlice<V, V::Item>>: HasReuseBuf<BoundTypes = <(V::BoundHandlesBool, Y) as FilterPair>::Filtered<V::BoundItems, V::Item>>,
    (N, <(V::OutputBool, <(V::FstOwnedBufferBool, <V::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or) as TyBoolPair>::Or): FilterPair,
    (N, <VecHalfBind<VecMaybeCreateSlice<V, V::Item>> as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
    (N, <VecHalfBind<VecMaybeCreateSlice<V, V::Item>> as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
    (N, <VecHalfBind<VecMaybeCreateSlice<V, V::Item>> as HasReuseBuf>::FstHandleBool): SelectPair,
    (N, <VecHalfBind<VecMaybeCreateSlice<V, V::Item>> as HasReuseBuf>::SndHandleBool): SelectPair,
    (N, <VecHalfBind<VecMaybeCreateSlice<V, V::Item>> as HasReuseBuf>::BoundHandlesBool): FilterPair,
{
    type ConcreteVectorLike = VectorSlice<V::Item>;
    type UsedVector = VecHalfBind<VecMaybeCreateSlice<V, V::Item>>;

    fn eval_in_place(self) -> <Self::Builder as VectorBuilder>::Wrapped<
        VecAttachUsedVec<Self::ConcreteVectorLike, Self::UsedVector>,
    > where 
        (
            <Self::ConcreteVectorLike as HasOutput>::OutputBool,
            <Self::UsedVector as HasOutput>::OutputBool,
        ): FilterPair,
        (
            <Self::ConcreteVectorLike as HasReuseBuf>::FstHandleBool,
            <Self::UsedVector as HasReuseBuf>::FstHandleBool,
        ): SelectPair,
        (
            <Self::ConcreteVectorLike as HasReuseBuf>::SndHandleBool,
            <Self::UsedVector as HasReuseBuf>::SndHandleBool,
        ): SelectPair,
        (
            <Self::ConcreteVectorLike as HasReuseBuf>::BoundHandlesBool,
            <Self::UsedVector as HasReuseBuf>::BoundHandlesBool,
        ): FilterPair,
        (
            <Self::ConcreteVectorLike as HasReuseBuf>::FstOwnedBufferBool,
            <Self::UsedVector as HasReuseBuf>::FstOwnedBufferBool,
        ): SelectPair,
        (
            <Self::ConcreteVectorLike as HasReuseBuf>::SndOwnedBufferBool,
            <Self::UsedVector as HasReuseBuf>::SndOwnedBufferBool,
        ): SelectPair,
        (
            <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
            <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool,
        ): TyBoolPair,
        <(
            <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
            <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool,
        ) as TyBoolPair>::Or: IsTrue,
        Self: Sized,
    {
        let builder = self.get_builder();
        let mut vec_iter = self.maybe_create_slice().half_bind().into_iter();
        unsafe {
            vec_iter.no_output_consume();
            builder.wrap(VecAttachUsedVec{vec: vec_iter.vec.get_bound_buf().unwrap(), used_vec: ptr::read(&vec_iter.vec), size: builder.size()})
        }
    }
}

impl<V: VectorLike> VectorEvalOps for RSVectorExpr<V> {
    type MaybeCreateBuffer<T: VectorLike>
        = VecMaybeCreateSlice<T, <T as Get>::Item>
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
        Self: Sized,
    {
        self.maybe_create_slice().unwrap()
    }
}

unsafe impl<'a, T> VectorOps for &'a RSMathVector<T> {
    type Unwrapped = &'a [T];
    type Builder = RSVectorExprBuilder;

    #[inline]
    fn unwrap(self) -> Self::Unwrapped {
        &*self.vec
    }
    #[inline]
    fn get_builder(&self) -> Self::Builder {
        RSVectorExprBuilder { size: self.size }
    }
    #[inline]
    fn size(&self) -> usize {
        self.size
    }
}

impl<T> VectorEvalOps for &RSMathVector<T> {
    type MaybeCreateBuffer<V: VectorLike>
        = VecMaybeCreateSlice<V, <V as Get>::Item>
    where
        <<V as HasReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
        (
            <V as HasReuseBuf>::FstHandleBool,
            <<V as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
        ): SelectPair,
        (
            <V as HasReuseBuf>::FstOwnedBufferBool,
            <<V as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
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
        Self: Sized,
    {
        self.maybe_create_slice().unwrap()
    }
}

unsafe impl<'a, T> VectorOps for &'a mut RSMathVector<T> {
    type Unwrapped = &'a mut [T];
    type Builder = RSVectorExprBuilder;

    #[inline]
    fn unwrap(self) -> Self::Unwrapped {
        &mut *self.vec
    }
    #[inline]
    fn get_builder(&self) -> Self::Builder {
        RSVectorExprBuilder { size: self.size }
    }
    #[inline]
    fn size(&self) -> usize {
        self.size
    }
}

impl<T> VectorEvalOps for &mut RSMathVector<T> {
    type MaybeCreateBuffer<V: VectorLike>
        = VecMaybeCreateSlice<V, <V as Get>::Item>
    where
        <<V as HasReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
        (
            <V as HasReuseBuf>::FstHandleBool,
            <<V as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
        ): SelectPair,
        (
            <V as HasReuseBuf>::FstOwnedBufferBool,
            <<V as HasReuseBuf>::FstHandleBool as TyBool>::Neg,
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
        Self: Sized,
    {
        self.maybe_create_slice().unwrap()
    }
}

macro_rules! conditional_syntax {
    (
        $cond:tt {$($tt:tt)*}
    ) => {
        $($tt)*
    };
    (
        {$($tt:tt)*}
    ) => {}
}

macro_rules! not_conditional_syntax {
    (
        {$($tt:tt)*}
    ) => {
        $($tt)*
    };
    (
        $cond:tt {$($tt:tt)*}
    ) => {}
}

macro_rules! impl_ops_for_wrapper {
    (
        $(
            <$($($lifetime:lifetime),+, )? $($generic:ident $(:)? $($lifetime_bound:lifetime |)? $($fst_trait_bound:path $(| $trait_bound:path)*)?),+ $(,{$size:ident})?>,
            $ty:ty,
            trait_vector: $trait_vector:ty,
            true_vector: $true_vector:ty
            $(, $subset:literal)?;
        )*
    ) => {
        $(
            not_conditional_syntax!(
                $($subset)?
                {
                    impl<$($($lifetime),+, )? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: Copy $(, const $size: usize)?> Mul<Z> for $ty where (<$trait_vector as HasOutput>::OutputBool, N): FilterPair, <$trait_vector as Get>::Item: Mul<Z>, Self: Sized {
                        type Output =  <<$ty as VectorOps>::Builder as VectorBuilder>::Wrapped<VecMulL<$true_vector, Z>>;

                        #[inline]
                        fn mul(self, rhs: Z) -> Self::Output {
                            self.mul_l(rhs)
                        }
                    }

                    impl<$($($lifetime),+, )? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: Copy $(, const $size: usize)?> Div<Z> for $ty where (<$trait_vector as HasOutput>::OutputBool, N): FilterPair, <$trait_vector as Get>::Item: Div<Z>, Self: Sized {
                        type Output = <<$ty as VectorOps>::Builder as VectorBuilder>::Wrapped<VecDivL<$true_vector, Z>>;

                        #[inline]
                        fn div(self, rhs: Z) -> Self::Output {
                            self.div_l(rhs)
                        }
                    }

                    impl<$($($lifetime),+, )? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: Copy $(, const $size: usize)?> Rem<Z> for $ty where (<$trait_vector as HasOutput>::OutputBool, N): FilterPair, <$trait_vector as Get>::Item: Rem<Z>, Self: Sized {
                        type Output = <<$ty as VectorOps>::Builder as VectorBuilder>::Wrapped<VecRemL<$true_vector, Z>>;

                        #[inline]
                        fn rem(self, rhs: Z) -> Self::Output {
                            self.rem_l(rhs)
                        }
                    }

                    impl<
                        $($($lifetime),+, )?
                        $($generic: $($lifetime_bound |)? $($fst_trait_bound $(| $trait_bound)*)?),+,
                        V2: VectorOps
                        $(, const $size: usize)?
                    > Add<V2> for $ty where
                        <$ty as VectorOps>::Builder: VectorBuilderUnion<V2::Builder>,
                        <$trait_vector as Get>::Item: Add<<V2::Unwrapped as Get>::Item>,
                        (<$trait_vector as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool): FilterPair,
                        (<(<$trait_vector as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
                        (<$trait_vector as HasReuseBuf>::BoundHandlesBool, <V2::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
                        (<$trait_vector as HasReuseBuf>::FstHandleBool, <V2::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
                        (<$trait_vector as HasReuseBuf>::SndHandleBool, <V2::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
                        (<$trait_vector as HasReuseBuf>::FstOwnedBufferBool, <V2::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
                        (<$trait_vector as HasReuseBuf>::SndOwnedBufferBool, <V2::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
                        (N, <V2::Unwrapped as HasOutput>::OutputBool): FilterPair,
                        (<(N, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
                        (N, <V2::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
                        (N, <V2::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
                        (N, <V2::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
                        (N, <V2::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
                        (N, <V2::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
                        $ty: Sized
                    {
                        type Output = <<<$ty as VectorOps>::Builder as VectorBuilderUnion<V2::Builder>>::Union as VectorBuilder>::Wrapped<VecAdd<$true_vector, V2::Unwrapped>>;

                        fn add(self, rhs: V2) -> Self::Output {
                            VectorOps::add(self,rhs)
                        }
                    }

                    impl<
                        $($($lifetime),+, )?
                        $($generic: $($lifetime_bound |)? $($fst_trait_bound $(| $trait_bound)*)?),+,
                        V2: VectorOps
                        $(, const $size: usize)?
                    > Sub<V2> for $ty where
                        <$ty as VectorOps>::Builder: VectorBuilderUnion<V2::Builder>,
                        <$trait_vector as Get>::Item: Sub<<V2::Unwrapped as Get>::Item>,
                        (<$trait_vector as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool): FilterPair,
                        (<(<$trait_vector as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
                        (<$trait_vector as HasReuseBuf>::BoundHandlesBool, <V2::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
                        (<$trait_vector as HasReuseBuf>::FstHandleBool, <V2::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
                        (<$trait_vector as HasReuseBuf>::SndHandleBool, <V2::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
                        (<$trait_vector as HasReuseBuf>::FstOwnedBufferBool, <V2::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
                        (<$trait_vector as HasReuseBuf>::SndOwnedBufferBool, <V2::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
                        (N, <V2::Unwrapped as HasOutput>::OutputBool): FilterPair,
                        (<(N, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
                        (N, <V2::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
                        (N, <V2::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
                        (N, <V2::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
                        (N, <V2::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
                        (N, <V2::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
                        $ty: Sized
                    {
                        type Output = <<<$ty as VectorOps>::Builder as VectorBuilderUnion<V2::Builder>>::Union as VectorBuilder>::Wrapped<VecSub<$true_vector, V2::Unwrapped>>;

                        fn sub(self, rhs: V2) -> Self::Output {
                            VectorOps::sub(self,rhs)
                        }
                    }
                }
            );

            conditional_syntax!(
                $($size)?
                {
                    impl<$($($lifetime),+, )? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: AddAssign<<$trait_vector as Get>::Item> $(, const $size: usize)?> AddAssign<$ty> for MathVector<Z $(, $size)?>
                    where
                        (N, <$trait_vector as HasOutput>::OutputBool): FilterPair,
                        (<(N, <$trait_vector as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
                        (N, <$trait_vector as HasReuseBuf>::BoundHandlesBool): FilterPair,
                        (N, <$trait_vector as HasReuseBuf>::FstHandleBool): SelectPair,
                        (N, <$trait_vector as HasReuseBuf>::SndHandleBool): SelectPair,
                        (N, <$trait_vector as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
                        (N, <$trait_vector as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
                        $trait_vector: HasReuseBuf<BoundTypes = <$trait_vector as Get>::BoundItems>
                    {
                        #[inline]
                        fn add_assign(&mut self, rhs: $ty) {
                            VectorOps::add_assign(self, rhs).consume();
                        }
                    }

                    impl<$($($lifetime),+, )? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: SubAssign<<$trait_vector as Get>::Item> $(, const $size: usize)?> SubAssign<$ty> for MathVector<Z $(, $size)?>
                    where
                        (N, <$trait_vector as HasOutput>::OutputBool): FilterPair,
                        (<(N, <$trait_vector as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
                        (N, <$trait_vector as HasReuseBuf>::BoundHandlesBool): FilterPair,
                        (N, <$trait_vector as HasReuseBuf>::FstHandleBool): SelectPair,
                        (N, <$trait_vector as HasReuseBuf>::SndHandleBool): SelectPair,
                        (N, <$trait_vector as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
                        (N, <$trait_vector as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
                        $trait_vector: HasReuseBuf<BoundTypes = <$trait_vector as Get>::BoundItems>
                    {
                        #[inline]
                        fn sub_assign(&mut self, rhs: $ty) {
                            VectorOps::sub_assign(self, rhs).consume();
                        }
                    }
                }
            );
        )*
    };
}

impl_ops_for_wrapper!(
    <V: VectorLike, {D}>, VectorExpr<V, D>, trait_vector: V, true_vector: V;
    <V: VectorLike, {D}>, Box<VectorExpr<V, D>>, trait_vector: V, true_vector: Box<V>;
    <'a, T, {D}>, &'a MathVector<T,D>, trait_vector: &'a [T; D], true_vector: &'a [T; D];
    <'a, T, {D}>, &'a mut MathVector<T,D>, trait_vector: &'a mut [T; D], true_vector: &'a mut [T; D];
    <'a, T, {D}>, &'a Box<MathVector<T,D>>, trait_vector: &'a [T; D], true_vector: &'a [T; D];
    <'a, T, {D}>, &'a mut Box<MathVector<T,D>>, trait_vector: &'a mut [T; D], true_vector: &'a mut [T; D];

    <V: VectorLike>, RSVectorExpr<V>, trait_vector: V, true_vector: V;
    <'a, T>, RefRSMathVector<'a, T>, trait_vector: &'a [T], true_vector: &'a [T], true;
    <'a, T>, RefMutRSMathVector<'a, T>, trait_vector: &'a mut [T], true_vector: &'a mut [T], true;
    <'a, T>, &'a RSMathVector<T>, trait_vector: &'a [T], true_vector: &'a [T];
    <'a, T>, &'a mut RSMathVector<T>, trait_vector: &'a mut [T], true_vector: &'a mut [T];
);
