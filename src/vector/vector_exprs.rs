use crate::{
    trait_specialization_utils::*,
    util_traits::HasOutput,
};
use std::{
    iter::Sum,
    mem::{self, ManuallyDrop, transmute},
    ops::*,
    slice::SliceIndex,
};
use super::vec_util_traits::*;
use super::vector_structs::*;
use super::vector_math::{GenericInnerProduct, VectorInnerProdOps};
use super::*;

/// a trait expressing that an implementor's data from [`Get`] is stored and accessible, allowing it to be indexed and borrowed
pub trait ConcreteVectorExpr: VectorOps + IndexMut<usize> where 
    Self::Unwrapped: Get<Item = Self::Output>,
    Self::Output: Sized,
{
    /// The inner [`VectorLike`] contained in the borrowed version of this vector
    type ReferencedInner<'a>: VectorLike<Item = &'a Self::Output> + IsRepeatable where 
        Self: 'a,
    ;
    /// the borrowed version of this vector
    type Referenced<'a>: VectorOps<Unwrapped = Self::ReferencedInner<'a>> + Index<usize, Output = Self::Output> where
        Self: 'a,
    ;
    /// A borrowed version of this vector except that its items are copied
    type Copied<'a>: VectorOps<Unwrapped = VecCopy<'a, Self::ReferencedInner<'a>, Self::Output>> where
        Self::Output: Copy,
        Self: 'a,
    ;
    /// The inner [`VectorLike`] contained in the mutably borrowed version of this vector
    type ReferencedMutInner<'a>: VectorLike<Item = &'a mut Self::Output> where
        Self: 'a,
    ;
    /// the mutably borrowed version of this vector
    type ReferencedMut<'a>: VectorOps<Unwrapped = Self::ReferencedMutInner<'a>> + IndexMut<usize, Output = Self::Output> + IndexMut<usize> where 
        Self: 'a,
    ;

    /// create a borrowed version of this vector which contains a reference to each of its items
    fn borrow<'a>(&'a self) -> Self::Referenced<'a>;

    /// create a mutably borrowed version of this vector which contain a mutable reference to each of its items
    fn borrow_mut<'a>(&'a mut self) -> Self::ReferencedMut<'a>;

    /// create a borrowed version of this vector which contains a copy each of its items
    fn copy<'a>(&'a self) -> Self::Copied<'a> where Self::Output: Copy;
}

pub trait InitializableVectorExpr: VectorOps {
    fn new_filled(builder: Self::Builder, val: <Self::Unwrapped as Get>::Item) -> Self where <Self::Unwrapped as Get>::Item: Copy;
    fn new_zeroed(builder: Self::Builder) -> Self where <Self::Unwrapped as Get>::Item: num_traits::Zero + Copy { //not sure on the Copy requirement
        Self::new_filled(
            builder,
            <<Self::Unwrapped as Get>::Item as num_traits::Zero>::zero()
        )
    } 
    fn new_oned(builder: Self::Builder) -> Self where <Self::Unwrapped as Get>::Item: num_traits::One + Copy { // TBD name //not sure on the Copy requirement
        Self::new_filled(
            builder,
            <<Self::Unwrapped as Get>::Item as num_traits::One>::one()
        )
    } 
}

/// as always: this being 'unsafe' is tbd
pub unsafe trait UninitVectorExpr: VectorOps {
    type Uninitialized: VectorOps<Builder = Self::Builder>;

    fn new_uninit(builder: Self::Builder) -> Self::Uninitialized;
    unsafe fn assume_init(uninit: Self::Uninitialized) -> Self;
    unsafe fn init_index(uninit: &mut Self::Uninitialized, index: usize, val: <Self::Unwrapped as Get>::Item);
    unsafe fn drop_index(uninit: &mut Self::Uninitialized, index: usize);
    /// drop one times (so Output and Buffers)
    unsafe fn drop_ots(uninit: &mut Self::Uninitialized);
}



/// A const sized vector wrapper
/// 
/// V: the underlying VectorLike type, generally inferred or generic
/// D: the size of the vector
#[repr(transparent)]
pub struct VectorExpr<V: VectorLike, const D: usize>(pub(crate) V); // note: VectorExpr only holds fully unused VectorLike objects

impl<V: VectorLike, const D: usize> VectorExpr<V, D> {
    /// evaluates the [`VectorExpr`] and returns the resulting vector (on the heap) alongside its output (if present)
    /// if the [`VectorExpr`] has no item (& thus results in a vector w/ ZST elements), see consume to not return that vector
    /// will try to use the first buffer if available (fails if the provided buffer is not bindable to the output)
    ///
    /// Warning:
    /// this method may cause a stack overflow if not compiled with `--release`
    ///
    /// Note:
    /// methods like [`VectorOps::sum`], [`VectorOps::product`], or [`VectorOps::fold`] can place build up outputs
    ///
    /// output is generally nested 2 element tuples,
    /// newer values to the right,
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
    /// 
    /// panics if `index` is out of bounds (outside of 0..D)
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
    /// marks this [`MathVector`] to have its buffer reused
    /// 
    /// buffer placed on the first buffer
    #[inline]
    pub fn reuse(self) -> VectorExpr<ReplaceArray<T, D>, D> {
        VectorExpr(ReplaceArray(self.unwrap().0))
    }
    /// marks this [`MathVector`] to have its buffer reused while keeping it on the heap
    /// 
    /// buffer placed on the first buffer
    #[inline]
    pub fn heap_reuse(self: Box<Self>) -> VectorExpr<Box<ReplaceArray<T, D>>, D> {
        // Safety, series of equivilent types:
        // Box<MathVector<T,D>>
        // Box<VectorExpr<VectorArray<T, D>, D>>, de-alias MathVector
        // Box<ManuallyDrop<[T; D]>>, VectorExpr & VectorArray are transparent
        // VectorExpr<Box<ReplaceArray<T, D>>, D>, VectorExpr & ReplaceArray are transparent
        unsafe { mem::transmute::<Box<Self>, VectorExpr<Box<ReplaceArray<T, D>>, D>>(self) }
    }

    /// converts this [`MathVector`] to a repeatable [`VectorExpr`] w/ Item = `&'a T`
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
    /// 
    /// safety: index is in bounds (0..D)
    #[inline]
    pub unsafe fn get_unchecked<I: SliceIndex<[T]>>(&self, index: I) -> &I::Output {
        unsafe { self.0.0.get_unchecked(index) }
    }

    /// mutably references the element at index without checking bounds
    /// 
    /// safety: index is in bounds (0..D)
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

impl<T, const D: usize> InitializableVectorExpr for MathVector<T, D> {
    fn new_filled(_: Self::Builder, val: <Self::Unwrapped as Get>::Item) -> Self where <Self::Unwrapped as Get>::Item: Copy {
        VectorExpr(VectorArray(ManuallyDrop::new([val; D])))
    }
}

unsafe impl<T, const D: usize> UninitVectorExpr for MathVector<T, D> {
    type Uninitialized = MathVector<MaybeUninit<T>, D>;

    fn new_uninit(_: Self::Builder) -> Self::Uninitialized {
        // safe bc the assume_init just moves the MaybeUninit inwards
        let inner: [MaybeUninit<T>; D] = unsafe { MaybeUninit::uninit().assume_init() };
        VectorExpr(VectorArray(ManuallyDrop::new(inner)))
    }

    unsafe fn assume_init(uninit: Self::Uninitialized) -> Self {
        unsafe {
            // FIXME: the copy isn't actually needed, just used to skirt around overly conservative size complaints
            std::mem::transmute_copy::<
                MathVector<MaybeUninit<T>, D>,
                MathVector<T, D>,
            >(&*ManuallyDrop::new(uninit))    
        }
    }

    unsafe fn init_index(uninit: &mut Self::Uninitialized, index: usize, val: <Self::Unwrapped as Get>::Item) {
        uninit.0.0[index] = MaybeUninit::new(val);
    }

    unsafe fn drop_index(uninit: &mut Self::Uninitialized, index: usize) {
        unsafe {
            MaybeUninit::assume_init_drop(uninit.0.0.get_unchecked_mut(index));
        }
    }

    // no actual one-time drops for this
    unsafe fn drop_ots(_: &mut Self::Uninitialized) {}
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
    type ReferencedInner<'a> = &'a [T; D] where Self: 'a;
    type Referenced<'a> = VectorExpr<&'a [T; D], D> where Self: 'a;
    type Copied<'a> = VectorExpr<VecCopy<'a, Self::ReferencedInner<'a>, T>, D>
    where
        Self::Output: 'a + Copy,
        Self: 'a,;
    type ReferencedMutInner<'a> = &'a mut [T; D] where Self: 'a;
    type ReferencedMut<'a> = VectorExpr<&'a mut [T; D], D> where Self: 'a;

    fn borrow<'a>(&'a self) -> Self::Referenced<'a> {
        VectorExpr(&self.0)
    }

    fn borrow_mut<'a>(&'a mut self) -> Self::ReferencedMut<'a> {
        VectorExpr(&mut self.0)
    }

    fn copy<'a>(&'a self) -> Self::Copied<'a> where Self::Output: Copy {
        VectorExpr(VecCopy{ vec: self.unwrap() })
    }
}

// used for VectorInPlaceEval
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
    type ReferencedInner<'a> = &'a [T; D] where Self: 'a;
    type Referenced<'a> = VectorExpr<&'a [T; D], D> where Self: 'a;
    type Copied<'a> = VectorExpr<VecCopy<'a, Self::ReferencedInner<'a>, T>, D>
    where
        Self::Output: Copy,
        Self: 'a,
    ;
    type ReferencedMutInner<'a> =  &'a mut [T; D] where Self: 'a;
    type ReferencedMut<'a> = VectorExpr<&'a mut [T; D], D> where Self: 'a;

    fn borrow<'a>(&'a self) -> Self::Referenced<'a> {
        VectorExpr(&*self.0.vec.0)
    }

    fn borrow_mut<'a>(&'a mut self) -> Self::ReferencedMut<'a> {
        VectorExpr(&mut *self.0.vec.0)
    }

    fn copy<'a>(&'a self) -> Self::Copied<'a> where Self::Output: Copy {
        self.borrow().copied()
    }
}


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

impl<T: AddAssign<<V::Unwrapped as Get>::Item>, V: VectorOps, const D: usize> AddAssign<V> for MathVector<T, D> where 
    V::Unwrapped: HasReuseBuf<BoundHandlesBool = N>,
    V::Unwrapped: HasOutput<OutputBool = N>,
    VectorExprBuilder<D>: VectorBuilderUnion<V::Builder>,
    (N, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
    (N, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
    (N, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
    (N, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
{
    #[inline]
    fn add_assign(&mut self, rhs: V) {
        VectorOps::add_assign(self, rhs).consume()
    }
}

impl<T: SubAssign<<V::Unwrapped as Get>::Item>, V: VectorOps, const D: usize> SubAssign<V> for MathVector<T, D> where 
    V::Unwrapped: HasReuseBuf<BoundHandlesBool = N>,
    V::Unwrapped: HasOutput<OutputBool = N>,
    VectorExprBuilder<D>: VectorBuilderUnion<V::Builder>,
    (N, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
    (N, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
    (N, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
    (N, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
{
    #[inline]
    fn sub_assign(&mut self, rhs: V) {
        VectorOps::sub_assign(self, rhs).consume()
    }
}

impl<T1: num_traits::Zero + AddAssign<T2>, T2, const D: usize> Sum<MathVector<T2, D>> for MathVector<T1, D> {
    #[inline]
    fn sum<I: Iterator<Item = MathVector<T2, D>>>(iter: I) -> Self {
        let mut sum = VectorExprBuilder::new().generate(|| T1::zero())
            .create_array()
            .bind()
            .consume();
        for vec in iter {
            sum += vec;
        }
        sum
    }
}


/// formats this vector as a column vector with stylized unicode brackets
/// 
/// example:
/// ```txt
/// ┌ 11  ┐
/// │ 2   │
/// └ 333 ┘
/// ```
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


/// a **R**untime **S**ized vector wrapper
/// 
/// V: the underlying VectorLike type, generally inferred or generic
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
    /// Retrieves an arbitrary value from a repeatable VectorExpr
    /// 
    /// panics if `index` is out of bounds (outside of 0..D)
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
    /// marks this [`RSMathVector`] to have its buffer reused
    /// 
    /// buffer placed on the first buffer
    #[inline]
    pub fn reuse(self) -> RSVectorExpr<ReplaceSlice<T>> {
        let size = self.size;
        RSVectorExpr {
            vec: ReplaceSlice(ManuallyDrop::new(self.unwrap().0)),
            size,
        }
    }

    /// converts this [`RSMathVector`] to a repeatable [`RSVectorExpr`] w/ Item = `&'a T`
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

    /// references the element at index without checking bounds
    /// 
    /// safety: index is in bounds (0..size)
    #[inline]
    pub unsafe fn get_unchecked<I: SliceIndex<[T]>>(&self, index: I) -> &I::Output {
        unsafe { self.vec.0.get_unchecked(index) }
    }

    /// mutably references the element at index without checking bounds
    /// 
    /// safety: index is in bounds (0..size)
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

impl<T> InitializableVectorExpr for RSMathVector<T> {
    fn new_filled(builder: Self::Builder, val: <Self::Unwrapped as Get>::Item) -> Self where <Self::Unwrapped as Get>::Item: Copy {
        RSVectorExpr{
            vec: VectorSlice(
                unsafe {transmute::<Box<[T]>, Box<ManuallyDrop<[T]>>>(vec![val; builder.size].into_boxed_slice())}
            ),
            size: builder.size,
        }
    }
}

unsafe impl<T> UninitVectorExpr for RSMathVector<T> {
    type Uninitialized = RSMathVector<MaybeUninit<T>>;

    fn new_uninit(builder: Self::Builder) -> Self::Uninitialized {
        RSVectorExpr{
            vec: VectorSlice(
                unsafe {
                    transmute::<
                        Box<[MaybeUninit<T>]>, 
                        Box<ManuallyDrop<[MaybeUninit<T>]>>,
                    >(Box::new_uninit_slice(builder.size))
                }
            ),
            size: builder.size,
        }
    }

    unsafe fn assume_init(uninit: Self::Uninitialized) -> Self {
        unsafe {
            transmute::<
                RSMathVector<MaybeUninit<T>>,
                RSMathVector<T>,
            >(uninit)
        }
    }

    unsafe fn init_index(uninit: &mut Self::Uninitialized, index: usize, val: <Self::Unwrapped as Get>::Item) {
        uninit.vec.0[index] = MaybeUninit::new(val);
    }

    unsafe fn drop_index(uninit: &mut Self::Uninitialized, index: usize) {
        unsafe {
            MaybeUninit::assume_init_drop(uninit.vec.0.get_unchecked_mut(index));
        }
    }

    // no actual one-time drops for this
    unsafe fn drop_ots(_: &mut Self::Uninitialized) {}
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
    type ReferencedInner<'a> = &'a [T] where Self: 'a;
    type Referenced<'a> = RefRSMathVector<'a, T> where Self: 'a;
    type Copied<'a> = RSVectorExpr<VecCopy<'a, &'a [T], T>> where
        Self::Output: Copy,
        Self: 'a,
    ;
    type ReferencedMutInner<'a> = &'a mut [T] where Self: 'a;
    type ReferencedMut<'a> = RefMutRSMathVector<'a, T>where Self: 'a;

    #[inline]
    fn borrow<'a>(&'a self) -> Self::Referenced<'a> {
        let size = self.size;
        RSVectorExpr { vec: &**self, size }
    }

    #[inline]
    fn borrow_mut<'a>(&'a mut self) -> Self::ReferencedMut<'a> {
        let size = self.size;
        RSVectorExpr { vec: &mut **self, size }
    }

    #[inline]
    fn copy<'a>(&'a self) -> Self::Copied<'a> where Self::Output: Copy {
        self.borrow().copied()
    }
}

/// formats this vector as a column vector with stylized unicode brackets
/// 
/// example:
/// ```txt
/// ┌ 11  ┐
/// │ 2   │
/// └ 333 ┘
/// ```
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

/// formats this vector as a column vector with stylized unicode brackets
/// 
/// example:
/// ```txt
/// ┌ 11  ┐
/// │ 2   │
/// └ 333 ┘
/// ```
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
    type ReferencedInner<'a> = &'a [T] where Self: 'a;
    type Referenced<'a> = RefRSMathVector<'a, T> where Self: 'a;
    type Copied<'a> = RSVectorExpr<VecCopy<'a, &'a [T], T>> where
        Self::Output: Copy,
        Self: 'a,
    ;
    type ReferencedMutInner<'a> = &'a mut [T] where Self: 'a;
    type ReferencedMut<'a> = RefMutRSMathVector<'a, T>where Self: 'a;

    fn borrow<'a>(&'a self) -> Self::Referenced<'a> {
        let size = self.size;
        RSVectorExpr{ vec: &**self.vec.vec.0, size }
    }
    fn borrow_mut<'a>(&'a mut self) -> Self::ReferencedMut<'a> {
        let size = self.size;
        RSVectorExpr{ vec: &mut **self.vec.vec.0, size }
    }
    fn copy<'a>(&'a self) -> Self::Copied<'a> where Self::Output: Copy {
        self.borrow().copied()
    }
}

impl<T: MulAssign<S>, S: Copy> MulAssign<S> for RSMathVector<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: S) {
        <&mut Self as VectorOps>::mul_assign(self, rhs).consume();
    }
}

impl<T: DivAssign<S>, S: Copy> DivAssign<S> for RSMathVector<T> {
    #[inline]
    fn div_assign(&mut self, rhs: S) {
        <&mut Self as VectorOps>::div_assign(self, rhs).consume();
    }
}

impl<T: RemAssign<S>, S: Copy> RemAssign<S> for RSMathVector<T> {
    #[inline]
    fn rem_assign(&mut self, rhs: S) {
        <&mut Self as VectorOps>::rem_assign(self, rhs).consume();
    }
}

impl<T: AddAssign<<V::Unwrapped as Get>::Item>, V: VectorOps> AddAssign<V> for RSMathVector<T> where 
    V::Unwrapped: HasReuseBuf<BoundHandlesBool = N>,
    V::Unwrapped: HasOutput<OutputBool = N>,
    RSVectorExprBuilder: VectorBuilderUnion<V::Builder>,
    (N, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
    (N, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
    (N, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
    (N, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
{
    #[inline]
    fn add_assign(&mut self, rhs: V) {
        VectorOps::add_assign(self, rhs).consume()
    }
}

impl<T: SubAssign<<V::Unwrapped as Get>::Item>, V: VectorOps> SubAssign<V> for RSMathVector<T> where 
    V::Unwrapped: HasReuseBuf<BoundHandlesBool = N>,
    V::Unwrapped: HasOutput<OutputBool = N>,
    RSVectorExprBuilder: VectorBuilderUnion<V::Builder>,
    (N, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
    (N, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
    (N, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
    (N, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
{
    #[inline]
    fn sub_assign(&mut self, rhs: V) {
        VectorOps::sub_assign(self, rhs).consume()
    }
}

/// formats this vector as a column vector with stylized unicode brackets
/// 
/// example:
/// ```txt
/// ┌ 11  ┐
/// │ 2   │
/// └ 333 ┘
/// ```
impl<'a, T: std::fmt::Display> std::fmt::Display for RefMutRSMathVector<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.deref().fmt(f)
    }
}


/// a vector wrapper which adds an [inner product](https://en.wikipedia.org/wiki/Inner_product_space) onto another vector wrapper
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
    type ReferencedInner<'a> = V::ReferencedInner<'a> where Self: 'a;
    type Referenced<'a> = VectorInnerProdExpr<V::Referenced<'a>, IP> where Self: 'a;
    type Copied<'a> = VectorInnerProdExpr<V::Copied<'a>, IP> where
        Self::Output: Copy,
        Self: 'a,
    ;
    type ReferencedMutInner<'a> = V::ReferencedMutInner<'a>  where Self: 'a;
    type ReferencedMut<'a> = VectorInnerProdExpr<V::ReferencedMut<'a>, IP> where Self: 'a;

    fn borrow<'a>(&'a self) -> Self::Referenced<'a> {
        VectorInnerProdExpr {
            vec: self.vec.borrow(),
            inner_prod: self.inner_prod
        }
    }

    fn borrow_mut<'a>(&'a mut self) -> Self::ReferencedMut<'a> {
        VectorInnerProdExpr {
            vec: self.vec.borrow_mut(),
            inner_prod: self.inner_prod
        }
    }

    fn copy<'a>(&'a self) -> Self::Copied<'a> where Self::Output: Copy {
        VectorInnerProdExpr {
            vec: self.vec.copy(),
            inner_prod: self.inner_prod
        }
    }
}

impl<V: VectorOps, IP: GenericInnerProduct> VectorInnerProdOps for VectorInnerProdExpr<V, IP> {
    type InnerProd = IP;
}

impl<V: VectorInPlaceEvalOps, IP: GenericInnerProduct> VectorInPlaceEvalOps for VectorInnerProdExpr<V, IP> 
where 
    <<V::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
    (<V::Unwrapped as HasReuseBuf>::BoundHandlesBool, Y): FilterPair,
    (<V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <<V::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg): TyBoolPair,
    (<<V::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): TyBoolPair,
    (<V::ConcreteVectorLike as HasOutput>::OutputBool, <V::UsedVector as HasOutput>::OutputBool): FilterPair,
    (<V::ConcreteVectorLike as HasReuseBuf>::FstOwnedBufferBool, <V::UsedVector as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
    (<V::ConcreteVectorLike as HasReuseBuf>::SndOwnedBufferBool, <V::UsedVector as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
    (<V::ConcreteVectorLike as HasReuseBuf>::FstHandleBool, <V::UsedVector as HasReuseBuf>::FstHandleBool): SelectPair,
    (<V::ConcreteVectorLike as HasReuseBuf>::SndHandleBool, <V::UsedVector as HasReuseBuf>::SndHandleBool): SelectPair,
    (<V::ConcreteVectorLike as HasReuseBuf>::BoundHandlesBool, <V::UsedVector as HasReuseBuf>::BoundHandlesBool): FilterPair,
    <V::Builder as VectorBuilder>::Wrapped<
        VecAttachUsedVec<V::ConcreteVectorLike, V::UsedVector>,
    >: Index<usize>,
{
    type ConcreteVectorLike = V::ConcreteVectorLike;
    type UsedVector = V::UsedVector;

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
        Self: Sized,
    {
        todo!()
    }
}
