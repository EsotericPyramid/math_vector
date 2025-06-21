//! Module containing all to do with Vectors and basic operations to do on them

use crate::{
    trait_specialization_utils::*, 
    util_structs::NoneIter, 
    util_traits::HasOutput,
};
use std::{
    iter::{
        Product,
        Sum,
    },
    mem::{
        MaybeUninit, 
        ManuallyDrop, 
        self,
    }, 
    ops::*, 
    ptr,
};

pub mod vec_util_traits;
pub mod vector_structs;
pub mod vector_builders;

use vec_util_traits::*;
use vector_structs::*;
use vector_builders::*;

/// A const sized vector wrapper
/// T: the underlying VectorLike type, generally inferred or generic
/// D: the size of the vector
#[repr(transparent)]
pub struct VectorExpr<V: VectorLike, const D: usize>(pub(crate) V); // note: VectorExpr only holds fully unused VectorLike objects

impl<V: VectorLike, const D: usize> VectorExpr<V, D> {
    /// converts the underlying VectorLike to a dynamic object
    /// stabilizes the overall type to a consitent one
    /// ex:
    /// ```
    ///     use math_vector::{vector::*};
    /// 
    ///     let mut vec = VectorExpr::from([1; 100]).make_dynamic();
    ///     for _ in 0..10 {
    ///         vec = vec.add(VectorExpr::from([1; 100])).make_dynamic();
    ///     }
    ///     let output = vec.eval();
    /// ```
    #[inline]
    pub fn make_dynamic(self) -> VectorExpr<Box<dyn VectorLike<
        GetBool = V::GetBool,
        Inputs = (),
        Item = V::Item,
        BoundItems = V::BoundItems,

        OutputBool = V::OutputBool,
        Output = V::Output,

        FstHandleBool = V::FstHandleBool,
        SndHandleBool = V::SndHandleBool,
        BoundHandlesBool = V::BoundHandlesBool,
        FstOwnedBufferBool = V::FstOwnedBufferBool,
        SndOwnedBufferBool = V::SndOwnedBufferBool,
        FstOwnedBuffer = V::FstOwnedBuffer,
        SndOwnedBuffer = V::SndOwnedBuffer,
        FstType = V::FstType,
        SndType = V::SndType,
        BoundTypes = V::BoundTypes,
    >>, D> where V: 'static {
        VectorExpr(Box::new(DynamicVectorLike{vec: self.unwrap(), inputs: None}) as Box<dyn VectorLike<
            GetBool = V::GetBool,
            Inputs = (),
            Item = V::Item,
            BoundItems = V::BoundItems,

            OutputBool = V::OutputBool,
            Output = V::Output,

            FstHandleBool = V::FstHandleBool,
            SndHandleBool = V::SndHandleBool,
            BoundHandlesBool = V::BoundHandlesBool,
            FstOwnedBufferBool = V::FstOwnedBufferBool,
            SndOwnedBufferBool = V::SndOwnedBufferBool,
            FstOwnedBuffer = V::FstOwnedBuffer,
            SndOwnedBuffer = V::SndOwnedBuffer,
            FstType = V::FstType,
            SndType = V::SndType,
            BoundTypes = V::BoundTypes,
        >>)
    }

    /// consumes the VectorExpr and returns the built up output
    /// 
    /// Note:
    /// methods like sum, product, or fold can place build up outputs
    /// 
    /// output is generally nested 2 element tuples
    /// newer values to the right
    /// binary operators merge the output of the 2 vectors
    #[inline] 
    pub fn consume(self) -> V::Output where V: HasReuseBuf<BoundTypes = V::BoundItems> {
        VectorIter::<V>{
            vec: unsafe { ptr::read(&ManuallyDrop::new(self).0) },
            live_input_start: 0,
            dead_output_start: 0,
            size: D
        }.consume()
    }

    /// evaluates the VectorExpr and returns the resulting vector alongside its output (if present)
    /// if the VectorExpr has no item (& thus results in a vector w/ ZST elements) or the item is irrelevent, see consume to not return that vector
    /// will try to use the first buffer if available (fails if the provided buffer is not bindable to the output)
    /// 
    /// Warning: 
    /// this method trying the evaluate the vector *onto the stack*, it is very possible to overflow the stack with larger vectors
    /// use heap_eval if this is a concern 
    /// 
    /// Note:
    /// methods like sum, product, or fold can place build up outputs
    /// 
    /// output is generally nested 2 element tuples
    /// newer values to the right
    /// binary operators merge the output of the 2 vectors
    #[inline]
    pub fn eval(self) -> <VecBind<VecMaybeCreateArray<V, V::Item, D>> as HasOutput>::Output 
    where 
        <V::FstHandleBool as TyBool>::Neg: Filter,
        (V::FstHandleBool, <V::FstHandleBool as TyBool>::Neg): SelectPair,
        (V::FstOwnedBufferBool, <V::FstHandleBool as TyBool>::Neg): TyBoolPair,
        (V::OutputBool, <(V::FstOwnedBufferBool, <V::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or): FilterPair,
        (V::BoundHandlesBool, Y): FilterPair,
        VecBind<VecMaybeCreateArray<V, V::Item, D>>: HasReuseBuf<BoundTypes = <VecBind<VecMaybeCreateArray<V, V::Item, D>> as Get>::BoundItems>
    {
        self.maybe_create_array().bind().consume()
    }


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
        (V::OutputBool, <(V::FstOwnedBufferBool, <V::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or): FilterPair,
        (V::BoundHandlesBool, Y): FilterPair,
        VecBind<VecMaybeCreateHeapArray<V, V::Item, D>>: HasReuseBuf<BoundTypes = <VecBind<VecMaybeCreateHeapArray<V, V::Item, D>> as Get>::BoundItems>
    {
        self.maybe_create_heap_array().bind().consume()
    }
}

impl<V: VectorLike + IsRepeatable, const D: usize> VectorExpr<V, D> {
    /// Retrieves an arbitrary value from a repeatable VectorExpr
    pub fn get(&mut self, index: usize) -> V::Item {
        // the nature of IsRepeatable means that any index can be called any number of times so this is fine
        if index >= D {panic!("math_vector Error: index access out of bound")}
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

impl<V: VectorLike, const D: usize> IntoIterator for VectorExpr<V, D> where V: HasReuseBuf<BoundTypes = V::BoundItems> {
    type IntoIter = VectorIter<V>;
    type Item = <V as Get>::Item;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        VectorIter{
            vec: unsafe { ptr::read(&ManuallyDrop::new(self).0) },
            live_input_start: 0,
            dead_output_start: 0,
            size: D
        }
    }
}


/// a simple type alias for a VectorExpr created from an array of type [T; D]
pub type MathVector<T, const D: usize> = VectorExpr<OwnedArray<T, D>, D>;

impl<T, const D: usize> MathVector<T, D> {
    /// marks this MathVector to have its buffer reused
    /// buffer placed on fst buffer
    #[inline] pub fn reuse(self) -> VectorExpr<ReplaceArray<T, D>, D> {VectorExpr(ReplaceArray(self.unwrap().0))}
    /// marks this MathVector to have its buffer reused while keeping it on the heap
    /// buffer placed on fst buffer
    #[inline] pub fn heap_reuse(self: Box<Self>) -> VectorExpr<Box<ReplaceArray<T, D>>, D> {
        // Safety, series of equivilent types:
        // Box<MathVector<T,D>>
        // Box<VectorExpr<OwnedArray<T, D>, D>>, de-alias MathVector
        // Box<ManuallyDrop<[T; D]>>, VectorExpr & OwnedArray are transparent
        // VectorExpr<Box<ReplaceArray<T, D>>, D>, VectorExpr & ReplaceArray are transparent
        unsafe { mem::transmute::<Box<Self>, VectorExpr<Box<ReplaceArray<T, D>>, D>>(self) }
    }

    /// converts this MathVector to a repeatable VectorExpr w/ Item = &'a T
    #[inline] pub fn referred<'a>(self) -> VectorExpr<ReferringOwnedArray<'a, T, D>, D> where T: 'a {
        VectorExpr(ReferringOwnedArray(unsafe {mem::transmute_copy::<ManuallyDrop<[T; D]>, [T; D]>(&self.unwrap().0)}, std::marker::PhantomData)) //FIXME: unecessary transmute copy to get the compiler to not complain
    }

    /// references the element at index without checking bounds
    /// safety: index is in bounds 
    #[inline] pub unsafe fn get_unchecked<I: std::slice::SliceIndex<[T]>>(&self, index: I) -> &I::Output { unsafe {
        self.0.0.get_unchecked(index)
    }}

    /// mutably references the element at index without checking bounds
    /// safety: index is in bounds
    #[inline] pub unsafe fn get_unchecked_mut<I: std::slice::SliceIndex<[T]>>(&mut self, index: I) -> &mut I::Output { unsafe {
        self.0.0.get_unchecked_mut(index)
    }}
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
        VectorExpr(OwnedArray(ManuallyDrop::new(value)))
    }
}

impl<T, const D: usize> Into<[T; D]> for MathVector<T, D> {
    #[inline] 
    fn into(self) -> [T; D] {
        self.unwrap().unwrap()
    }
}

impl<'a, T, const D: usize> From<&'a [T; D]> for &'a MathVector<T, D> {
    #[inline]
    fn from(value: &'a [T; D]) -> Self {
        unsafe { mem::transmute::<&'a [T; D], &'a MathVector<T, D>>(value) }
    }
} 

impl<'a, T, const D: usize> Into<&'a [T; D]> for &'a MathVector<T, D> {
    #[inline]
    fn into(self) -> &'a [T; D] {
        unsafe { mem::transmute::<&'a MathVector<T, D>, &'a [T; D]>(self) }
    }
} 

impl<'a, T, const D: usize> From<&'a mut [T; D]> for &'a mut MathVector<T, D> {
    #[inline]
    fn from(value: &'a mut [T; D]) -> Self {
        unsafe { mem::transmute::<&'a mut [T; D], &'a mut MathVector<T, D>>(value) }
    }
} 

impl<'a, T, const D: usize> Into<&'a mut [T; D]> for &'a mut MathVector<T, D> {
    #[inline]
    fn into(self) -> &'a mut [T; D] {
        unsafe { mem::transmute::<&'a mut MathVector<T, D>, &'a mut [T; D]>(self) }
    }
} 

impl<T, const D: usize> From<Box<[T; D]>> for Box<MathVector<T, D>> {
    #[inline]
    fn from(value: Box<[T; D]>) -> Self {
        unsafe { mem::transmute::<Box<[T; D]>, Box<MathVector<T,D>>>(value) }
    }
}

impl<T, const D: usize> Into<Box<[T; D]>> for Box<MathVector<T,D>> {
    #[inline] 
    fn into(self) -> Box<[T; D]> {
        unsafe { mem::transmute::<Box<MathVector<T,D>>, Box<[T; D]>>(self) }
    }
} 

impl<T, I, const D: usize> Index<I> for MathVector<T, D> where [T; D]: Index<I> {
    type Output = <[T; D] as Index<I>>::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        &self.0.0[index]
    }
}

impl<T, I, const D: usize> IndexMut<I> for MathVector<T, D> where [T; D]: IndexMut<I> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0.0[index]
    }
}


//impl<T1: AddAssign<<V2::Unwrapped as Get>::Item>, V2: VectorOps, const D: usize> AddAssign<V2> for MathVector<T1, D>
//where
//    <Self as VectorOps>::Builder: VectorBuilderUnion<V2::Builder>,
//    (N, <V2::Unwrapped as HasOutput>::OutputBool): FilterPair,
//    (<(N, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
//    (N, <V2::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
//    (N, <V2::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
//    (N, <V2::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
//    (N, <V2::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
//    (N, <V2::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
//    V2::Unwrapped: HasReuseBuf<BoundTypes = <V2::Unwrapped as Get>::BoundItems>
//{
//    #[inline]
//    fn add_assign(&mut self, rhs: V2) {
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


impl<T1: Sum<T2> + AddAssign<T2>, T2, const D: usize> Sum<MathVector<T2, D>> for MathVector<T1, D> {
    #[inline]
    fn sum<I: Iterator<Item = MathVector<T2, D>>>(iter: I) -> Self {
        let mut sum = vector_gen(|| Sum::sum(NoneIter::new())).create_array().bind().consume();
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
pub fn vector_index_gen<F: FnMut(usize) -> O, O, const D: usize>(f: F) -> VectorExpr<VecIndexGenerator<F, O>, D> {
    VectorExpr(VecIndexGenerator(f))
}

// TODO: finish implementing RSVectorExpr
pub struct RSVectorExpr<V: VectorLike>{pub(crate) vec: V, pub(crate) size: usize}

impl<V: VectorLike> RSVectorExpr<V> {
    /// converts this runtime sized vector into a const sized one
    /// 
    /// panics if this vector does not have size D
    #[inline]
    pub fn const_sized<const D: usize>(self) -> VectorExpr<V, D> {
        if self.size != D {panic!("math_vector error: cannot convert a RS vector with size {} into a const sized vector with size {}", self.size, D)}
        unsafe {mem::transmute_copy::<V, VectorExpr<V, D>>(&ManuallyDrop::new(self).vec)}
    }

    /// consumes the VectorExpr and returns the built up output
    /// 
    /// Note:
    /// methods like sum, product, or fold can place build up outputs
    /// 
    /// output is generally nested 2 element tuples
    /// newer values to the right
    /// binary operators merge the output of the 2 vectors
    #[inline]
    pub fn consume(self) -> V::Output where V: HasReuseBuf<BoundTypes = V::BoundItems> {
        self.into_iter().consume()
    }
}

impl<V: VectorLike + IsRepeatable> RSVectorExpr<V> {
    #[inline]
    pub fn get(&mut self, index: usize) -> V::Item {
        if index >= self.size {panic!("math_vector Error: index access out of bounds")}
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
        }
    }
}

impl<V: VectorLike> IntoIterator for RSVectorExpr<V> where V: HasReuseBuf<BoundTypes = V::BoundItems> {
    type IntoIter = VectorIter<V>;
    type Item = <V as Get>::Item;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        let size = self.size;
        VectorIter {
            vec: unsafe { ptr::read(&ManuallyDrop::new(self).vec) },
            live_input_start: 0,
            dead_output_start: 0,
            size
        }
    }
}


pub type RSMathVector<T> = RSVectorExpr<OwnedSlice<T>>;

impl<T> RSMathVector<T> {
    #[inline] pub fn reuse(self) -> RSVectorExpr<ReplaceSlice<T>> {
        todo!()
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

impl<T> From<Box<[T]>> for RSMathVector<T> {
    #[inline]
    fn from(value: Box<[T]>) -> Self {
        let size = value.len();
        unsafe { RSVectorExpr{vec: mem::transmute::<Box<[T]>, OwnedSlice<T>>(value), size} }
    }
}

impl<T> Into<Box<[T]>> for RSMathVector<T> {
    #[inline]
    fn into(self) -> Box<[T]> {
        unsafe {  mem::transmute_copy::<OwnedSlice<T>, Box<[T]>>(&ManuallyDrop::new(self).vec) }
    }
}

impl<T, I> Index<I> for RSMathVector<T> where [T]: Index<I> {
    type Output = <[T] as Index<I>>::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        &self.vec.0[index]
    }
}

impl<T, I> IndexMut<I> for RSMathVector<T> where [T]: IndexMut<I> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.vec.0[index]
    }
}


/// a VectorExpr iterator
pub struct VectorIter<V: VectorLike>{vec: V, live_input_start: usize, dead_output_start: usize, size: usize} // note: ranges are start inclusive, end exclusive

impl<V: VectorLike> VectorIter<V> {
    /// retrieves the next item without checking
    /// Safety: there must be another item to return
    #[inline]
    pub unsafe fn next_unchecked(&mut self) -> V::Item where V: HasReuseBuf<BoundTypes = V::BoundItems> { unsafe {
        let index = self.live_input_start;
        self.live_input_start += 1;
        let inputs = self.vec.get_inputs(index);
        let (item, bound_items) = self.vec.process(index, inputs);
        self.vec.assign_bound_bufs(index, bound_items);
        self.dead_output_start += 1;
        item
    }}

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
        assert!(self.live_input_start == self.size, "math_vector error: A VectorIter must be fully used before outputting");
        debug_assert!(self.dead_output_start == self.size, "math_vector internal error: A VectorIter's output buffers (somehow) weren't fully filled despite the inputs being fully used, likely an internal issue");
        unsafe {self.unchecked_output()}
    }

    /// fully consumes the VectorIter and then returns its output
    #[inline]
    pub fn consume(mut self) -> V::Output where V: HasReuseBuf<BoundTypes = V::BoundItems> {
        self.no_output_consume();
        unsafe {self.unchecked_output()} // safety: VectorIter was fully used
    }

    /// fully consumes the VectorIter without returning its output
    #[inline]
    pub fn no_output_consume(&mut self) where V: HasReuseBuf<BoundTypes = V::BoundItems> {
        while self.live_input_start < self.size {
            unsafe { let _ = self.next_unchecked(); }
        }
    }
}

impl<V: VectorLike> Drop for VectorIter<V> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            for i in 0..self.dead_output_start { //up to the start of the dead area in output
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

impl<V: VectorLike> Iterator for VectorIter<V> where V: HasReuseBuf<BoundTypes = V::BoundItems> {
    type Item = V::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.live_input_start < self.size { // != instead of < as it is known that start is always <= end so their equivilent
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

impl<V: VectorLike> ExactSizeIterator for VectorIter<V> where V: HasReuseBuf<BoundTypes = V::BoundItems> {}

impl<V: VectorLike> std::iter::FusedIterator for VectorIter<V> where V: HasReuseBuf<BoundTypes = V::BoundItems> {}


/// a trait with various vector operations
pub unsafe trait VectorOps {
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

    /// binds the vector's item to its fst buffer, adding the buffer to Output if owned by the vector
    #[inline] 
    fn bind(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecBind<Self::Unwrapped>> where 
        Self::Unwrapped:  VectorLike<FstHandleBool = Y>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, Y): FilterPair,
        VecBind<Self::Unwrapped>: HasReuseBuf<BoundTypes = <VecBind<Self::Unwrapped> as Get>::BoundItems>,
        Self: Sized
    {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecBind{vec: self.unwrap()}) }
    }

    /// maps the vector w/ the provided closure taking the vector's item and outputing the new item and a value to bind
    /// binds that value to the fst buffer, adding the buffer to Output if owned by the vector
    #[inline]
    fn map_bind<F: FnMut(<Self::Unwrapped as Get>::Item) -> (I, B), I, B>(self, f: F) -> <Self::Builder as VectorBuilder>::Wrapped<VecMapBind<Self::Unwrapped, F, I, B>> where 
        Self::Unwrapped:  VectorLike<FstHandleBool = Y>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, Y): FilterPair,
        VecMapBind<Self::Unwrapped, F, I, B>: HasReuseBuf<BoundTypes = <VecMapBind<Self::Unwrapped, F, I, B> as Get>::BoundItems>,
        Self: Sized
    {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecMapBind{vec: self.unwrap(), f})}
    }

    /// binds the vector's item to its fst buffer, adding the buffer to an internal output if owned by the vector
    /// 
    /// Note: 
    /// this internal output is not readily accessible and doesn't add much over bind
    /// As such, end users should generally just use bind
    #[inline] 
    fn half_bind(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecHalfBind<Self::Unwrapped>> where
        Self::Unwrapped:  VectorLike<FstHandleBool = Y>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, Y): FilterPair,
        VecHalfBind<Self::Unwrapped>: HasReuseBuf<BoundTypes = <VecBind<Self::Unwrapped> as Get>::BoundItems>,
        Self: Sized
    {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecHalfBind{vec: self.unwrap()}) }
    }

    /// swaps the vector's first and second buffers
    #[inline] fn buf_swap(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecBufSwap<Self::Unwrapped>> where Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecBufSwap{vec: self.unwrap()}) }
    }

    /// offsets (with rolling over) each element of the vector up by the given offset
    #[inline] 
    fn offset_up(self, offset: usize) -> <Self::Builder as VectorBuilder>::Wrapped<VecOffset<Self::Unwrapped>> where Self: Sized {
        let size = self.size();
        let builder = self.get_builder();
        unsafe { builder.wrap(VecOffset{vec: self.unwrap(), offset: offset % size, size}) }
    }

    /// offsets (with rolling over) each element of the vector down by the given offset
    #[inline] 
    fn offset_down(self, offset: usize) -> <Self::Builder as VectorBuilder>::Wrapped<VecOffset<Self::Unwrapped>> where Self: Sized {
        let size = self.size();
        let builder = self.get_builder();
        unsafe { builder.wrap(VecOffset{vec: self.unwrap(), offset: size - (offset % size), size}) }
    }

    /// reverses the vector
    #[inline]
    fn reverse(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecReverse<Self::Unwrapped>> where Self: Sized {
        let max_index = self.size() -1;
        let builder = self.get_builder();
        unsafe { builder.wrap(VecReverse{vec: self.unwrap(), max_index})}
    }

    /// maps the vector's items with the provided closure
    #[inline] fn map<F: FnMut(<Self::Unwrapped as Get>::Item) -> O, O>(self, f: F) -> <Self::Builder as VectorBuilder>::Wrapped<VecMap<Self::Unwrapped, F, O>> where (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecMap{vec: self.unwrap(), f}) }
    } 

    /// folds the vector's items into a single value using the provided closure
    /// 
    /// note: fold_ref should be used whenever possible due to implementation
    #[inline] fn fold<F: FnMut(O, <Self::Unwrapped as Get>::Item) -> O, O>(self, f: F, init: O) -> <Self::Builder as VectorBuilder>::Wrapped<VecFold<Self::Unwrapped, F, O>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecFold{vec: self.unwrap(), f, cell: Some(init)}) }
    }
    
    /// folds the vector's items into a single value using the provided closure while preserving the items
    /// 
    /// note: fold_ref should be used whenever possible due to implementation
    #[inline] fn copied_fold<F: FnMut(O, <Self::Unwrapped as Get>::Item) -> O, O>(self, f: F, init: O) -> <Self::Builder as VectorBuilder>::Wrapped<VecCopiedFold<Self::Unwrapped, F, O>> where <Self::Unwrapped as Get>::Item: Copy, (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecCopiedFold{vec: self.unwrap(), f, cell: Some(init)}) }
    }
    
    /// folds the vector's items into a single value using the provided closure
    #[inline] fn fold_ref<F: FnMut(&mut O, <Self::Unwrapped as Get>::Item), O>(self, f: F, init: O) -> <Self::Builder as VectorBuilder>::Wrapped<VecFoldRef<Self::Unwrapped, F, O>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecFoldRef{vec: self.unwrap(), f, cell: ManuallyDrop::new(init)}) }
    }
    
    /// folds the vector's items into a single value using the provided closure while preserving the items
    #[inline] fn copied_fold_ref<F: FnMut(&mut O, <Self::Unwrapped as Get>::Item), O>(self, f: F, init: O) -> <Self::Builder as VectorBuilder>::Wrapped<VecCopiedFoldRef<Self::Unwrapped, F, O>> where <Self::Unwrapped as Get>::Item: Copy, (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecCopiedFoldRef{vec: self.unwrap(), f, cell: ManuallyDrop::new(init)}) }
    }
    
    /// copies each of the vector's items, useful for turning &T -> T
    #[inline] fn copied<'a, I: Copy>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecCopy<'a, Self::Unwrapped, I>> where (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self::Unwrapped: Get<Item = &'a I>, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecCopy{vec: self.unwrap()}) }
    }

    /// clones each of the vector's items, useful for turning &T -> T
    #[inline] fn cloned<'a, I: Clone>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecClone<'a, Self::Unwrapped, I>> where (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self::Unwrapped: Get<Item = &'a I>, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecClone{vec: self.unwrap()}) }
    }

    /// negates each of the vector's items
    #[inline] fn neg(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecNeg<Self::Unwrapped>> where (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, <Self::Unwrapped as Get>::Item: Neg, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecNeg{vec: self.unwrap()}) }
    }

    /// multiples a scalar with the vector (vector items are rhs) (*may* be identitical to mul_l)
    #[inline] fn mul_r<S: Mul<<Self::Unwrapped as Get>::Item> + Copy>(self, scalar: S) -> <Self::Builder as VectorBuilder>::Wrapped<VecMulR<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecMulR{vec: self.unwrap(), scalar}) }
    }

    /// divides a scalar with the vector (vector items are rhs)
    #[inline] fn div_r<S: Div<<Self::Unwrapped as Get>::Item> + Copy>(self, scalar: S) -> <Self::Builder as VectorBuilder>::Wrapped<VecDivR<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecDivR{vec: self.unwrap(), scalar}) }
    }

    /// gets the remainder (ie. %) of a scalar with the vector (vector items are rhs)
    #[inline] fn rem_r<S: Rem<<Self::Unwrapped as Get>::Item> + Copy>(self, scalar: S) -> <Self::Builder as VectorBuilder>::Wrapped<VecRemR<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecRemR{vec: self.unwrap(), scalar}) }
    }

    /// multiplies the vector with a scalar (vector items are lhs) (*may* be identitcal to mul_r)
    #[inline] fn mul_l<S: Copy>(self, scalar: S) -> <Self::Builder as VectorBuilder>::Wrapped<VecMulL<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, <Self::Unwrapped as Get>::Item: Mul<S>, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecMulL{vec: self.unwrap(), scalar})}
    }

    /// divides the vector with a scalar (vector items are lhs)
    #[inline] fn div_l<S: Copy>(self, scalar: S) -> <Self::Builder as VectorBuilder>::Wrapped<VecDivL<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, <Self::Unwrapped as Get>::Item: Div<S>, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecDivL{vec: self.unwrap(), scalar})}
    }

    /// gets the remainder (ie. %) of the vector with a scalar (vector items are lhs)
    #[inline] fn rem_l<S: Copy>(self, scalar: S) -> <Self::Builder as VectorBuilder>::Wrapped<VecRemL<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, <Self::Unwrapped as Get>::Item: Rem<S>, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecRemL{vec: self.unwrap(), scalar})}
    }

    /// mul assigns (*=) the vector's items (&mut T) with a scalar
    #[inline] fn mul_assign<'a, I: 'a + MulAssign<S>, S: Copy>(self, scalar: S) -> <Self::Builder as VectorBuilder>::Wrapped<VecMulAssign<'a, Self::Unwrapped, I, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self::Unwrapped: Get<Item = &'a mut I>, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecMulAssign{vec: self.unwrap(), scalar})}
    }
    
    /// div assigns (/=) the vector's items (&mut T) with a scalar
    #[inline] fn div_assign<'a, I: 'a + DivAssign<S>, S: Copy>(self, scalar: S) -> <Self::Builder as VectorBuilder>::Wrapped<VecDivAssign<'a, Self::Unwrapped, I, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self::Unwrapped: Get<Item = &'a mut I>, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecDivAssign{vec: self.unwrap(), scalar})}
    }

    /// rem assigns (%=) the vector's items (&mut T) with a scalar
    #[inline] fn rem_assign<'a, I: 'a + RemAssign<S>, S: Copy>(self, scalar: S) -> <Self::Builder as VectorBuilder>::Wrapped<VecRemAssign<'a, Self::Unwrapped, I, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self::Unwrapped: Get<Item = &'a mut I>, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecRemAssign{vec: self.unwrap(), scalar})}
    }

    /// calculates the sum of the vector's elements and adds it to the output
    #[inline] fn sum<S: Sum<<Self::Unwrapped as Get>::Item> + AddAssign<<Self::Unwrapped as Get>::Item>>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecSum<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecSum{vec: self.unwrap(), scalar: ManuallyDrop::new(NoneIter::new().sum())})}
    }

    /// calculates the sum of the vector's elements, initialized at given value (not necessarily 0), and adds it to the output
    #[inline] fn initialized_sum<S: AddAssign<<Self::Unwrapped as Get>::Item>>(self, init: S) -> <Self::Builder as VectorBuilder>::Wrapped<VecSum<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecSum{vec: self.unwrap(), scalar: ManuallyDrop::new(init)})}
    }
    
    /// calculates the sum of the vector's elements, adding it to the output, while maintaining the vector's items
    #[inline] fn copied_sum<S: Sum<<Self::Unwrapped as Get>::Item> + AddAssign<<Self::Unwrapped as Get>::Item>>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecCopiedSum<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, <Self::Unwrapped as Get>::Item: Copy, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecCopiedSum{vec: self.unwrap(), scalar: ManuallyDrop::new(NoneIter::new().sum())})}
    }

    /// calculates the sum of the vector's elements, initialized at given value (not necessarily 0), adding it to the output while maintaing the vector's items
    #[inline] fn initialized_copied_sum<S: AddAssign<<Self::Unwrapped as Get>::Item>>(self, init: S) -> <Self::Builder as VectorBuilder>::Wrapped<VecCopiedSum<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, <Self::Unwrapped as Get>::Item: Copy, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecCopiedSum{vec: self.unwrap(), scalar: ManuallyDrop::new(init)})}
    }
    
    /// calculates the product of the vector's elements and adds it to the output
    #[inline] fn product<S: Product<<Self::Unwrapped as Get>::Item> + MulAssign<<Self::Unwrapped as Get>::Item>>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecProduct<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecProduct{vec: self.unwrap(), scalar: ManuallyDrop::new(NoneIter::new().product())})}
    }

    /// calculates the product of the vector's elements, initialized at given value (not necessarily 1), and adds it to the output
    #[inline] fn initialized_product<S: MulAssign<<Self::Unwrapped as Get>::Item>>(self, init: S) -> <Self::Builder as VectorBuilder>::Wrapped<VecProduct<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecProduct{vec: self.unwrap(), scalar: ManuallyDrop::new(init)})}
    }
    
    /// calculates the product of the vector's elements, adding it to the output, while maintaining the vector's items
    #[inline] fn copied_product<S: Product<<Self::Unwrapped as Get>::Item> + MulAssign<<Self::Unwrapped as Get>::Item>>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecCopiedProduct<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, <Self::Unwrapped as Get>::Item: Copy, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecCopiedProduct{vec: self.unwrap(), scalar: ManuallyDrop::new(NoneIter::new().product())})}
    }
    
    /// calculates the products of the vector's elements, initialized at given value (not necessarily 0), adding it to the output while maintaing the vector's items
    #[inline] fn initialized_copied_product<S: MulAssign<<Self::Unwrapped as Get>::Item>>(self, init: S) -> <Self::Builder as VectorBuilder>::Wrapped<VecCopiedProduct<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, <Self::Unwrapped as Get>::Item: Copy, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecCopiedProduct{vec: self.unwrap(), scalar: ManuallyDrop::new(init)})}
    }
        
    /// calculates the square of the vector's magnitude (ie. sum of each element's square) and adds it to the output
    #[inline] fn sqr_mag<S: Sum<<Self::Unwrapped as Get>::Item> + AddAssign<<<Self::Unwrapped as Get>::Item as Mul>::Output>>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecSqrMag<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, <Self::Unwrapped as Get>::Item: Copy + Mul, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecSqrMag{vec: self.unwrap(), scalar: ManuallyDrop::new(NoneIter::new().sum())})}
    }    

    /// calculates the square of the vector's magnitude (ie. sum of each element's square), adding it to the output, while maintaining the vector's items
    #[inline] fn copied_sqr_mag<S: Sum<<Self::Unwrapped as Get>::Item> + AddAssign<<<Self::Unwrapped as Get>::Item as Mul>::Output>>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecCopiedSqrMag<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, <Self::Unwrapped as Get>::Item: Copy + Mul, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecCopiedSqrMag{vec: self.unwrap(), scalar: ManuallyDrop::new(NoneIter::new().sum())})}
    }

    /// zips together the items of 2 vectors into 2 element tuples
    #[inline] fn zip<V: VectorOps>(self, other: V) -> <<Self::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<VecZip<Self::Unwrapped, V::Unwrapped>> 
    where
        Self::Builder: VectorBuilderUnion<V::Builder>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndHandleBool, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        Self: Sized,
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap(VecZip { l_vec: self.unwrap(), r_vec: other.unwrap() }) }
    }

    /// adds 2 vectors
    #[inline] fn add<V: VectorOps>(self, other: V) -> <<Self::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<VecAdd<Self::Unwrapped, V::Unwrapped>> 
    where
        Self::Builder: VectorBuilderUnion<V::Builder>,
        <Self::Unwrapped as Get>::Item: Add<<V::Unwrapped as Get>::Item>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndHandleBool, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        Self: Sized
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap(VecAdd { l_vec: self.unwrap(), r_vec: other.unwrap() }) }
    }

    /// substracts the other vector from self
    #[inline] fn sub<V: VectorOps>(self, other: V) -> <<Self::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<VecSub<Self::Unwrapped, V::Unwrapped>> 
    where
        Self::Builder: VectorBuilderUnion<V::Builder>,
        <Self::Unwrapped as Get>::Item: Sub<<V::Unwrapped as Get>::Item>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndHandleBool, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        Self: Sized
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap(VecSub { l_vec: self.unwrap(), r_vec: other.unwrap() }) }
    }

    /// component-wise multiplies 2 vectors
    #[inline] fn comp_mul<V: VectorOps>(self, other: V) -> <<Self::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<VecCompMul<Self::Unwrapped, V::Unwrapped>> 
    where
        Self::Builder: VectorBuilderUnion<V::Builder>,
        <Self::Unwrapped as Get>::Item: Mul<<V::Unwrapped as Get>::Item>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndHandleBool, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        Self: Sized
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap(VecCompMul { l_vec: self.unwrap(), r_vec: other.unwrap() }) }
    }

    /// component-wise divides self by other
    #[inline] fn comp_div<V: VectorOps>(self, other: V) -> <<Self::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<VecCompDiv<Self::Unwrapped, V::Unwrapped>> 
    where
        Self::Builder: VectorBuilderUnion<V::Builder>,
        <Self::Unwrapped as Get>::Item: Div<<V::Unwrapped as Get>::Item>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndHandleBool, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        Self: Sized
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap(VecCompDiv { l_vec: self.unwrap(), r_vec: other.unwrap() }) }
    }

    /// component-wise get remainder (%) of self by other
    #[inline] fn comp_rem<V: VectorOps>(self, other: V) -> <<Self::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<VecCompRem<Self::Unwrapped, V::Unwrapped>> 
    where
        Self::Builder: VectorBuilderUnion<V::Builder>,
        <Self::Unwrapped as Get>::Item: Rem<<V::Unwrapped as Get>::Item>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndHandleBool, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        Self: Sized
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap(VecCompRem { l_vec: self.unwrap(), r_vec: other.unwrap() }) }
    }

    /// add assigns (+=) the self's items (&mut T) with other
    #[inline] fn add_assign<'a, V: VectorOps, I: 'a + AddAssign<<V::Unwrapped as Get>::Item>>(self, other: V) -> <<Self::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<VecAddAssign<'a, Self::Unwrapped, V::Unwrapped, I>> 
    where
        Self::Builder: VectorBuilderUnion<V::Builder>,
        Self::Unwrapped: Get<Item = &'a mut I>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndHandleBool, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        Self: Sized
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap(VecAddAssign { l_vec: self.unwrap(), r_vec: other.unwrap() }) }
    }

    /// sub assigns (-=) the self's items (&mut T) with other
    #[inline] fn sub_assign<'a, V: VectorOps, I: 'a + SubAssign<<V::Unwrapped as Get>::Item>>(self, other: V) -> <<Self::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<VecSubAssign<'a, Self::Unwrapped, V::Unwrapped, I>> 
    where
        Self::Builder: VectorBuilderUnion<V::Builder>,
        Self::Unwrapped: Get<Item = &'a mut I>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndHandleBool, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        Self: Sized
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap(VecSubAssign { l_vec: self.unwrap(), r_vec: other.unwrap() }) }
    }

    /// mul assigns (*=) the self's items (&mut T) with other
    #[inline] fn comp_mul_assign<'a, V: VectorOps, I: 'a + MulAssign<<V::Unwrapped as Get>::Item>>(self, other: V) -> <<Self::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<VecCompMulAssign<'a, Self::Unwrapped, V::Unwrapped, I>> 
    where
        Self::Builder: VectorBuilderUnion<V::Builder>,
        Self::Unwrapped: Get<Item = &'a mut I>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndHandleBool, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        Self: Sized
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap(VecCompMulAssign { l_vec: self.unwrap(), r_vec: other.unwrap() }) }
    }

    /// div assigns (/=) the self's items (&mut T) with other
    #[inline] fn comp_div_assign<'a, V: VectorOps, I: 'a + DivAssign<<V::Unwrapped as Get>::Item>>(self, other: V) -> <<Self::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<VecCompDivAssign<'a, Self::Unwrapped, V::Unwrapped, I>> 
    where
        Self::Builder: VectorBuilderUnion<V::Builder>,
        Self::Unwrapped: Get<Item = &'a mut I>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndHandleBool, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        Self: Sized
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap(VecCompDivAssign { l_vec: self.unwrap(), r_vec: other.unwrap() }) }
    }

    /// rem assigns (%=) the self's items (&mut T) with other
    #[inline] fn comp_rem_assign<'a, V: VectorOps, I: 'a + RemAssign<<V::Unwrapped as Get>::Item>>(self, other: V) -> <<Self::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<VecCompRemAssign<'a, Self::Unwrapped, V::Unwrapped, I>> 
    where
        Self::Builder: VectorBuilderUnion<V::Builder>,
        Self::Unwrapped: Get<Item = &'a mut I>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndHandleBool, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        Self: Sized
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap(VecCompRemAssign { l_vec: self.unwrap(), r_vec: other.unwrap() }) }
    }

    /// calculates the dot product of 2 vectors and adds it to the output
    #[inline] fn dot<V: VectorOps, S: Sum<<<Self::Unwrapped as Get>::Item as Mul<<V::Unwrapped as Get>::Item>>::Output> + AddAssign<<<Self::Unwrapped as Get>::Item as Mul<<V::Unwrapped as Get>::Item>>::Output>>(self, other: V) -> <<Self::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<VecDot<Self::Unwrapped, V::Unwrapped, S>> 
    where
        Self::Builder: VectorBuilderUnion<V::Builder>,
        <Self::Unwrapped as Get>::Item: Mul<<V::Unwrapped as Get>::Item>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndHandleBool, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        Self: Sized
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap(VecDot { l_vec: self.unwrap(), r_vec: other.unwrap(), scalar: ManuallyDrop::new(NoneIter::new().sum())}) }
    }

    /// calculates the dot product of 2 vectors and adds it to the output
    #[inline] fn copied_dot<V: VectorOps, S: Sum<<<Self::Unwrapped as Get>::Item as Mul<<V::Unwrapped as Get>::Item>>::Output> + AddAssign<<<Self::Unwrapped as Get>::Item as Mul<<V::Unwrapped as Get>::Item>>::Output>>(self, other: V) -> <<Self::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<VecCopiedDot<Self::Unwrapped, V::Unwrapped, S>> 
    where
        Self::Builder: VectorBuilderUnion<V::Builder>,
        <Self::Unwrapped as Get>::Item: Mul<<V::Unwrapped as Get>::Item>,
        <Self::Unwrapped as Get>::Item: Copy,
        <V::Unwrapped as Get>::Item: Copy,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, Y): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndHandleBool, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        Self: Sized
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap(VecCopiedDot { l_vec: self.unwrap(), r_vec: other.unwrap(), scalar: ManuallyDrop::new(NoneIter::new().sum())}) }
    }
}

/// a trait with various vector operations for const sized vectors
// TODO: rename?
pub trait ArrayVectorOps<const D: usize>: VectorOps {
    /// attaches a &mut MathVector to the first buffer
    /// note: due to current borrow checker limitations surrounding for<'a>, this isn't very useful in reality
    #[inline]
    fn attach_array<'a, T>(self, buf: &'a mut MathVector<T, D>) -> <Self::Builder as VectorBuilder>::Wrapped<VecAttachArray<'a, Self::Unwrapped, T, D>> where Self::Unwrapped: HasReuseBuf<FstHandleBool = N>, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecAttachArray{vec: self.unwrap(), buf}) }
    }

    /// creates a array in the first buffer
    #[inline] 
    fn create_array<T>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecCreateArray<Self::Unwrapped, T, D>> where Self::Unwrapped: HasReuseBuf<FstHandleBool = N>, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecCreateArray{vec: self.unwrap(), buf: MaybeUninit::uninit().assume_init()}) }
    }

    /// creates a array on the heap in the first buffer
    /// note: a pre-existing buffer may or may not be owned by the vector
    #[inline] 
    fn create_heap_array<T>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecCreateHeapArray<Self::Unwrapped, T, D>> where Self::Unwrapped: HasReuseBuf<FstHandleBool = N>, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecCreateHeapArray{vec: self.unwrap(), buf: ManuallyDrop::new(Box::new(MaybeUninit::uninit().assume_init()))}) }
    }

    /// creates a array in the first buffer if there isn't already one there
    #[inline] 
    fn maybe_create_array<T>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecMaybeCreateArray<Self::Unwrapped, T, D>> 
    where 
        <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg): TyBoolPair,
        Self: Sized
    {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecMaybeCreateArray{vec: self.unwrap(), buf: MaybeUninit::uninit().assume_init()}) }
    }

    /// creates a array on the heap in the first buffer if there isn't already one there
    /// note: a pre-existing buffer may or may not be on the heap or owned by the vector
    #[inline] 
    fn maybe_create_heap_array<T>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecMaybeCreateHeapArray<Self::Unwrapped, T, D>> 
    where 
        <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg): TyBoolPair,
        Self: Sized
    {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecMaybeCreateHeapArray{vec: self.unwrap(), buf: ManuallyDrop::new(Box::new(MaybeUninit::uninit().assume_init()))}) }
    }
}

/// a trait enabling a vector to be made repeatable
pub trait RepeatableVectorOps: VectorOps {
    /// the underlying repeatable VectorLike to be returned
    type RepeatableVector<'a>: VectorLike + IsRepeatable where Self: 'a;
    /// the underlying VectorLike that was used to make the vector repeatable
    type UsedVector: VectorLike;
    //type HeapedUsedVector: VectorLike;

    /// turns the vector into a repeatable one
    /// note: 
    /// this is *non-trivial*,
    /// in this process, the original vector has to be evaluated and stored, needing computation & memory
    fn make_repeatable<'a>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecAttachUsedVec<Self::RepeatableVector<'a>, Self::UsedVector>> 
    where
        Self: 'a,
        (<Self::RepeatableVector<'a> as HasOutput>::OutputBool, <Self::UsedVector as HasOutput>::OutputBool): FilterPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::FstHandleBool, <Self::UsedVector as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::SndHandleBool, <Self::UsedVector as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::BoundHandlesBool, <Self::UsedVector as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::FstOwnedBufferBool, <Self::UsedVector as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::SndOwnedBufferBool, <Self::UsedVector as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg, <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): TyBoolPair,
        <(<<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg, <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool) as TyBoolPair>::Or: IsTrue
    ;
}


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
    #[inline] fn get_builder(&self) -> Self::Builder {VectorExprBuilder}
    #[inline] fn size(&self) -> usize {D}
}
impl<V: VectorLike, const D: usize> ArrayVectorOps<D> for VectorExpr<V, D> {}

impl<V: VectorLike, const D: usize> RepeatableVectorOps for VectorExpr<V, D> 
where 
    <V::FstHandleBool as TyBool>::Neg: Filter,
    (V::BoundHandlesBool, Y): FilterPair,
    (V::FstHandleBool, <V::FstHandleBool as TyBool>::Neg): SelectPair<
    Selected<V::FstOwnedBuffer, MathVector<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<<V as Get>::Item>, D>> = MathVector<V::Item, D>>,
    (V::FstOwnedBufferBool, <V::FstHandleBool as TyBool>::Neg): TyBoolPair,
    (<V::FstHandleBool as TyBool>::Neg, V::FstOwnedBufferBool): TyBoolPair,
    (V::OutputBool, <(V::FstOwnedBufferBool, <V::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or): FilterPair,
    VecHalfBind<VecMaybeCreateArray<V, V::Item, D>>: HasReuseBuf<BoundTypes = <(V::BoundHandlesBool, Y) as FilterPair>::Filtered<V::BoundItems, V::Item>>
{
    type RepeatableVector<'a> = ReferringOwnedArray<'a, V::Item, D> where Self: 'a;
    type UsedVector = VecHalfBind<VecMaybeCreateArray<V, V::Item, D>>;

    fn make_repeatable<'a>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecAttachUsedVec<Self::RepeatableVector<'a>, Self::UsedVector>> 
    where
        Self: 'a,
        (<Self::RepeatableVector<'a> as HasOutput>::OutputBool, <Self::UsedVector as HasOutput>::OutputBool): FilterPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::FstHandleBool, <Self::UsedVector as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::SndHandleBool, <Self::UsedVector as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::BoundHandlesBool, <Self::UsedVector as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::FstOwnedBufferBool, <Self::UsedVector as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::SndOwnedBufferBool, <Self::UsedVector as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg, <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): TyBoolPair,
        <(<<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg, <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool) as TyBoolPair>::Or: IsTrue
    {
        let builder = self.get_builder();
        let mut vec_iter = self.maybe_create_array().half_bind().into_iter();
        unsafe {
            vec_iter.no_output_consume();
            builder.wrap(VecAttachUsedVec{vec: vec_iter.vec.get_bound_buf().referred().unwrap(), used_vec: ptr::read(&vec_iter.vec)})
        }
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
    #[inline] fn get_builder(&self) -> Self::Builder {VectorExprBuilder}
    #[inline] fn size(&self) -> usize {D}
}
impl<V: VectorLike, const D: usize> ArrayVectorOps<D> for Box<VectorExpr<V, D>> {}

impl<V: VectorLike, const D: usize> RepeatableVectorOps for Box<VectorExpr<V, D>>
where 
    <V::FstHandleBool as TyBool>::Neg: Filter,
    (V::BoundHandlesBool, Y): FilterPair,
    (V::FstHandleBool, <V::FstHandleBool as TyBool>::Neg): SelectPair<
    Selected<V::FstOwnedBuffer, MathVector<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<<V as Get>::Item>, D>> = MathVector<V::Item, D>>,
    (V::FstOwnedBufferBool, <V::FstHandleBool as TyBool>::Neg): TyBoolPair,
    (<V::FstHandleBool as TyBool>::Neg, V::FstOwnedBufferBool): TyBoolPair,
    (V::OutputBool, <(V::FstOwnedBufferBool, <V::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or): FilterPair,
    VecHalfBind<VecMaybeCreateArray<Box<V>, V::Item, D>>: HasReuseBuf<BoundTypes = <(V::BoundHandlesBool, Y) as FilterPair>::Filtered<V::BoundItems, V::Item>>
{
    type RepeatableVector<'a> = ReferringOwnedArray<'a, V::Item, D> where Self: 'a;
    type UsedVector = VecHalfBind<VecMaybeCreateArray<Box<V>, V::Item, D>>;

    fn make_repeatable<'a>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecAttachUsedVec<Self::RepeatableVector<'a>, Self::UsedVector>> 
    where
        Self: 'a,
        (<Self::RepeatableVector<'a> as HasOutput>::OutputBool, <Self::UsedVector as HasOutput>::OutputBool): FilterPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::FstHandleBool, <Self::UsedVector as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::SndHandleBool, <Self::UsedVector as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::BoundHandlesBool, <Self::UsedVector as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::FstOwnedBufferBool, <Self::UsedVector as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::SndOwnedBufferBool, <Self::UsedVector as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg, <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): TyBoolPair,
        <(<<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg, <Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool) as TyBoolPair>::Or: IsTrue
    {
        let builder = self.get_builder();
        let mut vec_iter = self.maybe_create_array().half_bind().into_iter();
        unsafe {
            vec_iter.no_output_consume();
            builder.wrap(VecAttachUsedVec{vec: vec_iter.vec.get_bound_buf().referred().unwrap(), used_vec: ptr::read(&vec_iter.vec)})
        }
    }
}


//already repeatable / can't truly be made repeatable so not implemented
unsafe impl<'a, T, const D: usize> VectorOps for &'a MathVector<T, D> {
    type Unwrapped = &'a [T; D];
    type Builder = VectorExprBuilder<D>;

    #[inline] fn unwrap(self) -> Self::Unwrapped {&self.0}
    #[inline] fn get_builder(&self) -> Self::Builder {VectorExprBuilder}
    #[inline] fn size(&self) -> usize {D}
}
impl<'a, T, const D: usize> ArrayVectorOps<D> for &'a MathVector<T, D> {}

unsafe impl<'a, T, const D: usize> VectorOps for &'a mut MathVector<T, D> {
    type Unwrapped = &'a mut [T; D];
    type Builder = VectorExprBuilder<D>;

    #[inline] fn unwrap(self) -> Self::Unwrapped {&mut self.0}
    #[inline] fn get_builder(&self) -> Self::Builder {VectorExprBuilder}
    #[inline] fn size(&self) -> usize {D}
}
impl<'a, T, const D: usize> ArrayVectorOps<D> for &'a mut MathVector<T, D> {}

unsafe impl<'a, T, const D: usize> VectorOps for &'a Box<MathVector<T, D>> {
    type Unwrapped = &'a [T; D];
    type Builder = VectorExprBuilder<D>;

    #[inline] fn unwrap(self) -> Self::Unwrapped {&self.0}
    #[inline] fn get_builder(&self) -> Self::Builder {VectorExprBuilder}
    #[inline] fn size(&self) -> usize {D}
}
impl<'a, T, const D: usize> ArrayVectorOps<D> for &'a Box<MathVector<T, D>> {}

unsafe impl<'a, T, const D: usize> VectorOps for &'a mut Box<MathVector<T, D>> {
    type Unwrapped = &'a mut [T; D];
    type Builder = VectorExprBuilder<D>;

    #[inline] fn unwrap(self) -> Self::Unwrapped {&mut self.0}
    #[inline] fn get_builder(&self) -> Self::Builder {VectorExprBuilder}
    #[inline] fn size(&self) -> usize {D}
}
impl<'a, T, const D: usize> ArrayVectorOps<D> for &'a mut Box<MathVector<T, D>> {}



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
    #[inline] fn get_builder(&self) -> Self::Builder {RSVectorExprBuilder{size: self.size}}
    #[inline] fn size(&self) -> usize {self.size}
}
 
macro_rules! impl_ops_for_wrapper {
    (
        $(
            <$($($lifetime:lifetime),+, )? $($generic:ident $(:)? $($lifetime_bound:lifetime |)? $($fst_trait_bound:path $(| $trait_bound:path)*)?,)+ $({$size:ident})?>,
            $ty:ty,
            trait_vector: $trait_vector:ty,
            true_vector: $true_vector:ty;
        )*
    ) => {
        $(
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

            impl<$($($lifetime),+, )? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: AddAssign<<$trait_vector as Get>::Item> $(, const $size: usize)?> AddAssign<$ty> for MathVector<Z, D> 
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

            impl<$($($lifetime),+, )? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: SubAssign<<$trait_vector as Get>::Item> $(, const $size: usize)?> SubAssign<$ty> for MathVector<Z, D> 
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
);