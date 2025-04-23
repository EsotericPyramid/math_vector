use crate::{trait_specialization_utils::*, util_structs::NoneIter, util_traits::HasOutput};
use std::{mem::ManuallyDrop, ops::*};

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
pub struct VectorExpr<T: VectorLike, const D: usize>(pub(crate) T); // note: VectorExpr only holds fully unused VectorLike objects

impl<T: VectorLike, const D: usize> VectorExpr<T, D> {
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
        GetBool = T::GetBool,
        Inputs = (),
        Item = T::Item,
        BoundItems = T::BoundItems,

        OutputBool = T::OutputBool,
        Output = T::Output,

        FstHandleBool = T::FstHandleBool,
        SndHandleBool = T::SndHandleBool,
        BoundHandlesBool = T::BoundHandlesBool,
        FstOwnedBufferBool = T::FstOwnedBufferBool,
        SndOwnedBufferBool = T::SndOwnedBufferBool,
        FstOwnedBuffer = T::FstOwnedBuffer,
        SndOwnedBuffer = T::SndOwnedBuffer,
        FstType = T::FstType,
        SndType = T::SndType,
        BoundTypes = T::BoundTypes,
    >>, D> where T: 'static {
        VectorExpr(Box::new(DynamicVectorLike{vec: self.unwrap(), inputs: None}) as Box<dyn VectorLike<
            GetBool = T::GetBool,
            Inputs = (),
            Item = T::Item,
            BoundItems = T::BoundItems,

            OutputBool = T::OutputBool,
            Output = T::Output,

            FstHandleBool = T::FstHandleBool,
            SndHandleBool = T::SndHandleBool,
            BoundHandlesBool = T::BoundHandlesBool,
            FstOwnedBufferBool = T::FstOwnedBufferBool,
            SndOwnedBufferBool = T::SndOwnedBufferBool,
            FstOwnedBuffer = T::FstOwnedBuffer,
            SndOwnedBuffer = T::SndOwnedBuffer,
            FstType = T::FstType,
            SndType = T::SndType,
            BoundTypes = T::BoundTypes,
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
    pub fn consume(self) -> T::Output where T: HasReuseBuf<BoundTypes = T::BoundItems> {
        VectorIter::<T, D>{
            vec: unsafe { std::ptr::read(&std::mem::ManuallyDrop::new(self).0) },
            live_input_start: 0,
            dead_output_start: 0,
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
    pub fn eval(self) -> <VecBind<VecMaybeCreateBuf<T, T::Item, D>> as HasOutput>::Output 
    where 
        <T::FstHandleBool as TyBool>::Neg: Filter,
        (T::FstHandleBool, <T::FstHandleBool as TyBool>::Neg): SelectPair,
        (T::FstOwnedBufferBool, <T::FstHandleBool as TyBool>::Neg): TyBoolPair,
        (T::OutputBool, <(T::FstOwnedBufferBool, <T::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or): FilterPair,
        (T::BoundHandlesBool, Y): FilterPair,
        VecBind<VecMaybeCreateBuf<T, T::Item, D>>: HasReuseBuf<BoundTypes = <VecBind<VecMaybeCreateBuf<T, T::Item, D>> as Get>::BoundItems>
    {
        self.maybe_create_buf().bind().consume()
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
    pub fn heap_eval(self) -> <VecBind<VecMaybeCreateHeapBuf<T, T::Item, D>> as HasOutput>::Output 
    where 
        <T::FstHandleBool as TyBool>::Neg: Filter,
        (T::FstHandleBool, <T::FstHandleBool as TyBool>::Neg): SelectPair,
        (T::FstOwnedBufferBool, <T::FstHandleBool as TyBool>::Neg): TyBoolPair,
        (T::OutputBool, <(T::FstOwnedBufferBool, <T::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or): FilterPair,
        (T::BoundHandlesBool, Y): FilterPair,
        VecBind<VecMaybeCreateHeapBuf<T, T::Item, D>>: HasReuseBuf<BoundTypes = <VecBind<VecMaybeCreateHeapBuf<T, T::Item, D>> as Get>::BoundItems>
    {
        self.maybe_create_heap_buf().bind().consume()
    }
}

impl<V: VectorLike + IsRepeatable, const D: usize> VectorExpr<V, D> {
    /// Retrieves an arbitrary value from a repeatable VectorExpr
    pub fn get(&mut self, index: usize) -> V::Item {
        // the nature of IsRepeatable means that any index can be called any number of times so this is fine
        if index >= D {panic!("math_vector Error: index access out of bound")}
        unsafe {
            let inputs = self.0.get_inputs(index);
            let (item, _) = self.0.process(inputs);
            item
        }
    }

    /// TODO: remove?
    /// Note:   Some buffers do not drop pre-existing values when being filled as such values may be undefined data
    ///         however, this means that binding an index multiple times can cause a leak (ie. with Box<T>'s being bound)
    ///         Additionally, if the buffer is owned by the vector, the vector expr is also responsible for dropping filled indices
    ///         however, such filled indices filled via this method aren't tracked so further leaks can happen 
    ///         (assuming it isn't retroactivly noted as filled during evaluation/iteration)
    /// Note TLDR: this method is extremely prone to causing memory leaks
    pub fn binding_get(&mut self, index: usize) -> V::Item where V: HasReuseBuf<BoundTypes = V::BoundItems> {
        // the nature of IsRepeatable means that any index can be called any number of times so this is fine
        if index >= D {panic!("math_vector Error: index access out of bound")}
        unsafe {
            let inputs = self.0.get_inputs(index);
            let (item, bound_items) = self.0.process(inputs);
            self.0.assign_bound_bufs(index, bound_items); // NOTE: all current things which have IsRepeatable don't have any bound items, however, it is not restricted by the definition
            item
        }
    }
}

impl<T: VectorLike, const D: usize> Drop for VectorExpr<T, D> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            for i in 0..D {
                self.0.drop_inputs(i);
            }
            self.0.drop_output();
        }
    }
}

impl<T: VectorLike, const D: usize> IntoIterator for VectorExpr<T, D> where T: HasReuseBuf<BoundTypes = T::BoundItems> {
    type IntoIter = VectorIter<T, D>;
    type Item = <T as Get>::Item;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        VectorIter{
            vec: unsafe { std::ptr::read(&std::mem::ManuallyDrop::new(self).0) },
            live_input_start: 0,
            dead_output_start: 0,
        }
    }
}




/// a const-sized VectorExpr iterator
pub struct VectorIter<T: VectorLike, const D: usize>{vec: T, live_input_start: usize, dead_output_start: usize} // note: ranges are start inclusive, end exclusive

impl<T: VectorLike, const D: usize> VectorIter<T, D> {
    /// retrieves the next item without checking
    /// Safety: there must be another item to return
    #[inline]
    pub unsafe fn next_unchecked(&mut self) -> T::Item where T: HasReuseBuf<BoundTypes = T::BoundItems> { unsafe {
        let index = self.live_input_start;
        self.live_input_start += 1;
        let inputs = self.vec.get_inputs(index);
        let (item, bound_items) = self.vec.process(inputs);
        self.vec.assign_bound_bufs(index, bound_items);
        self.dead_output_start += 1;
        item
    }}

    /// retrieves the VectorIter's output without checking consumption
    /// Safety: the VectorLike must be fully consumed
    #[inline]
    pub unsafe fn unchecked_output(self) -> T::Output {
        // NOTE: manual drop shenanigans to prevent VectorIter from being dropped normally
        //       doing so would incorrectly drop HasReuseBuf & output 
        let mut man_drop_self = std::mem::ManuallyDrop::new(self);
        let output; 
        unsafe { 
            output = man_drop_self.vec.output(); 
            std::ptr::drop_in_place(&mut man_drop_self.vec);
        }
        output
    }

    /// retrieves the VectorIter's output
    /// the VectorIter must be fully consumed or this function will panic
    #[inline]
    pub fn output(self) -> T::Output {
        assert!(self.live_input_start == D, "math_vector error: A VectorIter must be fully used before outputting");
        debug_assert!(self.dead_output_start == D, "math_vector internal error: A VectorIter's output buffers (somehow) weren't fully filled despite the inputs being fully used, likely an internal issue");
        unsafe {self.unchecked_output()}
    }

    /// fully consumes the VectorIter and then returns its output
    #[inline]
    pub fn consume(mut self) -> T::Output where T: HasReuseBuf<BoundTypes = T::BoundItems> {
        self.no_output_consume();
        unsafe {self.unchecked_output()} // safety: VectorIter was fully used
    }

    /// fully consumes the VectorIter without returning its output
    #[inline]
    pub fn no_output_consume(&mut self) where T: HasReuseBuf<BoundTypes = T::BoundItems> {
        while self.live_input_start < D {
            unsafe { let _ = self.next_unchecked(); }
        }
    }
}

impl<T: VectorLike, const D: usize> Drop for VectorIter<T, D> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            // note: when VectorIter outputs, it is forgotten, so we can assume output hasn't been called
            self.vec.drop_output();
            for i in 0..self.dead_output_start { //up to the start of the dead area in output
                self.vec.drop_bound_bufs_index(i);
            }
            for i in self.live_input_start..D {
                self.vec.drop_inputs(i);
            }
        }
    }
}

impl<T: VectorLike, const D: usize> Iterator for VectorIter<T, D> where T: HasReuseBuf<BoundTypes = T::BoundItems> {
    type Item = T::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.live_input_start < D { // != instead of < as it is known that start is always <= end so their equivilent
            unsafe { Some(self.next_unchecked()) }
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = D - self.live_input_start;
        (size, Some(size))
    }
}

impl<T: VectorLike, const D: usize> ExactSizeIterator for VectorIter<T, D> where T: HasReuseBuf<BoundTypes = T::BoundItems> {}

impl<T: VectorLike, const D: usize> std::iter::FusedIterator for VectorIter<T, D> where T: HasReuseBuf<BoundTypes = T::BoundItems> {}

/// a simple type alias for a VectorExpr created from an array of type [T; D]
pub type MathVector<T, const D: usize> = VectorExpr<OwnedArray<T, D>, D>;

impl<T, const D: usize> MathVector<T, D> {
    // TODO: remove?
    #[inline] pub fn into_array(self) -> [T; D] {self.unwrap().unwrap()}
    // TODO: replace?
    #[inline] pub fn into_heap_array(self: Box<Self>) -> Box<[T; D]> {
        unsafe { std::mem::transmute::<Box<Self>, Box<[T; D]>>(self) }
    }
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
        unsafe { std::mem::transmute::<Box<Self>, VectorExpr<Box<ReplaceArray<T, D>>, D>>(self) }
    }

    /// converts this MathVector to a repeatable VectorExpr w/ Item = &'a T
    #[inline] pub fn referred<'a>(self) -> VectorExpr<ReferringOwnedArray<'a, T, D>, D> where T: 'a {
        VectorExpr(ReferringOwnedArray(unsafe {std::mem::transmute_copy::<ManuallyDrop<[T; D]>, [T; D]>(&self.unwrap().0)}, std::marker::PhantomData)) //FIXME: unecessary transmute copy to get the compiler to not complain
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
        VectorExpr(OwnedArray(std::mem::ManuallyDrop::new(value)))
    }
}

impl<T, const D: usize> Into<[T; D]> for MathVector<T, D> {
    #[inline] fn into(self) -> [T; D] {self.into_array()}
}

impl<'a, T, const D: usize> From<&'a [T; D]> for &'a MathVector<T, D> {
    #[inline]
    fn from(value: &'a [T; D]) -> Self {
        unsafe { std::mem::transmute::<&'a [T; D], &'a MathVector<T, D>>(value) }
    }
} 

impl<'a, T, const D: usize> Into<&'a [T; D]> for &'a MathVector<T, D> {
    #[inline]
    fn into(self) -> &'a [T; D] {
        unsafe { std::mem::transmute::<&'a MathVector<T, D>, &'a [T; D]>(self) }
    }
} 

impl<'a, T, const D: usize> From<&'a mut [T; D]> for &'a mut MathVector<T, D> {
    #[inline]
    fn from(value: &'a mut [T; D]) -> Self {
        unsafe { std::mem::transmute::<&'a mut [T; D], &'a mut MathVector<T, D>>(value) }
    }
} 

impl<'a, T, const D: usize> Into<&'a mut [T; D]> for &'a mut MathVector<T, D> {
    #[inline]
    fn into(self) -> &'a mut [T; D] {
        unsafe { std::mem::transmute::<&'a mut MathVector<T, D>, &'a mut [T; D]>(self) }
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

// TODO: Add addassign and subassign for arbitrary T2: VectorOps
impl<T1: AddAssign<T2>, T2, const D: usize> AddAssign<MathVector<T2, D>> for MathVector<T1, D> {
    #[inline]
    fn add_assign(&mut self, rhs: MathVector<T2, D>) {
        VectorOps::add_assign(self, rhs).consume();
    }
}

impl<T1: SubAssign<T2>, T2, const D: usize> SubAssign<MathVector<T2, D>> for MathVector<T1, D> {
    #[inline]
    fn sub_assign(&mut self, rhs: MathVector<T2, D>) {
        VectorOps::sub_assign(self, rhs).consume();
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


impl<T1: std::iter::Sum<T2> + AddAssign<T2>, T2, const D: usize> std::iter::Sum<MathVector<T2, D>> for MathVector<T1, D> {
    #[inline]
    fn sum<I: Iterator<Item = MathVector<T2, D>>>(iter: I) -> Self {
        let mut sum = vector_gen(|| std::iter::Sum::sum(NoneIter::new())).create_buf().bind().consume();
        for vec in iter {
            sum += vec;
        }
        sum
    }
}

#[inline]
pub fn vector_gen<F: FnMut() -> O, O, const D: usize>(f: F) -> VectorExpr<VecGenerator<F, O>, D> {
    VectorExpr(VecGenerator(f))
}

#[inline]
pub fn vector_index_gen<F: FnMut(usize) -> O, O, const D: usize>(f: F) -> VectorExpr<VecIndexGenerator<F, O>, D> {
    VectorExpr(VecIndexGenerator(f))
}

// TODO: finish implementing RSVectorExpr
pub struct RSVectorExpr<T: VectorLike>{vec: T, size: usize}

impl<T: VectorLike> Drop for RSVectorExpr<T> {
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

// a trait with various vector operations
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
        VecBind<Self::Unwrapped>: HasReuseBuf<BoundTypes = <VecBind<Self::Unwrapped> as Get>::BoundItems>,
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

    /// offsets each element of the vector down by the given offset
    #[inline] 
    fn offset_down(self, offset: usize) -> <Self::Builder as VectorBuilder>::Wrapped<VecOffset<Self::Unwrapped>> where Self: Sized {
        let size = self.size();
        let builder = self.get_builder();
        unsafe { builder.wrap(VecOffset{vec: self.unwrap(), offset: offset % size, size}) }
    }

    /// offsets each element of the vector up by the given offset
    #[inline] 
    fn offset_up(self, offset: usize) -> <Self::Builder as VectorBuilder>::Wrapped<VecOffset<Self::Unwrapped>> where Self: Sized {
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
    /// note: fold_ref should be used whenever possible due to implementation
    #[inline] fn fold<F: FnMut(O, <Self::Unwrapped as Get>::Item) -> O, O>(self, f: F, init: O) -> <Self::Builder as VectorBuilder>::Wrapped<VecFold<Self::Unwrapped, F, O>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecFold{vec: self.unwrap(), f, cell: Some(init)}) }
    }

    /// folds the vector's items into a single value using the provided closure
    #[inline] fn fold_ref<F: FnMut(&mut O, <Self::Unwrapped as Get>::Item), O>(self, f: F, init: O) -> <Self::Builder as VectorBuilder>::Wrapped<VecFoldRef<Self::Unwrapped, F, O>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecFoldRef{vec: self.unwrap(), f, cell: std::mem::ManuallyDrop::new(init)}) }
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

    /// rem assign (%=) the vector's items (&mut T) with a scalar
    #[inline] fn rem_assign<'a, I: 'a + RemAssign<S>, S: Copy>(self, scalar: S) -> <Self::Builder as VectorBuilder>::Wrapped<VecRemAssign<'a, Self::Unwrapped, I, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self::Unwrapped: Get<Item = &'a mut I>, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecRemAssign{vec: self.unwrap(), scalar})}
    }

    /// calculates the sum of the vector's elements and adds it to the output
    #[inline] fn sum<S: std::iter::Sum<<Self::Unwrapped as Get>::Item> + AddAssign<<Self::Unwrapped as Get>::Item>>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecSum<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecSum{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(NoneIter::new().sum())})}
    }

    /// calculates the sum of the vector's elements, initialized at given value (not necessarily 0), and adds it to the output
    #[inline] fn initialized_sum<S: AddAssign<<Self::Unwrapped as Get>::Item>>(self, init: S) -> <Self::Builder as VectorBuilder>::Wrapped<VecSum<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecSum{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(init)})}
    }

    /// calculates the product of the vector's elements and adds it to the output
    #[inline] fn product<S: std::iter::Product<<Self::Unwrapped as Get>::Item> + MulAssign<<Self::Unwrapped as Get>::Item>>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecProduct<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecProduct{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(NoneIter::new().product())})}
    }

    /// calculates the product of the vector's elements, initialized at given value (not necessarily 1), and adds it to the output
    #[inline] fn initialized_product<S: MulAssign<<Self::Unwrapped as Get>::Item>>(self, init: S) -> <Self::Builder as VectorBuilder>::Wrapped<VecProduct<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecProduct{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(init)})}
    }

    /// calculates the square of the vector's magnitude (ie. sum of each element's square) and adds it to the output
    #[inline] fn sqr_mag<S: std::iter::Sum<<Self::Unwrapped as Get>::Item> + AddAssign<<<Self::Unwrapped as Get>::Item as Mul>::Output>>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecSqrMag<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, <Self::Unwrapped as Get>::Item: Copy + Mul, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecSqrMag{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(NoneIter::new().sum())})}
    }    

    // TODO: remove?
    #[inline] fn initialized_sqr_mag<S: AddAssign<<<Self::Unwrapped as Get>::Item as Mul>::Output>>(self, init: S) -> <Self::Builder as VectorBuilder>::Wrapped<VecSqrMag<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, <Self::Unwrapped as Get>::Item: Copy + Mul, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecSqrMag{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(init)})}
    }

    /// calculates the sum of the vector's elements, adding it to the output, while maintaining the vector's items
    #[inline] fn copied_sum<S: std::iter::Sum<<Self::Unwrapped as Get>::Item> + AddAssign<<Self::Unwrapped as Get>::Item>>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecCopiedSum<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, <Self::Unwrapped as Get>::Item: Copy, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecCopiedSum{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(NoneIter::new().sum())})}
    }

    /// calculates the product of the vector's elements, adding it to the output, while maintaining the vector's items
    #[inline] fn copied_product<S: std::iter::Product<<Self::Unwrapped as Get>::Item> + MulAssign<<Self::Unwrapped as Get>::Item>>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecCopiedProduct<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, <Self::Unwrapped as Get>::Item: Copy, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecCopiedProduct{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(NoneIter::new().product())})}
    }

    /// calculates the square of the vector's magnitude (ie. sum of each element's square), adding it to the output, while maintaining the vector's items
    #[inline] fn copied_sqr_mag<S: std::iter::Sum<<Self::Unwrapped as Get>::Item> + AddAssign<<<Self::Unwrapped as Get>::Item as Mul>::Output>>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecCopiedSqrMag<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, <Self::Unwrapped as Get>::Item: Copy + Mul, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecCopiedSqrMag{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(NoneIter::new().sum())})}
    }

    /// calculates the sum of the vector's elements, initialized at given value (not necessarily 0), adding it to the output while maintaing the vector's items
    #[inline] fn initialized_copied_sum<S: AddAssign<<Self::Unwrapped as Get>::Item>>(self, init: S) -> <Self::Builder as VectorBuilder>::Wrapped<VecCopiedSum<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, <Self::Unwrapped as Get>::Item: Copy, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecCopiedSum{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(init)})}
    }

    /// calculates the products of the vector's elements, initialized at given value (not necessarily 0), adding it to the output while maintaing the vector's items
    #[inline] fn initialized_copied_product<S: MulAssign<<Self::Unwrapped as Get>::Item>>(self, init: S) -> <Self::Builder as VectorBuilder>::Wrapped<VecCopiedProduct<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, <Self::Unwrapped as Get>::Item: Copy, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecCopiedProduct{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(init)})}
    }

    // TODO: remove?
    #[inline] fn initialized_copied_sqr_mag<S: AddAssign<<<Self::Unwrapped as Get>::Item as Mul>::Output>>(self, init: S) -> <Self::Builder as VectorBuilder>::Wrapped<VecCopiedSqrMag<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, <Self::Unwrapped as Get>::Item: Copy + Mul, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecCopiedSqrMag{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(init)})}
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
    #[inline] fn dot<V: VectorOps, S: std::iter::Sum<<<Self::Unwrapped as Get>::Item as Mul<<V::Unwrapped as Get>::Item>>::Output> + AddAssign<<<Self::Unwrapped as Get>::Item as Mul<<V::Unwrapped as Get>::Item>>::Output>>(self, other: V) -> <<Self::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<VecDot<Self::Unwrapped, V::Unwrapped, S>> 
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
        unsafe { builder.wrap(VecDot { l_vec: self.unwrap(), r_vec: other.unwrap(), scalar: std::mem::ManuallyDrop::new(NoneIter::new().sum())}) }
    }

    // TODO: remove?
    #[inline] fn initialized_dot<V: VectorOps, S: AddAssign<<<Self::Unwrapped as Get>::Item as Mul<<V::Unwrapped as Get>::Item>>::Output>>(self, other: V, init: S) -> <<Self::Builder as VectorBuilderUnion<V::Builder>>::Union as VectorBuilder>::Wrapped<VecDot<Self::Unwrapped, V::Unwrapped, S>> 
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
        unsafe { builder.wrap(VecDot { l_vec: self.unwrap(), r_vec: other.unwrap(), scalar: std::mem::ManuallyDrop::new(init)}) }
    }
}

/// a trait with various vector operations for const sized vectors
// TODO: rename?
pub trait ArrayVectorOps<const D: usize>: VectorOps {
    /// attaches a &mut MathVector to the first buffer
    /// note: due to current borrow checker limitations surrounding for<'a>, this isn't very useful in reality
    #[inline]
    fn attach_buf<'a, T>(self, buf: &'a mut MathVector<T, D>) -> <Self::Builder as VectorBuilder>::Wrapped<VecAttachBuf<'a, Self::Unwrapped, T, D>> where Self::Unwrapped: HasReuseBuf<FstHandleBool = N>, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecAttachBuf{vec: self.unwrap(), buf}) }
    }

    /// creates a buffer in the first buffer
    #[inline] 
    fn create_buf<T>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecCreateBuf<Self::Unwrapped, T, D>> where Self::Unwrapped: HasReuseBuf<FstHandleBool = N>, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecCreateBuf{vec: self.unwrap(), buf: std::mem::MaybeUninit::uninit().assume_init()}) }
    }

    /// creates a buffer on the heap in the first buffer
    /// note: a pre-existing buffer may or may not be owned by the vector
    #[inline] 
    fn create_heap_buf<T>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecCreateHeapBuf<Self::Unwrapped, T, D>> where Self::Unwrapped: HasReuseBuf<FstHandleBool = N>, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecCreateHeapBuf{vec: self.unwrap(), buf: Box::new(std::mem::MaybeUninit::uninit().assume_init())}) }
    }

    /// creates a buffer in the first buffer if there isn't already one there
    #[inline] 
    fn maybe_create_buf<T>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecMaybeCreateBuf<Self::Unwrapped, T, D>> 
    where 
        <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg): TyBoolPair,
        Self: Sized
    {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecMaybeCreateBuf{vec: self.unwrap(), buf: std::mem::MaybeUninit::uninit().assume_init()}) }
    }

    /// creates a buffer on the heap in the first buffer if there isn't already one there
    /// note: a pre-existing buffer may or may not be on the heap or owned by the vector
    #[inline] 
    fn maybe_create_heap_buf<T>(self) -> <Self::Builder as VectorBuilder>::Wrapped<VecMaybeCreateHeapBuf<Self::Unwrapped, T, D>> 
    where 
        <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg): TyBoolPair,
        Self: Sized
    {
        let builder = self.get_builder();
        unsafe { builder.wrap(VecMaybeCreateHeapBuf{vec: self.unwrap(), buf: Box::new(std::mem::MaybeUninit::uninit().assume_init())}) }
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

macro_rules! if_lifetimes {
    (($item:item); $($lifetime:lifetime),+) => {$item};
    (($item:item); ) => {}
}

macro_rules! overload_operators {
    (
        <$($($lifetime:lifetime),+, )? $($generic:ident $(:)? $($lifetime_bound:lifetime |)? $($fst_trait_bound:path $(| $trait_bound:path)*)?),+, {$size:ident}>,
        $ty:ty,
        vector: $vector:ty,
        item: $item:ty
    ) => {
        impl<$($($lifetime),+, )? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: Copy, const $size: usize> Mul<Z> for $ty where (<$vector as HasOutput>::OutputBool, N): FilterPair, $item: Mul<Z>, Self: Sized {
            type Output = VectorExpr<VecMulL<<$ty as VectorOps>::Unwrapped, Z>, $size>;
    
            #[inline]
            fn mul(self, rhs: Z) -> Self::Output {
                self.mul_l(rhs)
            }
        }

        impl<$($($lifetime),+, )? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: Copy, const $size: usize> Div<Z> for $ty where (<$vector as HasOutput>::OutputBool, N): FilterPair, $item: Div<Z>, Self: Sized {
            type Output = VectorExpr<VecDivL<<$ty as VectorOps>::Unwrapped, Z>, $size>;
    
            #[inline]
            fn div(self, rhs: Z) -> Self::Output {
                self.div_l(rhs)
            }
        }

        impl<$($($lifetime),+, )? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: Copy, const $size: usize> Rem<Z> for $ty where (<$vector as HasOutput>::OutputBool, N): FilterPair, $item: Rem<Z>, Self: Sized {
            type Output = VectorExpr<VecRemL<<$ty as VectorOps>::Unwrapped, Z>, $size>;
    
            #[inline]
            fn rem(self, rhs: Z) -> Self::Output {
                self.rem_l(rhs)
            }
        }

        if_lifetimes!((
            impl<$($($lifetime),+, )? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: AddAssign<$item>, const $size: usize> AddAssign<$ty> for MathVector<Z, D> 
            where
                (N, <$vector as HasOutput>::OutputBool): FilterPair,
                (N, <$vector as HasReuseBuf>::FstHandleBool): SelectPair,
                (N, <$vector as HasReuseBuf>::SndHandleBool): SelectPair,
                (N, <$vector as HasReuseBuf>::BoundHandlesBool): SelectPair,
                (N, <$vector as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
                (N, <$vector as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
                (<(N, <$vector as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
            {
                #[inline]
                fn add_assign(&mut self, rhs: $ty) {
                    VectorOps::add_assign(self, rhs).consume();
                }
            }
        ); $($($lifetime),+)?);

        if_lifetimes!((
            impl<$($($lifetime),+, )? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: SubAssign<$item>, const $size: usize> SubAssign<$ty> for MathVector<Z, D> 
            where
                (N, <$vector as HasOutput>::OutputBool): FilterPair,
                (N, <$vector as HasReuseBuf>::FstHandleBool): SelectPair,
                (N, <$vector as HasReuseBuf>::SndHandleBool): SelectPair,
                (N, <$vector as HasReuseBuf>::BoundHandlesBool): SelectPair,
                (N, <$vector as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
                (N, <$vector as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
                (<(N, <$vector as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
            {
                #[inline]
                fn sub_assign(&mut self, rhs: $ty) {
                    VectorOps::sub_assign(self, rhs).consume();
                }
            }
        ); $($($lifetime),+)?);
    };
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
        unsafe { std::ptr::read(&std::mem::ManuallyDrop::new(self).0) } 
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
    VecHalfBind<VecMaybeCreateBuf<V, V::Item, D>>: HasReuseBuf<BoundTypes = <(V::BoundHandlesBool, Y) as FilterPair>::Filtered<V::BoundItems, V::Item>>
{
    type RepeatableVector<'a> = ReferringOwnedArray<'a, V::Item, D> where Self: 'a;
    type UsedVector = VecHalfBind<VecMaybeCreateBuf<V, V::Item, D>>;

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
        let mut vec_iter = self.maybe_create_buf().half_bind().into_iter();
        unsafe {
            vec_iter.no_output_consume();
            builder.wrap(VecAttachUsedVec{vec: vec_iter.vec.get_bound_buf().referred().unwrap(), used_vec: std::ptr::read(&vec_iter.vec)})
        }
    }
}

overload_operators!(<V: VectorLike, {D}>, VectorExpr<V, D>, vector: V, item: V::Item);


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
        Box::new(unsafe { std::ptr::read(&std::mem::ManuallyDrop::new(self).0) })
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
    VecHalfBind<VecMaybeCreateBuf<Box<V>, V::Item, D>>: HasReuseBuf<BoundTypes = <(V::BoundHandlesBool, Y) as FilterPair>::Filtered<V::BoundItems, V::Item>>
{
    type RepeatableVector<'a> = ReferringOwnedArray<'a, V::Item, D> where Self: 'a;
    type UsedVector = VecHalfBind<VecMaybeCreateBuf<Box<V>, V::Item, D>>;

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
        let mut vec_iter = self.maybe_create_buf().half_bind().into_iter();
        unsafe {
            vec_iter.no_output_consume();
            builder.wrap(VecAttachUsedVec{vec: vec_iter.vec.get_bound_buf().referred().unwrap(), used_vec: std::ptr::read(&vec_iter.vec)})
        }
    }
}

overload_operators!(<V: VectorLike, {D}>, Box<VectorExpr<V, D>>, vector: V, item: V::Item);

//already repeatable / can't truly be made repeatable so not implemented
unsafe impl<'a, T, const D: usize> VectorOps for &'a MathVector<T, D> {
    type Unwrapped = &'a [T; D];
    type Builder = VectorExprBuilder<D>;

    #[inline] fn unwrap(self) -> Self::Unwrapped {&self.0}
    #[inline] fn get_builder(&self) -> Self::Builder {VectorExprBuilder}
    #[inline] fn size(&self) -> usize {D}
}
impl<'a, T, const D: usize> ArrayVectorOps<D> for &'a MathVector<T, D> {}
overload_operators!(<'a, T, {D}>, &'a MathVector<T, D>, vector: &'a [T; D], item: &'a T);

unsafe impl<'a, T, const D: usize> VectorOps for &'a mut MathVector<T, D> {
    type Unwrapped = &'a mut [T; D];
    type Builder = VectorExprBuilder<D>;

    #[inline] fn unwrap(self) -> Self::Unwrapped {&mut self.0}
    #[inline] fn get_builder(&self) -> Self::Builder {VectorExprBuilder}
    #[inline] fn size(&self) -> usize {D}
}
impl<'a, T, const D: usize> ArrayVectorOps<D> for &'a mut MathVector<T, D> {}
overload_operators!(<'a, T, {D}>, &'a mut MathVector<T, D>, vector: &'a mut [T; D], item: &'a mut T);

unsafe impl<'a, T, const D: usize> VectorOps for &'a Box<MathVector<T, D>> {
    type Unwrapped = &'a [T; D];
    type Builder = VectorExprBuilder<D>;

    #[inline] fn unwrap(self) -> Self::Unwrapped {&self.0}
    #[inline] fn get_builder(&self) -> Self::Builder {VectorExprBuilder}
    #[inline] fn size(&self) -> usize {D}
}
impl<'a, T, const D: usize> ArrayVectorOps<D> for &'a Box<MathVector<T, D>> {}
overload_operators!(<'a, T, {D}>, &'a Box<MathVector<T, D>>, vector: &'a [T; D], item: &'a T);

unsafe impl<'a, T, const D: usize> VectorOps for &'a mut Box<MathVector<T, D>> {
    type Unwrapped = &'a mut [T; D];
    type Builder = VectorExprBuilder<D>;

    #[inline] fn unwrap(self) -> Self::Unwrapped {&mut self.0}
    #[inline] fn get_builder(&self) -> Self::Builder {VectorExprBuilder}
    #[inline] fn size(&self) -> usize {D}
}
impl<'a, T, const D: usize> ArrayVectorOps<D> for &'a mut Box<MathVector<T, D>> {}
overload_operators!(<'a, T, {D}>, &'a mut Box<MathVector<T, D>>, vector: &'a mut [T; D], item: &'a mut T);

 
macro_rules! impl_binary_ops_for_wrapper {
    (
        $(
            $($size:ident,)?
            <$($($lifetime:lifetime),+, )? $($generic:ident $(:)? $($lifetime_bound:lifetime |)? $($fst_trait_bound:path $(| $trait_bound:path)*)?),+>,
            $ty:ty,
            trait_vector: $trait_vector:ty,
            true_vector: $true_vector:ty;
        )*
    ) => {
        $(
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

impl_binary_ops_for_wrapper!(
    D, <V: VectorLike>, VectorExpr<V, D>, trait_vector: V, true_vector: V;
    D, <V: VectorLike>, Box<VectorExpr<V, D>>, trait_vector: V, true_vector: Box<V>;
    D, <'a, T>, &'a MathVector<T,D>, trait_vector: &'a [T; D], true_vector: &'a [T; D];
    D, <'a, T>, &'a mut MathVector<T,D>, trait_vector: &'a mut [T; D], true_vector: &'a mut [T; D];
    D, <'a, T>, &'a Box<MathVector<T,D>>, trait_vector: &'a [T; D], true_vector: &'a [T; D];
    D, <'a, T>, &'a mut Box<MathVector<T,D>>, trait_vector: &'a mut [T; D], true_vector: &'a mut [T; D];
);

/*

impl<V1: VectorLike, V2: VectorOps, const D: usize> Add<V2> for VectorExpr<V1,D> where 
    <VectorExpr<V1,D> as VectorOps>::Builder: VectorBuilderUnion<V2::Builder>,
    <V1 as Get>::Item: Add<<V2::Unwrapped as Get>::Item>,
    (<V1 as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool): FilterPair,
    (<(<V1 as HasOutput>::OutputBool, <V2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
    (<V1 as HasReuseBuf>::BoundHandlesBool, <V2::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
    (<V1 as HasReuseBuf>::FstHandleBool, <V2::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
    (<V1 as HasReuseBuf>::SndHandleBool, <V2::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
    (<V1 as HasReuseBuf>::FstOwnedBufferBool, <V2::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
    (<V1 as HasReuseBuf>::SndOwnedBufferBool, <V2::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
    VectorExpr<V1,D>: Sized
{
    type Output = <<<VectorExpr<V1,D> as VectorOps>::Builder as VectorBuilderUnion<V2::Builder>>::Union as VectorBuilder>::Wrapped<VecAdd<V1, V2::Unwrapped>>;

    fn add(self, rhs: V2) -> Self::Output {
        VectorOps::add(self,rhs)
    }
}

*/


// TODO: remove?
pub trait VectorVectorOps<V: VectorOps>: VectorOps {
    type DoubleWrapped<T: VectorLike>;

    fn assert_eq_len(&self, other: &V);
    unsafe fn double_wrap<T: VectorLike>(vec: T) -> Self::DoubleWrapped<T>;

    #[inline] fn zip(self, other: V) -> Self::DoubleWrapped<VecZip<Self::Unwrapped, V::Unwrapped>> 
    where
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool, <V::Unwrapped as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool, <V::Unwrapped as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndHandleBool, <V::Unwrapped as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::SndOwnedBufferBool, <V::Unwrapped as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        Self: Sized,
    {
        self.assert_eq_len(&other);
        unsafe { Self::double_wrap(VecZip { l_vec: self.unwrap(), r_vec: other.unwrap() }) }
    }

    #[inline] fn add(self, other: V) -> Self::DoubleWrapped<VecAdd<Self::Unwrapped, V::Unwrapped>> 
    where
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
        self.assert_eq_len(&other);
        unsafe { Self::double_wrap(VecAdd { l_vec: self.unwrap(), r_vec: other.unwrap() }) }
    }

    #[inline] fn sub(self, other: V) -> Self::DoubleWrapped<VecSub<Self::Unwrapped, V::Unwrapped>> 
    where
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
        self.assert_eq_len(&other);
        unsafe { Self::double_wrap(VecSub { l_vec: self.unwrap(), r_vec: other.unwrap() }) }
    }

    #[inline] fn comp_mul(self, other: V) -> Self::DoubleWrapped<VecCompMul<Self::Unwrapped, V::Unwrapped>> 
    where
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
        self.assert_eq_len(&other);
        unsafe { Self::double_wrap(VecCompMul { l_vec: self.unwrap(), r_vec: other.unwrap() }) }
    }

    #[inline] fn comp_div(self, other: V) -> Self::DoubleWrapped<VecCompDiv<Self::Unwrapped, V::Unwrapped>> 
    where
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
        self.assert_eq_len(&other);
        unsafe { Self::double_wrap(VecCompDiv { l_vec: self.unwrap(), r_vec: other.unwrap() }) }
    }

    #[inline] fn comp_rem(self, other: V) -> Self::DoubleWrapped<VecCompRem<Self::Unwrapped, V::Unwrapped>> 
    where
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
        self.assert_eq_len(&other);
        unsafe { Self::double_wrap(VecCompRem { l_vec: self.unwrap(), r_vec: other.unwrap() }) }
    }

    #[inline] fn add_assign<'a, I: 'a + AddAssign<<V::Unwrapped as Get>::Item>>(self, other: V) -> Self::DoubleWrapped<VecAddAssign<'a, Self::Unwrapped, V::Unwrapped, I>> 
    where
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
        self.assert_eq_len(&other);
        unsafe { Self::double_wrap(VecAddAssign { l_vec: self.unwrap(), r_vec: other.unwrap() }) }
    }

    #[inline] fn sub_assign<'a, I: 'a + SubAssign<<V::Unwrapped as Get>::Item>>(self, other: V) -> Self::DoubleWrapped<VecSubAssign<'a, Self::Unwrapped, V::Unwrapped, I>> 
    where
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
        self.assert_eq_len(&other);
        unsafe { Self::double_wrap(VecSubAssign { l_vec: self.unwrap(), r_vec: other.unwrap() }) }
    }

    #[inline] fn comp_mul_assign<'a, I: 'a + MulAssign<<V::Unwrapped as Get>::Item>>(self, other: V) -> Self::DoubleWrapped<VecCompMulAssign<'a, Self::Unwrapped, V::Unwrapped, I>> 
    where
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
        self.assert_eq_len(&other);
        unsafe { Self::double_wrap(VecCompMulAssign { l_vec: self.unwrap(), r_vec: other.unwrap() }) }
    }

    #[inline] fn comp_div_assign<'a, I: 'a + DivAssign<<V::Unwrapped as Get>::Item>>(self, other: V) -> Self::DoubleWrapped<VecCompDivAssign<'a, Self::Unwrapped, V::Unwrapped, I>> 
    where
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
        self.assert_eq_len(&other);
        unsafe { Self::double_wrap(VecCompDivAssign { l_vec: self.unwrap(), r_vec: other.unwrap() }) }
    }

    #[inline] fn comp_rem_assign<'a, I: 'a + RemAssign<<V::Unwrapped as Get>::Item>>(self, other: V) -> Self::DoubleWrapped<VecCompRemAssign<'a, Self::Unwrapped, V::Unwrapped, I>> 
    where
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
        self.assert_eq_len(&other);
        unsafe { Self::double_wrap(VecCompRemAssign { l_vec: self.unwrap(), r_vec: other.unwrap() }) }
    }

    #[inline] fn dot<S: std::iter::Sum<<<Self::Unwrapped as Get>::Item as Mul<<V::Unwrapped as Get>::Item>>::Output> + AddAssign<<<Self::Unwrapped as Get>::Item as Mul<<V::Unwrapped as Get>::Item>>::Output>>(self, other: V) -> Self::DoubleWrapped<VecDot<Self::Unwrapped, V::Unwrapped, S>> 
    where
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
        self.assert_eq_len(&other);
        unsafe { Self::double_wrap(VecDot { l_vec: self.unwrap(), r_vec: other.unwrap(), scalar: std::mem::ManuallyDrop::new(NoneIter::new().sum())}) }
    }

    #[inline] fn initialized_dot<S: AddAssign<<<Self::Unwrapped as Get>::Item as Mul<<V::Unwrapped as Get>::Item>>::Output>>(self, other: V, init: S) -> Self::DoubleWrapped<VecDot<Self::Unwrapped, V::Unwrapped, S>> 
    where
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
        self.assert_eq_len(&other);
        unsafe { Self::double_wrap(VecDot { l_vec: self.unwrap(), r_vec: other.unwrap(), scalar: std::mem::ManuallyDrop::new(init)}) }
    }
}

