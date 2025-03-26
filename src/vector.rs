use self::{vec_util_traits::{Get, HasReuseBuf, VectorLike}, vector_structs::OwnedArray};
use crate::{trait_specialization_utils::*, util_structs::NoneIter, util_traits::{HasOutput, IsRepeatable}};
use std::{mem::ManuallyDrop, ops::*};

pub mod vec_util_traits {
    // Note: traits here aren't meant to be used directly by end users

    use crate::trait_specialization_utils::*;
    use crate::util_traits::HasOutput;

    /// A way to get out items from a collection / generator which implicitly invalidates* that index
    /// Can output owned values
    /// *:  if IsRepeatable = Y, indices aren't actually invalidated so it is legal to call get at an index twice or more
    ///     note: IsRepeatable also implies that there is no bound item
    pub unsafe trait Get { 
        type GetBool: TyBool;
        type Inputs;
        type Item;
        type BoundItems;

        unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs; 

        unsafe fn drop_inputs(&mut self, index: usize);

        fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems);

        /// Note: generally not used to better manage dropping, may be removed in the future
        #[inline]
        unsafe fn get(&mut self, index: usize) -> (Self::Item, Self::BoundItems) { unsafe {
            let inputs = self.get_inputs(index);
            self.process(inputs)
        }}
    }

    pub trait HasReuseBuf {
        type FstHandleBool: TyBool;
        type SndHandleBool: TyBool;
        type BoundHandlesBool: TyBool;
        type FstOwnedBufferBool: TyBool;
        type SndOwnedBufferBool: TyBool;
        type FstOwnedBuffer;
        type SndOwnedBuffer;
        type FstType;
        type SndType;
        type BoundTypes;

        unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType); 
        unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType);
        unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes);
        unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer;
        unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer;
        unsafe fn drop_1st_buf_index(&mut self, index: usize);
        unsafe fn drop_2nd_buf_index(&mut self, index: usize);
        unsafe fn drop_bound_bufs_index(&mut self, index: usize);
    }

    pub trait VectorWrapperBuilder: Clone {
        type Wrapped<T: VectorLike>;

        ///Safety: The VectorLike passed to this function MUST match the implications of the wrapper (ATM (Oct 2024), just needs to be unused)
        unsafe fn wrap<T: VectorLike>(&self, vec: T) -> Self::Wrapped<T>;
    }

    ///really just a shorthand for the individual traits
    pub trait VectorLike: Get + HasOutput + HasReuseBuf {}

    impl<T: Get + HasOutput + HasReuseBuf> VectorLike for T {}
}


pub mod vector_structs;
use vec_util_traits::VectorWrapperBuilder;
use vector_structs::*;

#[repr(transparent)]
pub struct VectorExpr<T: VectorLike,const D: usize>(pub(crate) T); // note: VectorExpr only holds fully unused VectorLike objects

impl<T: VectorLike,const D: usize> VectorExpr<T,D> {
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
    >>,D> where T: 'static {
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

    #[inline] 
    pub fn consume(self) -> T::Output where T: HasReuseBuf<BoundTypes = T::BoundItems> {
        VectorIter::<T,D>{
            vec: unsafe { std::ptr::read(&std::mem::ManuallyDrop::new(self).0) },
            live_input_start: 0,
            dead_output_start: 0,
        }.consume()
    }

    #[inline]
    pub fn eval(self) -> <VecBind<VecMaybeCreateBuf<T,T::Item,D>> as HasOutput>::Output 
    where 
        <T::FstHandleBool as TyBool>::Neg: Filter,
        (T::FstHandleBool, <T::FstHandleBool as TyBool>::Neg): SelectPair,
        (T::FstOwnedBufferBool,<T::FstHandleBool as TyBool>::Neg): TyBoolPair,
        (T::OutputBool,<(T::FstOwnedBufferBool,<T::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or): FilterPair,
        (T::BoundHandlesBool,Y): FilterPair,
        VecBind<VecMaybeCreateBuf<T,T::Item,D>>: HasReuseBuf<BoundTypes = <VecBind<VecMaybeCreateBuf<T,T::Item,D>> as Get>::BoundItems>
    {
        self.maybe_create_buf().bind().consume()
    }

    #[inline]
    pub fn heap_eval(self) -> <VecBind<VecMaybeCreateHeapBuf<T,T::Item,D>> as HasOutput>::Output 
    where 
        <T::FstHandleBool as TyBool>::Neg: Filter,
        (T::FstHandleBool, <T::FstHandleBool as TyBool>::Neg): SelectPair,
        (T::FstOwnedBufferBool,<T::FstHandleBool as TyBool>::Neg): TyBoolPair,
        (T::OutputBool,<(T::FstOwnedBufferBool,<T::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or): FilterPair,
        (T::BoundHandlesBool,Y): FilterPair,
        VecBind<VecMaybeCreateHeapBuf<T,T::Item,D>>: HasReuseBuf<BoundTypes = <VecBind<VecMaybeCreateHeapBuf<T,T::Item,D>> as Get>::BoundItems>
    {
        self.maybe_create_heap_buf().bind().consume()
    }
}

impl<V: VectorLike + IsRepeatable,const D: usize> VectorExpr<V,D> {
    /// Note:   This method does NOT fill any buffers bound to the vector, if you need that, see binding_get
    pub fn get(&mut self,index: usize) -> V::Item {
        // the nature of IsRepeatable means that any index can be called any number of times so this is fine
        unsafe {
            let inputs = self.0.get_inputs(index);
            let (item,_) = self.0.process(inputs);
            item
        }
    }

    /// Note:   Some buffers do not drop pre-existing values when being filled as such values may be undefined data
    ///         however, this means that binding an index multiple times can cause a leak (ie. with Box<T>'s being bound)
    ///         Additionally, if the buffer is owned by the vector, the vector expr is also responsible for dropping filled indices
    ///         however, such filled indices filled via this method aren't tracked so further leaks can happen 
    ///         (assuming it isn't retroactivly noted as filled during evaluation/iteration)
    /// Note TLDR: this method is extremely prone to causing memory leaks
    pub fn binding_get(&mut self,index: usize) -> V::Item where V: HasReuseBuf<BoundTypes = V::BoundItems> {
        // the nature of IsRepeatable means that any index can be called any number of times so this is fine
        unsafe {
            let inputs = self.0.get_inputs(index);
            let (item,bound_items) = self.0.process(inputs);
            self.0.assign_bound_bufs(index, bound_items); // NOTE: all current things which have IsRepeatable don't have any bound items, however, it is not restricted by the definition
            item
        }
    }
}

impl<T: VectorLike,const D: usize> Drop for VectorExpr<T,D> {
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

impl<T: VectorLike,const D: usize> IntoIterator for VectorExpr<T,D> where T: HasReuseBuf<BoundTypes = T::BoundItems> {
    type IntoIter = VectorIter<T,D>;
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



#[derive(Clone)]
pub struct VectorExprBuilder<const D: usize>;

impl<const D: usize> VectorWrapperBuilder for VectorExprBuilder<D> {
    type Wrapped<T: VectorLike> = VectorExpr<T,D>;
    
    unsafe fn wrap<T: VectorLike>(&self, vec: T) -> Self::Wrapped<T> {VectorExpr(vec)}
}

pub struct VectorIter<T: VectorLike,const D: usize>{vec: T, live_input_start: usize, dead_output_start: usize} // note: ranges are start inclusive, end exclusive

impl<T: VectorLike,const D: usize> VectorIter<T,D> {
    #[inline]
    pub unsafe fn next_unchecked(&mut self) -> T::Item where T: HasReuseBuf<BoundTypes = T::BoundItems> { unsafe {
        let index = self.live_input_start;
        self.live_input_start += 1;
        let inputs = self.vec.get_inputs(index);
        let (item,bound_items) = self.vec.process(inputs);
        self.vec.assign_bound_bufs(index,bound_items);
        self.dead_output_start += 1;
        item
    }}

    #[inline]
    pub unsafe fn unchecked_output(self) -> T::Output {
        let mut man_drop_self = std::mem::ManuallyDrop::new(self);
        let output; 
        unsafe { 
            output = man_drop_self.vec.output(); 
            std::ptr::drop_in_place(&mut man_drop_self.vec);
        }
        output
    }

    #[inline]
    pub fn output(self) -> T::Output {
        assert!(self.live_input_start == D,"math_vector error: A VectorIter must be fully used before outputting");
        debug_assert!(self.dead_output_start == D,"math_vector internal error: A VectorIter's output buffers (somehow) weren't fully filled despite the inputs being fully used, likely an internal issue");
        unsafe {self.unchecked_output()}
    }

    #[inline]
    pub fn consume(mut self) -> T::Output where T: HasReuseBuf<BoundTypes = T::BoundItems> {
        self.no_output_consume();
        unsafe {self.unchecked_output()} // safety: VectorIter was fully used
    }

    #[inline]
    pub fn no_output_consume(&mut self) where T: HasReuseBuf<BoundTypes = T::BoundItems> {
        while self.live_input_start < D {
            unsafe { let _ = self.next_unchecked(); }
        }
    }
}

impl<T: VectorLike,const D: usize> Drop for VectorIter<T,D> {
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

impl<T: VectorLike,const D: usize> Iterator for VectorIter<T,D> where T: HasReuseBuf<BoundTypes = T::BoundItems> {
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
        (size,Some(size))
    }
}

impl<T: VectorLike,const D: usize> ExactSizeIterator for VectorIter<T,D> where T: HasReuseBuf<BoundTypes = T::BoundItems> {}

impl<T: VectorLike,const D: usize> std::iter::FusedIterator for VectorIter<T,D> where T: HasReuseBuf<BoundTypes = T::BoundItems> {}


pub type MathVector<T,const D: usize> = VectorExpr<OwnedArray<T,D>,D>;

impl<T,const D: usize> MathVector<T,D> {
    #[inline] pub fn into_array(self) -> [T; D] {self.unwrap().unwrap()}
    #[inline] pub fn into_heap_array(self: Box<Self>) -> Box<[T; D]> {
        unsafe { std::mem::transmute::<Box<Self>,Box<[T; D]>>(self) }
    }
    #[inline] pub fn reuse(self) -> VectorExpr<ReplaceArray<T,D>,D> {VectorExpr(ReplaceArray(self.unwrap().0))}
    #[inline] pub fn heap_reuse(self: Box<Self>) -> VectorExpr<ReplaceHeapArray<T,D>,D> {
        unsafe { VectorExpr(ReplaceHeapArray(std::mem::transmute::<Box<Self>,std::mem::ManuallyDrop<Box<[T; D]>>>(self))) }
    }

    #[inline] pub fn referred<'a>(self) -> VectorExpr<ReferringOwnedArray<'a,T,D>,D> where T: 'a {
        VectorExpr(ReferringOwnedArray(unsafe {std::mem::transmute_copy::<ManuallyDrop<[T; D]>,[T; D]>(&self.unwrap().0)},std::marker::PhantomData)) //FIXME: unecessary transmute copy to get the compiler to not complain
    }

    #[inline] pub unsafe fn get_unchecked<I: std::slice::SliceIndex<[T]>>(&self, index: I) -> &I::Output { unsafe {
        self.0.0.get_unchecked(index)
    }}

    #[inline] pub unsafe fn get_unchecked_mut<I: std::slice::SliceIndex<[T]>>(&mut self, index: I) -> &mut I::Output { unsafe {
        self.0.0.get_unchecked_mut(index)
    }}
}

impl<T: Clone,const D: usize> Clone for MathVector<T,D> {
    #[inline]
    fn clone(&self) -> Self {
        VectorExpr(self.0.clone())
    }
}

impl<T,const D: usize> Deref for MathVector<T,D> {
    type Target = [T; D];

    fn deref(&self) -> &Self::Target {
        &self.0.0
    }
}

impl<T,const D: usize> DerefMut for MathVector<T,D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0.0
    }
}

impl<T,const D: usize> From<[T; D]> for MathVector<T,D> {
    #[inline] 
    fn from(value: [T; D]) -> Self {
        VectorExpr(OwnedArray(std::mem::ManuallyDrop::new(value)))
    }
}

impl<T,const D: usize> Into<[T; D]> for MathVector<T,D> {
    #[inline] fn into(self) -> [T; D] {self.into_array()}
}

impl<'a,T,const D: usize> From<&'a [T; D]> for &'a MathVector<T,D> {
    #[inline]
    fn from(value: &'a [T; D]) -> Self {
        unsafe { std::mem::transmute::<&'a [T; D],&'a MathVector<T,D>>(value) }
    }
} 

impl<'a,T,const D: usize> Into<&'a [T; D]> for &'a MathVector<T,D> {
    #[inline]
    fn into(self) -> &'a [T; D] {
        unsafe { std::mem::transmute::<&'a MathVector<T,D>,&'a [T; D]>(self) }
    }
} 

impl<'a,T,const D: usize> From<&'a mut [T; D]> for &'a mut MathVector<T,D> {
    #[inline]
    fn from(value: &'a mut [T; D]) -> Self {
        unsafe { std::mem::transmute::<&'a mut [T; D],&'a mut MathVector<T,D>>(value) }
    }
} 

impl<'a,T,const D: usize> Into<&'a mut [T; D]> for &'a mut MathVector<T,D> {
    #[inline]
    fn into(self) -> &'a mut [T; D] {
        unsafe { std::mem::transmute::<&'a mut MathVector<T,D>,&'a mut [T; D]>(self) }
    }
} 

impl<T,I,const D: usize> Index<I> for MathVector<T,D> where [T; D]: Index<I> {
    type Output = <[T; D] as Index<I>>::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        &self.0.0[index]
    }
}

impl<T,I,const D: usize> IndexMut<I> for MathVector<T,D> where [T; D]: IndexMut<I> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0.0[index]
    }
}


impl<T1: AddAssign<T2>,T2,const D: usize> AddAssign<MathVector<T2,D>> for MathVector<T1,D> {
    #[inline]
    fn add_assign(&mut self, rhs: MathVector<T2,D>) {
        <&mut Self as VectorVectorOps<MathVector<T2,D>>>::add_assign(self,rhs).consume();
    }
}

impl<T1: SubAssign<T2>,T2,const D: usize> SubAssign<MathVector<T2,D>> for MathVector<T1,D> {
    #[inline]
    fn sub_assign(&mut self, rhs: MathVector<T2,D>) {
        <&mut Self as VectorVectorOps<MathVector<T2,D>>>::sub_assign(self,rhs).consume();
    }
}

impl<T: MulAssign<S>,S: Copy,const D: usize> MulAssign<S> for MathVector<T,D> {
    #[inline]
    fn mul_assign(&mut self, rhs: S) {
        <&mut Self as VectorOps>::mul_assign(self, rhs).consume();
    }
}

impl<T: DivAssign<S>,S: Copy,const D: usize> DivAssign<S> for MathVector<T,D> {
    #[inline]
    fn div_assign(&mut self, rhs: S) {
        <&mut Self as VectorOps>::div_assign(self, rhs).consume();
    }
}

impl<T: RemAssign<S>,S: Copy,const D: usize> RemAssign<S> for MathVector<T,D> {
    #[inline]
    fn rem_assign(&mut self, rhs: S) {
        <&mut Self as VectorOps>::rem_assign(self, rhs).consume();
    }
}


impl<T1: std::iter::Sum<T2> + AddAssign<T2>,T2,const D: usize> std::iter::Sum<MathVector<T2,D>> for MathVector<T1,D> {
    #[inline]
    fn sum<I: Iterator<Item = MathVector<T2,D>>>(iter: I) -> Self {
        let mut sum = vector_gen(|| std::iter::Sum::sum(NoneIter::new())).create_buf().bind().consume();
        for vec in iter {
            sum += vec;
        }
        sum
    }
}

#[inline]
pub fn vector_gen<F: FnMut() -> O,O,const D: usize>(f: F) -> VectorExpr<VecGenerator<F,O>,D> {
    VectorExpr(VecGenerator(f))
}

#[inline]
pub fn vector_index_gen<F: FnMut(usize) -> O,O,const D: usize>(f: F) -> VectorExpr<VecIndexGenerator<F,O>,D> {
    VectorExpr(VecIndexGenerator(f))
}


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

#[derive(Clone)]
pub struct RSVectorExprBuilder{size: usize}

impl VectorWrapperBuilder for RSVectorExprBuilder {
    type Wrapped<T: VectorLike> = RSVectorExpr<T>;

    unsafe fn wrap<T: VectorLike>(&self, vec: T) -> Self::Wrapped<T> {
        RSVectorExpr{vec, size: self.size}
    }
}



pub unsafe trait VectorOps {
    type Unwrapped: VectorLike;
    type WrapperBuilder: VectorWrapperBuilder;

    fn unwrap(self) -> Self::Unwrapped;
    fn get_wrapper_builder(&self) -> Self::WrapperBuilder;
    fn size(&self) -> usize;

    #[inline] 
    fn bind(self) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecBind<Self::Unwrapped>> where 
        Self::Unwrapped:  VectorLike<FstHandleBool = Y>,
        (<Self::Unwrapped as HasOutput>::OutputBool,<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool,Y): FilterPair,
        VecBind<Self::Unwrapped>: HasReuseBuf<BoundTypes = <VecBind<Self::Unwrapped> as Get>::BoundItems>,
        Self: Sized
    {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecBind{vec: self.unwrap()}) }
    }

    #[inline]
    fn map_bind<F: FnMut(<Self::Unwrapped as Get>::Item) -> (I,B),I,B>(self, f: F) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecMapBind<Self::Unwrapped,F,I,B>> where 
        Self::Unwrapped:  VectorLike<FstHandleBool = Y>,
        (<Self::Unwrapped as HasOutput>::OutputBool,<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool,Y): FilterPair,
        VecBind<Self::Unwrapped>: HasReuseBuf<BoundTypes = <VecBind<Self::Unwrapped> as Get>::BoundItems>,
        Self: Sized
    {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecMapBind{vec: self.unwrap(), f})}
    }

    #[inline] 
    fn half_bind(self) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecHalfBind<Self::Unwrapped>> where
        Self::Unwrapped:  VectorLike<FstHandleBool = Y>,
        (<Self::Unwrapped as HasOutput>::OutputBool,<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): FilterPair,
        (<Self::Unwrapped as HasReuseBuf>::BoundHandlesBool,Y): FilterPair,
        VecHalfBind<Self::Unwrapped>: HasReuseBuf<BoundTypes = <VecBind<Self::Unwrapped> as Get>::BoundItems>,
        Self: Sized
    {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecHalfBind{vec: self.unwrap()}) }
    }

    #[inline] fn buf_swap(self) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecBufSwap<Self::Unwrapped>> where Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecBufSwap{vec: self.unwrap()}) }
    }

    #[inline] 
    fn offset(self,offset: usize) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecOffset<Self::Unwrapped>> where Self: Sized {
        let size = self.size();
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecOffset{vec: self.unwrap(), offset: offset % size, size}) }
    }


    #[inline] fn map<F: FnMut(<Self::Unwrapped as Get>::Item) -> O,O>(self, f: F) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecMap<Self::Unwrapped,F,O>> where (<Self::Unwrapped as HasOutput>::OutputBool,N): FilterPair, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecMap{vec: self.unwrap(), f}) }
    } 

    //note: fold_ref should be used whenever possible due to implementation
    #[inline] fn fold<F: FnMut(O,<Self::Unwrapped as Get>::Item) -> O,O>(self, f: F, init: O) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecFold<Self::Unwrapped,F,O>> where (<Self::Unwrapped as HasOutput>::OutputBool,Y): FilterPair, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecFold{vec: self.unwrap(), f, cell: Some(init)}) }
    }

    #[inline] fn fold_ref<F: FnMut(&mut O,<Self::Unwrapped as Get>::Item),O>(self, f: F, init: O) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecFoldRef<Self::Unwrapped,F,O>> where (<Self::Unwrapped as HasOutput>::OutputBool,Y): FilterPair, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecFoldRef{vec: self.unwrap(), f, cell: std::mem::ManuallyDrop::new(init)}) }
    }

    #[inline] fn copied<'a,I: Copy>(self) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecCopy<'a,Self::Unwrapped,I>> where (<Self::Unwrapped as HasOutput>::OutputBool,N): FilterPair, Self::Unwrapped: Get<Item = &'a I>, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecCopy{vec: self.unwrap()}) }
    }

    #[inline] fn cloned<'a,I: Clone>(self) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecClone<'a,Self::Unwrapped,I>> where (<Self::Unwrapped as HasOutput>::OutputBool,N): FilterPair, Self::Unwrapped: Get<Item = &'a I>, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecClone{vec: self.unwrap()}) }
    }

    #[inline] fn neg(self) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecNeg<Self::Unwrapped>> where (<Self::Unwrapped as HasOutput>::OutputBool,N): FilterPair, <Self::Unwrapped as Get>::Item: Neg, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecNeg{vec: self.unwrap()}) }
    }

    #[inline] fn mul_r<S: Mul<<Self::Unwrapped as Get>::Item> + Copy>(self,scalar: S) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecMulR<Self::Unwrapped,S>> where (<Self::Unwrapped as HasOutput>::OutputBool,N): FilterPair, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecMulR{vec: self.unwrap(), scalar}) }
    }

    #[inline] fn div_r<S: Div<<Self::Unwrapped as Get>::Item> + Copy>(self,scalar: S) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecDivR<Self::Unwrapped,S>> where (<Self::Unwrapped as HasOutput>::OutputBool,N): FilterPair, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecDivR{vec: self.unwrap(), scalar}) }
    }

    #[inline] fn rem_r<S: Rem<<Self::Unwrapped as Get>::Item> + Copy>(self,scalar: S) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecRemR<Self::Unwrapped,S>> where (<Self::Unwrapped as HasOutput>::OutputBool,N): FilterPair, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecRemR{vec: self.unwrap(), scalar}) }
    }

    #[inline] fn mul_l<S: Copy>(self,scalar: S) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecMulL<Self::Unwrapped,S>> where (<Self::Unwrapped as HasOutput>::OutputBool,N): FilterPair, <Self::Unwrapped as Get>::Item: Mul<S>, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecMulL{vec: self.unwrap(), scalar})}
    }

    #[inline] fn div_l<S: Copy>(self,scalar: S) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecDivL<Self::Unwrapped,S>> where (<Self::Unwrapped as HasOutput>::OutputBool,N): FilterPair, <Self::Unwrapped as Get>::Item: Div<S>, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecDivL{vec: self.unwrap(), scalar})}
    }

    #[inline] fn rem_l<S: Copy>(self,scalar: S) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecRemL<Self::Unwrapped,S>> where (<Self::Unwrapped as HasOutput>::OutputBool,N): FilterPair, <Self::Unwrapped as Get>::Item: Rem<S>, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecRemL{vec: self.unwrap(), scalar})}
    }

    #[inline] fn mul_assign<'a,I: 'a + MulAssign<S>,S: Copy>(self,scalar: S) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecMulAssign<'a,Self::Unwrapped,I,S>> where (<Self::Unwrapped as HasOutput>::OutputBool,N): FilterPair, Self::Unwrapped: Get<Item = &'a mut I>, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecMulAssign{vec: self.unwrap(), scalar})}
    }

    #[inline] fn div_assign<'a,I: 'a + DivAssign<S>,S: Copy>(self,scalar: S) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecDivAssign<'a,Self::Unwrapped,I,S>> where (<Self::Unwrapped as HasOutput>::OutputBool,N): FilterPair, Self::Unwrapped: Get<Item = &'a mut I>, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecDivAssign{vec: self.unwrap(), scalar})}
    }

    #[inline] fn rem_assign<'a,I: 'a + RemAssign<S>,S: Copy>(self,scalar: S) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecRemAssign<'a,Self::Unwrapped,I,S>> where (<Self::Unwrapped as HasOutput>::OutputBool,N): FilterPair, Self::Unwrapped: Get<Item = &'a mut I>, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecRemAssign{vec: self.unwrap(), scalar})}
    }

    #[inline] fn sum<S: std::iter::Sum<<Self::Unwrapped as Get>::Item> + AddAssign<<Self::Unwrapped as Get>::Item>>(self) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecSum<Self::Unwrapped,S>> where (<Self::Unwrapped as HasOutput>::OutputBool,Y): FilterPair, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecSum{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(NoneIter::new().sum())})}
    }

    #[inline] fn initialized_sum<S: AddAssign<<Self::Unwrapped as Get>::Item>>(self,init: S) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecSum<Self::Unwrapped,S>> where (<Self::Unwrapped as HasOutput>::OutputBool,Y): FilterPair, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecSum{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(init)})}
    }

    #[inline] fn product<S: std::iter::Product<<Self::Unwrapped as Get>::Item> + MulAssign<<Self::Unwrapped as Get>::Item>>(self) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecProduct<Self::Unwrapped,S>> where (<Self::Unwrapped as HasOutput>::OutputBool,Y): FilterPair, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecProduct{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(NoneIter::new().product())})}
    }

    #[inline] fn sqr_mag<S: std::iter::Sum<<Self::Unwrapped as Get>::Item> + AddAssign<<<Self::Unwrapped as Get>::Item as Mul>::Output>>(self) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecSqrMag<Self::Unwrapped,S>> where (<Self::Unwrapped as HasOutput>::OutputBool,Y): FilterPair, <Self::Unwrapped as Get>::Item: Copy + Mul, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecSqrMag{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(NoneIter::new().sum())})}
    }

    #[inline] fn initialized_product<S: MulAssign<<Self::Unwrapped as Get>::Item>>(self,init: S) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecProduct<Self::Unwrapped,S>> where (<Self::Unwrapped as HasOutput>::OutputBool,Y): FilterPair, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecProduct{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(init)})}
    }

    #[inline] fn initialized_sqr_mag<S: AddAssign<<<Self::Unwrapped as Get>::Item as Mul>::Output>>(self,init: S) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecSqrMag<Self::Unwrapped,S>> where (<Self::Unwrapped as HasOutput>::OutputBool,Y): FilterPair, <Self::Unwrapped as Get>::Item: Copy + Mul, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecSqrMag{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(init)})}
    }

    #[inline] fn copied_sum<S: std::iter::Sum<<Self::Unwrapped as Get>::Item> + AddAssign<<Self::Unwrapped as Get>::Item>>(self) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecCopiedSum<Self::Unwrapped,S>> where (<Self::Unwrapped as HasOutput>::OutputBool,Y): FilterPair, <Self::Unwrapped as Get>::Item: Copy, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecCopiedSum{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(NoneIter::new().sum())})}
    }

    #[inline] fn copied_product<S: std::iter::Product<<Self::Unwrapped as Get>::Item> + MulAssign<<Self::Unwrapped as Get>::Item>>(self) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecCopiedProduct<Self::Unwrapped,S>> where (<Self::Unwrapped as HasOutput>::OutputBool,Y): FilterPair, <Self::Unwrapped as Get>::Item: Copy, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecCopiedProduct{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(NoneIter::new().product())})}
    }

    #[inline] fn copied_sqr_mag<S: std::iter::Sum<<Self::Unwrapped as Get>::Item> + AddAssign<<<Self::Unwrapped as Get>::Item as Mul>::Output>>(self) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecCopiedSqrMag<Self::Unwrapped,S>> where (<Self::Unwrapped as HasOutput>::OutputBool,Y): FilterPair, <Self::Unwrapped as Get>::Item: Copy + Mul, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecCopiedSqrMag{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(NoneIter::new().sum())})}
    }

    #[inline] fn initialized_copied_sum<S: AddAssign<<Self::Unwrapped as Get>::Item>>(self,init: S) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecCopiedSum<Self::Unwrapped,S>> where (<Self::Unwrapped as HasOutput>::OutputBool,Y): FilterPair, <Self::Unwrapped as Get>::Item: Copy, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecCopiedSum{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(init)})}
    }

    #[inline] fn initialized_copied_product<S: MulAssign<<Self::Unwrapped as Get>::Item>>(self,init: S) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecCopiedProduct<Self::Unwrapped,S>> where (<Self::Unwrapped as HasOutput>::OutputBool,Y): FilterPair, <Self::Unwrapped as Get>::Item: Copy, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecCopiedProduct{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(init)})}
    }

    #[inline] fn initialized_copied_sqr_mag<S: AddAssign<<<Self::Unwrapped as Get>::Item as Mul>::Output>>(self,init: S) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecCopiedSqrMag<Self::Unwrapped,S>> where (<Self::Unwrapped as HasOutput>::OutputBool,Y): FilterPair, <Self::Unwrapped as Get>::Item: Copy + Mul, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe { builder.wrap(VecCopiedSqrMag{vec: self.unwrap(), scalar: std::mem::ManuallyDrop::new(init)})}
    }
}

pub trait ArrayVectorOps<const D: usize>: VectorOps {
    // note: due to current borrow checker limitations surrounding for<'a>, this isn't very useful in reality
    #[inline]
    fn attach_buf<'a,T>(self,buf: &'a mut MathVector<T,D>) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecAttachBuf<'a,Self::Unwrapped,T,D>> where Self::Unwrapped: HasReuseBuf<FstHandleBool = N>, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe{ builder.wrap(VecAttachBuf{vec: self.unwrap(),buf}) }
    }

    #[inline] 
    fn create_buf<T>(self) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecCreateBuf<Self::Unwrapped,T,D>> where Self::Unwrapped: HasReuseBuf<FstHandleBool = N>, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe{ builder.wrap(VecCreateBuf{vec: self.unwrap(),buf: std::mem::MaybeUninit::uninit().assume_init()}) }
    }

    #[inline] 
    fn create_heap_buf<T>(self) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecCreateHeapBuf<Self::Unwrapped,T,D>> where Self::Unwrapped: HasReuseBuf<FstHandleBool = N>, Self: Sized {
        let builder = self.get_wrapper_builder();
        unsafe{ builder.wrap(VecCreateHeapBuf{vec: self.unwrap(),buf: std::mem::ManuallyDrop::new(Box::new(std::mem::MaybeUninit::uninit().assume_init()))}) }
    }

    #[inline] 
    fn maybe_create_buf<T>(self) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecMaybeCreateBuf<Self::Unwrapped,T,D>> 
    where 
        <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool,<<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool,<<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg): TyBoolPair,
        Self: Sized
    {
        let builder = self.get_wrapper_builder();
        unsafe{ builder.wrap(VecMaybeCreateBuf{vec: self.unwrap(),buf: std::mem::MaybeUninit::uninit().assume_init()}) }
    }

    #[inline] 
    fn maybe_create_heap_buf<T>(self) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecMaybeCreateHeapBuf<Self::Unwrapped,T,D>> 
    where 
        <<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
        (<Self::Unwrapped as HasReuseBuf>::FstHandleBool,<<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg): SelectPair,
        (<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool,<<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg): TyBoolPair,
        Self: Sized
    {
        let builder = self.get_wrapper_builder();
        unsafe{ builder.wrap(VecMaybeCreateHeapBuf{vec: self.unwrap(),buf: std::mem::ManuallyDrop::new(Box::new(std::mem::MaybeUninit::uninit().assume_init()))}) }
    }
}

pub trait RepeatableVectorOps: VectorOps {
    type RepeatableVector<'a>: VectorLike + IsRepeatable where Self: 'a;
    type UsedVector: VectorLike;
    //type HeapedUsedVector: VectorLike;

    fn make_repeatable<'a>(self) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecAttachUsedVec<Self::RepeatableVector<'a>,Self::UsedVector>> 
    where
        Self: 'a,
        (<Self::RepeatableVector<'a> as HasOutput>::OutputBool,<Self::UsedVector as HasOutput>::OutputBool): FilterPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::FstHandleBool,<Self::UsedVector as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::SndHandleBool,<Self::UsedVector as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::BoundHandlesBool,<Self::UsedVector as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::FstOwnedBufferBool,<Self::UsedVector as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::SndOwnedBufferBool,<Self::UsedVector as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg,<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): TyBoolPair,
        <(<<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg,<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool) as TyBoolPair>::Or: IsTrue
    ;
}

macro_rules! if_lifetimes {
    (($item:item); $($lifetime:lifetime),+) => {$item};
    (($item:item); ) => {}
}

macro_rules! overload_operators {
    (
        <$($($lifetime:lifetime),+,)? $($generic:ident $(:)? $($lifetime_bound:lifetime |)? $($fst_trait_bound:path $(| $trait_bound:path)*)?),+,{$size:ident}>,
        $ty:ty,
        vector: $vector:ty,
        item: $item:ty
    ) => {
        impl<$($($lifetime),+,)? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: Copy, const $size: usize> Mul<Z> for $ty where (<$vector as HasOutput>::OutputBool,N): FilterPair, $item: Mul<Z>, Self: Sized {
            type Output = VectorExpr<VecMulL<<$ty as VectorOps>::Unwrapped,Z>,$size>;
    
            #[inline]
            fn mul(self, rhs: Z) -> Self::Output {
                self.mul_l(rhs)
            }
        }

        impl<$($($lifetime),+,)? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: Copy, const $size: usize> Div<Z> for $ty where (<$vector as HasOutput>::OutputBool,N): FilterPair, $item: Div<Z>, Self: Sized {
            type Output = VectorExpr<VecDivL<<$ty as VectorOps>::Unwrapped,Z>,$size>;
    
            #[inline]
            fn div(self, rhs: Z) -> Self::Output {
                self.div_l(rhs)
            }
        }

        impl<$($($lifetime),+,)? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: Copy, const $size: usize> Rem<Z> for $ty where (<$vector as HasOutput>::OutputBool,N): FilterPair, $item: Rem<Z>, Self: Sized {
            type Output = VectorExpr<VecRemL<<$ty as VectorOps>::Unwrapped,Z>,$size>;
    
            #[inline]
            fn rem(self, rhs: Z) -> Self::Output {
                self.rem_l(rhs)
            }
        }

        if_lifetimes!((
            impl<$($($lifetime),+,)? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: AddAssign<$item>, const $size: usize> AddAssign<$ty> for MathVector<Z,D> 
            where
                (N,<$vector as HasOutput>::OutputBool): FilterPair,
                (N,<$vector as HasReuseBuf>::FstHandleBool): SelectPair,
                (N,<$vector as HasReuseBuf>::SndHandleBool): SelectPair,
                (N,<$vector as HasReuseBuf>::BoundHandlesBool): SelectPair,
                (N,<$vector as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
                (N,<$vector as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
                (<(N,<$vector as HasOutput>::OutputBool) as TyBoolPair>::Or,N): FilterPair,
            {
                #[inline]
                fn add_assign(&mut self, rhs: $ty) {
                    VectorVectorOps::add_assign(self, rhs).consume();
                }
            }
        ); $($($lifetime),+)?);

        if_lifetimes!((
            impl<$($($lifetime),+,)? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: SubAssign<$item>, const $size: usize> SubAssign<$ty> for MathVector<Z,D> 
            where
                (N,<$vector as HasOutput>::OutputBool): FilterPair,
                (N,<$vector as HasReuseBuf>::FstHandleBool): SelectPair,
                (N,<$vector as HasReuseBuf>::SndHandleBool): SelectPair,
                (N,<$vector as HasReuseBuf>::BoundHandlesBool): SelectPair,
                (N,<$vector as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
                (N,<$vector as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
                (<(N,<$vector as HasOutput>::OutputBool) as TyBoolPair>::Or,N): FilterPair,
            {
                #[inline]
                fn sub_assign(&mut self, rhs: $ty) {
                    VectorVectorOps::sub_assign(self, rhs).consume();
                }
            }
        ); $($($lifetime),+)?);
    };
}

unsafe impl<V: VectorLike,const D: usize> VectorOps for VectorExpr<V,D> {
    type Unwrapped = V;
    type WrapperBuilder = VectorExprBuilder<D>;

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
    #[inline] fn get_wrapper_builder(&self) -> Self::WrapperBuilder {VectorExprBuilder}
    #[inline] fn size(&self) -> usize {D}
}
impl<V: VectorLike,const D: usize> ArrayVectorOps<D> for VectorExpr<V,D> {}

impl<V: VectorLike,const D: usize> RepeatableVectorOps for VectorExpr<V,D> 
where 
    <V::FstHandleBool as TyBool>::Neg: Filter,
    (V::BoundHandlesBool,Y): FilterPair,
    (V::FstHandleBool,<V::FstHandleBool as TyBool>::Neg): SelectPair<
    Selected<V::FstOwnedBuffer, MathVector<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<<V as Get>::Item>, D>> = MathVector<V::Item,D>>,
    (V::FstOwnedBufferBool,<V::FstHandleBool as TyBool>::Neg): TyBoolPair,
    (<V::FstHandleBool as TyBool>::Neg,V::FstOwnedBufferBool): TyBoolPair,
    (V::OutputBool,<(V::FstOwnedBufferBool,<V::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or): FilterPair,
    VecHalfBind<VecMaybeCreateBuf<V,V::Item,D>>: HasReuseBuf<BoundTypes = <(V::BoundHandlesBool,Y) as FilterPair>::Filtered<V::BoundItems,V::Item>>
{
    type RepeatableVector<'a> = ReferringOwnedArray<'a,V::Item,D> where Self: 'a;
    type UsedVector = VecHalfBind<VecMaybeCreateBuf<V,V::Item,D>>;

    fn make_repeatable<'a>(self) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecAttachUsedVec<Self::RepeatableVector<'a>,Self::UsedVector>> 
    where
        Self: 'a,
        (<Self::RepeatableVector<'a> as HasOutput>::OutputBool,<Self::UsedVector as HasOutput>::OutputBool): FilterPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::FstHandleBool,<Self::UsedVector as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::SndHandleBool,<Self::UsedVector as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::BoundHandlesBool,<Self::UsedVector as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::FstOwnedBufferBool,<Self::UsedVector as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::SndOwnedBufferBool,<Self::UsedVector as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg,<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): TyBoolPair,
        <(<<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg,<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool) as TyBoolPair>::Or: IsTrue
    {
        let builder = self.get_wrapper_builder();
        let mut vec_iter = self.maybe_create_buf().half_bind().into_iter();
        unsafe {
            vec_iter.no_output_consume();
            builder.wrap(VecAttachUsedVec{vec: vec_iter.vec.get_bound_buf().referred().unwrap(),used_vec: std::ptr::read(&vec_iter.vec)})
        }
    }
}

overload_operators!(<V: VectorLike,{D}>, VectorExpr<V,D>, vector: V, item: V::Item);

unsafe impl<V: VectorLike,const D: usize> VectorOps for Box<VectorExpr<V,D>> {
    type Unwrapped = Box<V>;
    type WrapperBuilder = VectorExprBuilder<D>;

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
    #[inline] fn get_wrapper_builder(&self) -> Self::WrapperBuilder {VectorExprBuilder}
    #[inline] fn size(&self) -> usize {D}
}
impl<V: VectorLike,const D: usize> ArrayVectorOps<D> for Box<VectorExpr<V,D>> {}

impl<V: VectorLike,const D: usize> RepeatableVectorOps for Box<VectorExpr<V,D>>
where 
    <V::FstHandleBool as TyBool>::Neg: Filter,
    (V::BoundHandlesBool,Y): FilterPair,
    (V::FstHandleBool,<V::FstHandleBool as TyBool>::Neg): SelectPair<
    Selected<V::FstOwnedBuffer, MathVector<<<V::FstHandleBool as TyBool>::Neg as Filter>::Filtered<<V as Get>::Item>, D>> = MathVector<V::Item,D>>,
    (V::FstOwnedBufferBool,<V::FstHandleBool as TyBool>::Neg): TyBoolPair,
    (<V::FstHandleBool as TyBool>::Neg,V::FstOwnedBufferBool): TyBoolPair,
    (V::OutputBool,<(V::FstOwnedBufferBool,<V::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or): FilterPair,
    VecHalfBind<VecMaybeCreateBuf<Box<V>,V::Item,D>>: HasReuseBuf<BoundTypes = <(V::BoundHandlesBool,Y) as FilterPair>::Filtered<V::BoundItems,V::Item>>
{
    type RepeatableVector<'a> = ReferringOwnedArray<'a,V::Item,D> where Self: 'a;
    type UsedVector = VecHalfBind<VecMaybeCreateBuf<Box<V>,V::Item,D>>;

    fn make_repeatable<'a>(self) -> <Self::WrapperBuilder as VectorWrapperBuilder>::Wrapped<VecAttachUsedVec<Self::RepeatableVector<'a>,Self::UsedVector>> 
    where
        Self: 'a,
        (<Self::RepeatableVector<'a> as HasOutput>::OutputBool,<Self::UsedVector as HasOutput>::OutputBool): FilterPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::FstHandleBool,<Self::UsedVector as HasReuseBuf>::FstHandleBool): SelectPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::SndHandleBool,<Self::UsedVector as HasReuseBuf>::SndHandleBool): SelectPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::BoundHandlesBool,<Self::UsedVector as HasReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::FstOwnedBufferBool,<Self::UsedVector as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::RepeatableVector<'a> as HasReuseBuf>::SndOwnedBufferBool,<Self::UsedVector as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg,<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool): TyBoolPair,
        <(<<Self::Unwrapped as HasReuseBuf>::FstHandleBool as TyBool>::Neg,<Self::Unwrapped as HasReuseBuf>::FstOwnedBufferBool) as TyBoolPair>::Or: IsTrue
    {
        let builder = self.get_wrapper_builder();
        let mut vec_iter = self.maybe_create_buf().half_bind().into_iter();
        unsafe {
            while vec_iter.live_input_start < D {
                let _ = vec_iter.next_unchecked();
            }
            builder.wrap(VecAttachUsedVec{vec: vec_iter.vec.get_bound_buf().referred().unwrap(),used_vec: std::ptr::read(&vec_iter.vec)})
        }
    }
}

overload_operators!(<V: VectorLike,{D}>, Box<VectorExpr<V,D>>, vector: V, item: V::Item);

//already repeatable / can't truly be made repeatable so not implemented
unsafe impl<'a,T,const D: usize> VectorOps for &'a MathVector<T,D> {
    type Unwrapped = &'a [T; D];
    type WrapperBuilder = VectorExprBuilder<D>;

    #[inline] fn unwrap(self) -> Self::Unwrapped {&self.0}
    #[inline] fn get_wrapper_builder(&self) -> Self::WrapperBuilder {VectorExprBuilder}
    #[inline] fn size(&self) -> usize {D}
}
impl<'a,T,const D: usize> ArrayVectorOps<D> for &'a MathVector<T,D> {}
overload_operators!(<'a,T,{D}>, &'a MathVector<T,D>, vector: &'a [T; D], item: &'a T);

unsafe impl<'a,T,const D: usize> VectorOps for &'a mut MathVector<T,D> {
    type Unwrapped = &'a mut [T; D];
    type WrapperBuilder = VectorExprBuilder<D>;

    #[inline] fn unwrap(self) -> Self::Unwrapped {&mut self.0}
    #[inline] fn get_wrapper_builder(&self) -> Self::WrapperBuilder {VectorExprBuilder}
    #[inline] fn size(&self) -> usize {D}
}
impl<'a,T,const D: usize> ArrayVectorOps<D> for &'a mut MathVector<T,D> {}
overload_operators!(<'a,T,{D}>, &'a mut MathVector<T,D>, vector: &'a mut [T; D], item: &'a mut T);

unsafe impl<'a,T,const D: usize> VectorOps for &'a Box<MathVector<T,D>> {
    type Unwrapped = &'a [T; D];
    type WrapperBuilder = VectorExprBuilder<D>;

    #[inline] fn unwrap(self) -> Self::Unwrapped {&self.0}
    #[inline] fn get_wrapper_builder(&self) -> Self::WrapperBuilder {VectorExprBuilder}
    #[inline] fn size(&self) -> usize {D}
}
impl<'a,T,const D: usize> ArrayVectorOps<D> for &'a Box<MathVector<T,D>> {}
overload_operators!(<'a,T,{D}>, &'a Box<MathVector<T,D>>, vector: &'a [T; D], item: &'a T);

unsafe impl<'a,T,const D: usize> VectorOps for &'a mut Box<MathVector<T,D>> {
    type Unwrapped = &'a mut [T; D];
    type WrapperBuilder = VectorExprBuilder<D>;

    #[inline] fn unwrap(self) -> Self::Unwrapped {&mut self.0}
    #[inline] fn get_wrapper_builder(&self) -> Self::WrapperBuilder {VectorExprBuilder}
    #[inline] fn size(&self) -> usize {D}
}
impl<'a,T,const D: usize> ArrayVectorOps<D> for &'a mut Box<MathVector<T,D>> {}
overload_operators!(<'a,T,{D}>, &'a mut Box<MathVector<T,D>>, vector: &'a mut [T; D], item: &'a mut T);




pub trait VectorVectorOps<V: VectorOps>: VectorOps {
    type DoubleWrapped<T: VectorLike>;

    fn assert_eq_len(&self,other: &V);
    unsafe fn double_wrap<T: VectorLike>(vec: T) -> Self::DoubleWrapped<T>;

    #[inline] fn zip(self,other: V) -> Self::DoubleWrapped<VecZip<Self::Unwrapped,V::Unwrapped>> 
    where
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or,N): FilterPair,
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

    #[inline] fn add(self,other: V) -> Self::DoubleWrapped<VecAdd<Self::Unwrapped,V::Unwrapped>> 
    where
        <Self::Unwrapped as Get>::Item: Add<<V::Unwrapped as Get>::Item>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or,N): FilterPair,
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

    #[inline] fn sub(self,other: V) -> Self::DoubleWrapped<VecSub<Self::Unwrapped,V::Unwrapped>> 
    where
        <Self::Unwrapped as Get>::Item: Sub<<V::Unwrapped as Get>::Item>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or,N): FilterPair,
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

    #[inline] fn comp_mul(self,other: V) -> Self::DoubleWrapped<VecCompMul<Self::Unwrapped,V::Unwrapped>> 
    where
        <Self::Unwrapped as Get>::Item: Mul<<V::Unwrapped as Get>::Item>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or,N): FilterPair,
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

    #[inline] fn comp_div(self,other: V) -> Self::DoubleWrapped<VecCompDiv<Self::Unwrapped,V::Unwrapped>> 
    where
        <Self::Unwrapped as Get>::Item: Div<<V::Unwrapped as Get>::Item>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or,N): FilterPair,
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

    #[inline] fn comp_rem(self,other: V) -> Self::DoubleWrapped<VecCompRem<Self::Unwrapped,V::Unwrapped>> 
    where
        <Self::Unwrapped as Get>::Item: Rem<<V::Unwrapped as Get>::Item>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or,N): FilterPair,
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

    #[inline] fn add_assign<'a,I: 'a + AddAssign<<V::Unwrapped as Get>::Item>>(self,other: V) -> Self::DoubleWrapped<VecAddAssign<'a,Self::Unwrapped,V::Unwrapped,I>> 
    where
        Self::Unwrapped: Get<Item = &'a mut I>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or,N): FilterPair,
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

    #[inline] fn sub_assign<'a,I: 'a + SubAssign<<V::Unwrapped as Get>::Item>>(self,other: V) -> Self::DoubleWrapped<VecSubAssign<'a,Self::Unwrapped,V::Unwrapped,I>> 
    where
        Self::Unwrapped: Get<Item = &'a mut I>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or,N): FilterPair,
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

    #[inline] fn comp_mul_assign<'a,I: 'a + MulAssign<<V::Unwrapped as Get>::Item>>(self,other: V) -> Self::DoubleWrapped<VecCompMulAssign<'a,Self::Unwrapped,V::Unwrapped,I>> 
    where
        Self::Unwrapped: Get<Item = &'a mut I>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or,N): FilterPair,
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

    #[inline] fn comp_div_assign<'a,I: 'a + DivAssign<<V::Unwrapped as Get>::Item>>(self,other: V) -> Self::DoubleWrapped<VecCompDivAssign<'a,Self::Unwrapped,V::Unwrapped,I>> 
    where
        Self::Unwrapped: Get<Item = &'a mut I>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or,N): FilterPair,
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

    #[inline] fn comp_rem_assign<'a,I: 'a + RemAssign<<V::Unwrapped as Get>::Item>>(self,other: V) -> Self::DoubleWrapped<VecCompRemAssign<'a,Self::Unwrapped,V::Unwrapped,I>> 
    where
        Self::Unwrapped: Get<Item = &'a mut I>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or,N): FilterPair,
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

    #[inline] fn dot<S: std::iter::Sum<<<Self::Unwrapped as Get>::Item as Mul<<V::Unwrapped as Get>::Item>>::Output> + AddAssign<<<Self::Unwrapped as Get>::Item as Mul<<V::Unwrapped as Get>::Item>>::Output>>(self,other: V) -> Self::DoubleWrapped<VecDot<Self::Unwrapped,V::Unwrapped,S>> 
    where
        <Self::Unwrapped as Get>::Item: Mul<<V::Unwrapped as Get>::Item>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or,Y): FilterPair,
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

    #[inline] fn initialized_dot<S: AddAssign<<<Self::Unwrapped as Get>::Item as Mul<<V::Unwrapped as Get>::Item>>::Output>>(self,other: V,init: S) -> Self::DoubleWrapped<VecDot<Self::Unwrapped,V::Unwrapped,S>> 
    where
        <Self::Unwrapped as Get>::Item: Mul<<V::Unwrapped as Get>::Item>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <V::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or,Y): FilterPair,
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

macro_rules! impl_const_sized_double_vector_ops {
    (
        $size:ident;
        <$($($l_lifetime:lifetime),+,)? $($l_generic:ident $(:)? $($l_lifetime_bound:lifetime |)? $($l_fst_trait_bound:path $(| $l_trait_bound:path)*)?),+ $(, {$l_tt:tt})?>,
        $l_ty:ty,
        vector: $l_vector:ty,
        item: $l_item:ty;
        <$($($r_lifetime:lifetime),+,)? $($r_generic:ident $(:)? $($r_lifetime_bound:lifetime |)? $($r_fst_trait_bound:path $(| $r_trait_bound:path)*)?),+ $(, {$r_tt:tt})?>,
        $r_ty:ty,
        vector: $r_vector:ty,
        item: $r_item:ty
    ) => {
        impl<
            $($($l_lifetime),+,)? 
            $($($r_lifetime),+,)?
            $($l_generic: $($l_lifetime_bound |)? $($l_fst_trait_bound $(| $l_trait_bound)*)?),+,
            $($r_generic: $($r_lifetime_bound |)? $($r_fst_trait_bound $(| $r_trait_bound)*)?),+,
            $($l_tt,)?
            $($r_tt,)?
            const $size: usize
        > VectorVectorOps<$r_ty> for $l_ty {
            type DoubleWrapped<Z: VectorLike> = VectorExpr<Z,$size>;

            #[inline] fn assert_eq_len(&self,_: &$r_ty) {} //compile time checked, nothing necessary to assert
            #[inline] unsafe fn double_wrap<Z: VectorLike>(vec: Z) -> Self::DoubleWrapped<Z> {VectorExpr(vec)}
        }

        impl<
            $($($l_lifetime),+,)? 
            $($($r_lifetime),+,)?
            $($l_generic: $($l_lifetime_bound |)? $($l_fst_trait_bound $(| $l_trait_bound)*)?),+,
            $($r_generic: $($r_lifetime_bound |)? $($r_fst_trait_bound $(| $r_trait_bound)*)?),+,
            const $size: usize
        > Add<$r_ty> for $l_ty 
        where 
            $l_item: Add<$r_item>,
            (<$l_vector as HasOutput>::OutputBool, N): FilterPair,
            (<(<$l_vector as HasOutput>::OutputBool, N) as TyBoolPair>::Or,N): FilterPair,
            (<$l_vector as HasReuseBuf>::BoundHandlesBool, N): FilterPair,
            (<$l_vector as HasReuseBuf>::FstHandleBool, N): SelectPair,
            (<$l_vector as HasReuseBuf>::SndHandleBool, N): SelectPair,
            (<$l_vector as HasReuseBuf>::FstOwnedBufferBool, N): SelectPair,
            (<$l_vector as HasReuseBuf>::SndOwnedBufferBool, N): SelectPair,    
            (N, <$r_vector as HasOutput>::OutputBool): FilterPair,
            (<(N, <$r_vector as HasOutput>::OutputBool) as TyBoolPair>::Or,N): FilterPair,
            (N, <$r_vector as HasReuseBuf>::BoundHandlesBool): FilterPair,
            (N, <$r_vector as HasReuseBuf>::FstHandleBool): SelectPair,
            (N, <$r_vector as HasReuseBuf>::SndHandleBool): SelectPair,
            (N, <$r_vector as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
            (N, <$r_vector as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
            (<$l_vector as HasOutput>::OutputBool, <$r_vector as HasOutput>::OutputBool): FilterPair,
            (<(<$l_vector as HasOutput>::OutputBool, <$r_vector as HasOutput>::OutputBool) as TyBoolPair>::Or,N): FilterPair,
            (<$l_vector as HasReuseBuf>::BoundHandlesBool, <$r_vector as HasReuseBuf>::BoundHandlesBool): FilterPair,
            (<$l_vector as HasReuseBuf>::FstHandleBool, <$r_vector as HasReuseBuf>::FstHandleBool): SelectPair,
            (<$l_vector as HasReuseBuf>::SndHandleBool, <$r_vector as HasReuseBuf>::SndHandleBool): SelectPair,
            (<$l_vector as HasReuseBuf>::FstOwnedBufferBool, <$r_vector as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
            (<$l_vector as HasReuseBuf>::SndOwnedBufferBool, <$r_vector as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        {
            type Output = VectorExpr<VecAdd<<$l_ty as VectorOps>::Unwrapped,<$r_ty as VectorOps>::Unwrapped>,D>;

            #[inline] 
            fn add(self,rhs: $r_ty) -> Self::Output {
                <Self as VectorVectorOps<$r_ty>>::add(self,rhs)
            }
        }

        impl<
            $($($l_lifetime),+,)? 
            $($($r_lifetime),+,)?
            $($l_generic: $($l_lifetime_bound |)? $($l_fst_trait_bound $(| $l_trait_bound)*)?),+,
            $($r_generic: $($r_lifetime_bound |)? $($r_fst_trait_bound $(| $r_trait_bound)*)?),+,
            const $size: usize
        > Sub<$r_ty> for $l_ty 
        where 
            $l_item: Sub<$r_item>,
            (<$l_vector as HasOutput>::OutputBool, N): FilterPair,
            (<(<$l_vector as HasOutput>::OutputBool, N) as TyBoolPair>::Or,N): FilterPair,
            (<$l_vector as HasReuseBuf>::BoundHandlesBool, N): FilterPair,
            (<$l_vector as HasReuseBuf>::FstHandleBool, N): SelectPair,
            (<$l_vector as HasReuseBuf>::SndHandleBool, N): SelectPair,
            (<$l_vector as HasReuseBuf>::FstOwnedBufferBool, N): SelectPair,
            (<$l_vector as HasReuseBuf>::SndOwnedBufferBool, N): SelectPair,    
            (N, <$r_vector as HasOutput>::OutputBool): FilterPair,
            (<(N, <$r_vector as HasOutput>::OutputBool) as TyBoolPair>::Or,N): FilterPair,
            (N, <$r_vector as HasReuseBuf>::BoundHandlesBool): FilterPair,
            (N, <$r_vector as HasReuseBuf>::FstHandleBool): SelectPair,
            (N, <$r_vector as HasReuseBuf>::SndHandleBool): SelectPair,
            (N, <$r_vector as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
            (N, <$r_vector as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
            (<$l_vector as HasOutput>::OutputBool, <$r_vector as HasOutput>::OutputBool): FilterPair,
            (<(<$l_vector as HasOutput>::OutputBool, <$r_vector as HasOutput>::OutputBool) as TyBoolPair>::Or,N): FilterPair,
            (<$l_vector as HasReuseBuf>::BoundHandlesBool, <$r_vector as HasReuseBuf>::BoundHandlesBool): FilterPair,
            (<$l_vector as HasReuseBuf>::FstHandleBool, <$r_vector as HasReuseBuf>::FstHandleBool): SelectPair,
            (<$l_vector as HasReuseBuf>::SndHandleBool, <$r_vector as HasReuseBuf>::SndHandleBool): SelectPair,
            (<$l_vector as HasReuseBuf>::FstOwnedBufferBool, <$r_vector as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
            (<$l_vector as HasReuseBuf>::SndOwnedBufferBool, <$r_vector as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
        {
            type Output = VectorExpr<VecSub<<$l_ty as VectorOps>::Unwrapped,<$r_ty as VectorOps>::Unwrapped>,D>;

            #[inline] 
            fn sub(self,rhs: $r_ty) -> Self::Output {
                <Self as VectorVectorOps<$r_ty>>::sub(self,rhs)
            }
        }
    };
}

macro_rules! impl_some_const_sized_double_vector_ops {
    (
        $size:ident
        |
        | 
        <$($($r_lifetime:lifetime),+,)? $($r_generic:ident $(:)? $($r_lifetime_bound:lifetime |)? $($r_fst_trait_bound:path $(| $r_trait_bound:path)*)?),+ $(, {$r_tt:tt})?>,
        $r_ty:ty,
        vector: $r_vector:ty,
        item: $r_item:ty
    ) => {};
    (
        $size:ident 
        |
        <$($($fst_l_lifetime:lifetime),+,)? $($fst_l_generic:ident $(:)? $($fst_l_lifetime_bound:lifetime |)? $($fst_l_fst_trait_bound:path $(| $fst_l_trait_bound:path)*)?),+ $(, {$fst_l_tt:tt})?>,
        $fst_l_ty:ty,
        vector: $fst_l_vector:ty,
        item: $fst_l_item:ty
        $(; <$($($l_lifetime:lifetime),+,)? $($l_generic:ident $(:)? $($l_lifetime_bound:lifetime |)? $($l_fst_trait_bound:path $(| $l_trait_bound:path)*)?),+ $(, {$l_tt:tt})?>,
        $l_ty:ty,
        vector: $l_vector:ty,
        item: $l_item:ty)* 
        |
        <$($($r_lifetime:lifetime),+,)? $($r_generic:ident $(:)? $($r_lifetime_bound:lifetime |)? $($r_fst_trait_bound:path $(| $r_trait_bound:path)*)?),+ $(, {$r_tt:tt})?>,
        $r_ty:ty,
        vector: $r_vector:ty,
        item: $r_item:ty
    ) => {
        impl_const_sized_double_vector_ops!(
            $size;
            <$($($fst_l_lifetime),+,)? $($fst_l_generic: $($fst_l_lifetime_bound |)? $($fst_l_fst_trait_bound $(| $fst_l_trait_bound)*)?),+ $(, {$fst_l_tt})?>,
            $fst_l_ty,
            vector: $fst_l_vector,
            item: $fst_l_item;
            <$($($r_lifetime),+,)? $($r_generic: $($r_lifetime_bound |)? $($r_fst_trait_bound $(| $r_trait_bound)*)?),+ $(, {$r_tt})?>,
            $r_ty,
            vector: $r_vector,
            item: $r_item
        );

        impl_some_const_sized_double_vector_ops!(
            $size
            |
            $(
                <$($($l_lifetime),+,)? $($l_generic: $($l_lifetime_bound |)? $($l_fst_trait_bound $(| $l_trait_bound)*)?),+ $(, {$l_tt})?>,
                $l_ty,
                vector: $l_vector,
                item: $l_item
            );*
            |
            <$($($r_lifetime),+,)? $($r_generic: $($r_lifetime_bound |)? $($r_fst_trait_bound $(| $r_trait_bound)*)?),+ $(, {$r_tt})?>,
            $r_ty,
            vector: $r_vector,
            item: $r_item
        );
    };
}

macro_rules! impl_all_const_sized_double_vector_ops {
    (
        $size:ident
        |
        $(<$($($l_lifetime:lifetime),+,)? $($l_generic:ident $(:)? $($l_lifetime_bound:lifetime |)? $($l_fst_trait_bound:path $(| $l_trait_bound:path)*)?),+ $(, {$l_tt:tt})?>,
        $l_ty:ty,
        vector: $l_vector:ty,
        item: $l_item:ty);+
        |
    ) => {};
    (
        $size:ident 
        |
        $(<$($($l_lifetime:lifetime),+,)? $($l_generic:ident $(:)? $($l_lifetime_bound:lifetime |)? $($l_fst_trait_bound:path $(| $l_trait_bound:path)*)?),+ $(, {$l_tt:tt})?>,
        $l_ty:ty,
        vector: $l_vector:ty,
        item: $l_item:ty);+
        |
        <$($($fst_r_lifetime:lifetime),+,)? $($fst_r_generic:ident $(:)? $($fst_r_lifetime_bound:lifetime |)? $($fst_r_fst_trait_bound:path $(| $fst_r_trait_bound:path)*)?),+ $(, {$fst_r_tt:tt})?>,
        $fst_r_ty:ty,
        vector: $fst_r_vector:ty,
        item: $fst_r_item:ty
        $(; <$($($r_lifetime:lifetime),+,)? $($r_generic:ident $(:)? $($r_lifetime_bound:lifetime |)? $($r_fst_trait_bound:path $(| $r_trait_bound:path)*)?),+ $(, {$r_tt:tt})?>,
        $r_ty:ty,
        vector: $r_vector:ty,
        item: $r_item:ty)*
    ) => {      
        impl_some_const_sized_double_vector_ops!(
            $size
            |
            $(
                <$($($l_lifetime),+,)? $($l_generic: $($l_lifetime_bound |)? $($l_fst_trait_bound $(| $l_trait_bound)*)?),+ $(, {$l_tt})?>,
                $l_ty,
                vector: $l_vector,
                item: $l_item
            );+
            |
            <$($($fst_r_lifetime),+,)? $($fst_r_generic: $($fst_r_lifetime_bound |)? $($fst_r_fst_trait_bound $(| $fst_r_trait_bound)*)?),+ $(, {$fst_r_tt})?>,
            $fst_r_ty,
            vector: $fst_r_vector,
            item: $fst_r_item
        );

        impl_all_const_sized_double_vector_ops!{
            $size
            |
            $(
                <$($($l_lifetime),+,)? $($l_generic: $($l_lifetime_bound |)? $($l_fst_trait_bound $(| $l_trait_bound)*)?),+ $(, {$l_tt})?>,
                $l_ty,
                vector: $l_vector,
                item: $l_item
            );+
            |
            $(
                <$($($r_lifetime),+,)? $($r_generic: $($r_lifetime_bound |)? $($r_fst_trait_bound $(| $r_trait_bound)*)?),+ $(, {$r_tt})?>,
                $r_ty,
                vector: $r_vector,
                item: $r_item
            );*
        }
    };
}


impl_all_const_sized_double_vector_ops!(
    D 
    |
    <V1: VectorLike>, VectorExpr<V1,D>, vector: V1, item: V1::Item;
    <V1: VectorLike>, Box<VectorExpr<V1,D>>, vector: V1, item: V1::Item;
    <'a,T1>, &'a MathVector<T1,D>, vector: &'a [T1; D], item: &'a T1;
    <'a,T1>, &'a mut MathVector<T1,D>, vector: &'a mut [T1; D], item: &'a mut T1;
    <'a,T1>, &'a Box<MathVector<T1,D>>, vector: &'a [T1; D], item: &'a T1;
    <'a,T1>, &'a mut Box<MathVector<T1,D>>, vector: &'a mut [T1; D], item: &'a mut T1
    |
    <V2: VectorLike>, VectorExpr<V2,D>, vector: V2, item: V2::Item;
    <V2: VectorLike>, Box<VectorExpr<V2,D>>, vector: V2, item: V2::Item;
    <'b,T2>, &'b MathVector<T2,D>, vector: &'b [T2; D], item: &'b T2;
    <'b,T2>, &'b mut MathVector<T2,D>, vector: &'b mut [T2; D], item: &'b mut T2;
    <'b,T2>, &'b Box<MathVector<T2,D>>, vector: &'b [T2; D], item: &'b T2;
    <'b,T2>, &'b mut Box<MathVector<T2,D>>, vector: &'b mut [T2; D], item: &'b mut T2
);