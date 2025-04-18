use crate::{util_traits::{HasOutput, IsRepeatable}, vector::MathVector};
use std::mem::ManuallyDrop;
use crate::trait_specialization_utils::*;
use std::ops::*;

pub mod mat_util_traits;
pub mod matrix_structs;
pub mod vectorized_matrix_structs;
pub mod matrix_builders;

use mat_util_traits::*;
use matrix_builders::*;
use matrix_structs::*;
use vectorized_matrix_structs::*;


/// D1: # rows (dimension of vectors), D2: # columns (# of vectors)
// MatrixExpr assumes that the stored MatrixLike is fully unused
#[repr(transparent)]
pub struct MatrixExpr<M: MatrixLike, const D1: usize, const D2: usize>(M);

impl<M: MatrixLike, const D1: usize, const D2: usize> MatrixExpr<M, D1, D2> {
    #[inline] 
    pub fn make_dynamic(self) -> MatrixExpr<Box<dyn MatrixLike<
        GetBool = M::GetBool,
        AreInputsTransposed = N,
        Inputs = (),
        Item = M::Item,
        BoundItems = M::BoundItems,

        OutputBool = M::OutputBool,
        Output = M::Output,

        FstHandleBool = M::FstHandleBool,
        SndHandleBool = M::SndHandleBool,
        BoundHandlesBool = M::BoundHandlesBool,
        FstOwnedBufferBool = M::FstOwnedBufferBool,
        SndOwnedBufferBool = M::SndOwnedBufferBool,
        IsFstBufferTransposed = M::IsFstBufferTransposed,
        IsSndBufferTransposed = M::IsSndBufferTransposed,
        AreBoundBuffersTransposed = M::AreBoundBuffersTransposed,
        FstOwnedBuffer = M::FstOwnedBuffer,
        SndOwnedBuffer = M::SndOwnedBuffer,
        FstType = M::FstType,
        SndType = M::SndType,
        BoundTypes = M::BoundTypes,
    >>, D1, D2> where M: 'static {
        MatrixExpr(Box::new(DynamicMatrixLike{mat: self.unwrap(), inputs: None}) as Box<dyn MatrixLike<
            GetBool = M::GetBool,
            AreInputsTransposed = N,
            Inputs = (),
            Item = M::Item,
            BoundItems = M::BoundItems,

            OutputBool = M::OutputBool,
            Output = M::Output,

            FstHandleBool = M::FstHandleBool,
            SndHandleBool = M::SndHandleBool,
            BoundHandlesBool = M::BoundHandlesBool,
            FstOwnedBufferBool = M::FstOwnedBufferBool,
            SndOwnedBufferBool = M::SndOwnedBufferBool,
            IsFstBufferTransposed = M::IsFstBufferTransposed,
            IsSndBufferTransposed = M::IsSndBufferTransposed,
            AreBoundBuffersTransposed = M::AreBoundBuffersTransposed,
            FstOwnedBuffer = M::FstOwnedBuffer,
            SndOwnedBuffer = M::SndOwnedBuffer,
            FstType = M::FstType,
            SndType = M::SndType,
            BoundTypes = M::BoundTypes,
        >>)
    }

    #[inline] 
    pub fn consume(self) -> M::Output where M: Has2DReuseBuf<BoundTypes = M::BoundItems> {
        self.into_entry_iter().consume()
    }

    #[inline]
    pub fn eval(self) -> <MatBind<MatMaybeCreate2DBuf<M, M::Item, D1, D2>> as HasOutput>::Output 
    where
        (M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg): SelectPair,
        (M::FstOwnedBufferBool, <M::FstHandleBool as TyBool>::Neg): TyBoolPair,
        (M::OutputBool, <(M::FstOwnedBufferBool, <M::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or): FilterPair,
        <M::FstHandleBool as TyBool>::Neg: Filter,
        (M::BoundHandlesBool, Y): FilterPair,
        (M::IsFstBufferTransposed, M::AreBoundBuffersTransposed): TyBoolPair,
        MatBind<MatMaybeCreate2DBuf<M, M::Item, D1, D2>>: Has2DReuseBuf<BoundTypes = <MatBind<MatMaybeCreate2DBuf<M, M::Item, D1, D2>> as Get2D>::BoundItems>
    {
        self.maybe_create_2d_buf().bind().consume()
    }

    #[inline]
    pub fn heap_eval(self) -> <MatBind<MatMaybeCreate2DHeapBuf<M, M::Item, D1, D2>> as HasOutput>::Output 
    where
        (M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg): SelectPair,
        (M::FstOwnedBufferBool, <M::FstHandleBool as TyBool>::Neg): TyBoolPair,
        (M::OutputBool, <(M::FstOwnedBufferBool, <M::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or): FilterPair,
        <M::FstHandleBool as TyBool>::Neg: Filter,
        (M::BoundHandlesBool, Y): FilterPair,
        (M::IsFstBufferTransposed, M::AreBoundBuffersTransposed): TyBoolPair,
        MatBind<MatMaybeCreate2DHeapBuf<M, M::Item, D1, D2>>: Has2DReuseBuf<BoundTypes = <MatBind<MatMaybeCreate2DHeapBuf<M, M::Item, D1, D2>> as Get2D>::BoundItems>
    {
        self.maybe_create_2d_heap_buf().bind().consume()
    }


    #[inline]
    pub fn into_entry_iter(self) -> MatrixEntryIter<M, D1, D2> {
        MatrixEntryIter{
            mat: self.unwrap(),
            current_col: 0,
            live_input_row_start: 0,
            dead_output_row_start: 0 
        }
    }
}

impl<M: MatrixLike + IsRepeatable, const D1: usize, const D2: usize> MatrixExpr<M,D1,D2> {
    /// Note:   This method does NOT fill any buffers bound to the matrix, if you need that, see binding_get
    pub fn get(&mut self, col_index: usize, row_index: usize) -> M::Item {
        if (col_index >= D2) | (row_index >= D1) {panic!("math_vector Error: index access out of bound")}
        unsafe {
            let inputs = self.0.get_inputs(col_index, row_index);
            let (item, _) = self.0.process(inputs);
            item
        }
    } 

    /// Note:   Some buffers do not drop pre-existing values when being filled as such values may be undefined data
    ///         however, this means that binding an index multiple times can cause a leak (ie. with Box<T>'s being bound)
    ///         Additionally, if the buffer is owned by the matrix, the matrix expr is also responsible for dropping filled indices
    ///         however, such filled indices filled via this method aren't tracked so further leaks can happen 
    ///         (assuming it isn't retroactivly noted as filled during evaluation/iteration)
    /// Note TLDR: this method is extremely prone to causing memory leaks
    pub fn binding_get(&mut self, col_index: usize, row_index: usize) -> M::Item where M: Has2DReuseBuf<BoundTypes = M::BoundItems> {
        if (col_index >= D2) | (row_index >= D1) {panic!("math_vector Error: index access out of bound")}
        unsafe {
            let inputs = self.0.get_inputs(col_index, row_index);
            let (item, bound_items) = self.0.process(inputs);
            self.0.assign_bound_bufs(col_index, row_index, bound_items);
            item
        }
    }
}

impl<M: MatrixLike, const D1: usize, const D2: usize> Drop for MatrixExpr<M, D1, D2> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            for col_index in 0..D2 {
                for row_index in 0..D1 {
                    self.0.drop_inputs(col_index, row_index);
                }
            }
            self.0.drop_output();
        }
    }
}


pub struct MatrixEntryIter<M: MatrixLike, const D1: usize, const D2: usize>{mat: M, current_col: usize, live_input_row_start: usize, dead_output_row_start: usize}

impl<M: MatrixLike, const D1: usize, const D2: usize> MatrixEntryIter<M, D1, D2> {
    #[inline]
    pub fn raw_next(&mut self) -> Option<M::Item> where M: Has2DReuseBuf<BoundTypes = M::BoundItems> {
        unsafe {
            if self.live_input_row_start < D1 { //current vector isn't done
                let row_index = self.live_input_row_start;
                self.live_input_row_start += 1;
                let inputs = self.mat.get_inputs(self.current_col, row_index);
                let (item, bound_items) = self.mat.process(inputs);
                self.mat.assign_bound_bufs(self.current_col, row_index, bound_items);
                self.dead_output_row_start += 1;
                Some(item)
            } else if self.current_col < D2-1 {
                self.current_col += 1;
                self.live_input_row_start = 1; //we immediately and infallibly get the first one
                self.dead_output_row_start = 0;
                let inputs = self.mat.get_inputs(self.current_col, 0);
                let (item, bound_items) = self.mat.process(inputs);
                self.mat.assign_bound_bufs(self.current_col, 0, bound_items);
                self.dead_output_row_start += 1;
                Some(item)
            } else {
                None
            }
        }
    }

    #[inline]
    pub unsafe fn unchecked_output(self) -> M::Output {
        let mut man_drop_self = std::mem::ManuallyDrop::new(self);
        let output;
        unsafe { 
            output = man_drop_self.mat.output();
            std::ptr::drop_in_place(&mut man_drop_self.mat);
        }
        output
    }

    #[inline]
    pub fn output(self) -> M::Output {
        assert!((self.current_col == D2-1) & (self.live_input_row_start == D1), "math_vector error: A VectorIter must be fully used before outputting");
        debug_assert!(self.dead_output_row_start == D1, "math_vector internal error: A VectorIter's output buffers (somehow) weren't fully filled despite the inputs being fully used, likely an internal issue");
        unsafe {self.unchecked_output()}
    }

    #[inline]
    pub fn consume(mut self) -> M::Output where M: Has2DReuseBuf<BoundTypes = M::BoundItems> {
        self.no_output_consume();
        unsafe {self.unchecked_output()}
    }

    #[inline]
    pub fn no_output_consume(&mut self) where M: Has2DReuseBuf<BoundTypes = M::BoundItems> {
        let mat = &mut self.mat;
        let current_col = &mut self.current_col;
        let live_input_row_start = &mut self.live_input_row_start;
        let dead_output_row_start = &mut self.dead_output_row_start;
        unsafe {
            while *current_col < D2-1 {
                while *live_input_row_start < D1 {
                    let row_index = *live_input_row_start;
                    *live_input_row_start += 1;
                    let inputs = mat.get_inputs(*current_col, row_index);
                    let (_, bound_items) = mat.process(inputs);
                    mat.assign_bound_bufs(*current_col, row_index, bound_items);
                    *dead_output_row_start += 1;
                }
                *live_input_row_start = 0;
                *dead_output_row_start = 0;
                *current_col += 1;
            }
            while *live_input_row_start < D1 {
                let row_index = *live_input_row_start;
                *live_input_row_start += 1;
                let inputs = mat.get_inputs(*current_col, row_index);
                let (_, bound_items) = mat.process(inputs);
                mat.assign_bound_bufs(*current_col, row_index, bound_items);
                *dead_output_row_start += 1;
            }
        }
    }
}

impl<M: MatrixLike, const D1: usize, const D2: usize> Drop for MatrixEntryIter<M, D1, D2> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.mat.drop_output();
            for col_index in 0..self.current_col {
                for row_index in 0..D1 {
                    self.mat.drop_bound_bufs_index(col_index, row_index);
                }
            }
            for row_index in 0..self.dead_output_row_start {
                self.mat.drop_bound_bufs_index(self.current_col, row_index);
            }
            for row_index in self.live_input_row_start..D1 {
                self.mat.drop_inputs(self.current_col, row_index);
            }
            for col_index in self.current_col+1..D2 {
                for row_index in 0..D1 {
                    self.mat.drop_inputs(col_index, row_index);
                }
            }
        }
    }
}

impl<M: MatrixLike, const D1: usize, const D2: usize> Iterator for MatrixEntryIter<M, D1, D2> where M: Has2DReuseBuf<BoundTypes = M::BoundItems> {
    type Item = M::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {self.raw_next()}
}


pub type MathMatrix<T, const D1: usize, const D2: usize> = MatrixExpr<Owned2DArray<T, D1, D2>, D1, D2>;

impl<T, const D1: usize, const D2: usize> MathMatrix<T, D1, D2> {
    #[inline] pub fn into_2d_array(self) -> [[T; D1]; D2] {self.unwrap().unwrap()}
    #[inline] pub fn into_2d_heap_array(self: Box<Self>) -> Box<[[T; D1]; D2]> {
        unsafe { std::mem::transmute::<Box<Self>, Box<[[T; D1]; D2]>>(self) }
    }
    #[inline] pub fn reuse(self) -> MatrixExpr<Replace2DArray<T, D1, D2>, D1, D2> {MatrixExpr(Replace2DArray(self.unwrap().0))}
    #[inline] pub fn heap_reuse(self: Box<Self>) -> MatrixExpr<Replace2DHeapArray<T, D1, D2>, D1, D2> {
        unsafe { MatrixExpr(Replace2DHeapArray(std::mem::transmute::<Box<Self>, std::mem::ManuallyDrop<Box<[[T; D1]; D2]>>>(self))) }
    } 
    
    #[inline] pub fn referred<'a>(self) -> MatrixExpr<Referring2DArray<'a, T, D1, D2>, D1, D2> where T: 'a {
        MatrixExpr(Referring2DArray(unsafe {std::mem::transmute_copy::<ManuallyDrop<[[T; D1]; D2]>, [[T; D1]; D2]>(&self.unwrap().0)}, std::marker::PhantomData))
    }

    #[inline] pub unsafe fn get_unchecked<I: std::slice::SliceIndex<[[T; D1]]>>(&self, index: I) -> &I::Output { unsafe {
        self.0.0.get_unchecked(index)
    }}
    #[inline] pub unsafe fn get_unchecked_mut<I: std::slice::SliceIndex<[[T; D1]]>>(&mut self, index: I) -> &mut I::Output { unsafe {
        self.0.0.get_unchecked_mut(index)
    }}
}

impl<T: Clone, const D1: usize, const D2: usize> Clone for MathMatrix<T, D1, D2> {
    #[inline]
    fn clone(&self) -> Self {
        MatrixExpr(self.0.clone())
    }
}

impl<T, const D1: usize, const D2: usize> Deref for MathMatrix<T, D1, D2> {
    type Target = [[T; D1]; D2];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0.0
    }
}

impl<T, const D1: usize, const D2: usize> DerefMut for MathMatrix<T, D1, D2> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0.0
    }
}

impl<T, const D1: usize, const D2: usize> From<[[T; D1]; D2]> for MathMatrix<T, D1, D2> {
    #[inline] 
    fn from(value: [[T; D1]; D2]) -> Self {
        MatrixExpr(Owned2DArray(std::mem::ManuallyDrop::new(value)))
    }
}

impl<T, const D1: usize, const D2: usize> Into<[[T; D1]; D2]> for MathMatrix<T, D1, D2> {
    #[inline] fn into(self) -> [[T; D1]; D2] {self.into_2d_array()}
}

impl<'a, T, const D1: usize, const D2: usize> From<&'a [[T; D1]; D2]> for &'a MathMatrix<T, D1, D2> {
    #[inline]
    fn from(value: &'a [[T; D1]; D2]) -> Self {
        unsafe { std::mem::transmute::<&'a [[T; D1]; D2], &'a MathMatrix<T, D1, D2>>(value) }
    }
}

impl<'a, T, const D1: usize, const D2: usize> Into<&'a [[T; D1]; D2]> for &'a MathMatrix<T, D1, D2> {
    #[inline]
    fn into(self) -> &'a [[T; D1]; D2] {
        unsafe { std::mem::transmute::<&'a MathMatrix<T, D1, D2>, &'a [[T; D1]; D2]>(self) }
    }
}

impl<'a, T, const D1: usize, const D2: usize> From<&'a mut [[T; D1]; D2]> for &'a mut MathMatrix<T, D1, D2> {
    #[inline]
    fn from(value: &'a mut [[T; D1]; D2]) -> Self {
        unsafe { std::mem::transmute::<&'a mut [[T; D1]; D2], &'a mut MathMatrix<T, D1, D2>>(value) }
    }
}

impl<'a, T, const D1: usize, const D2: usize> Into<&'a mut [[T; D1]; D2]> for &'a mut MathMatrix<T, D1, D2> {
    #[inline]
    fn into(self) -> &'a mut [[T; D1]; D2] {
        unsafe { std::mem::transmute::<&'a mut MathMatrix<T, D1, D2>, &'a mut [[T; D1]; D2]>(self) }
    }
}

impl<T, const D1: usize, const D2: usize> From<MathVectoredMatrix<T, D1, D2>> for MathMatrix<T, D1, D2> {
    #[inline]
    fn from(value: MathVectoredMatrix<T, D1, D2>) -> Self {
        //  safety:
        //      MathVectoredMatrix<T, D1, D2> == VectorExpr<OwnedArray<VectorExpr<OwnedArray<T, D1>, D1>, D2>, D2>
        //      MathVectoredMatrix<T, D1, D2> == ManuallyDrop<[ManuallyDrop<[T; D1]>; D2]>
        //      MathVectoredMatrix<T, D1, D2> == ManuallyDrop<[[T; D1]; D2]>
        //      MathVectoredMatrix<T, D1, D2> == MatrixExpr<Owned2DArray<T, D1, D2>>
        //      MathVectoredMatrix<T, D1, D2> == MathMatrix<T, D1, D2>
        //
        //  FIXME: transmute_copy copies (:O), this shouldn't need to be done but gets the compiler to not complain about it
        unsafe { std::mem::transmute_copy::<MathVectoredMatrix<T, D1, D2>, MathMatrix<T, D1, D2>>(&std::mem::ManuallyDrop::new(value)) }
    }
}

impl<T, const D1: usize, const D2: usize> Into<MathVectoredMatrix<T, D1, D2>> for MathMatrix<T, D1, D2> {
    #[inline]
    fn into(self) -> MathVectoredMatrix<T, D1, D2> {
        //  safety:
        //      MathVectoredMatrix<T, D1, D2> == VectorExpr<OwnedArray<VectorExpr<OwnedArray<T, D1>, D1>, D2>, D2>
        //      MathVectoredMatrix<T, D1, D2> == ManuallyDrop<[ManuallyDrop<[T; D1]>; D2]>
        //      MathVectoredMatrix<T, D1, D2> == ManuallyDrop<[[T; D1]; D2]>
        //      MathVectoredMatrix<T, D1, D2> == MatrixExpr<Owned2DArray<T, D1, D2>>
        //      MathVectoredMatrix<T, D1, D2> == MathMatrix<T, D1, D2>
        //
        //  FIXME: transmute_copy copies (:O), this shouldn't need to be done but gets the compiler to not complain about it
        unsafe { std::mem::transmute_copy::<MathMatrix<T, D1, D2>, MathVectoredMatrix<T, D1, D2>>(&std::mem::ManuallyDrop::new(self)) }
    }
}

impl<T, I, const D1: usize, const D2: usize> Index<I> for MathMatrix<T, D1, D2> where [[T; D1]; D2]: Index<I, Output = [T; D1]> {
    type Output = MathVector<T, D1>;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        (&self.0.0[index]).into()
    }
}

impl<T, I, const D1: usize, const D2: usize> IndexMut<I> for MathMatrix<T, D1, D2> where [[T; D1]; D2]: IndexMut<I, Output = [T; D1]> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        (&mut self.0.0[index]).into()
    }
}



pub fn matrix_gen<F: FnMut() -> O, O, const D1: usize, const D2: usize>(f: F) -> MatrixExpr<MatGenerator<F, O>, D1, D2> {
    MatrixExpr(MatGenerator(f))
}


pub trait MatrixOps {
    type Unwrapped: MatrixLike;
    type Builder: MatrixBuilder;

    fn unwrap(self) -> Self::Unwrapped;
    fn get_builder(&self) -> Self::Builder;
    fn dimensions(&self) -> (usize, usize); // 0: num rows, 1: num columns

    #[inline]
    fn bind(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatBind<Self::Unwrapped>>
    where
        Self::Unwrapped: Has2DReuseBuf<FstHandleBool = Y>,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, Y): FilterPair,
        (<Self::Unwrapped as HasOutput>::OutputBool, <Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
        Self: Sized
    {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatBind{mat: self.unwrap()}) }
    }

    //TODO: map_bind

    #[inline]
    fn half_bind(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatHalfBind<Self::Unwrapped>>
    where 
        Self::Unwrapped:  MatrixLike<FstHandleBool = Y>,
        (<Self::Unwrapped as HasOutput>::OutputBool, <Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, Y): FilterPair,
        MatHalfBind<Self::Unwrapped>: Has2DReuseBuf<BoundTypes = <MatBind<Self::Unwrapped> as Get2D>::BoundItems>,
        Self: Sized
    {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatHalfBind{mat: self.unwrap()}) }
    }

    #[inline]
    fn buf_swap(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatBufSwap<Self::Unwrapped>> where (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatBufSwap{mat: self.unwrap()}) }
    }

    #[inline]
    fn offset_columns(self, offset: usize) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatColOffset<Self::Unwrapped>> where Self: Sized {
        let (_, cols) = self.dimensions();
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatColOffset{mat: self.unwrap(), offset: offset % cols, num_columns: cols}) }
    }

    #[inline]
    fn offset_rows(self, offset: usize) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatRowOffset<Self::Unwrapped>> where Self: Sized {
        let (rows, _) = self.dimensions();
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatRowOffset{mat: self.unwrap(), offset: offset % rows, num_rows: rows}) }
    }

    #[inline]
    fn columns(self) -> <Self::Builder as MatrixBuilder>::RowWrapped<MatColWrapper<MatColVectorExprs<Self::Unwrapped>, MatrixColumn<Self::Unwrapped>, Self::Builder>> where Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_row_vec(MatColWrapper{mat: MatColVectorExprs{mat: self.unwrap()}, builder: builder.clone()}) }
    }

    #[inline]
    fn rows(self) -> <Self::Builder as MatrixBuilder>::ColWrapped<MatRowWrapper<MatRowVectorExprs<Self::Unwrapped>, MatrixRow<Self::Unwrapped>, Self::Builder>> where Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_col_vec(MatRowWrapper{mat: MatRowVectorExprs{mat: self.unwrap()}, builder: builder.clone()}) }
    }


    #[inline]
    fn entry_map<F: FnMut(<Self::Unwrapped as Get2D>::Item) -> O, O>(self, f: F) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatEntryMap<Self::Unwrapped, F, O>> where (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatEntryMap{mat: self.unwrap(), f}) }
    }

    #[inline]
    fn entry_fold<F: FnMut(O, <Self::Unwrapped as Get2D>::Item) -> O, O>(self, f: F, init: O) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatEntryFold<Self::Unwrapped, F, O>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatEntryFold{mat: self.unwrap(), f, cell: Some(init)}) }
    }

    #[inline]
    fn entry_fold_ref<F: FnMut(&mut O, <Self::Unwrapped as Get2D>::Item), O>(self, f: F, init: O) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatEntryFoldRef<Self::Unwrapped, F, O>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatEntryFoldRef{mat: self.unwrap(), f, cell: std::mem::ManuallyDrop::new(init)}) }
    }

    #[inline]
    fn entry_copied_fold<F: FnMut(O, <Self::Unwrapped as Get2D>::Item) -> O, O>(self, f: F, init: O) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatEntryCopiedFold<Self::Unwrapped, F, O>> where <Self::Unwrapped as Get2D>::Item: Copy, (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatEntryCopiedFold{mat: self.unwrap(), f, cell: Some(init)}) }
    }

    #[inline]
    fn entry_copied_fold_ref<F: FnMut(&mut O, <Self::Unwrapped as Get2D>::Item), O>(self, f: F, init: O) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatEntryCopiedFoldRef<Self::Unwrapped, F, O>> where <Self::Unwrapped as Get2D>::Item: Copy, (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatEntryCopiedFoldRef{mat: self.unwrap(), f, cell: std::mem::ManuallyDrop::new(init)}) }
    }

    #[inline]
    fn copied<'a, I: 'a + Copy>(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatCopy<'a, Self::Unwrapped, I>> where Self::Unwrapped: Get2D<Item = &'a I>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatCopy{mat: self.unwrap()}) }
    }

    #[inline]
    fn cloned<'a, I: 'a + Clone>(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatClone<'a, Self::Unwrapped, I>> where Self::Unwrapped: Get2D<Item = &'a I>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatClone{mat: self.unwrap()}) }
    }

    #[inline] 
    fn neg(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatNeg<Self::Unwrapped>> where <Self::Unwrapped as Get2D>::Item: Neg, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatNeg{mat: self.unwrap()})}
    }

    #[inline]
    fn mul_r<S: Copy>(self, scalar: S) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatMulR<Self::Unwrapped, S>> where S: Mul<<Self::Unwrapped as Get2D>::Item>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatMulR{mat: self.unwrap(), scalar})}
    }

    #[inline]
    fn div_r<S: Copy>(self, scalar: S) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatDivR<Self::Unwrapped, S>> where S: Div<<Self::Unwrapped as Get2D>::Item>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatDivR{mat: self.unwrap(), scalar})}
    }

    #[inline]
    fn rem_r<S: Copy>(self, scalar: S) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatRemR<Self::Unwrapped, S>> where S: Rem<<Self::Unwrapped as Get2D>::Item>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatRemR{mat: self.unwrap(), scalar})}
    }

    #[inline]
    fn mul_l<S: Copy>(self, scalar: S) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatMulL<Self::Unwrapped, S>> where <Self::Unwrapped as Get2D>::Item: Mul<S>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatMulL{mat: self.unwrap(), scalar})}
    }

    #[inline]
    fn div_l<S: Copy>(self, scalar: S) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatDivL<Self::Unwrapped, S>> where <Self::Unwrapped as Get2D>::Item: Div<S>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatDivL{mat: self.unwrap(), scalar})}
    }

    #[inline]
    fn rem_l<S: Copy>(self, scalar: S) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatRemL<Self::Unwrapped, S>> where <Self::Unwrapped as Get2D>::Item: Rem<S>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatRemL{mat: self.unwrap(), scalar})}
    }

    #[inline]
    fn mul_assign<'a, I: 'a + MulAssign<S>, S: Copy>(self, scalar: S) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatMulAssign<'a, Self::Unwrapped, I, S>> where Self::Unwrapped: Get2D<Item = &'a mut I>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatMulAssign{mat: self.unwrap(), scalar}) }
    }

    #[inline]
    fn div_assign<'a, I: 'a + DivAssign<S>, S: Copy>(self, scalar: S) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatDivAssign<'a, Self::Unwrapped, I, S>> where Self::Unwrapped: Get2D<Item = &'a mut I>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatDivAssign{mat: self.unwrap(), scalar}) }
    }

    #[inline]
    fn rem_assign<'a, I: 'a + RemAssign<S>, S: Copy>(self, scalar: S) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatRemAssign<'a, Self::Unwrapped, I, S>> where Self::Unwrapped: Get2D<Item = &'a mut I>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatRemAssign{mat: self.unwrap(), scalar}) }
    }

    ///NOTE: WILL BE MOVED
    #[inline]
    fn mat_mul<M: MatrixOps>(self, other: M) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<FullMatMul<Self::Unwrapped, M::Unwrapped>> where 
        Self::Unwrapped: IsRepeatable,
        M::Unwrapped: IsRepeatable,
        <Self::Unwrapped as Get2D>::Item: Mul<<M::Unwrapped as Get2D>::Item>,
        <<Self::Unwrapped as Get2D>::Item as Mul<<M::Unwrapped as Get2D>::Item>>::Output: AddAssign,

        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    
        Self: Sized,
        M: Sized
    {
        if self.dimensions().1 != other.dimensions().0 {panic!("math_vector Error: cannot multiply matrices with incompatible sizes")}
        let shared_size = self.dimensions().1;
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(FullMatMul{l_mat: self.unwrap(), r_mat: other.unwrap(), shared_size}) }
    }


    #[inline]
    fn zip<M: MatrixOps>(self, other: M) -> <<Self::Builder as MatrixBuilderUnion<M::Builder>>::Union as MatrixBuilder>::MatrixWrapped<MatZip<Self::Unwrapped, M::Unwrapped>>
    where
        Self: Sized,
        Self::Builder: MatrixBuilderUnion<M::Builder>, 
        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap_mat(MatZip{ l_mat: self.unwrap(), r_mat: other.unwrap() }) }
    }

    #[inline]
    fn add<M: MatrixOps>(self, other: M) -> <<Self::Builder as MatrixBuilderUnion<M::Builder>>::Union as MatrixBuilder>::MatrixWrapped<MatAdd<Self::Unwrapped, M::Unwrapped>>
    where
        Self: Sized,
        Self::Builder: MatrixBuilderUnion<M::Builder>, 
        <Self::Unwrapped as Get2D>::Item: Add<<M::Unwrapped as Get2D>::Item>,
        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap_mat(MatAdd{ l_mat: self.unwrap(), r_mat: other.unwrap() }) }
    }

    #[inline]
    fn sub<M: MatrixOps>(self, other: M) -> <<Self::Builder as MatrixBuilderUnion<M::Builder>>::Union as MatrixBuilder>::MatrixWrapped<MatSub<Self::Unwrapped, M::Unwrapped>>
    where
        Self: Sized,
        Self::Builder: MatrixBuilderUnion<M::Builder>, 
        <Self::Unwrapped as Get2D>::Item: Sub<<M::Unwrapped as Get2D>::Item>,
        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap_mat(MatSub{ l_mat: self.unwrap(), r_mat: other.unwrap() }) }
    }

    #[inline]
    fn comp_mul<M: MatrixOps>(self, other: M) -> <<Self::Builder as MatrixBuilderUnion<M::Builder>>::Union as MatrixBuilder>::MatrixWrapped<MatCompMul<Self::Unwrapped, M::Unwrapped>>
    where
        Self: Sized,
        Self::Builder: MatrixBuilderUnion<M::Builder>, 
        <Self::Unwrapped as Get2D>::Item: Mul<<M::Unwrapped as Get2D>::Item>,
        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap_mat(MatCompMul{ l_mat: self.unwrap(), r_mat: other.unwrap() }) }
    }

    #[inline]
    fn comp_div<M: MatrixOps>(self, other: M) -> <<Self::Builder as MatrixBuilderUnion<M::Builder>>::Union as MatrixBuilder>::MatrixWrapped<MatCompDiv<Self::Unwrapped, M::Unwrapped>>
    where
        Self: Sized,
        Self::Builder: MatrixBuilderUnion<M::Builder>, 
        <Self::Unwrapped as Get2D>::Item: Div<<M::Unwrapped as Get2D>::Item>,
        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap_mat(MatCompDiv{ l_mat: self.unwrap(), r_mat: other.unwrap() }) }
    }

    #[inline]
    fn comp_rem<M: MatrixOps>(self, other: M) -> <<Self::Builder as MatrixBuilderUnion<M::Builder>>::Union as MatrixBuilder>::MatrixWrapped<MatCompRem<Self::Unwrapped, M::Unwrapped>>
    where
        Self: Sized,
        Self::Builder: MatrixBuilderUnion<M::Builder>, 
        <Self::Unwrapped as Get2D>::Item: Rem<<M::Unwrapped as Get2D>::Item>,
        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap_mat(MatCompRem{ l_mat: self.unwrap(), r_mat: other.unwrap() }) }
    }

    #[inline]
    fn add_assign<'a, M: MatrixOps, I: 'a + AddAssign<<M::Unwrapped as Get2D>::Item>>(self, other: M) -> <<Self::Builder as MatrixBuilderUnion<M::Builder>>::Union as MatrixBuilder>::MatrixWrapped<MatAddAssign<'a, Self::Unwrapped, M::Unwrapped, I>>
    where 
        Self: Sized,
        Self::Builder: MatrixBuilderUnion<M::Builder>, 
        Self::Unwrapped: Get2D<Item = &'a mut I>,
        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap_mat(MatAddAssign{ l_mat: self.unwrap(), r_mat: other.unwrap() }) }
    }

    #[inline]
    fn sub_assign<'a, M: MatrixOps, I: 'a + SubAssign<<M::Unwrapped as Get2D>::Item>>(self, other: M) -> <<Self::Builder as MatrixBuilderUnion<M::Builder>>::Union as MatrixBuilder>::MatrixWrapped<MatSubAssign<'a, Self::Unwrapped, M::Unwrapped, I>>
    where 
        Self: Sized,
        Self::Builder: MatrixBuilderUnion<M::Builder>, 
        Self::Unwrapped: Get2D<Item = &'a mut I>,
        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap_mat(MatSubAssign{ l_mat: self.unwrap(), r_mat: other.unwrap() }) }
    }

    #[inline]
    fn comp_mul_assign<'a, M: MatrixOps, I: 'a + MulAssign<<M::Unwrapped as Get2D>::Item>>(self, other: M) -> <<Self::Builder as MatrixBuilderUnion<M::Builder>>::Union as MatrixBuilder>::MatrixWrapped<MatCompMulAssign<'a, Self::Unwrapped, M::Unwrapped, I>>
    where 
        Self: Sized,
        Self::Builder: MatrixBuilderUnion<M::Builder>, 
        Self::Unwrapped: Get2D<Item = &'a mut I>,
        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap_mat(MatCompMulAssign{ l_mat: self.unwrap(), r_mat: other.unwrap() }) }
    }

    #[inline]
    fn comp_div_assign<'a, M: MatrixOps, I: 'a + DivAssign<<M::Unwrapped as Get2D>::Item>>(self, other: M) -> <<Self::Builder as MatrixBuilderUnion<M::Builder>>::Union as MatrixBuilder>::MatrixWrapped<MatCompDivAssign<'a, Self::Unwrapped, M::Unwrapped, I>>
    where 
        Self: Sized,
        Self::Builder: MatrixBuilderUnion<M::Builder>, 
        Self::Unwrapped: Get2D<Item = &'a mut I>,
        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap_mat(MatCompDivAssign{ l_mat: self.unwrap(), r_mat: other.unwrap() }) }
    }

    #[inline]
    fn comp_rem_assign<'a, M: MatrixOps, I: 'a + RemAssign<<M::Unwrapped as Get2D>::Item>>(self, other: M) -> <<Self::Builder as MatrixBuilderUnion<M::Builder>>::Union as MatrixBuilder>::MatrixWrapped<MatCompRemAssign<'a, Self::Unwrapped, M::Unwrapped, I>>
    where 
        Self: Sized,
        Self::Builder: MatrixBuilderUnion<M::Builder>, 
        Self::Unwrapped: Get2D<Item = &'a mut I>,
        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool, <M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    {
        let builder = self.get_builder().union(other.get_builder());
        unsafe { builder.wrap_mat(MatCompRemAssign{ l_mat: self.unwrap(), r_mat: other.unwrap() }) }
    }

}

pub trait ArrayMatrixOps<const D1: usize, const D2: usize>: MatrixOps {
    #[inline]
    fn attach_2d_buf<'a, T>(self, buf: &'a mut MathMatrix<T, D1, D2>) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatAttach2DBuf<'a, Self::Unwrapped, T, D1, D2>> where Self::Unwrapped: Has2DReuseBuf<FstHandleBool = N>, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatAttach2DBuf{mat: self.unwrap(), buf}) }
    }

    #[inline]
    fn create_2d_buf<T>(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatCreate2DBuf<Self::Unwrapped, T, D1, D2>> where Self::Unwrapped: Has2DReuseBuf<FstHandleBool = N>, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatCreate2DBuf{mat: self.unwrap(), buf: std::mem::MaybeUninit::uninit().assume_init()}) }
    }

    #[inline]
    fn create_2d_heap_buf<T>(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatCreate2DHeapBuf<Self::Unwrapped, T, D1, D2>> where Self::Unwrapped: Has2DReuseBuf<FstHandleBool = N>, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatCreate2DHeapBuf{mat: self.unwrap(), buf: std::mem::ManuallyDrop::new(Box::new(std::mem::MaybeUninit::uninit().assume_init()))}) }
    }

    #[inline]
    fn maybe_create_2d_buf<T>(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatMaybeCreate2DBuf<Self::Unwrapped, T, D1, D2>> 
    where 
        <<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg): TyBoolPair,
        Self: Sized
    {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatMaybeCreate2DBuf{mat: self.unwrap(), buf: std::mem::MaybeUninit::uninit().assume_init()}) }
    }

    #[inline]
    fn maybe_create_2d_heap_buf<T>(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatMaybeCreate2DHeapBuf<Self::Unwrapped, T, D1, D2>> 
    where 
        <<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg): TyBoolPair,
        Self: Sized
    {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatMaybeCreate2DHeapBuf{mat: self.unwrap(), buf: std::mem::ManuallyDrop::new(Box::new(std::mem::MaybeUninit::uninit().assume_init()))}) }
    }
}

pub trait RepeatableMatrixOps: MatrixOps {
    type RepeatableMatrix<'a>: MatrixLike + IsRepeatable where Self: 'a;
    type UsedMatrix: MatrixLike;
    //type HeapedUsedMatrix: MatrixLike;

    fn make_repeatable<'a>(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatAttachUsedMat<Self::RepeatableMatrix<'a>, Self::UsedMatrix>> where
        Self: 'a,
        (<Self::RepeatableMatrix<'a> as HasOutput>::OutputBool, <Self::UsedMatrix as HasOutput>::OutputBool): FilterPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::FstHandleBool, <Self::UsedMatrix as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::SndHandleBool, <Self::UsedMatrix as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::BoundHandlesBool, <Self::UsedMatrix as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::FstOwnedBufferBool, <Self::UsedMatrix as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::SndOwnedBufferBool, <Self::UsedMatrix as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::IsFstBufferTransposed, <Self::UsedMatrix as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::IsSndBufferTransposed, <Self::UsedMatrix as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::AreBoundBuffersTransposed, <Self::UsedMatrix as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
        (<<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg, <Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): TyBoolPair,
        <(<<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg, <Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool) as TyBoolPair>::Or: IsTrue
    ;
}

macro_rules! overload_operators {
    (
        <$($($lifetime:lifetime),+, )? $($generic:ident $(:)? $($lifetime_bound:lifetime |)? $($fst_trait_bound:path $(| $trait_bound:path)*)?),+, {$d1:ident, $d2:ident}>,
        $ty:ty,
        matrix: $matrix:ty,
        item: $item:ty
    ) => {
        impl<$($($lifetime),+, )? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: Copy, const $d1: usize, const $d2: usize> Mul<Z> for $ty where (<$matrix as HasOutput>::OutputBool, N): FilterPair, $item: Mul<Z>, Self: Sized {
            type Output = MatrixExpr<MatMulL<$matrix, Z>, $d1, $d2>;
    
            #[inline]
            fn mul(self, rhs: Z) -> Self::Output {
                self.mul_l(rhs)
            }
        }

        impl<$($($lifetime),+, )? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: Copy, const $d1: usize, const $d2: usize> Div<Z> for $ty where (<$matrix as HasOutput>::OutputBool, N): FilterPair, $item: Div<Z>, Self: Sized {
            type Output = MatrixExpr<MatDivL<$matrix, Z>, $d1, $d2>;
    
            #[inline]
            fn div(self, rhs: Z) -> Self::Output {
                self.div_l(rhs)
            }
        }

        impl<$($($lifetime),+, )? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: Copy, const $d1: usize, const $d2: usize> Rem<Z> for $ty where (<$matrix as HasOutput>::OutputBool, N): FilterPair, $item: Rem<Z>, Self: Sized {
            type Output = MatrixExpr<MatRemL<$matrix, Z>, $d1, $d2>;
    
            #[inline]
            fn rem(self, rhs: Z) -> Self::Output {
                self.rem_l(rhs)
            }
        }
    }
}

 
impl<M: MatrixLike, const D1: usize, const D2: usize> MatrixOps for MatrixExpr<M, D1, D2> {
    type Unwrapped = M;
    type Builder = MatrixExprBuilder<D1, D2>;

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

    #[inline] fn get_builder(&self) -> Self::Builder {MatrixExprBuilder}
    #[inline] fn dimensions(&self) -> (usize, usize) {(D1, D2)}
}
impl<M: MatrixLike, const D1: usize, const D2: usize> ArrayMatrixOps<D1, D2> for MatrixExpr<M, D1, D2> {}

impl<M: MatrixLike, const D1: usize, const D2: usize> RepeatableMatrixOps for MatrixExpr<M, D1, D2> where    
    <M::FstHandleBool as TyBool>::Neg: Filter,
    (M::BoundHandlesBool, Y): FilterPair,
    (M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg): SelectPair<
    Selected<M::FstOwnedBuffer, MathMatrix<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<<M as Get2D>::Item>, D1, D2>> = MathMatrix<M::Item, D1, D2>>,
    (M::FstOwnedBufferBool, <M::FstHandleBool as TyBool>::Neg): TyBoolPair,
    (<M::FstHandleBool as TyBool>::Neg, M::FstOwnedBufferBool): TyBoolPair,
    (M::OutputBool, <(M::FstOwnedBufferBool, <M::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or): FilterPair,
    MatHalfBind<MatMaybeCreate2DBuf<M, M::Item, D1, D2>>: Has2DReuseBuf<BoundTypes = <(M::BoundHandlesBool, Y) as FilterPair>::Filtered<M::BoundItems, M::Item>>
{
    type RepeatableMatrix<'a> = Referring2DArray<'a, M::Item, D1, D2> where Self: 'a;
    type UsedMatrix = MatHalfBind<MatMaybeCreate2DBuf<M, M::Item, D1, D2>>;

    fn make_repeatable<'a>(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatAttachUsedMat<Self::RepeatableMatrix<'a>, Self::UsedMatrix>> where
        Self: 'a,
        (<Self::RepeatableMatrix<'a> as HasOutput>::OutputBool, <Self::UsedMatrix as HasOutput>::OutputBool): FilterPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::FstHandleBool, <Self::UsedMatrix as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::SndHandleBool, <Self::UsedMatrix as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::BoundHandlesBool, <Self::UsedMatrix as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::FstOwnedBufferBool, <Self::UsedMatrix as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::SndOwnedBufferBool, <Self::UsedMatrix as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::IsFstBufferTransposed, <Self::UsedMatrix as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::IsSndBufferTransposed, <Self::UsedMatrix as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::AreBoundBuffersTransposed, <Self::UsedMatrix as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
        (<<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg, <Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): TyBoolPair,
        <(<<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg, <Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool) as TyBoolPair>::Or: IsTrue 
    {
        let builder = self.get_builder();
        let mut mat_iter = self.maybe_create_2d_buf().half_bind().into_entry_iter();
        unsafe {
            mat_iter.no_output_consume();
            builder.wrap_mat(MatAttachUsedMat{mat: mat_iter.mat.get_bound_buf().referred().unwrap(), used_mat: std::ptr::read(&mat_iter.mat)})
        }
    }
}

overload_operators!(<M: MatrixLike, {D1, D2}>, MatrixExpr<M, D1, D2>, matrix: M, item: M::Item);

impl<M: MatrixLike, const D1: usize, const D2: usize> MatrixOps for Box<MatrixExpr<M, D1, D2>> {
    type Unwrapped = Box<M>;
    type Builder = MatrixExprBuilder<D1, D2>;

    #[inline]
    fn unwrap(self) -> Self::Unwrapped {
        // safe because this is just move done manually as VectorExpr impls Drop
        // normally a problem as this leaves the fields of the struct at potentially 
        // invalid states which are assumed valid by the drop impl, however we just
        // disable dropping temporarily so this isn't a concern
        // does lead to leaking however, but it is ultimately fixed by wrap and the interim
        // (should) be non-panicking so leaking shouldn't happen

        // FIXME note: this could probably be just transmuted, may optimize better, applies for other owned Mats & Vecs
        unsafe { Box::new(std::ptr::read(&std::mem::ManuallyDrop::new(self).0)) } 
    }
    
    #[inline] fn get_builder(&self) -> Self::Builder {MatrixExprBuilder}
    #[inline] fn dimensions(&self) -> (usize, usize) {(D1, D2)}
}
impl<M: MatrixLike, const D1: usize, const D2: usize> ArrayMatrixOps<D1, D2> for Box<MatrixExpr<M, D1, D2>> {}

impl<M: MatrixLike, const D1: usize, const D2: usize> RepeatableMatrixOps for Box<MatrixExpr<M, D1, D2>> where    
    <M::FstHandleBool as TyBool>::Neg: Filter,
    (M::BoundHandlesBool, Y): FilterPair,
    (M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg): SelectPair<
    Selected<M::FstOwnedBuffer, MathMatrix<<<M::FstHandleBool as TyBool>::Neg as Filter>::Filtered<<M as Get2D>::Item>, D1, D2>> = MathMatrix<M::Item, D1, D2>>,
    (M::FstOwnedBufferBool, <M::FstHandleBool as TyBool>::Neg): TyBoolPair,
    (<M::FstHandleBool as TyBool>::Neg, M::FstOwnedBufferBool): TyBoolPair,
    (M::OutputBool, <(M::FstOwnedBufferBool, <M::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or): FilterPair,
    MatHalfBind<MatMaybeCreate2DBuf<Box<M>, M::Item, D1, D2>>: Has2DReuseBuf<BoundTypes = <(M::BoundHandlesBool, Y) as FilterPair>::Filtered<M::BoundItems, M::Item>>
{
    type RepeatableMatrix<'a> = Referring2DArray<'a, M::Item, D1, D2> where Self: 'a;
    type UsedMatrix = MatHalfBind<MatMaybeCreate2DBuf<Box<M>, M::Item, D1, D2>>;

    fn make_repeatable<'a>(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatAttachUsedMat<Self::RepeatableMatrix<'a>, Self::UsedMatrix>> where
        Self: 'a,
        (<Self::RepeatableMatrix<'a> as HasOutput>::OutputBool, <Self::UsedMatrix as HasOutput>::OutputBool): FilterPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::FstHandleBool, <Self::UsedMatrix as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::SndHandleBool, <Self::UsedMatrix as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::BoundHandlesBool, <Self::UsedMatrix as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::FstOwnedBufferBool, <Self::UsedMatrix as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::SndOwnedBufferBool, <Self::UsedMatrix as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::IsFstBufferTransposed, <Self::UsedMatrix as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::IsSndBufferTransposed, <Self::UsedMatrix as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::RepeatableMatrix<'a> as Has2DReuseBuf>::AreBoundBuffersTransposed, <Self::UsedMatrix as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
        (<<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg, <Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): TyBoolPair,
        <(<<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg, <Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool) as TyBoolPair>::Or: IsTrue 
    {
        let builder = self.get_builder();
        let mut mat_iter = self.maybe_create_2d_buf().half_bind().into_entry_iter();
        unsafe {
            mat_iter.no_output_consume();
            builder.wrap_mat(MatAttachUsedMat{mat: mat_iter.mat.get_bound_buf().referred().unwrap(), used_mat: std::ptr::read(&mat_iter.mat)})
        }
    }
}

overload_operators!(<M: MatrixLike, {D1, D2}>, Box<MatrixExpr<M, D1, D2>>, matrix: Box<M>, item: M::Item);


impl<'a, T, const D1: usize, const D2: usize> MatrixOps for &'a MathMatrix<T, D1, D2> {
    type Unwrapped = &'a [[T; D1]; D2];
    type Builder = MatrixExprBuilder<D1, D2>;
    
    #[inline] fn unwrap(self) -> Self::Unwrapped {&self.0}
    #[inline] fn get_builder(&self) -> Self::Builder {MatrixExprBuilder}
    #[inline] fn dimensions(&self) -> (usize, usize) {(D1, D2)}
}
impl<'a, T, const D1: usize, const D2: usize> ArrayMatrixOps<D1, D2> for &'a MathMatrix<T, D1, D2> {}
overload_operators!(<'a, T, {D1, D2}>, &'a MathMatrix<T, D1, D2>, matrix: &'a [[T; D1]; D2], item: &'a T);

impl<'a, T, const D1: usize, const D2: usize> MatrixOps for &'a mut MathMatrix<T, D1, D2> {
    type Unwrapped = &'a mut [[T; D1]; D2];
    type Builder = MatrixExprBuilder<D1, D2>;
    
    #[inline] fn unwrap(self) -> Self::Unwrapped {&mut self.0}
    #[inline] fn get_builder(&self) -> Self::Builder {MatrixExprBuilder}
    #[inline] fn dimensions(&self) -> (usize, usize) {(D1, D2)}
}
impl<'a, T, const D1: usize, const D2: usize> ArrayMatrixOps<D1, D2> for &'a mut MathMatrix<T, D1, D2> {}
overload_operators!(<'a, T, {D1, D2}>, &'a mut MathMatrix<T, D1, D2>, matrix: &'a mut [[T; D1]; D2], item: &'a mut T);

impl<'a, T, const D1: usize, const D2: usize> MatrixOps for &'a Box<MathMatrix<T, D1, D2>> {
    type Unwrapped = &'a [[T; D1]; D2];
    type Builder = MatrixExprBuilder<D1, D2>;
    
    #[inline] fn unwrap(self) -> Self::Unwrapped {&self.0}
    #[inline] fn get_builder(&self) -> Self::Builder {MatrixExprBuilder}
    #[inline] fn dimensions(&self) -> (usize, usize) {(D1, D2)}
}
impl<'a, T, const D1: usize, const D2: usize> ArrayMatrixOps<D1, D2> for &'a Box<MathMatrix<T, D1, D2>> {}
overload_operators!(<'a, T, {D1, D2}>, &'a Box<MathMatrix<T, D1, D2>>, matrix: &'a [[T; D1]; D2], item: &'a T);

impl<'a, T, const D1: usize, const D2: usize> MatrixOps for &'a mut Box<MathMatrix<T, D1, D2>> {
    type Unwrapped = &'a mut [[T; D1]; D2];
    type Builder = MatrixExprBuilder<D1, D2>;
    
    #[inline] fn unwrap(self) -> Self::Unwrapped {&mut self.0}
    #[inline] fn get_builder(&self) -> Self::Builder {MatrixExprBuilder}
    #[inline] fn dimensions(&self) -> (usize, usize) {(D1, D2)}
}
impl<'a, T, const D1: usize, const D2: usize> ArrayMatrixOps<D1, D2> for &'a mut Box<MathMatrix<T, D1, D2>> {}
overload_operators!(<'a, T, {D1, D2}>, &'a mut Box<MathMatrix<T, D1, D2>>, matrix: &'a mut [[T; D1]; D2], item: &'a mut T);


macro_rules! impl_binary_ops_for_wrapper {
    (
        $(
            $($d1:ident, $d2:ident;)?
            <$($($lifetime:lifetime),+, )? $($generic:ident $(:)? $($lifetime_bound:lifetime |)? $($fst_trait_bound:path $(| $trait_bound:path)*)?),+ $(, {$tt:tt})?>,
            $ty:ty,
            trait_matrix: $trait_matrix:ty,
            true_matrix: $true_matrix:ty;
        )*
    ) => {
        $(
            impl<
                $($($lifetime),+, )? 
                $($generic: $($lifetime_bound |)? $($fst_trait_bound $(| $trait_bound)*)?),+,
                M2: MatrixOps
                $(, const $d1: usize, const $d2: usize)?
            > Add<M2> for $ty where
                <$ty as MatrixOps>::Builder: MatrixBuilderUnion<M2::Builder>, 
                <$trait_matrix as Get2D>::Item: Add<<M2::Unwrapped as Get2D>::Item>,
                (<$trait_matrix as Get2D>::AreInputsTransposed, <M2::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
                (<$trait_matrix as HasOutput>::OutputBool, <M2::Unwrapped as HasOutput>::OutputBool): FilterPair,
                (<(<$trait_matrix as HasOutput>::OutputBool, <M2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
                (<$trait_matrix as Has2DReuseBuf>::BoundHandlesBool, <M2::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
                (<$trait_matrix as Has2DReuseBuf>::FstOwnedBufferBool, <M2::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
                (<$trait_matrix as Has2DReuseBuf>::SndOwnedBufferBool, <M2::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
                (<$trait_matrix as Has2DReuseBuf>::FstHandleBool, <M2::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
                (<$trait_matrix as Has2DReuseBuf>::SndHandleBool, <M2::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
                (<$trait_matrix as Has2DReuseBuf>::IsFstBufferTransposed, <M2::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
                (<$trait_matrix as Has2DReuseBuf>::IsSndBufferTransposed, <M2::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
                (<$trait_matrix as Has2DReuseBuf>::AreBoundBuffersTransposed, <M2::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
                (N, <M2::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
                (N, <M2::Unwrapped as HasOutput>::OutputBool): FilterPair,
                (<(N, <M2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
                (N, <M2::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
                (N, <M2::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
                (N, <M2::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
                (N, <M2::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
                (N, <M2::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
                (N, <M2::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
                (N, <M2::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
                (N, <M2::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
            {
                type Output = <<<$ty as MatrixOps>::Builder as MatrixBuilderUnion<M2::Builder>>::Union as MatrixBuilder>::MatrixWrapped<MatAdd<$true_matrix, M2::Unwrapped>>;

                fn add(self, rhs: M2) -> Self::Output {
                    MatrixOps::add(self, rhs)
                }
            }

            impl<
                $($($lifetime),+, )? 
                $($generic: $($lifetime_bound |)? $($fst_trait_bound $(| $trait_bound)*)?),+,
                M2: MatrixOps
                $(, const $d1: usize, const $d2: usize)?
            > Sub<M2> for $ty where
                <$ty as MatrixOps>::Builder: MatrixBuilderUnion<M2::Builder>, 
                <$trait_matrix as Get2D>::Item: Sub<<M2::Unwrapped as Get2D>::Item>,
                (<$trait_matrix as Get2D>::AreInputsTransposed, <M2::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
                (<$trait_matrix as HasOutput>::OutputBool, <M2::Unwrapped as HasOutput>::OutputBool): FilterPair,
                (<(<$trait_matrix as HasOutput>::OutputBool, <M2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
                (<$trait_matrix as Has2DReuseBuf>::BoundHandlesBool, <M2::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
                (<$trait_matrix as Has2DReuseBuf>::FstOwnedBufferBool, <M2::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
                (<$trait_matrix as Has2DReuseBuf>::SndOwnedBufferBool, <M2::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
                (<$trait_matrix as Has2DReuseBuf>::FstHandleBool, <M2::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
                (<$trait_matrix as Has2DReuseBuf>::SndHandleBool, <M2::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
                (<$trait_matrix as Has2DReuseBuf>::IsFstBufferTransposed, <M2::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
                (<$trait_matrix as Has2DReuseBuf>::IsSndBufferTransposed, <M2::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
                (<$trait_matrix as Has2DReuseBuf>::AreBoundBuffersTransposed, <M2::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
                (N, <M2::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
                (N, <M2::Unwrapped as HasOutput>::OutputBool): FilterPair,
                (<(N, <M2::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
                (N, <M2::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
                (N, <M2::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
                (N, <M2::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
                (N, <M2::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
                (N, <M2::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
                (N, <M2::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
                (N, <M2::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
                (N, <M2::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
            {
                type Output = <<<$ty as MatrixOps>::Builder as MatrixBuilderUnion<M2::Builder>>::Union as MatrixBuilder>::MatrixWrapped<MatSub<$true_matrix, M2::Unwrapped>>;

                fn sub(self, rhs: M2) -> Self::Output {
                    MatrixOps::sub(self, rhs)
                }
            }
        )*
    };
}

impl_binary_ops_for_wrapper!(
    D1, D2; <M1: MatrixLike>, MatrixExpr<M1, D1, D2>, trait_matrix: M1, true_matrix: M1;
    D1, D2; <M1: MatrixLike>, Box<MatrixExpr<M1, D1, D2>>, trait_matrix: M1, true_matrix: Box<M1>;
    D1, D2; <'a, T1>, &'a MathMatrix<T1, D1, D2>, trait_matrix: &'a [[T1; D1]; D2], true_matrix: &'a [[T1; D1]; D2];
    D1, D2; <'a, T1>, &'a mut MathMatrix<T1, D1, D2>, trait_matrix: &'a mut [[T1; D1]; D2], true_matrix: &'a mut [[T1; D1]; D2];
    D1, D2; <'a, T1>, &'a Box<MathMatrix<T1, D1, D2>>, trait_matrix: &'a [[T1; D1]; D2], true_matrix: &'a [[T1; D1]; D2];
    D1, D2; <'a, T1>, &'a mut Box<MathMatrix<T1, D1, D2>>, trait_matrix: &'a mut [[T1; D1]; D2], true_matrix: &'a mut [[T1; D1]; D2];
);