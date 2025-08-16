//! Module containing all to do with Matrices and basic operations to do on them

use crate::{
    trait_specialization_utils::*,
    util_structs::NoneIter, 
    util_traits::HasOutput, 
    vector::MathVector
};
use std::{
    iter::{
        Product, 
        Sum
    }, 
    mem::{
        self, 
        ManuallyDrop, 
        MaybeUninit
    }, 
    ops::*, 
    ptr
};

pub mod mat_util_traits;
pub mod matrix_structs;
pub mod matrix_builders;

use mat_util_traits::*;
use matrix_builders::*;
use matrix_structs::*;


/// A const sized matrix wrapper
/// D1: # rows (dimension of vectors), D2: # columns (# of vectors)
// MatrixExpr assumes that the stored MatrixLike is fully unused
#[repr(transparent)]
pub struct MatrixExpr<M: MatrixLike, const D1: usize, const D2: usize>(M);

impl<M: MatrixLike, const D1: usize, const D2: usize> MatrixExpr<M, D1, D2> {
    /// converts the underlying VectorLike to a dynamic object
    /// stabilizes the overall type to a consitent one
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

    /// consumes the MatrixExpr and returns the built up output
    /// 
    /// Note:
    /// methods like sum, product, or fold can place build up outputs
    /// 
    /// output is generally nested 2 element tuples
    /// newer values to the right
    /// binary operators merge the output of the 2 matrixes
    #[inline] 
    pub fn consume(self) -> M::Output where M: Has2DReuseBuf<BoundTypes = M::BoundItems> {
        self.into_entry_iter().consume()
    }

    /// evaluates the MatrixExpr and returns the resulting matrix alongside its output (if present)
    /// if the MatrixExpr has no item (& thus results in a matrix w/ ZST elements) or the item is irrelevent, see consume to not return that matrix
    /// will try to use the first buffer if available (fails if the provided buffer is not bindable to the output)
    /// 
    /// Warning: 
    /// this method trying the evaluate the matrix *onto the stack*, it is very possible to overflow the stack with larger matrixes
    /// use heap_eval if this is a concern 
    /// 
    /// Note:
    /// methods like sum, product, or fold can place build up outputs
    /// 
    /// output is generally nested 2 element tuples
    /// newer values to the right
    /// binary operators merge the output of the 2 matrixes
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

    /// evaluates the MatrixExpr and returns the resulting matrix alongside its output (if present)
    /// if the MatrixExpr has no item (& thus results in a matrix w/ ZST elements) or the item is irrelevent, see consume to not return that matrix
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
    /// binary operators merge the output of the 2 matrixes
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

    /// Creates a column-major iterator across the matrix's elements
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

impl<M: MatrixLike + Is2DRepeatable, const D1: usize, const D2: usize> MatrixExpr<M,D1,D2> {
    /// Retrieves the value at an arbitrary index of a repeatable matrix
    /// Note:   This method does NOT fill any buffers bound to the matrix
    pub fn get(&mut self, col_index: usize, row_index: usize) -> M::Item {
        if (col_index >= D2) | (row_index >= D1) {panic!("math_vector Error: index access out of bound")}
        unsafe {
            let inputs = self.0.get_inputs(col_index, row_index);
            let (item, _) = self.0.process(col_index, row_index, inputs);
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
            self.0.drop_1st_buffer();
            self.0.drop_2nd_buffer();
        }
    }
}

/// A column majored iterator over a matrix's elements
pub struct MatrixEntryIter<M: MatrixLike, const D1: usize, const D2: usize>{mat: M, current_col: usize, live_input_row_start: usize, dead_output_row_start: usize}

impl<M: MatrixLike, const D1: usize, const D2: usize> MatrixEntryIter<M, D1, D2> {
    /// retrieve the output of this matrix without checking consumption
    /// Safety: the matrix must have been fully consumed
    #[inline]
    pub unsafe fn unchecked_output(self) -> M::Output {
        let mut man_drop_self = ManuallyDrop::new(self);
        let output;
        unsafe { 
            output = man_drop_self.mat.output();
            ptr::drop_in_place(&mut man_drop_self.mat);
        }
        output
    }

    /// retrieve the output of this matrix
    #[inline]
    pub fn output(self) -> M::Output {
        assert!((self.current_col == D2-1) & (self.live_input_row_start == D1), "math_vector error: A VectorIter must be fully used before outputting");
        debug_assert!(self.dead_output_row_start == D1, "math_vector internal error: A VectorIter's output buffers (somehow) weren't fully filled despite the inputs being fully used, likely an internal issue");
        unsafe {self.unchecked_output()}
    }

    /// fully consumes the matrix and returns its output
    #[inline]
    pub fn consume(mut self) -> M::Output where M: Has2DReuseBuf<BoundTypes = M::BoundItems> {
        self.no_output_consume();
        unsafe {self.unchecked_output()}
    }

    /// fully consumes the matrix without returning its output
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
                    let (_, bound_items) = mat.process(*current_col, row_index, inputs);
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
                let (_, bound_items) = mat.process(*current_col, row_index, inputs);
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
            self.mat.drop_output();
            self.mat.drop_1st_buffer();
            self.mat.drop_2nd_buffer();
        }
    }
}

impl<M: MatrixLike, const D1: usize, const D2: usize> Iterator for MatrixEntryIter<M, D1, D2> where M: Has2DReuseBuf<BoundTypes = M::BoundItems> {
    type Item = M::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.live_input_row_start < D1 { //current vector isn't done
                let row_index = self.live_input_row_start;
                self.live_input_row_start += 1;
                let inputs = self.mat.get_inputs(self.current_col, row_index);
                let (item, bound_items) = self.mat.process(self.current_col, row_index, inputs);
                self.mat.assign_bound_bufs(self.current_col, row_index, bound_items);
                self.dead_output_row_start += 1;
                Some(item)
            } else if self.current_col < D2-1 {
                self.current_col += 1;
                self.live_input_row_start = 1; //we immediately and infallibly get the first one
                self.dead_output_row_start = 0;
                let inputs = self.mat.get_inputs(self.current_col, 0);
                let (item, bound_items) = self.mat.process(self.current_col, 0, inputs);
                self.mat.assign_bound_bufs(self.current_col, 0, bound_items);
                self.dead_output_row_start += 1;
                Some(item)
            } else {
                None
            }
        }
    }
}

/// a simple type alias for MatrixExpr created from an array of type [[T; D1]; D2]
pub type MathMatrix<T, const D1: usize, const D2: usize> = MatrixExpr<Owned2DArray<T, D1, D2>, D1, D2>;

impl<T, const D1: usize, const D2: usize> MathMatrix<T, D1, D2> {
    #[inline] pub fn into_2d_array(self) -> [[T; D1]; D2] {self.unwrap().unwrap()}
    #[inline] pub fn into_2d_heap_array(self: Box<Self>) -> Box<[[T; D1]; D2]> {
        unsafe { mem::transmute::<Box<Self>, Box<[[T; D1]; D2]>>(self) }
    }
    /// Marks this MathMatrix to have its buffer reused
    /// buffer placed in fst slot
    #[inline] pub fn reuse(self) -> MatrixExpr<Replace2DArray<T, D1, D2>, D1, D2> {MatrixExpr(Replace2DArray(self.unwrap().0))}
    /// Marks this MathMatrix to have its buffer reused while keeping it on the heap
    /// buffer placed in fst slot
    #[inline] pub fn heap_reuse(self: Box<Self>) -> MatrixExpr<Box<Replace2DArray<T, D1, D2>>, D1, D2> {
        // Safety, series of equivilent types:
        // Box<MathMatrix<T, D1, D2>>
        // Box<MatrixExpr<Owned2DArray<T, D1, D2>, D1, D2>>, de-alias MathMatrix
        // Box<ManuallyDrop<[[T; D1]; D2]>>, MatrixExpr and Owned2DArray are transparent
        // MatrixExpr<Box<Replace2DArray<T, D1, D2>>, D1, D2>, MatrixExpr and Replace2DArray are transparent
        unsafe { mem::transmute::<Box<Self>, MatrixExpr<Box<Replace2DArray<T, D1, D2>>, D1, D2>>(self) }
    } 
    
    /// converts this MathMatrix to a repeatable MatrixExpr w/ Item = &'a T
    #[inline] pub fn referred<'a>(self) -> MatrixExpr<Referring2DArray<'a, T, D1, D2>, D1, D2> where T: 'a {
        MatrixExpr(Referring2DArray(unsafe { mem::transmute_copy::<ManuallyDrop<[[T; D1]; D2]>, [[T; D1]; D2]>(&self.unwrap().0)}, std::marker::PhantomData))
    }

    /// references the element at index without checking bounds
    /// safety: index is in bounds
    #[inline] pub unsafe fn get_unchecked<I: std::slice::SliceIndex<[[T; D1]]>>(&self, index: I) -> &I::Output { unsafe {
        self.0.0.get_unchecked(index)
    }}
    /// mutably references the element at index without checking bounds
    /// safety: index is in bounds
    #[inline] pub unsafe fn get_unchecked_mut<I: std::slice::SliceIndex<[[T; D1]]>>(&mut self, index: I) -> &mut I::Output { unsafe {
        self.0.0.get_unchecked_mut(index)
    }}

    
}

impl<const D1: usize, const D2: usize> MathMatrix<f64, D1, D2> {
    pub fn rref(&mut self) {
        use std::collections::HashMap;
        use std::cmp::min;

        let mut pivots = Vec::with_capacity(D1);
        for row_idx in 0..D1 {
            let mut pivot = 0;
            while (pivot < D2) && (self[pivot][row_idx] == 0.0) {pivot += 1;}
            pivots.push(pivot);
            if pivot == D2 {continue;}
            let base_val = self[pivot][row_idx];

            for j in pivot..D2 {
                self[j][row_idx] /= base_val;
            }

            
            let mut multipliers = Vec::with_capacity(D1 - 1);
            for i in 0..D1 {
                multipliers.push(self[pivot][i]);
                self[pivot][i] = 0.0;
            }
            self[pivot][row_idx] = 1.0;
            multipliers[row_idx] = 0.0;
            for j in pivot +1..D2 {
                let target_row_col_val = self[j][row_idx];
                for i in 0..D1 {
                    self[j][i] -= multipliers[i] * target_row_col_val;
                }
            }
        }
        let mut sorted_pivots = pivots.clone();
        sorted_pivots.sort_unstable();
        let mut pivot_idx_map = HashMap::with_capacity(D1);
        for (idx, pivot) in sorted_pivots.into_iter().enumerate() {
            if pivot == D2 {break;}
            pivot_idx_map.insert(pivot, idx);
        }
        for src in 0..D1 {
            if let Some(dst) = pivot_idx_map.get(&pivots[src]) {
                let dst = *dst;
                
                if src != dst {
                    for i in min(pivots[src], pivots[dst])..D2 {
                        let temp = self[i][src];
                        self[i][src] = self[i][dst];
                        self[i][dst] = temp;
                    }
                    let temp = pivots[src];
                    pivots[src] = pivots[dst];
                    pivots[dst] = temp;
                }
            }
        }
    }

    fn det_inner(&mut self) -> f64 {
        assert_eq!(D1, D2, "math_vector error: can't get the determinant of a {}x{} matrix (not square)", D1, D2);

        let mut out = 1.0;
        let mut pivots = Vec::with_capacity(D2);
        for col_idx in 0..D2 {
            let mut pivot = 0;
            while (pivot < D1) && (self[col_idx][pivot] == 0.0) {pivot += 1;}
            pivots.push(pivot);
            if pivot == D1 {return 0.0;}
            let base_val = self[col_idx][pivot];
            out *= base_val;

            for j in pivot..D1 {
                self[col_idx][j] /= base_val;
            }
            for i in col_idx +1..D2 {
                let multiplier = self[i][pivot];
                self[i][pivot] = 0.0;
                for j in pivot+1..D1 {
                    let sub = multiplier * self[col_idx][j];
                    self[i][j] -= sub;
                }
            }
        }
        
        for src in 0..D2 {
            if pivots[src] != src {
                out = -out;
                let dst = pivots[src];
                let temp = pivots[dst];
                pivots[dst] = pivots[src];
                pivots[src] = temp;
            }
        }
        out
    }

    #[inline(always)]
    pub fn det(mut self) -> f64 {self.det_inner()}

    #[inline(always)]
    pub fn det_heap(mut self: Box<Self>) -> f64 {self.det_inner()}
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
        MatrixExpr(Owned2DArray(ManuallyDrop::new(value)))
    }
}

impl<T, const D1: usize, const D2: usize> Into<[[T; D1]; D2]> for MathMatrix<T, D1, D2> {
    #[inline] fn into(self) -> [[T; D1]; D2] {self.into_2d_array()}
}

impl<'a, T, const D1: usize, const D2: usize> From<&'a [[T; D1]; D2]> for &'a MathMatrix<T, D1, D2> {
    #[inline]
    fn from(value: &'a [[T; D1]; D2]) -> Self {
        unsafe { mem::transmute::<&'a [[T; D1]; D2], &'a MathMatrix<T, D1, D2>>(value) }
    }
}

impl<'a, T, const D1: usize, const D2: usize> Into<&'a [[T; D1]; D2]> for &'a MathMatrix<T, D1, D2> {
    #[inline]
    fn into(self) -> &'a [[T; D1]; D2] {
        unsafe { mem::transmute::<&'a MathMatrix<T, D1, D2>, &'a [[T; D1]; D2]>(self) }
    }
}

impl<'a, T, const D1: usize, const D2: usize> From<&'a mut [[T; D1]; D2]> for &'a mut MathMatrix<T, D1, D2> {
    #[inline]
    fn from(value: &'a mut [[T; D1]; D2]) -> Self {
        unsafe { mem::transmute::<&'a mut [[T; D1]; D2], &'a mut MathMatrix<T, D1, D2>>(value) }
    }
}

impl<'a, T, const D1: usize, const D2: usize> Into<&'a mut [[T; D1]; D2]> for &'a mut MathMatrix<T, D1, D2> {
    #[inline]
    fn into(self) -> &'a mut [[T; D1]; D2] {
        unsafe { mem::transmute::<&'a mut MathMatrix<T, D1, D2>, &'a mut [[T; D1]; D2]>(self) }
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
        unsafe { mem::transmute_copy::<MathVectoredMatrix<T, D1, D2>, MathMatrix<T, D1, D2>>(&ManuallyDrop::new(value)) }
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
        unsafe { mem::transmute_copy::<MathMatrix<T, D1, D2>, MathVectoredMatrix<T, D1, D2>>(&ManuallyDrop::new(self)) }
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


pub type MathVectoredMatrix<T, const D1: usize, const D2: usize> = MathVector<MathVector<T, D1>, D2>;

/// generates a Matrix of dimensions D1, D2 using the given closure (FnMut) with no inputs
pub fn matrix_gen<F: FnMut() -> O, O, const D1: usize, const D2: usize>(f: F) -> MatrixExpr<MatGenerator<F, O>, D1, D2> {
    MatrixExpr(MatGenerator(f))
}

/// generates a Matrix of dimensions D1, D2 using the given closure (FnMut) given the column and row indices as input
pub fn matrix_index_gen<F: FnMut(usize, usize) -> O, O, const D1: usize, const D2: usize>(f: F) -> MatrixExpr<MatIndexGenerator<F, O>, D1, D2> {
    MatrixExpr(MatIndexGenerator(f))
}

/// generates a Identity matrix of dimensions D, D
/// 
/// the "1" value is obtained from Product as the multiplicative identity
/// the "0" value is obtained from Sum as the additive identity
pub fn matrix_identiry_gen<T: Copy + Sum + Product, const D: usize>() -> MatrixExpr<MatIdentityGenerator<T>, D, D> {
    MatrixExpr(MatIdentityGenerator { zero: NoneIter::<T>::new().sum(), one: NoneIter::<T>::new().product() })
}

/// a trait with various matrix operations
pub trait MatrixOps {
    /// the underlying MatrixLike contained in Self
    type Unwrapped: MatrixLike;
    /// the type which builds the Wrapper around a MatrixLike
    type Builder: MatrixBuilder;

    /// get the underlying MatrixLike
    fn unwrap(self) -> Self::Unwrapped;
    /// get the builder for this Matrix's wrapper
    fn get_builder(&self) -> Self::Builder;
    /// get the dimensions of this matrix
    fn dimensions(&self) -> (usize, usize); // 0: num rows, 1: num columns

    /// binds the matrix's item to the buffer in the first slot, adding it to Output if owned
    #[inline]
    fn bind(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatBind<Self::Unwrapped>>
    where
        Self::Unwrapped: Has2DReuseBuf<FstHandleBool = Y>,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, Y): FilterPair,
        (<Self::Unwrapped as HasOutput>::OutputBool, <Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
        MatBind<Self::Unwrapped>: Has2DReuseBuf<BoundTypes = <MatBind<Self::Unwrapped> as Get2D>::BoundItems>,
        Self: Sized
    {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatBind{mat: self.unwrap()}) }
    }

    #[inline]
    fn map_bind<F: FnMut(<Self::Unwrapped as Get2D>::Item) -> (I, B), I, B>(self, f: F) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatMapBind<Self::Unwrapped, F, I, B>>
    where
        Self::Unwrapped: Has2DReuseBuf<FstHandleBool = Y>,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, Y): FilterPair,
        (<Self::Unwrapped as HasOutput>::OutputBool, <Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
        MatMapBind<Self::Unwrapped, F, I, B>: Has2DReuseBuf<BoundTypes = <MatMapBind<Self::Unwrapped, F, I, B> as Get2D>::BoundItems>,
        Self: Sized
    {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatMapBind{mat: self.unwrap(), f}) }
    }

    /// binds the matrix's item to its fst buffer, adding the buffer to an internal output if owned by the matrix
    /// 
    /// Note: 
    /// this internal output is not readily accessible and doesn't add much over bind
    /// As such, end users should generally just use bind
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

    /// swaps the vector's first and second buffer slots
    #[inline]
    fn buf_swap(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatBufSwap<Self::Unwrapped>> where (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatBufSwap{mat: self.unwrap()}) }
    }

    /// offsets (with rolling over) each element of the vector left by the given offset
    #[inline]
    fn offset_left(self, offset: usize) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatColOffset<Self::Unwrapped>> where Self: Sized {
        let (_, cols) = self.dimensions();
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatColOffset{mat: self.unwrap(), offset: offset % cols, num_columns: cols}) }
    }

    /// offsets (with rolling over) each element of the vector up by the given offset
    #[inline]
    fn offset_up(self, offset: usize) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatRowOffset<Self::Unwrapped>> where Self: Sized {
        let (rows, _) = self.dimensions();
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatRowOffset{mat: self.unwrap(), offset: offset % rows, num_rows: rows}) }
    }

    /// offsets (with rolling over) each element of the vector right by the given offset
    #[inline]
    fn offset_right(self, offset: usize) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatColOffset<Self::Unwrapped>> where Self: Sized {
        let (_, cols) = self.dimensions();
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatColOffset{mat: self.unwrap(), offset: cols - (offset % cols), num_columns: cols}) }
    }

    /// offsets (with rolling over) each element of the vector down by the given offset
    #[inline]
    fn offset_down(self, offset: usize) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatRowOffset<Self::Unwrapped>> where Self: Sized {
        let (rows, _) = self.dimensions();
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatRowOffset{mat: self.unwrap(), offset: rows - (offset % rows), num_rows: rows}) }
    }

    /// reverses the order of the matrix's columns
    #[inline] 
    fn reverse_cols(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatColReverse<Self::Unwrapped>> where Self: Sized {
        let (_, cols) = self.dimensions();
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatColReverse { mat: self.unwrap(), max_col_index: cols -1 })}
    }

    /// reverses the order of the matrix's rows
    #[inline] 
    fn reverse_rows(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatRowReverse<Self::Unwrapped>> where Self: Sized {
        let (rows, _) = self.dimensions();
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatRowReverse { mat: self.unwrap(), max_row_index: rows -1 })}
    }

    /// transposes the matrix
    #[inline]
    fn transpose(self) -> <Self::Builder as MatrixBuilder>::TransposedMatrixWrapped<MatTranspose<Self::Unwrapped>> where Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_trans_mat(MatTranspose { mat: self.unwrap() })}
    }

    /// converts the matrix into a VectorExpr of the columns (which are VectorExprs)
    #[inline]
    fn columns(self) -> <Self::Builder as MatrixBuilder>::RowWrapped<MatColWrapper<MatColVectorExprs<Self::Unwrapped>, MatrixColumn<Self::Unwrapped>, Self::Builder>> where Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_row_vec(MatColWrapper{mat: MatColVectorExprs{mat: self.unwrap()}, builder: builder.clone()}) }
    }

    /// converts the matrix into a VectorExpr of the rows (which are VectorExprs)
    #[inline]
    fn rows(self) -> <Self::Builder as MatrixBuilder>::ColWrapped<MatRowWrapper<MatRowVectorExprs<Self::Unwrapped>, MatrixRow<Self::Unwrapped>, Self::Builder>> where Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_col_vec(MatRowWrapper{mat: MatRowVectorExprs{mat: self.unwrap()}, builder: builder.clone()}) }
    }

    /// maps each entry of the matrix using the given closure
    #[inline]
    fn entry_map<F: FnMut(<Self::Unwrapped as Get2D>::Item) -> O, O>(self, f: F) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatEntryMap<Self::Unwrapped, F, O>> where (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatEntryMap{mat: self.unwrap(), f}) }
    }

    /// folds the matrix's items into a single value added to Output using the provided closure
    /// note: fold_ref should be used whenever possible due to implementation
    #[inline]
    fn entry_fold<F: FnMut(O, <Self::Unwrapped as Get2D>::Item) -> O, O>(self, f: F, init: O) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatEntryFold<Self::Unwrapped, F, O>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatEntryFold{mat: self.unwrap(), f, cell: Some(init)}) }
    }

    /// folds the matrix's items into a single value added to Output using the provided closure
    #[inline]
    fn entry_fold_ref<F: FnMut(&mut O, <Self::Unwrapped as Get2D>::Item), O>(self, f: F, init: O) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatEntryFoldRef<Self::Unwrapped, F, O>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatEntryFoldRef{mat: self.unwrap(), f, cell: ManuallyDrop::new(init)}) }
    }

    /// folds the matrix's items into a single value added to Output using the provided closure while preserving the item
    /// note: copied_fold_ref should be used whenever possible due to implementation
    #[inline]
    fn entry_copied_fold<F: FnMut(O, <Self::Unwrapped as Get2D>::Item) -> O, O>(self, f: F, init: O) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatEntryCopiedFold<Self::Unwrapped, F, O>> where <Self::Unwrapped as Get2D>::Item: Copy, (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatEntryCopiedFold{mat: self.unwrap(), f, cell: Some(init)}) }
    }

    /// folds the matrix's items into a single value added to Output using the provided closure while preserving the item
    #[inline]
    fn entry_copied_fold_ref<F: FnMut(&mut O, <Self::Unwrapped as Get2D>::Item), O>(self, f: F, init: O) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatEntryCopiedFoldRef<Self::Unwrapped, F, O>> where <Self::Unwrapped as Get2D>::Item: Copy, (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatEntryCopiedFoldRef{mat: self.unwrap(), f, cell: ManuallyDrop::new(init)}) }
    }

    /// copies each of the matrix's items, useful for turing &T -> T
    #[inline]
    fn copied<'a, I: 'a + Copy>(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatCopy<'a, Self::Unwrapped, I>> where Self::Unwrapped: Get2D<Item = &'a I>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatCopy{mat: self.unwrap()}) }
    }

    /// clones each of the matrix's items, useful for turing &T -> T
    #[inline]
    fn cloned<'a, I: 'a + Clone>(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatClone<'a, Self::Unwrapped, I>> where Self::Unwrapped: Get2D<Item = &'a I>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatClone{mat: self.unwrap()}) }
    }

    /// negates (the - unary operation) each of the matrix's items
    #[inline] 
    fn neg(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatNeg<Self::Unwrapped>> where <Self::Unwrapped as Get2D>::Item: Neg, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatNeg{mat: self.unwrap()})}
    }

    /// multiples a scalar with the matrix (matrix items are rhs) (*may* be identitical to mul_l) 
    #[inline]
    fn mul_r<S: Copy>(self, scalar: S) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatMulR<Self::Unwrapped, S>> where S: Mul<<Self::Unwrapped as Get2D>::Item>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatMulR{mat: self.unwrap(), scalar})}
    }

    /// divides a scalar with the matrix
    #[inline]
    fn div_r<S: Copy>(self, scalar: S) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatDivR<Self::Unwrapped, S>> where S: Div<<Self::Unwrapped as Get2D>::Item>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatDivR{mat: self.unwrap(), scalar})}
    }

    /// gets the remainder (ie. %) of a scalar with the matrix
    #[inline]
    fn rem_r<S: Copy>(self, scalar: S) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatRemR<Self::Unwrapped, S>> where S: Rem<<Self::Unwrapped as Get2D>::Item>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatRemR{mat: self.unwrap(), scalar})}
    }

    /// multiples the matrix with a scalar (matrix items are lhs) (*may* be identitical to mul_l) 
    #[inline]
    fn mul_l<S: Copy>(self, scalar: S) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatMulL<Self::Unwrapped, S>> where <Self::Unwrapped as Get2D>::Item: Mul<S>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatMulL{mat: self.unwrap(), scalar})}
    }

    /// divides the matrix with a scalar
    #[inline]
    fn div_l<S: Copy>(self, scalar: S) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatDivL<Self::Unwrapped, S>> where <Self::Unwrapped as Get2D>::Item: Div<S>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatDivL{mat: self.unwrap(), scalar})}
    }

    /// gets the remainder (ie. %) of the matrix with a scalar
    #[inline]
    fn rem_l<S: Copy>(self, scalar: S) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatRemL<Self::Unwrapped, S>> where <Self::Unwrapped as Get2D>::Item: Rem<S>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatRemL{mat: self.unwrap(), scalar})}
    }

    /// mul assigns (*=) the matrix's items (&mut T) with a scalar
    #[inline]
    fn mul_assign<'a, I: 'a + MulAssign<S>, S: Copy>(self, scalar: S) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatMulAssign<'a, Self::Unwrapped, I, S>> where Self::Unwrapped: Get2D<Item = &'a mut I>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatMulAssign{mat: self.unwrap(), scalar}) }
    }

    /// div assigns (/=) the matrix's items (&mut T) with a scalar
    #[inline]
    fn div_assign<'a, I: 'a + DivAssign<S>, S: Copy>(self, scalar: S) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatDivAssign<'a, Self::Unwrapped, I, S>> where Self::Unwrapped: Get2D<Item = &'a mut I>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatDivAssign{mat: self.unwrap(), scalar}) }
    }

    /// rem assigns (%=) the matrix's items (&mut T) with a scalar
    #[inline]
    fn rem_assign<'a, I: 'a + RemAssign<S>, S: Copy>(self, scalar: S) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatRemAssign<'a, Self::Unwrapped, I, S>> where Self::Unwrapped: Get2D<Item = &'a mut I>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatRemAssign{mat: self.unwrap(), scalar}) }
    }

    /// calculates the sum of the Matrix's entries and adds it to Output
    #[inline]
    fn entry_sum<S: Sum<<Self::Unwrapped as Get2D>::Item> + AddAssign<<Self::Unwrapped as Get2D>::Item>>(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatEntrySum<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatEntrySum{mat: self.unwrap(), scalar: ManuallyDrop::new(NoneIter::<<Self::Unwrapped as Get2D>::Item>::new().sum())}) }
    } 
    
    /// calculates the sum (initialized at `init`) of the Matrix's entries and adds it to Output
    #[inline]
    fn initialized_entry_sum<S: AddAssign<<Self::Unwrapped as Get2D>::Item>>(self, init: S) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatEntrySum<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatEntrySum{mat: self.unwrap(), scalar: ManuallyDrop::new(init)}) }
    } 
    
    /// calculates the sum of the Matrix's entries and adds it to Output while preserving the item
    #[inline]
    fn copied_entry_sum<S: Sum<<Self::Unwrapped as Get2D>::Item> + AddAssign<<Self::Unwrapped as Get2D>::Item>>(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatCopiedEntrySum<Self::Unwrapped, S>> where <Self::Unwrapped as Get2D>::Item: Copy, (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatCopiedEntrySum{mat: self.unwrap(), scalar: ManuallyDrop::new(NoneIter::<<Self::Unwrapped as Get2D>::Item>::new().sum())}) }
    } 
    
    /// calculates the sum (initialized at `init`) of the Matrix's entries and adds it to Output while preserving the item
    #[inline]
    fn initialized_copied_entry_sum<S: AddAssign<<Self::Unwrapped as Get2D>::Item>>(self, init: S) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatCopiedEntrySum<Self::Unwrapped, S>> where <Self::Unwrapped as Get2D>::Item: Copy, (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatCopiedEntrySum{mat: self.unwrap(), scalar: ManuallyDrop::new(init)}) }
    } 
    
    /// calculates the product of the Matrix's entries and adds it to Output
    #[inline]
    fn entry_product<S: Product<<Self::Unwrapped as Get2D>::Item> + MulAssign<<Self::Unwrapped as Get2D>::Item>>(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatEntryProd<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatEntryProd{mat: self.unwrap(), scalar: ManuallyDrop::new(NoneIter::<<Self::Unwrapped as Get2D>::Item>::new().product())}) }
    } 
    
    /// calculates the product (initialized at `init`) of the Matrix's entries and adds it to Output
    #[inline]
    fn initialized_entry_product<S: MulAssign<<Self::Unwrapped as Get2D>::Item>>(self, init: S) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatEntryProd<Self::Unwrapped, S>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatEntryProd{mat: self.unwrap(), scalar: ManuallyDrop::new(init)}) }
    } 
    
    /// calculates the product of the Matrix's entries and adds it to Output while preserving the item
    #[inline]
    fn copied_entry_product<S: Product<<Self::Unwrapped as Get2D>::Item> + MulAssign<<Self::Unwrapped as Get2D>::Item>>(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatCopiedEntryProd<Self::Unwrapped, S>> where <Self::Unwrapped as Get2D>::Item: Copy, (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatCopiedEntryProd{mat: self.unwrap(), scalar: ManuallyDrop::new(NoneIter::<<Self::Unwrapped as Get2D>::Item>::new().product())}) }
    } 
    
    /// calculates the product (initialized at `init`) of the Matrix's entries and adds it to Output while preserving the item
    #[inline]
    fn initialized_copied_entry_product<S: MulAssign<<Self::Unwrapped as Get2D>::Item>>(self, init: S) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatCopiedEntryProd<Self::Unwrapped, S>> where <Self::Unwrapped as Get2D>::Item: Copy, (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatCopiedEntryProd{mat: self.unwrap(), scalar: ManuallyDrop::new(init)}) }
    } 
    
    /// multiplies 2 matrices
    #[inline]
    fn mat_mul<M: MatrixOps>(self, other: M) -> <
        <
            <Self::Builder as MatrixBuilder>::ColBuilder as MatrixBuilderCompose<
                <M::Builder as MatrixBuilder>::RowBuilder
            >
        >::Composition as MatrixBuilder
    >::MatrixWrapped<
        FullMatMul<Self::Unwrapped, M::Unwrapped>
    > 
    where 
        Self::Unwrapped: Is2DRepeatable,
        M::Unwrapped: Is2DRepeatable,
        <Self::Builder as MatrixBuilder>::ColBuilder: MatrixBuilderCompose<<M::Builder as MatrixBuilder>::RowBuilder>,
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
        let builder = self.get_builder().decompose().0.compose(other.get_builder().decompose().1);
        unsafe { builder.wrap_mat(FullMatMul{l_mat: self.unwrap(), r_mat: other.unwrap(), shared_size}) }
    }

    /// zips together the items of 2 matrices into 2 element tuples
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

    /// adds 2 matrices
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

    /// substracts the other matrix from self
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

    /// component-wise multiplies 2 matrices
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

    /// component-wise divides self by other
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

    /// component-wise get remainder (%) of self by other
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

    /// add assigns (+=) self's items (&mut T) with other 
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

    /// sub assigns (-=) self's items (&mut T) with other
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

    /// mul assigns (*=) self's items (&mut T) with other
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

    /// div assigns (/=) self's items (&mut T) with other
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

    /// rem assigns (%=) self's items (&mut T) with other
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

/// a trait various matrix operations for const sized matrix
pub trait ArrayMatrixOps<const D1: usize, const D2: usize>: MatrixOps {
    /// attaches a &mut MathMatrix to the first buffer
    #[inline]
    fn attach_2d_buf<'a, T>(self, buf: &'a mut MathMatrix<T, D1, D2>) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatAttach2DBuf<'a, Self::Unwrapped, T, D1, D2>> where Self::Unwrapped: Has2DReuseBuf<FstHandleBool = N>, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatAttach2DBuf{mat: self.unwrap(), buf}) }
    }

    /// creates a buffer in the first buffer
    #[inline]
    fn create_2d_buf<T>(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatCreate2DBuf<Self::Unwrapped, T, D1, D2>> where Self::Unwrapped: Has2DReuseBuf<FstHandleBool = N>, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatCreate2DBuf{mat: self.unwrap(), buf: MaybeUninit::uninit().assume_init()}) }
    }

    /// creates a buffer on the heap in the first buffer
    #[inline]
    fn create_2d_heap_buf<T>(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatCreate2DHeapBuf<Self::Unwrapped, T, D1, D2>> where Self::Unwrapped: Has2DReuseBuf<FstHandleBool = N>, Self: Sized {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatCreate2DHeapBuf{mat: self.unwrap(), buf: ManuallyDrop::new(Box::new(MaybeUninit::uninit().assume_init()))}) }
    }

    /// creates a buffer in the first buffer if there isn't already one there
    #[inline]
    fn maybe_create_2d_buf<T>(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatMaybeCreate2DBuf<Self::Unwrapped, T, D1, D2>> 
    where 
        <<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg): TyBoolPair,
        Self: Sized
    {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatMaybeCreate2DBuf{mat: self.unwrap(), buf: MaybeUninit::uninit().assume_init()}) }
    }

    /// creates a buffer on the heap in the first buffer if there isn't already one there
    /// note: a pre-existing buffer may or may not be on the heap or owned by the vector
    #[inline]
    fn maybe_create_2d_heap_buf<T>(self) -> <Self::Builder as MatrixBuilder>::MatrixWrapped<MatMaybeCreate2DHeapBuf<Self::Unwrapped, T, D1, D2>> 
    where 
        <<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg): TyBoolPair,
        Self: Sized
    {
        let builder = self.get_builder();
        unsafe { builder.wrap_mat(MatMaybeCreate2DHeapBuf{mat: self.unwrap(), buf: ManuallyDrop::new(Box::new(MaybeUninit::uninit().assume_init()))}) }
    }
}

/// a trait enabling a matrix to be made repeatable
pub trait RepeatableMatrixOps: MatrixOps {
    /// the underlying repeatable MatrixLike to be returned
    type RepeatableMatrix<'a>: MatrixLike + Is2DRepeatable where Self: 'a;
    /// the underlying MatrixLike used to make the matrix repeatable
    type UsedMatrix: MatrixLike;
    //type HeapedUsedMatrix: MatrixLike;

    /// turns the matrix into a repeatable one
    /// note: 
    /// this is *non-trivial*,
    /// in this process, the original matrix has to be evaluated and stored, needing computation & memory
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
        unsafe { ptr::read(&ManuallyDrop::new(self).0) } 
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
            builder.wrap_mat(MatAttachUsedMat{mat: mat_iter.mat.get_bound_buf().referred().unwrap(), used_mat: ptr::read(&mat_iter.mat)})
        }
    }
}


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
        unsafe { Box::new(ptr::read(&ManuallyDrop::new(self).0)) } 
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
            builder.wrap_mat(MatAttachUsedMat{mat: mat_iter.mat.get_bound_buf().referred().unwrap(), used_mat: ptr::read(&mat_iter.mat)})
        }
    }
}



impl<'a, T, const D1: usize, const D2: usize> MatrixOps for &'a MathMatrix<T, D1, D2> {
    type Unwrapped = &'a [[T; D1]; D2];
    type Builder = MatrixExprBuilder<D1, D2>;
    
    #[inline] fn unwrap(self) -> Self::Unwrapped {&self.0}
    #[inline] fn get_builder(&self) -> Self::Builder {MatrixExprBuilder}
    #[inline] fn dimensions(&self) -> (usize, usize) {(D1, D2)}
}
impl<'a, T, const D1: usize, const D2: usize> ArrayMatrixOps<D1, D2> for &'a MathMatrix<T, D1, D2> {}

impl<'a, T, const D1: usize, const D2: usize> MatrixOps for &'a mut MathMatrix<T, D1, D2> {
    type Unwrapped = &'a mut [[T; D1]; D2];
    type Builder = MatrixExprBuilder<D1, D2>;
    
    #[inline] fn unwrap(self) -> Self::Unwrapped {&mut self.0}
    #[inline] fn get_builder(&self) -> Self::Builder {MatrixExprBuilder}
    #[inline] fn dimensions(&self) -> (usize, usize) {(D1, D2)}
}
impl<'a, T, const D1: usize, const D2: usize> ArrayMatrixOps<D1, D2> for &'a mut MathMatrix<T, D1, D2> {}

impl<'a, T, const D1: usize, const D2: usize> MatrixOps for &'a Box<MathMatrix<T, D1, D2>> {
    type Unwrapped = &'a [[T; D1]; D2];
    type Builder = MatrixExprBuilder<D1, D2>;
    
    #[inline] fn unwrap(self) -> Self::Unwrapped {&self.0}
    #[inline] fn get_builder(&self) -> Self::Builder {MatrixExprBuilder}
    #[inline] fn dimensions(&self) -> (usize, usize) {(D1, D2)}
}
impl<'a, T, const D1: usize, const D2: usize> ArrayMatrixOps<D1, D2> for &'a Box<MathMatrix<T, D1, D2>> {}

impl<'a, T, const D1: usize, const D2: usize> MatrixOps for &'a mut Box<MathMatrix<T, D1, D2>> {
    type Unwrapped = &'a mut [[T; D1]; D2];
    type Builder = MatrixExprBuilder<D1, D2>;
    
    #[inline] fn unwrap(self) -> Self::Unwrapped {&mut self.0}
    #[inline] fn get_builder(&self) -> Self::Builder {MatrixExprBuilder}
    #[inline] fn dimensions(&self) -> (usize, usize) {(D1, D2)}
}
impl<'a, T, const D1: usize, const D2: usize> ArrayMatrixOps<D1, D2> for &'a mut Box<MathMatrix<T, D1, D2>> {}


macro_rules! impl_ops_for_wrapper {
    (
        $(
            <$($($lifetime:lifetime),+, )? $($generic:ident $(:)? $($lifetime_bound:lifetime |)? $($fst_trait_bound:path $(| $trait_bound:path)*)?,)+ $({$d1:ident, $d2:ident})?>,
            $ty:ty,
            trait_matrix: $trait_matrix:ty,
            true_matrix: $true_matrix:ty;
        )*
    ) => {
        $(
            impl<$($($lifetime),+, )? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: Copy, $(const $d1: usize, const $d2: usize)?> Mul<Z> for $ty where (<$trait_matrix as HasOutput>::OutputBool, N): FilterPair, <$trait_matrix as Get2D>::Item: Mul<Z>, Self: Sized {
                type Output = <<$ty as MatrixOps>::Builder as MatrixBuilder>::MatrixWrapped<MatMulL<$true_matrix, Z>>;
            
                #[inline]
                fn mul(self, rhs: Z) -> Self::Output {
                    self.mul_l(rhs)
                }
            }
        
            impl<$($($lifetime),+, )? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: Copy, $(const $d1: usize, const $d2: usize)?> Div<Z> for $ty where (<$trait_matrix as HasOutput>::OutputBool, N): FilterPair, <$trait_matrix as Get2D>::Item: Div<Z>, Self: Sized {
                type Output = <<$ty as MatrixOps>::Builder as MatrixBuilder>::MatrixWrapped<MatDivL<$true_matrix, Z>>;
            
                #[inline]
                fn div(self, rhs: Z) -> Self::Output {
                    self.div_l(rhs)
                }
            }
        
            impl<$($($lifetime),+, )? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: Copy, $(const $d1: usize, const $d2: usize)?> Rem<Z> for $ty where (<$trait_matrix as HasOutput>::OutputBool, N): FilterPair, <$trait_matrix as Get2D>::Item: Rem<Z>, Self: Sized {
                type Output = <<$ty as MatrixOps>::Builder as MatrixBuilder>::MatrixWrapped<MatRemL<$true_matrix, Z>>;
            
                #[inline]
                fn rem(self, rhs: Z) -> Self::Output {
                    self.rem_l(rhs)
                }
            }
        
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

impl_ops_for_wrapper!(
    <M1: MatrixLike, {D1, D2}>, MatrixExpr<M1, D1, D2>, trait_matrix: M1, true_matrix: M1;
    <M1: MatrixLike, {D1, D2}>, Box<MatrixExpr<M1, D1, D2>>, trait_matrix: M1, true_matrix: Box<M1>;
    <'a, T1, {D1, D2}>, &'a MathMatrix<T1, D1, D2>, trait_matrix: &'a [[T1; D1]; D2], true_matrix: &'a [[T1; D1]; D2];
    <'a, T1, {D1, D2}>, &'a mut MathMatrix<T1, D1, D2>, trait_matrix: &'a mut [[T1; D1]; D2], true_matrix: &'a mut [[T1; D1]; D2];
    <'a, T1, {D1, D2}>, &'a Box<MathMatrix<T1, D1, D2>>, trait_matrix: &'a [[T1; D1]; D2], true_matrix: &'a [[T1; D1]; D2];
    <'a, T1, {D1, D2}>, &'a mut Box<MathMatrix<T1, D1, D2>>, trait_matrix: &'a mut [[T1; D1]; D2], true_matrix: &'a mut [[T1; D1]; D2];
);