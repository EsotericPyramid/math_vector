use crate::{util_traits::HasOutput, vector::{vec_util_traits::VectorLike, MathVector, VectorExpr}};
use self::mat_util_traits::{Get2D, Has2DReuseBuf, MatrixLike};
use crate::trait_specialization_utils::*;
use std::ops::*;

pub mod mat_util_traits {
    use crate::vector::vec_util_traits::VectorLike;

    // Note: traits here aren't meant to be used by end users
    use crate::trait_specialization_utils::TyBool;
    use crate::util_traits::HasOutput;

    pub unsafe trait Get2D {
        type GetBool: TyBool;
        type IsRepeatable: TyBool;
        type AreInputsTransposed: TyBool; // used to optimize access order
        type Inputs;
        type Item;
        type BoundItems;

        unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs; 

        unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize);

        fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems);

        #[inline]
        unsafe fn get(&mut self, col_index: usize, row_index: usize) -> (Self::Item, Self::BoundItems) { unsafe {
            let inputs = self.get_inputs(col_index,row_index);
            self.process(inputs)
        }}
    }

    pub trait Has2DReuseBuf {
        type FstHandleBool: TyBool;
        type SndHandleBool: TyBool;
        type BoundHandlesBool: TyBool;
        type FstOwnedBufferBool: TyBool;
        type SndOwnedBufferBool: TyBool;
        type IsFstBufferTransposed: TyBool;
        type IsSndBufferTransposed: TyBool;
        type AreBoundBuffersTransposed: TyBool;
        type FstOwnedBuffer;
        type SndOwnedBuffer;
        type FstType;
        type SndType;
        type BoundTypes;

        unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType); 
        unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType);
        unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes);
        unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize);
        unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize);
        unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize);
        unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer;
        unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer;
    }

    pub trait MatrixWrapperBuilder: Clone {
        type MatrixWrapped<T: MatrixLike>;
        type TransposedMatrixWrapped<T: MatrixLike>;
        type VectorWrapped<T: VectorLike>;
        type TransposedVectorWrapped<T: VectorLike>;

        unsafe fn wrap_mat<T: MatrixLike>(&self,mat: T) -> Self::MatrixWrapped<T>;
        unsafe fn wrap_trans_mat<T: MatrixLike>(&self,mat: T) -> Self::TransposedMatrixWrapped<T>;
        unsafe fn wrap_vec<T: VectorLike>(&self,vec: T) -> Self::VectorWrapped<T>;
        unsafe fn wrap_trans_vec<T: VectorLike>(&self,vec: T) -> Self::TransposedVectorWrapped<T>;        
    }

    ///really just a shorthand for the individual traits
    pub trait MatrixLike: Get2D + HasOutput + Has2DReuseBuf {}

    impl<T: Get2D + HasOutput + Has2DReuseBuf> MatrixLike for T {}
}


pub mod matrix_structs;
pub mod vectorized_matrix_structs;
use mat_util_traits::MatrixWrapperBuilder;
use matrix_structs::*;
use vectorized_matrix_structs::*;

/// D1: # rows (dimension of vectors), D2: # columns (# of vectors)
// MatrixExpr assumes that the stored MatrixLike is fully unused
#[repr(transparent)]
pub struct MatrixExpr<M: MatrixLike,const D1: usize,const D2: usize>(M);

impl<M: MatrixLike,const D1: usize,const D2: usize> MatrixExpr<M,D1,D2> {
    #[inline] 
    pub fn consume(self) -> M::Output where M: Has2DReuseBuf<BoundTypes = M::BoundItems> {
        self.into_entry_iter().consume()
    }

    #[inline]
    pub fn eval(self) -> <MatBind<MatMaybeCreate2DBuf<M,M::Item,D1,D2>> as HasOutput>::Output 
    where
        (M::FstHandleBool,<M::FstHandleBool as TyBool>::Neg): SelectPair,
        (M::FstOwnedBufferBool,<M::FstHandleBool as TyBool>::Neg): TyBoolPair,
        (M::OutputBool,<(M::FstOwnedBufferBool,<M::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or): FilterPair,
        <M::FstHandleBool as TyBool>::Neg: Filter,
        (M::BoundHandlesBool,Y): FilterPair,
        (M::IsFstBufferTransposed,M::AreBoundBuffersTransposed): TyBoolPair,
        MatBind<MatMaybeCreate2DBuf<M,M::Item,D1,D2>>: Has2DReuseBuf<BoundTypes = <MatBind<MatMaybeCreate2DBuf<M,M::Item,D1,D2>> as Get2D>::BoundItems>
    {
        self.maybe_create_2d_buf().bind().consume()
    }

    #[inline]
    pub fn heap_eval(self) -> <MatBind<MatMaybeCreate2DHeapBuf<M,M::Item,D1,D2>> as HasOutput>::Output 
    where
        (M::FstHandleBool,<M::FstHandleBool as TyBool>::Neg): SelectPair,
        (M::FstOwnedBufferBool,<M::FstHandleBool as TyBool>::Neg): TyBoolPair,
        (M::OutputBool,<(M::FstOwnedBufferBool,<M::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or): FilterPair,
        <M::FstHandleBool as TyBool>::Neg: Filter,
        (M::BoundHandlesBool,Y): FilterPair,
        (M::IsFstBufferTransposed,M::AreBoundBuffersTransposed): TyBoolPair,
        MatBind<MatMaybeCreate2DHeapBuf<M,M::Item,D1,D2>>: Has2DReuseBuf<BoundTypes = <MatBind<MatMaybeCreate2DHeapBuf<M,M::Item,D1,D2>> as Get2D>::BoundItems>
    {
        self.maybe_create_2d_heap_buf().bind().consume()
    }


    #[inline]
    pub fn into_entry_iter(self) -> MatrixEntryIter<M,D1,D2> {
        MatrixEntryIter{
            mat: self.unwrap(),
            current_col: 0,
            live_input_row_start: 0,
            dead_output_row_start: 0 
        }
    }
}

impl<M: MatrixLike,const D1: usize,const D2: usize> Drop for MatrixExpr<M,D1,D2> {
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

#[derive(Clone)]
pub struct MatrixExprBuilder<const D1: usize, const D2: usize>;

impl<const D1: usize, const D2: usize> MatrixWrapperBuilder for MatrixExprBuilder<D1,D2> {
    type MatrixWrapped<T: MatrixLike> = MatrixExpr<T,D1,D2>;
    type TransposedMatrixWrapped<T: MatrixLike> = MatrixExpr<T,D2,D1>;
    type VectorWrapped<T: VectorLike> = VectorExpr<T,D1>;
    type TransposedVectorWrapped<T: VectorLike> = VectorExpr<T,D2>;

    #[inline] unsafe fn wrap_mat<T: MatrixLike>(&self, mat: T) -> Self::MatrixWrapped<T> {MatrixExpr(mat)}
    #[inline] unsafe fn wrap_trans_mat<T: MatrixLike>(&self, mat: T) -> Self::TransposedMatrixWrapped<T> {MatrixExpr(mat)}
    #[inline] unsafe fn wrap_vec<T: VectorLike>(&self, vec: T) -> Self::VectorWrapped<T> {VectorExpr(vec)}
    #[inline] unsafe fn wrap_trans_vec<T: VectorLike>(&self, vec: T) -> Self::TransposedVectorWrapped<T> {VectorExpr(vec)}
}

pub struct MatrixEntryIter<M: MatrixLike,const D1: usize,const D2: usize>{mat: M, current_col: usize, live_input_row_start: usize, dead_output_row_start: usize}

impl<M: MatrixLike,const D1: usize,const D2: usize> MatrixEntryIter<M,D1,D2> {
    #[inline]
    pub fn raw_next(&mut self) -> Option<M::Item> where M: Has2DReuseBuf<BoundTypes = M::BoundItems> {
        unsafe {
            if self.live_input_row_start < D1 { //current vector isn't done
                let row_index = self.live_input_row_start;
                self.live_input_row_start += 1;
                let inputs = self.mat.get_inputs(self.current_col, row_index);
                let (item,bound_items) = self.mat.process(inputs);
                self.mat.assign_bound_bufs(self.current_col,row_index,bound_items);
                self.dead_output_row_start += 1;
                Some(item)
            } else if self.current_col < D2-1 {
                self.current_col += 1;
                self.live_input_row_start = 1; //we immediately and infallibly get the first one
                self.dead_output_row_start = 0;
                let inputs = self.mat.get_inputs(self.current_col, 0);
                let (item,bound_items) = self.mat.process(inputs);
                self.mat.assign_bound_bufs(self.current_col,0,bound_items);
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
        assert!((self.current_col == D2-1) & (self.live_input_row_start == D1),"math_vector error: A VectorIter must be fully used before outputting");
        debug_assert!(self.dead_output_row_start == D1,"math_vector internal error: A VectorIter's output buffers (somehow) weren't fully filled despite the inputs being fully used, likely an internal issue");
        unsafe {self.unchecked_output()}
    }

    #[inline]
    pub fn consume(mut self) -> M::Output where M: Has2DReuseBuf<BoundTypes = M::BoundItems> {
        let mat = &mut self.mat;
        let current_col = &mut self.current_col;
        let live_input_row_start = &mut self.live_input_row_start;
        let dead_output_row_start = &mut self.dead_output_row_start;
        unsafe{
            while *current_col < D2-1 {
                while *live_input_row_start < D1 {
                    let row_index = *live_input_row_start;
                    *live_input_row_start += 1;
                    let inputs = mat.get_inputs(*current_col, row_index);
                    let (_,bound_items) = mat.process(inputs);
                    mat.assign_bound_bufs(*current_col,row_index,bound_items);
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
                let (_,bound_items) = mat.process(inputs);
                mat.assign_bound_bufs(*current_col,row_index,bound_items);
                *dead_output_row_start += 1;
            }
            self.unchecked_output()
        }
    }
}

impl<M: MatrixLike,const D1: usize,const D2: usize> Drop for MatrixEntryIter<M,D1,D2> {
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

impl<M: MatrixLike,const D1: usize,const D2: usize> Iterator for MatrixEntryIter<M,D1,D2> where M: Has2DReuseBuf<BoundTypes = M::BoundItems> {
    type Item = M::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {self.raw_next()}
}


pub type MathMatrix<T,const D1: usize,const D2: usize> = MatrixExpr<Owned2DArray<T,D1,D2>,D1,D2>;

impl<T,const D1: usize,const D2: usize> MathMatrix<T,D1,D2> {
    #[inline] pub fn into_2d_array(self) -> [[T; D1]; D2] {self.unwrap().unwrap()}
    #[inline] pub fn into_2d_heap_array(self: Box<Self>) -> Box<[[T; D1]; D2]> {
        unsafe { std::mem::transmute::<Box<Self>,Box<[[T; D1]; D2]>>(self) }
    }
    #[inline] pub fn reuse(self) -> MatrixExpr<Replace2DArray<T,D1,D2>,D1,D2> {MatrixExpr(Replace2DArray(self.unwrap().0))}
    #[inline] pub fn heap_reuse(self: Box<Self>) -> MatrixExpr<Replace2DHeapArray<T,D1,D2>,D1,D2> {
        unsafe { MatrixExpr(Replace2DHeapArray(std::mem::transmute::<Box<Self>,std::mem::ManuallyDrop<Box<[[T; D1]; D2]>>>(self))) }
    } 
    
    #[inline] pub unsafe fn get_unchecked<I: std::slice::SliceIndex<[[T; D1]]>>(&self,index: I) -> &I::Output { unsafe {
        self.0.0.get_unchecked(index)
    }}
    #[inline] pub unsafe fn get_unchecked_mut<I: std::slice::SliceIndex<[[T; D1]]>>(&mut self,index: I) -> &mut I::Output { unsafe {
        self.0.0.get_unchecked_mut(index)
    }}
}

impl<T: Clone,const D1: usize,const D2: usize> Clone for MathMatrix<T,D1,D2> {
    #[inline]
    fn clone(&self) -> Self {
        MatrixExpr(self.0.clone())
    }
}

impl<T,const D1: usize,const D2: usize> Deref for MathMatrix<T,D1,D2> {
    type Target = [[T; D1]; D2];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0.0
    }
}

impl<T,const D1: usize,const D2: usize> DerefMut for MathMatrix<T,D1,D2> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0.0
    }
}

impl<T,const D1: usize,const D2: usize> From<[[T; D1]; D2]> for MathMatrix<T,D1,D2> {
    #[inline] 
    fn from(value: [[T; D1]; D2]) -> Self {
        MatrixExpr(Owned2DArray(std::mem::ManuallyDrop::new(value)))
    }
}

impl<T,const D1: usize,const D2: usize> Into<[[T; D1]; D2]> for MathMatrix<T,D1,D2> {
    #[inline] fn into(self) -> [[T; D1]; D2] {self.into_2d_array()}
}

impl<'a,T,const D1: usize,const D2: usize> From<&'a [[T; D1]; D2]> for &'a MathMatrix<T,D1,D2> {
    #[inline]
    fn from(value: &'a [[T; D1]; D2]) -> Self {
        unsafe { std::mem::transmute::<&'a [[T; D1]; D2],&'a MathMatrix<T,D1,D2>>(value) }
    }
}

impl<'a,T,const D1: usize,const D2: usize> Into<&'a [[T; D1]; D2]> for &'a MathMatrix<T,D1,D2> {
    #[inline]
    fn into(self) -> &'a [[T; D1]; D2] {
        unsafe { std::mem::transmute::<&'a MathMatrix<T,D1,D2>,&'a [[T; D1]; D2]>(self) }
    }
}

impl<'a,T,const D1: usize,const D2: usize> From<&'a mut [[T; D1]; D2]> for &'a mut MathMatrix<T,D1,D2> {
    #[inline]
    fn from(value: &'a mut [[T; D1]; D2]) -> Self {
        unsafe { std::mem::transmute::<&'a mut [[T; D1]; D2],&'a mut MathMatrix<T,D1,D2>>(value) }
    }
}

impl<'a,T,const D1: usize,const D2: usize> Into<&'a mut [[T; D1]; D2]> for &'a mut MathMatrix<T,D1,D2> {
    #[inline]
    fn into(self) -> &'a mut [[T; D1]; D2] {
        unsafe { std::mem::transmute::<&'a mut MathMatrix<T,D1,D2>,&'a mut [[T; D1]; D2]>(self) }
    }
}

impl<T,const D1: usize,const D2: usize> From<MathVectoredMatrix<T,D1,D2>> for MathMatrix<T,D1,D2> {
    #[inline]
    fn from(value: MathVectoredMatrix<T,D1,D2>) -> Self {
        //  safety:
        //      MathVectoredMatrix<T,D1,D2> == VectorExpr<OwnedArray<VectorExpr<OwnedArray<T,D1>,D1>,D2>,D2>
        //      MathVectoredMatrix<T,D1,D2> == ManuallyDrop<[ManuallyDrop<[T; D1]>; D2]>
        //      MathVectoredMatrix<T,D1,D2> == ManuallyDrop<[[T; D1]; D2]>
        //      MathVectoredMatrix<T,D1,D2> == MatrixExpr<Owned2DArray<T,D1,D2>>
        //      MathVectoredMatrix<T,D1,D2> == MathMatrix<T,D1,D2>
        //
        //  FIXME: transmute_copy copies (:O), this shouldn't need to be done but gets the compiler to not complain about it
        unsafe { std::mem::transmute_copy::<MathVectoredMatrix<T,D1,D2>,MathMatrix<T,D1,D2>>(&std::mem::ManuallyDrop::new(value)) }
    }
}

impl<T,const D1: usize,const D2: usize> Into<MathVectoredMatrix<T,D1,D2>> for MathMatrix<T,D1,D2> {
    #[inline]
    fn into(self) -> MathVectoredMatrix<T,D1,D2> {
        //  safety:
        //      MathVectoredMatrix<T,D1,D2> == VectorExpr<OwnedArray<VectorExpr<OwnedArray<T,D1>,D1>,D2>,D2>
        //      MathVectoredMatrix<T,D1,D2> == ManuallyDrop<[ManuallyDrop<[T; D1]>; D2]>
        //      MathVectoredMatrix<T,D1,D2> == ManuallyDrop<[[T; D1]; D2]>
        //      MathVectoredMatrix<T,D1,D2> == MatrixExpr<Owned2DArray<T,D1,D2>>
        //      MathVectoredMatrix<T,D1,D2> == MathMatrix<T,D1,D2>
        //
        //  FIXME: transmute_copy copies (:O), this shouldn't need to be done but gets the compiler to not complain about it
        unsafe { std::mem::transmute_copy::<MathMatrix<T,D1,D2>,MathVectoredMatrix<T,D1,D2>>(&std::mem::ManuallyDrop::new(self)) }
    }
}

impl<T,I,const D1: usize,const D2: usize> Index<I> for MathMatrix<T,D1,D2> where [[T; D1]; D2]: Index<I, Output = [T; D1]> {
    type Output = MathVector<T,D1>;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        (&self.0.0[index]).into()
    }
}

impl<T,I,const D1: usize,const D2: usize> IndexMut<I> for MathMatrix<T,D1,D2> where [[T; D1]; D2]: IndexMut<I, Output = [T; D1]> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        (&mut self.0.0[index]).into()
    }
}



pub fn matrix_gen<F: FnMut() -> O,O,const D1: usize,const D2: usize>(f: F) -> MatrixExpr<MatGenerator<F,O>,D1,D2> {
    MatrixExpr(MatGenerator(f))
}


pub trait MatrixOps {
    type Unwrapped: MatrixLike;
    type WrapperBuilder: MatrixWrapperBuilder;

    fn unwrap(self) -> Self::Unwrapped;
    fn get_wrapper_builder(&self) -> Self::WrapperBuilder;
    fn dimensions(&self) -> (usize,usize); // 0: num rows, 1: num columns

    #[inline]
    fn bind(self) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatBind<Self::Unwrapped>>
    where
        Self::Unwrapped: Has2DReuseBuf<FstHandleBool = Y>,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool,Y): FilterPair,
        (<Self::Unwrapped as HasOutput>::OutputBool, <Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed,<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
        Self: Sized
    {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatBind{mat: self.unwrap()}) }
    }

    #[inline]
    fn buf_swap(self) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatBufSwap<Self::Unwrapped>> where (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatBufSwap{mat: self.unwrap()}) }
    }

    #[inline]
    fn offset_columns(self,offset: usize) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatColOffset<Self::Unwrapped>> where Self: Sized {
        let (_, cols) = self.dimensions();
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatColOffset{mat: self.unwrap(),offset: offset % cols, num_columns: cols}) }
    }

    #[inline]
    fn offset_rows(self,offset: usize) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatRowOffset<Self::Unwrapped>> where Self: Sized {
        let (rows, _) = self.dimensions();
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatRowOffset{mat: self.unwrap(),offset: offset % rows, num_rows: rows}) }
    }

    #[inline]
    fn columns(self) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::TransposedVectorWrapped<MatColWrapper<MatColVectorExprs<Self::Unwrapped>,MatrixColumn<Self::Unwrapped>,Self::WrapperBuilder>> where Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_trans_vec(MatColWrapper{mat: MatColVectorExprs{mat: self.unwrap()}, wrapper_builder: wrapper_builder.clone()}) }
    }

    #[inline]
    fn rows(self) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::VectorWrapped<MatRowWrapper<MatRowVectorExprs<Self::Unwrapped>,MatrixRow<Self::Unwrapped>,Self::WrapperBuilder>> where Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_vec(MatRowWrapper{mat: MatRowVectorExprs{mat: self.unwrap()}, wrapper_builder: wrapper_builder.clone()}) }
    }


    #[inline]
    fn entry_map<F: FnMut(<Self::Unwrapped as Get2D>::Item) -> O,O>(self,f: F) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatEntryMap<Self::Unwrapped,F,O>> where (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatEntryMap{mat: self.unwrap(),f}) }
    }

    #[inline]
    fn entry_fold<F: FnMut(O,<Self::Unwrapped as Get2D>::Item) -> O,O>(self,f: F,init: O) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatEntryFold<Self::Unwrapped,F,O>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatEntryFold{mat: self.unwrap(),f,cell: Some(init)}) }
    }

    #[inline]
    fn entry_fold_ref<F: FnMut(&mut O,<Self::Unwrapped as Get2D>::Item),O>(self,f: F,init: O) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatEntryFoldRef<Self::Unwrapped,F,O>> where (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatEntryFoldRef{mat: self.unwrap(),f,cell: std::mem::ManuallyDrop::new(init)}) }
    }

    #[inline]
    fn entry_copied_fold<F: FnMut(O,<Self::Unwrapped as Get2D>::Item) -> O,O>(self,f: F,init: O) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatEntryCopiedFold<Self::Unwrapped,F,O>> where <Self::Unwrapped as Get2D>::Item: Copy, (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatEntryCopiedFold{mat: self.unwrap(),f,cell: Some(init)}) }
    }

    #[inline]
    fn entry_copied_fold_ref<F: FnMut(&mut O,<Self::Unwrapped as Get2D>::Item),O>(self,f: F,init: O) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatEntryCopiedFoldRef<Self::Unwrapped,F,O>> where <Self::Unwrapped as Get2D>::Item: Copy, (<Self::Unwrapped as HasOutput>::OutputBool, Y): FilterPair, Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatEntryCopiedFoldRef{mat: self.unwrap(),f,cell: std::mem::ManuallyDrop::new(init)}) }
    }

    #[inline]
    fn copied<'a,I: 'a + Copy>(self) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatCopy<'a,Self::Unwrapped,I>> where Self::Unwrapped: Get2D<Item = &'a I>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatCopy{mat: self.unwrap()}) }
    }

    #[inline]
    fn cloned<'a,I: 'a + Clone>(self) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatClone<'a,Self::Unwrapped,I>> where Self::Unwrapped: Get2D<Item = &'a I>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatClone{mat: self.unwrap()}) }
    }

    #[inline] 
    fn neg(self) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatNeg<Self::Unwrapped>> where <Self::Unwrapped as Get2D>::Item: Neg, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatNeg{mat: self.unwrap()})}
    }

    #[inline]
    fn mul_r<S: Copy>(self,scalar: S) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatMulR<Self::Unwrapped,S>> where S: Mul<<Self::Unwrapped as Get2D>::Item>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatMulR{mat: self.unwrap(),scalar})}
    }

    #[inline]
    fn div_r<S: Copy>(self,scalar: S) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatDivR<Self::Unwrapped,S>> where S: Div<<Self::Unwrapped as Get2D>::Item>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatDivR{mat: self.unwrap(),scalar})}
    }

    #[inline]
    fn rem_r<S: Copy>(self,scalar: S) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatRemR<Self::Unwrapped,S>> where S: Rem<<Self::Unwrapped as Get2D>::Item>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatRemR{mat: self.unwrap(),scalar})}
    }

    #[inline]
    fn mul_l<S: Copy>(self,scalar: S) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatMulL<Self::Unwrapped,S>> where <Self::Unwrapped as Get2D>::Item: Mul<S>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatMulL{mat: self.unwrap(),scalar})}
    }

    #[inline]
    fn div_l<S: Copy>(self,scalar: S) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatDivL<Self::Unwrapped,S>> where <Self::Unwrapped as Get2D>::Item: Div<S>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatDivL{mat: self.unwrap(),scalar})}
    }

    #[inline]
    fn rem_l<S: Copy>(self,scalar: S) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatRemL<Self::Unwrapped,S>> where <Self::Unwrapped as Get2D>::Item: Rem<S>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatRemL{mat: self.unwrap(),scalar})}
    }

    #[inline]
    fn mul_assign<'a,I: 'a + MulAssign<S>,S: Copy>(self,scalar: S) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatMulAssign<'a,Self::Unwrapped,I,S>> where Self::Unwrapped: Get2D<Item = &'a mut I>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatMulAssign{mat: self.unwrap(),scalar}) }
    }

    #[inline]
    fn div_assign<'a,I: 'a + DivAssign<S>,S: Copy>(self,scalar: S) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatDivAssign<'a,Self::Unwrapped,I,S>> where Self::Unwrapped: Get2D<Item = &'a mut I>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatDivAssign{mat: self.unwrap(),scalar}) }
    }

    #[inline]
    fn rem_assign<'a,I: 'a + RemAssign<S>,S: Copy>(self,scalar: S) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatRemAssign<'a,Self::Unwrapped,I,S>> where Self::Unwrapped: Get2D<Item = &'a mut I>, (<Self::Unwrapped as HasOutput>::OutputBool, N): FilterPair, Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatRemAssign{mat: self.unwrap(),scalar}) }
    }

    ///NOTE: WILL BE MOVED
    #[inline]
    fn mat_mul<M: MatrixOps>(self,other: M) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<FullMatMul<Self::Unwrapped,M::Unwrapped>> where 
        Self::Unwrapped: MatrixLike<IsRepeatable = Y>,
        M::Unwrapped: MatrixLike<IsRepeatable = Y>,
        <Self::Unwrapped as Get2D>::Item: Mul<<M::Unwrapped as Get2D>::Item>,
        <<Self::Unwrapped as Get2D>::Item as Mul<<M::Unwrapped as Get2D>::Item>>::Output: AddAssign,

        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
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
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(FullMatMul{l_mat: self.unwrap(),r_mat: other.unwrap(), shared_size}) }
    }
}

pub trait ArrayMatrixOps<const D1: usize,const D2: usize>: MatrixOps {
    #[inline]
    fn attach_2d_buf<'a,T>(self,buf: &'a mut MathMatrix<T,D1,D2>) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatAttach2DBuf<'a,Self::Unwrapped,T,D1,D2>> where Self::Unwrapped: Has2DReuseBuf<FstHandleBool = N>, Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatAttach2DBuf{mat: self.unwrap(),buf}) }
    }

    #[inline]
    fn create_2d_buf<T>(self) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatCreate2DBuf<Self::Unwrapped,T,D1,D2>> where Self::Unwrapped: Has2DReuseBuf<FstHandleBool = N>, Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatCreate2DBuf{mat: self.unwrap(),buf: std::mem::MaybeUninit::uninit().assume_init()}) }
    }

    #[inline]
    fn create_2d_heap_buf<T>(self) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatCreate2DHeapBuf<Self::Unwrapped,T,D1,D2>> where Self::Unwrapped: Has2DReuseBuf<FstHandleBool = N>, Self: Sized {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatCreate2DHeapBuf{mat: self.unwrap(),buf: std::mem::ManuallyDrop::new(Box::new(std::mem::MaybeUninit::uninit().assume_init()))}) }
    }

    #[inline]
    fn maybe_create_2d_buf<T>(self) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatMaybeCreate2DBuf<Self::Unwrapped,T,D1,D2>> 
    where 
        <<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool,<<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool,<<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg): TyBoolPair,
        Self: Sized
    {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatMaybeCreate2DBuf{mat: self.unwrap(),buf: std::mem::MaybeUninit::uninit().assume_init()}) }
    }

    #[inline]
    fn maybe_create_2d_heap_buf<T>(self) -> <Self::WrapperBuilder as MatrixWrapperBuilder>::MatrixWrapped<MatMaybeCreate2DHeapBuf<Self::Unwrapped,T,D1,D2>> 
    where 
        <<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg: Filter,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool,<<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool,<<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool as TyBool>::Neg): TyBoolPair,
        Self: Sized
    {
        let wrapper_builder = self.get_wrapper_builder();
        unsafe { wrapper_builder.wrap_mat(MatMaybeCreate2DHeapBuf{mat: self.unwrap(),buf: std::mem::ManuallyDrop::new(Box::new(std::mem::MaybeUninit::uninit().assume_init()))}) }
    }
}

macro_rules! overload_operators {
    (
        <$($($lifetime:lifetime),+,)? $($generic:ident $(:)? $($lifetime_bound:lifetime |)? $($fst_trait_bound:path $(| $trait_bound:path)*)?),+,{$d1:ident,$d2:ident}>,
        $ty:ty,
        matrix: $matrix:ty,
        item: $item:ty
    ) => {
        impl<$($($lifetime),+,)? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: Copy, const $d1: usize, const $d2: usize> Mul<Z> for $ty where (<$matrix as HasOutput>::OutputBool,N): FilterPair, $item: Mul<Z>, Self: Sized {
            type Output = MatrixExpr<MatMulL<$matrix,Z>,$d1,$d2>;
    
            #[inline]
            fn mul(self, rhs: Z) -> Self::Output {
                self.mul_l(rhs)
            }
        }

        impl<$($($lifetime),+,)? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: Copy, const $d1: usize, const $d2: usize> Div<Z> for $ty where (<$matrix as HasOutput>::OutputBool,N): FilterPair, $item: Div<Z>, Self: Sized {
            type Output = MatrixExpr<MatDivL<$matrix,Z>,$d1,$d2>;
    
            #[inline]
            fn div(self, rhs: Z) -> Self::Output {
                self.div_l(rhs)
            }
        }

        impl<$($($lifetime),+,)? $($generic: $($lifetime_bound +)? $($fst_trait_bound $(+ $trait_bound)*)?),+, Z: Copy, const $d1: usize, const $d2: usize> Rem<Z> for $ty where (<$matrix as HasOutput>::OutputBool,N): FilterPair, $item: Rem<Z>, Self: Sized {
            type Output = MatrixExpr<MatRemL<$matrix,Z>,$d1,$d2>;
    
            #[inline]
            fn rem(self, rhs: Z) -> Self::Output {
                self.rem_l(rhs)
            }
        }
    }
}


impl<M: MatrixLike,const D1: usize,const D2: usize> MatrixOps for MatrixExpr<M,D1,D2> {
    type Unwrapped = M;
    type WrapperBuilder = MatrixExprBuilder<D1,D2>;

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

    #[inline] fn get_wrapper_builder(&self) -> Self::WrapperBuilder {MatrixExprBuilder}
    #[inline] fn dimensions(&self) -> (usize,usize) {(D1,D2)}
}
impl<M: MatrixLike,const D1: usize,const D2: usize> ArrayMatrixOps<D1,D2> for MatrixExpr<M,D1,D2> {}
overload_operators!(<M: MatrixLike,{D1,D2}>, MatrixExpr<M,D1,D2>, matrix: M, item: M::Item);

impl<M: MatrixLike,const D1: usize,const D2: usize> MatrixOps for Box<MatrixExpr<M,D1,D2>> {
    type Unwrapped = Box<M>;
    type WrapperBuilder = MatrixExprBuilder<D1,D2>;

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
    
    #[inline] fn get_wrapper_builder(&self) -> Self::WrapperBuilder {MatrixExprBuilder}
    #[inline] fn dimensions(&self) -> (usize,usize) {(D1,D2)}
}
impl<M: MatrixLike,const D1: usize,const D2: usize> ArrayMatrixOps<D1,D2> for Box<MatrixExpr<M,D1,D2>> {}
overload_operators!(<M: MatrixLike,{D1,D2}>, Box<MatrixExpr<M,D1,D2>>, matrix: Box<M>, item: M::Item);


impl<'a,T,const D1: usize,const D2: usize> MatrixOps for &'a MathMatrix<T,D1,D2> {
    type Unwrapped = &'a [[T; D1]; D2];
    type WrapperBuilder = MatrixExprBuilder<D1,D2>;
    
    #[inline] fn unwrap(self) -> Self::Unwrapped {&self.0}
    #[inline] fn get_wrapper_builder(&self) -> Self::WrapperBuilder {MatrixExprBuilder}
    #[inline] fn dimensions(&self) -> (usize,usize) {(D1,D2)}
}
impl<'a,T,const D1: usize,const D2: usize> ArrayMatrixOps<D1,D2> for &'a MathMatrix<T,D1,D2> {}
overload_operators!(<'a,T,{D1,D2}>, &'a MathMatrix<T,D1,D2>, matrix: &'a [[T; D1]; D2], item: &'a T);

impl<'a,T,const D1: usize,const D2: usize> MatrixOps for &'a mut MathMatrix<T,D1,D2> {
    type Unwrapped = &'a mut [[T; D1]; D2];
    type WrapperBuilder = MatrixExprBuilder<D1,D2>;
    
    #[inline] fn unwrap(self) -> Self::Unwrapped {&mut self.0}
    #[inline] fn get_wrapper_builder(&self) -> Self::WrapperBuilder {MatrixExprBuilder}
    #[inline] fn dimensions(&self) -> (usize,usize) {(D1,D2)}
}
impl<'a,T,const D1: usize,const D2: usize> ArrayMatrixOps<D1,D2> for &'a mut MathMatrix<T,D1,D2> {}
overload_operators!(<'a,T,{D1,D2}>, &'a mut MathMatrix<T,D1,D2>, matrix: &'a mut [[T; D1]; D2], item: &'a mut T);

impl<'a,T,const D1: usize,const D2: usize> MatrixOps for &'a Box<MathMatrix<T,D1,D2>> {
    type Unwrapped = &'a [[T; D1]; D2];
    type WrapperBuilder = MatrixExprBuilder<D1,D2>;
    
    #[inline] fn unwrap(self) -> Self::Unwrapped {&self.0}
    #[inline] fn get_wrapper_builder(&self) -> Self::WrapperBuilder {MatrixExprBuilder}
    #[inline] fn dimensions(&self) -> (usize,usize) {(D1,D2)}
}
impl<'a,T,const D1: usize,const D2: usize> ArrayMatrixOps<D1,D2> for &'a Box<MathMatrix<T,D1,D2>> {}
overload_operators!(<'a,T,{D1,D2}>, &'a Box<MathMatrix<T,D1,D2>>, matrix: &'a [[T; D1]; D2], item: &'a T);

impl<'a,T,const D1: usize,const D2: usize> MatrixOps for &'a mut Box<MathMatrix<T,D1,D2>> {
    type Unwrapped = &'a mut [[T; D1]; D2];
    type WrapperBuilder = MatrixExprBuilder<D1,D2>;
    
    #[inline] fn unwrap(self) -> Self::Unwrapped {&mut self.0}
    #[inline] fn get_wrapper_builder(&self) -> Self::WrapperBuilder {MatrixExprBuilder}
    #[inline] fn dimensions(&self) -> (usize,usize) {(D1,D2)}
}
impl<'a,T,const D1: usize,const D2: usize> ArrayMatrixOps<D1,D2> for &'a mut Box<MathMatrix<T,D1,D2>> {}
overload_operators!(<'a,T,{D1,D2}>, &'a mut Box<MathMatrix<T,D1,D2>>, matrix: &'a mut [[T; D1]; D2], item: &'a mut T);


pub trait EqDimMatrixMatrixOps<M: MatrixOps>: MatrixOps {
    type MatrixDoubleWrapped<T: MatrixLike>;
    type TransposedMatrixDoubleWrapped<T: MatrixLike>;
    type VectorDoubleWrapped<T: VectorLike>;
    type TransposedVectorDoubleWrapped<T: VectorLike>;

    fn assert_eq_dim(&self,other: &M);
    unsafe fn double_wrap_mat<T: MatrixLike>(mat: T) -> Self::MatrixDoubleWrapped<T>;
    unsafe fn double_wrap_trans_mat<T: MatrixLike>(mat: T) -> Self::TransposedMatrixDoubleWrapped<T>;
    unsafe fn double_wrap_vec<T: VectorLike>(vec: T) -> Self::VectorDoubleWrapped<T>;
    unsafe fn double_wrap_trans_vec<T: VectorLike>(vec: T) -> Self::TransposedVectorDoubleWrapped<T>;

    #[inline]
    fn zip(self,other: M) -> Self::MatrixDoubleWrapped<MatZip<Self::Unwrapped,M::Unwrapped>>
    where
        Self: Sized, 
        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    {
        self.assert_eq_dim(&other);
        unsafe { Self::double_wrap_mat(MatZip{ l_mat: self.unwrap(), r_mat: other.unwrap() }) }
    }

    #[inline]
    fn add(self,other: M) -> Self::MatrixDoubleWrapped<MatAdd<Self::Unwrapped,M::Unwrapped>>
    where
        Self: Sized, 
        <Self::Unwrapped as Get2D>::Item: Add<<M::Unwrapped as Get2D>::Item>,
        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    {
        self.assert_eq_dim(&other);
        unsafe { Self::double_wrap_mat(MatAdd{ l_mat: self.unwrap(), r_mat: other.unwrap() }) }
    }

    #[inline]
    fn sub(self,other: M) -> Self::MatrixDoubleWrapped<MatSub<Self::Unwrapped,M::Unwrapped>>
    where
        Self: Sized, 
        <Self::Unwrapped as Get2D>::Item: Sub<<M::Unwrapped as Get2D>::Item>,
        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    {
        self.assert_eq_dim(&other);
        unsafe { Self::double_wrap_mat(MatSub{ l_mat: self.unwrap(), r_mat: other.unwrap() }) }
    }

    #[inline]
    fn comp_mul(self,other: M) -> Self::MatrixDoubleWrapped<MatCompMul<Self::Unwrapped,M::Unwrapped>>
    where
        Self: Sized, 
        <Self::Unwrapped as Get2D>::Item: Mul<<M::Unwrapped as Get2D>::Item>,
        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    {
        self.assert_eq_dim(&other);
        unsafe { Self::double_wrap_mat(MatCompMul{ l_mat: self.unwrap(), r_mat: other.unwrap() }) }
    }

    #[inline]
    fn comp_div(self,other: M) -> Self::MatrixDoubleWrapped<MatCompDiv<Self::Unwrapped,M::Unwrapped>>
    where
        Self: Sized, 
        <Self::Unwrapped as Get2D>::Item: Div<<M::Unwrapped as Get2D>::Item>,
        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    {
        self.assert_eq_dim(&other);
        unsafe { Self::double_wrap_mat(MatCompDiv{ l_mat: self.unwrap(), r_mat: other.unwrap() }) }
    }

    #[inline]
    fn comp_rem(self,other: M) -> Self::MatrixDoubleWrapped<MatCompRem<Self::Unwrapped,M::Unwrapped>>
    where
        Self: Sized, 
        <Self::Unwrapped as Get2D>::Item: Rem<<M::Unwrapped as Get2D>::Item>,
        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    {
        self.assert_eq_dim(&other);
        unsafe { Self::double_wrap_mat(MatCompRem{ l_mat: self.unwrap(), r_mat: other.unwrap() }) }
    }

    #[inline]
    fn add_assign<'a,I: 'a + AddAssign<<M::Unwrapped as Get2D>::Item>>(self,other: M) -> Self::MatrixDoubleWrapped<MatAddAssign<'a,Self::Unwrapped,M::Unwrapped,I>>
    where 
        Self: Sized, 
        Self::Unwrapped: Get2D<Item = &'a mut I>,
        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    {
        self.assert_eq_dim(&other);
        unsafe { Self::double_wrap_mat(MatAddAssign{ l_mat: self.unwrap(), r_mat: other.unwrap() }) }
    }

    #[inline]
    fn sub_assign<'a,I: 'a + SubAssign<<M::Unwrapped as Get2D>::Item>>(self,other: M) -> Self::MatrixDoubleWrapped<MatSubAssign<'a,Self::Unwrapped,M::Unwrapped,I>>
    where 
        Self: Sized, 
        Self::Unwrapped: Get2D<Item = &'a mut I>,
        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    {
        self.assert_eq_dim(&other);
        unsafe { Self::double_wrap_mat(MatSubAssign{ l_mat: self.unwrap(), r_mat: other.unwrap() }) }
    }

    #[inline]
    fn comp_mul_assign<'a,I: 'a + MulAssign<<M::Unwrapped as Get2D>::Item>>(self,other: M) -> Self::MatrixDoubleWrapped<MatCompMulAssign<'a,Self::Unwrapped,M::Unwrapped,I>>
    where 
        Self: Sized, 
        Self::Unwrapped: Get2D<Item = &'a mut I>,
        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    {
        self.assert_eq_dim(&other);
        unsafe { Self::double_wrap_mat(MatCompMulAssign{ l_mat: self.unwrap(), r_mat: other.unwrap() }) }
    }

    #[inline]
    fn comp_div_assign<'a,I: 'a + DivAssign<<M::Unwrapped as Get2D>::Item>>(self,other: M) -> Self::MatrixDoubleWrapped<MatCompDivAssign<'a,Self::Unwrapped,M::Unwrapped,I>>
    where 
        Self: Sized, 
        Self::Unwrapped: Get2D<Item = &'a mut I>,
        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    {
        self.assert_eq_dim(&other);
        unsafe { Self::double_wrap_mat(MatCompDivAssign{ l_mat: self.unwrap(), r_mat: other.unwrap() }) }
    }

    #[inline]
    fn comp_rem_assign<'a,I: 'a + RemAssign<<M::Unwrapped as Get2D>::Item>>(self,other: M) -> Self::MatrixDoubleWrapped<MatCompRemAssign<'a,Self::Unwrapped,M::Unwrapped,I>>
    where 
        Self: Sized, 
        Self::Unwrapped: Get2D<Item = &'a mut I>,
        (<Self::Unwrapped as Get2D>::AreInputsTransposed, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
        (<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool): FilterPair,
        (<(<Self::Unwrapped as HasOutput>::OutputBool,<M::Unwrapped as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::BoundHandlesBool, <M::Unwrapped as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::FstHandleBool, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::SndHandleBool, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
        (<Self::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
    {
        self.assert_eq_dim(&other);
        unsafe { Self::double_wrap_mat(MatCompRemAssign{ l_mat: self.unwrap(), r_mat: other.unwrap() }) }
    }
}

macro_rules! impl_const_sized_eq_dim_mat_mat_ops {
    (
        $d1:ident;
        $d2:ident;
        <$($($l_lifetime:lifetime),+,)? $($l_generic:ident $(:)? $($l_lifetime_bound:lifetime |)? $($l_fst_trait_bound:path $(| $l_trait_bound:path)*)?),+ $(, {$l_tt:tt})?>,
        $l_ty:ty,
        matrix: $l_matrix:ty,
        item: $l_item:ty;
        <$($($r_lifetime:lifetime),+,)? $($r_generic:ident $(:)? $($r_lifetime_bound:lifetime |)? $($r_fst_trait_bound:path $(| $r_trait_bound:path)*)?),+ $(, {$r_tt:tt})?>,
        $r_ty:ty,
        matrix: $r_matrix:ty,
        item: $r_item:ty
    ) => {
        impl<
            $($($l_lifetime),+,)? 
            $($($r_lifetime),+,)?
            $($l_generic: $($l_lifetime_bound |)? $($l_fst_trait_bound $(| $l_trait_bound)*)?),+,
            $($r_generic: $($r_lifetime_bound |)? $($r_fst_trait_bound $(| $r_trait_bound)*)?),+,
            $($l_tt,)?
            $($r_tt,)?
            const $d1: usize,
            const $d2: usize
        > EqDimMatrixMatrixOps<$r_ty> for $l_ty {
            type MatrixDoubleWrapped<Z: MatrixLike> = MatrixExpr<Z,$d1,$d2>;
            type TransposedMatrixDoubleWrapped<Z: MatrixLike> = MatrixExpr<Z,$d2,$d1>;
            type VectorDoubleWrapped<Z: VectorLike> = VectorExpr<Z,$d1>;
            type TransposedVectorDoubleWrapped<Z: VectorLike> = VectorExpr<Z,$d2>;

            #[inline] fn assert_eq_dim(&self,_: &$r_ty) {} //compile time checked through const equivalence
            #[inline] unsafe fn double_wrap_mat<Z: MatrixLike>(mat: Z) -> Self::MatrixDoubleWrapped<Z> {MatrixExpr(mat)}
            #[inline] unsafe fn double_wrap_trans_mat<Z: MatrixLike>(mat: Z) -> Self::TransposedMatrixDoubleWrapped<Z> {MatrixExpr(mat)}
            #[inline] unsafe fn double_wrap_vec<Z: VectorLike>(vec: Z) -> Self::VectorDoubleWrapped<Z> {VectorExpr(vec)}
            #[inline] unsafe fn double_wrap_trans_vec<Z: VectorLike>(vec: Z) -> Self::TransposedVectorDoubleWrapped<Z> {VectorExpr(vec)}
        }

        impl<
            $($($l_lifetime),+,)? 
            $($($r_lifetime),+,)?
            $($l_generic: $($l_lifetime_bound |)? $($l_fst_trait_bound $(| $l_trait_bound)*)?),+,
            $($r_generic: $($r_lifetime_bound |)? $($r_fst_trait_bound $(| $r_trait_bound)*)?),+,
            $($l_tt,)?
            $($r_tt,)?
            const $d1: usize,
            const $d2: usize
        > Add<$r_ty> for $l_ty 
        where
            $l_item: Add<$r_item>,
            (<$l_matrix as Get2D>::AreInputsTransposed, N): TyBoolPair,
            (<$l_matrix as HasOutput>::OutputBool, N): FilterPair,
            (<(<$l_matrix as HasOutput>::OutputBool, N) as TyBoolPair>::Or, N): FilterPair,
            (<$l_matrix as Has2DReuseBuf>::BoundHandlesBool, N): FilterPair,
            (<$l_matrix as Has2DReuseBuf>::FstOwnedBufferBool, N): SelectPair,
            (<$l_matrix as Has2DReuseBuf>::SndOwnedBufferBool, N): SelectPair,
            (<$l_matrix as Has2DReuseBuf>::FstHandleBool, N): SelectPair,
            (<$l_matrix as Has2DReuseBuf>::SndHandleBool, N): SelectPair,
            (<$l_matrix as Has2DReuseBuf>::IsFstBufferTransposed, N): TyBoolPair,
            (<$l_matrix as Has2DReuseBuf>::IsSndBufferTransposed, N): TyBoolPair,
            (<$l_matrix as Has2DReuseBuf>::AreBoundBuffersTransposed, N): TyBoolPair,
            (N, <$r_matrix as Get2D>::AreInputsTransposed): TyBoolPair,
            (N,<$r_matrix as HasOutput>::OutputBool): FilterPair,
            (<(N,<$r_matrix as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
            (N, <$r_matrix as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
            (N, <$r_matrix as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
            (N, <$r_matrix as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
            (N, <$r_matrix as Has2DReuseBuf>::FstHandleBool): SelectPair,
            (N, <$r_matrix as Has2DReuseBuf>::SndHandleBool): SelectPair,
            (N, <$r_matrix as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
            (N, <$r_matrix as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
            (N, <$r_matrix as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
            (<$l_matrix as Get2D>::AreInputsTransposed, <$r_matrix as Get2D>::AreInputsTransposed): TyBoolPair,
            (<$l_matrix as HasOutput>::OutputBool,<$r_matrix as HasOutput>::OutputBool): FilterPair,
            (<(<$l_matrix as HasOutput>::OutputBool,<$r_matrix as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
            (<$l_matrix as Has2DReuseBuf>::BoundHandlesBool, <$r_matrix as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
            (<$l_matrix as Has2DReuseBuf>::FstOwnedBufferBool, <$r_matrix as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
            (<$l_matrix as Has2DReuseBuf>::SndOwnedBufferBool, <$r_matrix as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
            (<$l_matrix as Has2DReuseBuf>::FstHandleBool, <$r_matrix as Has2DReuseBuf>::FstHandleBool): SelectPair,
            (<$l_matrix as Has2DReuseBuf>::SndHandleBool, <$r_matrix as Has2DReuseBuf>::SndHandleBool): SelectPair,
            (<$l_matrix as Has2DReuseBuf>::IsFstBufferTransposed, <$r_matrix as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
            (<$l_matrix as Has2DReuseBuf>::IsSndBufferTransposed, <$r_matrix as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
            (<$l_matrix as Has2DReuseBuf>::AreBoundBuffersTransposed, <$r_matrix as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
        {
            type Output = MatrixExpr<MatAdd<<$l_ty as MatrixOps>::Unwrapped,<$r_ty as MatrixOps>::Unwrapped>,$d1,$d2>;

            #[inline]
            fn add(self,rhs: $r_ty) -> Self::Output {
                <Self as EqDimMatrixMatrixOps<$r_ty>>::add(self,rhs)
            }
        }

        impl<
            $($($l_lifetime),+,)? 
            $($($r_lifetime),+,)?
            $($l_generic: $($l_lifetime_bound |)? $($l_fst_trait_bound $(| $l_trait_bound)*)?),+,
            $($r_generic: $($r_lifetime_bound |)? $($r_fst_trait_bound $(| $r_trait_bound)*)?),+,
            $($l_tt,)?
            $($r_tt,)?
            const $d1: usize,
            const $d2: usize
        > Sub<$r_ty> for $l_ty 
        where
            $l_item: Sub<$r_item>,
            (<$l_matrix as Get2D>::AreInputsTransposed, N): TyBoolPair,
            (<$l_matrix as HasOutput>::OutputBool, N): FilterPair,
            (<(<$l_matrix as HasOutput>::OutputBool, N) as TyBoolPair>::Or, N): FilterPair,
            (<$l_matrix as Has2DReuseBuf>::BoundHandlesBool, N): FilterPair,
            (<$l_matrix as Has2DReuseBuf>::FstOwnedBufferBool, N): SelectPair,
            (<$l_matrix as Has2DReuseBuf>::SndOwnedBufferBool, N): SelectPair,
            (<$l_matrix as Has2DReuseBuf>::FstHandleBool, N): SelectPair,
            (<$l_matrix as Has2DReuseBuf>::SndHandleBool, N): SelectPair,
            (<$l_matrix as Has2DReuseBuf>::IsFstBufferTransposed, N): TyBoolPair,
            (<$l_matrix as Has2DReuseBuf>::IsSndBufferTransposed, N): TyBoolPair,
            (<$l_matrix as Has2DReuseBuf>::AreBoundBuffersTransposed, N): TyBoolPair,
            (N, <$r_matrix as Get2D>::AreInputsTransposed): TyBoolPair,
            (N,<$r_matrix as HasOutput>::OutputBool): FilterPair,
            (<(N,<$r_matrix as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
            (N, <$r_matrix as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
            (N, <$r_matrix as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
            (N, <$r_matrix as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
            (N, <$r_matrix as Has2DReuseBuf>::FstHandleBool): SelectPair,
            (N, <$r_matrix as Has2DReuseBuf>::SndHandleBool): SelectPair,
            (N, <$r_matrix as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
            (N, <$r_matrix as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
            (N, <$r_matrix as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
            (<$l_matrix as Get2D>::AreInputsTransposed, <$r_matrix as Get2D>::AreInputsTransposed): TyBoolPair,
            (<$l_matrix as HasOutput>::OutputBool,<$r_matrix as HasOutput>::OutputBool): FilterPair,
            (<(<$l_matrix as HasOutput>::OutputBool,<$r_matrix as HasOutput>::OutputBool) as TyBoolPair>::Or, N): FilterPair,
            (<$l_matrix as Has2DReuseBuf>::BoundHandlesBool, <$r_matrix as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
            (<$l_matrix as Has2DReuseBuf>::FstOwnedBufferBool, <$r_matrix as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
            (<$l_matrix as Has2DReuseBuf>::SndOwnedBufferBool, <$r_matrix as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
            (<$l_matrix as Has2DReuseBuf>::FstHandleBool, <$r_matrix as Has2DReuseBuf>::FstHandleBool): SelectPair,
            (<$l_matrix as Has2DReuseBuf>::SndHandleBool, <$r_matrix as Has2DReuseBuf>::SndHandleBool): SelectPair,
            (<$l_matrix as Has2DReuseBuf>::IsFstBufferTransposed, <$r_matrix as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
            (<$l_matrix as Has2DReuseBuf>::IsSndBufferTransposed, <$r_matrix as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
            (<$l_matrix as Has2DReuseBuf>::AreBoundBuffersTransposed, <$r_matrix as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
        {
            type Output = MatrixExpr<MatSub<<$l_ty as MatrixOps>::Unwrapped,<$r_ty as MatrixOps>::Unwrapped>,$d1,$d2>;

            #[inline]
            fn sub(self,rhs: $r_ty) -> Self::Output {
                <Self as EqDimMatrixMatrixOps<$r_ty>>::sub(self,rhs)
            }
        }
    };
}

macro_rules! impl_some_const_sized_eq_dim_mat_mat_ops {
    (
        $d1:ident;
        $d2:ident
        |
        | 
        <$($($r_lifetime:lifetime),+,)? $($r_generic:ident $(:)? $($r_lifetime_bound:lifetime |)? $($r_fst_trait_bound:path $(| $r_trait_bound:path)*)?),+ $(, {$r_tt:tt})?>,
        $r_ty:ty,
        matrix: $r_matrix:ty,
        item: $r_item:ty
    ) => {};
    (
        $d1:ident;
        $d2:ident
        |
        <$($($fst_l_lifetime:lifetime),+,)? $($fst_l_generic:ident $(:)? $($fst_l_lifetime_bound:lifetime |)? $($fst_l_fst_trait_bound:path $(| $fst_l_trait_bound:path)*)?),+ $(, {$fst_l_tt:tt})?>,
        $fst_l_ty:ty,
        matrix: $fst_l_matrix:ty,
        item: $fst_l_item:ty
        $(; <$($($l_lifetime:lifetime),+,)? $($l_generic:ident $(:)? $($l_lifetime_bound:lifetime |)? $($l_fst_trait_bound:path $(| $l_trait_bound:path)*)?),+ $(, {$l_tt:tt})?>,
        $l_ty:ty,
        matrix: $l_matrix:ty,
        item: $l_item:ty)* 
        |
        <$($($r_lifetime:lifetime),+,)? $($r_generic:ident $(:)? $($r_lifetime_bound:lifetime |)? $($r_fst_trait_bound:path $(| $r_trait_bound:path)*)?),+ $(, {$r_tt:tt})?>,
        $r_ty:ty,
        matrix: $r_matrix:ty,
        item: $r_item:ty
    ) => {
        impl_const_sized_eq_dim_mat_mat_ops!(
            $d1;
            $d2;
            <$($($fst_l_lifetime),+,)? $($fst_l_generic: $($fst_l_lifetime_bound |)? $($fst_l_fst_trait_bound $(| $fst_l_trait_bound)*)?),+ $(, {$fst_l_tt})?>,
            $fst_l_ty,
            matrix: $fst_l_matrix,
            item: $fst_l_item;
            <$($($r_lifetime),+,)? $($r_generic: $($r_lifetime_bound |)? $($r_fst_trait_bound $(| $r_trait_bound)*)?),+ $(, {$r_tt})?>,
            $r_ty,
            matrix: $r_matrix,
            item: $r_item
        );

        impl_some_const_sized_eq_dim_mat_mat_ops!(
            $d1;
            $d2
            |
            $(
                <$($($l_lifetime),+,)? $($l_generic: $($l_lifetime_bound |)? $($l_fst_trait_bound $(| $l_trait_bound)*)?),+ $(, {$l_tt})?>,
                $l_ty,
                matrix: $l_matrix,
                item: $l_item
            );*
            |
            <$($($r_lifetime),+,)? $($r_generic: $($r_lifetime_bound |)? $($r_fst_trait_bound $(| $r_trait_bound)*)?),+ $(, {$r_tt})?>,
            $r_ty,
            matrix: $r_matrix,
            item: $r_item
        );
    };
}

macro_rules! impl_all_const_sized_eq_dim_mat_mat_ops {
    (
        $d1:ident;
        $d2:ident
        |
        $(<$($($l_lifetime:lifetime),+,)? $($l_generic:ident $(:)? $($l_lifetime_bound:lifetime |)? $($l_fst_trait_bound:path $(| $l_trait_bound:path)*)?),+ $(, {$l_tt:tt})?>,
        $l_ty:ty,
        matrix: $l_matrix:ty,
        item: $l_item:ty);+
        |
    ) => {};
    (
        $d1:ident;
        $d2:ident 
        |
        $(<$($($l_lifetime:lifetime),+,)? $($l_generic:ident $(:)? $($l_lifetime_bound:lifetime |)? $($l_fst_trait_bound:path $(| $l_trait_bound:path)*)?),+ $(, {$l_tt:tt})?>,
        $l_ty:ty,
        matrix: $l_matrix:ty,
        item: $l_item:ty);+
        |
        <$($($fst_r_lifetime:lifetime),+,)? $($fst_r_generic:ident $(:)? $($fst_r_lifetime_bound:lifetime |)? $($fst_r_fst_trait_bound:path $(| $fst_r_trait_bound:path)*)?),+ $(, {$fst_r_tt:tt})?>,
        $fst_r_ty:ty,
        matrix: $fst_r_matrix:ty,
        item: $fst_r_item:ty
        $(; <$($($r_lifetime:lifetime),+,)? $($r_generic:ident $(:)? $($r_lifetime_bound:lifetime |)? $($r_fst_trait_bound:path $(| $r_trait_bound:path)*)?),+ $(, {$r_tt:tt})?>,
        $r_ty:ty,
        matrix: $r_matrix:ty,
        item: $r_item:ty)*
    ) => {
        impl_some_const_sized_eq_dim_mat_mat_ops!(
            $d1;
            $d2
            |
            $(
                <$($($l_lifetime),+,)? $($l_generic: $($l_lifetime_bound |)? $($l_fst_trait_bound $(| $l_trait_bound)*)?),+ $(, {$l_tt})?>,
                $l_ty,
                matrix: $l_matrix,
                item: $l_item
            );+
            |
            <$($($fst_r_lifetime),+,)? $($fst_r_generic: $($fst_r_lifetime_bound |)? $($fst_r_fst_trait_bound $(| $fst_r_trait_bound)*)?),+ $(, {$fst_r_tt})?>,
            $fst_r_ty,
            matrix: $fst_r_matrix,
            item: $fst_r_item
        );

        impl_all_const_sized_eq_dim_mat_mat_ops!{
            $d1;
            $d2
            |
            $(
                <$($($l_lifetime),+,)? $($l_generic: $($l_lifetime_bound |)? $($l_fst_trait_bound $(| $l_trait_bound)*)?),+ $(, {$l_tt})?>,
                $l_ty,
                matrix: $l_matrix,
                item: $l_item
            );+
            |
            $(
                <$($($r_lifetime),+,)? $($r_generic: $($r_lifetime_bound |)? $($r_fst_trait_bound $(| $r_trait_bound)*)?),+ $(, {$r_tt})?>,
                $r_ty,
                matrix: $r_matrix,
                item: $r_item
            );*
        }
    };
}

impl_all_const_sized_eq_dim_mat_mat_ops!(
    D1;
    D2 
    |
    <M1: MatrixLike>, MatrixExpr<M1,D1,D2>, matrix: M1, item: M1::Item;
    <M1: MatrixLike>, Box<MatrixExpr<M1,D1,D2>>, matrix: M1, item: M1::Item;
    <'a,T1>, &'a MathMatrix<T1,D1,D2>, matrix: &'a [[T1; D1]; D2], item: &'a T1;
    <'a,T1>, &'a mut MathMatrix<T1,D1,D2>, matrix: &'a mut [[T1; D1]; D2], item: &'a mut T1;
    <'a,T1>, &'a Box<MathMatrix<T1,D1,D2>>, matrix: &'a [[T1; D1]; D2], item: &'a T1;
    <'a,T1>, &'a mut Box<MathMatrix<T1,D1,D2>>, matrix: &'a mut [[T1; D1]; D2], item: &'a mut T1
    |
    <M2: MatrixLike>, MatrixExpr<M2,D1,D2>, matrix: M2, item: M2::Item;
    <M2: MatrixLike>, Box<MatrixExpr<M2,D1,D2>>, matrix: M2, item: M2::Item;
    <'b,T2>, &'b MathMatrix<T2,D1,D2>, matrix: &'b [[T2; D1]; D2], item: &'b T2;
    <'b,T2>, &'b mut MathMatrix<T2,D1,D2>, matrix: &'b mut [[T2; D1]; D2], item: &'b mut T2;
    <'b,T2>, &'b Box<MathMatrix<T2,D1,D2>>, matrix: &'b [[T2; D1]; D2], item: &'b T2;
    <'b,T2>, &'b mut Box<MathMatrix<T2,D1,D2>>, matrix: &'b mut [[T2; D1]; D2], item: &'b mut T2
);