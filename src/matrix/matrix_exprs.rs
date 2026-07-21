use crate::{
    trait_specialization_utils::*,
    util_traits::HasOutput,
    vector::vector_exprs::MathVector, 
};
use super::{
    mat_util_traits::*,
    matrix_structs::*,
    matrix_builders::*,
    MatrixOps,
    ArrayMatrixOps,
};
use std::{
    mem::{self, transmute, ManuallyDrop},
    ops::*,
};

/// a trait expressing that an implementor's data from [`Get`] is stored and accessible, allowing it to be indexed and borrowed
/// 
/// the implementation of index must be indexing into a column and then row
pub trait ConcreteMatrixExpr: MatrixOps + IndexMut<usize> where 
    Self::Output: IndexMut<usize>,
    Self::Unwrapped: Get2D<Item = <Self::Output as Index<usize>>::Output>,
{
    /// The inner [`MatrixLike`] contained in the borrowed version of this matrix
    type ReferencedInner<'a>: MatrixLike<Item = &'a <Self::Output as Index<usize>>::Output> + Is2DRepeatable
        where Self: 'a;
    /// the borrowed version of this matrix
    type Referenced<'a>: MatrixOps<Unwrapped = Self::ReferencedInner<'a>> + Index<usize, Output = Self::Output>
        where Self: 'a;
    /// A borrowed version of this matrix except that its items are copied
    type Copied<'a>: MatrixOps<Unwrapped = MatCopy<'a, Self::ReferencedInner<'a>, <Self::Output as Index<usize>>::Output>>
        where <Self::Output as Index<usize>>::Output: Copy, Self: 'a;
    /// The inner [`MatrixLike`] contained in the mutably borrowed version of this matrix
    type ReferencedMutInner<'a>: MatrixLike<Item = &'a mut <Self::Output as Index<usize>>::Output> 
        where Self: 'a;
    /// the mutably borrowed version of this matrix
    type ReferencedMut<'a>: MatrixOps<Unwrapped = Self::ReferencedMutInner<'a>> + IndexMut<usize, Output = Self::Output> + IndexMut<usize>
        where Self: 'a;

    /// create a borrowed version of this vector which contains a reference to each of its items
    fn borrow<'a>(&'a self) -> Self::Referenced<'a>;

    /// create a mutably borrowed version of this vector which contain a mutable reference to each of its items
    fn borrow_mut<'a>(&'a mut self) -> Self::ReferencedMut<'a>;

    /// create a borrowed version of this vector which contains a copy each of its items
    fn copy<'a>(&'a self) -> Self::Copied<'a> where 
        <Self::Output as Index<usize>>::Output: Copy,
    ;
}

/// A const sized matrix wrapper
/// D1: # rows (dimension of vectors), D2: # columns (# of vectors)
// MatrixExpr assumes that the stored MatrixLike is fully unused
#[repr(transparent)]
pub struct MatrixExpr<M: MatrixLike, const D1: usize, const D2: usize>(pub(crate) M);

impl<M: MatrixLike, const D1: usize, const D2: usize> MatrixExpr<M, D1, D2> {
    /// converts the underlying VectorLike to a dynamic object
    /// stabilizes the overall type to a consitent one
    #[inline]
    #[allow(clippy::type_complexity)] // you try writing this type more simply
    pub fn make_dynamic(
        self,
    ) -> MatrixExpr<
        Box<
            dyn MatrixLike<
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
            >,
        >,
        D1,
        D2,
    >
    where
        M: 'static,
    {
        MatrixExpr(Box::new(DynamicMatrixLike {
            mat: self.unwrap(),
            inputs: None,
        })
            as Box<
                dyn MatrixLike<
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
                >,
            >
        )
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
    pub fn heap_eval(self) -> <MatBind<MatMaybeCreate2DHeapArray<M, M::Item, D1, D2>> as HasOutput>::Output
    where
        (M::FstHandleBool, <M::FstHandleBool as TyBool>::Neg): SelectPair,
        (M::FstOwnedBufferBool, <M::FstHandleBool as TyBool>::Neg): TyBoolPair,
        (M::OutputBool, <(M::FstOwnedBufferBool, <M::FstHandleBool as TyBool>::Neg) as TyBoolPair>::Or): FilterPair,
        <M::FstHandleBool as TyBool>::Neg: Filter,
        (M::BoundHandlesBool, Y): FilterPair,
        (M::IsFstBufferTransposed, M::AreBoundBuffersTransposed): TyBoolPair,
        MatBind<MatMaybeCreate2DHeapArray<M, M::Item, D1, D2>>: Has2DReuseBuf<BoundTypes = <MatBind<MatMaybeCreate2DHeapArray<M, M::Item, D1, D2>> as Get2D>::BoundItems>
    {
        self.maybe_create_2d_heap_array().bind().consume()
    }
}

impl<M: MatrixLike + Is2DRepeatable, const D1: usize, const D2: usize> MatrixExpr<M, D1, D2> {
    /// Retrieves the value at an arbitrary index of a repeatable matrix
    /// Note:   This method does NOT fill any buffers bound to the matrix
    pub fn get(&mut self, col_index: usize, row_index: usize) -> M::Item {
        if (col_index >= D2) | (row_index >= D1) {
            panic!("math_vector Error: index access out of bound")
        }
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

/// a simple type alias for MatrixExpr created from an array of type [[T; D1]; D2]
pub type MathMatrix<T, const D1: usize, const D2: usize> =
    MatrixExpr<MatrixArray<T, D1, D2>, D1, D2>;

impl<T, const D1: usize, const D2: usize> MathMatrix<T, D1, D2> {
    /// Marks this MathMatrix to have its buffer reused
    /// buffer placed in fst slot
    #[inline]
    pub fn reuse(self) -> MatrixExpr<ReplaceMatrixArray<T, D1, D2>, D1, D2> {
        MatrixExpr(ReplaceMatrixArray(self.unwrap().0))
    }
    /// Marks this MathMatrix to have its buffer reused while keeping it on the heap
    /// buffer placed in fst slot
    #[inline]
    pub fn heap_reuse(self: Box<Self>) -> MatrixExpr<Box<ReplaceMatrixArray<T, D1, D2>>, D1, D2> {
        // Safety, series of equivilent types:
        // Box<MathMatrix<T, D1, D2>>
        // Box<MatrixExpr<MatrixArray<T, D1, D2>, D1, D2>>, de-alias MathMatrix
        // Box<ManuallyDrop<[[T; D1]; D2]>>, MatrixExpr and MatrixArray are transparent
        // MatrixExpr<Box<ReplaceMatrixArray<T, D1, D2>>, D1, D2>, MatrixExpr and ReplaceMatrixArray are transparent
        unsafe {
            mem::transmute::<Box<Self>, MatrixExpr<Box<ReplaceMatrixArray<T, D1, D2>>, D1, D2>>(self)
        }
    }

    /// converts this MathMatrix to a repeatable MatrixExpr w/ Item = &'a T
    #[inline]
    pub fn referred<'a>(self) -> MatrixExpr<ReferringMatrixArray<'a, T, D1, D2>, D1, D2>
    where
        T: 'a,
    {
        MatrixExpr(ReferringMatrixArray(
            unsafe {
                mem::transmute_copy::<ManuallyDrop<[[T; D1]; D2]>, [[T; D1]; D2]>(&self.unwrap().0)
            },
            std::marker::PhantomData,
        ))
    }

    /// references the element at index without checking bounds
    /// safety: index is in bounds
    #[inline]
    pub unsafe fn get_unchecked<I: std::slice::SliceIndex<[[T; D1]]>>(
        &self,
        index: I,
    ) -> &I::Output {
        unsafe { self.0.0.get_unchecked(index) }
    }
    /// mutably references the element at index without checking bounds
    /// safety: index is in bounds
    #[inline]
    pub unsafe fn get_unchecked_mut<I: std::slice::SliceIndex<[[T; D1]]>>(
        &mut self,
        index: I,
    ) -> &mut I::Output {
        unsafe { self.0.0.get_unchecked_mut(index) }
    }
}

impl<T: alga::general::Field + Copy, const D1: usize, const D2: usize> MathMatrix<T, D1, D2> {
    pub fn rref(&mut self) {
        use std::cmp::min;
        use std::collections::HashMap;

        let mut pivots = Vec::with_capacity(D1);
        for row_idx in 0..D1 {
            let mut pivot = 0;
            while (pivot < D2) && (self[pivot][row_idx].is_zero()) {
                pivot += 1;
            }
            pivots.push(pivot);
            if pivot == D2 {
                continue;
            }
            let base_val = self[pivot][row_idx];

            for j in pivot..D2 {
                self[j][row_idx] /= base_val;
            }

            let mut multipliers = Vec::with_capacity(D1 - 1);
            for i in 0..D1 {
                multipliers.push(self[pivot][i]);
                self[pivot][i] = T::zero();
            }
            self[pivot][row_idx] = T::one();
            multipliers[row_idx] = T::zero();
            for j in pivot + 1..D2 {
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
            if pivot == D2 {
                break;
            }
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
                    pivots.swap(src, dst);
                }
            }
        }
    }

    fn det_inner(&mut self) -> T {
        assert_eq!(
            D1, D2,
            "math_vector error: can't get the determinant of a {}x{} matrix (not square)",
            D1, D2
        );

        let mut out = T::one();
        let mut pivots = Vec::with_capacity(D2);
        for col_idx in 0..D2 {
            let mut pivot = 0;
            while (pivot < D1) && (self[col_idx][pivot].is_zero()) {
                pivot += 1;
            }
            pivots.push(pivot);
            if pivot == D1 {
                return T::zero();
            }
            let base_val = self[col_idx][pivot];
            out *= base_val;

            for j in pivot..D1 {
                self[col_idx][j] /= base_val;
            }
            for i in col_idx + 1..D2 {
                let multiplier = self[i][pivot];
                self[i][pivot] = T::zero();
                for j in pivot + 1..D1 {
                    let sub = multiplier * self[col_idx][j];
                    self[i][j] -= sub;
                }
            }
        }

        for src in 0..D2 {
            if pivots[src] != src {
                out = T::zero() - out; //scuff math moment
                let dst = pivots[src];
                pivots.swap(dst, src);
            }
        }
        out
    }

    #[inline(always)]
    pub fn det(mut self) -> T {
        self.det_inner()
    }

    #[inline(always)]
    #[allow(clippy::boxed_local)] //the box is needed to keep it on the heap
    pub fn det_heap(mut self: Box<Self>) -> T {
        self.det_inner()
    }
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
        MatrixExpr(MatrixArray(ManuallyDrop::new(value)))
    }
}

impl<T, const D1: usize, const D2: usize> From<MathMatrix<T, D1, D2>> for [[T; D1]; D2] {
    #[inline]
    fn from(value: MathMatrix<T, D1, D2>) -> Self {
        value.unwrap().unwrap()
    }
}

impl<T, const D1: usize, const D2: usize> From<Box<[[T; D1]; D2]>> for Box<MathMatrix<T, D1, D2>> {
    #[inline]
    fn from(value: Box<[[T; D1]; D2]>) -> Self {
        unsafe { mem::transmute::<Box<[[T; D1]; D2]>, Box<MathMatrix<T, D1, D2>>>(value) }
    }
}

impl<T, const D1: usize, const D2: usize> From<Box<MathMatrix<T, D1, D2>>> for Box<[[T; D1]; D2]> {
    #[inline]
    fn from(value: Box<MathMatrix<T, D1, D2>>) -> Self {
        unsafe { mem::transmute::<Box<MathMatrix<T, D1, D2>>, Box<[[T; D1]; D2]>>(value) }
    }
}

impl<'a, T, const D1: usize, const D2: usize> From<&'a [[T; D1]; D2]>
    for &'a MathMatrix<T, D1, D2>
{
    #[inline]
    fn from(value: &'a [[T; D1]; D2]) -> Self {
        unsafe { mem::transmute::<&'a [[T; D1]; D2], &'a MathMatrix<T, D1, D2>>(value) }
    }
}

impl<'a, T, const D1: usize, const D2: usize> From<&'a MathMatrix<T, D1, D2>>
    for &'a [[T; D1]; D2]
{
    #[inline]
    fn from(value: &'a MathMatrix<T, D1, D2>) -> Self {
        unsafe { mem::transmute::<&'a MathMatrix<T, D1, D2>, &'a [[T; D1]; D2]>(value) }
    }
}

impl<'a, T, const D1: usize, const D2: usize> From<&'a mut [[T; D1]; D2]>
    for &'a mut MathMatrix<T, D1, D2>
{
    #[inline]
    fn from(value: &'a mut [[T; D1]; D2]) -> Self {
        unsafe { mem::transmute::<&'a mut [[T; D1]; D2], &'a mut MathMatrix<T, D1, D2>>(value) }
    }
}

impl<'a, T, const D1: usize, const D2: usize> From<&'a mut MathMatrix<T, D1, D2>>
    for &'a mut [[T; D1]; D2]
{
    #[inline]
    fn from(value: &'a mut MathMatrix<T, D1, D2>) -> Self {
        unsafe { mem::transmute::<&'a mut MathMatrix<T, D1, D2>, &'a mut [[T; D1]; D2]>(value) }
    }
}

impl<T, const D1: usize, const D2: usize> From<MathVectoredMatrix<T, D1, D2>>
    for MathMatrix<T, D1, D2>
{
    #[inline]
    fn from(value: MathVectoredMatrix<T, D1, D2>) -> Self {
        //  safety:
        //      MathVectoredMatrix<T, D1, D2> == VectorExpr<VectorArray<VectorExpr<VectorArray<T, D1>, D1>, D2>, D2>
        //      MathVectoredMatrix<T, D1, D2> == ManuallyDrop<[ManuallyDrop<[T; D1]>; D2]>
        //      MathVectoredMatrix<T, D1, D2> == ManuallyDrop<[[T; D1]; D2]>
        //      MathVectoredMatrix<T, D1, D2> == MatrixExpr<MatrixArray<T, D1, D2>>
        //      MathVectoredMatrix<T, D1, D2> == MathMatrix<T, D1, D2>
        //
        //  FIXME: transmute_copy copies (:O), this shouldn't need to be done but gets the compiler to not complain about it
        unsafe {
            mem::transmute_copy::<MathVectoredMatrix<T, D1, D2>, MathMatrix<T, D1, D2>>(
                &ManuallyDrop::new(value),
            )
        }
    }
}

impl<T, const D1: usize, const D2: usize> From<MathMatrix<T, D1, D2>>
    for MathVectoredMatrix<T, D1, D2>
{
    #[inline]
    fn from(value: MathMatrix<T, D1, D2>) -> MathVectoredMatrix<T, D1, D2> {
        //  safety:
        //      MathVectoredMatrix<T, D1, D2> == VectorExpr<VectorArray<VectorExpr<VectorArray<T, D1>, D1>, D2>, D2>
        //      MathVectoredMatrix<T, D1, D2> == ManuallyDrop<[ManuallyDrop<[T; D1]>; D2]>
        //      MathVectoredMatrix<T, D1, D2> == ManuallyDrop<[[T; D1]; D2]>
        //      MathVectoredMatrix<T, D1, D2> == MatrixExpr<MatrixArray<T, D1, D2>>
        //      MathVectoredMatrix<T, D1, D2> == MathMatrix<T, D1, D2>
        //
        //  FIXME: transmute_copy copies (:O), this shouldn't need to be done but gets the compiler to not complain about it
        unsafe {
            mem::transmute_copy::<MathMatrix<T, D1, D2>, MathVectoredMatrix<T, D1, D2>>(
                &ManuallyDrop::new(value),
            )
        }
    }
}

type MathVectoredMatrix<T, const D1: usize, const D2: usize> = 
    MathVector<MathVector<T, D1>, D2>;

impl<T, I, const D1: usize, const D2: usize> Index<I> for MathMatrix<T, D1, D2>
where
    [[T; D1]; D2]: Index<I, Output = [T; D1]>,
{
    type Output = MathVector<T, D1>;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        (&self.0.0[index]).into()
    }
}

impl<'a, T, I, const D1: usize, const D2: usize> Index<I> for MatrixExpr<&'a [[T; D1]; D2], D1, D2>
where 
    [[T; D1]; D2]: Index<I, Output = [T; D1]>,
{
    type Output = MathVector<T, D1>;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        (&self.0[index]).into()
    }
}

impl<'a, T, I, const D1: usize, const D2: usize> Index<I> for MatrixExpr<&'a mut [[T; D1]; D2], D1, D2>
where 
    [[T; D1]; D2]: Index<I, Output = [T; D1]>,
{
    type Output = MathVector<T, D1>;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        (&self.0[index]).into()
    }
}

impl<T, I, const D1: usize, const D2: usize> IndexMut<I> for MathMatrix<T, D1, D2>
where
    [[T; D1]; D2]: IndexMut<I, Output = [T; D1]>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        (&mut self.0.0[index]).into()
    }
}

impl<'a, T, I, const D1: usize, const D2: usize> IndexMut<I> for MatrixExpr<&'a mut [[T; D1]; D2], D1, D2>
where 
    [[T; D1]; D2]: IndexMut<I, Output = [T; D1]>,
{
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        (&mut self.0[index]).into()
    }
}

impl<T: MulAssign<S>, S: Copy, const D1: usize, const D2: usize> MulAssign<S> for MathMatrix<T, D1, D2> {
    #[inline]
    fn mul_assign(&mut self, rhs: S) {
        MatrixOps::mul_assign(self, rhs).consume();
    }
}

impl<T: DivAssign<S>, S: Copy, const D1: usize, const D2: usize> DivAssign<S> for MathMatrix<T, D1, D2> {
    #[inline]
    fn div_assign(&mut self, rhs: S) {
        MatrixOps::div_assign(self, rhs).consume();
    }
}

impl<T: RemAssign<S>, S: Copy, const D1: usize, const D2: usize> RemAssign<S> for MathMatrix<T, D1, D2> {
    #[inline]
    fn rem_assign(&mut self, rhs: S) {
        MatrixOps::rem_assign(self, rhs).consume();
    }
}

impl<T: AddAssign<<M::Unwrapped as Get2D>::Item>, M: MatrixOps, const D1: usize, const D2: usize> AddAssign<M> for MathMatrix<T, D1, D2> 
where 
    M::Unwrapped: Has2DReuseBuf<BoundHandlesBool = N>,
    M::Unwrapped: HasOutput<OutputBool = N>,
    MatrixExprBuilder<D1, D2>: MatrixBuilderUnion<M::Builder>,
    (N, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
{
    #[inline]
    fn add_assign(&mut self, rhs: M) {
        MatrixOps::add_assign(self, rhs).consume();
    }
}

impl<T: SubAssign<<M::Unwrapped as Get2D>::Item>, M: MatrixOps, const D1: usize, const D2: usize> SubAssign<M> for MathMatrix<T, D1, D2> 
where 
    M::Unwrapped: Has2DReuseBuf<BoundHandlesBool = N>,
    M::Unwrapped: HasOutput<OutputBool = N>,
    MatrixExprBuilder<D1, D2>: MatrixBuilderUnion<M::Builder>,
    (N, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
{
    #[inline]
    fn sub_assign(&mut self, rhs: M) {
        MatrixOps::sub_assign(self, rhs).consume();
    }
}

impl<T: std::fmt::Display, const D1: usize, const D2: usize> std::fmt::Display
    for MathMatrix<T, D1, D2>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn add_line(
            f: &mut std::fmt::Formatter<'_>,
            l_end: char,
            r_end: char,
            strings: &[Vec<String>],
            col_max_lengths: &[usize],
            row: usize,
        ) -> std::fmt::Result {
            write!(f, "\n{} {:2$}", l_end, strings[0][row], col_max_lengths[0])?;
            for col in 1..strings.len() {
                write!(f, ", {:1$}", strings[col][row], col_max_lengths[col])?;
            }
            write!(f, " {}", r_end)?;
            Ok(())
        }

        let mut strings = Vec::with_capacity(D2);
        let mut col_max_lengths = Vec::with_capacity(D2);
        for col in <&[[_; _]; _]>::from(self) {
            let mut col_strings = Vec::with_capacity(D1);
            let mut max_length = 0;
            for v in col {
                let str = v.to_string();
                max_length = std::cmp::max(max_length, str.len());
                col_strings.push(str);
            }
            strings.push(col_strings);
            col_max_lengths.push(max_length);
        }
        if (D1 != 0) & (D2 != 0) {
            if D1 > 1 {
                add_line(f, '┌', '┐', &strings, &col_max_lengths, 0)?;
                for row in 1..D1 - 1 {
                    add_line(f, '│', '│', &strings, &col_max_lengths, row)?;
                }
                add_line(f, '└', '┘', &strings, &col_max_lengths, D1 - 1)?;
            } else {
                add_line(f, '[', ']', &strings, &col_max_lengths, 0)?;
            }
        } else {
            write!(f, "\n[]")?;
        }

        Ok(())
    }
}



#[derive(Clone)]
pub struct RSMatrixExpr<M: MatrixLike> {
    pub(crate) mat: M,
    pub(crate) num_rows: usize,
    pub(crate) num_cols: usize,
}

impl<M: MatrixLike> RSMatrixExpr<M> {
    #[inline]
    pub fn const_sized<const D1: usize, const D2: usize>(self) -> MatrixExpr<M, D1, D2> {
        todo!()
    }
}

impl<M: MatrixLike + Is2DRepeatable> RSMatrixExpr<M> {
    /// Retrieves the value at an arbitrary index of a repeatable matrix
    /// Note:   This method does NOT fill any buffers bound to the matrix
    pub fn get(&mut self, col_index: usize, row_index: usize) -> M::Item {
        if (col_index >= self.num_cols) | (row_index >= self.num_rows) {
            panic!("math_vector Error: index access out of bound")
        }
        unsafe {
            let inputs = self.mat.get_inputs(col_index, row_index);
            let (item, _) = self.mat.process(col_index, row_index, inputs);
            item
        }
    }
}

impl<M: MatrixLike> Drop for RSMatrixExpr<M> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            for col_index in 0..self.num_cols {
                for row_index in 0..self.num_rows {
                    self.mat.drop_inputs(col_index, row_index);
                }
            }
            self.mat.drop_output();
            self.mat.drop_1st_buffer();
            self.mat.drop_2nd_buffer();
        }
    }
}

pub type RSMathDopeMatrix<T> = RSMatrixExpr<MatrixDopeSlice<T>>;

impl<T> RSMathDopeMatrix<T> {
    //TODO: add missing fns

    #[inline]
    pub fn borrow(&self) -> RefRSMathDopeMatrix<'_, T> {
        let num_rows = self.num_rows;
        let num_cols = self.num_cols;
        RSMatrixExpr{ 
            mat: RefMatrixDopeSlice { 
                mat: unsafe { transmute::<&[ManuallyDrop<T>], &[T]>(&*self.mat.mat) }, 
                height: num_rows 
            },
            num_rows,
            num_cols,
        }
    }

    #[inline]
    pub fn borrow_mut(&mut self) -> RefMutRSMathDopeMatrix<'_, T> {
        let num_rows = self.num_rows;
        let num_cols = self.num_cols;
        RSMatrixExpr{ 
            mat: RefMutMatrixDopeSlice { 
                mat: unsafe { transmute::<&mut [ManuallyDrop<T>], &mut [T]>(&mut *self.mat.mat) }, 
                height: num_rows 
            },
            num_rows,
            num_cols,
        }
    }
}

/// NOTE: this may get removed / modified in the future for cases where this can be impl'd better without going through an Iliffe Matrix
impl<T, U> From<U> for RSMathDopeMatrix<T> where RSMathIliffeMatrix<T>: From<U> {
    fn from(value: U) -> Self {
        let iliffe_mat_expr = value.into();
        let RSMatrixExpr { mat: _, num_rows, num_cols } = iliffe_mat_expr;
        // some shenaniganery bc you cant move mat out from iliffe_mat_expr since iliffe_mat_expr impl's Drop and mat isn't Copy
        let iliffe_mat = iliffe_mat_expr.unwrap();
        let mut dope_mat = Vec::with_capacity(num_rows * num_cols);
        for column in iliffe_mat.0 {
            dope_mat.extend(column.into_iter());
        }
        RSMatrixExpr { 
            mat: MatrixDopeSlice { mat: dope_mat.into_boxed_slice(), height: num_rows }, 
            num_rows, 
            num_cols 
        }
    }
}

impl<T> Index<usize> for RSMathDopeMatrix<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let slice_start = index * self.mat.height;
        let raw = &self.mat.mat[slice_start..slice_start + self.mat.height];
        unsafe{ transmute::<&[ManuallyDrop<T>], &[T]>(raw) }
    }
}

impl<T> IndexMut<usize> for RSMathDopeMatrix<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let slice_start = index * self.mat.height;
        let raw = &mut self.mat.mat[slice_start..slice_start + self.mat.height];
        unsafe{ transmute::<&mut [ManuallyDrop<T>], &mut [T]>(raw) }
    }
}


impl<T: MulAssign<S>, S: Copy> MulAssign<S> for RSMathDopeMatrix<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: S) {
        MatrixOps::mul_assign(self, rhs).consume();
    }
}

impl<T: DivAssign<S>, S: Copy> DivAssign<S> for RSMathDopeMatrix<T> {
    #[inline]
    fn div_assign(&mut self, rhs: S) {
        MatrixOps::div_assign(self, rhs).consume();
    }
}

impl<T: RemAssign<S>, S: Copy> RemAssign<S> for RSMathDopeMatrix<T> {
    #[inline]
    fn rem_assign(&mut self, rhs: S) {
        MatrixOps::rem_assign(self, rhs).consume();
    }
}

impl<T: AddAssign<<M::Unwrapped as Get2D>::Item>, M: MatrixOps> AddAssign<M> for RSMathDopeMatrix<T> 
where 
    M::Unwrapped: Has2DReuseBuf<BoundHandlesBool = N>,
    M::Unwrapped: HasOutput<OutputBool = N>,
    RSMatrixExprBuilder: MatrixBuilderUnion<M::Builder>,
    (N, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
{
    #[inline]
    fn add_assign(&mut self, rhs: M) {
        MatrixOps::add_assign(self, rhs).consume();
    }
}

impl<T: SubAssign<<M::Unwrapped as Get2D>::Item>, M: MatrixOps> SubAssign<M> for RSMathDopeMatrix<T> 
where 
    M::Unwrapped: Has2DReuseBuf<BoundHandlesBool = N>,
    M::Unwrapped: HasOutput<OutputBool = N>,
    RSMatrixExprBuilder: MatrixBuilderUnion<M::Builder>,
    (N, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
{
    #[inline]
    fn sub_assign(&mut self, rhs: M) {
        MatrixOps::sub_assign(self, rhs).consume();
    }
}

pub type RefRSMathDopeMatrix<'a, T> = RSMatrixExpr<RefMatrixDopeSlice<'a, T>>;

// TODO if possible: `From` impl's (bc any normal 2d structure like Vec<Vec<T>> just doesn't have such a contiguous slice to use)

impl<'a, T> Index<usize> for RefRSMathDopeMatrix<'a, T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let slice_start = index * self.mat.height;
        &self.mat.mat[slice_start..slice_start + self.mat.height]
    }
} 

pub type RefMutRSMathDopeMatrix<'a, T> = RSMatrixExpr<RefMutMatrixDopeSlice<'a, T>>;

// TODO if possible: `From` impl's

impl<'a, T> Index<usize> for RefMutRSMathDopeMatrix<'a, T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let slice_start = index * self.mat.height;
        &self.mat.mat[slice_start..slice_start + self.mat.height]
    }
}

impl<'a, T> IndexMut<usize> for RefMutRSMathDopeMatrix<'a, T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let slice_start = index * self.mat.height;
        &mut self.mat.mat[slice_start..slice_start + self.mat.height]
    }
}

pub type RSMathIliffeMatrix<T> = RSMatrixExpr<MatrixIliffeSlice<T>>;
pub type RefRSMathIliffeMatrix<'a, T> = RSMatrixExpr<&'a [&'a [T]]>;
pub type RefBoxRSMathIliffeMatrix<'a, T> = RSMatrixExpr<&'a [Box<[T]>]>;
pub type RefMutRSMathIliffeMatrix<'a, T> = RSMatrixExpr<&'a mut [&'a mut [T]]>;
pub type RefMutBoxRSMathIliffeMatrix<'a, T> = RSMatrixExpr<&'a mut [Box<[T]>]>;

impl<T> RSMathIliffeMatrix<T> {
    //TODO: add missing fns

    #[inline]
    pub fn borrow(&self) -> RefBoxRSMathIliffeMatrix<'_, T> {
        let num_rows = self.num_rows;
        let num_cols = self.num_cols;
        RSMatrixExpr{ 
            mat: unsafe {transmute::<&[Box<[ManuallyDrop<T>]>], &[Box<[T]>]>(&*self.mat.0)},
            num_rows,
            num_cols,
        }
    }

    #[inline]
    pub fn borrow_mut(&mut self) -> RefMutBoxRSMathIliffeMatrix<'_, T> {
        let num_rows = self.num_rows;
        let num_cols = self.num_cols;
        RSMatrixExpr{ 
            mat: unsafe { transmute::<&mut [Box<[ManuallyDrop<T>]>], &mut [Box<[T]>]>(&mut *self.mat.0) }, 
            num_rows,
            num_cols,
        }
    }
}

/// not entirely free as this includes a check that the Box<[Box<[T]>]> is rectangular, but that is highly minimal
impl<T> From<Box<[Box<[T]>]>> for RSMathIliffeMatrix<T> {
    fn from(value: Box<[Box<[T]>]>) -> Self {
        let num_cols = value.len();
        let num_rows = if num_cols > 0 {value[0].len()} else {0};
        for column in value.iter() {
            assert_eq!(num_rows, column.len(), "math_vector error: input data type was not rectangular and couldn't be made into a matrix");
        }
        RSMatrixExpr { 
            mat: MatrixIliffeSlice(
                unsafe { transmute::<Box<[Box<[T]>]>, Box<[Box<[ManuallyDrop<T>]>]>>(value) }
            ),
            num_rows, 
            num_cols 
        }
    }
}

/// not entirely free as this includes a check that the Vec<Box<[T]>> is rectangular, but that is highly minimal
impl<T> From<Vec<Box<[T]>>> for RSMathIliffeMatrix<T> {
    fn from(value: Vec<Box<[T]>>) -> Self {
        Self::from(value.into_boxed_slice())
    }
}

/// interpretted as a Box<[]> of columns
/// *NOT A FREE CONVERSION*
impl<T> From<Box<[Vec<T>]>> for RSMathIliffeMatrix<T> {
    fn from(value: Box<[Vec<T>]>) -> Self {
        let num_cols = value.len();
        let num_rows = if num_cols > 0 {value[0].len()} else {0};
        let mut illife_matrix = Vec::with_capacity(num_cols);
        for column in value {
            assert_eq!(num_rows, column.len(), "math_vector error: input data type was not rectangular and couldn't be made into a matrix");
            let column = unsafe { transmute::<Box<[T]>, Box<[ManuallyDrop<T>]>>(column.into_boxed_slice()) };
            illife_matrix.push(column);
        }
        RSMatrixExpr { 
            mat: MatrixIliffeSlice(illife_matrix.into_boxed_slice()), 
            num_rows, 
            num_cols 
        }
    }
}

/// interpretted as a Vec of columns
/// *NOT A FREE CONVERSION*
impl<T> From<Vec<Vec<T>>> for RSMathIliffeMatrix<T> {
    fn from(value: Vec<Vec<T>>) -> Self {
        Self::from(value.into_boxed_slice())
    }
}

impl<T, I> Index<I> for RSMathIliffeMatrix<T> where [Box<[T]>]: Index<I> {
    type Output = <[Box<[T]>] as Index<I>>::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        // SAFETY: this transmute isn't *strictly* needed but it allows for a cleaner where bound, safe bc `ManuallyDrop` is `repr(transparent)`
        &(unsafe { transmute::<&[Box<[ManuallyDrop<T>]>], &[Box<[T]>]>(&self.mat.0) })[index]
    }
}

impl<T> IndexMut<usize> for RSMathIliffeMatrix<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        // SAFETY: this transmute isn't *strictly* needed but it allows for a cleaner where bound, safe bc `ManuallyDrop` is `repr(transparent)`
        &mut (unsafe { transmute::<&mut [Box<[ManuallyDrop<T>]>], &mut [Box<[T]>]>(&mut self.mat.0) })[index]
    }
}



impl<T: MulAssign<S>, S: Copy> MulAssign<S> for RSMathIliffeMatrix<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: S) {
        MatrixOps::mul_assign(self, rhs).consume();
    }
}

impl<T: DivAssign<S>, S: Copy> DivAssign<S> for RSMathIliffeMatrix<T> {
    #[inline]
    fn div_assign(&mut self, rhs: S) {
        MatrixOps::div_assign(self, rhs).consume();
    }
}

impl<T: RemAssign<S>, S: Copy> RemAssign<S> for RSMathIliffeMatrix<T> {
    #[inline]
    fn rem_assign(&mut self, rhs: S) {
        MatrixOps::rem_assign(self, rhs).consume();
    }
}

impl<T: AddAssign<<M::Unwrapped as Get2D>::Item>, M: MatrixOps> AddAssign<M> for RSMathIliffeMatrix<T> 
where 
    M::Unwrapped: Has2DReuseBuf<BoundHandlesBool = N>,
    M::Unwrapped: HasOutput<OutputBool = N>,
    RSMatrixExprBuilder: MatrixBuilderUnion<M::Builder>,
    (N, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
{
    #[inline]
    fn add_assign(&mut self, rhs: M) {
        MatrixOps::add_assign(self, rhs).consume();
    }
}

impl<T: SubAssign<<M::Unwrapped as Get2D>::Item>, M: MatrixOps> SubAssign<M> for RSMathIliffeMatrix<T> 
where 
    M::Unwrapped: Has2DReuseBuf<BoundHandlesBool = N>,
    M::Unwrapped: HasOutput<OutputBool = N>,
    RSMatrixExprBuilder: MatrixBuilderUnion<M::Builder>,
    (N, <M::Unwrapped as Get2D>::AreInputsTransposed): TyBoolPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::FstHandleBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::SndHandleBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
    (N, <M::Unwrapped as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair,
{
    #[inline]
    fn sub_assign(&mut self, rhs: M) {
        MatrixOps::sub_assign(self, rhs).consume();
    }
}


/// not entirely free as this includes a check that it is rectangular, but that is highly minimal
impl<'a, T: 'a, S: Deref<Target = [T]>> From<&'a [S]> for RSMatrixExpr<&'a [S]> {
    fn from(value: &'a [S]) -> Self {
        let num_cols = value.len();
        let num_rows = if num_cols > 0 {value[0].len()} else {0};
        for column in value {
            assert_eq!(num_rows, column.len(), "math_vector error: input data type was not rectangular and couldn't be made into a matrix");
        }
        RSMatrixExpr { mat: value, num_rows, num_cols }
    }
}

impl<'a, T: 'a, S: Deref<Target = [T]>, I> Index<I> for RSMatrixExpr<&'a [S]> where [S]: Index<I, Output = S> {
    type Output = S;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        &self.mat[index]
    }
}

/// not entirely free as this includes a check that it is rectangular, but that is highly minimal
impl<'a, T: 'a, S: DerefMut<Target = [T]>> From<&'a mut [S]> for RSMatrixExpr<&'a mut [S]> {
    fn from(value: &'a mut [S]) -> Self {
        let num_cols = value.len();
        let num_rows = if num_cols > 0 {value[0].len()} else {0};
        for column in value.iter() {
            assert_eq!(num_rows, column.len(), "math_vector error: input data type was not rectangular and couldn't be made into a matrix");
        }
        RSMatrixExpr { mat: value, num_rows, num_cols }
    }
}

impl<'a, T: 'a, S: DerefMut<Target = [T]>, I> Index<I> for RSMatrixExpr<&'a mut [S]> where [S]: Index<I, Output = S> {
    type Output = S;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        &self.mat[index]
    }
}

impl<'a, T: 'a, S: DerefMut<Target = [T]>, I> IndexMut<I> for RSMatrixExpr<&'a mut [S]> where [S]: IndexMut<I, Output = S> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.mat[index]
    }
}
