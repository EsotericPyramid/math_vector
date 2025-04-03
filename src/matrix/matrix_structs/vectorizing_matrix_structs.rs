use crate::trait_specialization_utils::*;
use crate::util_traits::*;
use crate::matrix::mat_util_traits::*;
use crate::matrix::VectorizedMatrix;
use crate::vector::vec_util_traits::*;

pub struct MatrixColumn<M: MatrixLike>{pub(crate) mat: *mut M, pub(crate) column_num: usize}

unsafe impl<M: MatrixLike> Get for MatrixColumn<M> {
    type GetBool = M::GetBool;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline]
    unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {unsafe { (*self.mat).get_inputs(self.column_num, index)} }

    #[inline]
    unsafe fn drop_inputs(&mut self, index: usize) {unsafe { (*self.mat).drop_inputs(self.column_num, index)} }
    
    #[inline]
    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {unsafe { (*self.mat).process(inputs)}}
}

unsafe impl<M: IsRepeatable + MatrixLike> IsRepeatable for MatrixColumn<M> {}

impl<M: MatrixLike> HasOutput for MatrixColumn<M> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

impl<M: MatrixLike> HasReuseBuf for MatrixColumn<M> {
    type FstHandleBool = N;
    type SndHandleBool = N;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = N;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = ();
    type FstType = ();
    type SndType = ();
    type BoundTypes = M::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, _: usize, _: Self::SndType) {}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) { unsafe {(*self.mat).assign_bound_bufs(self.column_num, index, val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {(*self.mat).drop_bound_bufs_index(self.column_num, index)}}
}


pub struct MatColVectorExprs<M: MatrixLike>{pub(crate) mat: M}

unsafe impl<M: MatrixLike> Get for MatColVectorExprs<M> {
    type GetBool = M::GetBool;
    type Inputs = usize;
    type Item = MatrixColumn<M>;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {index}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(MatrixColumn{mat: &mut self.mat as *mut M, column_num: inputs}, ())}
}

unsafe impl<M: IsRepeatable + MatrixLike> IsRepeatable for MatColVectorExprs<M> {}

impl<M: MatrixLike> HasOutput for MatColVectorExprs<M> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.mat.drop_output()}}
}

impl<M: MatrixLike> HasReuseBuf for MatColVectorExprs<M> {
    type FstHandleBool = N;
    type SndHandleBool = N;
    type BoundHandlesBool = N;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = N;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = ();
    type FstType = ();
    type SndType = ();
    type BoundTypes = ();

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, _: usize, _: Self::SndType) {}
    #[inline] unsafe fn assign_bound_bufs(&mut self, _: usize, _: Self::BoundTypes) {}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, _: usize) {}
}

unsafe impl<M: MatrixLike> VectorizedMatrix for MatColVectorExprs<M> {}


pub struct MatrixRow<M: MatrixLike>{pub(crate) mat: *mut M, pub(crate) row_num: usize}

unsafe impl<M: MatrixLike> Get for MatrixRow<M> {
    type GetBool = M::GetBool;
    type Inputs = M::Inputs;
    type Item = M::Item;
    type BoundItems = M::BoundItems;

    #[inline]
    unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {unsafe { (*self.mat).get_inputs(index, self.row_num)} }

    #[inline]
    unsafe fn drop_inputs(&mut self, index: usize) {unsafe { (*self.mat).drop_inputs(index, self.row_num)} }
    
    #[inline]
    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {unsafe { (*self.mat).process(inputs)}}
}

unsafe impl<M: IsRepeatable + MatrixLike> IsRepeatable for MatrixRow<M> {}

impl<M: MatrixLike> HasOutput for MatrixRow<M> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) {}
}

impl<M: MatrixLike> HasReuseBuf for MatrixRow<M> {
    type FstHandleBool = N;
    type SndHandleBool = N;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = N;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = ();
    type FstType = ();
    type SndType = ();
    type BoundTypes = M::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, _: usize, _: Self::SndType) {}
    #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) { unsafe {(*self.mat).assign_bound_bufs(index, self.row_num, val)}}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {(*self.mat).drop_bound_bufs_index(index, self.row_num)}}
}


pub struct MatRowVectorExprs<M: MatrixLike>{pub(crate) mat: M}

unsafe impl<M: MatrixLike> Get for MatRowVectorExprs<M> {
    type GetBool = M::GetBool;
    type Inputs = usize;
    type Item = MatrixRow<M>;
    type BoundItems = ();

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {index}
    #[inline] unsafe fn drop_inputs(&mut self, _: usize) {}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {(MatrixRow{mat: &mut self.mat as *mut M, row_num: inputs}, ())}
}

unsafe impl<M: IsRepeatable + MatrixLike> IsRepeatable for MatRowVectorExprs<M> {}

impl<M: MatrixLike> HasOutput for MatRowVectorExprs<M> {
    type OutputBool = N;
    type Output = ();

    #[inline] unsafe fn output(&mut self) -> Self::Output {}
    #[inline] unsafe fn drop_output(&mut self) { unsafe {self.mat.drop_output()}}
}

impl<M: MatrixLike> HasReuseBuf for MatRowVectorExprs<M> {
    type FstHandleBool = N;
    type SndHandleBool = N;
    type BoundHandlesBool = N;
    type FstOwnedBufferBool = N;
    type SndOwnedBufferBool = N;
    type FstOwnedBuffer = ();
    type SndOwnedBuffer = ();
    type FstType = ();
    type SndType = ();
    type BoundTypes = ();

    #[inline] unsafe fn assign_1st_buf(&mut self, _: usize, _: Self::FstType) {}
    #[inline] unsafe fn assign_2nd_buf(&mut self, _: usize, _: Self::SndType) {}
    #[inline] unsafe fn assign_bound_bufs(&mut self, _: usize, _: Self::BoundTypes) {}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::FstOwnedBuffer {}
    #[inline] unsafe fn drop_1st_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self, _: usize) {}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self, _: usize) {}
}

unsafe impl<M: MatrixLike> VectorizedMatrix for MatRowVectorExprs<M> {}
