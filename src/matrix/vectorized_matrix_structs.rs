//use crate::trait_specialization_utils::*;
use crate::util_traits::*;
use crate::vector::{vec_util_traits::*,MathVector};
use super::mat_util_traits::MatrixWrapperBuilder;





//implies the returned vectors must be of the exact same length
pub unsafe trait VectorizedMatrix: VectorLike<Item = Self::Vector> {type Vector: VectorLike;}

pub type MathVectoredMatrix<T,const D1: usize,const D2: usize> = MathVector<MathVector<T,D1>,D2>;

pub struct MatColWrapper<M: VectorLike<Item = V>, V: VectorLike, Wrap: MatrixWrapperBuilder>{pub(crate) mat: M, pub(crate) wrapper_builder: Wrap}

unsafe impl<M: VectorLike<Item = V>, V: VectorLike, Wrap: MatrixWrapperBuilder> Get for MatColWrapper<M,V,Wrap> {
    type GetBool = M::GetBool;
    type IsRepeatable = M::GetBool;
    type Inputs = M::Inputs;
    type Item = Wrap::VectorWrapped<M::Item>;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {self.mat.get_inputs(index)}
    #[inline] unsafe fn drop_inputs(&mut self, index: usize) {self.mat.drop_inputs(index);}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
        let (col, bound) =  self.mat.process(inputs);
        (unsafe { self.wrapper_builder.wrap_vec(col) }, bound)
    }
}

impl<M: VectorLike<Item = V>, V: VectorLike, Wrap: MatrixWrapperBuilder> HasOutput for MatColWrapper<M,V,Wrap> {
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output {self.mat.output()}
    #[inline] unsafe fn drop_output(&mut self) {self.mat.drop_output();}
}

impl<M: VectorLike<Item = V>, V: VectorLike, Wrap: MatrixWrapperBuilder> HasReuseBuf for MatColWrapper<M,V,Wrap> {
    type FstHandleBool = M::FstHandleBool;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = M::FstOwnedBufferBool;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type FstOwnedBuffer = M::FstOwnedBuffer;
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = M::FstType;
    type SndType = M::SndType;
    type BoundTypes = M::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self,index: usize,val: Self::FstType) {self.mat.assign_1st_buf(index,val)}
    #[inline] unsafe fn assign_2nd_buf(&mut self,index: usize,val: Self::SndType) {self.mat.assign_2nd_buf(index,val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self,index: usize,val: Self::BoundTypes) {self.mat.assign_bound_bufs(index,val)}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {self.mat.get_1st_buffer()}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.mat.get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self,index: usize) {self.mat.drop_1st_buf_index(index)}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self,index: usize) {self.mat.drop_2nd_buf_index(index)}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self,index: usize) {self.mat.drop_bound_bufs_index(index)}    
}



pub struct MatRowWrapper<M: VectorLike<Item = V>, V: VectorLike, Wrap: MatrixWrapperBuilder>{pub(crate) mat: M, pub(crate) wrapper_builder: Wrap}

unsafe impl<M: VectorLike<Item = V>, V: VectorLike, Wrap: MatrixWrapperBuilder> Get for MatRowWrapper<M,V,Wrap> {
    type GetBool = M::GetBool;
    type IsRepeatable = M::GetBool;
    type Inputs = M::Inputs;
    type Item = Wrap::TransposedVectorWrapped<M::Item>;
    type BoundItems = M::BoundItems;

    #[inline] unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {self.mat.get_inputs(index)}
    #[inline] unsafe fn drop_inputs(&mut self, index: usize) {self.mat.drop_inputs(index);}
    #[inline] fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
        let (col, bound) =  self.mat.process(inputs);
        (unsafe { self.wrapper_builder.wrap_trans_vec(col) }, bound)
    }
}

impl<M: VectorLike<Item = V>, V: VectorLike, Wrap: MatrixWrapperBuilder> HasOutput for MatRowWrapper<M,V,Wrap> {
    type OutputBool = M::OutputBool;
    type Output = M::Output;

    #[inline] unsafe fn output(&mut self) -> Self::Output {self.mat.output()}
    #[inline] unsafe fn drop_output(&mut self) {self.mat.drop_output();}
}

impl<M: VectorLike<Item = V>, V: VectorLike, Wrap: MatrixWrapperBuilder> HasReuseBuf for MatRowWrapper<M,V,Wrap> {
    type FstHandleBool = M::FstHandleBool;
    type SndHandleBool = M::SndHandleBool;
    type BoundHandlesBool = M::BoundHandlesBool;
    type FstOwnedBufferBool = M::FstOwnedBufferBool;
    type SndOwnedBufferBool = M::SndOwnedBufferBool;
    type FstOwnedBuffer = M::FstOwnedBuffer;
    type SndOwnedBuffer = M::SndOwnedBuffer;
    type FstType = M::FstType;
    type SndType = M::SndType;
    type BoundTypes = M::BoundTypes;

    #[inline] unsafe fn assign_1st_buf(&mut self,index: usize,val: Self::FstType) {self.mat.assign_1st_buf(index,val)}
    #[inline] unsafe fn assign_2nd_buf(&mut self,index: usize,val: Self::SndType) {self.mat.assign_2nd_buf(index,val)}
    #[inline] unsafe fn assign_bound_bufs(&mut self,index: usize,val: Self::BoundTypes) {self.mat.assign_bound_bufs(index,val)}
    #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer {self.mat.get_1st_buffer()}
    #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer {self.mat.get_2nd_buffer()}
    #[inline] unsafe fn drop_1st_buf_index(&mut self,index: usize) {self.mat.drop_1st_buf_index(index)}
    #[inline] unsafe fn drop_2nd_buf_index(&mut self,index: usize) {self.mat.drop_2nd_buf_index(index)}
    #[inline] unsafe fn drop_bound_bufs_index(&mut self,index: usize) {self.mat.drop_bound_bufs_index(index)}    
}

//TODO: add lazy Mat multiplication
//pub struct MatMul<M1: VectorizedMatrix<Vector = V1>, V1: VectorLike, M2: VectorizedMatrix<Vector = V2, IsRepeatable = Y>, V2: VectorLike>{pub(crate) l_mat: M1, pub(crate) r_mat: M2, pub(crate) shared_dimension: usize}
//
//unsafe impl<M1: VectorizedMatrix<Vector = V1>, V1: VectorLike, M2: VectorizedMatrix<Vector = V2, IsRepeatable = Y>, V2: VectorLike> Get for MatMul<M1,V1,M2,V2> where 
//    (M1::BoundHandlesBool,M2::BoundHandlesBool): FilterPair,
//{
//    type GetBool = Y;
//    type IsRepeatable = N;
//    type Inputs = M1::Inputs;
//    type Item = ();
//    type BoundItems = M1::BoundItems;
//
//    unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs {self.l_mat.get_inputs(index)}
//    unsafe fn drop_inputs(&mut self, index: usize) {self.l_mat.drop_inputs(index)}
//    fn process(&mut self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
//        let (l_col,l_bound) =  self.l_mat.process(inputs);
//        let mut l_vectorlike = VecMap()
//    }
//}