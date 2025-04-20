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

///really just a shorthand for the individual traits
pub trait VectorLike: Get + HasOutput + HasReuseBuf {}

impl<T: Get + HasOutput + HasReuseBuf> VectorLike for T {}

pub trait VectorBuilder: Clone {
    type Wrapped<T: VectorLike>;

    ///Safety: The VectorLike passed to this function MUST match the implications of the wrapper (ATM (Oct 2024), just needs to be unused)
    unsafe fn wrap<T: VectorLike>(&self, vec: T) -> Self::Wrapped<T>;
}

pub trait VectorBuilderUnion<T: VectorBuilder>: VectorBuilder {
    type Union: VectorBuilder;

    fn union(self, other: T) -> Self::Union;
}

/// Implies that the struct's impl of Get is repeatable & can be called multiple times at a given idx
pub unsafe trait IsRepeatable {}