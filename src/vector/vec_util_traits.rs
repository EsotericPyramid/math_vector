//! Module containing the traits which define a VectorLike type (primative lazy vector)

// Note: traits here aren't meant to be used directly by end users

use crate::{
    trait_specialization_utils::*,
    util_traits::HasOutput,
};

/// A way to get out items from a collection / generator which implicitly invalidates* that index
/// Can output owned values
/// 
/// Get has 2 parts: `get_inputs` and `process`
/// `get_inputs`: "gets the inputs" for process, is infallible and implicitly invalidates that index
/// `process`: "processes" inputs from `get_inputs` into the Item and BoundItems, can be fallible (can panic) but doesn't effect the validity of an index
/// 
/// thus, the flow to get the actual item (& BoundItems), you run `process(get_inputs(index))`, which is shortcutted w/ `get`
/// 
/// *:  if IsRepeatable = Y, indices aren't actually invalidated so it is legal to call `get_inputs` at an index twice or more
pub unsafe trait Get { 
    /// can you actually get something from this type (ie. is Item not a ZST?)
    type GetBool: TyBool;
    /// the inputs for `process` retrieved via `get_inputs`
    type Inputs;
    /// the main value returned by Get which may undergo further processing via additional wrappers
    type Item;
    /// Additional values which are "bound" to a specific place, generally within HasReuseBuf
    type BoundItems;

    /// "gets the inputs" for process, is infallible and implicitly invalidates that index
    /// 
    /// Safety:
    ///     index must be in bounds (not determinable via this trait)
    ///     called at most once at each index*
    ///     mutually exclusive with `drop_inputs` for each index
    /// 
    /// *:  if IsRepeatable = Y, indices aren't actually invalidated so it is legal to call `get_index` at an index twice or more
    unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs; 

    /// drops the memory that would be invalidated by `get_inputs` at the given index, is infallible
    /// 
    /// Safety:
    ///     index must be in bounds (not determinable via this trait)
    ///     called at most once at each index
    ///     mutually exclusive with `drop_inputs` for each index*
    /// 
    /// *:  if IsRepeatable = Y, indices aren't actually invalidated so it is legal `get_index` and `drop_inputs` at an index
    unsafe fn drop_inputs(&mut self, index: usize);

    /// processes the inputs retrieved from `get_inputs` into Item and BoundItems, is (potentially) fallible
    fn process(&mut self, index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems);

    /// A shortcut for calling `get_inputs` and `process` at an index
    /// Note: generally not used to better manage dropping, may be removed in the future
    #[inline]
    unsafe fn get(&mut self, index: usize) -> (Self::Item, Self::BoundItems) { unsafe {
        let inputs = self.get_inputs(index);
        self.process(index, inputs)
    }}
}

/// A way to designate buffers of memory to be written to
/// 
/// HasReuseBuf has unbound buffers and bound buffers:
/// unbound buffers (2 slots, refered as first (fst or 1st) and second (snd or 2nd)) are generally not used by external items
/// they are, instead, generally used by wrappers also implementing HasReuseBuf
/// In doing that, they are used "transparently" within the same method (ie. assign_1st_buf calls self.inner.assign_1st_buf)
/// Or, they are "bound" and are used within the bound buffer methods
/// 
/// bound buffers have been "bound" to a specific purpose and are generally used by external items
/// 
/// The general flow for using is:
/// - obtain some unbound buffer (created or attached)
/// - bind that buffer
/// - wrappers call the bound buffer methods to actually use the buffer
pub trait HasReuseBuf {
    /// is there an unbound buffer in the first slot
    type FstHandleBool: TyBool;
    /// is there an unbound buffer in the second slot
    type SndHandleBool: TyBool;
    /// are there any bound buffers
    type BoundHandlesBool: TyBool;
    /// is the first buffer owned by this type
    type FstOwnedBufferBool: TyBool;
    /// is the second buffer owned by this type
    type SndOwnedBufferBool: TyBool;
    /// the type of the buffer in the first slot (should be a ZST if no buffer or not owned)
    type FstOwnedBuffer;
    /// the type of the buffer in the second slot (should be a ZST if no buffer or not owned)
    type SndOwnedBuffer;
    /// the type written to the buffer in the first slot (should be a ZST if no buffer)
    type FstType;
    /// the type written to the buffer in the second slot (should be a ZST if no buffer)
    type SndType;
    /// the aggregate type of the various types written to bound buffers (should be a ZST if no buffer)
    type BoundTypes;

    /// write the val to the first buffer at index,
    /// safety: index in range (not determinable via this trait)
    /// note: `drop_1st_buf_index` should be called at indexes where this is called if the first buffer can't be outputted
    unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType); 
    /// write the val to the second buffer at index,
    /// safety: index in range (not determinable via this trait)
    /// note: `drop_2nd_buf_index` should be called at indexes where this is called if the second buffer can't be outputted
    unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType);
    /// write the val to the bound buffers at index,
    /// safety: index in range (not determinable via this trait)
    /// note: `drop_bound_bufs_index` should be called at indexs where this is called if the bound buffers can't be outputted (via HasOutput)
    unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes);
    /// get the first buffer
    /// safety: the first buffer has been filled (assign_1st_buf called at all indices in range)
    unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer;
    /// get the second buffer
    /// safety: the second buffer has been filled (assign_2nd_buf called at all indices in range)
    unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer;
    /// drop the first buffer
    /// safety: the first buffer has not been returned
    unsafe fn drop_1st_buffer(&mut self);
    /// drop the second buffer
    /// safety: the second buffer has been returned
    unsafe fn drop_2nd_buffer(&mut self);
    /// drops the assigned value at index in the first buffer
    /// safety: 
    /// index in range (not determinable via this trait)
    /// `assign_1st_buf` called at this index since last call of this at that index
    unsafe fn drop_1st_buf_index(&mut self, index: usize);
    /// drops the assigned value at index in the second buffer
    /// safety: 
    /// index in range (not determinable via this trait)
    /// `assign_2nd_buf` called at this index since last call of this at that index
    unsafe fn drop_2nd_buf_index(&mut self, index: usize);
    /// drops the assigned value at index in the bound buffers
    /// safety: 
    /// index in range (not determinable via this trait)
    /// `assign_bound_bufs` called at this index since last call of this at that index
    unsafe fn drop_bound_bufs_index(&mut self, index: usize);
}

/// a simple trait describing the full interface of a math_vector vector
/// still lacks sizing information (provided by wrappers, ie. `VectorExpr<V: VectorLike>`), and vector operations (see `VectorOps`)
/// 
/// really just a shorthand for the individual traits (Get, HasOutput, and HasReuseBuf)
/// automatically implemented for all types implementing all of the individual traits
pub trait VectorLike: Get + HasOutput + HasReuseBuf {}

impl<T: Get + HasOutput + HasReuseBuf> VectorLike for T {}

/// A way for a type to "build" wrappers around VectorLikes which encode sizing information
/// or in other words, implementors carry minimal sizing information which can be applied to VectorLikes
pub trait VectorBuilder: Clone {
    type Wrapped<T: VectorLike>;

    ///Safety: The VectorLike passed to this function MUST match the implications of the wrapper (ATM (Oct 2024), just needs to be unused)
    unsafe fn wrap<T: VectorLike>(&self, vec: T) -> Self::Wrapped<T>;
}

/// Enables an union operation between 2 VectorBuilders into a single VectorBuilder
pub trait VectorBuilderUnion<T: VectorBuilder>: VectorBuilder {
    /// the resulting type of the Union
    type Union: VectorBuilder;

    /// union 2 VectorBuilders into a single VectorBuilder
    /// additionally checks that the sizing information of each VectorBuilder is equal
    fn union(self, other: T) -> Self::Union;
}

/// Implies that the struct's impl of Get is repeatable and can be called multiple times at a given idx
pub unsafe trait IsRepeatable {}