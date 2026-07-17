//! Module containing the traits which define the [`VectorLike`] trait (primative lazy vector)

// Note: traits here aren't meant to be used directly by end users

use crate::{trait_specialization_utils::*, util_traits::HasOutput};

/// A way to get out items from a collection / generator which implicitly invalidates* that index
/// Can output owned values
///
/// Get has 2 parts: `get_inputs` and `process`
/// `get_inputs`: "gets the inputs" for process, is infallible and implicitly invalidates that index
/// `process`: "processes" inputs from `get_inputs` into the Item and BoundItems, can be fallible (can panic) but doesn't effect the validity of an index
///
/// thus, the flow to get the actual item (and BoundItems), you run `process(get_inputs(index))`, which is shortcutted w/ `get`
/// 
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
    /// - index must be in bounds (not determinable via this trait)
    /// - called at most once at each index*
    /// - mutually exclusive with `drop_inputs` for each index
    ///
    /// *:  if IsRepeatable = Y, indices aren't actually invalidated so it is legal to call `get_index` and `drop_inputs` at an index twice or more
    unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs;

    /// drops the memory that would be invalidated by `get_inputs` at the given index, is infallible
    ///
    /// Safety:
    /// - index must be in bounds (not determinable via this trait)
    /// - called at most once at each index
    /// - mutually exclusive with `drop_inputs` for each index*
    ///
    /// *:  if IsRepeatable = Y, indices aren't actually invalidated so it is legal `get_index` and `drop_inputs` at an index
    unsafe fn drop_inputs(&mut self, index: usize);

    /// processes the inputs retrieved from `get_inputs` into Item and BoundItems, is (potentially) fallible
    fn process(&mut self, index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems);

    /// A shortcut for calling `get_inputs` and `process` at an index
    /// Note: generally not used to better manage dropping
    #[inline]
    unsafe fn get(&mut self, index: usize) -> (Self::Item, Self::BoundItems) {
        unsafe {
            let inputs = self.get_inputs(index);
            self.process(index, inputs)
        }
    }
}

/// A way to designate buffers of memory to be written to
///
/// HasReuseBuf has unbound buffers and bound buffers:
/// unbound buffers (2 slots, refered as first (fst or 1st) and second (snd or 2nd)) are generally not used by external items
/// they are, instead, generally used by wrappers also implementing HasReuseBuf
/// In doing that, they are used "transparently" within the same method (ie. `assign_1st_buf()` calls `self.inner.assign_1st_buf()`)
/// Or, they are "bound" and are used within the bound buffer methods and connected to output (generally [`HasOutput`])
///
/// bound buffers have been "bound" to a specific purpose and are generally used by external items
///
/// The general flow for using is:
/// - obtain some unbound buffer (create or attach it)
/// - bind that buffer
/// - wrappers call the bound buffer methods to actually use the buffer
pub trait HasReuseBuf {
    /// is there an unbound buffer in the first slot
    type FstHandleBool: TyBool;
    /// is there an unbound buffer in the second slot
    type SndHandleBool: TyBool;
    /// are there any bound buffers
    type BoundHandlesBool: TyBool;
    /// is the first buffer owned by this type (ie. is there a buffer to output in the first slot)
    type FstOwnedBufferBool: TyBool;
    /// is the second buffer owned by this type (ie. is there a buffer to output in the second slot)
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
    /// 
    /// safety: 
    /// - index in range (not determinable via this trait)
    /// - the first buffer hasn't been invalidated 
    /// 
    /// memory safety: 
    /// - called at most once per index (unless [`Self::drop_1st_buf_index`] is used in between)
    /// - call [`Self::drop_1st_buf_index`] at indexes where this is called if the first buffer can't be outputted
    unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType);
    /// write the val to the second buffer at index,
    /// 
    /// safety: 
    /// - index in range (not determinable via this trait)
    /// - the second buffer hasn't been invalidated
    /// 
    /// memory safety: 
    /// - called at most once per index (unless [`Self::drop_2nd_buf_index`] is used in between)
    /// - call [`Self::drop_2nd_buf_index`] at indexes where this is called if the second buffer can't be outputted
    /// note: 
    unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType);
    /// write the val to the bound buffers at index,
    /// 
    /// safety: 
    /// - index in range (not determinable via this trait)
    /// - the bound buffers haven't been invalidated (note: not invalidated through this trait) (FIXME: clarify when this can be invalidated)
    /// 
    /// memory safety: 
    /// - called at most once per index (unless [`Self::drop_bound_bufs_index`] is used in between)
    /// - call [`Self::drop_bound_bufs_index`] at indexs where this is called if the bound buffers can't be outputted (via HasOutput)
    unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes);
    /// get the first buffer
    /// 
    /// safety: 
    /// - the first buffer is still valid
    /// - the first buffer has been filled (assign_1st_buf called at all indices in range)
    /// - invalidates the first buffer
    unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer;
    /// get the second buffer
    /// 
    /// safety: 
    /// - the second buffer is still valid
    /// - the second buffer has been filled (assign_2nd_buf called at all indices in range)
    /// - invalidates the second buffer
    unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer;
    /// drop the first buffer
    /// 
    /// safety: 
    /// - the first buffer is still valid
    /// - invalidates the first buffer
    /// 
    /// memory safety:
    /// - the contents have been dropped with [`Self::drop_1st_buf_index`]
    unsafe fn drop_1st_buffer(&mut self);
    /// drop the second buffer
    /// 
    /// safety: 
    /// - the second buffer is still valid
    /// - invalidates the second buffer
    /// 
    /// memory safety:
    /// - the contents have been dropped with [`Self::drop_2nd_buf_index`]
    unsafe fn drop_2nd_buffer(&mut self);
    /// drops the assigned value at index in the first buffer
    /// 
    /// safety: 
    /// - the first buffer is still valid
    /// - index in range (not determinable via this trait)
    /// - index is valid (from [`Self::assign_1st_buf`])
    /// - invalidates the index
    unsafe fn drop_1st_buf_index(&mut self, index: usize);
    /// drops the assigned value at index in the second buffer
    /// 
    /// safety: 
    /// - the second buffer is still valid
    /// - index in range (not determinable via this trait)
    /// - index is valid (from [`Self::assign_2nd_buf`])
    /// - invalidates the index
    unsafe fn drop_2nd_buf_index(&mut self, index: usize);
    /// drops the assigned value at index in the bound buffers
    /// 
    /// safety: 
    /// - the bound buffers are still valid
    /// - index in range (not determinable via this trait)
    /// - index is valid (from [`Self::assign_bound_bufs`])
    /// - invalidates the index
    unsafe fn drop_bound_bufs_index(&mut self, index: usize);
}

/// a simple trait describing the full interface of a math_vector vector
/// still lacks sizing information (provided by wrappers, ie. `VectorExpr<V: VectorLike>`), and vector operations (see `VectorOps`)
///
/// really just a shorthand for the individual traits ([`Get`], [`HasOutput`], and [`HasReuseBuf`])
/// 
/// automatically implemented for all types implementing all of the individual traits
pub trait VectorLike: Get + HasOutput + HasReuseBuf {}

impl<T: Get + HasOutput + HasReuseBuf> VectorLike for T {}

/// Implies that the struct's impl of Get is repeatable and can be called multiple times at a given idx
///
/// Additional implications:
/// - no exposed part of Get will change behaviour if repeated 
/// - the calls to Get are trivial (ie. don't worry about costs calling multiple times vs. storing outputs)
/// - no *bound* reused bufs (ie. HasReuseBuf::BoundHandlesBool == N) (it is allowed to have them available)
/// 
/// note that you *still cannot call methods from other traits multiple times*, their semantics are unchanged
pub unsafe trait IsRepeatable {}
