use crate::trait_specialization_utils::*;
use crate::util_traits::*;
use crate::vector::vec_util_traits::*;
use std::ops::*;
use std::mem::ManuallyDrop;


macro_rules! is_unit {
    (()) => {
        N
    };
    ($ty:ty) => {
        Y
    };
}

macro_rules! is_present {
    ($tokens:tt) => {
        Y
    };
    () => {
        N
    }
}

macro_rules! if_present {
    ({$($tokens:tt)*}, $bool:tt) => {
        $($tokens)*
    };
    ({$($tokens:tt)*}, ) => {
        
    }
}

macro_rules! optional_type {
    () => {
        ()
    };
    ($ty:ty) => {
        $ty
    }
}

macro_rules! optional_expr {
    () => {
        ()
    };
    ($expr:expr_2021) => {
        $expr
    }
}

macro_rules! optimized_or {
    ($ty_bool:ty, $tokens:tt) => {
        Y
    };
    ($ty_bool:ty, ) => {
        $ty_bool
    }
}

macro_rules! vec_structs {
    (
        $(
            $comment:literal;
            $struct:ident<$($($lifetime:lifetime),+, )? {$vec_generic:ident} $(, $($generic:ident $(: $($generic_lifetime:lifetime |)? $fst_generic_bound:path $(| $generic_bound:path)*)?),+)?>{$vec:ident $(, $($field:ident: $field_ty:ty),+)?}
            $(where $($bound_ty:ty: $fst_where_bound:path $(| $where_bound:path)*),+)?;
            $(output: $outputted_field:ident: $output_ty:ty, )?
            get: $item:ty, |$self:ident, $(($is_mut:tt))? $input:ident| $get_expr:expr_2021 $(, $is_repeatable:ty)?;
        )*
    ) => {
        $(
            #[doc=$comment]
            pub struct $struct<$($($lifetime),+, )? $vec_generic: VectorLike $(, $($generic $(: $($generic_lifetime +)? $fst_generic_bound $(+ $generic_bound)*)?),+)?> $(where $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {pub(crate) $vec: $vec_generic $(, $(pub(crate) $field: $field_ty),+)?}
    
            unsafe impl<$($($lifetime),+, )? $vec_generic: VectorLike $(, $($generic $(: $fst_generic_bound $(+ $generic_bound)*)?),+)?> Get for $struct<$($($lifetime),+, )? $vec_generic $(, $($generic),+)?> $(where $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {
                type GetBool = is_unit!($item);
                type Inputs = <$vec_generic as Get>::Inputs;
                type Item = $item;
                type BoundItems = <$vec_generic as Get>::BoundItems;
    
                #[inline]
                unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {self.$vec.get_inputs(index)}}
    
                #[inline]
                unsafe fn drop_inputs(&mut self, index: usize) { unsafe {self.$vec.drop_inputs(index)}}
    
                #[inline]
                fn process($self: &mut Self, index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
                    let ($($is_mut)? $input, bound_items) = $self.$vec.process(index, inputs);
                    ($get_expr, bound_items)
                }
            }
    
            if_present!({unsafe impl<$($($lifetime),+, )? $vec_generic: IsRepeatable + VectorLike $(, $($generic $(: $fst_generic_bound $(+ $generic_bound)*)?),+)?> IsRepeatable for $struct<$($($lifetime),+, )? $vec_generic $(, $($generic),+)?> $(where $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {}}, $($is_repeatable)?);
     
            impl<$($($lifetime),+, )? $vec_generic: VectorLike $(, $($generic $(: $fst_generic_bound $(+ $generic_bound)*)?),+)?> HasOutput for $struct<$($($lifetime),+, )? $vec_generic $(, $($generic),+)?> 
            where ($vec_generic::OutputBool, is_present!($($outputted_field)?)): FilterPair $(, $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {
                type OutputBool = optimized_or!($vec_generic::OutputBool, $($outputted_field)?);
                type Output = <($vec_generic::OutputBool, is_present!($($outputted_field)?)) as FilterPair>::Filtered<$vec_generic::Output, optional_type!($($output_ty)?)>;
    
                #[inline]
                unsafe fn output(&mut self) -> Self::Output { unsafe {
                    <($vec_generic::OutputBool, is_present!($($outputted_field)?)) as FilterPair>::filter(self.$vec.output(), optional_expr!($(self.$outputted_field.output())?))
                }}
    
                #[inline]
                unsafe fn drop_output(&mut self) { unsafe {
                    self.$vec.drop_output();
                    $(self.$outputted_field.output();)?
                }}
            }
    
            impl<$($($lifetime),+, )? $vec_generic: VectorLike $(, $($generic $(: $fst_generic_bound $(+ $generic_bound)*)?),+)?> HasReuseBuf for $struct<$($($lifetime),+, )? $vec_generic $(, $($generic),+)?> 
            $(where $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {
                type FstHandleBool = <$vec_generic as HasReuseBuf>::FstHandleBool;
                type SndHandleBool = <$vec_generic as HasReuseBuf>::SndHandleBool;
                type BoundHandlesBool = <$vec_generic as HasReuseBuf>::BoundHandlesBool;
                type FstOwnedBufferBool = <$vec_generic as HasReuseBuf>::FstOwnedBufferBool;
                type SndOwnedBufferBool = <$vec_generic as HasReuseBuf>::SndOwnedBufferBool;
                type FstOwnedBuffer = <$vec_generic as HasReuseBuf>::FstOwnedBuffer;
                type SndOwnedBuffer = <$vec_generic as HasReuseBuf>::SndOwnedBuffer;
                type FstType = <$vec_generic as HasReuseBuf>::FstType;
                type SndType = <$vec_generic as HasReuseBuf>::SndType;
                type BoundTypes = <$vec_generic as HasReuseBuf>::BoundTypes;
    
                #[inline] unsafe fn assign_1st_buf(&mut self, index: usize, val: Self::FstType) { unsafe {self.$vec.assign_1st_buf(index, val)}}
                #[inline] unsafe fn assign_2nd_buf(&mut self, index: usize, val: Self::SndType) { unsafe {self.$vec.assign_2nd_buf(index, val)}}
                #[inline] unsafe fn assign_bound_bufs(&mut self, index: usize, val: Self::BoundTypes) { unsafe {self.$vec.assign_bound_bufs(index, val)}}
                #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {self.$vec.get_1st_buffer()}}
                #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.$vec.get_2nd_buffer()}}
                #[inline] unsafe fn drop_1st_buffer(&mut self) { unsafe {self.$vec.drop_1st_buffer()}}
                #[inline] unsafe fn drop_2nd_buffer(&mut self) { unsafe {self.$vec.drop_2nd_buffer()}}
                #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) { unsafe {self.$vec.drop_1st_buf_index(index)}}
                #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.$vec.drop_2nd_buf_index(index)}}
                #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {self.$vec.drop_bound_bufs_index(index)}}
            }
        )*
    };
    (
        $(
            $comment:literal;
            $struct:ident<$($($lifetime:lifetime),+, )? {$l_vec_generic:ident, $r_vec_generic:ident} $(, $($generic:ident $(: $($generic_lifetime:lifetime |)? $fst_generic_bound:path $(| $generic_bound:path)*)?),+)?>{$l_vec:ident, $r_vec:ident $(, $($field:ident: $field_ty:ty),+)?}
            $(where $($bound_ty:ty: $fst_where_bound:path $(| $where_bound:path)*),+)?;
            $(output: $outputted_field:ident: $output_ty:ty, )?
            get: $item:ty, |$self:ident, $(($l_is_mut:tt))? $l_input:ident, $(($r_is_mut:tt))? $r_input:ident| $get_expr:expr_2021 $(, $is_repeatable:ty)?;
        )*
    ) => {
        $(
            #[doc=$comment]
            pub struct $struct<$($($lifetime),+, )? $l_vec_generic: VectorLike, $r_vec_generic: VectorLike $(, $($generic $(: $($generic_lifetime +)? $fst_generic_bound $(+ $generic_bound)*)?),+)?> $(where $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {pub(crate) $l_vec: $l_vec_generic, pub(crate) $r_vec: $r_vec_generic $(, $(pub(crate) $field: $field_ty),+)?}
    
            unsafe impl<$($($lifetime),+, )? $l_vec_generic: VectorLike, $r_vec_generic: VectorLike $(, $($generic $(: $($generic_lifetime +)? $fst_generic_bound $(+ $generic_bound)*)?),+)?> Get for $struct<$($($lifetime),+, )? $l_vec_generic, $r_vec_generic $(, $($generic),+)?> where ($l_vec_generic::BoundHandlesBool, $r_vec_generic::BoundHandlesBool): FilterPair $(, $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {
                type GetBool = is_unit!($item);
                type Inputs = ($l_vec_generic::Inputs, $r_vec_generic::Inputs);
                type Item = $item;
                type BoundItems = <($l_vec_generic::BoundHandlesBool, $r_vec_generic::BoundHandlesBool) as FilterPair>::Filtered<$l_vec_generic::BoundItems, $r_vec_generic::BoundItems>;
    
                #[inline]
                unsafe fn get_inputs(&mut self, index: usize) -> Self::Inputs { unsafe {(self.$l_vec.get_inputs(index), self.$r_vec.get_inputs(index))}}
    
                #[inline]
                unsafe fn drop_inputs(&mut self, index: usize) { unsafe {
                    self.$l_vec.drop_inputs(index);
                    self.$r_vec.drop_inputs(index);
                }}
    
                #[inline]
                fn process($self: &mut Self, index: usize, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
                    let ($($l_is_mut)? $l_input, l_bound_items) = $self.$l_vec.process(index, inputs.0);
                    let ($($r_is_mut)? $r_input, r_bound_items) = $self.$r_vec.process(index, inputs.1);
                    ($get_expr, <($l_vec_generic::BoundHandlesBool, $r_vec_generic::BoundHandlesBool) as FilterPair>::filter(l_bound_items, r_bound_items))
                }
            }
        
            if_present!({unsafe impl<$($($lifetime),+, )? $l_vec_generic: VectorLike, $r_vec_generic: VectorLike $(, $($generic $(: $($generic_lifetime +)? $fst_generic_bound $(+ $generic_bound)*)?),+)?> IsRepeatable for $struct<$($($lifetime),+, )? $l_vec_generic, $r_vec_generic $(, $($generic),+)?> $(where $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {}}, $($is_repeatable)?);
    
            impl<$($($lifetime),+, )? $l_vec_generic: VectorLike, $r_vec_generic: VectorLike $(, $($generic $(: $($generic_lifetime +)? $fst_generic_bound $(+ $generic_bound)*)?),+)?> HasOutput for $struct<$($($lifetime),+, )? $l_vec_generic, $r_vec_generic $(, $($generic),+)?> where ($l_vec_generic::OutputBool, $r_vec_generic::OutputBool): FilterPair, (<($l_vec_generic::OutputBool, $r_vec_generic::OutputBool) as TyBoolPair>::Or, is_present!($($outputted_field)?)): FilterPair $(, $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {
                type OutputBool = optimized_or!(<($l_vec_generic::OutputBool, $r_vec_generic::OutputBool) as TyBoolPair>::Or, $($outputted_field)?);
                type Output = <(<($l_vec_generic::OutputBool, $r_vec_generic::OutputBool) as TyBoolPair>::Or, is_present!($($outputted_field)?)) as FilterPair>::Filtered<<($l_vec_generic::OutputBool, $r_vec_generic::OutputBool) as FilterPair>::Filtered<$l_vec_generic::Output, $r_vec_generic::Output>, optional_type!($($output_ty)?)>;
            
                #[inline]
                unsafe fn output(&mut self) -> Self::Output { unsafe {
                    <(<($l_vec_generic::OutputBool, $r_vec_generic::OutputBool) as TyBoolPair>::Or, is_present!($($outputted_field)?)) as FilterPair>::filter(
                        <($l_vec_generic::OutputBool, $r_vec_generic::OutputBool) as FilterPair>::filter(self.$l_vec.output(), self.$r_vec.output()),
                        optional_expr!($(self.$outputted_field.output())?)
                    )
                }}
    
                #[inline]
                unsafe fn drop_output(&mut self) { unsafe {
                    self.$l_vec.drop_output();
                    self.$r_vec.drop_output();
                    $(self.$outputted_field.output();)?
                }}
            }
    
            impl<$($($lifetime),+, )? $l_vec_generic: VectorLike, $r_vec_generic: VectorLike $(, $($generic $(: $($generic_lifetime +)? $fst_generic_bound $(+ $generic_bound)*)?),+)?> HasReuseBuf for $struct<$($($lifetime),+, )? $l_vec_generic, $r_vec_generic $(, $($generic),+)?> 
            where 
                (<$l_vec_generic as HasReuseBuf>::FstOwnedBufferBool, <$r_vec_generic as HasReuseBuf>::FstOwnedBufferBool): SelectPair,
                (<$l_vec_generic as HasReuseBuf>::SndOwnedBufferBool, <$r_vec_generic as HasReuseBuf>::SndOwnedBufferBool): SelectPair,
                (<$l_vec_generic as HasReuseBuf>::FstHandleBool, <$r_vec_generic as HasReuseBuf>::FstHandleBool): SelectPair,
                (<$l_vec_generic as HasReuseBuf>::SndHandleBool, <$r_vec_generic as HasReuseBuf>::SndHandleBool): SelectPair,
                (<$l_vec_generic as HasReuseBuf>::BoundHandlesBool, <$r_vec_generic as HasReuseBuf>::BoundHandlesBool): FilterPair
                $(, $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)?
            {
                type FstHandleBool = <(<$l_vec_generic as HasReuseBuf>::FstHandleBool, <$r_vec_generic as HasReuseBuf>::FstHandleBool) as TyBoolPair>::Xor;
                type SndHandleBool = <(<$l_vec_generic as HasReuseBuf>::SndHandleBool, <$r_vec_generic as HasReuseBuf>::SndHandleBool) as TyBoolPair>::Xor;
                type BoundHandlesBool = <(<$l_vec_generic as HasReuseBuf>::BoundHandlesBool, <$r_vec_generic as HasReuseBuf>::BoundHandlesBool) as TyBoolPair>::Or;
                type FstOwnedBufferBool = <(<$l_vec_generic as HasReuseBuf>::FstOwnedBufferBool, <$r_vec_generic as HasReuseBuf>::FstOwnedBufferBool) as TyBoolPair>::Xor; 
                type SndOwnedBufferBool = <(<$l_vec_generic as HasReuseBuf>::SndOwnedBufferBool, <$r_vec_generic as HasReuseBuf>::SndOwnedBufferBool) as TyBoolPair>::Xor; 
                type FstOwnedBuffer = <(<$l_vec_generic as HasReuseBuf>::FstOwnedBufferBool, <$r_vec_generic as HasReuseBuf>::FstOwnedBufferBool) as SelectPair>::Selected<<$l_vec_generic as HasReuseBuf>::FstOwnedBuffer, <$r_vec_generic as HasReuseBuf>::FstOwnedBuffer>;
                type SndOwnedBuffer = <(<$l_vec_generic as HasReuseBuf>::SndOwnedBufferBool, <$r_vec_generic as HasReuseBuf>::SndOwnedBufferBool) as SelectPair>::Selected<<$l_vec_generic as HasReuseBuf>::SndOwnedBuffer, <$r_vec_generic as HasReuseBuf>::SndOwnedBuffer>;
                type FstType = <(<$l_vec_generic as HasReuseBuf>::FstHandleBool, <$r_vec_generic as HasReuseBuf>::FstHandleBool) as SelectPair>::Selected<<$l_vec_generic as HasReuseBuf>::FstType, <$r_vec_generic as HasReuseBuf>::FstType>;
                type SndType = <(<$l_vec_generic as HasReuseBuf>::SndHandleBool, <$r_vec_generic as HasReuseBuf>::SndHandleBool) as SelectPair>::Selected<<$l_vec_generic as HasReuseBuf>::SndType, <$r_vec_generic as HasReuseBuf>::SndType>;
                type BoundTypes = <(<$l_vec_generic as HasReuseBuf>::BoundHandlesBool, <$r_vec_generic as HasReuseBuf>::BoundHandlesBool) as FilterPair>::Filtered<<$l_vec_generic as HasReuseBuf>::BoundTypes, <$r_vec_generic as HasReuseBuf>::BoundTypes>;
            
                #[inline] unsafe fn assign_1st_buf<'z>(&'z mut self, index: usize, val: Self::FstType) { unsafe {
                    let (l_val, r_val) = <(<$l_vec_generic as HasReuseBuf>::FstHandleBool, <$r_vec_generic as HasReuseBuf>::FstHandleBool) as SelectPair>::deselect(val);
                    self.$l_vec.assign_1st_buf(index, l_val);
                    self.$r_vec.assign_1st_buf(index, r_val);
                }}
                #[inline] unsafe fn assign_2nd_buf<'z>(&'z mut self, index: usize, val: Self::SndType) { unsafe {
                    let (l_val, r_val) = <(<$l_vec_generic as HasReuseBuf>::SndHandleBool, <$r_vec_generic as HasReuseBuf>::SndHandleBool) as SelectPair>::deselect(val);
                    self.$l_vec.assign_2nd_buf(index, l_val);
                    self.$r_vec.assign_2nd_buf(index, r_val);
                }}
                #[inline] unsafe fn assign_bound_bufs<'z>(&'z mut self, index: usize, val: Self::BoundTypes) { unsafe {
                    let (l_val, r_val) = <(<$l_vec_generic as HasReuseBuf>::BoundHandlesBool, <$r_vec_generic as HasReuseBuf>::BoundHandlesBool) as FilterPair>::defilter(val);
                    self.$l_vec.assign_bound_bufs(index, l_val);
                    self.$r_vec.assign_bound_bufs(index, r_val);
                }}
                #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {
                    <(<$l_vec_generic as HasReuseBuf>::FstOwnedBufferBool, <$r_vec_generic as HasReuseBuf>::FstOwnedBufferBool) as SelectPair>::select(self.$l_vec.get_1st_buffer(), self.$r_vec.get_1st_buffer())
                }}
                #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {
                    <(<$l_vec_generic as HasReuseBuf>::SndOwnedBufferBool, <$r_vec_generic as HasReuseBuf>::SndOwnedBufferBool) as SelectPair>::select(self.$l_vec.get_2nd_buffer(), self.$r_vec.get_2nd_buffer())
                }}
                #[inline] unsafe fn drop_1st_buffer(&mut self) { unsafe {
                    self.$l_vec.drop_1st_buffer();
                    self.$r_vec.drop_1st_buffer();
                }}
                #[inline] unsafe fn drop_2nd_buffer(&mut self) { unsafe {
                    self.$l_vec.drop_2nd_buffer();
                    self.$r_vec.drop_2nd_buffer();
                }}
                #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) { unsafe {
                    self.$l_vec.drop_1st_buf_index(index);
                    self.$r_vec.drop_1st_buf_index(index);
                }}
                #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {
                    self.$l_vec.drop_2nd_buf_index(index);
                    self.$r_vec.drop_2nd_buf_index(index);
                }}
                #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {
                    self.$l_vec.drop_bound_bufs_index(index);
                    self.$r_vec.drop_bound_bufs_index(index);
                }}
            }
        )*
    }
}

vec_structs!(
    "Struct mapping a vector's items using its closure (FnMut)";
    VecMap<{V}, F: FnMut(V::Item) -> O, O>{vec, f: F}; get: O, |self, input| (self.f)(input);
    "Struct folding a vector's items using its closure";
    VecFold<{V}, F: FnMut(O, V::Item) -> O, O>{vec, f: F, cell: Option<O>}; output: cell: O, get: (), |self, input| self.cell = Some((self.f)(self.cell.take().unwrap(), input));
    "Struct folding a vector's items using its closure";
    VecFoldRef<{V}, F: FnMut(&mut O, V::Item), O>{vec, f: F, cell: ManuallyDrop<O>}; output: cell: O, get: (), |self, input| (self.f)(&mut self.cell, input); // note: use of this is preferred to VecFold
    
    "Struct folding a vector's items using its closure while preserving the items";
    VecCopiedFold<{V}, F: FnMut(O, V::Item) -> O, O>{vec, f: F, cell: Option<O>} where V::Item: Copy; output: cell: O, get: V::Item, |self, input| {self.cell = Some((self.f)(self.cell.take().unwrap(), input)); input};
    "Struct folding a vector's items using its closure while preserving the items";
    VecCopiedFoldRef<{V}, F: FnMut(&mut O, V::Item), O>{vec, f: F, cell: ManuallyDrop<O>} where V::Item: Copy; output: cell: O, get: V::Item, |self, input| {(self.f)(&mut self.cell, input); input}; // note: use of this is preferred to VecFold
    
    "Struct copying a vector's items, useful for &T -> T";
    VecCopy<'a, {V}, I: 'a | Copy>{vec} where V: Get<Item = &'a I>; get: I, |self, input| *input, Y;
    "Struct cloning a vector's items, useful for &T -> T";
    VecClone<'a, {V}, I: 'a | Clone>{vec} where V: Get<Item = &'a I>; get: I, |self, input| input.clone();
    
    "Struct negating (-) a vector's items";
    VecNeg<{V}>{vec} where V::Item: Neg; get: <V::Item as Neg>::Output, |self, input| -input;
    
    "Struct multiplying a scalar by a vector (vector is rhs)";
    VecMulR<{V}, S: Copy>{vec, scalar: S} where S: Mul<V::Item>; get: <S as Mul<V::Item>>::Output, |self, input| self.scalar * input;
    "Struct dividing a scalar by a vector";
    VecDivR<{V}, S: Copy>{vec, scalar: S} where S: Div<V::Item>; get: <S as Div<V::Item>>::Output, |self, input| self.scalar / input;
    "Struct getting remainer (%) of a scalar by a vector";
    VecRemR<{V}, S: Copy>{vec, scalar: S} where S: Rem<V::Item>; get: <S as Rem<V::Item>>::Output, |self, input| self.scalar % input;
    "Struct multiplying a vector by a scalar (vector is lhs)";
    VecMulL<{V}, S: Copy>{vec, scalar: S} where V::Item: Mul<S>; get: <V::Item as Mul<S>>::Output, |self, input| input * self.scalar;
    "Struct dividing a vector by a scalar";
    VecDivL<{V}, S: Copy>{vec, scalar: S} where V::Item: Div<S>; get: <V::Item as Div<S>>::Output, |self, input| input / self.scalar;
    "Struct getting remainer (%) of a vector by a scalar";
    VecRemL<{V}, S: Copy>{vec, scalar: S} where V::Item: Rem<S>; get: <V::Item as Rem<S>>::Output, |self, input| input % self.scalar;
    
    "Struct mul assigning (*=) a vector's item (&mut T) by a scalar";
    VecMulAssign<'a, {V}, I: 'a | MulAssign<S>, S: Copy>{vec, scalar: S} where V: Get<Item = &'a mut I>; get: (), |self, input| *input *= self.scalar;
    "Struct div assigning (/=) a vector's item (&mut T) by a scalar";
    VecDivAssign<'a, {V}, I: 'a | DivAssign<S>, S: Copy>{vec, scalar: S} where V: Get<Item = &'a mut I>; get: (), |self, input| *input /= self.scalar;
    "Struct rem assigning (%=) a vector's item (&mut T) by a scalar";
    VecRemAssign<'a, {V}, I: 'a | RemAssign<S>, S: Copy>{vec, scalar: S} where V: Get<Item = &'a mut I>; get: (), |self, input| *input %= self.scalar;
    
    "Struct summing up a vector's items, adding it to Output";
    VecSum<{V}, S>{vec, scalar: ManuallyDrop<S>} where S: AddAssign<V::Item>; output: scalar: S, get: (), |self, input| *self.scalar += input;
    "Struct multiplying together a vector's items, adding it to Output";
    VecProduct<{V}, S>{vec, scalar: ManuallyDrop<S>} where S: MulAssign<V::Item>; output: scalar: S, get: (), |self, input| *self.scalar *= input;
    "Struct calculating the square magnitude of a vector, adding it to Output";
    VecSqrMag<{V}, S>{vec, scalar: ManuallyDrop<S>} where V::Item: Copy | Mul, S: AddAssign<<V::Item as Mul>::Output>; output: scalar: S, get: (), |self, input| *self.scalar += input*input;
    
    "Struct summing up a vector's items, adding it to Output while preserving the item";
    VecCopiedSum<{V}, S>{vec, scalar: ManuallyDrop<S>} where V::Item: Copy, S: AddAssign<V::Item>; output: scalar: S, get: V::Item, |self, input| {*self.scalar += input; input};
    "Struct multiplying together a vector's items, adding it to Output while preserving the item";
    VecCopiedProduct<{V}, S>{vec, scalar: ManuallyDrop<S>} where V::Item: Copy, S: MulAssign<V::Item>; output: scalar: S, get: V::Item, |self, input| {*self.scalar *= input; input};
    "Struct calculating the square magnitude of a vector, adding it to Output while preserving the item";
    VecCopiedSqrMag<{V}, S>{vec, scalar: ManuallyDrop<S>} where V::Item: Copy | Mul, S: AddAssign<<V::Item as Mul>::Output>; output: scalar: S, get: V::Item, |self, input| {*self.scalar += input*input; input};
);

vec_structs!(
    "Struct zipping together the items of 2 vectors into a 2 element tuple";
    VecZip<{V1, V2}>{l_vec, r_vec}; get: (V1::Item, V2::Item), |self, l_input, r_input| (l_input, r_input), Y;
    
    "Struct adding 2 vectors";
    VecAdd<{V1, V2}>{l_vec, r_vec} where V1::Item: Add<V2::Item>; get: <V1::Item as Add<V2::Item>>::Output, |self, l_input, r_input| l_input + r_input;
    "Struct subtracting a vector from another";
    VecSub<{V1, V2}>{l_vec, r_vec} where V1::Item: Sub<V2::Item>; get: <V1::Item as Sub<V2::Item>>::Output, |self, l_input, r_input| l_input - r_input;
    "Struct component-wise multiplying 2 vectors";
    VecCompMul<{V1, V2}>{l_vec, r_vec} where V1::Item: Mul<V2::Item>; get: <V1::Item as Mul<V2::Item>>::Output, |self, l_input, r_input| l_input * r_input;
    "Struct component-wise dividing a vector by another";
    VecCompDiv<{V1, V2}>{l_vec, r_vec} where V1::Item: Div<V2::Item>; get: <V1::Item as Div<V2::Item>>::Output, |self, l_input, r_input| l_input / r_input;
    "Struct component-wise getting remainder (%) of a vector by another";
    VecCompRem<{V1, V2}>{l_vec, r_vec} where V1::Item: Rem<V2::Item>; get: <V1::Item as Rem<V2::Item>>::Output, |self, l_input, r_input| l_input % r_input;
    
    "Struct add assigning (+=) a vector (of item &mut T) with another";
    VecAddAssign<'a, {V1, V2}, I: 'a | AddAssign<V2::Item>>{l_vec, r_vec} where V1: Get<Item = &'a mut I>; get: (), |self, l_input, r_input| *l_input += r_input;
    "Struct sub assigning (-=) a vector (of item &mut T) with another";
    VecSubAssign<'a, {V1, V2}, I: 'a | SubAssign<V2::Item>>{l_vec, r_vec} where V1: Get<Item = &'a mut I>; get: (), |self, l_input, r_input| *l_input -= r_input;
    "Struct component-wise mul assigning (*=) a vector (of item &mut T) with another";
    VecCompMulAssign<'a, {V1, V2}, I: 'a | MulAssign<V2::Item>>{l_vec, r_vec} where V1: Get<Item = &'a mut I>; get: (), |self, l_input, r_input| *l_input *= r_input;
    "Struct component-wise div assigning (/=) a vector (of item &mut T) with another";
    VecCompDivAssign<'a, {V1, V2}, I: 'a | DivAssign<V2::Item>>{l_vec, r_vec} where V1: Get<Item = &'a mut I>; get: (), |self, l_input, r_input| *l_input /= r_input;
    "Struct component-wise rem assigning (%=) a vector (of item &mut T) with another";
    VecCompRemAssign<'a, {V1, V2}, I: 'a | RemAssign<V2::Item>>{l_vec, r_vec} where V1: Get<Item = &'a mut I>; get: (), |self, l_input, r_input| *l_input %= r_input;
    
    "Struct calculating the dot product between 2 vectors, adding it to Output";
    VecDot<{V1, V2}, S>{l_vec, r_vec, scalar: ManuallyDrop<S>} where V1::Item: Mul<V2::Item>, S: AddAssign<<V1::Item as Mul<V2::Item>>::Output>; output: scalar: S, get: (), |self, l_input, r_input| *self.scalar += l_input * r_input;
    
    "Struct calculating the dot product between 2 vectors, adding it to Output, while preserving the items by zipping them in 2 element tuples";
    VecCopiedDot<{V1, V2}, S>{l_vec, r_vec, scalar: ManuallyDrop<S>} where V1::Item: Mul<V2::Item>, V1::Item: Copy, V2::Item: Copy, S: AddAssign<<V1::Item as Mul<V2::Item>>::Output>; output: scalar: S, get: (V1::Item, V2::Item), |self, l_input, r_input| {*self.scalar += l_input * r_input; (l_input, r_input)};
);