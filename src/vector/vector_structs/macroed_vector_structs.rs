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

macro_rules! vec_struct {
    (
        $struct:ident<$($($lifetime:lifetime),+, )? {$vec_generic:ident} $(, $($generic:ident $(: $($generic_lifetime:lifetime |)? $fst_generic_bound:path $(| $generic_bound:path)*)?),+)?>{$vec:ident $(, $($field:ident: $field_ty:ty),+)?}
        $(where $($bound_ty:ty: $fst_where_bound:path $(| $where_bound:path)*),+)?;
        $(output: $outputted_field:ident: $output_ty:ty, )?
        get: $item:ty, |$self:ident, $(($is_mut:tt))? $input:ident| $get_expr:expr_2021 $(, $is_repeatable:ty)?
    ) => {
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
            fn process($self: &mut Self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
                let ($($is_mut)? $input, bound_items) = $self.$vec.process(inputs);
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
            #[inline] unsafe fn drop_1st_buf_index(&mut self, index: usize) { unsafe {self.$vec.drop_1st_buf_index(index)}}
            #[inline] unsafe fn drop_2nd_buf_index(&mut self, index: usize) { unsafe {self.$vec.drop_2nd_buf_index(index)}}
            #[inline] unsafe fn drop_bound_bufs_index(&mut self, index: usize) { unsafe {self.$vec.drop_bound_bufs_index(index)}}
        }
    };
    (
        $struct:ident<$($($lifetime:lifetime),+, )? {$l_vec_generic:ident, $r_vec_generic:ident} $(, $($generic:ident $(: $($generic_lifetime:lifetime |)? $fst_generic_bound:path $(| $generic_bound:path)*)?),+)?>{$l_vec:ident, $r_vec:ident $(, $($field:ident: $field_ty:ty),+)?}
        $(where $($bound_ty:ty: $fst_where_bound:path $(| $where_bound:path)*),+)?;
        $(output: $outputted_field:ident: $output_ty:ty, )?
        get: $item:ty, |$self:ident, $(($l_is_mut:tt))? $l_input:ident, $(($r_is_mut:tt))? $r_input:ident| $get_expr:expr_2021 $(, $is_repeatable:ty)?
    ) => {
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
            fn process($self: &mut Self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
                let ($($l_is_mut)? $l_input, l_bound_items) = $self.$l_vec.process(inputs.0);
                let ($($r_is_mut)? $r_input, r_bound_items) = $self.$r_vec.process(inputs.1);
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
    }
}

vec_struct!(VecMap<{T}, F: FnMut(T::Item) -> O, O>{vec, f: F}; get: O, |self, input| (self.f)(input));
vec_struct!(VecFold<{T}, F: FnMut(O, T::Item) -> O, O>{vec, f: F, cell: Option<O>}; output: cell: O, get: (), |self, input| self.cell = Some((self.f)(self.cell.take().unwrap(), input)));
vec_struct!(VecFoldRef<{T}, F: FnMut(&mut O, T::Item), O>{vec, f: F, cell: ManuallyDrop<O>}; output: cell: O, get: (), |self, input| (self.f)(&mut self.cell, input)); // note: use of this is preferred to VecFold

vec_struct!(VecCopiedFold<{T}, F: FnMut(O, T::Item) -> O, O>{vec, f: F, cell: Option<O>} where T::Item: Copy; output: cell: O, get: T::Item, |self, input| {self.cell = Some((self.f)(self.cell.take().unwrap(), input)); input});
vec_struct!(VecCopiedFoldRef<{T}, F: FnMut(&mut O, T::Item), O>{vec, f: F, cell: ManuallyDrop<O>} where T::Item: Copy; output: cell: O, get: T::Item, |self, input| {(self.f)(&mut self.cell, input); input}); // note: use of this is preferred to VecFold

vec_struct!(VecCopy<'a, {T}, I: 'a | Copy>{vec} where T: Get<Item = &'a I>; get: I, |self, input| *input, Y);
vec_struct!(VecClone<'a, {T}, I: 'a | Clone>{vec} where T: Get<Item = &'a I>; get: I, |self, input| input.clone());

vec_struct!(VecNeg<{T}>{vec} where T::Item: Neg; get: <T::Item as Neg>::Output, |self, input| -input);

vec_struct!(VecMulR<{T}, S: Copy>{vec, scalar: S} where S: Mul<T::Item>; get: <S as Mul<T::Item>>::Output, |self, input| self.scalar * input);
vec_struct!(VecDivR<{T}, S: Copy>{vec, scalar: S} where S: Div<T::Item>; get: <S as Div<T::Item>>::Output, |self, input| self.scalar / input);
vec_struct!(VecRemR<{T}, S: Copy>{vec, scalar: S} where S: Rem<T::Item>; get: <S as Rem<T::Item>>::Output, |self, input| self.scalar % input);
vec_struct!(VecMulL<{T}, S: Copy>{vec, scalar: S} where T::Item: Mul<S>; get: <T::Item as Mul<S>>::Output, |self, input| input * self.scalar);
vec_struct!(VecDivL<{T}, S: Copy>{vec, scalar: S} where T::Item: Div<S>; get: <T::Item as Div<S>>::Output, |self, input| input / self.scalar);
vec_struct!(VecRemL<{T}, S: Copy>{vec, scalar: S} where T::Item: Rem<S>; get: <T::Item as Rem<S>>::Output, |self, input| input % self.scalar);

vec_struct!(VecMulAssign<'a, {T}, I: 'a | MulAssign<S>, S: Copy>{vec, scalar: S} where T: Get<Item = &'a mut I>; get: (), |self, input| *input *= self.scalar);
vec_struct!(VecDivAssign<'a, {T}, I: 'a | DivAssign<S>, S: Copy>{vec, scalar: S} where T: Get<Item = &'a mut I>; get: (), |self, input| *input /= self.scalar);
vec_struct!(VecRemAssign<'a, {T}, I: 'a | RemAssign<S>, S: Copy>{vec, scalar: S} where T: Get<Item = &'a mut I>; get: (), |self, input| *input %= self.scalar);

vec_struct!(VecSum<{T}, S>{vec, scalar: ManuallyDrop<S>} where S: AddAssign<T::Item>; output: scalar: S, get: (), |self, input| *self.scalar += input);
vec_struct!(VecProduct<{T}, S>{vec, scalar: ManuallyDrop<S>} where S: MulAssign<T::Item>; output: scalar: S, get: (), |self, input| *self.scalar *= input);
vec_struct!(VecSqrMag<{T}, S>{vec, scalar: ManuallyDrop<S>} where T::Item: Copy | Mul, S: AddAssign<<T::Item as Mul>::Output>; output: scalar: S, get: (), |self, input| *self.scalar += input*input);

vec_struct!(VecCopiedSum<{T}, S>{vec, scalar: ManuallyDrop<S>} where T::Item: Copy, S: AddAssign<T::Item>; output: scalar: S, get: T::Item, |self, input| {*self.scalar += input; input});
vec_struct!(VecCopiedProduct<{T}, S>{vec, scalar: ManuallyDrop<S>} where T::Item: Copy, S: MulAssign<T::Item>; output: scalar: S, get: T::Item, |self, input| {*self.scalar *= input; input});
vec_struct!(VecCopiedSqrMag<{T}, S>{vec, scalar: ManuallyDrop<S>} where T::Item: Copy | Mul, S: AddAssign<<T::Item as Mul>::Output>; output: scalar: S, get: T::Item, |self, input| {*self.scalar += input*input; input});


vec_struct!(VecZip<{T1, T2}>{l_vec, r_vec}; get: (T1::Item, T2::Item), |self, l_input, r_input| (l_input, r_input), Y);

vec_struct!(VecAdd<{T1, T2}>{l_vec, r_vec} where T1::Item: Add<T2::Item>; get: <T1::Item as Add<T2::Item>>::Output, |self, l_input, r_input| l_input + r_input);
vec_struct!(VecSub<{T1, T2}>{l_vec, r_vec} where T1::Item: Sub<T2::Item>; get: <T1::Item as Sub<T2::Item>>::Output, |self, l_input, r_input| l_input - r_input);
vec_struct!(VecCompMul<{T1, T2}>{l_vec, r_vec} where T1::Item: Mul<T2::Item>; get: <T1::Item as Mul<T2::Item>>::Output, |self, l_input, r_input| l_input * r_input);
vec_struct!(VecCompDiv<{T1, T2}>{l_vec, r_vec} where T1::Item: Div<T2::Item>; get: <T1::Item as Div<T2::Item>>::Output, |self, l_input, r_input| l_input / r_input);
vec_struct!(VecCompRem<{T1, T2}>{l_vec, r_vec} where T1::Item: Rem<T2::Item>; get: <T1::Item as Rem<T2::Item>>::Output, |self, l_input, r_input| l_input % r_input);

vec_struct!(VecAddAssign<'a, {T1, T2}, I: 'a | AddAssign<T2::Item>>{l_vec, r_vec} where T1: Get<Item = &'a mut I>; get: (), |self, l_input, r_input| *l_input += r_input);
vec_struct!(VecSubAssign<'a, {T1, T2}, I: 'a | SubAssign<T2::Item>>{l_vec, r_vec} where T1: Get<Item = &'a mut I>; get: (), |self, l_input, r_input| *l_input -= r_input);
vec_struct!(VecCompMulAssign<'a, {T1, T2}, I: 'a | MulAssign<T2::Item>>{l_vec, r_vec} where T1: Get<Item = &'a mut I>; get: (), |self, l_input, r_input| *l_input *= r_input);
vec_struct!(VecCompDivAssign<'a, {T1, T2}, I: 'a | DivAssign<T2::Item>>{l_vec, r_vec} where T1: Get<Item = &'a mut I>; get: (), |self, l_input, r_input| *l_input /= r_input);
vec_struct!(VecCompRemAssign<'a, {T1, T2}, I: 'a | RemAssign<T2::Item>>{l_vec, r_vec} where T1: Get<Item = &'a mut I>; get: (), |self, l_input, r_input| *l_input %= r_input);

vec_struct!(VecDot<{T1, T2}, S>{l_vec, r_vec, scalar: ManuallyDrop<S>} where T1::Item: Mul<T2::Item>, S: AddAssign<<T1::Item as Mul<T2::Item>>::Output>; output: scalar: S, get: (), |self, l_input, r_input| *self.scalar += l_input * r_input);