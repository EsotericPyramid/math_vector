use crate::trait_specialization_utils::*;
use crate::util_traits::*;
use crate::matrix::mat_util_traits::*;
use std::mem::ManuallyDrop;
use std::ops::*;

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


macro_rules! mat_struct {
    ( // Get2D (+ non-lazy)* -> Get2D
        $struct:ident<$($($lifetime:lifetime),+, )? {$mat_generic:ident} $(, $($generic:ident $(: $($generic_lifetime:lifetime |)? $fst_generic_bound:path $(| $generic_bound:path)*)?),+)?>{$mat:ident $(, $($field:ident: $field_ty:ty),+)?}
        $(where $($bound_ty:ty: $fst_where_bound:path $(| $where_bound:path)*),+)?;
        $(output: $outputted_field:ident: $output_ty:ty, )?
        get2D: $item:ty, |$self:ident, $(($is_mut:tt))? $input:ident| $get_expr:expr_2021 $(, $is_repeatable:ty)?
    ) => {
        pub struct $struct<$($($lifetime),+, )? $mat_generic: MatrixLike $(, $($generic $(: $($generic_lifetime +)? $fst_generic_bound $(+ $generic_bound)*)?),+)?> $(where $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {pub(crate) $mat: $mat_generic $(, $(pub(crate) $field: $field_ty),+)?}

        unsafe impl<$($($lifetime),+, )? $mat_generic: MatrixLike $(, $($generic $(: $fst_generic_bound $(+ $generic_bound)*)?),+)?> Get2D for $struct<$($($lifetime),+, )? $mat_generic $(, $($generic),+)?> $(where $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {
            type GetBool = is_unit!($item);
            type AreInputsTransposed = <$mat_generic as Get2D>::AreInputsTransposed;
            type Inputs = <$mat_generic as Get2D>::Inputs;
            type Item = $item;
            type BoundItems = <$mat_generic as Get2D>::BoundItems;

            #[inline]
            unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs { unsafe {self.$mat.get_inputs(col_index, row_index)}}

            #[inline]
            unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) { unsafe {self.$mat.drop_inputs(col_index, row_index)}}

            #[inline]
            fn process($self: &mut Self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
                let ($($is_mut)? $input, bound_items) = $self.$mat.process(inputs);
                ($get_expr, bound_items)
            }
        }

        impl<$($($lifetime),+, )? $mat_generic: MatrixLike $(, $($generic $(: $fst_generic_bound $(+ $generic_bound)*)?),+)?> HasOutput for $struct<$($($lifetime),+, )? $mat_generic $(, $($generic),+)?> 
        where ($mat_generic::OutputBool, is_present!($($outputted_field)?)): FilterPair $(, $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {
            type OutputBool = optimized_or!($mat_generic::OutputBool, $($outputted_field)?);
            type Output = <($mat_generic::OutputBool, is_present!($($outputted_field)?)) as FilterPair>::Filtered<$mat_generic::Output, optional_type!($($output_ty)?)>;

            #[inline]
            unsafe fn output(&mut self) -> Self::Output { unsafe {
                <($mat_generic::OutputBool, is_present!($($outputted_field)?)) as FilterPair>::filter(self.$mat.output(), optional_expr!($(self.$outputted_field.output())?))
            }}

            #[inline]
            unsafe fn drop_output(&mut self) { unsafe {
                self.$mat.drop_output();
                $(self.$outputted_field.output();)?
            }}
        }

        impl<$($($lifetime),+, )? $mat_generic: MatrixLike $(, $($generic $(: $fst_generic_bound $(+ $generic_bound)*)?),+)?> Has2DReuseBuf for $struct<$($($lifetime),+, )? $mat_generic $(, $($generic),+)?> 
        $(where $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {
            type FstHandleBool = <$mat_generic as Has2DReuseBuf>::FstHandleBool;
            type SndHandleBool = <$mat_generic as Has2DReuseBuf>::SndHandleBool;
            type BoundHandlesBool = <$mat_generic as Has2DReuseBuf>::BoundHandlesBool;
            type FstOwnedBufferBool = <$mat_generic as Has2DReuseBuf>::FstOwnedBufferBool;
            type SndOwnedBufferBool = <$mat_generic as Has2DReuseBuf>::SndOwnedBufferBool;
            type IsFstBufferTransposed = <$mat_generic as Has2DReuseBuf>::IsFstBufferTransposed;
            type IsSndBufferTransposed = <$mat_generic as Has2DReuseBuf>::IsSndBufferTransposed;
            type AreBoundBuffersTransposed = <$mat_generic as Has2DReuseBuf>::AreBoundBuffersTransposed;
            type FstOwnedBuffer = <$mat_generic as Has2DReuseBuf>::FstOwnedBuffer;
            type SndOwnedBuffer = <$mat_generic as Has2DReuseBuf>::SndOwnedBuffer;
            type FstType = <$mat_generic as Has2DReuseBuf>::FstType;
            type SndType = <$mat_generic as Has2DReuseBuf>::SndType;
            type BoundTypes = <$mat_generic as Has2DReuseBuf>::BoundTypes;

            #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) { unsafe {self.$mat.assign_1st_buf(col_index, row_index, val)}}
            #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) { unsafe {self.$mat.assign_2nd_buf(col_index, row_index, val)}}
            #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) { unsafe {self.$mat.assign_bound_bufs(col_index, row_index, val)}}
            #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {self.$mat.get_1st_buffer()}}
            #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {self.$mat.get_2nd_buffer()}}
            #[inline] unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.$mat.drop_1st_buf_index(col_index, row_index)}}
            #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.$mat.drop_2nd_buf_index(col_index, row_index)}}
            #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) { unsafe {self.$mat.drop_bound_bufs_index(col_index, row_index)}}
        }
    };
    ( // Get2D + Get2D (+ non-lazy)* -> Get2D
        $struct:ident<$($($lifetime:lifetime),+, )? {$l_mat_generic:ident, $r_mat_generic:ident} $(, $($generic:ident $(: $($generic_lifetime:lifetime |)? $fst_generic_bound:path $(| $generic_bound:path)*)?),+)?>{$l_mat:ident, $r_mat:ident $(, $($field:ident: $field_ty:ty),+)?}
        $(where $($bound_ty:ty: $fst_where_bound:path $(| $where_bound:path)*),+)?;
        $(output: $outputted_field:ident: $output_ty:ty, )?
        get2D: $item:ty, |$self:ident, $(($l_is_mut:tt))? $l_input:ident, $(($r_is_mut:tt))? $r_input:ident| $get_expr:expr_2021 $(, $is_repeatable:ty)?
    ) => {
        pub struct $struct<$($($lifetime),+, )? $l_mat_generic: MatrixLike, $r_mat_generic: MatrixLike $(, $($generic $(: $($generic_lifetime +)? $fst_generic_bound $(+ $generic_bound)*)?),+)?> $(where $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {pub(crate) $l_mat: $l_mat_generic, pub(crate) $r_mat: $r_mat_generic $(, $(pub(crate) $field: $field_ty),+)?}

        unsafe impl<$($($lifetime),+, )? $l_mat_generic: MatrixLike, $r_mat_generic: MatrixLike $(, $($generic $(: $($generic_lifetime +)? $fst_generic_bound $(+ $generic_bound)*)?),+)?> Get2D for $struct<$($($lifetime),+, )? $l_mat_generic, $r_mat_generic $(, $($generic),+)?> where ($l_mat_generic::BoundHandlesBool, $r_mat_generic::BoundHandlesBool): FilterPair, (<$l_mat_generic as Get2D>::AreInputsTransposed, <$r_mat_generic as Get2D>::AreInputsTransposed): TyBoolPair $(, $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {
            type GetBool = is_unit!($item);
            type AreInputsTransposed = <(<$l_mat_generic as Get2D>::AreInputsTransposed, <$r_mat_generic as Get2D>::AreInputsTransposed) as TyBoolPair>::And;
            type Inputs = ($l_mat_generic::Inputs, $r_mat_generic::Inputs);
            type Item = $item;
            type BoundItems = <($l_mat_generic::BoundHandlesBool, $r_mat_generic::BoundHandlesBool) as FilterPair>::Filtered<$l_mat_generic::BoundItems, $r_mat_generic::BoundItems>;

            #[inline]
            unsafe fn get_inputs(&mut self, col_index: usize, row_index: usize) -> Self::Inputs { unsafe {(self.$l_mat.get_inputs(col_index, row_index), self.$r_mat.get_inputs(col_index, row_index))}}

            #[inline]
            unsafe fn drop_inputs(&mut self, col_index: usize, row_index: usize) { unsafe {
                self.$l_mat.drop_inputs(col_index, row_index);
                self.$r_mat.drop_inputs(col_index, row_index);
            }}

            #[inline]
            fn process($self: &mut Self, inputs: Self::Inputs) -> (Self::Item, Self::BoundItems) {
                let ($($l_is_mut)? $l_input, l_bound_items) = $self.$l_mat.process(inputs.0);
                let ($($r_is_mut)? $r_input, r_bound_items) = $self.$r_mat.process(inputs.1);
                ($get_expr, <($l_mat_generic::BoundHandlesBool, $r_mat_generic::BoundHandlesBool) as FilterPair>::filter(l_bound_items, r_bound_items))
            }
        }

        if_present!({unsafe impl<$($($lifetime),+, )? $l_mat_generic: MatrixLike + Is2DRepeatable, $r_mat_generic: MatrixLike + Is2DRepeatable $(, $($generic $(: $($generic_lifetime +)? $fst_generic_bound $(+ $generic_bound)*)?),+)?> Is2DRepeatable for $struct<$($($lifetime),+, )? $l_mat_generic, $r_mat_generic $(, $($generic),+)?> $(where $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {}}, $($is_repeatable)?);
    
        impl<$($($lifetime),+, )? $l_mat_generic: MatrixLike, $r_mat_generic: MatrixLike $(, $($generic $(: $($generic_lifetime +)? $fst_generic_bound $(+ $generic_bound)*)?),+)?> HasOutput for $struct<$($($lifetime),+, )? $l_mat_generic, $r_mat_generic $(, $($generic),+)?> where ($l_mat_generic::OutputBool, $r_mat_generic::OutputBool): FilterPair, (<($l_mat_generic::OutputBool, $r_mat_generic::OutputBool) as TyBoolPair>::Or, is_present!($($outputted_field)?)): FilterPair $(, $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)? {
            type OutputBool = optimized_or!(<($l_mat_generic::OutputBool, $r_mat_generic::OutputBool) as TyBoolPair>::Or, $($outputted_field)?);
            type Output = <(<($l_mat_generic::OutputBool, $r_mat_generic::OutputBool) as TyBoolPair>::Or, is_present!($($outputted_field)?)) as FilterPair>::Filtered<<($l_mat_generic::OutputBool, $r_mat_generic::OutputBool) as FilterPair>::Filtered<$l_mat_generic::Output, $r_mat_generic::Output>, optional_type!($($output_ty)?)>;
        
            #[inline]
            unsafe fn output(&mut self) -> Self::Output { unsafe {
                <(<($l_mat_generic::OutputBool, $r_mat_generic::OutputBool) as TyBoolPair>::Or, is_present!($($outputted_field)?)) as FilterPair>::filter(
                    <($l_mat_generic::OutputBool, $r_mat_generic::OutputBool) as FilterPair>::filter(self.$l_mat.output(), self.$r_mat.output()),
                    optional_expr!($(self.$outputted_field.output())?)
                )
            }}

            #[inline]
            unsafe fn drop_output(&mut self) { unsafe {
                self.$l_mat.drop_output();
                self.$r_mat.drop_output();
                $(self.$outputted_field.output();)?
            }}
        }

        impl<$($($lifetime),+, )? $l_mat_generic: MatrixLike, $r_mat_generic: MatrixLike $(, $($generic $(: $($generic_lifetime +)? $fst_generic_bound $(+ $generic_bound)*)?),+)?> Has2DReuseBuf for $struct<$($($lifetime),+, )? $l_mat_generic, $r_mat_generic $(, $($generic),+)?> 
        where 
            (<$l_mat_generic as Has2DReuseBuf>::FstOwnedBufferBool, <$r_mat_generic as Has2DReuseBuf>::FstOwnedBufferBool): SelectPair,
            (<$l_mat_generic as Has2DReuseBuf>::SndOwnedBufferBool, <$r_mat_generic as Has2DReuseBuf>::SndOwnedBufferBool): SelectPair,
            (<$l_mat_generic as Has2DReuseBuf>::FstHandleBool, <$r_mat_generic as Has2DReuseBuf>::FstHandleBool): SelectPair,
            (<$l_mat_generic as Has2DReuseBuf>::SndHandleBool, <$r_mat_generic as Has2DReuseBuf>::SndHandleBool): SelectPair,
            (<$l_mat_generic as Has2DReuseBuf>::BoundHandlesBool, <$r_mat_generic as Has2DReuseBuf>::BoundHandlesBool): FilterPair,
            (<$l_mat_generic as Has2DReuseBuf>::IsFstBufferTransposed, <$r_mat_generic as Has2DReuseBuf>::IsFstBufferTransposed): TyBoolPair,
            (<$l_mat_generic as Has2DReuseBuf>::IsSndBufferTransposed, <$r_mat_generic as Has2DReuseBuf>::IsSndBufferTransposed): TyBoolPair,
            (<$l_mat_generic as Has2DReuseBuf>::AreBoundBuffersTransposed, <$r_mat_generic as Has2DReuseBuf>::AreBoundBuffersTransposed): TyBoolPair
            $(, $($bound_ty: $fst_where_bound $(+ $where_bound)*),+)?
        {
            type FstHandleBool = <(<$l_mat_generic as Has2DReuseBuf>::FstHandleBool, <$r_mat_generic as Has2DReuseBuf>::FstHandleBool) as TyBoolPair>::Xor;
            type SndHandleBool = <(<$l_mat_generic as Has2DReuseBuf>::SndHandleBool, <$r_mat_generic as Has2DReuseBuf>::SndHandleBool) as TyBoolPair>::Xor;
            type BoundHandlesBool = <(<$l_mat_generic as Has2DReuseBuf>::BoundHandlesBool, <$r_mat_generic as Has2DReuseBuf>::BoundHandlesBool) as TyBoolPair>::Or;
            type FstOwnedBufferBool = <(<$l_mat_generic as Has2DReuseBuf>::FstOwnedBufferBool, <$r_mat_generic as Has2DReuseBuf>::FstOwnedBufferBool) as TyBoolPair>::Xor; 
            type SndOwnedBufferBool = <(<$l_mat_generic as Has2DReuseBuf>::SndOwnedBufferBool, <$r_mat_generic as Has2DReuseBuf>::SndOwnedBufferBool) as TyBoolPair>::Xor; 
            type IsFstBufferTransposed = <(<$l_mat_generic as Has2DReuseBuf>::IsFstBufferTransposed, <$r_mat_generic as Has2DReuseBuf>::IsFstBufferTransposed) as TyBoolPair>::Xor;
            type IsSndBufferTransposed = <(<$l_mat_generic as Has2DReuseBuf>::IsSndBufferTransposed, <$r_mat_generic as Has2DReuseBuf>::IsSndBufferTransposed) as TyBoolPair>::Xor;
            type AreBoundBuffersTransposed = <(<$l_mat_generic as Has2DReuseBuf>::AreBoundBuffersTransposed, <$r_mat_generic as Has2DReuseBuf>::AreBoundBuffersTransposed) as TyBoolPair>::Xor;
            type FstOwnedBuffer = <(<$l_mat_generic as Has2DReuseBuf>::FstOwnedBufferBool, <$r_mat_generic as Has2DReuseBuf>::FstOwnedBufferBool) as SelectPair>::Selected<<$l_mat_generic as Has2DReuseBuf>::FstOwnedBuffer, <$r_mat_generic as Has2DReuseBuf>::FstOwnedBuffer>;
            type SndOwnedBuffer = <(<$l_mat_generic as Has2DReuseBuf>::SndOwnedBufferBool, <$r_mat_generic as Has2DReuseBuf>::SndOwnedBufferBool) as SelectPair>::Selected<<$l_mat_generic as Has2DReuseBuf>::SndOwnedBuffer, <$r_mat_generic as Has2DReuseBuf>::SndOwnedBuffer>;
            type FstType = <(<$l_mat_generic as Has2DReuseBuf>::FstHandleBool, <$r_mat_generic as Has2DReuseBuf>::FstHandleBool) as SelectPair>::Selected<<$l_mat_generic as Has2DReuseBuf>::FstType, <$r_mat_generic as Has2DReuseBuf>::FstType>;
            type SndType = <(<$l_mat_generic as Has2DReuseBuf>::SndHandleBool, <$r_mat_generic as Has2DReuseBuf>::SndHandleBool) as SelectPair>::Selected<<$l_mat_generic as Has2DReuseBuf>::SndType, <$r_mat_generic as Has2DReuseBuf>::SndType>;
            type BoundTypes = <(<$l_mat_generic as Has2DReuseBuf>::BoundHandlesBool, <$r_mat_generic as Has2DReuseBuf>::BoundHandlesBool) as FilterPair>::Filtered<<$l_mat_generic as Has2DReuseBuf>::BoundTypes, <$r_mat_generic as Has2DReuseBuf>::BoundTypes>;
        
            #[inline] unsafe fn assign_1st_buf(&mut self, col_index: usize, row_index: usize, val: Self::FstType) { unsafe {
                let (l_val, r_val) = <(<$l_mat_generic as Has2DReuseBuf>::FstHandleBool, <$r_mat_generic as Has2DReuseBuf>::FstHandleBool) as SelectPair>::deselect(val);
                self.$l_mat.assign_1st_buf(col_index, row_index, l_val);
                self.$r_mat.assign_1st_buf(col_index, row_index, r_val);
            }}
            #[inline] unsafe fn assign_2nd_buf(&mut self, col_index: usize, row_index: usize, val: Self::SndType) { unsafe {
                let (l_val, r_val) = <(<$l_mat_generic as Has2DReuseBuf>::SndHandleBool, <$r_mat_generic as Has2DReuseBuf>::SndHandleBool) as SelectPair>::deselect(val);
                self.$l_mat.assign_2nd_buf(col_index, row_index, l_val);
                self.$r_mat.assign_2nd_buf(col_index, row_index, r_val);
            }}
            #[inline] unsafe fn assign_bound_bufs(&mut self, col_index: usize, row_index: usize, val: Self::BoundTypes) { unsafe {
                let (l_val, r_val) = <(<$l_mat_generic as Has2DReuseBuf>::BoundHandlesBool, <$r_mat_generic as Has2DReuseBuf>::BoundHandlesBool) as FilterPair>::defilter(val);
                self.$l_mat.assign_bound_bufs(col_index, row_index, l_val);
                self.$r_mat.assign_bound_bufs(col_index, row_index, r_val);
            }}
            #[inline] unsafe fn get_1st_buffer(&mut self) -> Self::FstOwnedBuffer { unsafe {
                <(<$l_mat_generic as Has2DReuseBuf>::FstOwnedBufferBool, <$r_mat_generic as Has2DReuseBuf>::FstOwnedBufferBool) as SelectPair>::select(self.$l_mat.get_1st_buffer(), self.$r_mat.get_1st_buffer())
            }}
            #[inline] unsafe fn get_2nd_buffer(&mut self) -> Self::SndOwnedBuffer { unsafe {
                <(<$l_mat_generic as Has2DReuseBuf>::SndOwnedBufferBool, <$r_mat_generic as Has2DReuseBuf>::SndOwnedBufferBool) as SelectPair>::select(self.$l_mat.get_2nd_buffer(), self.$r_mat.get_2nd_buffer())
            }}
            #[inline] unsafe fn drop_1st_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {
                self.$l_mat.drop_1st_buf_index(col_index, row_index);
                self.$r_mat.drop_1st_buf_index(col_index, row_index);
            }}
            #[inline] unsafe fn drop_2nd_buf_index(&mut self, col_index: usize, row_index: usize) { unsafe {
                self.$l_mat.drop_2nd_buf_index(col_index, row_index);
                self.$r_mat.drop_2nd_buf_index(col_index, row_index);
            }}
            #[inline] unsafe fn drop_bound_bufs_index(&mut self, col_index: usize, row_index: usize) { unsafe {
                self.$l_mat.drop_bound_bufs_index(col_index, row_index);
                self.$r_mat.drop_bound_bufs_index(col_index, row_index);
            }}
        }
    }
}

mat_struct!(MatEntryMap<{M}, F: FnMut(M::Item) -> O, O>{mat, f: F}; get2D: O, |self, input| (self.f)(input));
mat_struct!(MatEntryFold<{M}, F: FnMut(O, M::Item) -> O, O>{mat, f: F, cell: Option<O>}; output: cell: O, get2D: (), |self, input| self.cell = Some((self.f)(self.cell.take().unwrap(), input)));
mat_struct!(MatEntryFoldRef<{M}, F: FnMut(&mut O, M::Item), O>{mat, f: F, cell: ManuallyDrop<O>}; output: cell: O, get2D: (), |self, input| (self.f)(&mut self.cell, input)); // note: use of this is preferred to MatEntryFold

mat_struct!(MatEntryCopiedFold<{M}, F: FnMut(O, M::Item) -> O, O>{mat, f: F, cell: Option<O>} where M::Item: Copy; output: cell: O, get2D: M::Item, |self, input| {self.cell = Some((self.f)(self.cell.take().unwrap(), input)); input});
mat_struct!(MatEntryCopiedFoldRef<{M}, F: FnMut(&mut O, M::Item), O>{mat, f: F, cell: ManuallyDrop<O>} where M::Item: Copy; output: cell: O, get2D: M::Item, |self, input| {(self.f)(&mut self.cell, input); input});

mat_struct!(MatCopy<'a, {M}, I: 'a | Copy>{mat} where M: Get2D<Item = &'a I>; get2D: I, |self, input| *input, Y);
mat_struct!(MatClone<'a, {M}, I: 'a | Clone>{mat} where M: Get2D<Item = &'a I>; get2D: I, |self, input| input.clone());

mat_struct!(MatNeg<{M}>{mat} where M::Item: Neg; get2D: <M::Item as Neg>::Output, |self, input| -input);

mat_struct!(MatMulR<{M}, S: Copy>{mat, scalar: S} where S: Mul<M::Item>; get2D: <S as Mul<M::Item>>::Output, |self, input| self.scalar * input);
mat_struct!(MatDivR<{M}, S: Copy>{mat, scalar: S} where S: Div<M::Item>; get2D: <S as Div<M::Item>>::Output, |self, input| self.scalar / input);
mat_struct!(MatRemR<{M}, S: Copy>{mat, scalar: S} where S: Rem<M::Item>; get2D: <S as Rem<M::Item>>::Output, |self, input| self.scalar % input);
mat_struct!(MatMulL<{M}, S: Copy>{mat, scalar: S} where M::Item: Mul<S>; get2D: <M::Item as Mul<S>>::Output, |self, input| input * self.scalar);
mat_struct!(MatDivL<{M}, S: Copy>{mat, scalar: S} where M::Item: Div<S>; get2D: <M::Item as Div<S>>::Output, |self, input| input / self.scalar);
mat_struct!(MatRemL<{M}, S: Copy>{mat, scalar: S} where M::Item: Rem<S>; get2D: <M::Item as Rem<S>>::Output, |self, input| input % self.scalar);

mat_struct!(MatDivAssign<'a, {M}, I: 'a | DivAssign<S>, S: Copy>{mat, scalar: S} where M: Get2D<Item = &'a mut I>; get2D: (), |self, input| *input /= self.scalar);
mat_struct!(MatMulAssign<'a, {M}, I: 'a | MulAssign<S>, S: Copy>{mat, scalar: S} where M: Get2D<Item = &'a mut I>; get2D: (), |self, input| *input *= self.scalar);
mat_struct!(MatRemAssign<'a, {M}, I: 'a | RemAssign<S>, S: Copy>{mat, scalar: S} where M: Get2D<Item = &'a mut I>; get2D: (), |self, input| *input %= self.scalar);

mat_struct!(MatEntrySum<{M}, S>{mat, scalar: ManuallyDrop<S>} where S: AddAssign<M::Item>; output: scalar: S, get2D: (), |self, input| *self.scalar += input);
mat_struct!(MatEntryProd<{M}, S>{mat, scalar: ManuallyDrop<S>} where S: MulAssign<M::Item>; output: scalar: S, get2D: (), |self, input| *self.scalar *= input);

mat_struct!(MatCopiedEntrySum<{M}, S>{mat, scalar: ManuallyDrop<S>} where M::Item: Copy, S: AddAssign<M::Item>; output: scalar: S, get2D: M::Item, |self, input| {*self.scalar += input; input});
mat_struct!(MatCopiedEntryProd<{M}, S>{mat, scalar: ManuallyDrop<S>} where M::Item: Copy, S: MulAssign<M::Item>; output: scalar: S, get2D: M::Item, |self, input| {*self.scalar *= input; input});


mat_struct!(MatZip<{M1, M2}>{l_mat, r_mat}; get2D: (M1::Item, M2::Item), |self, l_input, r_input| (l_input, r_input), Y);

mat_struct!(MatAdd<{M1, M2}>{l_mat, r_mat} where M1::Item: Add<M2::Item>; get2D: <M1::Item as Add<M2::Item>>::Output, |self, l_input, r_input| l_input + r_input);
mat_struct!(MatSub<{M1, M2}>{l_mat, r_mat} where M1::Item: Sub<M2::Item>; get2D: <M1::Item as Sub<M2::Item>>::Output, |self, l_input, r_input| l_input - r_input);
mat_struct!(MatCompMul<{M1, M2}>{l_mat, r_mat} where M1::Item: Mul<M2::Item>; get2D: <M1::Item as Mul<M2::Item>>::Output, |self, l_input, r_input| l_input * r_input);
mat_struct!(MatCompDiv<{M1, M2}>{l_mat, r_mat} where M1::Item: Div<M2::Item>; get2D: <M1::Item as Div<M2::Item>>::Output, |self, l_input, r_input| l_input / r_input);
mat_struct!(MatCompRem<{M1, M2}>{l_mat, r_mat} where M1::Item: Rem<M2::Item>; get2D: <M1::Item as Rem<M2::Item>>::Output, |self, l_input, r_input| l_input % r_input);

mat_struct!(MatAddAssign<'a, {M1, M2}, I: 'a | AddAssign<M2::Item>>{l_mat, r_mat} where M1: Get2D<Item = &'a mut I>; get2D: (), |self, l_input, r_input| *l_input += r_input);
mat_struct!(MatSubAssign<'a, {M1, M2}, I: 'a | SubAssign<M2::Item>>{l_mat, r_mat} where M1: Get2D<Item = &'a mut I>; get2D: (), |self, l_input, r_input| *l_input -= r_input);
mat_struct!(MatCompMulAssign<'a, {M1, M2}, I: 'a | MulAssign<M2::Item>>{l_mat, r_mat} where M1: Get2D<Item = &'a mut I>; get2D: (), |self, l_input, r_input| *l_input *= r_input);
mat_struct!(MatCompDivAssign<'a, {M1, M2}, I: 'a | DivAssign<M2::Item>>{l_mat, r_mat} where M1: Get2D<Item = &'a mut I>; get2D: (), |self, l_input, r_input| *l_input /= r_input);
mat_struct!(MatCompRemAssign<'a, {M1, M2}, I: 'a | RemAssign<M2::Item>>{l_mat, r_mat} where M1: Get2D<Item = &'a mut I>; get2D: (), |self, l_input, r_input| *l_input %= r_input);