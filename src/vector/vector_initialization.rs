use std::mem::{transmute, ManuallyDrop, MaybeUninit};

use super::{
    vector_structs::{VectorArray, VectorSlice},
    vec_util_traits::Get,
    VectorOps, 
    MathVector, 
    RSMathVector, 
    VectorExpr,
    RSVectorExpr,
};


pub trait InitializableVectorExpr: VectorOps {
    fn new_filled(builder: Self::Builder, val: <Self::Unwrapped as Get>::Item) -> Self where <Self::Unwrapped as Get>::Item: Copy;
    fn new_zeroed(builder: Self::Builder) -> Self where <Self::Unwrapped as Get>::Item: num_traits::Zero + Copy { //not sure on the Copy requirement
        Self::new_filled(
            builder,
            <<Self::Unwrapped as Get>::Item as num_traits::Zero>::zero()
        )
    } 
    fn new_oned(builder: Self::Builder) -> Self where <Self::Unwrapped as Get>::Item: num_traits::One + Copy { // TBD name //not sure on the Copy requirement
        Self::new_filled(
            builder,
            <<Self::Unwrapped as Get>::Item as num_traits::One>::one()
        )
    } 
}

/// as always: this being 'unsafe' is tbd
pub unsafe trait UninitVectorExpr: InitializableVectorExpr {
    type Uninitialized: VectorOps;

    fn new_uninit(builder: Self::Builder) -> Self::Uninitialized;
    unsafe fn assume_init(uninit: Self::Uninitialized) -> Self;
    unsafe fn init_index(uninit: &mut Self::Uninitialized, index: usize, val: <Self::Unwrapped as Get>::Item);
    unsafe fn drop_index(uninit: &mut Self::Uninitialized, index: usize);
    unsafe fn drop_ots(uninit: &mut Self::Uninitialized);
}

impl<T, const D: usize> InitializableVectorExpr for MathVector<T, D> {
    fn new_filled(_: Self::Builder, val: <Self::Unwrapped as Get>::Item) -> Self where <Self::Unwrapped as Get>::Item: Copy {
        VectorExpr(VectorArray(ManuallyDrop::new([val; D])))
    }
}

unsafe impl<T, const D: usize> UninitVectorExpr for MathVector<T, D> {
    type Uninitialized = MathVector<MaybeUninit<T>, D>;

    fn new_uninit(_: Self::Builder) -> Self::Uninitialized {
        // safe bc the assume_init just moves the MaybeUninit inwards
        let inner: [MaybeUninit<T>; D] = unsafe { MaybeUninit::uninit().assume_init() };
        VectorExpr(VectorArray(ManuallyDrop::new(inner)))
    }

    unsafe fn assume_init(uninit: Self::Uninitialized) -> Self {
        unsafe {
            // FIXME: the copy isn't actually needed, just used to skirt around overly conservative size complaints
            std::mem::transmute_copy::<
                MathVector<MaybeUninit<T>, D>,
                MathVector<T, D>,
            >(&*ManuallyDrop::new(uninit))    
        }
    }

    unsafe fn init_index(uninit: &mut Self::Uninitialized, index: usize, val: <Self::Unwrapped as Get>::Item) {
        uninit.0.0[index] = MaybeUninit::new(val);
    }

    unsafe fn drop_index(uninit: &mut Self::Uninitialized, index: usize) {
        unsafe {
            MaybeUninit::assume_init_drop(uninit.0.0.get_unchecked_mut(index));
        }
    }

    // no actual one-time drops for this
    unsafe fn drop_ots(_: &mut Self::Uninitialized) {}
}

impl<T> InitializableVectorExpr for RSMathVector<T> {
    fn new_filled(builder: Self::Builder, val: <Self::Unwrapped as Get>::Item) -> Self where <Self::Unwrapped as Get>::Item: Copy {
        RSVectorExpr{
            vec: VectorSlice(
                unsafe {transmute::<Box<[T]>, Box<ManuallyDrop<[T]>>>(vec![val; builder.size].into_boxed_slice())}
            ),
            size: builder.size,
        }
    }
}

unsafe impl<T> UninitVectorExpr for RSMathVector<T> {
    type Uninitialized = RSMathVector<MaybeUninit<T>>;

    fn new_uninit(builder: Self::Builder) -> Self::Uninitialized {
        RSVectorExpr{
            vec: VectorSlice(
                unsafe {
                    transmute::<
                        Box<[MaybeUninit<T>]>, 
                        Box<ManuallyDrop<[MaybeUninit<T>]>>,
                    >(Box::new_uninit_slice(builder.size))
                }
            ),
            size: builder.size,
        }
    }

    unsafe fn assume_init(uninit: Self::Uninitialized) -> Self {
        unsafe {
            transmute::<
                RSMathVector<MaybeUninit<T>>,
                RSMathVector<T>,
            >(uninit)
        }
    }

    unsafe fn init_index(uninit: &mut Self::Uninitialized, index: usize, val: <Self::Unwrapped as Get>::Item) {
        uninit.vec.0[index] = MaybeUninit::new(val);
    }

    unsafe fn drop_index(uninit: &mut Self::Uninitialized, index: usize) {
        unsafe {
            MaybeUninit::assume_init_drop(uninit.vec.0.get_unchecked_mut(index));
        }
    }

    // no actual one-time drops for this
    unsafe fn drop_ots(_: &mut Self::Uninitialized) {}
}