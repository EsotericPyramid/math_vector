pub mod trait_specialization_utils {
    use std::mem::transmute_copy;

    pub trait TyBool {type Neg: TyBool; fn as_bool() -> bool;}
    pub struct Y;
    pub struct N;

    impl TyBool for Y {type Neg = N; #[inline] fn as_bool() -> bool {false}}
    impl TyBool for N {type Neg = Y; #[inline] fn as_bool() -> bool {true}}

    pub trait IsTrue {}

    impl IsTrue for Y {}

    pub trait TyBoolPair {
        type And: TyBool; 
        type Or: TyBool;
        type Xor: TyBool;
    }

    impl TyBoolPair for (N, N) {type And = N; type Or = N; type Xor = N;}
    impl TyBoolPair for (N, Y) {type And = N; type Or = Y; type Xor = Y;}
    impl TyBoolPair for (Y, N) {type And = N; type Or = Y; type Xor = Y;}
    impl TyBoolPair for (Y, Y) {type And = Y; type Or = Y; type Xor = N;}

    pub trait Filter {
        type Filtered<T>;

        fn filter<T>(x: T) -> Self::Filtered<T>;
    }

    pub trait FilterPair: TyBoolPair {
        type Filtered<T1, T2>;

        fn filter<T1, T2>(x1: T1, x2: T2) -> Self::Filtered<T1, T2>;
        ///Safety: requires that T1 & T2 are a ZST if the corresponding bool in the FilterPair is false
        unsafe fn defilter<T1, T2>(filtered: Self::Filtered<T1, T2>) -> (T1, T2);
    }

    impl Filter for N {type Filtered<T> = (); #[inline] fn filter<T>(_: T) -> Self::Filtered<T> {}}
    impl Filter for Y {type Filtered<T> = T; #[inline] fn filter<T>(x: T) -> Self::Filtered<T> {x}}
    
    impl FilterPair for (N, N) {
        type Filtered<T1, T2> = (); 
        
        #[inline] fn filter<T1, T2>(_: T1, _: T2) -> Self::Filtered<T1, T2> {}
        #[inline] unsafe fn defilter<T1, T2>(_: Self::Filtered<T1, T2>) -> (T1, T2) { unsafe {(transmute_copy(&()), transmute_copy(&()))}}
    }
    impl FilterPair for (N, Y) {
        type Filtered<T1, T2> = T2; 
        
        #[inline] fn filter<T1, T2>(_: T1, x: T2) -> Self::Filtered<T1, T2> {x}
        #[inline] unsafe fn defilter<T1, T2>(filtered: Self::Filtered<T1, T2>) -> (T1, T2) { unsafe {(transmute_copy(&()), filtered)}}
    }
    impl FilterPair for (Y, N) {
        type Filtered<T1, T2> = T1;
        
        #[inline] fn filter<T1, T2>(x: T1, _: T2) -> Self::Filtered<T1, T2> {x}
        #[inline] unsafe fn defilter<T1, T2>(filtered: Self::Filtered<T1, T2>) -> (T1, T2) { unsafe {(filtered, transmute_copy(&()))}}
    }
    impl FilterPair for (Y, Y) {
        type Filtered<T1, T2> = (T1, T2); 
        
        #[inline] fn filter<T1, T2>(x1: T1, x2: T2) -> Self::Filtered<T1, T2> {(x1, x2)}
        #[inline] unsafe fn defilter<T1, T2>(filtered: Self::Filtered<T1, T2>) -> (T1, T2) {filtered}
    }

    pub trait SelectPair: TyBoolPair { //A more specific version of filter where at most 1 of the inputs is outputted
        type Selected<T1, T2>;

        fn select<T1, T2>(x1: T1, x2: T2) -> Self::Selected<T1, T2>;
        unsafe fn deselect<T1, T2>(filtered: Self::Selected<T1, T2>) -> (T1, T2);
        fn select_ref<'a, T1, T2>(x1: &'a T1, x2: &'a T2) -> &'a Self::Selected<T1, T2>;
        fn select_ref_mut<'a, T1, T2>(x1: &'a mut T1, x2: &'a mut T2) -> &'a mut Self::Selected<T1, T2>;
    }

    impl SelectPair for (N, N) {
        type Selected<T1, T2> = ();

        #[inline] fn select<T1, T2>(_: T1, _: T2) -> Self::Selected<T1, T2> {}
        #[inline] unsafe fn deselect<T1, T2>(_: Self::Selected<T1, T2>) -> (T1, T2) { unsafe {(transmute_copy(&()), transmute_copy(&()))}}
        #[inline] fn select_ref<'a, T1, T2>(_: &'a T1, _: &'a T2) -> &'a Self::Selected<T1, T2> {Box::leak(Box::new(()))} //oh no, leaking a (), a type of size 0, whatever will we do...
        #[inline] fn select_ref_mut<'a, T1, T2>(_: &'a mut T1, _: &'a mut T2) -> &'a mut Self::Selected<T1, T2> {Box::leak(Box::new(()))} //although this does assume Rust will realize that the & is useless and elides it
    }

    impl SelectPair for (N, Y) {
        type Selected<T1, T2> = T2;

        #[inline] fn select<T1, T2>(_: T1, x: T2) -> Self::Selected<T1, T2> {x} 
        #[inline] unsafe fn deselect<T1, T2>(filtered: Self::Selected<T1, T2>) -> (T1, T2) { unsafe {(transmute_copy(&()), filtered)}}
        #[inline] fn select_ref<'a, T1, T2>(_: &'a T1, x: &'a T2) -> &'a Self::Selected<T1, T2> {x}
        #[inline] fn select_ref_mut<'a, T1, T2>(_: &'a mut T1, x: &'a mut T2) -> &'a mut Self::Selected<T1, T2> {x}
    }

    impl SelectPair for (Y, N) {
        type Selected<T1, T2> = T1;

        #[inline] fn select<T1, T2>(x: T1, _: T2) -> Self::Selected<T1, T2> {x} 
        #[inline] unsafe fn deselect<T1, T2>(filtered: Self::Selected<T1, T2>) -> (T1, T2) { unsafe {(filtered, transmute_copy(&()))}}
        #[inline] fn select_ref<'a, T1, T2>(x: &'a T1, _: &'a T2) -> &'a Self::Selected<T1, T2> {x}
        #[inline] fn select_ref_mut<'a, T1, T2>(x: &'a mut T1, _: &'a mut T2) -> &'a mut Self::Selected<T1, T2> {x}
    }
}

pub mod util_traits {
    use crate::trait_specialization_utils::*;

    pub trait HasOutput {
        type OutputBool: TyBool;
        type Output;

        unsafe fn output(&mut self) -> Self::Output;
        unsafe fn drop_output(&mut self); // &mut for std::ptr::drop_in_place which needs a *mut and not a *const
    }

    // implementation assumes that the value is always valid unless output is called
    impl<T> HasOutput for std::mem::ManuallyDrop<T> {
        type OutputBool = Y;
        type Output = T;

        unsafe fn output(&mut self) -> Self::Output { unsafe {std::ptr::read(&**self)}}
        unsafe fn drop_output(&mut self) { unsafe {std::mem::ManuallyDrop::drop(self)}}
    }

    impl<T> HasOutput for Option<T> {
        type OutputBool = Y;
        type Output = T;

        unsafe fn output(&mut self) -> Self::Output {self.take().unwrap()}
        unsafe fn drop_output(&mut self) {} // normal drop should be called & can handle Option fine
    }

    #[inline] fn debox<T: ?Sized>(boxed: &mut Box<T>) -> &mut T {&mut *boxed}

    impl<T: HasOutput + ?Sized> HasOutput for Box<T> {
        type OutputBool = T::OutputBool;
        type Output = T::Output;
    
        #[inline] unsafe fn output(&mut self) -> Self::Output { unsafe {(debox(self)).output()}}
        #[inline] unsafe fn drop_output(&mut self) { unsafe {(debox(self)).drop_output()}}
    }

    /// Implies that the struct's impl of Get &/| Get2D is repeatable & can be called multiple times at a given idx
    pub unsafe trait IsRepeatable {}
}

pub(crate) mod util_structs {
    pub struct NoneIter<T>(std::marker::PhantomData<T>); // used to abuse Sum and Product to get the additive & multiplicative identity values

    impl<T> NoneIter<T> {
        #[inline] pub fn new() -> NoneIter<T> {NoneIter(std::marker::PhantomData)} 
    }

    impl<T> Iterator for NoneIter<T> {
        type Item = T;

        #[inline] fn next(&mut self) -> Option<Self::Item> {None}
    }
}

pub mod vector;

pub mod matrix;

#[cfg(test)]
mod test {
    use rand::Rng; 
    use crate::{matrix::{matrix_gen, MathMatrix, MatrixOps}, vector::{vector_gen, MathVector, RepeatableVectorOps, VectorOps, }};
    use std::time::*;


    
    #[test]
    fn mat_vec_mul() {
        let mut rng = rand::rng();
        let vec: MathVector<f64, 10000> = vector_gen(|| rng.random()).eval();
        let mat: Box<MathVector<MathVector<f64, 10000>, 10000>> = vector_gen(|| vector_gen(|| rng.random()).eval()).heap_eval();
        let now = Instant::now();
        let out = mat.zip(vec).map(|(mat_vec, scalar)| (mat_vec.reuse() * scalar).eval()).sum::<MathVector<f64, 10000>>().consume();
        let elapsed = now.elapsed();
        println!("{}", out.into_array()[0]);
        println!("Elapsed: {}", elapsed.as_nanos());
    }

    #[test]
    fn mat_mat_mul() {
        let mut rng = rand::rng();
        let mat1: Box<MathVector<MathVector<f64, 1000>, 1000>> = vector_gen(|| vector_gen(|| rng.random()).eval()).heap_eval();
        let mat2: Box<MathVector<MathVector<f64, 1000>, 1000>> = vector_gen(|| vector_gen(|| rng.random()).eval()).heap_eval();
        let now = Instant::now();
        let out = mat2.map(|vec| (&mat1).zip(vec).map(|(mat_vec, scalar)| (mat_vec * scalar).eval()).sum::<MathVector<f64, 1000>>().consume()).heap_eval();
        let elapsed = now.elapsed();
        println!("{}", out.map(|vec| vec.into_array()).heap_eval()[0][0]);
        println!("Elapsed: {}", elapsed.as_nanos());
    }

    #[test]
    fn vec_angle_cos() {
        let mut rng = rand::rng();
        let mut total = 0.0;
        let mut time = Duration::new(0, 0);
        for _ in 0..10000 {
            let vec1: MathVector<f64, 10000> = vector_gen(|| rng.random()).eval();
            let vec2: MathVector<f64, 10000> = vector_gen(|| rng.random()).eval();
            let now = Instant::now();
            let ((vec1_sqr_mag, vec2_sqr_mag), dot_product): ((f64, f64), f64) = (vec1.copied_sqr_mag()).dot(vec2.copied_sqr_mag()).consume();
            let elapsed = now.elapsed();
            time += elapsed;
            let mag = dot_product/((vec1_sqr_mag * vec2_sqr_mag).sqrt());
            total += mag;
        }
        println!("{}", total);
        println!("Elapsed: {}", time.as_nanos());
    }

    #[test]
    fn repeatable_vectors_test() {
        // although IsRepeatable would likely mostly be only used internally, it has minimal external use
        let mut rng = rand::rng();
        let vec1: MathVector<f64, 10000> = vector_gen(|| rng.random()).eval();
        let vec2: MathVector<f64, 10000> = vector_gen(|| rng.random()).eval();
        let mut vec3 = (vec1.reuse().comp_mul(vec2)).copied_sum::<f64>().make_repeatable().copied();
        for _ in 0..200 { // enabled by IsRepeatable
            println!("{}", vec3.get(rng.random_range(0..10000)));
        }
        let (sum, product) = vec3.product::<f64>().consume();
        println!("sum: {}, product: {}", sum, product);
    }

    #[test]
    fn full_matrix_mul_test() {
        let mut rng = rand::rng();
        let mat1: Box<MathMatrix<f64, 1000, 1000>> = matrix_gen(|| rng.random()).heap_eval();
        let mat2: Box<MathMatrix<f64, 1000, 1000>> = matrix_gen(|| rng.random()).heap_eval(); 
        let now = Instant::now();
        let out = (mat1).mat_mul(mat2).heap_eval();
        let elapsed = now.elapsed();
        println!("{}", out[0][0]);
        println!("Elapsed: {}", elapsed.as_nanos());
    }

    #[test]
    fn vector_variation_test() {
        let mut rng = rand::rng();
        let mut total = 0.0;
        let mut normal_time = Duration::new(0, 0);
        let mut heap_time = Duration::new(0, 0);
        let mut dynamic_time = Duration::new(0, 0);
        let mut dyn_heap_time = Duration::new(0, 0);
        for _ in 0..1000 {
            let vec1 = vector_gen::<_, f64, 10000>(|| rng.random()).eval();
            let vec2 = vector_gen::<_, f64, 10000>(|| rng.random()).eval();
            let now = Instant::now();
            let res = (vec1.reuse() + vec2).eval();
            let elapsed = now.elapsed();
            normal_time += elapsed;
            total += res[0];

            let vec1 = vector_gen::<_, f64, 10000>(|| rng.random()).heap_eval();
            let vec2 = vector_gen::<_, f64, 10000>(|| rng.random()).heap_eval();
            let now = Instant::now();
            let res = (vec1.heap_reuse() + vec2).heap_eval();
            let elapsed = now.elapsed();
            heap_time += elapsed;
            total += res[0];

            let vec1 = vector_gen::<_, f64, 10000>(|| rng.random()).eval();
            let vec2 = vector_gen::<_, f64, 10000>(|| rng.random()).eval();
            let now = Instant::now();
            let res = (vec1.reuse() + vec2).make_dynamic().eval();
            let elapsed = now.elapsed();
            dynamic_time += elapsed;
            total += res[0];

            let vec1 = vector_gen::<_, f64, 10000>(|| rng.random()).heap_eval();
            let vec2 = vector_gen::<_, f64, 10000>(|| rng.random()).heap_eval();
            let now = Instant::now();
            let res = (vec1.heap_reuse() + vec2).make_dynamic().heap_eval();
            let elapsed = now.elapsed();
            dyn_heap_time += elapsed;
            total += res[0];
        }
        println!("{}", total);
        println!("Normal Time:    {}", normal_time.as_nanos());
        println!("Heap Time:      {}", heap_time.as_nanos());
        println!("Dynamic Time:   {}", dynamic_time.as_nanos());
        println!("Dyn Heap Time:  {}", dyn_heap_time.as_nanos());
    }
}