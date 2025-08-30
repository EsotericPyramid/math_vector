//! A Library for lazily evaluated (where possible) linear algebra

/// provides utilities for helping with trait specialization
/// 
/// - Y and N boolean types
/// - binary boolean operators via TyBoolPair
/// - Filter & Select to use boolean pairs to trim down types if unused
pub mod trait_specialization_utils {
    use std::mem::transmute_copy;

    /// a trait describing a single boolean type
    pub trait TyBool { 
        /// The inverse of Self (ie. N -> Y, Y -> N) 
        type Neg: TyBool; 

        /// Converts the type into a normal boolean
        fn as_bool() -> bool;
    }

    /// the provided "true" boolean type
    pub struct Y;

    /// the provided "false" boolean type
    pub struct N;

    impl TyBool for Y {type Neg = N; #[inline] fn as_bool() -> bool {false}}
    impl TyBool for N {type Neg = Y; #[inline] fn as_bool() -> bool {true}}

    /// a trait indicating if the boolean type is true
    pub trait IsTrue: TyBool {}

    impl IsTrue for Y {}

    /// A pair (ie. tuple w/ 2 elements) of boolean types
    /// can preform binary operators via the And, Or, and Xor Assoc types
    pub trait TyBoolPair {
        type And: TyBool; 
        type Or: TyBool;
        type Xor: TyBool;
    }

    impl TyBoolPair for (N, N) {type And = N; type Or = N; type Xor = N;}
    impl TyBoolPair for (N, Y) {type And = N; type Or = Y; type Xor = Y;}
    impl TyBoolPair for (Y, N) {type And = N; type Or = Y; type Xor = Y;}
    impl TyBoolPair for (Y, Y) {type And = Y; type Or = Y; type Xor = N;}

    /// a trait allowing a boolean type to filter a type
    /// 
    /// Y filter T -> T
    /// N filter T -> ()
    pub trait Filter: TyBool {
        /// the type T after filtering
        type Filtered<T>;

        /// filters the given value
        fn filter<T>(x: T) -> Self::Filtered<T>;
    }

    /// a trait allowing a boolean pair to filter a pair of types
    /// 
    /// Y, Y filter T1, T2 -> (T1, T2)
    /// Y, N filter T1, T2 -> T1
    /// N, Y filter T1, T2 -> T2
    /// N, N filter T1, T2 -> ()
    pub trait FilterPair: TyBoolPair {
        /// the returned type after filtering T1 and T2
        type Filtered<T1, T2>;

        /// filters the given T1 and T2 values
        fn filter<T1, T2>(x1: T1, x2: T2) -> Self::Filtered<T1, T2>;

        /// defilters the given filtered value
        /// ie. Y, N defilter T1 -> (T1, T2)
        /// 
        /// Safety: 
        /// requires that T1 and T2 are a ZST if the corresponding bool in the FilterPair is false
        /// ie. when doing Y, N defilter T1 -> (T1, T2), T2 must be a ZST (like ())
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

    /// A more specific version of filter where at most 1 of the inputs is outputted
    pub trait SelectPair: TyBoolPair { 
        /// the returned type after selecting T1 & T2
        type Selected<T1, T2>;

        /// selects the given T1 and T2 values
        fn select<T1, T2>(x1: T1, x2: T2) -> Self::Selected<T1, T2>;
        /// selects the given &'a T1 and &'a T2 values
        fn select_ref<'a, T1, T2>(x1: &'a T1, x2: &'a T2) -> &'a Self::Selected<T1, T2>;
        /// selects the given &'a mut T1 and &'a mut T2 values
        fn select_ref_mut<'a, T1, T2>(x1: &'a mut T1, x2: &'a mut T2) -> &'a mut Self::Selected<T1, T2>;

        /// deselectss the given selected value
        /// ie. Y, N deselect T1 -> (T1, T2)
        /// 
        /// Safety: 
        /// requires that T1 and T2 are a ZST if the corresponding bool in the SelctPair is false
        /// ie. when doing Y, N deselect T1 -> (T1, T2), T2 must be a ZST (like ())
        unsafe fn deselect<T1, T2>(filtered: Self::Selected<T1, T2>) -> (T1, T2);
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

/// provides utility traits for the library
pub mod util_traits {
    use crate::trait_specialization_utils::*;
    use std::mem::ManuallyDrop;

    /// a trait allowing a type to (possibly) "output" an owned value
    /// Note:
    /// a type implementing this should be assumed to be leaky
    /// to prevent leaks, you must call either output or drop_output
    pub trait HasOutput {
        /// does the type actually output a type (generally non-ZST)
        type OutputBool: TyBool;
        /// what type is outputted
        type Output;

        /// Makes the type output its value
        /// Safety: 
        /// can be called once (further calls may be safe depending on implementing type)
        /// mutually exclusive with `drop_output`
        unsafe fn output(&mut self) -> Self::Output;
        /// Make the type drop its output
        /// Safety:
        /// can be called once
        /// mutually exclusive with `output`
        /// must be called after drop methods from HasReuseBuf / Has2DReuseBuf
        unsafe fn drop_output(&mut self); 
    }

    /// implementation assumes that the value is always valid unless output is called
    impl<T> HasOutput for ManuallyDrop<T> {
        type OutputBool = Y;
        type Output = T;

        unsafe fn output(&mut self) -> Self::Output { unsafe {std::ptr::read(&**self)}}
        unsafe fn drop_output(&mut self) { unsafe {ManuallyDrop::drop(self)}}
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

    // NOTE: blanket impls bc outputting implies some level of ownership which 
    //      can only claimed once and so it shouldnt _really_ be done from 
    //      a reference, thus these impls just force the no output impl
    impl<'a, T: ?Sized> HasOutput for &'a T {
        type OutputBool = N;
        type Output = ();

        #[inline] unsafe fn output(&mut self) -> Self::Output {}
        #[inline] unsafe fn drop_output(&mut self) {}
    }

    impl<'a, T: ?Sized> HasOutput for &'a mut T {
        type OutputBool = N;
        type Output = ();

        #[inline] unsafe fn output(&mut self) -> Self::Output {}
        #[inline] unsafe fn drop_output(&mut self) {}
    }
}

/// provides utility structs for the Library
pub(crate) mod util_structs {
    use std::marker::PhantomData;

    /// an iterator which always return None
    /// used internally to abuse `Sum` and `Product` to get the additive & multiplicative identity values
    pub struct NoneIter<T>(PhantomData<T>); 

    impl<T> NoneIter<T> {
        #[inline] pub fn new() -> NoneIter<T> {NoneIter(PhantomData)} 
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
    use std::{hint::black_box, time::*};


    /// preforms a multiplication between a matrix and a vector
    /// prints:
    ///     a value from the resulting vector
    ///     duration of the calculation in nanoseconds
    /// 
    // TODO: update this use a proper Matrix type
    #[test]
    fn mat_vec_mul() {
        let mut rng = rand::rng();
        let vec: MathVector<f64, 10000> = vector_gen(|| rng.random()).eval();
        let mat: Box<MathVector<MathVector<f64, 10000>, 10000>> = vector_gen(|| vector_gen(|| rng.random()).eval()).heap_eval();
        let now = Instant::now();
        let out = mat.zip(vec).map(|(mat_vec, scalar)| (mat_vec.reuse() * scalar).eval()).sum::<MathVector<f64, 10000>>().consume();
        black_box(out);
        let elapsed = now.elapsed();
        
        println!("Elapsed: {}", elapsed.as_nanos());
    }

    /// uses the dot product of 2 vectors to find the cosine of the angle between them (x10000 times)
    /// meant to test stacking outputs
    /// prints:
    ///     the total of the cosines
    ///     the time elapsed in nanoseconds
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

    /// component-wise multiplies 2 vectors and gets the sum and product of all the elements
    /// and grabs 200 random values from the multiplication of the 2 vectors
    /// tests the ability to grab arbitrary values from a repeatable vector
    /// prints:
    ///     200 random values from the multiplication of 2 vectors
    ///     sum and product of all the elemements
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

    /// preforms a multiplication between a matrix and a matrix
    /// prints:
    ///     a value from the resulting matrix
    ///     duration of the calculation in nanoseconds
    #[test]
    fn mat_mat_mul_test() {
        let mut rng = rand::rng();
        let mat1: Box<MathMatrix<f64, 1000, 1000>> = matrix_gen(|| rng.random()).heap_eval();
        let mat2: Box<MathMatrix<f64, 1000, 1000>> = matrix_gen(|| rng.random()).heap_eval(); 
        let now = Instant::now();
        let out = (mat1).mat_mul(mat2).heap_eval();
        let elapsed = now.elapsed();
        println!("{}", out[0][0]);
        println!("Elapsed: {}", elapsed.as_nanos());
    }

    /// tests the preformance difference of a light calculation between different vector variants 
    /// variants tested:
    ///     Normal: `VectorExpr<_>`
    ///     Heaped: `Box<VectorExpr<_>>`
    ///     Dynamic: `VectorExpr<dyn VectorLike>`
    ///     Heaped Dynamic: `Box<VectorExpr<dyn VectorLike>>`
    /// prints preformance of each in nanoseconds
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

    ///tests rref basic correctness
    #[test]
    fn rref_test() {
        let mut mat = MathMatrix::from([
            [0.242, 0.740, 0.959, 0.774],
            [0.454, 0.501, 0.535, 0.969],
            [0.442, 0.081, 0.973, 0.506],
        ]).transpose().eval();
        mat.rref();
        println!("rref: {:#?}", mat.into_2d_array());
    }

    //tests det basic correctness
    #[test]
    fn det_test() {
        let mat = MathMatrix::from([
            [0.242, 0.740, 0.959],
            [0.454, 0.501, 0.535],
            [0.442, 0.081, 0.973],
        ]).transpose().eval();
        let det = mat.det();
        println!("det: {}", det);
    }

    // tests the performance of rref & det
    #[test]
    fn mat_math_performance_test() {
        let mut rng = rand::rng();
        let mut rref_mat = matrix_gen::<_, f64, 1000, 2000>(|| rng.random()).heap_eval();
        let det_mat = matrix_gen::<_, f64, 1500, 1500>(|| rng.random()).heap_eval();

        let now = Instant::now();
        rref_mat.rref();
        black_box(rref_mat);
        let rref_elapsed = now.elapsed();

        let now = Instant::now();
        black_box(det_mat.det_heap());
        let det_elapsed = now.elapsed();

        println!("rref: {}", rref_elapsed.as_nanos());
        println!("det: {}", det_elapsed.as_nanos());
    }

    // tests mat_vec_mul basic correctness
    #[test]
    fn mat_vec_mul_test() {
        let mat = MathMatrix::from([
            [0.242, 0.740, 0.959],
            [0.454, 0.501, 0.535],
            [0.442, 0.081, 0.973],
        ]).transpose().eval();
        let vec = MathVector::from([
            0.774,
            0.969,
            0.506,            
        ]);
        let out_vec = mat.mat_vec_mul::<_, f64>(vec).eval();
        let print: [_; _] = out_vec.into();
        println!("mat * vec: \n{:#?}", print);
    }

    /// tests mat_vec_mul performance
    #[test]
    fn mat_vec_mul_performance_test() {
        let mut rng = rand::rng();
        let mat = matrix_gen::<_, f64, 10000, 10000>(|| rng.random()).heap_eval();
        let vec = vector_gen::<_, f64, 10000>(|| rng.random()).eval();

        let now = Instant::now();
        black_box(mat.mat_vec_mul::<_, f64>(vec).eval());
        let elapsed = now.elapsed();

        println!("{}", elapsed.as_nanos());
    }
}