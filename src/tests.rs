use rand::Rng; 
use crate::{matrix::{matrix_gen, MathMatrix, MatrixOps, MatrixEvalOps}, vector::{vector_gen, MathVector, RepeatableVectorOps, VectorEvalOps, VectorOps}};
use std::{hint::black_box, time::*};

#[test]
fn vector_display() {
    let vec = MathVector::from([0; 0]);
    let str = format!("{}", vec);
    println!("0d vec: {}", str);
    assert_eq!("\n[]", str);
    let vec = MathVector::from([1]);
    let str = format!("{}", vec);
    println!("1d vec: {}", str);
    assert_eq!("\n[ 1 ]", str);
    let vec = MathVector::from([1, 2]);
    let str = format!("{}", vec);
    println!("2d vec: {}", str);
    assert_eq!("\n┌ 1 ┐\n└ 2 ┘", str);
    let vec = MathVector::from([1, 2, 3]);
    let str = format!("{}", vec);
    println!("3d vec: {}", str);
    assert_eq!("\n┌ 1 ┐\n│ 2 │\n└ 3 ┘", str);
    let vec = MathVector::from([11, 2, 333]);
    let str = format!("{}", vec);
    println!("padded 3d vec: {}", str);
    assert_eq!("\n┌ 11  ┐\n│ 2   │\n└ 333 ┘", str);
}

/// returns log of the error, higher is better
/// roughly speaking is the number of non-leading-zero digits correct
fn f64_accuracy(experimental: f64, real: f64) -> f64 {
    if real == 0.0 {
        if experimental == 0.0 {
            return f64::INFINITY
        } else {
            return -f64::INFINITY;
        }
    }
    -f64::log10(((experimental - real) / real).abs())
}

#[test]
fn f64_accuracy_check() {
    assert!(f64_accuracy(1.0, 1.0).is_infinite());
    assert!({let x = f64_accuracy(1.0, 0.88888888888888); (x > 0.0) & (x < 1.0)});
    assert!({let x = f64_accuracy(0.9, 0.88888888888888); (x > 1.0) & (x < 2.0)});
    assert!({let x = f64_accuracy(0.89, 0.88888888888888); (x > 2.0) & (x < 3.0)});
    assert!({let x = f64_accuracy(0.889, 0.88888888888888); (x > 3.0) & (x < 4.0)});
    assert!({let x = f64_accuracy(0.8889, 0.88888888888888); (x > 4.0) & (x < 5.0)});
    assert!(f64_accuracy(0.0, 0.0) == f64::INFINITY);
    assert!(f64_accuracy(1.0, 0.0) == -f64::INFINITY);
}

/// uses the dot product of 2 vectors to find the cosine of the angle between them (x10000 times)
/// meant to test stacking outputs
/// prints:
///     the time elapsed in nanoseconds
#[test]
#[ignore]
fn vec_angle_cos() {
    let mut rng = rand::rng();
    let mut time = Duration::new(0, 0);
    for _ in 0..10000 {
        let vec1: MathVector<f64, 10000> = black_box(vector_gen(|| rng.random()).eval());
        let vec2: MathVector<f64, 10000> = black_box(vector_gen(|| rng.random()).eval());
        let now = Instant::now();
        let ((vec1_sqr_mag, vec2_sqr_mag), dot_product): ((f64, f64), f64) = (vec1.copied_sqr_mag()).dot(vec2.copied_sqr_mag()).consume();
        let mag = dot_product/((vec1_sqr_mag * vec2_sqr_mag).sqrt());
        black_box(mag);
        let elapsed = now.elapsed();
        time += elapsed;
    }
    println!("Elapsed: {}", time.as_nanos());
}

/// component-wise multiplies 2 vectors and gets the sum and product of all the elements
/// and grabs 200 random values from the multiplication of the 2 vectors
/// tests the ability to grab arbitrary values from a repeatable vector
/// prints:
///     200 random values from the multiplication of 2 vectors
///     sum and product of all the elemements
#[test]
#[ignore]
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

/// checks Mat * Mat basic correctness, tests for ~12 digits of accuracy (uses `f64_accuracy`)
#[test]
fn mat_mat_mul_test() {
    let mat1 = MathMatrix::from([
        [0.242, 0.740, 0.959],
        [0.454, 0.501, 0.535],
        [0.442, 0.081, 0.973],
    ]).transpose().eval();
    let mat2 = MathMatrix::from([
        [0.242, 0.740, 0.959],
        [0.454, 0.501, 0.535],
        [0.442, 0.081, 0.973],
    ]).transpose().eval();
    let out_mat = mat1.mat_mul(mat2).eval();
    (&out_mat).copied().zip(MathMatrix::from([
        [0.818402, 0.627499, 1.561085],
        [0.573792, 0.630296, 1.223976],
        [0.573804, 0.446474, 1.413942],
    ]).transpose()).entry_map(|(v1, v2)| assert!(f64_accuracy(v1, v2) > 12.0)).consume();
    println!("{:#?}", <[[_; _]; _]>::from(out_mat));
}

/// preforms a multiplication between a matrix and a matrix
/// prints:
///     duration of the calculation in nanoseconds
#[test]
#[ignore]
fn mat_mat_mul_preformance_test() {
    let mut rng = rand::rng();
    let mat1: Box<MathMatrix<f64, 1000, 1000>> = black_box(matrix_gen(|| rng.random()).heap_eval());
    let mat2: Box<MathMatrix<f64, 1000, 1000>> = black_box(matrix_gen(|| rng.random()).heap_eval()); 
    let now = Instant::now();
    let out = (mat1).mat_mul(mat2).heap_eval();
    black_box(out);
    let elapsed = now.elapsed();
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
#[ignore]
fn vector_variation_test() {
    let mut rng = rand::rng();
    let mut normal_time = Duration::new(0, 0);
    let mut heap_time = Duration::new(0, 0);
    let mut dynamic_time = Duration::new(0, 0);
    let mut dyn_heap_time = Duration::new(0, 0);
    for _ in 0..1000 {
        let vec1 = black_box(vector_gen::<_, f64, 10000>(|| rng.random()).eval());
        let vec2 = black_box(vector_gen::<_, f64, 10000>(|| rng.random()).eval());
        let now = Instant::now();
        let res = (vec1.reuse() + vec2).eval();
        black_box(res);
        let elapsed = now.elapsed();
        normal_time += elapsed;

        let vec1 = black_box(vector_gen::<_, f64, 10000>(|| rng.random()).heap_eval());
        let vec2 = black_box(vector_gen::<_, f64, 10000>(|| rng.random()).heap_eval());
        let now = Instant::now();
        let res = (vec1.heap_reuse() + vec2).heap_eval();
        black_box(res);
        let elapsed = now.elapsed();
        heap_time += elapsed;

        let vec1 = black_box(vector_gen::<_, f64, 10000>(|| rng.random()).eval());
        let vec2 = black_box(vector_gen::<_, f64, 10000>(|| rng.random()).eval());
        let now = Instant::now();
        let res = (vec1.reuse() + vec2).make_dynamic().eval();
        black_box(res);
        let elapsed = now.elapsed();
        dynamic_time += elapsed;

        let vec1 = black_box(vector_gen::<_, f64, 10000>(|| rng.random()).heap_eval());
        let vec2 = black_box(vector_gen::<_, f64, 10000>(|| rng.random()).heap_eval());
        let now = Instant::now();
        let res = (vec1.heap_reuse() + vec2).make_dynamic().heap_eval();
        black_box(res);
        let elapsed = now.elapsed();
        dyn_heap_time += elapsed;
    }
    println!("Normal Time:    {}", normal_time.as_nanos());
    println!("Heap Time:      {}", heap_time.as_nanos());
    println!("Dynamic Time:   {}", dynamic_time.as_nanos());
    println!("Dyn Heap Time:  {}", dyn_heap_time.as_nanos());
}

///tests rref basic correctness, tests for ~12 digits of accuracy (uses `f64_accuracy`) (real values only given to 14)
#[test]
fn rref_test() {
    let mut mat = MathMatrix::from([
        [0.242, 0.740, 0.959, 0.774],
        [0.454, 0.501, 0.535, 0.969],
        [0.442, 0.081, 0.973, 0.506],
    ]).transpose().eval();
    mat.rref();
    (&mat).zip(MathMatrix::from([
        [1.0, 0.0, 0.0, 1.451144618141667],
        [0.0, 1.0, 0.0, 0.84263819341021],
        [0.0, 0.0, 1.0, -0.209311012214639],
    ]).transpose()).entry_map(|(v1, v2)| assert!(f64_accuracy(*v1, v2) > 12.0, "Rref Accuracy Fail")).consume();
    println!("rref: {:#?}", <[[_; _]; _]>::from(mat));
}

//tests det basic correctness, tests for ~12 digits of accuracy (uses `f64_accuracy`)
#[test]
fn det_test() {
    let mat = MathMatrix::from([
        [0.242, 0.740, 0.959],
        [0.454, 0.501, 0.535],
        [0.442, 0.081, 0.973],
    ]).transpose().eval();
    let det = mat.det();
    assert!(f64_accuracy(det, -0.221516496) > 12.0, "Det Accuracy Fail");
    println!("det: {}", det);
}

// tests the performance of rref & det
#[test]
#[ignore]
fn mat_math_performance_test() {
    let mut rng = rand::rng();
    let mut rref_mat = black_box(matrix_gen::<_, f64, 1000, 2000>(|| rng.random()).heap_eval());
    let det_mat = black_box(matrix_gen::<_, f64, 1500, 1500>(|| rng.random()).heap_eval());

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
    (&out_vec).copied().zip(MathVector::from([1.389622, 1.107575, 0.912935])).map(|(v1, v2)| assert!(f64_accuracy(v1, v2) > 12.0, "Mat * Vec Accuracy Fail")).consume();
    let print: [_; _] = out_vec.into();
    println!("mat * vec: \n{:#?}", print);
}

/// tests mat_vec_mul performance
#[test]
#[ignore]
fn mat_vec_mul_performance_test() {
    let mut rng = rand::rng();
    let mat = black_box(matrix_gen::<_, f64, 10000, 10000>(|| rng.random()).heap_eval());
    let vec = black_box(vector_gen::<_, f64, 10000>(|| rng.random()).eval());

    let now = Instant::now();
    black_box(mat.mat_vec_mul::<_, f64>(vec).eval());
    let elapsed = now.elapsed();

    println!("{}", elapsed.as_nanos());
}

// tests vec_mat_mul basic correctness
#[test]
fn vec_mat_mul_test() {
    let vec = MathVector::from([0.774, 0.969, 0.506]);
    let mat = MathMatrix::from([
        [0.242, 0.740, 0.959],
        [0.454, 0.501, 0.535],
        [0.442, 0.081, 0.973],
    ]).transpose().eval();
    let out_vec = vec.vec_mat_mul::<_, f64>(mat).eval();
    (&out_vec).copied().zip(MathVector::from([0.850886, 1.099215, 1.753019])).map(|(v1, v2)| assert!(f64_accuracy(v1, v2) > 12.0, "Mat * Vec Accuracy Fail")).consume();
    let print: [_; _] = out_vec.into();
    println!("vec * mat: \n{:?}", print);
}

/// tests vec_mat_mul performance
#[test]
#[ignore]
fn vec_mat_mul_performance_test() {
    let mut rng = rand::rng();
    let vec = black_box(vector_gen::<_, f64, 10000>(|| rng.random()).eval());
    let mat = black_box(matrix_gen::<_, f64, 10000, 10000>(|| rng.random()).heap_eval());

    let now = Instant::now();
    black_box(vec.vec_mat_mul::<_, f64>(mat).eval());
    let elapsed = now.elapsed();

    println!("{}", elapsed.as_nanos());
}