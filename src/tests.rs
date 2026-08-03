//! A collection of tests (who woulda guessed)
//! 
//! contains part basic tests for correctness as well as benchmarks (which are `#[ignored]` by default)

use crate::prelude::*;
use crate::vector::vector_builders::{HeapedVectorExprBuilder, InitializableVectorBuilder};
use crate::{
    vector::VectorInPlaceEvalOps,
    matrix::matrix_exprs::ConcreteMatrixExpr,
};
use rand::Rng;
use std::{hint::black_box, time::*};

/// A quantity of 8 byte types (ex u64) which is known to not fit on the stack
/// 
/// The exact correct quantity likely depends the targetted architecture (and possibly compiler options (haven't checked)).
/// This number works at least for my system (2023 Macbook Pro; M2 Max, 32 GB)
const UNSTACKABLE_SIZE: usize = 1000000;

/// Checks that vectors display as expected:
/// 
/// ie. VectorExpr::from([1, 2, 3]) becomes
/// ```text
/// ┌ 1 ┐
/// │ 2 │
/// └ 3 ┘
/// ```
/// 
/// (prints results to stdout (use `-- --no-capture` to see))
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

/// Checks that matrices display as expected:
/// 
/// something like this is expected:
/// ```text
/// ┌ 1, 2, 3 ┐
/// │ 4, 5, 6 │
/// └ 7, 8, 9 ┘
/// ```
/// 
/// (prints results to stdout (use `-- --no-capture` to see))
#[test]
fn matrix_display() {
    let mat = MathMatrix::from([[0; 0]; 0]).transpose().eval();
    let string = mat.to_string();
    assert_eq!(string, "\n[]");
    println!("0x0 mat: {}", string);
    let mat = MathMatrix::from([[1, 2, 3]]).transpose().eval();
    let string = mat.to_string();
    assert_eq!(string, "\n[ 1, 2, 3 ]");
    println!("1x3 mat: {}", string);
    let mat = MathMatrix::from([[1, 2, 3], [4, 5, 6]]).transpose().eval();
    let string = mat.to_string();
    assert_eq!(string, "\n┌ 1, 2, 3 ┐\n└ 4, 5, 6 ┘");
    println!("2x3 mat: {}", string);
    let mat = MathMatrix::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        .transpose()
        .eval();
    let string = mat.to_string();
    println!("3x3 mat: {}", string);
    assert_eq!(string, "\n┌ 1, 2, 3 ┐\n│ 4, 5, 6 │\n└ 7, 8, 9 ┘");
    let mat = MathMatrix::from([[111, 2, 3], [4, 5, 66666], [77, 8, 9]])
        .transpose()
        .eval();
    let string = mat.to_string();
    assert_eq!(
        string,
        "\n┌ 111, 2, 3     ┐\n│ 4  , 5, 66666 │\n└ 77 , 8, 9     ┘"
    );
    println!("padded 3x3 mat: {}", string);
}

/// returns the negative log of the error, higher is better
/// 
/// roughly speaking is the number of non-leading-zero digits correct
/// 
/// "edge" cases: (these are actually required for this to be as continuous as possible)
/// - if `real = 0.0` and `experimental = 0.0`, then this returns +inf
/// - if `real = 0.0` and `experimental != 0.0`, then this return -inf
/// - if `real == experimental`, then this return +inf
/// 
/// ex: 
/// - `f64_accuracy(1.0,  0.88888888888)` is between 0 and 1
/// - `f64_accuracy(0.9,  0.88888888888)` is between 1 and 2
/// - `f64_accuracy(0.89, 0.88888888888)` is between 2 and 3
/// - etc.
fn f64_accuracy(experimental: f64, real: f64) -> f64 {
    if real == 0.0 {
        if experimental == 0.0 {
            return f64::INFINITY;
        } else {
            return -f64::INFINITY;
        }
    }
    -f64::log10(((experimental - real) / real).abs())
}

/// checks if [`f64_accuracy`] behaves as expected (see its def for details)
#[test]
fn f64_accuracy_check() {
    assert!(f64_accuracy(1.0, 1.0).is_infinite());
    assert!({
        let x = f64_accuracy(1.0, 0.88888888888888);
        (x > 0.0) & (x < 1.0)
    });
    assert!({
        let x = f64_accuracy(0.9, 0.88888888888888);
        (x > 1.0) & (x < 2.0)
    });
    assert!({
        let x = f64_accuracy(0.89, 0.88888888888888);
        (x > 2.0) & (x < 3.0)
    });
    assert!({
        let x = f64_accuracy(0.889, 0.88888888888888);
        (x > 3.0) & (x < 4.0)
    });
    assert!({
        let x = f64_accuracy(0.8889, 0.88888888888888);
        (x > 4.0) & (x < 5.0)
    });
    assert!(f64_accuracy(0.0, 0.0) == f64::INFINITY);
    assert!(f64_accuracy(1.0, 0.0) == -f64::INFINITY);
}

/// checks the correctness of basic binary operations on array vectors (`+`, `-`, `*`, `/`)
/// 
/// they are first checked individually and then a compound expression of them is checked
#[test]
fn vec_basic_arithmetic_ops_test() {
    let vec1 = MathVector::from([1, 2, 3]);
    let vec2 = MathVector::from([4, 6, 8]);
    let add = (&vec1 + &vec2).eval();
    let sub = (&vec1 - &vec2).eval();
    let mul = (&vec1 * 3).eval();
    let div = (&vec2 / 2).eval();
    assert_eq!(<[_; _]>::from(add), [5, 8, 11], "Add failed");
    assert_eq!(<[_; _]>::from(sub), [-3, -4, -5], "Sub failed");
    assert_eq!(<[_; _]>::from(mul), [3, 6, 9], "Mul failed");
    assert_eq!(<[_; _]>::from(div), [2, 3, 4], "Div failed");
    let compound = ((&vec1 + &vec2) + (&vec1 - &vec2) - (&vec1 * 3) - (&vec2 / 2)).eval();
    assert_eq!(
        <[_; _]>::from(compound),
        [-3, -5, -7],
        "Compound Expr Failed"
    );
}

/// checks the correctness of basic binary operations on runtime sized vectors (`+`, `-`, `*`, `/`)
/// 
/// they are first checked individually and then a compound expression of them is checked
#[test]
fn rs_vec_basic_arithmetic_ops_test() {
    // NOTE: the type specification (specfically the `i32` bit) is needed to avoid a bounds checking infinite recursion error, idfk why
    let vec1: RSMathVector<i32> = RSMathVector::from(vec![1, 2, 3]);
    let vec2 = RSMathVector::from(vec![4, 6, 8]);
    let add = (&vec1 + &vec2).eval();
    let sub = (&vec1 - &vec2).eval();
    let mul = (&vec1 * 3).eval();
    let div = (&vec2 / 2).eval();
    assert_eq!(<Vec<_>>::from(add), vec![5, 8, 11], "Add failed");
    assert_eq!(<Vec<_>>::from(sub), vec![-3, -4, -5], "Sub failed");
    assert_eq!(<Vec<_>>::from(mul), vec![3, 6, 9], "Mul failed");
    assert_eq!(<Vec<_>>::from(div), vec![2, 3, 4], "Div failed");
    let compound = ((&vec1 + &vec2) + (&vec1 - &vec2) - (&vec1 * 3) - (&vec2 / 2)).eval();
    assert_eq!(
        <Vec<_>>::from(compound),
        vec![-3, -5, -7],
        "Compound Expr Failed"
    );
}

/// checking if array vectors get allocated on the heap without first being on the stack
/// 
/// this test (& thus proper heaping) is **KNOWN** to fail without compiling with `--release`
/// 
/// also doubles as a list of known valid ways to generate array vectors on the heap
#[test]
fn boxed_array_vector_test() {
    black_box(VectorExprBuilder::<UNSTACKABLE_SIZE>.gen_zeroed::<u64>().heap_eval());
    black_box(HeapedVectorExprBuilder::<UNSTACKABLE_SIZE>.gen_zeroed::<u64>().eval());
    black_box(HeapedVectorExprBuilder::<UNSTACKABLE_SIZE>.new_zeroed::<u64>());
    black_box(Box::new(VectorExprBuilder::<UNSTACKABLE_SIZE>.new_zeroed::<u64>())); 
}

/// vector benchmark: calculates the cosine of the angle between 2 10000 dimension vectors via their dot product (x10000 times)
/// 
/// the compilation of this w/o error also signifies that outputs can be stacked w/o issue
/// 
/// prints:
/// - the time elapsed in nanoseconds
/// 
/// note: can fail from stack overflow without `--release`
#[test]
#[ignore]
fn vec_angle_cos() {
    let mut rng = rand::rng();
    let mut time = Duration::new(0, 0);
    for _ in 0..10000 {
        let vec1: MathVector<f64, 10000> = black_box(VectorExprBuilder.generate(|| rng.random()).eval());
        let vec2: MathVector<f64, 10000> = black_box(VectorExprBuilder.generate(|| rng.random()).eval());
        let now = Instant::now();
        let ((vec1_sqr_mag, vec2_sqr_mag), dot_product): ((f64, f64), f64) =
            (vec1.copied_sqr_mag()).dot(vec2.copied_sqr_mag()).consume();
        let mag = dot_product / ((vec1_sqr_mag * vec2_sqr_mag).sqrt());
        black_box(mag);
        let elapsed = now.elapsed();
        time += elapsed;
    }
    println!("Elapsed: {}", time.as_nanos());
}

/// a test whos primary purpose is to check that evaled in place vectors compile w/o issue
/// 
/// this test can't fail in runtime thus it is `#[ignore]`
/// 
/// generates 2 random vectors, component multiplies them, calculates the sum, evals in place, prefix sums it, then calculates the product
/// 
/// prints:
/// - the 2 starting vectors
/// - their component multiplication
/// - a prefix sum of their multiplication
/// - the sum before the prefix 
/// - the product after the prefix
#[test]
#[ignore]
fn eval_in_place_vectors_test() {
    let mut rng = rand::rng();
    let vec1: MathVector<u64, 10> = VectorExprBuilder.generate(|| rng.random_range(0..10)).eval();
    let vec2: MathVector<u64, 10> = VectorExprBuilder.generate(|| rng.random_range(0..10)).eval();
    println!("vec1: {}", vec1);
    println!("vec2: {}", vec2);
    let mut vec3 = (vec1.reuse().comp_mul(vec2))
        .copied_sum::<u64>()
        .eval_in_place();
    println!("vec3: ");
    for i in 0..10 {
        println!("{}", vec3[i]);
    }
    println!("prefix_summing...");
    for i in 1..10 {
        vec3[i] += vec3[i-1];
    }
    println!("vec3: ");
    for i in 0..10 {
        println!("{}", vec3[i]);
    }
    let (sum, product) = vec3.product::<u64>().consume();
    println!("sum: {}, product: {}", sum, product);
}

/// a test whos primary purpose is to check that repeatable vectors compile w/o issue
/// 
/// this test can't fail in runtime thus it is `#[ignore]`
/// 
/// component-wise multiplies 2 vectors and gets the sum and product of all the elements
/// and grabs 200 random values from the multiplication of the 2 vectors
/// 
/// tests the ability to grab arbitrary values from a repeatable vector (obtained via evaling in place)
/// 
/// prints:
/// - 200 random values from the multiplication of 2 vectors
/// - sum and product of all the elemements
#[test]
#[ignore]
fn repeatable_vectors_test() {
    // although IsRepeatable would likely mostly be only used internally, it has minimal external use
    let mut rng = rand::rng();
    let vec1: MathVector<f64, 10000> = VectorExprBuilder.generate(|| rng.random()).eval();
    let vec2: MathVector<f64, 10000> = VectorExprBuilder.generate(|| rng.random()).eval();
    let mut vec3 = (vec1.reuse().comp_mul(vec2))
        .copied_sum::<f64>()
        .eval_in_place();
    for _ in 0..200 {
        // enabled by IsRepeatable
        println!("{}", vec3.get(rng.random_range(0..10000)));
    }
    let (sum, product) = vec3.product::<f64>().consume();
    println!("sum: {}, product: {}", sum, product);
}

/// checks the correctness of matrix multiplication, tests for ~12 digits of accuracy (see ``f64_accuracy``)
/// 
/// multiplies 2 3x3 hardcoded matrices and compares the result against the hardcoded result
/// 
/// prints:
/// - the input matrices
/// - the resulting matrix
#[test]
fn mat_mat_mul_test() {
    let mat1 = MathMatrix::from([
        [0.242, 0.740, 0.959],
        [0.454, 0.501, 0.535],
        [0.442, 0.081, 0.973],
    ])
    .transpose()
    .eval();
    println!("input 1: {}", mat1);
    let mat2 = MathMatrix::from([
        [0.242, 0.740, 0.959],
        [0.454, 0.501, 0.535],
        [0.442, 0.081, 0.973],
    ])
    .transpose()
    .eval();
    println!("input 2: {}", mat2);
    let out_mat = mat1.mat_mul(mat2).eval();
    (&out_mat)
        .copied()
        .zip(
            MathMatrix::from([
                [0.818402, 0.627499, 1.561085],
                [0.573792, 0.630296, 1.223976],
                [0.573804, 0.446474, 1.413942],
            ])
            .transpose(),
        )
        .entry_map(|(v1, v2)| assert!(f64_accuracy(v1, v2) > 12.0))
        .consume();
    println!("Multiplication: {}", out_mat);
}

/// preforms a multiplication between 2 1000x1000 matrices ()
/// 
/// I believe this is in total 2 billion floating point operations (2 GFLOP)
/// 
/// prints:
///     duration of the calculation in nanoseconds
#[test]
#[ignore]
fn mat_mat_mul_preformance_test() {
    let mut rng = rand::rng();
    let mat1: Box<MathMatrix<f64, 1000, 1000>> = black_box(MatrixExprBuilder.generate(|| rng.random()).heap_eval());
    let mat2: Box<MathMatrix<f64, 1000, 1000>> = black_box(MatrixExprBuilder.generate(|| rng.random()).heap_eval());
    let now = Instant::now();
    let out = (mat1).mat_mul(mat2).heap_eval();
    black_box(out);
    let elapsed = now.elapsed();
    println!("Elapsed: {}", elapsed.as_nanos());
}

/// tests the preformance difference of a light calculation between different vector variants and sizes
/// 
/// to run, use `cargo test --release vector_variation_test -- --no-capture --include-ignored`.
/// 
/// sizes tested: powers of 2 from 8 to 32768
/// 
/// variants tested:
///     Normal: `VectorExpr<_>`
///     Heaped: `Box<VectorExpr<_>>`
///     Dynamic: `VectorExpr<dyn VectorLike>`
///     Heaped Dynamic: `Box<VectorExpr<dyn VectorLike>>`
/// 
/// for each size and variant combo, 2^24 additions are performed
/// prints preformance of each combo in nanoseconds
/// 
/// results (on my 2023 Macbook Pro):
///     Heaped is better than Normal at size >= 256, and worse size <= 128
///     Dyn Heap is better than Dynamic at size >= 64, and comparable at smallar sizes (and better at size 8 but that feels like a separate trend)
///     Dyn Heap is comparable to Heaped most of the time (if not a little faster)
#[test]
#[ignore]
fn vector_variation_test() {
    let mut rng = rand::rng();
    macro_rules! test_for_builder {
        (
            $rng:ident;
            $($builder:expr;)*
        ) => {
            $(
                let mut normal_time = Duration::new(0, 0);
                let mut heap_time = Duration::new(0, 0);
                let mut dynamic_time = Duration::new(0, 0);
                let mut dyn_heap_time = Duration::new(0, 0);
                let builder = $builder;
                for _ in 0..((1 << 24) / builder.size()) {
                    let vec1 = black_box(builder.generate::<_, f64>(|| $rng.random()).eval());
                    let vec2 = black_box(builder.generate::<_, f64>(|| $rng.random()).eval());
                    let now = Instant::now();
                    let res = (vec1.reuse() + vec2).eval();
                    black_box(res);
                    let elapsed = now.elapsed();
                    normal_time += elapsed;
            
                    let vec1 = black_box(builder.generate::<_, f64>(|| $rng.random()).heap_eval());
                    let vec2 = black_box(builder.generate::<_, f64>(|| $rng.random()).heap_eval());
                    let now = Instant::now();
                    let res = (vec1.reuse() + vec2).eval();
                    black_box(res);
                    let elapsed = now.elapsed();
                    heap_time += elapsed;
            
                    let vec1 = black_box(builder.generate::<_, f64>(|| $rng.random()).eval());
                    let vec2 = black_box(builder.generate::<_, f64>(|| $rng.random()).eval());
                    let now = Instant::now();
                    let res = (vec1.reuse() + vec2).make_dynamic().eval();
                    black_box(res);
                    let elapsed = now.elapsed();
                    dynamic_time += elapsed;
            
                    let vec1 = black_box(builder.generate::<_, f64>(|| $rng.random()).heap_eval());
                    let vec2 = black_box(builder.generate::<_, f64>(|| $rng.random()).heap_eval());
                    let now = Instant::now();
                    let res = (vec1.reuse() + vec2).make_dynamic().eval();
                    black_box(res);
                    let elapsed = now.elapsed();
                    dyn_heap_time += elapsed;
                }
                println!("Size: {}", builder.size());
                println!("\tNormal Time:    {:>9}", normal_time.as_nanos());
                println!("\tHeap Time:      {:>9}", heap_time.as_nanos());
                println!("\tDynamic Time:   {:>9}", dynamic_time.as_nanos());
                println!("\tDyn Heap Time:  {:>9}", dyn_heap_time.as_nanos());
            )*
        };
    }
    test_for_builder!(
        rng;
        VectorExprBuilder::<8>;
        VectorExprBuilder::<16>;
        VectorExprBuilder::<32>;
        VectorExprBuilder::<64>;
        VectorExprBuilder::<128>;
        VectorExprBuilder::<256>;
        VectorExprBuilder::<512>;
        VectorExprBuilder::<1024>;
        VectorExprBuilder::<2048>;
        VectorExprBuilder::<4096>;
        VectorExprBuilder::<8192>;
        VectorExprBuilder::<16384>;
        VectorExprBuilder::<32768>;
    );
}

/// tests rref basic correctness, tests for ~12 digits of accuracy (uses `f64_accuracy`) (real values only given to 14)
/// 
/// performs rref on a 3x4 matrix to simulate solving a system of 3 equations with 3 variables
/// and on a 3x3 matrix augmented with the identity matrix to simulate calculating the inverse matrix
/// 
/// prints for each rref:
/// - input matrix
/// - resulting determinant value
#[test]
fn rref_test() {
    // solving a system of equations
    let mut mat = MathMatrix::from([
        [0.242, 0.740, 0.959, 0.774],
        [0.454, 0.501, 0.535, 0.969],
        [0.442, 0.081, 0.973, 0.506],
    ])
    .transpose()
    .eval();
    println!("init: {}", mat);
    mat.rref();
    (&mat)
        .zip(
            MathMatrix::from([
                [1.0, 0.0, 0.0, 1.451144618141667],
                [0.0, 1.0, 0.0, 0.84263819341021],
                [0.0, 0.0, 1.0, -0.209311012214639],
            ])
            .transpose(),
        )
        .entry_map(|(v1, v2)| assert!(f64_accuracy(*v1, v2) > 12.0, "Rref Accuracy Fail"))
        .consume();
    println!("rref: {}", mat);

    // finding an inverse
    let mut mat = MathMatrix::from([
        [2.0, 1.0, 3.0, 1.0, 0.0, 0.0],
        [0.0, 2.0, 4.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 2.0, 0.0, 0.0, 1.0],
    ])
    .transpose()
    .eval();
    println!("init: {}", mat);
    mat.rref();
    (&mat)
        .zip(
            MathMatrix::from([
                [1.0, 0.0, 0.0, 0.0, -0.5, 1.0],
                [0.0, 1.0, 0.0, -2.0, -0.5, 4.0],
                [0.0, 0.0, 1.0, 1.0, 0.5, -2.0],
            ])
            .transpose(),
        )
        .entry_map(|(v1, v2)| assert!(f64_accuracy(*v1, v2) > 12.0, "Rref Accuracy Fail"))
        .consume();
    println!("rref: {}", mat);
}

/// tests det basic correctness, tests for ~12 digits of accuracy (uses `f64_accuracy`)
/// 
/// performs det on a 3x3 matrix 
/// 
/// prints:
/// - input matrix
/// - resulting determinant value
#[test]
fn det_test() {
    let mat = MathMatrix::from([
        [0.242, 0.740, 0.959],
        [0.454, 0.501, 0.535],
        [0.442, 0.081, 0.973],
    ])
    .transpose()
    .eval();
    println!("init: {}", mat);
    let det = mat.det();
    assert!(f64_accuracy(det, -0.221516496) > 12.0, "Det Accuracy Fail");
    println!("det: {}", det);
}

/// tests the performance of rref & det
/// 
/// rref is performed on a 1000x2000 matrix.
/// det is performed on a 1500x1500 matrix.
#[test]
#[ignore]
fn mat_math_performance_test() {
    let mut rng = rand::rng();
    let mut rref_mat = black_box(MatrixExprBuilder::<1000, 2000>.generate::<_, f64>(|| rng.random()).heap_eval());
    let det_mat = black_box(MatrixExprBuilder::<1500, 1500>.generate::<_, f64>(|| rng.random()).heap_eval());

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

/// tests mat_vec_mul basic correctness to ~12 digits (see [`f64_accuracy`])
/// 
/// uses a hardcoded 3x3 matrix and dim 3 vector to do so
#[test]
fn mat_vec_mul_test() {
    let mat = MathMatrix::from([
        [0.242, 0.740, 0.959],
        [0.454, 0.501, 0.535],
        [0.442, 0.081, 0.973],
    ])
    .transpose()
    .eval();
    println!("mat: {}", mat);
    let vec = MathVector::from([0.774, 0.969, 0.506]);
    println!("vec: {}", vec);
    let out_vec = mat.mat_vec_mul::<_, f64>(vec).eval();
    (&out_vec)
        .copied()
        .zip(MathVector::from([1.389622, 1.107575, 0.912935]))
        .map(|(v1, v2)| assert!(f64_accuracy(v1, v2) > 12.0, "Mat * Vec Accuracy Fail"))
        .consume();
    println!("mat * vec: {}", out_vec);
}

/// tests mat_vec_mul performance by multiplying a 10000x10000 matrix with a dim 10000 vector
/// 
/// this contains 200 million floating point operations (200 MFLOP)
/// 
/// prints:
///     duration of the calculation in nanoseconds
#[test]
#[ignore]
fn mat_vec_mul_performance_test() {
    let mut rng = rand::rng();
    let mat = black_box(MatrixExprBuilder::<10000, 10000>.generate::<_, f64>(|| rng.random()).heap_eval());
    let vec = black_box(VectorExprBuilder::<10000>.generate::<_, f64>(|| rng.random()).eval());

    let now = Instant::now();
    black_box(mat.mat_vec_mul::<_, f64>(vec).eval());
    let elapsed = now.elapsed();

    println!("{}", elapsed.as_nanos());
}

/// same as [`mat_vec_mul_test`], just with a row vector on the other side
#[test]
fn vec_mat_mul_test() {
    let vec = MathVector::from([0.774, 0.969, 0.506]);
    println!("vec: {}", vec);
    let mat = MathMatrix::from([
        [0.242, 0.740, 0.959],
        [0.454, 0.501, 0.535],
        [0.442, 0.081, 0.973],
    ])
    .transpose()
    .eval();
    println!("mat: {}", mat);
    let out_vec = vec.vec_mat_mul::<_, f64>(mat).eval();
    (&out_vec)
        .copied()
        .zip(MathVector::from([0.850886, 1.099215, 1.753019]))
        .map(|(v1, v2)| assert!(f64_accuracy(v1, v2) > 12.0, "Mat * Vec Accuracy Fail"))
        .consume();
    println!("vec * mat: \n{}", out_vec);
}

/// same as [`mat_vec_mul_performance_test`], just with a row vector on the other side
#[test]
#[ignore]
fn vec_mat_mul_performance_test() {
    let mut rng = rand::rng();
    let vec = black_box(VectorExprBuilder::<10000>.generate::<_, f64>(|| rng.random()).eval());
    let mat = black_box(MatrixExprBuilder::<10000, 10000>.generate::<_, f64>(|| rng.random()).heap_eval());

    let now = Instant::now();
    black_box(vec.vec_mat_mul::<_, f64>(mat).eval());
    let elapsed = now.elapsed();

    println!("{}", elapsed.as_nanos());
}
