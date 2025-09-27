# Math Vector

A lazily-evaluated linear algebra for Rust. 

## Usage

Provides Wrapper types (ex. `VectorExpr` & `MatrixExpr`) which
represent expressions. These types can be evaluated in 
contrete collections / values by using certain methods 
(ex. `collect()` & `eval()`). These expressions are built up
from base values (wrapped concrete collections or numerical values)
used in methods on the expression (ex. `a.add(b)`). Many 
standard operations with overloads are overloaded (ex. `+`, `*`, `-`, `/`) (footnotes 1, 2).

note 1: `vec1 * vec2` is not a valid operation, see dot or comp_mul operations
note 2: `scalar * vec` should be valid but due to orphan rules is not implemented, 
        `vec * scalar` is implemented though and for associative Mul impl is identitical, if not see mul_r

## Miscelaneous

This library is large but not all encompassing it is currently missing:
 -  eigenvector and eigenvalue calculation
 -  fast matrix multiplication
 -  and probably more that I don't realize

Although I would love to do so, not all operations are implemented lazily, specifically:
 -  rref
 -  det
