# Math Vector

A lazily-evaluated linear algebra library for Rust. 

## Usage

Provides Wrapper types (ex. `VectorExpr` & `MatrixExpr`) which
represent expressions. These expressions are built up
from wrapped concrete collections or numerical values
used in methods on the expression (ex. `a.add(b)`). They can then be evaluated into 
contrete collections / values by using certain methods 
(ex. `collect()` & `eval()`). Many 
standard operations with overloads are overloaded (ex. `+`, `-`, `*`, `/`, `%`) (notes 1-3)
as well as their assign counterpoints when possible (ex. `+=`, `-=`, `*=`, `/=`, `%=`) (note 4).

note 1: `vec1 * vec2` is not a valid operation, see dot or comp_mul operations<br>
note 2: `scalar * vec` should be valid but due to orphan rules cannot implemented, 
        `vec * scalar` is implemented though and for commutative scalar multiplication is identitical, if not see `mul_r`. (also applies to `/` and `%`)<br>
note 3: `scalar / vec` (& `scaler % vec`) (really div_r and rem_r, see note 2) as far as I'm aware aren't really a thing but is defined such that `(scalar / vec)[idx] == scalar / vec[idx]`.<br>
note 4: By the nature of assign ops, the type of mutated value cannot change so I can't pull the same lazily-evaluation shenanigans so sadly these are readily-evaluated & can only be done on concrete vectors

## Miscelaneous

This library is large but not all encompassing it is currently missing:
 -  eigenvector and eigenvalue calculation
 -  fast matrix multiplication
 -  matrix inverse calculation (explicitly, it is still possible, see `rref_test` for an example)
 -  matrix diagonalization
 -  considerations regarding complex vectors (ex: no provided inner product for complex vectors)
 -  and probably more that I don't realize

Additionally, it will likely never contain more abstract things like:
 -  Vectorspaces & subspaces (or any other mathematical set regarding linear algebra)
 -  using functions as vectors / infinite dimensional vectors (can't be meaningfully evaluated)

Although I would love to do so, not all operations are implemented lazily, specifically:
 -  `rref`
 -  `det`
 -  all the assign operations

if you are wondering what more there is to do, check `todo.txt`