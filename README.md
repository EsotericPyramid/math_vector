# Math Vector

A lazily-evaluated linear algebra for Rust. 

## Usage

Provides Wrapper types (ex. `VectorExpr` & `MatrixExpr`) which
represent expressions. These types can be evaluated in 
contrete collections / values by using certain methods 
(ex. `collect()` & `eval()`). These expressions are built up
from base values (wrapped concrete collections or numerical values)
used in methods on the expression (ex. `a.add(b)`). Many 
standard operations with overloads are overloaded (ex. `+`, `*`, `-`, `/`)*.



*`vec1 * vec2` is not a valid operation, see dot or comp_mul operations 

