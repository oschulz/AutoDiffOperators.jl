# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

"""
    AutoDiffOperators.MatrixShapedOperators

Linear operators with matrix shape and semantics that are accessed only
through multiplication and adjoint multiplication, not through element
access.

Operator traits are part of the type domain and operator application is
free of array mutation, so operators of this kind are compatible with
program tracing (e.g. via Reactant) as long as the functions and values
they are built from are.

Types from `Base` and `LinearAlgebra` (e.g. `Diagonal`) are used
directly where suitable, so that their ecosystem-wide specializations
(GPU arrays, program tracing, BLAS) take effect; custom operator types
only cover what those types can't express.

This module only depends on `LinearAlgebra`.
"""
module MatrixShapedOperators

using LinearAlgebra

include("matrix_shaped_operator.jl")
include("wrapped_matrix_operator.jl")
include("matrix_free_operator.jl")
include("row_gram_operator.jl")
include("operator_arithmetic.jl")
include("structured_operators.jl")

end # module MatrixShapedOperators
