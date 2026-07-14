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

This module only depends on `LinearAlgebra`.
"""
module MatrixShapedOperators

using LinearAlgebra

using Base.Threads: @threads

include("abstract_operator.jl")
include("matrix_free_operator.jl")

end # module MatrixShapedOperators
