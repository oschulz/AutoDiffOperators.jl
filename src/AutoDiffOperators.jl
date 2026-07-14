# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

"""
    AutoDiffOperators

Provides Julia operators that act via automatic differentiation.
"""
module AutoDiffOperators

using Base.Threads: nthreads, @threads

using LinearAlgebra

using ADTypes: ADTypes, AbstractADType, NoAutoDiff
import DifferentiationInterface as DI

using FunctionWrappers: FunctionWrapper

export AbstractADType, NoAutoDiff

include("util.jl")
include("matrix_shaped_operators/MatrixShapedOperators.jl")

using .MatrixShapedOperators
using .MatrixShapedOperators: mulfunc_operator, supports_batched_mul,
    mul_impl, add_impl
export MatrixShapedOperator, MatrixFreeOperator, WrappedMatrixOperator, asoperator,
    RowGramOperator, gram_factor,
    MatrixShapedSum, MatrixShapedProduct, UniformScalingOperator,
    diagonal_operator, blockdiag_operator

include("ad_selector.jl")
include("jacobian.jl")
include("gradient.jl")
include("fwd_rev_ad_selector.jl")

end # module
