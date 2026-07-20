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

# Operator types live in MatrixShapedOperators, re-export the operator
# API for convenience of downstream code:
using MatrixShapedOperators
export MatrixShapedOperator, MatrixShaped, MulFuncOperator, mulfunc_operator,
    MatrixAsOperator, asoperator, asmatrix,
    RowGramOperator, gram_factor, lower_cholesky, WoodburyOperator,
    MatrixShapedSum, MatrixShapedProduct, UniformScalingOperator,
    diagonal_operator, blockdiag_operator

include("ad_selector.jl")
include("jacobian.jl")
include("gradient.jl")
include("fwd_rev_ad_selector.jl")

end # module
