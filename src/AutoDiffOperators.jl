# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

"""
    AutoDiffOperators

Provides Julia operators that act via automatic differentiation.
"""
module AutoDiffOperators

using Base.Threads: nthreads

using LinearAlgebra

using Compat: @compat

using ADTypes: ADTypes, AbstractADType, NoAutoDiff
import DifferentiationInterface as DI

using FunctionWrappers: FunctionWrapper

using MatrixShapedOperators
using MatrixShapedOperators: mulfunc_operator, check_mulfunc_operator_support

export AbstractADType, NoAutoDiff

include("util.jl")
include("ad_selector.jl")
include("jacobian.jl")
include("ad_jacobian.jl")
include("gradient.jl")
include("fwd_rev_ad_selector.jl")

end # module
