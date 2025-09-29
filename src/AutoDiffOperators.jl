# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

"""
    AutoDiffOperators

Provides Julia operators that act via automatic differentiation.
"""
module AutoDiffOperators

using Base.Threads: nthreads, threadid, @threads

using LinearAlgebra

using ADTypes: ADTypes, AbstractADType
using ADTypes: NoAutoDiff

import DifferentiationInterface as DI

include("util.jl")
include("mulfunc_operator.jl")
include("ad_selector.jl")
include("jacobian.jl")
include("gradient.jl")
include("fwd_rev_ad_selector.jl")

end # module
