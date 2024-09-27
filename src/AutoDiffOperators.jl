# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

"""
    AutoDiffOperators

Provides Julia operators that act via automatic differentiation.
"""
module AutoDiffOperators

using Base.Threads: nthreads, threadid, @threads

using LinearAlgebra

import ADTypes
import AbstractDifferentiation

using AffineMaps: Mul
using FunctionChains: fchain

include("mulfunc_operator.jl")
include("ad_selector.jl")
include("jacobian.jl")
include("gradient.jl")

end # module
