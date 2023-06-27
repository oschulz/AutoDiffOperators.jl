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
include("matrix_like_operator.jl")
include("ad_selector.jl")
include("jacobian.jl")
include("gradient.jl")

@static if !isdefined(Base, :get_extension)
    using Requires
end

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9" include("../ext/AutoDiffOperatorsEnzymeExt.jl")
        @require FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000" include("../ext/AutoDiffOperatorsFiniteDifferencesExt.jl")
        @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" include("../ext/AutoDiffOperatorsForwardDiffExt.jl")
        @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" include("../ext/AutoDiffOperatorsZygoteExt.jl")
        @require LinearMaps = "7a12625a-238d-50fd-b39a-03d52299707e" include("../ext/AutoDiffOperatorsLinearMapsExt.jl")
    end
end

end # module
