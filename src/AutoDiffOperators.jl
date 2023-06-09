# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

"""
    AutoDiffOperators

Provides Julia operators that act via automatic differentiation.
"""
module AutoDiffOperators

using LinearAlgebra

import ADTypes
import AbstractDifferentiation

include("matrix_like_operator.jl")
include("ad_selector.jl")
include("jacobian.jl")
include("gradient.jl")

@static if !isdefined(Base, :get_extension)
    using Requires
end

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" include("../ext/AutoDiffOperatorsForwardDiffExt.jl")
        @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" include("../ext/AutoDiffOperatorsZygoteExt.jl")
        @require LinearMaps = "7a12625a-238d-50fd-b39a-03d52299707e" include("../ext/AutoDiffOperatorsLinearMapsExt.jl")
    end
end

end # module
