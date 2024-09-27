# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package AutoDiffOperators" begin
    include("test_aqua.jl")
    include("test_linear_maps.jl")
    include("test_finitedifferences.jl")
    include("test_forwarddiff.jl")
    include("test_zygote.jl")
    #Enzyme (up to at least v0.13.5) still seems to have trouble with
    # Julia v1.11 (at least with Julia v1.11.0-rc4)
    if VERSION < v"1.11.0-dev"
        include("test_enzyme.jl")
    end
    include("test_differentiation_interface.jl")
    include("test_fwd_rev_ad_selector.jl")
    include("test_docs.jl")
end # testset
