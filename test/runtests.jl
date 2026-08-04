# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package AutoDiffOperators" begin
    include("test_aqua.jl")
    include("test_util.jl")
    include("test_traced_arrays.jl")
    include("test_static_arrays.jl")
    include("test_linear_maps.jl")
    include("test_sciml_operators.jl")
    include("test_adselector.jl")
    include("test_finitedifferences.jl")
    include("test_forwarddiff.jl")
    include("test_zygote.jl")
    include("test_mooncake.jl")
    include("test_enzyme.jl")
    # Reactant only supports 64-bit Linux and macOS and Julia 1.10's parallel
    # precompilation is prone to deadlock on Reactant's large extension set:
    if Sys.islinux() && Sys.ARCH === :x86_64 &&
       isempty(VERSION.prerelease) && VERSION >= v"1.11"
        import Pkg
        Base.identify_package("Reactant") === nothing && Pkg.add("Reactant")
        include("test_reactant.jl")
    end
    include("test_fwd_rev_ad_selector.jl")
    include("test_docs.jl")
end # testset
