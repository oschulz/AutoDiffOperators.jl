# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package AutoDiffOperators" begin
    include("test_aqua.jl")
    include("test_util.jl")
    include("test_traced_arrays.jl")
    include("test_mulfunc_operator.jl")
    include("test_linear_maps.jl")
    include("test_adselector.jl")
    include("test_finitedifferences.jl")
    include("test_forwarddiff.jl")
    include("test_zygote.jl")
    include("test_mooncake.jl")
    include("test_enzyme.jl")
    # Reactant provides prebuilt binaries only for 64-bit Linux and macOS:
    if Sys.WORD_SIZE == 64 && (Sys.islinux() || Sys.isapple()) && isempty(VERSION.prerelease)
        include("test_reactant.jl")
    end
    include("test_fwd_rev_ad_selector.jl")
    include("test_docs.jl")
end # testset
