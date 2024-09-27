# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using Test

using LinearAlgebra
using ADTypes: AutoForwardDiff, AutoFiniteDifferences
import ForwardDiff, FiniteDifferences

include("testutils.jl")


@testset "test ForwardDiff" begin
    structargs = false
    ad_backend = ADSelector(ForwardDiff, FiniteDifferences)
    @test ADSelector(fwd = ForwardDiff, rev = FiniteDifferences) == ad_backend

    fwd_adsel = ADSelector(ForwardDiff)
    rev_adsel = ADSelector(FiniteDifferences)

    for ad in [ad_backend]
        @testset "fwd and rev sel for $ad" begin
            @test @inferred(forward_ad_selector(ad)) == fwd_adsel
            @test @inferred(reverse_ad_selector(ad)) == rev_adsel
            @test_throws ArgumentError AutoDiffOperators.supports_structargs(ad)
        end

        test_adsel_functionality(ad)
    end
end
