# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using Test

using LinearAlgebra
using ADTypes: AutoForwardDiff, AutoFiniteDifferences
import ForwardDiff, FiniteDifferences

include("testutils.jl")


@testset "test ForwardDiff" begin
    structargs = false
    ad = ADSelector(ForwardDiff, FiniteDifferences)
    @test ADSelector(fwd = ForwardDiff, rev = FiniteDifferences) == ad

    fwd_adsel = ADSelector(ForwardDiff)
    rev_adsel = ADSelector(FiniteDifferences)

    @testset "fwd and rev sel for $ad" begin
        @test @inferred(forward_adtype(ad)) == fwd_adsel
        @test @inferred(reverse_adtype(ad)) == rev_adsel
    end

    test_adsel_functionality(ad)
end
