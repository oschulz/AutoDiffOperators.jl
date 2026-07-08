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


@testset "nothing as fwd/rev selector" begin
    @test ADSelector(ForwardDiff, nothing) isa AutoForwardDiff
    @test ADSelector(nothing, FiniteDifferences) isa AutoFiniteDifferences
    @test ADSelector(fwd = nothing, rev = FiniteDifferences) isa AutoFiniteDifferences
    @test @inferred(ADSelector(AutoForwardDiff(), nothing)) isa AutoForwardDiff
    @test @inferred(ADSelector(nothing, AutoForwardDiff())) isa AutoForwardDiff
    @test @inferred(ADSelector(nothing, nothing)) isa NoAutoDiff
end
