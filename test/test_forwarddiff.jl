# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using Test

using LinearAlgebra
using ADTypes: AutoForwardDiff
import ForwardDiff

include("testutils.jl")


@testset "test ForwardDiff" begin
    ADT = AutoForwardDiff
    ad_module = ForwardDiff
    structargs = false
    ad = ADSelector(ad_module)
    fwd_adsel = ad
    rev_adsel = ad

    @test ADSelector(Val(nameof(ad_module))) isa ADT
    @test ADSelector(nameof(ad_module)) isa ADT
    @test ADSelector(ad_module) isa ADT
    @test_deprecated ADModule(:ForwardDiff) isa ADT
    @test_deprecated ADModule(ForwardDiff) isa ADT

    @testset "fwd and rev sel for $ad" begin
        @test @inferred(forward_ad_selector(ad)) == fwd_adsel
        @test @inferred(reverse_ad_selector(ad)) == rev_adsel
        @test @inferred(AutoDiffOperators.supports_structargs(ad)) == false
    end

    test_adsel_functionality(ad)
end
