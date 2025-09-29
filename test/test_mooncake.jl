# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using Test

using LinearAlgebra
using ADTypes: NoAutoDiff, AutoMooncake, AutoMooncakeForward
import Mooncake

include("testutils.jl")


@testset "test mooncake" begin
    ADT = AutoMooncake
    ad_module = Mooncake
    structargs = false
    ad = ADSelector(ad_module)
    fwd_adsel = AutoMooncakeForward()
    rev_adsel = AutoMooncake()

    @test ADSelector(ad) === ad
    @test ADSelector(Val(nameof(ad_module))) isa ADT
    @test ADSelector(nameof(ad_module)) isa ADT
    @test convert(ADSelector, Val(nameof(ad_module))) isa ADT
    @test convert(ADSelector, nameof(ad_module)) isa ADT
    @test convert(ADSelector, ad_module) isa ADT
    @test ADSelector(ad_module) isa ADT

    @testset "fwd and rev sel for $ad" begin
        @test @inferred(forward_adtype(ad)) == fwd_adsel
        @test @inferred(reverse_adtype(ad)) == rev_adsel
    end

    test_adsel_functionality(ad)
end
