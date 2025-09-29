# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using Test

using LinearAlgebra
using ADTypes: AutoEnzyme
import Enzyme

include("testutils.jl")


@testset "test Enzyme" begin
    ADT = AutoEnzyme
    ad_module = Enzyme
    structargs = false
    ad = ADSelector(ad_module)
    fwd_adsel = AutoEnzyme(function_annotation = Enzyme.Const, mode = Enzyme.ForwardWithPrimal)
    rev_adsel = AutoEnzyme(function_annotation = Enzyme.Const, mode = Enzyme.ReverseWithPrimal)

    @test ad.mode isa Nothing
    @test @inferred(forward_adtype(ad)).mode isa Enzyme.ForwardMode{true}
    @test @inferred(forward_adtype(forward_adtype(ad))).mode isa Enzyme.ForwardMode
    @test_throws ArgumentError forward_adtype(reverse_adtype(ad))
    @test @inferred(reverse_adtype(ad)).mode isa Enzyme.ReverseMode{true}
    @test @inferred(reverse_adtype(reverse_adtype(ad))).mode isa Enzyme.ReverseMode
    @test_throws ArgumentError reverse_adtype(forward_adtype(ad))

    @test ADSelector(ad) === ad
    @test ADSelector(Val(nameof(ad_module))) isa ADT
    @test ADSelector(nameof(ad_module)) isa ADT
    @test ADSelector(ad_module) isa ADT
    @test convert(ADSelector, Val(nameof(ad_module))) isa ADT
    @test convert(ADSelector, nameof(ad_module)) isa ADT
    @test convert(ADSelector, ad_module) isa ADT

    @testset "fwd and rev sel for $ad" begin
        @test @inferred(forward_adtype(ad)) == fwd_adsel
        @test @inferred(reverse_adtype(ad)) == rev_adsel
    end

    test_adsel_functionality(ad)
end
