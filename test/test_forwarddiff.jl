# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using Test

using LinearAlgebra
using AbstractDifferentiation, ADTypes
using ForwardDiff

include("testutils.jl")


@testset "test_forwarddiff.jl" begin
    @test @inferred(convert_ad(AbstractDifferentiation.AbstractBackend, ADTypes.AutoForwardDiff())) isa AbstractDifferentiation.ForwardDiffBackend
    @test @inferred(convert_ad(AbstractDifferentiation.AbstractBackend, AbstractDifferentiation.ForwardDiffBackend())) isa AbstractDifferentiation.ForwardDiffBackend
    @test @inferred(convert_ad(AbstractDifferentiation.AbstractBackend, AutoDiffOperators.ADModule(:ForwardDiff))) isa AbstractDifferentiation.ForwardDiffBackend
    @test @inferred(convert_ad(ADTypes.AbstractADType, ADTypes.AutoForwardDiff())) isa ADTypes.AutoForwardDiff
    @test @inferred(convert_ad(ADTypes.AbstractADType, AbstractDifferentiation.ForwardDiffBackend())) isa ADTypes.AutoForwardDiff
    @test @inferred(convert_ad(ADTypes.AbstractADType, AutoDiffOperators.ADModule(:ForwardDiff))) isa ADTypes.AutoForwardDiff
    @test @inferred(convert_ad(AutoDiffOperators.ADModule, ADTypes.AutoForwardDiff())) isa AutoDiffOperators.ADModule
    @test @inferred(convert_ad(AutoDiffOperators.ADModule, AbstractDifferentiation.ForwardDiffBackend())) isa AutoDiffOperators.ADModule
    @test @inferred(convert_ad(AutoDiffOperators.ADModule, AutoDiffOperators.ADModule(:ForwardDiff))) isa AutoDiffOperators.ADModule

    for ad in (
        AbstractDifferentiation.ForwardDiffBackend(),
        ADTypes.AutoForwardDiff(),
        AutoDiffOperators.ADModule(:ForwardDiff),
    )
        @testset "fwd and rev for $ad" begin
            @test @inferred(forward_ad_selector(ad)) == ad
            @test @inferred(reverse_ad_selector(ad)) == ad
        end

        test_adsel_functionality(ad)
    end
end
