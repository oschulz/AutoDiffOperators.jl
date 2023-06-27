# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using Test

using LinearAlgebra
using AbstractDifferentiation, ADTypes
using Enzyme

include("testutils.jl")


@testset "test_enzyme.jl" begin
    @test @inferred(convert_ad(ADTypes.AbstractADType, ADTypes.AutoEnzyme())) isa ADTypes.AutoEnzyme
    @test @inferred(convert_ad(ADTypes.AbstractADType, AutoDiffOperators.ADModule(:Enzyme))) isa ADTypes.AutoEnzyme
    @test @inferred(convert_ad(AutoDiffOperators.ADModule, ADTypes.AutoEnzyme())) isa AutoDiffOperators.ADModule
    @test @inferred(convert_ad(AutoDiffOperators.ADModule, AutoDiffOperators.ADModule(:Enzyme))) isa AutoDiffOperators.ADModule

    for ad in (
        ADTypes.AutoEnzyme(),
        AutoDiffOperators.ADModule(:Enzyme),
    )
        @testset "fwd and rev for $ad" begin
            @test @inferred(forward_ad_selector(ad)) == ad
            @test @inferred(reverse_ad_selector(ad)) == ad

            @test @inferred(AutoDiffOperators.supports_structargs(forward_ad_selector(ad))) == false
            @test @inferred(AutoDiffOperators.supports_structargs(reverse_ad_selector(ad))) == false
        end

        test_adsel_functionality(ad)
    end
end
