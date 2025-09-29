using AutoDiffOperators
using Test

using LinearAlgebra
using ADTypes: AutoFiniteDifferences
import FiniteDifferences

include("testutils.jl")


@testset "test FiniteDifferences" begin
    ADT = AutoFiniteDifferences
    ad_module = FiniteDifferences
    structargs = false
    ad = ADSelector(ad_module)
    fwd_adsel = ad
    rev_adsel = ad

    @test ADSelector(ad) === ad
    @test ADSelector(Val(nameof(ad_module))) isa ADT
    @test ADSelector(nameof(ad_module)) isa ADT
    @test ADSelector(ad_module) isa ADT
    @test convert(ADSelector, Val(nameof(ad_module))) isa ADT
    @test convert(ADSelector, nameof(ad_module)) isa ADT
    @test convert(ADSelector, ad_module) isa ADT
    @test_deprecated ADModule(:FiniteDifferences) isa ADT
    @test_deprecated ADModule(FiniteDifferences) isa ADT

    @testset "fwd and rev sel for $ad" begin
        @test @inferred(forward_ad_selector(ad)) == fwd_adsel
        @test @inferred(reverse_ad_selector(ad)) == rev_adsel
    end

    test_adsel_functionality(ad)
end
