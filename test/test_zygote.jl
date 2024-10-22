using AutoDiffOperators
using Test

using LinearAlgebra
using ADTypes: AutoZygote
import Zygote, ForwardDiff

include("testutils.jl")


@testset "test Zygote" begin
    ADT = AutoZygote
    ad_module = Zygote
    structargs = false
    ad = ADSelector(ad_module)
    fwd_adsel = ADSelector(ForwardDiff)
    rev_adsel = ad

    @test ADSelector(ad) === ad
    @test ADSelector(Val(nameof(ad_module))) isa ADT
    @test ADSelector(nameof(ad_module)) isa ADT
    @test ADSelector(ad_module) isa ADT
    @test convert(ADSelector, Val(nameof(ad_module))) isa ADT
    @test convert(ADSelector, nameof(ad_module)) isa ADT
    @test convert(ADSelector, ad_module) isa ADT
    @test_deprecated ADModule(:Zygote) isa ADT
    @test_deprecated ADModule(Zygote) isa ADT

    @testset "fwd and rev sel for $ad" begin
        @test @inferred(forward_ad_selector(ad)) == fwd_adsel
        @test @inferred(reverse_ad_selector(ad)) == rev_adsel
        @test @inferred(AutoDiffOperators.supports_structargs(ad)) == true
    end

    test_adsel_functionality(ad)
end
