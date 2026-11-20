# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using Test

using LinearAlgebra
using ADTypes: NoAutoDiff, AutoMooncake, AutoMooncakeForward
import Mooncake

isdefined(Main, :test_adsel_functionality) || include("testutils.jl")


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

    # Concurrent use of a prepared function must not depend on what the
    # caller does with the dummy argument after preparation:
    let dummy_x = rand(10^5) .+ 1, x = rand(10^5) .+ 1
        f_∇f = valgrad_func(x -> sum(log, x), ad, dummy_x)
        fill!(dummy_x, -1)
        results = fetch.([Threads.@spawn f_∇f(x) for _ in 1:4*Threads.nthreads()])
        @test all(r -> r[1] ≈ sum(log, x) && r[2] ≈ inv.(x), results)
    end
end
