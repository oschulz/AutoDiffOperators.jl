# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using Test

using LinearAlgebra
using AbstractDifferentiation, ADTypes
using FiniteDifferences

include("testutils.jl")


@testset "test_zygote.jl" begin
    f(X) = diff((x -> x^2).(X))
    g(X) = sum(f(X))
    x = rand(Float32, 5)

    y_f_ref = f(x)
    J_f_ref = ForwardDiff.jacobian(f, x)
    J_z_l = rand(Float32, size(y_f_ref, 1))
    J_z_r = rand(Float32, size(x, 1))

    y_g_ref = g(x)
    grad_g_x_ref = ForwardDiff.gradient(g, x)

    @test @inferred(convert_ad(AbstractDifferentiation.AbstractBackend, AbstractDifferentiation.FiniteDifferencesBackend())) isa AbstractDifferentiation.FiniteDifferencesBackend
    @test @inferred(convert_ad(AbstractDifferentiation.AbstractBackend, AutoDiffOperators.ADModule(:FiniteDifferences))) isa AbstractDifferentiation.FiniteDifferencesBackend
    @test @inferred(convert_ad(AutoDiffOperators.ADModule, AbstractDifferentiation.FiniteDifferencesBackend())) isa AutoDiffOperators.ADModule
    @test @inferred(convert_ad(AutoDiffOperators.ADModule, AutoDiffOperators.ADModule(:FiniteDifferences))) isa AutoDiffOperators.ADModule
    
    for ad in (
        AbstractDifferentiation.FiniteDifferencesBackend(),
        AutoDiffOperators.ADModule(:FiniteDifferences),
    )
        @testset "fwd and rev for $ad" begin
            @test @inferred(forward_ad_selector(ad)) == ad
            @test @inferred(reverse_ad_selector(ad)) == ad

            @test @inferred(AutoDiffOperators.supports_structargs(forward_ad_selector(ad))) == true
            @test @inferred(AutoDiffOperators.supports_structargs(reverse_ad_selector(ad))) == true
        end

        test_adsel_functionality(ad)
    end
end
