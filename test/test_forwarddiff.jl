# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using Test

using LinearAlgebra
using AbstractDifferentiation, ADTypes
using ForwardDiff

@testset "test_forwarddiff.jl" begin
    f(X) = diff((x -> x^2).(X))
    g(X) = sum(f(X))
    x = rand(Float32, 5)

    y_f_ref = f(x)
    J_f_ref = ForwardDiff.jacobian(f, x)
    J_z_l = rand(Float32, size(y_f_ref, 1))
    J_z_r = rand(Float32, size(x, 1))

    y_g_ref = g(x)
    grad_g_x_ref = ForwardDiff.gradient(g, x)

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
        @testset "ad" begin
            @test @inferred(forward_ad_selector(ad)) == ad
            @test @inferred(reverse_ad_selector(ad)) == ad

            wj_y, J = with_jacobian(f, x, ad)
            @test wj_y ≈ y_f_ref
            @test Matrix(J) ≈ J_f_ref
            @test J * J_z_r ≈ J_f_ref * J_z_r
            @test J_z_l' * J ≈ J_z_l' * J_f_ref
            @test jacobian_matrix(f, x, ad) ≈ J_f_ref
            @test with_gradient(g, x, ad)[1] ≈ y_g_ref
            @test with_gradient(g, x, ad)[2] ≈ grad_g_x_ref
        end
    end
end
