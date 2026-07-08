# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using Test

using LinearAlgebra
using LinearMaps
using ADTypes: AutoForwardDiff, AutoFiniteDifferences
import ForwardDiff, FiniteDifferences

include("testutils.jl")


@testset "test ForwardDiff" begin
    structargs = false
    ad = ADSelector(ForwardDiff, FiniteDifferences)
    @test ADSelector(fwd = ForwardDiff, rev = FiniteDifferences) == ad

    fwd_adsel = ADSelector(ForwardDiff)
    rev_adsel = ADSelector(FiniteDifferences)

    @testset "fwd and rev sel for $ad" begin
        @test @inferred(forward_adtype(ad)) == fwd_adsel
        @test @inferred(reverse_adtype(ad)) == rev_adsel
    end

    test_adsel_functionality(ad)
end


@testset "nothing as fwd/rev selector" begin
    @test ADSelector(ForwardDiff, nothing) isa AutoForwardDiff
    @test ADSelector(nothing, FiniteDifferences) isa AutoFiniteDifferences
    @test ADSelector(fwd = nothing, rev = FiniteDifferences) isa AutoFiniteDifferences
    @test @inferred(ADSelector(AutoForwardDiff(), nothing)) isa AutoForwardDiff
    @test @inferred(ADSelector(nothing, AutoForwardDiff())) isa AutoForwardDiff
    @test @inferred(ADSelector(nothing, nothing)) isa NoAutoDiff
end


@testset "reverse-only selector" begin
    ad = ADSelector(NoAutoDiff(), FiniteDifferences)
    @test @inferred(forward_adtype(ad)) isa NoAutoDiff
    @test @inferred(reverse_adtype(ad)) isa AutoFiniteDifferences

    f(X) = diff((x -> x^2).(X))
    x = rand(Float32, 5)
    f_x_ref = f(x)
    J_ref = ForwardDiff.jacobian(f, x)
    z_l = rand(Float32, 4)
    z_r = rand(Float32, 5)

    @test_throws ArgumentError jvp_func(f, x, ad)
    @test_throws ArgumentError with_jvp(f, x, z_r, ad)

    y, J = with_jacobian(f, x, LinearMap, ad)
    @test y ≈ f_x_ref
    @test z_l' * J ≈ z_l' * J_ref
    @test_throws ErrorException J * z_r

    y, J_mat = with_jacobian(f, x, DenseMatrix, ad)
    @test y ≈ f_x_ref
    @test J_mat ≈ J_ref
end


@testset "operator eltype promotion" begin
    ad = ADSelector(ForwardDiff)
    f(X) = diff((x -> x^2).(X))
    x_int = [1, 2, 3, 4, 5]
    _, J = with_jacobian(f, x_int, LinearMap, ad)
    @test eltype(J) == Float64
end
