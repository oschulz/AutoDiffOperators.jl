# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using Test

using LinearAlgebra
using LinearMaps
using MatrixShapedOperators
using ADTypes: AutoForwardDiff, AutoFiniteDifferences
import ForwardDiff, FiniteDifferences

isdefined(Main, :test_adsel_functionality) || include("testutils.jl")


@testset "test FwdRevADSelector" begin
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

    # LinearMaps has no representation for a forward-less operator:
    @test_throws ArgumentError with_jacobian(f, x, LinearMap, ad)
    # a selector with neither mode is rejected before f is evaluated:
    let calls = Ref(0), g = X -> (calls[] += 1; X)
        @test_throws ArgumentError with_jacobian(g, x, MatrixShapedOperator, ADSelector(nothing, nothing))
        @test_throws ArgumentError with_jacobian(g, x, LinearMap, ADSelector(nothing, nothing))
        @test calls[] == 0
    end

    y, J = with_jacobian(f, x, MatrixShapedOperator, ad)
    @test y ≈ f_x_ref
    @test J' * z_l ≈ J_ref' * z_l
    @test z_l' * J ≈ z_l' * J_ref
    @test_throws ArgumentError J * z_r
    # explicit materialization picks the available AD mode:
    @test Matrix(J) ≈ J_ref
    MatrixShapedOperators.test_operator(J; directions = :adjoint)

    y, J_mat = with_jacobian(f, x, AbstractMatrix, ad)
    @test y ≈ f_x_ref
    @test J_mat ≈ J_ref
end


@testset "unsupported operator types" begin
    f(X) = diff((x -> x^2).(X))
    x = rand(Float32, 5)
    # fails fast, before any AD preparation:
    @test_throws ArgumentError with_jacobian(f, x, Vector, ADSelector(ForwardDiff))
    @test_throws ArgumentError with_jacobian(f, x, Vector{Float32}, ADSelector(ForwardDiff))
    # the v0.3 explicit-matrix request is rejected independently of the
    # available AD modes:
    @test_throws ArgumentError with_jacobian(f, x, Matrix, ADSelector(ForwardDiff))
    @test_throws ArgumentError with_jacobian(f, x, Matrix, ADSelector(NoAutoDiff(), ForwardDiff))
end


@testset "operator eltype promotion" begin
    ad = ADSelector(ForwardDiff)
    f(X) = diff((x -> x^2).(X))
    x_int = [1, 2, 3, 4, 5]
    _, J = with_jacobian(f, x_int, LinearMap, ad)
    @test eltype(J) == Float64

    J_ref = ForwardDiff.jacobian(f, float.(x_int))
    _, J = with_jacobian(f, x_int, MatrixShapedOperator, ad)
    @test eltype(J) == Float64
    @test eltype(J._x) == Float64
    # multiplication and materialization must use the same
    # linearization point (a float snapshot of x):
    @test Matrix(J) ≈ J_ref
    @test J * Matrix{Float64}(I, 5, 5) ≈ J_ref

    # White-box probe of that shared snapshot: with separate float
    # copies, multiplication would keep using the construction-time
    # point after J._x changes. Mutating the internal field is not
    # supported use, and this relies on the prepared ForwardDiff
    # pushforward reading the point at call time:
    J._x[1] = 10
    J_ref_mutated = ForwardDiff.jacobian(f, J._x)
    @test Matrix(J) ≈ J_ref_mutated
    @test J * Matrix{Float64}(I, 5, 5) ≈ J_ref_mutated
end
