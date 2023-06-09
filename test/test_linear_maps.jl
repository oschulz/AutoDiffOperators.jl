# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using LinearMaps
using Test

using LinearAlgebra

@testset "test_linear_maps" begin
    s = randn(Float32)
    A = randn(Float32, 4, 5)
    x_l = randn(Float32, 4)
    x_r = randn(Float32, 5)
    B_r = randn(Float32, 5, 6)
    B_l = randn(Float32, 3, 4)
    x_ll = randn(Float32, 3)
    x_rr = randn(Float32, 6)

    op = MatrixLikeOperator(A)

    @test @inferred(LinearMap(op)) isa LinearMaps.FunctionMap
    @test @inferred(LinearMaps.FunctionMap(op)) isa LinearMaps.FunctionMap

    @test @inferred(convert(LinearMap, op)) isa LinearMaps.FunctionMap
    @test @inferred(convert(LinearMaps.FunctionMap, op)) isa LinearMaps.FunctionMap

    lm = LinearMap(op)

    @test @inferred(size(lm)) == size(op)
    @test issymmetric(lm) == issymmetric(op)
    @test ishermitian(lm) == ishermitian(op)
    @test isposdef(lm) == isposdef(op)

    @test @inferred(Matrix(lm)) == Matrix(op)

    @test @inferred(lm * x_r) == op * x_r
    @test @inferred(x_l' * lm) == x_l' * op

    @test @inferred(LinearMap(B_l) * op) isa LinearMap
    @test @inferred((LinearMap(B_l) * op) * x_r) == (B_l * op) * x_r
    @test @inferred(x_ll' * (LinearMap(B_l) * op)) == x_ll' * (B_l * op)

    @test @inferred(lm * LinearMap(B_r)) isa LinearMap
    @test @inferred((lm * LinearMap(B_r)) * x_rr) == (op * B_r) * x_rr
    @test @inferred(x_l' * (lm * LinearMap(B_r))) ≈ x_l' * (op * B_r)
end
