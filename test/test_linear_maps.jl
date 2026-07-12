# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using LinearMaps
using Test

using LinearAlgebra

@testset "test_linear_maps" begin
    A = randn(Float32, 4, 5)
    x_l = randn(Float32, 4)
    x_r = randn(Float32, 5)
 
    jvp = Base.Fix1(*, A)
    vjp = Base.Fix1(*, A')
    sz = size(A)
    T = eltype(A)

    for OP in [LinearMap, FunctionMap]
        op = AutoDiffOperators.mulfunc_operator(OP, T, sz, jvp, vjp, Val(false), Val(false), Val(false))

        @test @inferred(op * x_r) == A * x_r
        @test @inferred(op' * x_l) == A' * x_l
    end

    @testset "MulFuncOperator conversion" begin
        mfop = MulFuncOperator{T,false,false,false}(jvp, vjp, sz)

        for lm in [
            LinearMap(mfop), LinearMap{T}(mfop), FunctionMap(mfop), FunctionMap{T}(mfop),
            convert(LinearMap, mfop), convert(LinearMap{T}, mfop),
            convert(FunctionMap, mfop), convert(FunctionMap{T}, mfop)
        ]
            @test lm isa FunctionMap{T}
            @test lm * x_r == A * x_r
            @test lm' * x_l == A' * x_l
        end

        B = LinearMap(randn(Float32, 5, 4))
        @test Matrix(mfop * B) ≈ A * Matrix(B)
        C = LinearMap(randn(Float32, 3, 4))
        @test Matrix(C * mfop) ≈ Matrix(C) * A
    end
end
