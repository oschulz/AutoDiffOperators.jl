# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using LinearMaps
using Test

using LinearAlgebra
import ForwardDiff

@testset "test_linear_maps" begin
    A = randn(Float32, 4, 5)
    x_l = randn(Float32, 4)
    x_r = randn(Float32, 5)

    @testset "MulFuncOperator conversion" begin
        jvp = Base.Fix1(*, A)
        vjp = Base.Fix1(*, A')
        mfop = MulFuncOperator{eltype(A)}(jvp, vjp, size(A))

        for lm in [
            LinearMap(mfop), LinearMap{eltype(A)}(mfop), FunctionMap(mfop),
            convert(LinearMap, mfop), convert(FunctionMap, mfop)
        ]
            @test lm isa FunctionMap{eltype(A)}
            @test lm * x_r == A * x_r
            @test lm' * x_l == A' * x_l
        end

        B = LinearMap(randn(Float32, 5, 4))
        @test Matrix(mfop * B) ≈ A * Matrix(B)
        C = LinearMap(randn(Float32, 3, 4))
        @test Matrix(C * mfop) ≈ Matrix(C) * A
    end

    @testset "with_jacobian" begin
        f(X) = diff((x -> x^2).(X))
        x = rand(Float32, 5)
        J_ref = ForwardDiff.jacobian(f, x)

        for OP in [LinearMap, FunctionMap]
            f_x, J = with_jacobian(f, x, OP, ADSelector(ForwardDiff))
            @test f_x ≈ f(x)
            @test J isa FunctionMap
            @test J * x_r ≈ J_ref * x_r
            @test J' * x_l ≈ J_ref' * x_l
        end
    end
end
