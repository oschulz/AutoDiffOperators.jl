# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using SciMLOperators
using Test

using LinearAlgebra
import ForwardDiff

@testset "test_sciml_operators" begin
    A = randn(Float32, 4, 5)
    x_l = randn(Float32, 4)
    x_r = randn(Float32, 5)

    jvp = Base.Fix1(*, A)
    vjp = Base.Fix1(*, A')
    sz = size(A)
    T = eltype(A)

    for OP in [SciMLOperators.AbstractSciMLOperator, SciMLOperators.FunctionOperator]
        op = AutoDiffOperators.mulfunc_operator(OP, T, sz, jvp, vjp, Val(false), Val(false), Val(false))

        @test op isa SciMLOperators.FunctionOperator
        @test eltype(op) == T
        @test size(op) == sz
        @test SciMLOperators.islinear(op)
        @test op * x_r ≈ A * x_r
        @test op' * x_l ≈ A' * x_l
        @test mul!(similar(x_l), op, x_r) ≈ A * x_r
    end

    @testset "MulFuncOperator conversion" begin
        mfop = MulFuncOperator{T,false,false,false}(jvp, vjp, sz)

        for op in [SciMLOperators.FunctionOperator(mfop), convert(SciMLOperators.AbstractSciMLOperator, mfop)]
            @test op isa SciMLOperators.FunctionOperator
            @test op * x_r ≈ A * x_r
            @test op' * x_l ≈ A' * x_l
        end
    end

    @testset "with_jacobian" begin
        f(X) = diff((x -> x^2).(X))
        x = rand(Float32, 5)
        J_ref = ForwardDiff.jacobian(f, x)

        f_x, J = with_jacobian(f, x, SciMLOperators.AbstractSciMLOperator, ADSelector(ForwardDiff))
        @test f_x ≈ f(x)
        @test J isa SciMLOperators.FunctionOperator
        @test J * rand(Float32, 5) isa AbstractVector{<:Real}
        z_r = rand(Float32, 5)
        z_l = rand(Float32, 4)
        @test J * z_r ≈ J_ref * z_r
        @test J' * z_l ≈ J_ref' * z_l
    end
end
