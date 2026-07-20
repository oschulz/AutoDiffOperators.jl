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

    @testset "MulFuncOperator conversion" begin
        jvp = Base.Fix1(*, A)
        vjp = Base.Fix1(*, A')
        mfop = MulFuncOperator{eltype(A)}(jvp, vjp, size(A))

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

        for OP in [SciMLOperators.AbstractSciMLOperator, SciMLOperators.FunctionOperator]
            f_x, J = with_jacobian(f, x, OP, ADSelector(ForwardDiff))
            @test f_x ≈ f(x)
            @test J isa SciMLOperators.FunctionOperator
            @test eltype(J) == eltype(x)
            @test size(J) == (4, 5)
            @test SciMLOperators.islinear(J)
            z_r = rand(Float32, 5)
            z_l = rand(Float32, 4)
            @test J * z_r ≈ J_ref * z_r
            @test J' * z_l ≈ J_ref' * z_l
            @test mul!(similar(z_l), J, z_r) ≈ J_ref * z_r
        end
    end
end
