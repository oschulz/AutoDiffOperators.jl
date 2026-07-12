# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using Test

using LinearAlgebra

@testset "mulfunc_operator" begin
    A = randn(Float32, 4, 5)

    @test @inferred(
        AutoDiffOperators.mulfunc_operator(
            Matrix, eltype(A), size(A),
            Base.Fix1(*, A), Base.Fix1(*, A'),
            Val(false), Val(false), Val(false)
        )
    ) ≈ A
end

@testset "MulFuncOperator" begin
    A = randn(Float32, 4, 5)
    ovp = Base.Fix1(*, A)
    vop = Base.Fix1(*, A')
    x_l = randn(Float32, 4)
    x_r = randn(Float32, 5)
    X_l = randn(Float32, 4, 3)
    X_r = randn(Float32, 5, 3)

    op = @inferred(
        AutoDiffOperators.mulfunc_operator(
            MulFuncOperator, eltype(A), size(A), ovp, vop,
            Val(false), Val(false), Val(false)
        )
    )
    @test op isa MulFuncOperator{Float32,false,false,false}
    @test op == MulFuncOperator{Float32,false,false,false}(ovp, vop, size(A))

    @test @inferred(eltype(op)) == Float32
    @test @inferred(size(op)) == (4, 5)
    @test @inferred(size(op, 1)) == 4
    @test @inferred(size(op, 2)) == 5
    @test size(op, 3) == 1
    @test_throws ArgumentError size(op, 0)

    @test @inferred(issymmetric(op)) == false
    @test @inferred(ishermitian(op)) == false
    @test @inferred(isposdef(op)) == false

    @test occursin("MulFuncOperator", sprint(show, op))

    @test @inferred(op * x_r) == A * x_r
    @test @inferred(op * X_r) ≈ A * X_r
    @test @inferred(op' * x_l) == A' * x_l
    @test @inferred(op' * X_l) ≈ A' * X_l
    @test @inferred(x_l' * op) ≈ x_l' * A
    @test @inferred(transpose(x_l) * op) ≈ transpose(x_l) * A
    @test @inferred(transpose(op) * x_l) == transpose(A) * x_l
    @test op'' == op

    @test_throws DimensionMismatch op * x_l
    @test_throws DimensionMismatch op * X_l
    @test_throws DimensionMismatch op' * x_r

    @test size(op * zeros(Float32, 5, 0)) == (4, 0)

    @testset "scalar scaling" begin
        sop = @inferred(2 * op)
        @test sop isa MulFuncOperator{Float32,false,false,false}
        @test sop * x_r ≈ 2 * (A * x_r)
        @test sop' * x_l ≈ 2 * (A' * x_l)
        @test op * 2 == sop
        @test (2.0 * op) isa MulFuncOperator{Float64}
    end

    @testset "composition" begin
        cop = @inferred(op' * op)
        @test cop isa MulFuncOperator{Float32}
        @test size(cop) == (5, 5)
        @test cop * x_r ≈ A' * (A * x_r)
        @test cop' * x_r ≈ A' * (A * x_r)
        @test_throws DimensionMismatch op * op
    end

    @testset "mul!" begin
        y = similar(x_l)
        @test mul!(y, op, x_r) === y
        @test y ≈ A * x_r
        Y = similar(X_l)
        @test mul!(Y, op, X_r) === Y
        @test Y ≈ A * X_r

        y2 = rand(Float32, 4)
        y2_ref = 2 * (A * x_r) + 3 * y2
        @test mul!(y2, op, x_r, 2, 3) === y2
        @test y2 ≈ y2_ref
    end

    @testset "materialization" begin
        @test @inferred(Matrix(op)) ≈ A
        @test convert(Matrix, op) ≈ A
    end

    @testset "traits and trait validation" begin
        B = randn(Float32, 5, 5)
        S = B'B + I
        sym_op = MulFuncOperator{Float32,true,true,true}(Base.Fix1(*, S), Base.Fix1(*, S), size(S))
        @test issymmetric(sym_op) && ishermitian(sym_op) && isposdef(sym_op)
        @test issymmetric(sym_op') && ishermitian(sym_op') && isposdef(sym_op')
        @test isposdef(2 * sym_op) == false
        @test issymmetric(2 * sym_op) && ishermitian(2 * sym_op)

        @test_throws ArgumentError MulFuncOperator{Float32,true,false,false}(ovp, vop, size(A))
        @test_throws ArgumentError MulFuncOperator{Float32,1,0,0}(ovp, vop, (5, 5))
        @test_throws ArgumentError MulFuncOperator{ComplexF64,false,false,false}(ovp, vop, size(A))
    end
end
