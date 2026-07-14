# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using Test

using LinearAlgebra

struct _DiagMulTestFunc{T<:AbstractVector} <: Function
    d::T
end
(m::_DiagMulTestFunc)(x) = m.d .* x
AutoDiffOperators.supports_batched_mul(::_DiagMulTestFunc) = true

# Minimal MatrixShapedOperator subtype, to test the generic fallbacks:
struct _WrappedMatrixTestOp{T<:Number,M<:AbstractMatrix{T}} <: MatrixShapedOperator{T}
    A::M
end
Base.size(op::_WrappedMatrixTestOp) = size(op.A)
Base.adjoint(op::_WrappedMatrixTestOp) = _WrappedMatrixTestOp(copy(adjoint(op.A)))
Base.:(*)(op::_WrappedMatrixTestOp, x::AbstractVector{<:Number}) = op.A * x
LinearAlgebra.issymmetric(op::_WrappedMatrixTestOp) = issymmetric(op.A)
LinearAlgebra.ishermitian(op::_WrappedMatrixTestOp) = ishermitian(op.A)
LinearAlgebra.isposdef(op::_WrappedMatrixTestOp) = isposdef(op.A)

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

@testset "MatrixFreeOperator" begin
    A = randn(Float32, 4, 5)
    ovp = Base.Fix1(*, A)
    vop = Base.Fix1(*, A')
    x_l = randn(Float32, 4)
    x_r = randn(Float32, 5)
    X_l = randn(Float32, 4, 3)
    X_r = randn(Float32, 5, 3)

    op = @inferred(
        AutoDiffOperators.mulfunc_operator(
            MatrixFreeOperator, eltype(A), size(A), ovp, vop,
            Val(false), Val(false), Val(false)
        )
    )
    @test op isa MatrixFreeOperator{Float32,false,false,false}
    @test op == MatrixFreeOperator{Float32,false,false,false}(ovp, vop, size(A))

    @test @inferred(eltype(op)) == Float32
    @test @inferred(size(op)) == (4, 5)
    @test @inferred(size(op, 1)) == 4
    @test @inferred(size(op, 2)) == 5
    @test size(op, 3) == 1
    @test_throws ArgumentError size(op, 0)

    @test @inferred(issymmetric(op)) == false
    @test @inferred(ishermitian(op)) == false
    @test @inferred(isposdef(op)) == false

    @test occursin("MatrixFreeOperator", sprint(show, op))

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
        @test sop isa MatrixFreeOperator{Float32,false,false,false}
        @test sop * x_r ≈ 2 * (A * x_r)
        @test sop' * x_l ≈ 2 * (A' * x_l)
        @test op * 2 == sop
        @test (2.0 * op) isa MatrixFreeOperator{Float64}
    end

    @testset "composition" begin
        cop = @inferred(op' * op)
        @test cop isa MatrixFreeOperator{Float32}
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

    @testset "batched mul support" begin
        d = randn(Float32, 5)
        dop_ref = Diagonal(d)
        dmul = _DiagMulTestFunc(d)
        @test AutoDiffOperators.supports_batched_mul(dmul)
        # Fix1-multiplication by a matrix accepts matrix arguments natively:
        @test AutoDiffOperators.supports_batched_mul(ovp)
        @test !AutoDiffOperators.supports_batched_mul(x -> A * x)

        dop = MatrixFreeOperator{Float32,true,true,false}(dmul, dmul, (5, 5))
        @test dop * x_r ≈ dop_ref * x_r
        @test dop * X_r ≈ dop_ref * X_r
        @test dop' * X_r ≈ dop_ref' * X_r

        @test AutoDiffOperators.supports_batched_mul((2 * dop).ovp)
        @test (2 * dop) * X_r ≈ 2 * (dop_ref * X_r)
        @test AutoDiffOperators.supports_batched_mul((dop' * dop).ovp)
        nb_op = MatrixFreeOperator{Float32,false,false,false}(x -> A * x, x -> A' * x, size(A))
        @test !AutoDiffOperators.supports_batched_mul((nb_op * dop).ovp)
        @test (dop' * dop) * X_r ≈ dop_ref' * (dop_ref * X_r)
    end

    @testset "traits and trait validation" begin
        B = randn(Float32, 5, 5)
        S = B'B + I
        sym_op = MatrixFreeOperator{Float32,true,true,true}(Base.Fix1(*, S), Base.Fix1(*, S), size(S))
        @test issymmetric(sym_op) && ishermitian(sym_op) && isposdef(sym_op)
        @test issymmetric(sym_op') && ishermitian(sym_op') && isposdef(sym_op')
        @test isposdef(2 * sym_op) == false
        @test issymmetric(2 * sym_op) && ishermitian(2 * sym_op)

        @test_throws ArgumentError MatrixFreeOperator{Float32,true,false,false}(ovp, vop, size(A))
        @test_throws ArgumentError MatrixFreeOperator{Float32,1,0,0}(ovp, vop, (5, 5))
        @test_throws ArgumentError MatrixFreeOperator{ComplexF64,false,false,false}(ovp, vop, size(A))
    end
end

@testset "MatrixShapedOperator fallbacks" begin
    A = randn(Float32, 4, 5)
    x_l = randn(Float32, 4)
    x_r = randn(Float32, 5)
    X_r = randn(Float32, 5, 3)

    op = _WrappedMatrixTestOp(A)
    @test op isa MatrixShapedOperator{Float32}
    @test eltype(op) == Float32
    @test size(op, 1) == 4 && size(op, 2) == 5

    @test op * x_r ≈ A * x_r
    @test op * X_r ≈ A * X_r
    @test op' * x_l ≈ A' * x_l
    @test x_l' * op ≈ x_l' * A
    @test transpose(op) * x_l ≈ transpose(A) * x_l
    @test Matrix(op) ≈ A
    @test mul!(similar(x_l), op, x_r) ≈ A * x_r

    sop = 3 * op
    @test sop isa MatrixFreeOperator{Float32}
    @test sop * x_r ≈ 3 * (A * x_r)
    @test sop' * x_l ≈ 3 * (A' * x_l)

    cop = op' * op
    @test cop isa MatrixFreeOperator{Float32}
    @test cop * x_r ≈ A' * (A * x_r)
    @test cop * X_r ≈ A' * (A * X_r)
end
