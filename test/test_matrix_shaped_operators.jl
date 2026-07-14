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
        @test cop isa MatrixShapedProduct{Float32}
        @test cop.factors == (op', op)
        @test size(cop) == (5, 5)
        @test cop * x_r ≈ A' * (A * x_r)
        @test cop * X_r ≈ A' * (A * X_r)
        @test cop' * x_r ≈ A' * (A * x_r)
        @test cop' == cop'
        @test Matrix(cop) ≈ A' * A
        @test (cop * op') isa MatrixShapedProduct
        @test (cop * op').factors == (op', op, op')
        @test (op * cop).factors == (op, op', op)
        @test (cop * cop).factors == (op', op, op', op)
        @test occursin("MatrixShapedProduct", sprint(show, cop))
        @test_throws DimensionMismatch op * op

        Ms = [randn(Float32, 5, 5) for _ in 1:10]
        pv = AutoDiffOperators.MatrixShapedProduct(map(Base.Fix1(*, 0.5f0) ∘ LinearAlgebra.Symmetric, Ms))
        @test pv isa MatrixShapedProduct{Float32,<:AbstractVector}
        pv_ref = foldl(*, map(M -> 0.5f0 * LinearAlgebra.Symmetric(M), Ms))
        @test pv * x_r[1:5] ≈ pv_ref * x_r[1:5]
        @test (pv * cop).factors isa AbstractVector
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
    @test cop isa MatrixShapedProduct{Float32}
    @test cop * x_r ≈ A' * (A * x_r)
    @test cop * X_r ≈ A' * (A * X_r)
end

@testset "RowGramOperator" begin
    A = randn(Float32, 4, 6)
    G_ref = A * A'
    x = randn(Float32, 4)
    X = randn(Float32, 4, 3)

    for factor in [A, _WrappedMatrixTestOp(A)]
        g = @inferred(RowGramOperator(factor))
        @test g isa RowGramOperator{Float32}
        @test gram_factor(g) === factor
        @test @inferred(size(g)) == (4, 4)
        @test issymmetric(g) && ishermitian(g) && isposdef(g)
        @test g' === g
        @test @inferred(g * x) ≈ G_ref * x
        @test g * X ≈ G_ref * X
        @test Matrix(g) ≈ G_ref
    end

    # column-Gram via the adjoint factor:
    W = randn(Float32, 4, 6)
    gc = RowGramOperator(_WrappedMatrixTestOp(W)')
    @test gc * randn(Float32, 6) isa AbstractVector{Float32}
    @test Matrix(gc) ≈ W' * W

    @test occursin("RowGramOperator", sprint(show, RowGramOperator(A)))
    @test RowGramOperator(A) == RowGramOperator(A)
    @test_throws ArgumentError RowGramOperator(randn(ComplexF64, 3, 3))
end

@testset "MatrixShapedSum" begin
    A = randn(Float32, 5, 5)
    B = randn(Float32, 5, 5)
    op_a = _WrappedMatrixTestOp(A)
    op_b = _WrappedMatrixTestOp(B)
    x = randn(Float32, 5)
    X = randn(Float32, 5, 3)

    u = UniformScalingOperator(2.0f0, 5)
    @test u isa MatrixShapedOperator{Float32}
    @test size(u) == (5, 5)
    @test u * x ≈ 2 * x
    @test u * X ≈ 2 * X
    @test u' === u
    @test issymmetric(u) && ishermitian(u) && isposdef(u)
    @test !isposdef(UniformScalingOperator(-1.0f0, 5))
    @test Matrix(u) ≈ 2 * I(5)

    s = op_a + op_b
    @test s isa MatrixShapedSum{Float32}
    @test size(s) == (5, 5)
    @test s * x ≈ (A + B) * x
    @test s * X ≈ (A + B) * X
    @test Matrix(s) ≈ A + B

    si = op_a + op_b + I
    @test si isa MatrixShapedSum
    @test si.terms[3] isa UniformScalingOperator
    @test si * x ≈ (A + B + I) * x
    @test Matrix(I + op_a) ≈ A + I
    @test Matrix(op_a + 2 * I) ≈ A + 2 * I
    @test Matrix((op_a + I) + (op_b + 2 * I)) ≈ A + B + 3 * I
    @test Matrix((op_a + I) + op_b) ≈ A + B + I
    @test Matrix(op_a + (op_b + I)) ≈ A + B + I

    @test s' * x ≈ (A + B)' * x
    @test Matrix(si') ≈ (A + B + I)'

    g = RowGramOperator(randn(Float32, 5, 7))
    m = g + I
    @test issymmetric(m) && ishermitian(m) && isposdef(m)
    @test Matrix(m) ≈ Matrix(g) + I
    @test !issymmetric(op_a + op_b) || issymmetric(A) && issymmetric(B)

    @test_throws DimensionMismatch op_a + _WrappedMatrixTestOp(randn(Float32, 3, 3))
    @test occursin("MatrixShapedSum", sprint(show, s))

    @testset "vector terms" begin
        Ms = [randn(Float32, 5, 5) for _ in 1:20]
        ops = map(_WrappedMatrixTestOp, Ms)
        sv = MatrixShapedSum(ops)
        @test sv isa MatrixShapedSum{Float32,<:AbstractVector}
        @test sv * x ≈ sum(Ms) * x
        @test sv * X ≈ sum(Ms) * X
        @test Matrix(sv') ≈ sum(Ms)'
        @test issymmetric(sv) == all(issymmetric, Ms)

        svi = sv + I
        @test svi isa MatrixShapedSum{Float32,<:AbstractVector}
        @test length(svi.terms) == 21
        @test svi * x ≈ (sum(Ms) + I) * x
        @test (op_a + sv).terms isa AbstractVector
        @test Matrix(op_a + sv) ≈ A + sum(Ms)
        @test Matrix(sv + sv) ≈ 2 * sum(Ms)
    end
end

@testset "diagonal_operator and blockdiag_operator" begin
    d1 = randn(Float32, 4)
    d2 = randn(Float32, 3)
    x4 = randn(Float32, 4)
    X4 = randn(Float32, 4, 3)

    dop = @inferred(diagonal_operator(d1))
    @test dop isa MatrixFreeOperator{Float32,true,true,false}
    @test size(dop) == (4, 4)
    @test dop * x4 ≈ Diagonal(d1) * x4
    @test dop * X4 ≈ Diagonal(d1) * X4
    @test dop' * x4 ≈ Diagonal(d1) * x4
    @test AutoDiffOperators.supports_batched_mul(dop.ovp)

    # all-diagonal blocks collapse to a single diagonal operator:
    bd = blockdiag_operator(diagonal_operator(d1), diagonal_operator(d2))
    @test bd isa AutoDiffOperators.MatrixShapedOperators._DiagonalMFOperator
    @test Matrix(bd) ≈ Matrix(Diagonal(vcat(d1, d2)))

    # generic blocks (mixed operators and matrices, non-square):
    A = randn(Float32, 2, 4)
    B = randn(Float32, 3, 3)
    bd2 = blockdiag_operator(_WrappedMatrixTestOp(A), B, diagonal_operator(d2))
    ref = zeros(Float32, 2 + 3 + 3, 4 + 3 + 3)
    ref[1:2, 1:4] = A
    ref[3:5, 5:7] = B
    ref[6:8, 8:10] = Matrix(Diagonal(d2))
    @test size(bd2) == size(ref)
    z = randn(Float32, size(ref, 2))
    Z = randn(Float32, size(ref, 2), 3)
    @test bd2 * z ≈ ref * z
    @test bd2 * Z ≈ ref * Z
    @test bd2' * randn(Float32, size(ref, 1)) isa AbstractVector{Float32}
    @test Matrix(bd2) ≈ ref
    @test Matrix(bd2') ≈ ref'

    # row-Gram blocks combine into a row-Gram of block-diagonal factors:
    g1 = RowGramOperator(randn(Float32, 3, 5))
    g2 = RowGramOperator(diagonal_operator(d2))
    gbd = blockdiag_operator(g1, g2)
    @test gbd isa RowGramOperator
    @test Matrix(gbd) ≈ [Matrix(g1) zeros(Float32, 3, 3); zeros(Float32, 3, 3) Matrix(g2)]

    # single-block cases:
    @test blockdiag_operator(dop) === dop
    @test blockdiag_operator(g1) === g1
    @test Matrix(blockdiag_operator(B)) ≈ B

    @testset "vector blocks" begin
        ds = [randn(Float32, 3) for _ in 1:30]
        bdv = blockdiag_operator(map(diagonal_operator, ds))
        @test bdv isa AutoDiffOperators.MatrixShapedOperators._DiagonalMFOperator
        @test Matrix(bdv) ≈ Matrix(Diagonal(reduce(vcat, ds)))

        Bs = [randn(Float32, 2, 3) for _ in 1:10]
        bdg = blockdiag_operator(map(_WrappedMatrixTestOp, Bs))
        @test size(bdg) == (20, 30)
        zz = randn(Float32, 30)
        ref = cat(Bs...; dims = (1, 2))
        @test bdg * zz ≈ ref * zz
        @test bdg * randn(Float32, 30, 4) isa AbstractMatrix{Float32}
        @test Matrix(bdg') ≈ ref'

        gs = [RowGramOperator(randn(Float32, 3, 4)) for _ in 1:5]
        gbdv = blockdiag_operator(gs)
        @test gbdv isa RowGramOperator
        @test Matrix(gbdv) ≈ cat(map(Matrix, gs)...; dims = (1, 2))
    end
end
