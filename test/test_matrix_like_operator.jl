# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using Test

using LinearAlgebra

@testset "test_matrix_like_operator" begin
    s = randn(Float32)
    A = randn(Float32, 4, 5)
    x_l = randn(Float32, 4)
    x_r = randn(Float32, 5)
    B_r = randn(Float32, 5, 6)
    B_l = randn(Float32, 3, 4)
    x_ll = randn(Float32, 3)
    x_rr = randn(Float32, 6)

    @test @inferred(MatrixLikeOperator(A)) isa MatrixLikeOperator{Float32, false, false, false}
    op = MatrixLikeOperator(A)

    @test @inferred(MatrixLikeOperator(op)) === op
    @test @inferred(Base.IndexStyle(op)) == Base.IndexCartesian()

    @test @inferred(Array(op)) == A
    @test @inferred(Matrix(op)) == A
    @test @inferred(Array{Float32}(op)) == A
    @test @inferred(Matrix{Float32}(op)) == A
    @test @inferred(convert(Array, op)) == A
    @test @inferred(convert(Matrix, op)) == A
    @test @inferred(convert(Array{Float32}, op)) == A
    @test @inferred(convert(Matrix{Float32}, op)) == A

    @test typeof(@inferred(adjoint(adjoint(op)))) == typeof(op)
    @test @inferred(adjoint(adjoint(op)) == op)
    @test @inferred(adjoint(adjoint(op)) ≈ op)
    @test typeof(@inferred(transpose(transpose(op)))) == typeof(op)
    @test @inferred(transpose(transpose(op)) == op)
    @test @inferred(transpose(transpose(op)) ≈ op)

    @test Matrix(@inferred(op * s)) == A * s
    @test Matrix(@inferred(s * op)) == s * A
    @test @inferred(op * x_r) == A * x_r
    @test @inferred(x_l' * op) == x_l' * A
    @test @inferred(transpose(x_l) * op) == transpose(x_l) * A
    @test Matrix(@inferred(op * B_r)) ≈ A * B_r
    @test Matrix(@inferred(B_l * op)) ≈ B_l * A

    @test @inferred(mul!(similar(A*s), op, s)) == mul!(similar(A*s), A, s)
    @test @inferred(mul!(similar(s * A), s, op)) == mul!(similar(s * A), s, A)
    @test @inferred(mul!(similar(A*x_r), op, x_r)) == mul!(similar(A*x_r), A, x_r)
    @test @inferred(mul!(similar(x_l' * A), x_l', op)) ≈ mul!(similar(x_l' * A), x_l', A)
    @test @inferred(mul!(similar(A*B_r), op, B_r)) ≈ mul!(similar(A*B_r), A, B_r)
    @test @inferred(mul!(similar(A*adjoint(A)), op, adjoint(op))) ≈ mul!(similar(A*adjoint(A)), A, adjoint(A))
    @test @inferred(mul!(similar(A*transpose(A)), op, transpose(op))) ≈ mul!(transpose(A*adjoint(A)), A, transpose(A))
    @test @inferred(mul!(similar(similar(B_l * A)), B_l, op)) ≈ mul!(similar(B_l * A), B_l, A)

    @test_throws ArgumentError op[3, 4]
    @test @inferred(op[:, 1]) == A[:, 1]
    @test @inferred(op[:, 5]) == A[:, 5]
    @test @inferred(op[1, :]) == A[1, :]
    @test @inferred(op[4, :]) == A[4, :]
    @test @inferred(op[2:3, 4]) == A[2:3, 4]
    @test @inferred(op[3, 3:4]) == A[3, 3:4]
    @test @inferred(op[:, 3:4]) == A[:, 3:4]
    @test @inferred(op[2:3, :]) == A[2:3, :]
    @test @inferred(op[2:3, 3:4]) == A[2:3, 3:4]
    @test @inferred(op[:, :]) == A[:, :]
end
