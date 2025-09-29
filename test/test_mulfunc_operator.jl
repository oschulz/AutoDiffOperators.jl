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
