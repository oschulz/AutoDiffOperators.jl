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
end
