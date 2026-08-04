# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using Test

using LinearAlgebra
using StaticArrays
using MatrixShapedOperators: MatrixShapedOperator
import ForwardDiff

# The extension keeps static array types static: the generic
# _similar_type fallback infers similar(::SVector) as MVector.
struct _Point3{T} <: FieldVector{3,T}
    x::T
    y::T
    z::T
end

@testset "test_static_arrays" begin
    @test AutoDiffOperators._similar_type(SVector{3,Float64}) === SVector{3,Float64}
    @test AutoDiffOperators._similar_type(_Point3{Float64}) === _Point3{Float64}

    f(x) = SVector(x[1]^2 - x[3], x[2] * x[1])
    x = SVector(0.6, -0.3, 0.4)
    z_r = SVector(0.5, -1.0, 0.25)
    z_l = SVector(1.5, -0.5)
    J_ref = ForwardDiff.jacobian(f, x)
    ad = ADSelector(ForwardDiff)

    f_x, J_z = with_jvp(f, x, z_r, ad)
    @test f_x isa SVector{2} && f_x ≈ f(x)
    @test J_z isa SVector{2} && J_z ≈ J_ref * z_r

    @test jvp_func(f, x, ad)(z_r) isa SVector{2}

    f_x, vjp = with_vjp_func(f, x, ad)
    @test f_x isa SVector{2}
    @test vjp(z_l) isa SVector{3}
    @test vjp(z_l) ≈ J_ref' * z_l

    f_x, J_mat = with_jacobian(f, x, AbstractMatrix, ad)
    @test J_mat isa SMatrix{2,3}
    @test J_mat ≈ J_ref

    f_x, J = with_jacobian(f, x, MatrixShapedOperator, ad)
    @test J * z_r isa SVector{2}
    @test J' * z_l isa SVector{3}
    # backend-preserving vs dense-CPU materialization:
    @test AbstractMatrix(J) isa SMatrix{2,3}
    @test Matrix(J) isa Matrix
    @test Matrix(J) ≈ J_ref

    g(x) = sum(f(x) .^ 2)
    g_x, grad_g_x = with_gradient(g, x, ad)
    @test grad_g_x isa SVector{3}
    @test grad_g_x ≈ ForwardDiff.gradient(g, x)
end
