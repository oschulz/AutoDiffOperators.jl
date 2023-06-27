# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using Test

using ForwardDiff
using LinearMaps

include("approx_cmp.jl")

function test_adsel_functionality(ad::ADSelector)
    @testset "functionality of $ad" begin
        f(X) = diff((x -> x^2).(X))
        g(X) = sum(f(X))
        x = rand(Float32, 5)

        y_f_ref = f(x)
        J_f_ref = ForwardDiff.jacobian(f, x)
        J_z_l = rand(Float32, size(y_f_ref, 1))
        J_z_r = rand(Float32, size(x, 1))

        y_g_ref = g(x)
        grad_g_x_ref = ForwardDiff.gradient(g, x)

        f_nv(x) = sum(x.a .* x.a) * x.b
        x_nv = (a = [1.0, 2.0], b = 2.0)
        y_nv_ref = 10.0
        grad_nv_ref = (a = [4.0, 8.0], b = 5.0)

        @test_deprecated typeof(with_jacobian(f, x, ad)) == typeof(with_jacobian(f, x, MatrixLikeOperator, ad))
        wj_y, J = with_jacobian(f, x, MatrixLikeOperator, ad)
        @test wj_y ≈ y_f_ref
        @test Matrix(J) ≈ J_f_ref
        @test with_jacobian(f, x, LinearMap, ad)[2] isa FunctionMap
        @test Matrix(with_jacobian(f, x, LinearMap, ad)[2]) ≈ J_f_ref
        @test J * J_z_r ≈ J_f_ref * J_z_r
        @test J_z_l' * J ≈ J_z_l' * J_f_ref
        @test approx_cmp(with_jacobian(f, x, Matrix, ad), (y_f_ref, J_f_ref))
        @test_deprecated jacobian_matrix(f, x, ad) ≈ J_f_ref
        @test with_gradient(g, x, ad)[1] ≈ y_g_ref
        @test with_gradient(g, x, ad)[2] ≈ grad_g_x_ref

        let δx = similar(x)
            fill!(δx, NaN)
            @test with_gradient!!(g, δx, x, ad)[1] ≈ y_g_ref
            @test with_gradient!!(g, δx, x, ad)[2] ≈ grad_g_x_ref
        end

        @test valgrad_func(g, ad)(x)[1] ≈ y_g_ref
        @test valgrad_func(g, ad)(x)[2] ≈ grad_g_x_ref
        @test gradient_func(g, ad)(x) ≈ grad_g_x_ref
        let δx = similar(x)
            fill!(δx, NaN)
            @test gradient!_func(g, ad)(δx, x) === δx
            @test δx ≈ grad_g_x_ref
        end

        if AutoDiffOperators.supports_structargs(reverse_ad_selector(ad))
            @test approx_cmp(with_gradient(f_nv, x_nv, ad), (y_nv_ref, grad_nv_ref))
            @test approx_cmp(valgrad_func(f_nv, ad)(x_nv), (y_nv_ref, grad_nv_ref))
            @test approx_cmp(gradient_func(f_nv, ad)(x_nv), grad_nv_ref)
        end
    end
end
