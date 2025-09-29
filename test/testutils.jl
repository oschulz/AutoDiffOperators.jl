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

        wj_y, J = @inferred with_jacobian(f, x, LinearMap, ad)
        @test wj_y ≈ y_f_ref
        @test J isa FunctionMap
        @test Matrix(J) ≈ J_f_ref
        @test @inferred(J * J_z_r) ≈ J_f_ref * J_z_r
        @test @inferred(J_z_l' * J) ≈ J_z_l' * J_f_ref
        @test approx_cmp(@inferred(with_jacobian(f, x, Matrix, ad)), (y_f_ref, J_f_ref))
        @test with_gradient(g, x, ad)[1] ≈ y_g_ref
        @test with_gradient(g, x, ad)[2] ≈ grad_g_x_ref
        @test only_gradient(g, x, ad) ≈ grad_g_x_ref

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
    end
end
