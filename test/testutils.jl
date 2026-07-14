# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using Test

using ForwardDiff
using LinearMaps

using ADTypes: NoAutoDiff
using FunctionWrappers: FunctionWrapper

include("approx_cmp.jl")

function test_adsel_functionality(ad::ADSelector)
    @testset "functionality of $ad" begin
        f(X) = diff((x -> x^2).(X))
        g(X) = sum(f(X))
        x = rand(Float32, 5)

        f_x_ref = f(x)
        J_f_ref = ForwardDiff.jacobian(f, x)
        J_z_l = rand(Float32, size(f_x_ref, 1))
        J_z_r = rand(Float32, size(x, 1))

        g_x_ref = g(x)
        grad_g_x_ref = ForwardDiff.gradient(g, x)

        ad_fwd = @inferred(forward_adtype(ad))
        if ad_fwd isa NoAutoDiff
            @test_throws ArgumentError valid_forward_adtype(ad)
        else
            @test @inferred(valid_forward_adtype(ad)) == ad_fwd
        end

        ad_rev = @inferred(reverse_adtype(ad))
        if ad_rev isa NoAutoDiff
            @test_throws ArgumentError valid_reverse_adtype(ad)
        else
            @test @inferred(valid_reverse_adtype(ad)) == ad_rev
        end

        @test @inferred(with_jvp(f, x, J_z_r, ad)) isa Tuple{Vararg{Any,2}}
        f_x, J_z = with_jvp(f, x, J_z_r, ad)
        @test f_x ≈ f_x_ref
        @test J_z ≈ J_f_ref * J_z_r


        @test @inferred jvp_func(f, x, ad) isa Function
        f_jvp = jvp_func(f, x, ad)
        @test @inferred(f_jvp(J_z_r)) isa AbstractVector{<:Number}
        J_z = f_jvp(J_z_r)
        @test J_z ≈ J_f_ref * J_z_r

        @test @inferred with_vjp_func(f, x, ad) isa Tuple{Vararg{Any,2}}
        f_x, f_vjp = with_vjp_func(f, x, ad)
        @test @inferred(f_vjp(J_z_l)) isa AbstractVector{<:Number}
        z_J = f_vjp(J_z_l)
        @test f_x ≈ f_x_ref
        @test z_J ≈ J_f_ref' * J_z_l


        @test approx_cmp(@inferred(with_jacobian(f, x, DenseMatrix, ad)), (f_x_ref, J_f_ref))

        f_x, J = @inferred with_jacobian(f, x, LinearMap, ad)
        @test f_x ≈ f_x_ref
        @test J isa FunctionMap
        @test Matrix(J) ≈ J_f_ref
        @test @inferred(J * J_z_r) ≈ J_f_ref * J_z_r
        @test @inferred(J_z_l' * J) ≈ J_z_l' * J_f_ref

        f_x, J = @inferred with_jacobian(f, x, MatrixFreeOperator, ad)
        @test f_x ≈ f_x_ref
        @test J isa MatrixFreeOperator
        @test Matrix(J) ≈ J_f_ref
        @test @inferred(J * J_z_r) ≈ J_f_ref * J_z_r
        @test @inferred(J' * J_z_l) ≈ J_f_ref' * J_z_l
        @test @inferred(J_z_l' * J) ≈ J_z_l' * J_f_ref
        let Z_r = rand(Float32, size(x, 1), 3), Z_l = rand(Float32, size(f_x_ref, 1), 3)
            @test J(Z_r) ≈ J_f_ref * Z_r
            @test J'(Z_l) ≈ J_f_ref' * Z_l
        end


        @test @inferred(with_gradient(g, x, ad)) isa Tuple{Vararg{Any,2}}
        g_x, grad_g_x = with_gradient(g, x, ad)
        @test g_x ≈ g_x_ref
        @test grad_g_x ≈ grad_g_x_ref

        let δx = zero(x)
            @test @inferred(with_gradient!(g, δx, x, ad)) isa Tuple{Vararg{Any,2}}
            g_x, grad_g_x = with_gradient!(g, δx, x, ad)
            @test g_x ≈ g_x_ref
            @test grad_g_x === δx
            @test grad_g_x ≈ grad_g_x_ref
        end

        let δx = zero(x)
            @test @inferred(with_gradient!!(g, δx, x, ad)) isa Tuple{Vararg{Any,2}}
            g_x, grad_g_x = with_gradient!!(g, δx, x, ad)
            @test g_x ≈ g_x_ref
            @test (grad_g_x === δx) == !isbits(δx)
            @test grad_g_x ≈ grad_g_x_ref
        end


        @test @inferred(only_gradient(g, x, ad)) isa AbstractVector{<:Number}
        grad_g_x = only_gradient(g, x, ad)
        @test grad_g_x ≈ grad_g_x_ref

        let δx = zero(x)
            @test @inferred(only_gradient!(g, δx, x, ad)) isa AbstractVector{<:Number}
            grad_g_x = only_gradient!(g, δx, x, ad)
            @test grad_g_x === δx
            @test grad_g_x ≈ grad_g_x_ref
        end

        let δx = zero(x)
            @test @inferred(only_gradient!!(g, δx, x, ad)) isa AbstractVector{<:Number}
            grad_g_x = only_gradient!!(g, δx, x, ad)
            @test (grad_g_x === δx) == !isbits(δx)
            @test grad_g_x ≈ grad_g_x_ref
        end


        @test @inferred(valgrad_func(g, ad)) isa Function
        f_∇f = valgrad_func(g, ad)
        @test @inferred(f_∇f(x)) isa Tuple{Vararg{Any,2}}
        g_x, grad_g_x = f_∇f(x)
        @test g_x ≈ g_x_ref
        @test grad_g_x ≈ grad_g_x_ref

        @test @inferred(valgrad_func(g, ad, x)) isa Function
        f_∇f_prep = valgrad_func(g, ad, x)
        @test @inferred(f_∇f_prep(x)) isa Tuple{Vararg{Any,2}}
        g_x, grad_g_x = f_∇f_prep(x)
        @test g_x ≈ g_x_ref
        @test grad_g_x ≈ grad_g_x_ref

        @test @inferred(gradient_func(g, ad)) isa Function
        ∇f = gradient_func(g, ad)
        @test @inferred(∇f(x)) isa AbstractVector{<:Number}
        grad_g_x = ∇f(x)
        @test grad_g_x ≈ grad_g_x_ref

        @test @inferred(gradient_func(g, ad, x)) isa Function
        ∇f_prep = gradient_func(g, ad, x)
        @test @inferred(∇f_prep(x)) isa AbstractVector{<:Number}
        grad_g_x = ∇f_prep(x)
        @test grad_g_x ≈ grad_g_x_ref

        # used for array types that opt out of AD preparation and function
        # wrappers (e.g. Reactant arrays):
        @testset "unprepped AD functions" begin
            if !(ad_fwd isa NoAutoDiff)
                f_jvp_u = AutoDiffOperators._UnpreppedJVPFunc(valid_forward_adtype(ad), f, x)
                @test f_jvp_u(J_z_r) ≈ J_f_ref * J_z_r
            end

            if !(ad_rev isa NoAutoDiff)
                f_vjp_u = AutoDiffOperators._UnpreppedVJPFunc(valid_reverse_adtype(ad), f, x)
                @test f_vjp_u(J_z_l) ≈ J_f_ref' * J_z_l
            end
        end
    end
end
