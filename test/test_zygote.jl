using AutoDiffOperators
using Test

using LinearAlgebra
using MatrixShapedOperators: MatrixShapedOperator
using ADTypes: AutoZygote
import Zygote, ForwardDiff

isdefined(Main, :test_adsel_functionality) || include("testutils.jl")


@testset "test Zygote" begin
    ADT = AutoZygote
    ad_module = Zygote
    structargs = false
    ad = ADSelector(ad_module)
    fwd_adsel = ADSelector(ForwardDiff)
    rev_adsel = ad

    @test ADSelector(ad) === ad
    @test ADSelector(Val(nameof(ad_module))) isa ADT
    @test ADSelector(nameof(ad_module)) isa ADT
    @test ADSelector(ad_module) isa ADT
    @test convert(ADSelector, Val(nameof(ad_module))) isa ADT
    @test convert(ADSelector, nameof(ad_module)) isa ADT
    @test convert(ADSelector, ad_module) isa ADT

    @testset "fwd and rev sel for $ad" begin
        @test @inferred(forward_adtype(ad)) == fwd_adsel
        @test @inferred(reverse_adtype(ad)) == rev_adsel
    end

    test_adsel_functionality(ad)
end


@testset "frozen linearization point" begin
    f(X) = diff((x -> x^2).(X))
    x = collect(1.0:5.0)
    _, J = with_jacobian(f, x, MatrixShapedOperator, ADSelector(ForwardDiff, Zygote))
    J_ref = ForwardDiff.jacobian(f, copy(x))
    z_r, z_l = rand(5), rand(4)
    # The operator must stay at its construction point when the caller
    # mutates x afterwards: the Zygote pullback is prepared at
    # construction, and re-reading backends and explicit
    # materialization must agree with it:
    x .*= 3
    @test Matrix(J) ≈ J_ref
    @test J * z_r ≈ J_ref * z_r
    @test J' * z_l ≈ J_ref' * z_l
end
