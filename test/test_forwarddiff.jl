# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using Test

using LinearAlgebra
using ADTypes: AutoForwardDiff
import ForwardDiff

isdefined(Main, :test_adsel_functionality) || include("testutils.jl")


# An array type that `copy` turns into a plain array:
struct _CopiesToArray{T} <: AbstractVector{T}
    data::Vector{T}
end
Base.size(x::_CopiesToArray) = size(x.data)
Base.getindex(x::_CopiesToArray, i::Int) = x.data[i]
Base.convert(::Type{_CopiesToArray{T}}, x::AbstractVector) where T = _CopiesToArray(Vector{T}(x))


@testset "test ForwardDiff" begin
    ADT = AutoForwardDiff
    ad_module = ForwardDiff
    structargs = false
    ad = ADSelector(ad_module)
    fwd_adsel = ad
    rev_adsel = ad

    @test ADSelector(ad) === ad
    @test ADSelector(Val(nameof(ad_module))) isa ADT
    @test ADSelector(nameof(ad_module)) isa ADT
    @test convert(ADSelector, Val(nameof(ad_module))) isa ADT
    @test convert(ADSelector, nameof(ad_module)) isa ADT
    @test convert(ADSelector, ad_module) isa ADT
    @test ADSelector(ad_module) isa ADT

    @testset "fwd and rev sel for $ad" begin
        @test @inferred(forward_adtype(ad)) == fwd_adsel
        @test @inferred(reverse_adtype(ad)) == rev_adsel
    end

    test_adsel_functionality(ad)

    # Types are valid functions as well, also for prepared AD:
    let x = rand(5), z = rand(5)
        @test @inferred(jvp_func(Array, x, ad))(z) ≈ z
        y, f_vjp = @inferred with_vjp_func(Array, x, ad)
        @test y == x && f_vjp(z) ≈ z
    end

    # Prepared gradients keep the type of the dummy argument:
    let x = _CopiesToArray([1.0, 2.0, 3.0]), g = x -> sum(abs2, x)
        @test valgrad_func(g, ad, x)(x) == (14.0, [2.0, 4.0, 6.0])
        @test gradient_func(g, ad, x)(x) == [2.0, 4.0, 6.0]
    end
end
