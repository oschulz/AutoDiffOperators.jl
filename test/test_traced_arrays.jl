# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using Test
using AutoDiffOperators
import ForwardDiff

# A real-valued Number type that is not a Real, standing in for tracing-based
# array eltypes like Reactant's TracedRNumber:
struct _NonRealNum <: Number
    x::Float64
end
Base.float(::Type{_NonRealNum}) = _NonRealNum
Base.real(::Type{_NonRealNum}) = _NonRealNum
Base.:+(a::_NonRealNum, b::_NonRealNum) = _NonRealNum(a.x + b.x)
Base.zero(::Type{_NonRealNum}) = _NonRealNum(0.0)

AutoDiffOperators._traced_array_kind(::Vector{_NonRealNum}) = Val(:MockTraced)

Test.@testset "test_traced_arrays" begin
    ad = ADSelector(ForwardDiff)
    x = fill(_NonRealNum(1.0), 3)

    # arrays with real-valued non-Real number elements must dispatch and
    # select the unprepared, wrapper-free AD path:
    @test AutoDiffOperators.with_floatlike_contents(x) === x
    @test @inferred(gradient_func(sum, ad, x)) isa AutoDiffOperators._GradOnlyFunc{Nothing}
    @test @inferred(valgrad_func(sum, ad, x)) isa AutoDiffOperators._ValGradFunc{Nothing}
    @test @inferred(jvp_func(sum, x, ad)) isa AutoDiffOperators._UnpreppedJVPFunc
    y, vjp = with_vjp_func(sum, x, ad)
    @test y == _NonRealNum(3.0)
    @test vjp isa AutoDiffOperators._UnpreppedVJPFunc

    # complex numbers must be rejected, real-AD conventions would silently
    # give incorrect results for them:
    f_scalar = x -> abs2(sum(x))
    f_vec = x -> 2 .* x
    xc = complex.(randn(4))
    zc = complex.(randn(4))
    @test_throws ArgumentError with_gradient(f_scalar, xc, ad)
    @test_throws ArgumentError only_gradient(f_scalar, xc, ad)
    @test_throws ArgumentError gradient_func(f_scalar, ad, xc)
    @test_throws ArgumentError valgrad_func(f_scalar, ad, xc)
    @test_throws ArgumentError jvp_func(f_vec, xc, ad)
    @test_throws ArgumentError with_vjp_func(f_vec, xc, ad)
    @test_throws ArgumentError with_jvp(f_vec, xc, zc, ad)
    @test_throws ArgumentError with_jacobian(f_vec, xc, DenseMatrix, ad)
end
