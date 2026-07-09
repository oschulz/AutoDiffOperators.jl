# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using Test
using AutoDiffOperators
import ForwardDiff

# A Number type that is not a Real, standing in for tracing-based array
# eltypes like Reactant's TracedRNumber:
struct _NonRealNum <: Number
    x::Float64
end
Base.float(::Type{_NonRealNum}) = _NonRealNum
Base.:+(a::_NonRealNum, b::_NonRealNum) = _NonRealNum(a.x + b.x)
Base.zero(::Type{_NonRealNum}) = _NonRealNum(0.0)

AutoDiffOperators._traced_array_kind(::Vector{_NonRealNum}) = Val(:MockTraced)

Test.@testset "test_traced_arrays" begin
    ad = ADSelector(ForwardDiff)
    x = fill(_NonRealNum(1.0), 3)

    # arrays with non-Real number elements must dispatch and select the
    # unprepared, wrapper-free AD path:
    @test AutoDiffOperators.with_floatlike_contents(x) === x
    @test @inferred(gradient_func(sum, ad, x)) isa AutoDiffOperators._GradOnlyFunc{Nothing}
    @test @inferred(valgrad_func(sum, ad, x)) isa AutoDiffOperators._ValGradFunc{Nothing}
    @test @inferred(jvp_func(sum, x, ad)) isa AutoDiffOperators._UnpreppedJVPFunc
    y, vjp = with_vjp_func(sum, x, ad)
    @test y == _NonRealNum(3.0)
    @test vjp isa AutoDiffOperators._UnpreppedVJPFunc
end
