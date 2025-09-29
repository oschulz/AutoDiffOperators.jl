# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

module AutoDiffOperatorsEnzymeExt

using Enzyme

import AutoDiffOperators
using AutoDiffOperators: with_floatlike_contents
using AutoDiffOperators: _primal_return_type, _similar_type, _matrix_type, _JVPFunc, _VJPFunc

import ADTypes
using ADTypes: AutoEnzyme


function AutoDiffOperators._adsel_enzyme_forward(ad::AutoEnzyme{M, A}) where {M, A}
    mode = _enzyme_forward_withprimal(ad.mode)
    return AutoEnzyme(function_annotation = Enzyme.Const, mode = mode)
end

_enzyme_forward_withprimal(mode::Enzyme.ForwardMode{true}) = mode
_enzyme_forward_withprimal(::Nothing) = Enzyme.ForwardWithPrimal
_enzyme_forward_withprimal(mode) = throw(ArgumentError("Enzyme mode $mode is not a forward mode with primal"))

_enzyme_forward_mode(mode::Enzyme.ForwardMode) = mode
_enzyme_forward_mode(::Nothing) = Enzyme.Forward
_enzyme_forward_mode(mode) = throw(ArgumentError("Enzyme mode $mode is not a forward mode"))


function AutoDiffOperators._adsel_enzyme_reverse(ad::AutoEnzyme{M, A}) where {M, A}
    mode = _enzyme_reverse_withprimal(ad.mode)
    return AutoEnzyme(function_annotation = Enzyme.Const, mode = mode)
end

_enzyme_reverse_withprimal(mode::Enzyme.ReverseMode{true}) = mode
_enzyme_reverse_withprimal(::Nothing) = Enzyme.ReverseWithPrimal
_enzyme_reverse_withprimal(mode) = throw(ArgumentError("Enzyme mode $mode is not a reverse mode with primal"))

_enzyme_reverse_mode(mode::Enzyme.ReverseMode) = mode
_enzyme_reverse_mode(::Nothing) = Enzyme.Reverse
_enzyme_reverse_mode(mode) = throw(ArgumentError("Enzyme mode $mode is not a reverse mode"))


# Works, but DifferentiationInterface implementations are more sophisticated.
# Keep this around for now in case custom Enzyme use is required in the
# Future, e.g. for Reactant:
#=

function AutoDiffOperators._with_jacobian_matrix_impl(f, x::AbstractVector{<:Real}, ad::AutoEnzyme)
    mode = _enzyme_forward_mode(ad.mode)
    return _with_jacobian_matrix_enzyme(mode, f, x)
end

function _with_jacobian_matrix_enzyme(mode::Enzyme.ForwardMode, f, x::AbstractVector{<:Real})
    float_x = with_floatlike_contents(x)
    y = f(float_x)
    R = _matrix_type(typeof(y), typeof(float_x))
    result = Enzyme.jacobian(mode, f, float_x)
    J = convert(R, _get_jac_matrix(result))::R
    y, J
end

function _with_jacobian_matrix_enzyme(mode::Enzyme.ReverseMode, f, x::AbstractVector{<:Real})
    float_x = with_floatlike_contents(x)
    y = f(float_x)
    R = _matrix_type(typeof(y), typeof(float_x))
    n_y = length(y)  # number of outputs required by Enzyme in reverse mode
    result = Enzyme.jacobian(mode, f, float_x, n_outs = Val(n_y))
    J = convert(R, _get_jac_matrix(result.derivs))::R
    y, J
end

_get_jac_matrix(result::NamedTuple) = _get_jac_matrix(result.derivs)
_get_jac_matrix(result::Tuple{AbstractMatrix{<:Real}}) = result[1]


function AutoDiffOperators._with_jvp_impl(f, x::AbstractVector{<:Real}, z::AbstractVector{<:Real}, ad::AutoEnzyme)
    mode = _enzyme_forward_withprimal(ad.mode)
    float_x = with_floatlike_contents(x)
    float_z = with_floatlike_contents(z)
    T_z = typeof(float_z)
    T_f_x = _primal_return_type(f, float_x)
    T_J_z = _similar_type(T_z, T_f_x)
    J_z, f_x = autodiff(mode, f, Duplicated(float_x, float_z))
    return convert(T_f_x, f_x)::T_f_x, convert(T_J_z, J_z)::T_J_z
end

AutoDiffOperators._jvp_func_impl(f::F, x::AbstractVector{<:Real}, ad::AutoEnzyme) where F = _JVPFunc(nothing, ad, f, x)


function AutoDiffOperators._with_vjp_func_impl(f::F, x::AbstractVector{<:Real}, ad::AutoEnzyme) where F
    float_x = with_floatlike_contents(x)
    y = f(float_x)
    mf! = _Mutating_Func(f)
    return y, _VJPFunc((;y = y), ad, mf!, float_x)
end

function (f_vjp::_VJPFunc{<:NamedTuple{(:y,)},<:AutoEnzyme})(z)
    mf! = f_vjp.f
    δx = similar(f_vjp.x, float(eltype(f_vjp.x)))
    fill!(δx, zero(eltype(δx)))
    y = similar(f_vjp.aux.y)
    δy = deepcopy(z)
    mode = f_vjp.ad.mode

    Enzyme.autodiff(mode, Const(mf!), Duplicated(y, δy), Duplicated(f_vjp.x, δx))

    return δx
end

struct _Mutating_Func{F} <: Function
    f::F
end

_Mutating_Func(::Type{FT}) where FT = _Mutating_Func{Type{FT}}(FT)

function (mf!::_Mutating_Func)(y, x)
    y .= mf!.f(x)
    return nothing
end


# ToDo: StaticArray support
function AutoDiffOperators._with_gradient_impl(f, x::AbstractVector{<:Real}, ad::AutoEnzyme)
    δx = similar(x, float(eltype(x)))
    return AutoDiffOperators._with_gradient_impl!(f, δx, x, ad)
end

# ToDo: StaticArray support
function AutoDiffOperators._with_gradient_impl!(f, δx::AbstractVector{<:Real}, x::AbstractVector{<:Real}, ad::AutoEnzyme)
    mode = _enzyme_reverse_withprimal(ad.mode)
    float_x = with_floatlike_contents(x)
    fill!(δx, zero(eltype(δx)))
    _, y = autodiff(mode, f, Active, Duplicated(float_x, δx))
    return y, δx
end

# ToDo: StaticArray support
function AutoDiffOperators._only_gradient_impl(f, x::AbstractVector{<:Real}, ad::AutoEnzyme)
    δx = similar(x, float(eltype(x)))
    return AutoDiffOperators._only_gradient_impl!(f, δx, x, ad)
end

# ToDo: StaticArray support
function AutoDiffOperators._only_gradient_impl!(f, δx::AbstractVector{<:Real}, x::AbstractVector{<:Real}, ad::AutoEnzyme)
    mode = _enzyme_reverse_mode(ad.mode)
    float_x = with_floatlike_contents(x)
    fill!(δx, zero(eltype(δx)))
    autodiff(mode, f, Active, Duplicated(float_x, δx))
    return δx
end

=#

end # module AutoDiffOperatorsEnzymeExt
