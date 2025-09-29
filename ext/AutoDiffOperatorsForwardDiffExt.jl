# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

module AutoDiffOperatorsForwardDiffExt

using ForwardDiff

using AutoDiffOperators: AutoDiffOperators, similar_onehot, with_jvp

import ADTypes
using ADTypes: AbstractADType, AutoForwardDiff

using LinearAlgebra


# DifferentiationInterface implementations seem more performant than these:
#=

struct _JacVecProdTag{F, T} end

function _dual_along(f::F, x::AbstractVector{T1}, z::AbstractVector{T2}) where {F, T1, T2}
    T =  promote_type(T1, T2)
    T_Dual = _JacVecProdTag{Core.Typeof(f), T}
    # ToDo: use `StructArrays.StructArray`? Would add StructArrays to deps.
    f(ForwardDiff.Dual{T_Dual}.(x, z))
end


function AutoDiffOperators._with_jvp_impl(f, x::AbstractVector{<:Real}, z::AbstractVector{<:Real}, ::AutoForwardDiff)
    dual_y = _dual_along(f, x, z)
    ForwardDiff.value.(dual_y), ForwardDiff.partials.(dual_y, 1)
end


function AutoDiffOperators._jvp_func_impl(f::F, x::AbstractVector{<:Real}, ad::AutoForwardDiff) where F
    return AutoDiffOperators._JVPFunc(nothing, ad, f, x)
end


function AutoDiffOperators._with_vjp_func_impl(f::F, x::AbstractVector{<:Real}, ad::AutoForwardDiff) where F
    f(x), _FwdModeVJPFunc(nothing, ad, f, x)
end

struct _FwdModeVJPFunc{P,AD<:AbstractADType,F,T<:AbstractVector{<:Number}} <: Function
    aux::P
    ad::AD
    f::F
    x::T
end

function _FwdModeVJPFunc(aux::P, ad::AD, ::Type{FT}, x::T) where {P,AD<:AbstractADType,FT,T<:AbstractVector{<:Number}}
    return _FwdModeVJPFunc{P,AD,Type{FT},T}(aux, ad, FT, x)
end

function (vjp::_FwdModeVJPFunc{Nothing})(z::AbstractVector{<:Real})
    # ToDo: Reduce memory allocation? Would require a `with_jvp!` function.
    f, x, ad = vjp.f, vjp.x, vjp.ad
    U = promote_type(eltype(f(x)), eltype(x))
    n = size(x, 1)
    result = similar(x, U)
    Base.Threads.@threads for i in eachindex(x)
        tmp = similar_onehot(x, U, n, i)
        result[i] = dot(z, with_jvp(f, x, tmp, ad)[2])
    end
    return result
end

=#

end # module AutoDiffOperatorsForwardDiffExt
