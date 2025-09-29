# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

module AutoDiffOperatorsFiniteDifferencesExt

using FiniteDifferences

import AutoDiffOperators
import ADTypes
using ADTypes: AutoFiniteDifferences


const default_method = FiniteDifferences.central_fdm(5, 1)
@inline AutoDiffOperators.ADSelector(::Val{:FiniteDifferences}) = AutoFiniteDifferences(fdm = default_method)


_get_method(ad::AutoFiniteDifferences) = ad.fdm


function AutoDiffOperators.with_gradient(f, x::AbstractVector{<:Real}, ad::AutoFiniteDifferences)
    f(x), only(FiniteDifferences.grad(_get_method(ad), f, x))
end


function AutoDiffOperators.with_jvp(f, x::AbstractVector{<:Real}, z::AbstractVector{<:Real}, ad::AutoFiniteDifferences)
    f(x), FiniteDifferences.jvp(_get_method(ad), f, (x, z))
end


struct _FiniteDifferencesVJPFunc{F,T,M<:FiniteDifferenceMethod} <: Function
    f::F
    x::T
    method::M
end
_FiniteDifferencesVJPFunc(::Type{FT}, x::T, method::M) where {FT,T,M} = _FwdModeVJPFunc{Type{FT},T,M}(FT, x, method)

function (vjp::_FiniteDifferencesVJPFunc)(z)
    only(FiniteDifferences.j′vp(vjp.method, vjp.f, z, vjp.x))
end

function AutoDiffOperators.with_vjp_func(f, x::AbstractVector{<:Real}, ad::AutoFiniteDifferences)
    f(x), _FiniteDifferencesVJPFunc(f, x, _get_method(ad))
end


function AutoDiffOperators.with_jacobian(f, x::AbstractVector{<:Real}, ::Type{<:Matrix}, ad::AutoFiniteDifferences)
    f(x), only(FiniteDifferences.jacobian(_get_method(ad), f, x))
end


end # module FiniteDifferences
