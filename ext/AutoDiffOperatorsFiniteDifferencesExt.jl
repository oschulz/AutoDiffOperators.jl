# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

module AutoDiffOperatorsFiniteDifferencesExt

@static if isdefined(Base, :get_extension)
    using FiniteDifferences
else
    using ..FiniteDifferences
end

import AutoDiffOperators
import AbstractDifferentiation
import ADTypes


Base.Module(::AutoDiffOperators.ADModule{:FiniteDifferences}) = FiniteDifferences

const FiniteDifferencesAD = Union{
    AbstractDifferentiation.FiniteDifferencesBackend,
    AutoDiffOperators.ADModule{:FiniteDifferences}
}


AutoDiffOperators.supports_structargs(::FiniteDifferencesAD) = true


# ADTypes doesn't have a backend for FiniteDifferences yet.

function AutoDiffOperators.convert_ad(::Type{AbstractDifferentiation.AbstractBackend}, ::AutoDiffOperators.ADModule{:FiniteDifferences})
    AbstractDifferentiation.FiniteDifferencesBackend(default_method)
end

function AutoDiffOperators.convert_ad(::Type{AutoDiffOperators.ADModule}, ad::AbstractDifferentiation.FiniteDifferencesBackend)
    AutoDiffOperators.ADModule{:FiniteDifferences}()
end


const default_method = FiniteDifferences.central_fdm(5, 1)
_get_method(ad::AbstractDifferentiation.FiniteDifferencesBackend) = ad.method
_get_method(::AutoDiffOperators.ADModule{:FiniteDifferences}) = default_method


function AutoDiffOperators.with_gradient(f, x, ad::FiniteDifferencesAD)
    f(x), only(FiniteDifferences.grad(_get_method(ad), f, x))
end


function AutoDiffOperators.with_jvp(f, x, z, ad::FiniteDifferencesAD)
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

function AutoDiffOperators.with_vjp_func(f, x, ad::FiniteDifferencesAD)
    f(x), _FiniteDifferencesVJPFunc(f, x, _get_method(ad))
end


function AutoDiffOperators.with_jacobian(f, x::AbstractVector{<:Real}, ::Type{<:Matrix}, ad::FiniteDifferencesAD)
    f(x), only(FiniteDifferences.jacobian(_get_method(ad), f, x))
end


end # module FiniteDifferences
