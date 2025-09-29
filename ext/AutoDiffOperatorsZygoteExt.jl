# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

module AutoDiffOperatorsZygoteExt

using Zygote

import AutoDiffOperators
import ADTypes
using ADTypes: AutoZygote


@inline AutoDiffOperators.ADSelector(::Val{:Zygote}) = AutoZygote()

AutoDiffOperators.forward_ad_selector(::ADTypes.AutoZygote) = ADTypes.AutoForwardDiff()


function AutoDiffOperators.with_jvp(f, x::AbstractVector{<:Real}, z::AbstractVector{<:Real}, ad::AutoZygote)
    fwd_ad = forward_ad_selector(ad)
    @assert !(fwd_ad isa AutoZygote)
    AutoDiffOperators.with_jvp(f, x, z, fwd_ad)
end



function AutoDiffOperators.with_vjp_func(f, x::AbstractVector{<:Real}, ::AutoZygote)
    y = f(x)
    _, pullback = Zygote.pullback(f, x)
    _with_vjp_func_result(x, y, pullback)
end

_with_vjp_func_result(::Any, y::Any, pullback) = y, only ∘ pullback

function _with_vjp_func_result(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, pullback)
    f_vjp = _ZygoteRealVecVJP(pullback, size(x, 1))
    return y, f_vjp
end

struct _ZygoteRealVecVJP <: Function
    pullback::Function
    n_x::Int
end

function (f::_ZygoteRealVecVJP)(z::AbstractVector{<:Real})
    result = similar(z, f.n_x)
    result[:] = only(f.pullback(z))
    return result
end


function AutoDiffOperators.with_jacobian(f, x::AbstractVector{<:Real}, ::Type{<:Matrix}, ad::AutoZygote)
    y = f(x)
    R = promote_type(eltype(x), eltype(y))
    n_y, n_x = length(y), length(x)
    J = similar(y, R, (n_y, n_x))
    if 8 * n_y < n_x  # Heuristic
        J[:,:] = only(Zygote.jacobian(f, x)) # Zygote.[with]jacobian is not type-stable
    else
        J[:,:] = Zygote.ForwardDiff.jacobian(f, x) # ForwardDiff.jacobian is not type-stable
    end
    f(x), J
end


end # module Zygote
