# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

module AutoDiffOperatorsDifferentiationInterfaceExt

import DifferentiationInterface as DI

import AutoDiffOperators
using AutoDiffOperators: DiffIfAD

using AutoDiffOperators: with_floatlike_contents, forward_ad_selector, reverse_ad_selector

const _di_v0_6 = isdefined(DI, :Context)


_get_fwd_backend(ad::DiffIfAD) = forward_ad_selector(ad.backend)
_get_rev_backend(ad::DiffIfAD) = reverse_ad_selector(ad.backend)


function AutoDiffOperators.with_gradient(f, x::AbstractVector{<:Real}, ad::DiffIfAD)
    float_x = with_floatlike_contents(x)
    f_x, δx = DI.value_and_gradient(f, _get_rev_backend(ad), float_x)
    return f_x, oftype(float_x, δx)
end

function AutoDiffOperators.with_gradient!!(f, δx::AbstractVector{<:Real}, x::AbstractVector{<:Real}, ad::DiffIfAD)
    float_x = with_floatlike_contents(x)
    f_x, r_δx = DI.value_and_gradient!(f, δx, _get_rev_backend(ad), float_x)
    return f_x, oftype(δx, r_δx)
end


function AutoDiffOperators.only_gradient(f, x::AbstractVector{<:Real}, ad::DiffIfAD)
    float_x = with_floatlike_contents(x)
    δx = DI.gradient(f, _get_rev_backend(ad), float_x)
    return oftype(float_x, δx)
end


function AutoDiffOperators.with_jvp(f, x::AbstractVector{<:Real}, z::AbstractVector{<:Real}, ad::DiffIfAD)
    float_x = with_floatlike_contents(x)
    float_z = with_floatlike_contents(z)
    f_x, J_z = DI.value_and_pushforward(f, _get_fwd_backend(ad), float_x, float_z)
    return f_x, oftype(float_z, J_z)
end


function AutoDiffOperators.jvp_func(f, x::AbstractVector{<:Real}, ad::DiffIfAD)
    float_x = with_floatlike_contents(x)
    backend = _get_fwd_backend(ad)

    prep = @static if _di_v0_6
        DI.prepare_pushforward_same_point(f, backend, float_x, (float_x,))
    else # DifferentiationInterface v0.5
        DI.prepare_pushforward_same_point(f, backend, float_x, float_x)
    end

    f_vjp = _DIJVPFunc{Core.Typeof(f),typeof(prep),typeof(float_x),typeof(backend)}(f, prep, float_x, backend)
    return f_vjp
end

struct _DIJVPFunc{F,P,T,AD} <: Function
    f::F
    prep::P
    x::T
    backend::AD
end

@static if _di_v0_6
    function (f_jvp::_DIJVPFunc)(z::AbstractVector{<:Real})
        float_z = with_floatlike_contents(z)
        J_z = DI.pushforward(f_jvp.f, f_jvp.prep, f_jvp.backend, f_jvp.x, (float_z,))[1]
        return oftype(float_z, J_z)
    end
else
    # DifferentiationInterface v0.5
    function (f_jvp::_DIJVPFunc)(z::AbstractVector{<:Real})
        float_z = with_floatlike_contents(z)
        J_z = DI.pushforward(f_jvp.f, f_jvp.backend, f_jvp.x, float_z, f_jvp.prep)
        return oftype(float_z, J_z)
    end
end



function AutoDiffOperators.with_vjp_func(f, x::AbstractVector{<:Real}, ad::DiffIfAD)
    float_x = with_floatlike_contents(x)
    # ToDo: Avoid duplicate calculation of f(x)
    f_x = f(float_x)
    backend = _get_rev_backend(ad)

    prep = @static if _di_v0_6
        DI.prepare_pullback_same_point(f, backend, float_x, (f_x,))
    else # DifferentiationInterface v0.5
        DI.prepare_pullback_same_point(f, backend, float_x, f_x)
    end
    
    f_vjp = _DIVJPFunc{Core.Typeof(f),typeof(prep),typeof(float_x),typeof(backend)}(f, prep, float_x, backend)
    return f_x, f_vjp
end

struct _DIVJPFunc{F,P,T,AD} <: Function
    f::F
    prep::P
    x::T
    backend::AD
end

@static if _di_v0_6
    function (f_vjp::_DIVJPFunc)(z::AbstractVector{<:Real})
        float_z = with_floatlike_contents(z)
        z_J = DI.pullback(f_vjp.f, f_vjp.prep, f_vjp.backend, f_vjp.x, (float_z,))[1]
        return oftype(float_z, z_J)
    end
else # DifferentiationInterface v0.5
    function (f_vjp::_DIVJPFunc)(z::AbstractVector{<:Real})
        float_z = with_floatlike_contents(z)
        z_J = DI.pullback(f_vjp.f, f_vjp.backend, f_vjp.x, float_z, f_vjp.prep)
        return oftype(float_z, z_J)
    end
end


function AutoDiffOperators.with_jacobian(f, x::AbstractVector{<:Real}, ::Type{<:Matrix}, ad::DiffIfAD)
    float_x = with_floatlike_contents(x)
    # ToDo: Use heuristic to choose forward or reverse mode based on size of x and f_x?
    f_x, J = DI.value_and_jacobian(f, _get_fwd_backend(ad), float_x)
    # Enforce type stability:
    typed_J = similar(float_x, (size(f_x)..., size(float_x)...))
    typed_J .= J
    return f_x, typed_J
end


end # module AutoDiffOperatorsDifferentiationInterfaceExt
