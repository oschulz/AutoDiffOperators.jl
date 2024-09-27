# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

module AutoDiffOperatorsEnzymeExt

using Enzyme

import AutoDiffOperators
import ADTypes
using ADTypes: AutoEnzyme

const _enzyme_v0_13 = isdefined(Enzyme, :ForwardWithPrimal)


@inline AutoDiffOperators.ADSelector(::Val{:Enzyme}) = AutoEnzyme()


function AutoDiffOperators.with_gradient(f, x::AbstractVector{<:Real}, ad::AutoEnzyme)
    δx = similar(x, float(eltype(x)))
    AutoDiffOperators.with_gradient!!(f, δx, x, ad)
end

function AutoDiffOperators.with_gradient!!(f, δx::AbstractVector{<:Real}, x::AbstractVector{<:Real}, ::AutoEnzyme)
    fill!(δx, zero(eltype(x)))
    _, y = autodiff(ReverseWithPrimal, f, Active, Duplicated(x, δx))
    y, δx
end


@static if _enzyme_v0_13
    function AutoDiffOperators.with_jvp(f, x::AbstractVector{<:Real}, z::AbstractVector{<:Real}, ::AutoEnzyme)
        J_z, f_x = autodiff(ForwardWithPrimal, f, Duplicated(x, z))
        return f_x, J_z
    end
else
    # Enzyme v0.12
    function AutoDiffOperators.with_jvp(f, x::AbstractVector{<:Real}, z::AbstractVector{<:Real}, ::AutoEnzyme)
        f_x, J_z = autodiff(Forward, f, Duplicated, Duplicated(x, z))
        return f_x, J_z
    end
end

# ToDo: Broadcast specialization of `with_jvp` for multiple z-values using `Enzyme.BatchDuplicated`.


#=

# How do to this in a thread-safe fashion?

struct _Enzyme_VJP_WithTape{REV,CF<:Const,TP,TX<:AbstractVector{<:Real},DX<:AbstractVector{<:Real},DY<:AbstractVector{<:Real}} <: Function
    reverse::REV
    cf::CF
    tape::TP
    x::TX
    δx::DX
    δy::DY
end

function (vjp::_Enzyme_VJP_WithTape)(z)
    fill!(vjp.δx, zero(eltype(vjp.δx)))
    vjp.δy .= z
    vjp.reverse(vjp.cf, Duplicated(vjp.x, vjp.δx), vjp.tape)
    deepcopy(vjp.δx)
end

function AutoDiffOperators.with_vjp_func(f, x::AbstractVector{<:Real}, ::AutoEnzyme)
    δx = similar(x, float(eltype(x)))
    fill!(δx, zero(eltype(δx)))
    forward, reverse = autodiff_thunk(ReverseSplitWithPrimal, Const{Core.Typeof(f)}, Duplicated, Duplicated{Core.Typeof(δx)})
    tape, y, δy  = forward(Const(f), Duplicated(x, δx))
    return y, _Enzyme_VJP_WithTape(reverse, Const(f), tape, x, δx, δy)
end

=#


struct _Mutating_Func{F} <: Function
    f::F
end

function (mf!::_Mutating_Func)(y, x)
    y .= mf!.f(x)
    return nothing
end


struct _Enzyme_VJP_NoTape{MF<:_Mutating_Func,T,U} <: Function
    mf!::MF
    x::T
    y::U
end

function (vjp::_Enzyme_VJP_NoTape)(z)
    δx = similar(vjp.x, float(eltype(vjp.x)))
    fill!(δx, zero(eltype(δx)))

    y = similar(vjp.y)
    δy = deepcopy(z)

    Enzyme.autodiff(Reverse, Const(vjp.mf!), Duplicated(y, δy), Duplicated(vjp.x, δx))

    return δx
end

function AutoDiffOperators.with_vjp_func(f, x::AbstractVector{<:Real}, ::AutoEnzyme)
    y = f(x)
    mf! = _Mutating_Func{Core.Typeof(f)}(f)
    return y, _Enzyme_VJP_NoTape(mf!, x, y)
end


# ToDo: Broadcast specialization of functions returned by `with_vjp_func` for multiple z-values.

function AutoDiffOperators.with_jacobian(f, x::AbstractVector{<:Real}, ::Type{<:Matrix}, ad::AutoEnzyme)
    y = f(x)
    R = promote_type(eltype(x), eltype(y))
    n_y, n_x = length(y), length(x)
    # Enzyme.jacobian is not type-stable:
    J = similar(y, R, (n_y, n_x))
    if 4 * n_y < n_x && n_y <= 8  # Heuristic
        @static if _enzyme_v0_13
            J[:,:] = Enzyme.jacobian(Enzyme.Reverse, f, x, n_outs = Val(n_y))[1]
        else # Enzyme v0.12
            J[:,:] = Enzyme.jacobian(Enzyme.Reverse, f, x, Val(n_y))
        end
    else
        @static if _enzyme_v0_13
            J[:,:] = Enzyme.jacobian(Enzyme.Forward, f, x)[1]
        else # Enzyme v0.12
            J[:,:] = Enzyme.jacobian(Enzyme.Forward, f, x)
        end
    end
    f(x), J
end


end # module Enzyme
