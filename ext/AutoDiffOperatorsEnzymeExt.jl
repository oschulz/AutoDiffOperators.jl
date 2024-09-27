# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

module AutoDiffOperatorsEnzymeExt

using Enzyme

import AutoDiffOperators
import AbstractDifferentiation
import ADTypes


Base.Module(::AutoDiffOperators.ADModule{:Enzyme}) = Enzyme

const EnzymeAD = Union{
    ADTypes.AutoEnzyme,
    AutoDiffOperators.ADModule{:Enzyme}
}


# ToDo: Support structured arguments (since Enzyme itself is capable of it).
AutoDiffOperators.supports_structargs(::EnzymeAD) = false

# ToDo: Add custom selector for Enzyme that contains chunk size, etc.

# No AbstractDifferentiation doesn't have a backend for Enzyme yet.

function AutoDiffOperators.convert_ad(::Type{ADTypes.AbstractADType}, ::AutoDiffOperators.ADModule{:Enzyme})
    ADTypes.AutoEnzyme()
end

function AutoDiffOperators.convert_ad(::Type{AutoDiffOperators.ADModule}, ad::ADTypes.AutoEnzyme)
    AutoDiffOperators.ADModule{:Enzyme}()
end


function AutoDiffOperators.with_gradient(f, x::AbstractVector{<:Real}, ad::EnzymeAD)
    δx = similar(x, float(eltype(x)))
    AutoDiffOperators.with_gradient!!(f, δx, x, ad)
end

function AutoDiffOperators.with_gradient!!(f, δx::AbstractVector{<:Real}, x::AbstractVector{<:Real}, ::EnzymeAD)
    fill!(δx, zero(eltype(x)))
    _, y = autodiff(ReverseWithPrimal, f, Duplicated(x, δx))
    y, δx
end


function AutoDiffOperators.with_jvp(f, x::AbstractVector{<:Real}, z::AbstractVector{<:Real}, ::EnzymeAD)
    f_x, J_z = autodiff(Forward, f, Duplicated, Duplicated(x, z))
    return f_x, J_z
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

function AutoDiffOperators.with_vjp_func(f, x::AbstractVector{<:Real}, ::EnzymeAD)
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

function AutoDiffOperators.with_vjp_func(f, x::AbstractVector{<:Real}, ::EnzymeAD)
    y = f(x)
    mf! = _Mutating_Func{Core.Typeof(f)}(f)
    return y, _Enzyme_VJP_NoTape(mf!, x, y)
end


# ToDo: Broadcast specialization of functions returned by `with_vjp_func` for multiple z-values.

@static if VERSION >= v"1.9"
    # Causes crashes on (at least) Julia v1.6:
    function AutoDiffOperators.with_jacobian(f, x::AbstractVector{<:Real}, ::Type{<:Matrix}, ad::EnzymeAD)
        y = f(x)
        R = promote_type(eltype(x), eltype(y))
        n_y, n_x = length(y), length(x)
        J = similar(y, R, (n_y, n_x))
        if 4 * n_y < n_x && n_y <= 8  # Heuristic
            J[:,:] = Enzyme.jacobian(Enzyme.Reverse, f, x, Val(n_y)) # Enzyme.jacobian is not type-stable
        else
            J[:,:] = Enzyme.jacobian(Enzyme.Forward, f, x) # Enzyme.jacobian is not type-stable
        end
        f(x), J
    end
end


end # module Enzyme
