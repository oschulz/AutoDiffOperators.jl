# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).


"""
    with_gradient(f, x::AbstractVector{<:Number}, ad::ADSelector)

Returns a tuple (f(x), ∇f(x)) with the gradient ∇f(x) of `f` at `x`.

See also [`with_gradient!!(f, δx, x, ad)`](@ref) for the "maybe-in-place"
variant of this function.
"""
function with_gradient end
export with_gradient

_grad_sensitivity(y::Number) = one(y)
_grad_sensitivity(@nospecialize(::Complex)) = error("f(x) is a complex number, but with_gradient expects it to a real number")
_grad_sensitivity(@nospecialize(::T)) where T = error("f(x) is of type $(nameof(T)), but with_gradient expects it to a real number")

function with_gradient(f, x::AbstractVector{<:Number}, ad::ADSelector)
    y, vjp = with_vjp_func(f, x, ad)
    y isa Real || throw(ArgumentError("with_gradient expects f(x) to return a real number"))
    grad_f_x = vjp(_grad_sensitivity(y))
    return y, grad_f_x
end


"""
    with_gradient!!(f, δx, x::AbstractVector{<:Number}, ad::ADSelector)

Returns a tuple (f(x), ∇f(x)) with the gradient `∇f(x)`` of `f` at `x`.

`δx` may or may not be reused/overwritten and returned as `∇f(x)`.

The default implementation falls back to [`with_gradient(f, x, ad)`](@ref),
subtypes of `ADSelector` may specialized `with_gradient!!` to provide more
efficient implementations.
"""
function with_gradient!! end
export with_gradient!!

# ToDo: Copy result of with_gradient to δx if mutable, convert to same type if immutable:
function with_gradient!!(f, @nospecialize(δx::AbstractVector{<:Number}), x::AbstractVector{<:Number}, ad::ADSelector)
    return with_gradient(f, x, ad::ADSelector)
end


"""
    only_gradient(f, x::AbstractVector{<:Number}, ad::ADSelector)

Returns the gradient ∇f(x) of `f` at `x`.

See also [`with_gradient(f, x, ad)`](@ref).
"""
function only_gradient end
export only_gradient

only_gradient(f, x::AbstractVector{<:Number}, ad::ADSelector) = with_gradient(f, x, ad)[2]



struct _ValGradFunc{F,AD} <: Function
    f::F
    ad::AD
end
_ValGradFunc(::Type{FT}, ad::AD) where {FT,AD<:ADSelector}  = _ValGradFunc{Type{FT},AD}(FT, ad)

(f::_ValGradFunc)(x::AbstractVector{<:Number}) = with_gradient(f.f, x, f.ad)

"""
    valgrad_func(f, ad::ADSelector)

Returns a function `f_∇f` that calculates the value and gradient of `f`
at given points, so that `f_∇f(x)` is equivalent to
[`with_gradient(f, x, ad)`](@ref).
"""
function valgrad_func end
export valgrad_func

valgrad_func(f, ad::ADSelector) = _ValGradFunc(f, ad)



struct _GenericGradientFunc{F,AD} <: Function
    f::F
    ad::AD
end
_GenericGradientFunc(::Type{FT}, ad::AD) where {FT,AD<:ADSelector}  = _GenericGradientFunc{Type{FT},AD}(FT, ad)

(f::_GenericGradientFunc)(x::AbstractVector{<:Number}) = only_gradient(f.f, x, f.ad)

"""
    gradient_func(f, ad::ADSelector)

Returns a tuple `(f, ∇f)` with the functions `f(x)` and `∇f(x)`.
"""
function gradient_func end
export gradient_func

gradient_func(f, ad::ADSelector) = _GenericGradientFunc(f, ad)


struct _GenericGradient!Func{F,AD} <: Function
    f::F
    ad::AD
end
_GenericGradient!Func(::Type{FT}, ad::AD) where {FT,AD<:ADSelector}  = _GenericGradient!Func{Type{FT},AD}(FT, ad)

function (f!::_GenericGradient!Func)(δx::AbstractVector{<:Number}, x::AbstractVector{<:Number})
    _, δx_new = with_gradient(f!.f, x, f!.ad)
    if !(δx === δx_new)
        δx .= δx_new
    end
    return δx
end

"""
    gradient!_func(f, ad::ADSelector)

Returns a tuple `(f, ∇f!)` with the functions `f(x)` and `∇f!(δx, x)`.
"""
function gradient!_func end
export gradient!_func

gradient!_func(f, ad::ADSelector) = _GenericGradient!Func(f, ad)
