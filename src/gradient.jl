# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).


"""
    with_gradient(f, x::AbstractVector{<:Number}, ad::ADSelector)

Returns a tuple (f(x), ∇f(x)) with the gradient ∇f(x) of `f` at `x`.

See also [`with_gradient!!(f, δx, x, ad)`](@ref) for the "maybe-in-place"
variant of this function.
"""
function with_gradient end
export with_gradient

function with_gradient(f::F, x::AbstractVector{<:Real}, ad::ADSelector) where F
    ad_rev = valid_reverse_adtype(ad)
    _with_gradient_impl(f, x, ad_rev)
end

function _with_gradient_impl(f::F, x::AbstractVector{<:Real}, ad::AbstractADType) where F
    float_x = with_floatlike_contents(x)
    T_f_x = _primal_return_type(f, float_x)
    f_x, δx = DI.value_and_gradient(f, ad, float_x)
    return convert(T_f_x, f_x)::T_f_x, _oftype(float_x, δx)
end


"""
    with_gradient!(f, δx, x::AbstractVector{<:Number}, ad::ADSelector)

Fills `δx` with the the gradient `∇f(x)` of `f` at `x` and returns the tuple
`(f(x), δx)`.
"""
function with_gradient! end
export with_gradient!

function with_gradient!(f::F, δx::AbstractVector{<:Real}, x::AbstractVector{<:Real}, ad::ADSelector) where F
    ad_rev = valid_reverse_adtype(ad)
    _with_gradient_impl!(f, δx, x, ad_rev)
end

function _with_gradient_impl!(f::F, δx::AbstractVector{<:Real}, x::AbstractVector{<:Real}, ad::AbstractADType) where F
    float_x = with_floatlike_contents(x)
    T_f_x = _primal_return_type(f, float_x)
    f_x, _ = DI.value_and_gradient!(f, δx, ad, float_x)
    return convert(T_f_x, f_x)::T_f_x, δx
end


"""
    with_gradient!!(f, δx, x::AbstractVector{<:Number}, ad::ADSelector)

Returns a tuple (f(x), ∇f(x)) with the gradient `∇f(x)` of `f` at `x`.

`δx` may or may not be reused/overwritten and returned as `∇f(x)`.
"""
function with_gradient!! end
export with_gradient!!

function with_gradient!!(f::F, δx::AbstractVector{<:Real}, x::AbstractVector{<:Real}, ad::ADSelector) where F
    ad_rev = valid_reverse_adtype(ad)
    return _with_gradient_impl!!(_is_immutable_type(typeof(δx)), f, δx, x, ad_rev)
end

function _with_gradient_impl!!(::Val{true}, f::F, ::AbstractVector{<:Real}, x::AbstractVector{<:Real}, ad::AbstractADType) where F
    return with_gradient(f, x, ad)
end

function _with_gradient_impl!!(::Val{false}, f::F, δx::AbstractVector{<:Real}, x::AbstractVector{<:Real}, ad::AbstractADType) where F
    return with_gradient!(f, δx, x, ad)
end


"""
    only_gradient(f, x::AbstractVector{<:Number}, ad::ADSelector)

Returns the gradient ∇f(x) of `f` at `x`.

See also [`with_gradient(f, x, ad)`](@ref).
"""
function only_gradient end
export only_gradient

function only_gradient(f::F, x::AbstractVector{<:Real}, ad::ADSelector) where F
    ad_rev = valid_reverse_adtype(ad)
    _only_gradient_impl(f, x, ad_rev)
end

function _only_gradient_impl(f::F, x::AbstractVector{<:Real}, ad::AbstractADType) where F
    float_x = with_floatlike_contents(x)
    δx = DI.gradient(f, ad, float_x)
    return _oftype(float_x, δx)
end


"""
    only_gradient!(f, δx, x::AbstractVector{<:Number}, ad::ADSelector)

Fills δx with the `∇f(x)` of `f` at `x` and returns it.
"""
function only_gradient! end
export only_gradient!

function only_gradient!(f::F, δx::AbstractVector{<:Real}, x::AbstractVector{<:Real}, ad::ADSelector) where F
    ad_rev = valid_reverse_adtype(ad)
    _only_gradient_impl!(f, δx, x, ad_rev)
end

function _only_gradient_impl!(f::F, δx::AbstractVector{<:Real}, x::AbstractVector{<:Real}, ad::AbstractADType) where F
    float_x = with_floatlike_contents(x)
    DI.gradient!(f, δx, ad, float_x)
    return δx
end


"""
    only_gradient!!(f, δx, x::AbstractVector{<:Number}, ad::ADSelector)

Returns the gradient `∇f(x)` of `f` at `x`.

`δx` may or may not be reused/overwritten and returned as `∇f(x)`.
"""
function only_gradient!! end
export only_gradient!!

function only_gradient!!(f::F, δx::AbstractVector{<:Real}, x::AbstractVector{<:Real}, ad::ADSelector) where F
    ad_rev = valid_reverse_adtype(ad)
    return _only_gradient_impl!!(_is_immutable_type(typeof(δx)), f, δx, x, ad_rev)
end

function _only_gradient_impl!!(::Val{true}, f::F, ::AbstractVector{<:Real}, x::AbstractVector{<:Real}, ad::AbstractADType) where F
    return only_gradient(f, x, ad)
end

function _only_gradient_impl!!(::Val{false}, f::F, δx::AbstractVector{<:Real}, x::AbstractVector{<:Real}, ad::AbstractADType) where F
    return only_gradient!(f, δx, x, ad)
end


"""
    valgrad_func(f, ad::ADSelector)

Returns a function `f_∇f` that calculates the value and gradient of `f`
at given points, so that `f_∇f(x)` is equivalent to
[`with_gradient(f, x, ad)`](@ref).
"""
function valgrad_func end
export valgrad_func

valgrad_func(f::F, ad::ADSelector) where F = _valgrad_func_impl(f, valid_reverse_adtype(ad))
_valgrad_func_impl(f::F, ad::AbstractADType) where F = _ValGradFunc(nothing, ad, f)

# ToDo: valgrad_func(f, ad::AbstractADType, dummy_x::AbstractVector{<:Number}) with aux

struct _ValGradFunc{P,AD<:AbstractADType,F} <: Function
    aux::P
    ad::AD
    f::F
end

function _ValGradFunc(aux::P, ad::AD, ::Type{FT}) where {P,AD<:AbstractADType,FT}
    return _ValGradFunc{P,AD,Type{FT}}(aux, ad, FT)
end

(f::_ValGradFunc{Nothing})(x::AbstractVector{<:Number}) = with_gradient(f.f, x, f.ad)


"""
    gradient_func(f, ad::ADSelector)

Returns a function `∇f` that calculates the gradient of `f` at a given
point `x`, so that `∇f(x)` is equivalent to [`only_gradient(f, x, ad)`](@ref).
"""
function gradient_func end
export gradient_func

gradient_func(f::F, ad::ADSelector) where F = _gradient_func_impl(f, valid_reverse_adtype(ad))
_gradient_func_impl(f::F, ad::AbstractADType) where F = _GradOnlyFunc(nothing, ad, f)

# ToDo: gradient_func(f, ad::AbstractADType, dummy_x::AbstractVector{<:Number}) with DI prep

struct _GradOnlyFunc{P,AD<:AbstractADType,F} <: Function
    aux::P
    ad::AD
    f::F
end

function _GradOnlyFunc(aux::P, ad::AD, ::Type{FT}) where {P,AD<:AbstractADType,FT}
    return _GradOnlyFunc{P,AD,Type{FT}}(aux, ad, FT)
end

(f::_GradOnlyFunc{Nothing})(x::AbstractVector{<:Number}) = only_gradient(f.f, x, f.ad)


"""
    gradient!_func(f, ad::ADSelector)

Returns a function `∇f!` that fills a given vector `δx` with gradient of `f`
at a given point `x`, so that `∇f!(δx, x)` is equivalent to
[`only_gradient!(f, δx, x, ad)`](@ref).
"""
function gradient!_func end
export gradient!_func

gradient!_func(f::F, ad::ADSelector) where F = _gradient!_func_impl(f, valid_reverse_adtype(ad))
_gradient!_func_impl(f::F, ad::AbstractADType) where F = _GradOnly!Func(nothing, ad, f)

# ToDo: gradient!_func(f, ad::AbstractADType, dummy_x::AbstractVector{<:Number}) with DI prep

struct _GradOnly!Func{P,AD<:AbstractADType,F} <: Function
    aux::P
    ad::AD
    f::F
end

function _GradOnly!Func(aux::P, ad::AD, ::Type{FT}) where {P,AD<:AbstractADType,FT}
    return _GradOnly!Func{P,AD,Type{FT}}(aux, ad, FT)
end

function (f!::_GradOnly!Func{Nothing})(δx::AbstractVector{<:Number}, x::AbstractVector{<:Number})
    float_x = with_floatlike_contents(x)
    only_gradient!(f!.f, δx, float_x, f!.ad)
    return δx
end
