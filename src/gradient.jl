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

Fills `δx` with the gradient `∇f(x)` of `f` at `x` and returns the tuple
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

Fills `δx` with the gradient `∇f(x)` of `f` at `x` and returns it.
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
    valgrad_func(f, ad::ADSelector, dummy_x::AbstractVector{<:Real})

Returns a function `f_∇f` that calculates the value and gradient of `f`
at given points, so that `f_∇f(x)` is equivalent to
[`with_gradient(f, x, ad)`](@ref).

Passing a `dummy_x` that matches later arguments of `f_∇f` in type and size
enables backend-specific preparation. `f_∇f` is thread-safe either way.
"""
function valgrad_func end
export valgrad_func

valgrad_func(f::F, ad::ADSelector) where F = _valgrad_func_impl(f, valid_reverse_adtype(ad))
_valgrad_func_impl(f::F, ad::AbstractADType) where F = _ValGradFunc(nothing, ad, f)

function valgrad_func(f::F, ad::ADSelector, dummy_x::AbstractVector{<:Real}) where F
    return _valgrad_func_impl(f, valid_reverse_adtype(ad), dummy_x)
end

function _valgrad_func_impl(f::F, ad::AbstractADType, dummy_x::AbstractVector{<:Real}) where F
    float_x = with_floatlike_contents(dummy_x)
    return _valgrad_func_impl(_traced_array_kind(float_x), f, ad, float_x)
end

function _valgrad_func_impl(::Nothing, f::F, ad::AbstractADType, float_x::AbstractVector{<:Real}) where F
    Tx = typeof(float_x)
    Ty = _concrete_return_realtype(f, float_x)
    prep = DI.prepare_gradient(f, ad, float_x)
    aux = _DIPrep(_borrowable_object(_CacheLikeUse(), prep))
    f_∇f = _ValGradFunc(aux, ad, f)
    return _WrappedFunction{Tuple{Ty,Tx},Tx}(f_∇f)
end

_valgrad_func_impl(::Val, f::F, ad::AbstractADType, ::AbstractVector{<:Real}) where F = _ValGradFunc(nothing, ad, f)

struct _ValGradFunc{P,AD<:AbstractADType,F} <: Function
    aux::P
    ad::AD
    f::F
end

function _ValGradFunc(aux::P, ad::AD, ::Type{FT}) where {P,AD<:AbstractADType,FT}
    return _ValGradFunc{P,AD,Type{FT}}(aux, ad, FT)
end

(f::_ValGradFunc{Nothing})(x::AbstractVector{<:Number}) = with_gradient(f.f, x, f.ad)

function (f_∇f::_ValGradFunc{<:_DIPrep})(x::AbstractVector{<:Number})
    f = f_∇f.f
    float_x = with_floatlike_contents(x)
    prep = f_∇f.aux.prep
    f_x, δx = @_borrow_maybewrite prep begin
        DI.value_and_gradient(f, prep, f_∇f.ad, float_x)
    end
    return f_x, _oftype(float_x, δx)
end


"""
    gradient_func(f, ad::ADSelector)
    gradient_func(f, ad::ADSelector, dummy_x::AbstractVector{<:Real})

Returns a function `∇f` that calculates the gradient of `f` at a given
point `x`, so that `∇f(x)` is equivalent to [`only_gradient(f, x, ad)`](@ref).

Passing a `dummy_x` that matches later arguments of `∇f` in type and size
enables backend-specific preparation. `∇f` is thread-safe either way.
"""
function gradient_func end
export gradient_func

gradient_func(f::F, ad::ADSelector) where F = _gradient_func_impl(f, valid_reverse_adtype(ad))
_gradient_func_impl(f::F, ad::AbstractADType) where F = _GradOnlyFunc(nothing, ad, f)

function gradient_func(f::F, ad::ADSelector, dummy_x::AbstractVector{<:Real}) where F
    return _gradient_func_impl(f, valid_reverse_adtype(ad), dummy_x)
end

function _gradient_func_impl(f::F, ad::AbstractADType, dummy_x::AbstractVector{<:Real}) where F
    float_x = with_floatlike_contents(dummy_x)
    return _gradient_func_impl(_traced_array_kind(float_x), f, ad, float_x)
end

function _gradient_func_impl(::Nothing, f::F, ad::AbstractADType, float_x::AbstractVector{<:Real}) where F
    Tx = typeof(float_x)
    prep = DI.prepare_gradient(f, ad, float_x)
    aux = _DIPrep(_borrowable_object(_CacheLikeUse(), prep))
    ∇f = _GradOnlyFunc(aux, ad, f)
    return _WrappedFunction{Tx,Tx}(∇f)
end

_gradient_func_impl(::Val, f::F, ad::AbstractADType, ::AbstractVector{<:Real}) where F = _GradOnlyFunc(nothing, ad, f)

struct _GradOnlyFunc{P,AD<:AbstractADType,F} <: Function
    aux::P
    ad::AD
    f::F
end

function _GradOnlyFunc(aux::P, ad::AD, ::Type{FT}) where {P,AD<:AbstractADType,FT}
    return _GradOnlyFunc{P,AD,Type{FT}}(aux, ad, FT)
end

(f::_GradOnlyFunc{Nothing})(x::AbstractVector{<:Number}) = only_gradient(f.f, x, f.ad)

function (∇f::_GradOnlyFunc{<:_DIPrep})(x::AbstractVector{<:Number})
    f = ∇f.f
    float_x = with_floatlike_contents(x)
    prep = ∇f.aux.prep
    δx = @_borrow_maybewrite prep begin
        DI.gradient(f, prep, ∇f.ad, float_x)
    end
    return _oftype(float_x, δx)
end
