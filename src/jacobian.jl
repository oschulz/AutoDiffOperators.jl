# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).


"""
    with_jacobian(f, x::AbstractVector{<:Number}, OP, ad::ADSelector)

Returns a tuple `(f(x), J)` with a multiplicative Jacobian operator `J`
of type `OP`.

Example:

```julia
using AutoDiffOperators, LinearMaps
y, J = with_jacobian(f, x, LinearMap, ad)
y == f(x)
_, J_explicit = with_jacobian(f, x, DenseMatrix, ad)
J * z_r ≈ J_explicit * z_r
z_l' * J ≈ z_l' * J_explicit
```

`OP` may be
[`LinearMaps.LinearMap`](https://github.com/JuliaLinearAlgebra/LinearMaps.jl)
(resp. `LinearMaps.FunctionMap`) or `Matrix`. Other operator types can be
supported by specializing
[`mulfunc_operator`](@ref) for the operator type.

The default implementation of `with_jacobian` uses
[`jvp_func`](@ref) and [`with_vjp_func`](@ref) to implement (adjoint)
multiplication of `J` with (adjoint) vectors.
"""
function with_jacobian end
export with_jacobian


function with_jacobian(f::F, x::AbstractVector{T}, ::Type{OP}, ad::ADSelector) where {F,T<:Real,OP}
    ad_fwd = forward_adtype(ad)
    ad_rev = reverse_adtype(ad)
    f_jvp = _maybe_jvp_func(ad_fwd, f, x, ad)
    y, f_vjp = _maybe_with_vjp_func(ad_rev, f, x, ad)
    sz = Dims((size(y,1), size(x,1)))
    J = mulfunc_operator(OP, T, sz, f_jvp, f_vjp, Val(false), Val(false), Val(false))
    return y, J
end

_maybe_jvp_func(ad_fwd::AbstractADType, f::F, x, ::ADSelector) where F = jvp_func(f, x, ad_fwd)
_maybe_jvp_func(::NoAutoDiff, f, x, ::Type{AD}) where {AD<:ADSelector} = _NoJVPFunc{AD}()

struct _NoJVPFunc{AD<:ADSelector} <: Function end
function (::_NoJVPFunc{AD})(::AbstractVector{<:Number}) where AD
    throw(ErrorException("No forward-mode automatic differentiation available for AD-selector $(nameof(AD)), can't compute Jacobian * vector products"))
end

_maybe_with_vjp_func(ad_rev::AbstractADType, f::F, x, ::ADSelector) where F = with_vjp_func(f, x, ad_rev)
_maybe_with_vjp_func(::NoAutoDiff, f, x, ::Type{AD}) where {AD<:ADSelector} = f(x), _NoVJPFunc{AD}()

struct _NoVJPFunc{AD<:ADSelector} <: Function end
function (::_NoVJPFunc{AD})(::AbstractVector{<:Number}) where AD
    throw(ErrorException("No reverse-mode automatic differentiation available for AD-selector $(nameof(AD)), can't compute vector * Jacobian product"))
end


function with_jacobian(f::F, x::AbstractVector{<:Real}, ::Type{<:DenseMatrix}, ad::ADSelector) where F
    return _with_jacobian_matrix(f, x, ad)
end

function _with_jacobian_matrix(f::F, x::AbstractVector{<:Real}, ad::ADSelector) where F
    ad_fwd = valid_forward_adtype(ad)
    return _with_jacobian_matrix_impl(f, x, ad_fwd)
end

function _with_jacobian_matrix_impl(f::F, x::AbstractVector{<:Real}, ad::AbstractADType) where F
    float_x = with_floatlike_contents(x)
    T_x = typeof(float_x)
    T_f_x = _concrete_return_vector_type(f, float_x)
    T_J = _matrix_type(T_f_x, T_x)
    f_x, J = DI.value_and_jacobian(f, ad, float_x)
    return convert(T_f_x, f_x)::T_f_x, convert(T_J, J)::T_J
end


"""
    with_jvp(f, x::AbstractVector{<:Number}, z::AbstractVector{<:Number}, ad::ADSelector)

Returns a tuple `(f(x), J * z)`.
"""
function with_jvp end
export with_jvp

function with_jvp(f::F, x::AbstractVector{<:Real}, z::AbstractVector{<:Real}, ad::ADSelector) where F
    ad_fwd = valid_forward_adtype(ad)
    return _with_jvp_impl(f, x, z, ad_fwd)
end

function _with_jvp_impl(f::F, x::AbstractVector{<:Real}, z::AbstractVector{<:Real}, ad::AbstractADType) where F
    float_x = with_floatlike_contents(x)
    float_z = with_floatlike_contents(z)
    T_z = typeof(float_z)
    T_f_x = _primal_return_type(f, float_x)
    T_J_z = _similar_type(T_z, T_f_x)
    f_x, J_zs = DI.value_and_pushforward(f, ad, float_x, (float_z,))
    return convert(T_f_x, f_x)::T_f_x, convert(T_J_z, only(J_zs))::T_J_z
end


# ToDo: add `with_jvp!(f, Jz, x, z, ad::ADSelector)`?


"""
    jvp_func(f, x::AbstractVector{<:Number}, ad::ADSelector)

Returns a function `jvp` with `jvp(z) == J * z`.
"""
function jvp_func end
export jvp_func

function jvp_func(f::F, x::AbstractVector{<:Real}, ad::ADSelector) where F
    ad_fwd = valid_forward_adtype(ad)
    return _jvp_func_impl(f, x, ad_fwd)
end

function _jvp_func_impl(f::F, x::AbstractVector{<:Real}, ad::AbstractADType) where F
    float_x = with_floatlike_contents(x)
    Tx = typeof(float_x)
    Ty = _concrete_return_vector_type(f, float_x)
    prep = DI.prepare_pushforward_same_point(f, ad, float_x, (float_x,))
    aux = _DIPrep(_borrowable_object(_CacheLikeUse(), prep))
    f_jvp = _JVPFunc(aux, ad, f, float_x, Ty)
    wrapped_f_jvp = _WrappedFunction{Ty,Tx}(f_jvp)
    return wrapped_f_jvp
end

struct _JVPFunc{P,AD<:AbstractADType,F,Tx<:AbstractVector{<:Number},Ty<:AbstractVector{<:Number}} <: Function
    aux::P
    ad::AD
    f::F
    x::Tx
end
function _JVPFunc(aux::P, ad::AD, f::F, x::Tx, ::Type{Ty}) where {P,AD,F,Tx,Ty}
    return _JVPFunc{P,AD,F,Tx,Ty}(aux, ad, f, x)
end
function _JVPFunc(aux::P, ad::AD, ::Type{FT}, x::Tx, ::Type{Ty}) where {P,AD,FT,Tx,Ty}
    return _JVPFunc{P,AD,Type{FT},Tx,Ty}(aux, ad, FT, x)
end

function (f_jvp::_JVPFunc{<:_DIPrep,AD,F,Tx,Ty})(z::AbstractVector{<:Real}) where {AD,F,Tx,Ty}
    @assert !any(isnan, z)
    prep = f_jvp.aux.prep
    prep_instance, prep_handle = _borrow_maybewrite(prep)
    try
        f = f_jvp.f
        float_x = f_jvp.x
        float_z = convert(Tx, z)::Tx
        J_z = only(DI.pushforward(f, prep_instance, f_jvp.ad, float_x, (float_z,)))
        @assert !any(isnan, J_z)
        return convert(Ty, J_z)::Ty
    finally
        _return_borrowed(prep, prep_instance, prep_handle)
    end
end


"""
    with_vjp_func(f, x::AbstractVector{<:Number}, ad::ADSelector)

Returns a tuple `(f(x), vjp)` with the function `vjp(z) ≈ J' * z`.
"""
function with_vjp_func end
export with_vjp_func

function with_vjp_func(f::F, x::AbstractVector{<:Real}, ad::ADSelector) where F
    ad_rev = reverse_adtype(ad)
    return _with_vjp_func_impl(f, x, ad_rev)
end

function _with_vjp_func_impl(f::F, x::AbstractVector{<:Real}, ad::AbstractADType) where F
    float_x = with_floatlike_contents(x)
    Tx = typeof(float_x)
    f_x = f(float_x)
    Ty = typeof(f_x)
    prep = DI.prepare_pullback_same_point(f, ad, float_x, (f_x,))
    aux = _DIPrep(_borrowable_object(_CacheLikeUse(), prep))
    f_vjp = _VJPFunc(aux, ad, f, float_x, Ty)
    f_vjp_wrapped = _WrappedFunction{Tx,Ty}(f_vjp)
    return f_x, f_vjp_wrapped
end

struct _VJPFunc{P,AD<:AbstractADType,F,Tx<:AbstractVector{<:Number},Ty<:AbstractVector{<:Number}} <: Function
    aux::P
    ad::AD
    f::F
    x::Tx
end
function _VJPFunc(aux::P, ad::AD, f::F, x::Tx, ::Type{Ty}) where {P,AD,F,Tx,Ty}
    return _VJPFunc{P,AD,F,Tx,Ty}(aux, ad, f, x)
end
function _VJPFunc(aux::P, ad::AD, ::Type{FT}, x::Tx, ::Type{Ty}) where {P,AD,FT,Tx,Ty}
    return _VJPFunc{P,AD,Type{FT},Tx,Ty}(aux, ad, FT, x)
end

function (f_vjp::_VJPFunc{<:_DIPrep,AD,F,Tx,Ty})(z::AbstractVector{<:Real}) where {AD,F,Tx,Ty}
    @assert !any(isnan, z)
    prep = f_vjp.aux.prep
    prep_instance, prep_handle = _borrow_maybewrite(prep)
    try
        f = f_vjp.f
        float_x = f_vjp.x
        float_z = convert(Ty, z)::Ty
        z_J = DI.pullback(f, prep_instance, f_vjp.ad, float_x, (float_z,))[1]
        @assert !any(isnan, z_J)
        return convert(Tx, z_J)::Tx
    finally
        _return_borrowed(prep, prep_instance, prep_handle)
    end
end
