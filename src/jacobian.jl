# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).


@deprecate jacobian_matrix(f, x, ad::ADSelector) with_jacobian(f, x, Matrix, ad)[2]


"""
    with_jacobian(f, x, OP, ad::ADSelector)

Returns a tuple `(f(x), J)` with a multiplicative Jabobian operator `J`
of type `OP`.

Example:

```julia
using AutoDiffOperators, LinearMaps
y, J = with_jacobian(f, x, LinearMap, ad)
y == f(x)
_, J_explicit = with_jacobian(f, x, Matrix, ad)
J * z_r ≈ J_explicit * z_r
z_l' * J ≈ z_l' * J_explicit
```

`OP` may be
[`LinearMaps.LinearMap`](https://github.com/JuliaLinearAlgebra/LinearMaps.jl)
(resp. `LinearMaps.FunctionMap`) or `Matrix`. Other operator types can be
supported by specializing
[`AutoDiffOperators.mulfunc_operator`](@ref) for the operator type.

The default implementation of `with_jacobian` uses
[`jvp_func`](@ref) and [`with_vjp_func`](@ref) to implement (adjoint)
multiplication of `J` with (adjoint) vectors.
"""
function with_jacobian end

@deprecate with_jacobian(f, x::AbstractVector{<:Real}, ad::ADSelector) with_jacobian(f, x, MatrixLikeOperator, ad)

function with_jacobian(f, x::AbstractVector{T}, ::Type{OP}, ad::ADSelector) where {T<:Real,OP}
    y, vjp = with_vjp_func(f, x, ad)
    jvp = jvp_func(f, x, ad)
    sz = Dims((size(y,1), size(x,1)))
    J = mulfunc_operator(OP, T, sz, jvp, vjp, Val(false), Val(false), Val(false))
    return y, J
end
export with_jacobian


struct _JVPFunc{F,V,AD<:ADSelector} <: Function
    f::F
    x::V
    ad::AD
end
_JVPFunc(::Type{FT}, x::V, ad::AD) where {FT,V,AD<:ADSelector}  = _JVPFunc{Type{FT},V,AD}(FT, x, ad)

(jvp_func::_JVPFunc)(z) = with_jvp(jvp_func.f, jvp_func.x, z, jvp_func.ad)[2]

"""
    jvp_func(f, x, ad::ADSelector)

Returns a function `jvp` with `jvp(z) == J * z`.
"""
jvp_func(f, x, ad::ADSelector) = _JVPFunc(f, x, forward_ad_selector(ad))
export jvp_func


"""
    with_jvp(f, x, z, ad::ADSelector)

Returns a tuple `(f(x), J * z)`.
"""
function with_jvp end
export with_jvp

with_jvp(f, x, z, ad::FwdRevADSelector) = with_jvp(f, x, z, forward_ad_selector(ad))


# ToDo: add `with_jvp!(f, Jz, x, z, ad::ADSelector)`?


"""
    with_vjp_func(f, x, ad::ADSelector)

Returns a tuple `(f(x), vjp)` with the function `vjp(z) ≈ J' * z`.
"""
function with_vjp_func end
export with_vjp_func

with_vjp_func(f, x, ad::FwdRevADSelector) = with_vjp_func(f, x, reverse_ad_selector(ad))



struct _FwdModeVJPFunc{F,T<:AbstractVector{<:Real},AD<:ADSelector} <: Function
    f::F
    x::T
    ad::AD
end
_FwdModeVJPFunc(::Type{FT}, x::T, ad::AD) where {FT,T<:AbstractVector{<:Real},AD<:ADSelector}  = _FwdModeVJPFunc{Type{FT},T,AD}(FT, x, ad)

function _similar_onehot(A::AbstractArray{<:Number}, ::Type{T}, n::Integer, i::Integer) where {T<:Number}
    result = similar(A, T, (n,))
    fill!(result, zero(T))
    result[i] = one(T)
    return result
end

function (vjp::_FwdModeVJPFunc)(z::AbstractVector{<:Real})
    # ToDo: Reduce memory allocation? Would require a `with_jvp!` function.
    f, x, ad = vjp.f, vjp.x, vjp.ad
    U = promote_type(eltype(f(x)), eltype(x))
    n = size(x, 1)
    result = similar(x, U)
    Base.Threads.@threads for i in eachindex(x)
        tmp = _similar_onehot(x, U, n, i)
        result[i] = dot(z, with_jvp(f, x, tmp, ad)[2])
    end
    result
end
