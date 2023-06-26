# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).


"""
    jacobian_matrix(f, x, ad::ADSelector)

Returns the explicit Jacobian matrix of `f` at `x`

The Jacobian matrix is computed using the automatic differentiation backend
selected by `ad`.
"""
function jacobian_matrix end
export jacobian_matrix

jacobian_matrix(f, x, ad::FwdRevADSelector) = jacobian_matrix(f, x, forward_ad_selector(ad))

function jacobian_matrix(f, x, ad::ADSelector)
    J = with_jacobian(f, x, ad)[2]
    J_matrix = similar(x, size(J))
    copyto!(J_matrix, J)
    return J_matrix
end


"""
    with_jacobian(f, x, ad::ADSelector)

Returns a tuple `(f(x), J)` with a multiplicative Jabobian operator `J.

`J` behaves like jacobian_matrix(f, x, ad) in respect to
multiplication:

```julia
y, J = with_jacobian(f, x, ad)
y == f(x)
J_explicit = jacobian_matrix(f, x, ad)
J * z ≈ J_explicit * z
z * J ≈ z * J_explicit
```

The default implementation of `with_jacobian` relies on
[`with_vjp_func`](@ref) and [`jvp_func`](@ref).
"""
function with_jacobian(f, x::AbstractVector{T}, ad::ADSelector) where T
    y, vjp = with_vjp_func(f, x, ad)
    jvp = jvp_func(f, x, ad)
    J = MatrixLikeOperator{T,false,false,false}(jvp, vjp, (size(y,1), size(x,1)))
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
