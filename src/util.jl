# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).


"""
    AutoDiffOperators.with_floatlike_contents(A::AbstractArray)

If the elements of `A` are integer-like, convert them using `float`,
otherwise return `A` unchanged.
"""
function with_floatlike_contents end

with_floatlike_contents(A::AbstractArray{T}) where {T<:Real} = _with_floatlike_contents_impl(A, float(T))

_with_floatlike_contents_impl(A::AbstractArray{T}, ::Type{T}) where {T<:Real} = A
_with_floatlike_contents_impl(A::AbstractArray{T}, ::Type{U}) where {T<:Real,U<:Real} = float.(A)


"""
    AutoDiffOperators.similar_onehot(A::AbstractArray, ::Type{T}, n::Integer, i::Integer)

Return an array similar to `A`, but with `n` elements of type `T`, all set to
zero but the `i`-th element set to one.
"""
function similar_onehot end

function similar_onehot(A::AbstractArray{<:Number}, ::Type{T}, n::Integer, i::Integer) where {T<:Number}
    result = similar(A, T, (n,))
    fill!(result, zero(T))
    result[i] = one(T)
    return result
end


# _similar_type(::T) where T = Core.Compiler.return_type(similar, Tuple{T})
_similar_type(::Type{T}) where T = Core.Compiler.return_type(similar, Tuple{T})

_oftype(::T, x::T) where T = x
_oftype(::T, x::U) where {T,U} = convert(T, x)::T

_oftype(::T, x::T) where {T<:AbstractArray} = x
function _oftype(::T, x::U) where {T<:AbstractArray,U<:AbstractArray}
    R = _similar_type(T)
    return convert(R, x)::R
end

_oftype!!(::T, x::T) where T = x
_oftype!!(y::T, x::U) where {T,U} = oftype(y, x)

_oftype!!(::T, x::T) where {T<:AbstractArray} = x
_oftype!!(y::T, x::U) where {T<:AbstractArray,U<:AbstractArray} = _oftype_array_impl!!(Val(isbitstype(T)), y, x)
# Assume T is immutable:
_oftype_array_impl!!(::Val{true}, y::T, x::U) where {T<:AbstractArray,U<:AbstractArray} = oftype(y, x)
# Assume T is mutable:
_oftype_array_impl!!(::Val{false}, y::T, x::U) where {T<:AbstractArray,U<:AbstractArray} = copyto!(y, x)::T
