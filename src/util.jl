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
