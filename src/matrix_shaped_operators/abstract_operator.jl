# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).


"""
    abstract type MatrixShapedOperator{T<:Number}

Abstract supertype for linear operators with matrix shape and semantics
that are accessed only through multiplication and adjoint multiplication.

Subtypes must implement:

* `Base.:(*)(op::SomeOperator, x::AbstractVector{<:Number})`
* `Base.adjoint(op::SomeOperator)`
* `Base.size(op::SomeOperator)`
* `LinearAlgebra.issymmetric(op::SomeOperator)`
* `LinearAlgebra.ishermitian(op::SomeOperator)`
* `LinearAlgebra.isposdef(op::SomeOperator)`

and may specialize multiplication with matrices, which is column-wise by
default. Element types must be real-valued numbers, though not
necessarily `Real` (e.g. tracing-number types), realness is checked via
`real(T) === T`.

`Base.transpose`, multiplication with adjoint/transposed vectors, scalar
scaling, operator composition via `*`, operator superposition via `+`
(see [`SuperposedOperator`](@ref)), `LinearAlgebra.mul!` and
materialization via `Base.Matrix` are provided generically.
"""
abstract type MatrixShapedOperator{T<:Number} end
export MatrixShapedOperator

function _check_real_eltype(::Type{T}) where {T<:Number}
    T <: Real || real(T) === T || throw(ArgumentError(
        "MatrixShapedOperators only support real-valued numbers, got element type $T"
    ))
    return nothing
end

Base.eltype(::Type{<:MatrixShapedOperator{T}}) where T = T

function Base.size(op::MatrixShapedOperator, d::Integer)
    d >= 1 || throw(ArgumentError("dimension out of range, got $d"))
    return d <= 2 ? size(op)[d] : 1
end

Base.transpose(op::MatrixShapedOperator) = adjoint(op)


function Base.:(*)(op::MatrixShapedOperator{T}, X::AbstractMatrix{<:Number}) where T
    size(X, 1) == size(op, 2) || throw(DimensionMismatch(
        "operator of size $(size(op)) can't be multiplied with matrix of size $(size(X))"
    ))
    size(X, 2) == 0 && return similar(X, promote_type(T, eltype(X)), size(op, 1), 0)
    return _mapcols(Base.Fix1(*, op), X)
end

_mapcols(f, X::AbstractMatrix) = reduce(hcat, [f(X[:, j]) for j in axes(X, 2)])

function Base.:(*)(x_l::LinearAlgebra.Adjoint{<:Number,<:AbstractVector{<:Number}}, op::MatrixShapedOperator)
    return adjoint(adjoint(op) * adjoint(x_l))
end

function Base.:(*)(x_l::LinearAlgebra.Transpose{<:Number,<:AbstractVector{<:Number}}, op::MatrixShapedOperator)
    return transpose(transpose(op) * transpose(x_l))
end


function LinearAlgebra.mul!(y::AbstractVecOrMat{<:Number}, op::MatrixShapedOperator, x::AbstractVecOrMat{<:Number})
    return mul!(y, op, x, true, false)
end

function LinearAlgebra.mul!(
    y::AbstractVecOrMat{<:Number}, op::MatrixShapedOperator, x::AbstractVecOrMat{<:Number},
    alpha::Number, beta::Number
)
    w = op * x
    if iszero(beta)
        y .= alpha .* w
    else
        y .= alpha .* w .+ beta .* y
    end
    return y
end


"""
    MatrixShapedOperators.similar_onehot(A::AbstractArray, ::Type{T}, n::Integer, i::Integer)

Return an array similar to `A`, but with `n` elements of type `T`, all set to
zero but the `i`-th element set to one.
"""
function similar_onehot end
export similar_onehot

function similar_onehot(A::AbstractArray{<:Number}, ::Type{T}, n::Integer, i::Integer) where {T<:Number}
    result = similar(A, T, (n,))
    fill!(result, zero(T))
    result[i] = one(T)
    return result
end


function Base.Matrix(op::MatrixShapedOperator{T}) where T
    m, n = size(op)
    A = Matrix{T}(undef, (m, n))
    @threads for j in axes(A, 2)
        A[:, j] = op * similar_onehot(A, T, n, j)
    end
    return A
end

Base.convert(::Type{Matrix}, op::MatrixShapedOperator) = Matrix(op)
