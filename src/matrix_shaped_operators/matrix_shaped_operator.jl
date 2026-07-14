# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).


"""
    abstract type MatrixShapedOperator{T<:Number}

Abstract supertype for linear operators with matrix shape and semantics
that are accessed only through multiplication and adjoint multiplication.

Subtypes must implement:

* `MatrixShapedOperators.mul_impl(op::SomeOperator, x::AbstractVector{<:Number})`
* `Base.adjoint(op::SomeOperator)`
* `Base.size(op::SomeOperator)`
* `LinearAlgebra.issymmetric(op::SomeOperator)`
* `LinearAlgebra.ishermitian(op::SomeOperator)`
* `LinearAlgebra.isposdef(op::SomeOperator)`

and may specialize [`mul_impl`](@ref) for multiplication with matrices,
which is column-wise by default. `Base.:(*)` and `Base.:(+)` themselves
are only defined for `MatrixShapedOperator` and dispatch to
[`mul_impl`](@ref) and [`add_impl`](@ref) after argument checking, to
keep the method footprint on `Base` operators small. Element types must be real-valued numbers, though not
necessarily `Real` (e.g. tracing-number types), realness is checked via
`real(T) === T`.

`Base.transpose`, multiplication with adjoint/transposed vectors, scalar
scaling, operator sums via `+` (see [`MatrixShapedSum`](@ref)), operator
products via `*` (see [`MatrixShapedProduct`](@ref)), `LinearAlgebra.mul!`
and materialization via `Base.Matrix` are provided generically.
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


"""
    MatrixShapedOperators.mul_impl(a, b)

Implements multiplication for [`MatrixShapedOperator`](@ref)s: with a
vector or matrix `b` (matrices are applied column-wise by default), with
another operator (a [`MatrixShapedProduct`](@ref) by default) or with a
number `a` or `b`.

`Base.:(*)` involving matrix-shaped operators dispatches to `mul_impl`
after argument checking, subtypes specialize `mul_impl` instead of
`Base.:(*)`.
"""
function mul_impl end

"""
    MatrixShapedOperators.add_impl(a, b)

Implements addition of [`MatrixShapedOperator`](@ref)s, resulting in a
[`MatrixShapedSum`](@ref) by default. `Base.:(+)` involving matrix-shaped
operators dispatches to `add_impl`, subtypes specialize `add_impl`
instead of `Base.:(+)`.
"""
function add_impl end

function Base.:(*)(op::MatrixShapedOperator, x::AbstractVector{<:Number})
    size(x, 1) == size(op, 2) || throw(DimensionMismatch(
        "operator of size $(size(op)) can't be multiplied with vector of length $(length(x))"
    ))
    return mul_impl(op, x)
end

function Base.:(*)(op::MatrixShapedOperator{T}, X::AbstractMatrix{<:Number}) where T
    size(X, 1) == size(op, 2) || throw(DimensionMismatch(
        "operator of size $(size(op)) can't be multiplied with matrix of size $(size(X))"
    ))
    size(X, 2) == 0 && return similar(X, promote_type(T, eltype(X)), size(op, 1), 0)
    return mul_impl(op, X)
end

function Base.:(*)(a::MatrixShapedOperator, b::MatrixShapedOperator)
    size(a, 2) == size(b, 1) || throw(DimensionMismatch(
        "operator of size $(size(a)) can't be composed with operator of size $(size(b))"
    ))
    return mul_impl(a, b)
end

Base.:(*)(s::Number, op::MatrixShapedOperator) = mul_impl(s, op)
Base.:(*)(op::MatrixShapedOperator, s::Number) = mul_impl(s, op)

Base.:(+)(a::MatrixShapedOperator, b::MatrixShapedOperator) = add_impl(a, b)

mul_impl(op::MatrixShapedOperator, X::AbstractMatrix{<:Number}) = _mapcols(Base.Fix1(*, op), X)

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
