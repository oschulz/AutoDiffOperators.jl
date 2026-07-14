# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).


"""
    abstract type MatrixShapedOperator{T<:Number}

Abstract supertype for linear operators with matrix shape and semantics
that are accessed only through application and operator algebra, not
through element access.

Operators are applied to vectors, and column-wise to matrices treated as
batches of vectors, using function-call syntax:

```julia
y = op(x)   # apply to a vector
Y = op(X)   # apply to the columns of a matrix
```

`Base.:(*)` and `Base.:(+)` implement lazy operator algebra:
multiplication and addition of operators with each other and with
`AbstractMatrix` values (which get wrapped via
[`WrappedMatrixOperator`](@ref)) result in [`MatrixShapedProduct`](@ref)
resp. [`MatrixShapedSum`](@ref) operators. Multiplication with plain and
adjoint/transposed *vectors* applies the operator directly, `op * x` is
equivalent to `op(x)` for a vector `x`.

Subtypes must implement:

* `MatrixShapedOperators.mul_impl(op::SomeOperator, x::AbstractVector{<:Number})`
* `Base.adjoint(op::SomeOperator)`
* `Base.size(op::SomeOperator)`
* `LinearAlgebra.issymmetric(op::SomeOperator)`
* `LinearAlgebra.ishermitian(op::SomeOperator)`
* `LinearAlgebra.isposdef(op::SomeOperator)`

and may specialize [`mul_impl`](@ref) for application to matrices, which
is column-wise by default. `Base.:(*)`, `Base.:(+)` and operator
application are only defined for `MatrixShapedOperator` itself and
dispatch to [`mul_impl`](@ref) and [`add_impl`](@ref) after argument
checking, to keep the method footprint on `Base` operators small.
Element types must be real-valued numbers, though not necessarily `Real`
(e.g. tracing-number types), realness is checked via `real(T) === T`.

`Base.transpose`, scalar scaling, `LinearAlgebra.mul!` and
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


"""
    MatrixShapedOperators.mul_impl(a, b)

Implements operator application and multiplication for
[`MatrixShapedOperator`](@ref)s: application to a vector or matrix `b`
(matrices are applied column-wise by default), lazy multiplication with
another operator (a [`MatrixShapedProduct`](@ref) by default) and scalar
scaling.

Operator application and `Base.:(*)` involving matrix-shaped operators
dispatch to `mul_impl` after argument checking, subtypes specialize
`mul_impl` instead.
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


# Operator application:

function (op::MatrixShapedOperator)(x::AbstractVector{<:Number})
    size(x, 1) == size(op, 2) || throw(DimensionMismatch(
        "operator of size $(size(op)) can't be applied to vector of length $(length(x))"
    ))
    return mul_impl(op, x)
end

function (op::MatrixShapedOperator{T})(X::AbstractMatrix{<:Number}) where T
    size(X, 1) == size(op, 2) || throw(DimensionMismatch(
        "operator of size $(size(op)) can't be applied to matrix of size $(size(X))"
    ))
    size(X, 2) == 0 && return similar(X, promote_type(T, eltype(X)), size(op, 1), 0)
    return mul_impl(op, X)
end

mul_impl(op::MatrixShapedOperator, X::AbstractMatrix{<:Number}) = _mapcols(op, X)

_mapcols(f, X::AbstractMatrix) = reduce(hcat, [f(X[:, j]) for j in axes(X, 2)])

# Application of matrix-shaped operators and matrix-like objects alike,
# in operator implementations that hold factors of either kind:
_apply(A, x::AbstractVecOrMat{<:Number}) = A * x
_apply(op::MatrixShapedOperator, x::AbstractVecOrMat{<:Number}) = op(x)


# Operator algebra:

function Base.:(*)(a::MatrixShapedOperator, b::MatrixShapedOperator)
    size(a, 2) == size(b, 1) || throw(DimensionMismatch(
        "operator of size $(size(a)) can't be multiplied with operator of size $(size(b))"
    ))
    return mul_impl(a, b)
end

Base.:(*)(a::MatrixShapedOperator, B::AbstractMatrix{<:Number}) = a * asoperator(B)
Base.:(*)(A::AbstractMatrix{<:Number}, b::MatrixShapedOperator) = asoperator(A) * b

Base.:(*)(s::Number, op::MatrixShapedOperator) = mul_impl(s, op)
Base.:(*)(op::MatrixShapedOperator, s::Number) = mul_impl(s, op)

Base.:(+)(a::MatrixShapedOperator, b::MatrixShapedOperator) = add_impl(a, b)

Base.:(+)(a::MatrixShapedOperator, B::AbstractMatrix{<:Number}) = a + asoperator(B)
Base.:(+)(A::AbstractMatrix{<:Number}, b::MatrixShapedOperator) = asoperator(A) + b


# Multiplication with vectors applies the operator:

Base.:(*)(op::MatrixShapedOperator, x::AbstractVector{<:Number}) = op(x)

function Base.:(*)(x_l::LinearAlgebra.Adjoint{<:Number,<:AbstractVector{<:Number}}, op::MatrixShapedOperator)
    return adjoint(adjoint(op)(adjoint(x_l)))
end

function Base.:(*)(x_l::LinearAlgebra.Transpose{<:Number,<:AbstractVector{<:Number}}, op::MatrixShapedOperator)
    return transpose(transpose(op)(transpose(x_l)))
end


function LinearAlgebra.mul!(y::AbstractVecOrMat{<:Number}, op::MatrixShapedOperator, x::AbstractVecOrMat{<:Number})
    return mul!(y, op, x, true, false)
end

function LinearAlgebra.mul!(
    y::AbstractVecOrMat{<:Number}, op::MatrixShapedOperator, x::AbstractVecOrMat{<:Number},
    alpha::Number, beta::Number
)
    w = op(x)
    if iszero(beta)
        y .= alpha .* w
    else
        y .= alpha .* w .+ beta .* y
    end
    return y
end


function Base.Matrix(op::MatrixShapedOperator{T}) where T
    n = size(op, 2)
    return op(Matrix{T}(I, n, n))
end

Base.convert(::Type{Matrix}, op::MatrixShapedOperator) = Matrix(op)
