# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).


"""
    struct RowGramOperator{T<:Number,OP} <: MatrixShapedOperator{T}

Represents the row-Gram operator `A * A'` of a matrix or matrix-shaped
operator `A`, storing only `A` itself.

Constructor:

```julia
op = RowGramOperator(A)
op * x == A * (A' * x)
```

`op` is symmetric, hermitian and declared positive definite (it is
positive semi-definite in general and positive definite if `A` has full
row rank). The underlying `A` can be retrieved via [`gram_factor`](@ref).

Column-Gram operators `A' * A` are represented by `RowGramOperator(A')`.

`A` may be an `AbstractMatrix`, a [`MatrixShapedOperator`](@ref) or any
matrix-like object that supports multiplication with vectors and
matrices, `adjoint` and `size`.
"""
struct RowGramOperator{T<:Number,OP} <: MatrixShapedOperator{T}
    A::OP

    function RowGramOperator{T,OP}(A) where {T<:Number,OP}
        _check_real_eltype(T)
        new{T,OP}(A)
    end
end
export RowGramOperator

RowGramOperator(A::OP) where OP = RowGramOperator{eltype(A),OP}(A)

"""
    gram_factor(op::RowGramOperator)

Returns `A` for `op` representing `A * A'`.
"""
gram_factor(op::RowGramOperator) = op.A
export gram_factor

Base.size(op::RowGramOperator) = (size(op.A, 1), size(op.A, 1))

Base.adjoint(op::RowGramOperator) = op

LinearAlgebra.issymmetric(::RowGramOperator) = true
LinearAlgebra.ishermitian(::RowGramOperator) = true
LinearAlgebra.isposdef(::RowGramOperator) = true

Base.:(==)(a::RowGramOperator, b::RowGramOperator) = a.A == b.A

function Base.show(io::IO, op::RowGramOperator)
    print(io, "RowGramOperator(")
    show(io, op.A)
    print(io, ")")
end

Base.:(*)(op::RowGramOperator, x::AbstractVector{<:Number}) = op.A * (op.A' * x)
Base.:(*)(op::RowGramOperator, X::AbstractMatrix{<:Number}) = op.A * (op.A' * X)
