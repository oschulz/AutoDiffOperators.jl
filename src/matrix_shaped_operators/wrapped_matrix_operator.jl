# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).


"""
    struct WrappedMatrixOperator{T<:Number,M<:AbstractMatrix{T}} <: MatrixShapedOperator{T}

Wraps an `AbstractMatrix` as a matrix-shaped operator, e.g. to make a
plain matrix participate in lazy operator algebra. Operator `*` and `+`
wrap `AbstractMatrix` operands automatically.

Application delegates to multiplication with the wrapped matrix, so
specialized matrix types (e.g. `Diagonal`, GPU arrays) keep their
optimized code paths. Symmetry/definiteness traits are delegated to the
wrapped matrix as well, which may be costly for large dense matrices.

`Base.parent` returns the wrapped matrix.
"""
struct WrappedMatrixOperator{T<:Number,M<:AbstractMatrix{T}} <: MatrixShapedOperator{T}
    A::M

    function WrappedMatrixOperator{T,M}(A) where {T<:Number,M<:AbstractMatrix{T}}
        _check_real_eltype(T)
        new{T,M}(A)
    end
end
export WrappedMatrixOperator

WrappedMatrixOperator(A::M) where {T<:Number,M<:AbstractMatrix{T}} = WrappedMatrixOperator{T,M}(A)
WrappedMatrixOperator(op::WrappedMatrixOperator) = op

Base.parent(op::WrappedMatrixOperator) = op.A

Base.size(op::WrappedMatrixOperator) = size(op.A)

Base.adjoint(op::WrappedMatrixOperator) = WrappedMatrixOperator(adjoint(op.A))

LinearAlgebra.issymmetric(op::WrappedMatrixOperator) = issymmetric(op.A)
LinearAlgebra.ishermitian(op::WrappedMatrixOperator) = ishermitian(op.A)
LinearAlgebra.isposdef(op::WrappedMatrixOperator) = isposdef(op.A)

Base.:(==)(a::WrappedMatrixOperator, b::WrappedMatrixOperator) = a.A == b.A

function Base.show(io::IO, op::WrappedMatrixOperator)
    print(io, "WrappedMatrixOperator(")
    show(io, op.A)
    print(io, ")")
end

mul_impl(op::WrappedMatrixOperator, x::AbstractVector{<:Number}) = op.A * x
mul_impl(op::WrappedMatrixOperator, X::AbstractMatrix{<:Number}) = op.A * X


"""
    asoperator(A)

Returns `A` if it already is a [`MatrixShapedOperator`](@ref), wraps an
`AbstractMatrix` in a [`WrappedMatrixOperator`](@ref).
"""
asoperator(op::MatrixShapedOperator) = op
asoperator(A::AbstractMatrix{<:Number}) = WrappedMatrixOperator(A)
export asoperator
