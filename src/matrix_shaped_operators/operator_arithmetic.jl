# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).


"""
    struct UniformScalingOperator{T<:Number} <: MatrixShapedOperator{T}

Represents `λ * I` with an explicit size.

Constructor:

```julia
op = UniformScalingOperator(λ, n)
op * x == λ .* x
```

Typically arises from adding `LinearAlgebra.UniformScaling` values like
`I` to matrix-shaped operators (see [`MatrixShapedSum`](@ref)).
"""
struct UniformScalingOperator{T<:Number} <: MatrixShapedOperator{T}
    λ::T
    n::Int

    function UniformScalingOperator{T}(λ, n::Integer) where {T<:Number}
        _check_real_eltype(T)
        new{T}(λ, n)
    end
end
export UniformScalingOperator

UniformScalingOperator(λ::T, n::Integer) where {T<:Number} = UniformScalingOperator{T}(λ, n)

Base.size(op::UniformScalingOperator) = (op.n, op.n)

Base.adjoint(op::UniformScalingOperator) = op

LinearAlgebra.issymmetric(::UniformScalingOperator) = true
LinearAlgebra.ishermitian(::UniformScalingOperator) = true
LinearAlgebra.isposdef(op::UniformScalingOperator) = op.λ > 0

Base.:(==)(a::UniformScalingOperator, b::UniformScalingOperator) = a.λ == b.λ && a.n == b.n

function Base.show(io::IO, op::UniformScalingOperator)
    print(io, "UniformScalingOperator(")
    show(io, op.λ)
    print(io, ", ")
    show(io, op.n)
    print(io, ")")
end

mul_impl(op::UniformScalingOperator, x::AbstractVector{<:Number}) = op.λ .* x
mul_impl(op::UniformScalingOperator, X::AbstractMatrix{<:Number}) = op.λ .* X


"""
    struct MatrixShapedSum{T<:Number,OPS<:Tuple} <: MatrixShapedOperator{T}

Represents the additive superposition `terms[1] + terms[2] + ...` of
matrix-shaped operators.

Typically constructed via operator addition instead of directly:

```julia
op_a + op_b isa MatrixShapedSum
op_a + op_b + 2 * I isa MatrixShapedSum
```

Adding a `LinearAlgebra.UniformScaling` adds a
[`UniformScalingOperator`](@ref) term. All terms must have equal size.
The operator is symmetric/hermitian/positive definite if all of its
terms are.

The terms may be given as a tuple (best for a few terms of different
type) or as a vector (best for many terms of equal type).
"""
struct MatrixShapedSum{T<:Number,OPS<:Union{Tuple,AbstractVector}} <: MatrixShapedOperator{T}
    terms::OPS

    function MatrixShapedSum{T,OPS}(terms::OPS) where {T<:Number,OPS<:Union{Tuple,AbstractVector}}
        isempty(terms) && throw(ArgumentError("MatrixShapedSum requires at least one term"))
        _check_real_eltype(T)
        sz = size(first(terms))
        all(t -> size(t) == sz, terms) || throw(DimensionMismatch(
            "MatrixShapedSum terms must all have equal size"
        ))
        new{T,OPS}(terms)
    end
end
export MatrixShapedSum

function MatrixShapedSum(terms::OPS) where {OPS<:Union{Tuple,AbstractVector}}
    T = mapreduce(eltype, promote_type, terms)
    MatrixShapedSum{T,OPS}(terms)
end

Base.size(op::MatrixShapedSum) = size(first(op.terms))

LinearAlgebra.issymmetric(op::MatrixShapedSum) = all(issymmetric, op.terms)
LinearAlgebra.ishermitian(op::MatrixShapedSum) = all(ishermitian, op.terms)
LinearAlgebra.isposdef(op::MatrixShapedSum) = all(isposdef, op.terms)

Base.adjoint(op::MatrixShapedSum) = MatrixShapedSum(map(adjoint, op.terms))

Base.:(==)(a::MatrixShapedSum, b::MatrixShapedSum) = a.terms == b.terms

function Base.show(io::IO, op::MatrixShapedSum)
    print(io, "MatrixShapedSum(")
    show(io, op.terms)
    print(io, ")")
end

mul_impl(op::MatrixShapedSum, x::AbstractVector{<:Number}) = _sum_mul(op, x)
mul_impl(op::MatrixShapedSum, X::AbstractMatrix{<:Number}) = _sum_mul(op, X)

_sum_mul(op::MatrixShapedSum, x::AbstractVecOrMat{<:Number}) = mapreduce(Base.Fix2(_apply, x), +, op.terms)


_vec_terms(terms::Tuple) = [terms...]
_vec_terms(terms::AbstractVector) = terms

_cat_terms(a::Tuple, b::Tuple) = (a..., b...)
_cat_terms(a, b) = vcat(_vec_terms(a), _vec_terms(b))

add_impl(a::MatrixShapedOperator, b::MatrixShapedOperator) = MatrixShapedSum((a, b))
add_impl(a::MatrixShapedSum, b::MatrixShapedOperator) = MatrixShapedSum(_cat_terms(a.terms, (b,)))
add_impl(a::MatrixShapedOperator, b::MatrixShapedSum) = MatrixShapedSum(_cat_terms((a,), b.terms))
add_impl(a::MatrixShapedSum, b::MatrixShapedSum) = MatrixShapedSum(_cat_terms(a.terms, b.terms))

Base.:(+)(a::MatrixShapedOperator, J::UniformScaling) = a + UniformScalingOperator(J.λ, size(a, 1))
Base.:(+)(J::UniformScaling, a::MatrixShapedOperator) = a + J


"""
    struct MatrixShapedProduct{T<:Number,OPS<:Tuple} <: MatrixShapedOperator{T}

Represents the product `factors[1] * factors[2] * ...` of matrix-shaped
operators, applied right-to-left.

Typically constructed via operator multiplication instead of directly:

```julia
op_a * op_b isa MatrixShapedProduct
```

Adjacent factor sizes must match. The factors may be given as a tuple
(best for a few factors of different type) or as a vector (best for many
factors of equal type). No symmetry/definiteness traits are
derived for the product (they are not preserved by operator products in
general); use [`RowGramOperator`](@ref) for products of the form `A * A'`.
"""
struct MatrixShapedProduct{T<:Number,OPS<:Union{Tuple,AbstractVector}} <: MatrixShapedOperator{T}
    factors::OPS

    function MatrixShapedProduct{T,OPS}(factors::OPS) where {T<:Number,OPS<:Union{Tuple,AbstractVector}}
        isempty(factors) && throw(ArgumentError("MatrixShapedProduct requires at least one factor"))
        _check_real_eltype(T)
        for i in firstindex(factors):(lastindex(factors) - 1)
            size(factors[i], 2) == size(factors[i + 1], 1) || throw(DimensionMismatch(
                "operator of size $(size(factors[i])) can't be composed with operator of size $(size(factors[i + 1]))"
            ))
        end
        new{T,OPS}(factors)
    end
end
export MatrixShapedProduct

function MatrixShapedProduct(factors::OPS) where {OPS<:Union{Tuple,AbstractVector}}
    T = mapreduce(eltype, promote_type, factors)
    MatrixShapedProduct{T,OPS}(factors)
end

Base.size(op::MatrixShapedProduct) = (size(first(op.factors), 1), size(last(op.factors), 2))

LinearAlgebra.issymmetric(::MatrixShapedProduct) = false
LinearAlgebra.ishermitian(::MatrixShapedProduct) = false
LinearAlgebra.isposdef(::MatrixShapedProduct) = false

Base.adjoint(op::MatrixShapedProduct) = MatrixShapedProduct(reverse(map(adjoint, op.factors)))

Base.:(==)(a::MatrixShapedProduct, b::MatrixShapedProduct) = a.factors == b.factors

function Base.show(io::IO, op::MatrixShapedProduct)
    print(io, "MatrixShapedProduct(")
    show(io, op.factors)
    print(io, ")")
end

mul_impl(op::MatrixShapedProduct, x::AbstractVector{<:Number}) = _product_mul(op, x)
mul_impl(op::MatrixShapedProduct, X::AbstractMatrix{<:Number}) = _product_mul(op, X)

_product_mul(op::MatrixShapedProduct, x::AbstractVecOrMat{<:Number}) = foldr(_apply, op.factors; init = x)


mul_impl(a::MatrixShapedOperator, b::MatrixShapedOperator) = MatrixShapedProduct((a, b))
mul_impl(a::MatrixShapedProduct, b::MatrixShapedOperator) = MatrixShapedProduct(_cat_terms(a.factors, (b,)))
mul_impl(a::MatrixShapedOperator, b::MatrixShapedProduct) = MatrixShapedProduct(_cat_terms((a,), b.factors))
mul_impl(a::MatrixShapedProduct, b::MatrixShapedProduct) = MatrixShapedProduct(_cat_terms(a.factors, b.factors))
