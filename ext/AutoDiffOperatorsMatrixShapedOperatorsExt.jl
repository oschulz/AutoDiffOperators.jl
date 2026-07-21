# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

module AutoDiffOperatorsMatrixShapedOperatorsExt

using MatrixShapedOperators
using MatrixShapedOperators: mulfunc_operator

import AutoDiffOperators
using AutoDiffOperators: ADSelector, forward_adtype, reverse_adtype,
    with_floatlike_contents, _maybe_jvp_func, _maybe_with_vjp_func,
    _with_jacobian_matrix, _traced_array_kind
using AutoDiffOperators: AbstractADType, NoAutoDiff

const DI = AutoDiffOperators.DI


"""
    ADJacobian{T<:Number} <: MatrixShapedOperator{T}

Represents the Jacobian of a function `f` at a point `x` as a
matrix-shaped operator, computed via automatic differentiation on
demand.

Constructed via `with_jacobian`, requested by the abstract operator
type:

```julia
using AutoDiffOperators, MatrixShapedOperators

y, J = with_jacobian(f, x, MatrixShapedOperator, ad)
J * z        # Jacobian-vector product via forward-mode AD
J' * z       # vector-Jacobian product via reverse-mode AD
Matrix(J)    # explicit Jacobian in a single AD pass
```

Unlike a `MulFuncOperator` holding opaque AD closures, `ADJacobian`
keeps `f`, `x` and the `ADSelector` accessible as fields (`x` with
float-like contents).
"""
struct ADJacobian{T<:Number,F,V<:AbstractVector{<:Number},AD<:ADSelector,JF,VF} <: MatrixShapedOperator{T}
    f::F
    x::V
    ad::AD
    jvp::JF
    vjp::VF
    sz::Dims{2}
end

Base.size(J::ADJacobian) = J.sz

MatrixShapedOperators.explicit_mul_impl(J::ADJacobian, z::AbstractVector{<:Number}) = J.jvp(z)

MatrixShapedOperators.BatchedMulStyle(::ADJacobian) = MatrixShapedOperators.BatchedMul()

# The jvp/vjp closures derive deterministically from (f, x, ad), so
# they take part in neither equality nor hashing:
Base.:(==)(a::ADJacobian, b::ADJacobian) =
    a.f == b.f && a.x == b.x && a.ad == b.ad && a.sz == b.sz

Base.hash(J::ADJacobian, h::UInt) =
    hash(J.sz, hash(J.ad, hash(J.x, hash(J.f, hash(:ADJacobian, h)))))

function Base.show(io::IO, J::ADJacobian)
    print(io, "ADJacobian(")
    show(io, J.f)
    print(io, ", ")
    show(io, J.x)
    print(io, ", ")
    show(io, J.ad)
    print(io, ")")
end


"""
    ADJacobianAdjoint{T<:Number} <: MatrixShapedOperator{T}

The adjoint of an [`ADJacobian`](@ref), available via `Base.parent`;
applies vector-Jacobian products via reverse-mode AD.
"""
struct ADJacobianAdjoint{T<:Number,OP<:ADJacobian{T}} <: MatrixShapedOperator{T}
    J::OP
end

Base.parent(J′::ADJacobianAdjoint) = J′.J

Base.adjoint(J::ADJacobian) = ADJacobianAdjoint(J)
Base.adjoint(J′::ADJacobianAdjoint) = parent(J′)

Base.size(J′::ADJacobianAdjoint) = reverse(size(parent(J′)))

MatrixShapedOperators.explicit_mul_impl(J′::ADJacobianAdjoint, z::AbstractVector{<:Number}) = parent(J′).vjp(z)

MatrixShapedOperators.BatchedMulStyle(::ADJacobianAdjoint) = MatrixShapedOperators.BatchedMul()

Base.:(==)(a::ADJacobianAdjoint, b::ADJacobianAdjoint) = parent(a) == parent(b)

Base.hash(J′::ADJacobianAdjoint, h::UInt) = hash(parent(J′), hash(:ADJacobianAdjoint, h))

function Base.show(io::IO, J′::ADJacobianAdjoint)
    print(io, "adjoint(")
    show(io, parent(J′))
    print(io, ")")
end


# Explicit materialization uses a single AD Jacobian pass instead of
# operator application to an identity matrix:
Base.AbstractMatrix(J::ADJacobian) = _with_jacobian_matrix(J.f, J.x, J.ad)[2]
Base.AbstractMatrix(J′::ADJacobianAdjoint) = copy(adjoint(AbstractMatrix(parent(J′))))


# Batched application computes all Jacobian-vector products of a column
# batch in a single DI pushforward resp. pullback call. Traced arrays
# keep the column-wise path, whose per-column functions carry the
# tracing-compatible specializations:

function MatrixShapedOperators.batched_mul_impl(J::ADJacobian, X::AbstractMatrix{<:Number})
    return _batched_jvp(_traced_array_kind(J.x), forward_adtype(J.ad), J, X)
end

function MatrixShapedOperators.batched_mul_impl(J′::ADJacobianAdjoint, X::AbstractMatrix{<:Number})
    J = parent(J′)
    return _batched_vjp(_traced_array_kind(J.x), reverse_adtype(J.ad), J, X)
end

_column_tuple(X::AbstractMatrix) = Tuple(X[:, j] for j in axes(X, 2))

_jvp_cols(J::ADJacobian, X::AbstractMatrix) = reduce(hcat, [J.jvp(X[:, j]) for j in axes(X, 2)])
_vjp_cols(J::ADJacobian, X::AbstractMatrix) = reduce(hcat, [J.vjp(X[:, j]) for j in axes(X, 2)])

_batched_jvp(::Val, ::AbstractADType, J::ADJacobian, X::AbstractMatrix) = _jvp_cols(J, X)
_batched_jvp(::Nothing, ::NoAutoDiff, J::ADJacobian, X::AbstractMatrix) = _jvp_cols(J, X)

function _batched_jvp(::Nothing, ad_fwd::AbstractADType, J::ADJacobian, X::AbstractMatrix)
    f, x = J.f, J.x
    tx = _column_tuple(X)
    prep = DI.prepare_pushforward_same_point(f, ad_fwd, x, tx)
    ty = DI.pushforward(f, prep, ad_fwd, x, tx)
    return reduce(hcat, ty)
end

_batched_vjp(::Val, ::AbstractADType, J::ADJacobian, X::AbstractMatrix) = _vjp_cols(J, X)
_batched_vjp(::Nothing, ::NoAutoDiff, J::ADJacobian, X::AbstractMatrix) = _vjp_cols(J, X)

function _batched_vjp(::Nothing, ad_rev::AbstractADType, J::ADJacobian, X::AbstractMatrix)
    f, x = J.f, J.x
    ty = _column_tuple(X)
    prep = DI.prepare_pullback_same_point(f, ad_rev, x, ty)
    tx = DI.pullback(f, prep, ad_rev, x, ty)
    return reduce(hcat, tx)
end


# Requesting the abstract operator type yields an ADJacobian:
function AutoDiffOperators.with_jacobian(
    f::F, x::AbstractVector{<:Number},
    ::Union{Type{MatrixShapedOperator},Type{ADJacobian}}, ad::ADSelector
) where F
    ad_fwd = forward_adtype(ad)
    ad_rev = reverse_adtype(ad)
    f_jvp = _maybe_jvp_func(ad_fwd, f, x, ad)
    y, f_vjp = _maybe_with_vjp_func(ad_rev, f, x, ad)
    T = promote_type(float(eltype(x)), float(eltype(y)))
    sz = Dims((size(y, 1), size(x, 1)))
    float_x = with_floatlike_contents(x)
    J = ADJacobian{T,F,typeof(float_x),typeof(ad),typeof(f_jvp),typeof(f_vjp)}(f, float_x, ad, f_jvp, f_vjp, sz)
    return y, J
end

# Other operator types are generated via the mulfunc_operator seam
# (specialized by the MatrixShapedOperators package extensions for e.g.
# LinearMaps and SciMLOperators types):
function AutoDiffOperators.with_jacobian(f::F, x::AbstractVector{<:Number}, ::Type{OP}, ad::ADSelector) where {F,OP}
    ad_fwd = forward_adtype(ad)
    ad_rev = reverse_adtype(ad)
    f_jvp = _maybe_jvp_func(ad_fwd, f, x, ad)
    y, f_vjp = _maybe_with_vjp_func(ad_rev, f, x, ad)
    T = promote_type(float(eltype(x)), float(eltype(y)))
    sz = Dims((size(y, 1), size(x, 1)))
    J = mulfunc_operator(OP, T, sz, f_jvp, f_vjp)
    return y, J
end


end # module AutoDiffOperatorsMatrixShapedOperatorsExt
