# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).


"""
    AutoDiffOperators.ADJacobian{T<:Number} <: MatrixShapedOperator{T}

Represents the Jacobian of a function `f` at a point `x` as a
matrix-shaped operator, computed via automatic differentiation on
demand.

User code should not construct an `ADJacobian` directly, but use
[`with_jacobian`](@ref) instead.

If the `ad` selector lacks forward resp. reverse mode, application in the
corresponding direction fails.

All fields are internal and subject to change.
"""
struct ADJacobian{T<:Number,F,V<:AbstractVector{<:Number},AD<:ADSelector,JF,VF} <: MatrixShapedOperator{T}
    _f::F
    _x::V
    _ad::AD
    _jvp::JF
    _vjp::VF
    _sz::Dims{2}
end
@compat public ADJacobian

Base.size(J::ADJacobian) = J._sz

# A missing AD mode is represented by a `nothing` jvp/vjp field (the
# missing-direction convention of the MatrixShapedOperators seam),
# application in that direction fails like for a MulFuncOperator
# without that direction:

MatrixShapedOperators.explicit_mul_impl(J::ADJacobian, z::AbstractVector{<:Number}) = _jvp_apply(J._jvp, J, z)

_jvp_apply(jvp, ::ADJacobian, z::AbstractVector{<:Number}) = jvp(z)
_jvp_apply(::Nothing, J::ADJacobian, ::AbstractVector{<:Number}) =
    throw(ArgumentError(
        "no forward-mode automatic differentiation available for AD-selector $(nameof(typeof(J._ad))), can't compute Jacobian-vector products"
    ))

# The jvp/vjp closures derive deterministically from (f, x, ad), so
# they take part in neither equality nor hashing; the element type
# participates like for a MulFuncOperator:
Base.:(==)(a::ADJacobian, b::ADJacobian) =
    eltype(a) == eltype(b) && a._f == b._f && a._x == b._x && a._ad == b._ad && a._sz == b._sz

Base.hash(J::ADJacobian{T}, h::UInt) where T =
    hash(J._sz, hash(J._ad, hash(J._x, hash(J._f, hash(T, hash(:ADJacobian, h))))))

# Equal up to the point of linearization:
Base.isapprox(a::ADJacobian, b::ADJacobian; kwargs...) =
    eltype(a) == eltype(b) && a._f == b._f && a._ad == b._ad && a._sz == b._sz &&
    isapprox(a._x, b._x; kwargs...)

function Base.show(io::IO, J::ADJacobian)
    print(io, "ADJacobian(")
    show(io, J._f)
    print(io, ", ")
    summary(io, J._x)
    print(io, ", ")
    show(io, J._ad)
    print(io, ")")
end


"""
    AutoDiffOperators.ADJacobianAdjoint{T<:Number} <: MatrixShapedOperator{T}

The adjoint of an [`ADJacobian`](@ref), which it exposes via
`Base.parent`; applies vector-Jacobian products via reverse-mode AD.
"""
struct ADJacobianAdjoint{T<:Number,OP<:ADJacobian{T}} <: MatrixShapedOperator{T}
    _J::OP
end
@compat public ADJacobianAdjoint

Base.parent(J′::ADJacobianAdjoint) = J′._J

Base.adjoint(J::ADJacobian) = ADJacobianAdjoint(J)
Base.adjoint(J′::ADJacobianAdjoint) = parent(J′)

Base.size(J′::ADJacobianAdjoint) = reverse(size(parent(J′)))

MatrixShapedOperators.explicit_mul_impl(J′::ADJacobianAdjoint, z::AbstractVector{<:Number}) =
    _vjp_apply(parent(J′)._vjp, parent(J′), z)

_vjp_apply(vjp, ::ADJacobian, z::AbstractVector{<:Number}) = vjp(z)
_vjp_apply(::Nothing, J::ADJacobian, ::AbstractVector{<:Number}) =
    throw(ArgumentError(
        "no reverse-mode automatic differentiation available for AD-selector $(nameof(typeof(J._ad))), can't compute vector-Jacobian products"
    ))

Base.:(==)(a::ADJacobianAdjoint, b::ADJacobianAdjoint) = parent(a) == parent(b)

Base.isapprox(a::ADJacobianAdjoint, b::ADJacobianAdjoint; kwargs...) =
    isapprox(parent(a), parent(b); kwargs...)

Base.hash(J′::ADJacobianAdjoint, h::UInt) = hash(parent(J′), hash(:ADJacobianAdjoint, h))

function Base.show(io::IO, J′::ADJacobianAdjoint)
    print(io, "adjoint(")
    show(io, parent(J′))
    print(io, ")")
end


# Explicit materialization uses a single AD Jacobian pass instead of
# operator application to an identity matrix:
Base.AbstractMatrix(J::ADJacobian) = _with_jacobian_matrix(J._f, J._x, J._ad)[2]
Base.AbstractMatrix(J′::ADJacobianAdjoint) = copy(adjoint(AbstractMatrix(parent(J′))))


# Batched application stays column-wise (the MatrixShapedOperators
# fallback), reusing the same-point-prepared jvp/vjp closures - a
# DI-level tuple-tangent batch would re-prepare on every call and
# instantiate a fresh tuple type per batch width, at measurably higher
# cost than the column loop.


# A selector with neither AD mode is rejected before `f` is evaluated
# or any AD preparation runs:
_check_have_ad_mode(::NoAutoDiff, ::NoAutoDiff) = throw(ArgumentError(
    "with_jacobian requires an AD-selector with at least one of forward and reverse mode"
))
_check_have_ad_mode(::Any, ::Any) = nothing

# Requesting the abstract operator type yields an ADJacobian. The jvp
# and vjp preparations and the `_x` field must share a single snapshot of
# the linearization point - separate `with_floatlike_contents` calls
# would give each its own float copy for non-float input:
function with_jacobian(
    f::F, x::AbstractVector{<:Number},
    ::Union{Type{MatrixShapedOperator},Type{ADJacobian}}, ad::ADSelector
) where F
    ad_fwd = forward_adtype(ad)
    ad_rev = reverse_adtype(ad)
    _check_have_ad_mode(ad_fwd, ad_rev)
    float_x = with_floatlike_contents(x)
    f_jvp = _maybe_jvp_func(ad_fwd, f, float_x, ad)
    y, f_vjp = _maybe_with_vjp_func(ad_rev, f, float_x, ad)
    T = promote_type(eltype(float_x), float(eltype(y)))
    sz = Dims((size(y, 1), size(x, 1)))
    J = ADJacobian{T,F,typeof(float_x),typeof(ad),typeof(f_jvp),typeof(f_vjp)}(f, float_x, ad, f_jvp, f_vjp, sz)
    return y, J
end

# Other operator types are generated via the mulfunc_operator seam
# (specialized by the MatrixShapedOperators package extensions for e.g.
# LinearMaps and SciMLOperators types); the support check fails fast
# before any AD preparation work:
function with_jacobian(f::F, x::AbstractVector{<:Number}, ::Type{OP}, ad::ADSelector) where {F,OP}
    check_mulfunc_operator_support(OP)
    ad_fwd = forward_adtype(ad)
    ad_rev = reverse_adtype(ad)
    _check_have_ad_mode(ad_fwd, ad_rev)
    f_jvp = _maybe_jvp_func(ad_fwd, f, x, ad)
    y, f_vjp = _maybe_with_vjp_func(ad_rev, f, x, ad)
    T = promote_type(float(eltype(x)), float(eltype(y)))
    sz = Dims((size(y, 1), size(x, 1)))
    J = mulfunc_operator(OP, T, sz, f_jvp, f_vjp)
    return y, J
end
