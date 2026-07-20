# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).


"""
    struct ADJacobian{T<:Number} <: MatrixShapedOperator{T}

Represents the Jacobian of a function `f` at a point `x` as a
matrix-shaped operator, computed via automatic differentiation on
demand.

Constructed via [`with_jacobian`](@ref):

```julia
y, J = with_jacobian(f, x, ADJacobian, ad)
J * z        # Jacobian-vector product via forward-mode AD
J' * z       # vector-Jacobian product via reverse-mode AD
Matrix(J)    # explicit Jacobian in a single AD pass
```

Unlike a `MulFuncOperator` holding opaque AD closures, `ADJacobian`
keeps `f`, `x` and the [`ADSelector`](@ref) accessible as fields
(`x` with float-like contents).
"""
struct ADJacobian{T<:Number,F,V<:AbstractVector{<:Number},AD<:ADSelector,JF,VF} <: MatrixShapedOperator{T}
    f::F
    x::V
    ad::AD
    jvp::JF
    vjp::VF
    sz::Dims{2}
end
export ADJacobian

Base.size(J::ADJacobian) = J.sz

MatrixShapedOperators.explicit_mul_impl(J::ADJacobian, z::AbstractVector{<:Number}) = J.jvp(z)

MatrixShapedOperators.BatchedMulStyle(J::ADJacobian) = MatrixShapedOperators.BatchedMulStyle(J.jvp)

Base.:(==)(a::ADJacobian, b::ADJacobian) =
    a.f == b.f && a.x == b.x && a.ad == b.ad && a.sz == b.sz

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
    struct ADJacobianAdjoint{T<:Number} <: MatrixShapedOperator{T}

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

MatrixShapedOperators.BatchedMulStyle(J′::ADJacobianAdjoint) = MatrixShapedOperators.BatchedMulStyle(parent(J′).vjp)

Base.:(==)(a::ADJacobianAdjoint, b::ADJacobianAdjoint) = parent(a) == parent(b)

function Base.show(io::IO, J′::ADJacobianAdjoint)
    print(io, "adjoint(")
    show(io, parent(J′))
    print(io, ")")
end


# Explicit materialization uses a single AD Jacobian pass instead of
# operator application to an identity matrix:
Base.AbstractMatrix(J::ADJacobian) = _with_jacobian_matrix(J.f, J.x, J.ad)[2]
Base.AbstractMatrix(J′::ADJacobianAdjoint) = copy(adjoint(AbstractMatrix(parent(J′))))
