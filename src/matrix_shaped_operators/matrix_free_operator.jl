# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

"""
    MatrixShapedOperators.mulfunc_operator(
        ::Type{OP},
        ::Type{T}, sz::Dims{2}, ovp, vop,
        ::Val{sym}, ::Val{herm}, ::Val{posdef}
    ) where {OP, T<:Real, sym, herm, posdef}

Generates a linear operator object of type `OP` that supports multiplication
with (adjoint) vectors based on a multiplication function `ovp` and an
adjoint multiplication function `vop`.

An operator
`op = mulfunc_operator(OP, T, sz, ovp, vop, Val(sym), ::Val{herm}, ::Val{posdef})`
must show the following behavior:

```julia
op isa OP
eltype(op) == T
size(op) == sz
op * x_r == ovp(x_r)
x_l' * op == vop(x_l)
issymmetric(op) == sym
ishermitian(op) == herm
isposdef(op) == posdef
```

where `x_l` and `x_r` are vectors of size `sz[1]` and `sz[2]` respectively.
"""
function mulfunc_operator end
export mulfunc_operator

function mulfunc_operator(
    ::Type{Matrix},
    ::Type{T}, sz::Dims{2}, ovp, vop,
    ::Val{sym}, ::Val{herm}, ::Val{posdef}
) where {T<:Real, sym, herm, posdef}
    A = Matrix{T}(undef, sz)
    @threads for j in axes(A, 2)
        A[:, j] = ovp(similar_onehot(A, T, sz[2], j))
    end
    return A
end


"""
    struct MatrixFreeOperator{T<:Number,sym,herm,posdef,F,G} <: MatrixShapedOperator{T}

Linear operator that multiplies via a function `ovp` and adjoint-multiplies
via a function `vop`.

Constructor:

```julia
op = MatrixFreeOperator{T,sym,herm,posdef}(ovp, vop, sz::Dims{2})
op * x_r == ovp(x_r)
op' * x_l == vop(x_l)
```

The type parameters `sym`, `herm` and `posdef` are `Bool` values that
declare whether the operator is symmetric, hermitian and positive definite.

Multiplication with matrices applies `ovp` column-wise, unless the
multiplication functions declare
[`MatrixShapedOperators.supports_batched_mul`](@ref) support. Scalar
scaling drops a positive-definiteness declaration since the scalar may be
negative.

Since the operator carries no runtime trait fields and its fields are only
its multiplication functions and its size, it stays compatible with
program tracing (e.g. Reactant tracing/compilation) if `ovp` and `vop`
are.

Use `AutoDiffOperators.with_jacobian(f, x, MatrixFreeOperator, ad)` to
obtain Jacobians as `MatrixFreeOperator`s.

Package extensions provide conversion to `LinearMaps.LinearMap` (via
`LinearMaps.LinearMap(op)`) and `SciMLOperators.FunctionOperator` (via
`SciMLOperators.FunctionOperator(op)`).
"""
struct MatrixFreeOperator{T<:Number,sym,herm,posdef,F,G} <: MatrixShapedOperator{T}
    ovp::F
    vop::G
    sz::Dims{2}

    function MatrixFreeOperator{T,sym,herm,posdef,F,G}(ovp, vop, sz::Dims{2}) where {T<:Number,sym,herm,posdef,F,G}
        _check_real_eltype(T)
        sym isa Bool && herm isa Bool && posdef isa Bool || throw(ArgumentError(
            "MatrixFreeOperator trait type parameters must be Bool values"
        ))
        !(sym || herm || posdef) || sz[1] == sz[2] || throw(ArgumentError(
            "MatrixFreeOperator of size $sz can't be symmetric, hermitian or positive definite"
        ))
        new{T,sym,herm,posdef,F,G}(ovp, vop, sz)
    end
end
export MatrixFreeOperator

function MatrixFreeOperator{T,sym,herm,posdef}(ovp::F, vop::G, sz::Dims{2}) where {T<:Number,sym,herm,posdef,F,G}
    MatrixFreeOperator{T,sym,herm,posdef,F,G}(ovp, vop, sz)
end

function mulfunc_operator(
    ::Type{<:MatrixFreeOperator},
    ::Type{T}, sz::Dims{2}, ovp, vop,
    ::Val{sym}, ::Val{herm}, ::Val{posdef}
) where {T<:Number, sym, herm, posdef}
    MatrixFreeOperator{T,sym,herm,posdef}(ovp, vop, sz)
end


Base.size(op::MatrixFreeOperator) = op.sz

LinearAlgebra.issymmetric(::MatrixFreeOperator{T,sym}) where {T,sym} = sym
LinearAlgebra.ishermitian(::MatrixFreeOperator{T,sym,herm}) where {T,sym,herm} = herm
LinearAlgebra.isposdef(::MatrixFreeOperator{T,sym,herm,posdef}) where {T,sym,herm,posdef} = posdef

function Base.:(==)(a::MatrixFreeOperator, b::MatrixFreeOperator)
    eltype(a) == eltype(b) &&
        issymmetric(a) == issymmetric(b) && ishermitian(a) == ishermitian(b) &&
        isposdef(a) == isposdef(b) &&
        a.ovp == b.ovp && a.vop == b.vop && a.sz == b.sz
end

function Base.show(io::IO, op::MatrixFreeOperator{T,sym,herm,posdef}) where {T,sym,herm,posdef}
    print(io, "MatrixFreeOperator{", T, ",", sym, ",", herm, ",", posdef, "}(")
    show(io, op.ovp)
    print(io, ", ")
    show(io, op.vop)
    print(io, ", ")
    show(io, op.sz)
    print(io, ")")
end

function Base.adjoint(op::MatrixFreeOperator{T,sym,herm,posdef}) where {T,sym,herm,posdef}
    MatrixFreeOperator{T,sym,herm,posdef}(op.vop, op.ovp, reverse(op.sz))
end


function Base.:(*)(op::MatrixFreeOperator, x::AbstractVector{<:Number})
    length(x) == op.sz[2] || throw(DimensionMismatch(
        "operator of size $(op.sz) can't be multiplied with vector of length $(length(x))"
    ))
    return op.ovp(x)
end

function Base.:(*)(op::MatrixFreeOperator{T}, X::AbstractMatrix{<:Number}) where T
    size(X, 1) == op.sz[2] || throw(DimensionMismatch(
        "operator of size $(op.sz) can't be multiplied with matrix of size $(size(X))"
    ))
    supports_batched_mul(op.ovp) && return op.ovp(X)
    size(X, 2) == 0 && return similar(X, promote_type(T, eltype(X)), op.sz[1], 0)
    return _mapcols(op.ovp, X)
end


"""
    MatrixShapedOperators.supports_batched_mul(f)::Bool

Declares whether a multiplication function `f`, as used by
[`MatrixFreeOperator`](@ref) and [`mulfunc_operator`](@ref), can be
applied to matrices directly, treating the matrix columns as a batch of
vectors.

Defaults to `false`, in which case `MatrixFreeOperator` applies `f`
column-wise. Specialize it for multiplication function types that handle
matrix arguments natively (e.g. diagonal operators via broadcasting),
which can be much more efficient, especially under program tracing.
"""
supports_batched_mul(::Any) = false
export supports_batched_mul

supports_batched_mul(f::ComposedFunction) = supports_batched_mul(f.outer) && supports_batched_mul(f.inner)

# Multiplication by a matrix-shaped operator or a matrix accepts matrix
# arguments natively:
supports_batched_mul(::Base.Fix1{typeof(*),<:MatrixShapedOperator}) = true
supports_batched_mul(::Base.Fix1{typeof(*),<:AbstractMatrix{<:Number}}) = true


struct _ScaledFunc{S<:Number,F} <: Function
    s::S
    f::F
end
(sf::_ScaledFunc)(x) = sf.s .* sf.f(x)

supports_batched_mul(sf::_ScaledFunc) = supports_batched_mul(sf.f)

function Base.:(*)(s::Number, op::MatrixFreeOperator{T,sym,herm}) where {T,sym,herm}
    U = promote_type(T, typeof(s))
    MatrixFreeOperator{U,sym,herm,false}(_ScaledFunc(s, op.ovp), _ScaledFunc(s, op.vop), op.sz)
end

function Base.:(*)(s::Number, op::MatrixShapedOperator{T}) where T
    U = promote_type(T, typeof(s))
    MatrixFreeOperator{U,issymmetric(op),ishermitian(op),false}(
        _ScaledFunc(s, Base.Fix1(*, op)), _ScaledFunc(s, Base.Fix1(*, adjoint(op))), size(op)
    )
end

Base.:(*)(op::MatrixShapedOperator, s::Number) = s * op

function Base.:(*)(a::MatrixFreeOperator{T}, b::MatrixFreeOperator{U}) where {T,U}
    a.sz[2] == b.sz[1] || throw(DimensionMismatch(
        "operator of size $(a.sz) can't be composed with operator of size $(b.sz)"
    ))
    MatrixFreeOperator{promote_type(T,U),false,false,false}(
        a.ovp ∘ b.ovp, b.vop ∘ a.vop, (a.sz[1], b.sz[2])
    )
end

function Base.:(*)(a::MatrixShapedOperator{T}, b::MatrixShapedOperator{U}) where {T,U}
    size(a, 2) == size(b, 1) || throw(DimensionMismatch(
        "operator of size $(size(a)) can't be composed with operator of size $(size(b))"
    ))
    MatrixFreeOperator{promote_type(T,U),false,false,false}(
        Base.Fix1(*, a) ∘ Base.Fix1(*, b), Base.Fix1(*, adjoint(b)) ∘ Base.Fix1(*, adjoint(a)),
        (size(a, 1), size(b, 2))
    )
end
