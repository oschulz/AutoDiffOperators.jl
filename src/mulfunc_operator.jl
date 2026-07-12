# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

"""
    AutoDiffOperators.mulfunc_operator(
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
    struct MulFuncOperator{T<:Number,sym,herm,posdef,F,G}

Linear operator that multiplies via a function `ovp` and adjoint-multiplies
via a function `vop`.

Constructor:

```julia
op = MulFuncOperator{T,sym,herm,posdef}(ovp, vop, sz::Dims{2})
op * x_r == ovp(x_r)
op' * x_l == vop(x_l)
```

The type parameters `sym`, `herm` and `posdef` are `Bool` values that
declare whether the operator is symmetric, hermitian and positive definite.

The operator supports multiplication and adjoint multiplication with
vectors and matrices (applied column-wise), scaling with real scalars
(scaling drops a positive-definiteness declaration since the scalar may be
negative), composition with other `MulFuncOperator`s via `*`,
materialization via `Matrix(op)` and `LinearAlgebra.mul!`.

Operator traits are type parameters and multiplication allocates no
intermediate arrays that are mutated afterwards, so operators of this type
are compatible with program tracing (e.g. Reactant tracing/compilation) if
their `ovp` and `vop` functions are.

Use [`with_jacobian`](@ref)`(f, x, MulFuncOperator, ad)` to obtain
Jacobians as `MulFuncOperator`s.

Package extensions provide conversion to `LinearMaps.LinearMap` (via
`LinearMaps.LinearMap(op)`) and `SciMLOperators.FunctionOperator` (via
`SciMLOperators.FunctionOperator(op)`).
"""
struct MulFuncOperator{T<:Number,sym,herm,posdef,F,G}
    ovp::F
    vop::G
    sz::Dims{2}

    function MulFuncOperator{T,sym,herm,posdef,F,G}(ovp, vop, sz::Dims{2}) where {T<:Number,sym,herm,posdef,F,G}
        T <: Real || real(T) === T || throw(ArgumentError(
            "MulFuncOperator only supports real-valued numbers, got eltype $T"
        ))
        sym isa Bool && herm isa Bool && posdef isa Bool || throw(ArgumentError(
            "MulFuncOperator trait type parameters must be Bool values"
        ))
        !(sym || herm || posdef) || sz[1] == sz[2] || throw(ArgumentError(
            "MulFuncOperator of size $sz can't be symmetric, hermitian or positive definite"
        ))
        new{T,sym,herm,posdef,F,G}(ovp, vop, sz)
    end
end
export MulFuncOperator

function MulFuncOperator{T,sym,herm,posdef}(ovp::F, vop::G, sz::Dims{2}) where {T<:Number,sym,herm,posdef,F,G}
    MulFuncOperator{T,sym,herm,posdef,F,G}(ovp, vop, sz)
end

function mulfunc_operator(
    ::Type{<:MulFuncOperator},
    ::Type{T}, sz::Dims{2}, ovp, vop,
    ::Val{sym}, ::Val{herm}, ::Val{posdef}
) where {T<:Number, sym, herm, posdef}
    MulFuncOperator{T,sym,herm,posdef}(ovp, vop, sz)
end


Base.eltype(::Type{<:MulFuncOperator{T}}) where T = T

Base.size(op::MulFuncOperator) = op.sz

function Base.size(op::MulFuncOperator, d::Integer)
    d >= 1 || throw(ArgumentError("dimension out of range, got $d"))
    return d <= 2 ? op.sz[d] : 1
end

LinearAlgebra.issymmetric(::MulFuncOperator{T,sym}) where {T,sym} = sym
LinearAlgebra.ishermitian(::MulFuncOperator{T,sym,herm}) where {T,sym,herm} = herm
LinearAlgebra.isposdef(::MulFuncOperator{T,sym,herm,posdef}) where {T,sym,herm,posdef} = posdef

function Base.:(==)(a::MulFuncOperator, b::MulFuncOperator)
    eltype(a) == eltype(b) &&
        issymmetric(a) == issymmetric(b) && ishermitian(a) == ishermitian(b) &&
        isposdef(a) == isposdef(b) &&
        a.ovp == b.ovp && a.vop == b.vop && a.sz == b.sz
end

function Base.show(io::IO, op::MulFuncOperator{T,sym,herm,posdef}) where {T,sym,herm,posdef}
    print(io, "MulFuncOperator{", T, ",", sym, ",", herm, ",", posdef, "}(")
    show(io, op.ovp)
    print(io, ", ")
    show(io, op.vop)
    print(io, ", ")
    show(io, op.sz)
    print(io, ")")
end

function Base.adjoint(op::MulFuncOperator{T,sym,herm,posdef}) where {T,sym,herm,posdef}
    MulFuncOperator{T,sym,herm,posdef}(op.vop, op.ovp, reverse(op.sz))
end

Base.transpose(op::MulFuncOperator) = adjoint(op)


function Base.:(*)(op::MulFuncOperator, x::AbstractVector{<:Number})
    length(x) == op.sz[2] || throw(DimensionMismatch(
        "operator of size $(op.sz) can't be multiplied with vector of length $(length(x))"
    ))
    return op.ovp(x)
end

function Base.:(*)(op::MulFuncOperator{T}, X::AbstractMatrix{<:Number}) where T
    size(X, 1) == op.sz[2] || throw(DimensionMismatch(
        "operator of size $(op.sz) can't be multiplied with matrix of size $(size(X))"
    ))
    size(X, 2) == 0 && return similar(X, promote_type(T, eltype(X)), op.sz[1], 0)
    return _mapcols(op.ovp, X)
end

_mapcols(f, X::AbstractMatrix) = reduce(hcat, [f(X[:, j]) for j in axes(X, 2)])

function Base.:(*)(x_l::LinearAlgebra.Adjoint{<:Number,<:AbstractVector{<:Number}}, op::MulFuncOperator)
    return adjoint(adjoint(op) * adjoint(x_l))
end

function Base.:(*)(x_l::LinearAlgebra.Transpose{<:Number,<:AbstractVector{<:Number}}, op::MulFuncOperator)
    return transpose(transpose(op) * transpose(x_l))
end


struct _ScaledFunc{S<:Number,F} <: Function
    s::S
    f::F
end
(sf::_ScaledFunc)(x) = sf.s .* sf.f(x)

function Base.:(*)(s::Number, op::MulFuncOperator{T,sym,herm}) where {T,sym,herm}
    U = promote_type(T, typeof(s))
    MulFuncOperator{U,sym,herm,false}(_ScaledFunc(s, op.ovp), _ScaledFunc(s, op.vop), op.sz)
end

Base.:(*)(op::MulFuncOperator, s::Number) = s * op

function Base.:(*)(a::MulFuncOperator{T}, b::MulFuncOperator{U}) where {T,U}
    a.sz[2] == b.sz[1] || throw(DimensionMismatch(
        "operator of size $(a.sz) can't be composed with operator of size $(b.sz)"
    ))
    MulFuncOperator{promote_type(T,U),false,false,false}(
        a.ovp ∘ b.ovp, b.vop ∘ a.vop, (a.sz[1], b.sz[2])
    )
end


function LinearAlgebra.mul!(y::AbstractVecOrMat{<:Number}, op::MulFuncOperator, x::AbstractVecOrMat{<:Number})
    return mul!(y, op, x, true, false)
end

function LinearAlgebra.mul!(
    y::AbstractVecOrMat{<:Number}, op::MulFuncOperator, x::AbstractVecOrMat{<:Number},
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


function Base.Matrix(op::MulFuncOperator{T,sym,herm,posdef}) where {T,sym,herm,posdef}
    return mulfunc_operator(Matrix, T, op.sz, op.ovp, op.vop, Val(sym), Val(herm), Val(posdef))
end

Base.convert(::Type{Matrix}, op::MulFuncOperator) = Matrix(op)
