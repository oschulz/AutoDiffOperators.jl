# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).


"""
    op_mul(op::MatrixLikeOperator, x::AbstractVector{<:Number})

Returns `op * x`.

For details, see [`MatrixLikeOperator`](@ref).
"""
function op_mul end


"""
    adjoint_op_mul(op::MatrixLikeOperator, x::AbstractVector{<:Number})

Returns `op' * x`.

For details, see [`MatrixLikeOperator`](@ref).
"""
function adjoint_op_mul end


"""
    abstract type MatrixLikeOperator{T<:Number} <: AbstractMatrix{T} end

Abstract type for matrix-like operators that support multiplication with
vectors.

An `op::MatrixLikeOperator` can be constructed from a multiplication function
`op_mul(x)` that implements `op * x`, and ajoint multiplication function
`adjoint_op_mul(x)` that implements `op' * x`, and a size `sz` via

```julia
MatrixLikeOperator(
    ovp, op_mul_function, adjoint_op_mul_function, sz::Dims
    issymmetric::Bool, ishermitian::Bool, isposdef::Bool
)
```

Typically specialized subtypes of `MatrixLikeOperator` should be used for
specific kinds of operators, though.

MatrixLikeOperator supports conversion to
[LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl) operator
types via constructor and `convert` methods:

```julia
LinearMaps.FunctionMap(op::MatrixLikeOperator) isa LinearMaps.FunctionMap
LinearMaps.LinearMap(op::MatrixLikeOperator) isa LinearMaps.FunctionMap
```

Subtypes must implement:

* `Base.size(op::MatrixLikeOperator)`
* `AutoDiffOperators.op_mul(op::MatrixLikeOperator, x::AbstractVector{<:Number})`
* `AutoDiffOperators.adjoint_op_mul(op::MatrixLikeOperator, x::AbstractVector{<:Number})`

And may implement, resp. specialize (if possible):

* `Base.adjoint(op::MatrixLikeOperator)`
* `Base.transpose(op::MatrixLikeOperator)`
* `AutoDiffOperators.transpose_op_mul(op::MatrixLikeOperator, x::AbstractVector{<:Number})`
"""
abstract type MatrixLikeOperator{T<:Number,sym,herm,posdef} <: AbstractMatrix{T} end
export MatrixLikeOperator


function MatrixLikeOperator{T,sym,herm,posdef}(ovp::F, vop::G, sz::Dims) where {T<:Number,sym,herm,posdef,F,G}
    _MulFuncOperator{T,sym,herm,posdef,F,G}(ovp, vop, sz)
end

MatrixLikeOperator{T,sym,herm,posdef}(A::MatrixLikeOperator{T,sym,herm,posdef}) where {T<:Number,sym,herm,posdef} = A

function MatrixLikeOperator{T,sym,herm,posdef}(A::AbstractMatrix{T})  where {T<:Number,sym,herm,posdef}
    _MulFuncOperator{T, sym, herm, posdef}(Base.Fix1(*, A), Base.Fix1(*, A'), size(A))
end

MatrixLikeOperator(A::MatrixLikeOperator{T}) where T = A

MatrixLikeOperator(A::AbstractMatrix{T}) where T = MatrixLikeOperator{T,false,false,false}(A)

@inline Base.IndexStyle(::Type{<:MatrixLikeOperator}) = IndexCartesian()

LinearAlgebra.issymmetric(::MatrixLikeOperator{T,sym,herm,posdef}) where {T,sym,herm,posdef} = sym
LinearAlgebra.ishermitian(::MatrixLikeOperator{T,sym,herm,posdef}) where {T,sym,herm,posdef} = herm
LinearAlgebra.isposdef(::MatrixLikeOperator{T,sym,herm,posdef}) where {T,sym,herm,posdef} = posdef

Base.transpose(op::MatrixLikeOperator{<:Real}) = adjoint(op)
transpose_op_mul(op::MatrixLikeOperator{<:Real}, x::AbstractVector{<:Real}) = adjoint_op_mul(op, x)


"""
    struct AutoDiffOperators.AdjointMatrixLikeOperator{T<:Number,sym,herm,posdef} <: MatrixLikeOperator{T,sym,herm,posdef}

Represents the adjoint of a [`MatrixLikeOperator`](@ref).

User code should not instantiate this type directly, use `adjoint(::MatrixLikeOperator)` instead.
"""
struct AdjointMatrixLikeOperator{T<:Number,sym,herm,posdef,OP<:MatrixLikeOperator{T,sym,herm,posdef}} <: MatrixLikeOperator{T,sym,herm,posdef}
    parent::OP
end

@inline Base.parent(op::AdjointMatrixLikeOperator) = op.parent
@inline Base.size(op::AdjointMatrixLikeOperator) = reverse(size(parent(op)))

@inline Base.adjoint(op::MatrixLikeOperator) = AdjointMatrixLikeOperator(op)
@inline Base.adjoint(op::AdjointMatrixLikeOperator) = parent(op)

@inline op_mul(op::AdjointMatrixLikeOperator, x::AbstractVector{<:Number}) = adjoint_op_mul(parent(op), x)
@inline adjoint_op_mul(op::AdjointMatrixLikeOperator, x::AbstractVector{<:Number}) = op_mul(parent(op), x)


const _AdjointNumVector{T} = LinearAlgebra.Adjoint{T,<:AbstractVector{T}}

_jvp_func(op::MatrixLikeOperator) = Base.Fix1(op_mul, op)
_vjp_func(op::MatrixLikeOperator) = Base.Fix1(adjoint_op_mul, op)

_jvp_func(op::AdjointMatrixLikeOperator) = _vjp_func(op')
_vjp_func(op::AdjointMatrixLikeOperator) = _jvp_func(op')



@inline Base.:(*)(a::Number, b::MatrixLikeOperator) = _op_mul(a, b)
@inline Base.:(*)(a::MatrixLikeOperator, b::Number) = _op_mul(a, b)
@inline Base.:(*)(a::AbstractVector{<:Number}, b::MatrixLikeOperator) = _op_mul(a, b)
@inline Base.:(*)(a::MatrixLikeOperator, b::AbstractVector{<:Number}) = _op_mul(a, b)
@inline Base.:(*)(a::AbstractMatrix{<:Number}, b::MatrixLikeOperator) = _op_mul(a, b)
@inline Base.:(*)(a::MatrixLikeOperator, b::AbstractMatrix{<:Number}) = _op_mul(a, b)
@inline Base.:(*)(a::MatrixLikeOperator, b::MatrixLikeOperator) = _op_mul(a, b)

# Disambiguation:
@inline Base.:(*)(a::MatrixLikeOperator, b::Diagonal{<:Number}) = _op_mul(a, b)
@inline Base.:(*)(a::MatrixLikeOperator, b::LinearAlgebra.AbstractTriangular{<:Number}) = _op_mul(a, b)
@inline Base.:(*)(a::Diagonal{<:Number}, b::MatrixLikeOperator) = _op_mul(a, b)
@inline Base.:(*)(a::LinearAlgebra.AbstractTriangular{<:Number}, b::MatrixLikeOperator) = _op_mul(a, b)

@static if VERSION < v"1.9"
    # Julia v1.6 Disambiguation:
    const _LinAlgRHSCH = LinearAlgebra.RealHermSymComplexHerm
    const _LinAlgRHSCS = LinearAlgebra.RealHermSymComplexSym
    @inline Base.:(*)(a::MatrixLikeOperator, b::LinearAlgebra.Adjoint{<:Number,<:_LinAlgRHSCH}) = _op_mul(a, b)
    @inline Base.:(*)(a::LinearAlgebra.Adjoint{<:Number,<:_LinAlgRHSCH}, b::MatrixLikeOperator) = _op_mul(a, b)
    @inline Base.:(*)(a::MatrixLikeOperator, b::LinearAlgebra.Adjoint{<:Number,<:LinearAlgebra.AbstractRotation}) = _op_mul(a, b)
    @inline Base.:(*)(a::LinearAlgebra.Adjoint{<:Number,<:LinearAlgebra.AbstractRotation}, b::MatrixLikeOperator) = _op_mul(a, b)
    @inline Base.:(*)(a::MatrixLikeOperator, b::LinearAlgebra.Transpose{<:Number,<:_LinAlgRHSCS}) = _op_mul(a, b)
    @inline Base.:(*)(a::LinearAlgebra.Transpose{<:Number,<:_LinAlgRHSCS}, b::MatrixLikeOperator) = _op_mul(a, b)
end

@inline Base.:(*)(a::LinearAlgebra.Transpose{<:Number,<:AbstractVector}, b::MatrixLikeOperator) = _op_mul(a, b)
@inline Base.:(*)(a::LinearAlgebra.Adjoint{<:Number,<:AbstractVector}, b::MatrixLikeOperator) = _op_mul(a, b)

@inline function _op_mul(s::Number, A::MatrixLikeOperator{T,sym,herm,posdef}) where {T,sym,herm,posdef}
    MatrixLikeOperator{promote_type(T,typeof(s)), sym, herm, posdef}(
        Base.Fix1(*, s) ∘ _jvp_func(A),
        _vjp_func(A) ∘ Base.Fix1(*, s),
        size(A)
    )
end
@inline _op_mul(A::MatrixLikeOperator, s::Number) = _op_mul(s, A)
@inline _op_mul(s::Number, A::AdjointMatrixLikeOperator) = AdjointMatrixLikeOperator(_op_mul(s, parent(A)))

@inline _op_mul(op::MatrixLikeOperator, x_r::AbstractVector{<:Number}) = op_mul(op, x_r)

@inline function _op_mul(x_l::LinearAlgebra.Adjoint{<:Number,<:AbstractVector}, op::MatrixLikeOperator)
    adjoint(adjoint_op_mul(op, adjoint(x_l)))
end

@inline function _op_mul(x_l::LinearAlgebra.Transpose{<:Number,<:AbstractVector}, op::MatrixLikeOperator)
    transpose(transpose_op_mul(op, transpose(x_l)))
end

@inline _op_mul(A::MatrixLikeOperator, B::AbstractMatrix{<:Number}) = _op_mul(A, MatrixLikeOperator(B))
@inline _op_mul(A::AbstractMatrix{<:Number}, B::MatrixLikeOperator) = _op_mul(MatrixLikeOperator(A), B)

function _op_mul(
    A::MatrixLikeOperator{T,sym,herm,posdef},
    B::MatrixLikeOperator{U,sym2,herm2,posdef2}
) where {T<:Number,sym,herm,posdef,U<:Number,sym2,herm2,posdef2}
    MatrixLikeOperator{promote_type(T,U), sym && sym2, herm && herm2, posdef && posdef2}(
        _jvp_func(A) ∘ _jvp_func(B),
        _vjp_func(B) ∘ _vjp_func(A),
        (size(A, 1), size(B, 2))
    )
end



Base.@propagate_inbounds LinearAlgebra.mul!(Y::AbstractMatrix{<:Number}, A::MatrixLikeOperator, X::Number) = _op_mul!(Y, A, X)
Base.@propagate_inbounds LinearAlgebra.mul!(Y::AbstractVector{<:Number}, A::MatrixLikeOperator, X::AbstractVector{<:Number}) = _op_mul!(Y, A, X)
Base.@propagate_inbounds LinearAlgebra.mul!(Y::AbstractMatrix{<:Number}, A::MatrixLikeOperator, X::AbstractMatrix{<:Number}) = _op_mul!(Y, A, X)
Base.@propagate_inbounds LinearAlgebra.mul!(Y::AbstractMatrix{<:Number}, A::Number, X::MatrixLikeOperator) = _op_mul!(Y, A, X)
Base.@propagate_inbounds LinearAlgebra.mul!(Y::AbstractMatrix{<:Number}, A::AbstractMatrix{<:Number}, X::MatrixLikeOperator) = _op_mul!(Y, A, X)
Base.@propagate_inbounds LinearAlgebra.mul!(Y::AbstractMatrix{<:Number}, A::MatrixLikeOperator, X::MatrixLikeOperator) = _op_mul!(Y, A, X)
# Disambiguation:
Base.@propagate_inbounds LinearAlgebra.mul!(Y::AbstractMatrix{<:Number}, A::LinearAlgebra.AbstractTriangular{<:Number}, X::MatrixLikeOperator) = _op_mul!(Y, A, X)
Base.@propagate_inbounds LinearAlgebra.mul!(Y::AbstractMatrix{<:Number}, A::MatrixLikeOperator, X::LinearAlgebra.AbstractTriangular{<:Number}) = _op_mul!(Y, A, X)

Base.@propagate_inbounds function _op_mul!(y::AbstractMatrix{<:Number}, op::MatrixLikeOperator, s::Number)
    return copyto!(y, op * s)
end

Base.@propagate_inbounds function _op_mul!(y::AbstractMatrix{<:Number}, s::Number, op::MatrixLikeOperator)
    return _op_mul!(y, op, s)
end

Base.@propagate_inbounds function _op_mul!(y::AbstractVector{<:Number}, op::MatrixLikeOperator, x::AbstractVector{<:Number})
    @boundscheck _mul!_dimcheck(y, op, x)
    return copyto!(y, op * x)
end

Base.@propagate_inbounds function _op_mul!(Y::AbstractMatrix{<:Number}, op::MatrixLikeOperator, op2::MatrixLikeOperator)
    @boundscheck _mul!_dimcheck(Y, op, op2)
    return copyto!(Y, op * op2)
end

Base.@propagate_inbounds function _op_mul!(Y::AbstractMatrix{<:Number}, op::MatrixLikeOperator, X::AbstractMatrix{<:Number})
    @boundscheck _mul!_dimcheck(Y, op, X)
    _op_mul_impl!(Y, op, X)
end

function _op_mul_impl!(Y::AbstractMatrix{<:Number}, op::MatrixLikeOperator, X::AbstractMatrix{<:Number})
    first_k_Y = firstindex(X, 1)
    first_k_X = firstindex(X, 2)
    Base.Threads.@threads for k in Base.OneTo(size(Y, 2))
        y = view(Y, :, k-1 + first_k_Y)
        x = view(X, :, k-1 + first_k_X)
        _op_mul!(y, op, x)
    end
    return Y
end

Base.@propagate_inbounds function _op_mul!(Y::AbstractMatrix{<:Number}, X::AbstractMatrix{<:Number}, op::MatrixLikeOperator)
    # return copyto!(Y, X * op)
    _op_mul!(Y', op', X')
    return Y
end


function _mul!_dimcheck(C::AbstractVecOrMat, A::AbstractMatrix, B::AbstractVecOrMat)
    if (size(B, 1) != size(A, 2) || size(C, 1) != size(A, 1) || size(C, 2) != size(B, 2))
        throw(DimensionMismatch("mul!(C,A,B) with size(A) = $(size(A)), size(B) = $(size(B)), size(C) = $(size(C))"))
    end
    return nothing
end



Base.@propagate_inbounds Base.copyto!(A::AbstractMatrix{<:Number}, op::MatrixLikeOperator) = _op_copyto!(A, op)
# Disambiguation:
Base.@propagate_inbounds Base.copyto!(A::PermutedDimsArray{<:Number,2}, op::MatrixLikeOperator) = _op_copyto!(A, op)

Base.@propagate_inbounds function _op_copyto!(A::AbstractMatrix{<:Number}, op::MatrixLikeOperator)
    @boundscheck size(A) == size(op) || throw(DimensionMismatch("copyto!(A, op) with size(A) = $(size(A)), size(op) = $(size(op))"))
    _op_copyto_impl!(A, op)
end


function _op_copyto_impl!(A::AbstractMatrix{<:Number}, op::MatrixLikeOperator)
    first_j_A = firstindex(A, 1)
    first_j_op = firstindex(op, 1)
    Base.Threads.@threads for j in Base.OneTo(size(op, 2))
        _get_column!(view(A, :, j-1+first_j_A), op, j-1+first_j_op)
    end
    return A
end

function _op_copyto_impl!(A::AbstractMatrix{<:Number}, op::AdjointMatrixLikeOperator)
    _op_copyto_impl!(A', op')
    return A
end


Base.@propagate_inbounds function _get_column!(c::AbstractVector{<:Number}, op::MatrixLikeOperator, j::Int)
    @boundscheck begin
        size(c, 1) == size(op, 1) || throw(DimensionMismatch("get_column!(c, op, j) with size(c) = $(size(c)), size(op) = $(size(op)), j = $j"))
        j in axes(op, 2) || throw(BoundsError(op, (:,j)))
    end
    _get_column_impl!(c, op, j)
end

function _get_column_impl!(c::AbstractVector{<:Number}, op::MatrixLikeOperator, j::Int)
    x = similar(c, (size(op, 2),))
    fill!(x, zero(eltype(x)))
    x[j] = one(eltype(x))
    mul!(c, op, x)
end

Base.@propagate_inbounds function _get_row!(c::AbstractVector{<:Number}, op::MatrixLikeOperator, i::Int)
    @boundscheck begin
        size(c, 1) == size(op, 2) || throw(DimensionMismatch("get_row!(c, op, i) with size(c) = $(size(c)), size(op) = $(size(op)), i = $i"))
        i in axes(op, 1) || throw(BoundsError(op, (i,:)))
    end
    _get_row_impl!(c, op, i)
end

function _get_row_impl!(c::AbstractVector{<:Number}, op::MatrixLikeOperator, i::Int)
    x = similar(c, (size(op, 1),))
    fill!(x, zero(eltype(x)))
    x[i] = one(eltype(x))
    mul!(c', x', op)
    return c
end 


_to_index_keep_colon(I) = Base.to_index(I)
_to_index_keep_colon(I::Colon) = I

Base.@propagate_inbounds function Base.getindex(A::MatrixLikeOperator, I, J)
    @boundscheck checkbounds(A, I, J)
    _getindex_impl(A, _to_index_keep_colon(I), _to_index_keep_colon(J))
end


function _getindex_impl(A::MatrixLikeOperator, i::Integer, j::Integer)
    throw(ArgumentError("$(nameof(typeof(A))) doesn't support element-wise getindex"))
end

function _getindex_impl(op::MatrixLikeOperator{T}, ::Colon, j::Int) where T
    c = Vector{T}(undef, size(op, 1))
    return _get_column!(c, op, j)
end

function _getindex_impl(op::MatrixLikeOperator, I::AbstractVector{<:Integer}, j::Int)
    _getindex_impl(op, :, j)[I]
end

function _getindex_impl(op::MatrixLikeOperator{T}, i::Int, ::Colon) where T
    c = Vector{T}(undef, size(op, 2))
    return _get_row!(c, op, i)
end

function _getindex_impl(op::MatrixLikeOperator, i::Int, J::AbstractVector{<:Integer})
    _getindex_impl(op, i, :)[J]
end

function _getindex_impl(op::MatrixLikeOperator{T}, ::Colon, ::Colon) where T
    A = Matrix{T}(undef, size(op))
    return copyto!(A, op)
end

# ToDo: Optimize implementation?
function _getindex_impl(op::MatrixLikeOperator{T}, I::Union{Colon,AbstractVector{<:Integer}}, J::Union{Colon,AbstractVector{<:Integer}}) where T
    _getindex_impl(op, :, :)[I, J]
end


struct _MulFuncOperator{T<:Number,sym,herm,posdef,F,G} <: MatrixLikeOperator{T,sym,herm,posdef}
    ovp::F
    vop::G
    sz::Dims
end

function _MulFuncOperator{T,sym,herm,posdef}(ovp::F, vop::G, sz::Dims) where {T<:Number,sym,herm,posdef,F,G}
    _MulFuncOperator{T,sym,herm,posdef,F,G}(ovp, vop, sz)
end

_jvp_func(op::_MulFuncOperator) = op.ovp
_vjp_func(op::_MulFuncOperator) = op.vop

Base.size(op::_MulFuncOperator) = op.sz
op_mul(op::_MulFuncOperator, x::AbstractVector{<:Number}) = op.ovp(x)
adjoint_op_mul(op::_MulFuncOperator, x::AbstractVector{<:Number}) = op.vop(x)


function Base.show(io::IO, m::MIME"text/plain", op::_MulFuncOperator{T}) where T
    print(io, nameof(MatrixLikeOperator{T}), "{", T, "}", "(",)
    show(io, op.ovp)
    print(io, " ,")
    show(io, op.vop)
    print(io, " ,")
    show(io, op.sz)
    print(io, ")")
end
