# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).


"""
    diagonal_operator(d::AbstractVector{<:Number})

Returns a [`WrappedMatrixOperator`](@ref) around
`LinearAlgebra.Diagonal(d)`, so that specializations for `Diagonal`
(e.g. on GPU or under program tracing) take effect.
"""
diagonal_operator(d::AbstractVector{<:Number}) = WrappedMatrixOperator(Diagonal(d))
export diagonal_operator

const _DiagonalWrappedOperator{T} = WrappedMatrixOperator{T,<:Diagonal}


struct _BlockDiagMul{FS,RS} <: Function
    fs::FS
    ranges::RS
end

function (m::_BlockDiagMul)(x::AbstractVecOrMat{<:Number})
    return reduce(vcat, map((f, r) -> f(_sel_rows(x, r)), m.fs, m.ranges))
end

_sel_rows(x::AbstractVector, r::AbstractUnitRange) = x[r]
_sel_rows(X::AbstractMatrix, r::AbstractUnitRange) = X[r, :]

# The blocks are applied as matrix-shaped operators, which accept matrix
# arguments natively:
supports_batched_mul(::_BlockDiagMul) = true

function _block_ranges(lens::NTuple{N,Integer}) where N
    stops = cumsum(lens)
    starts = (1, map(s -> s + 1, Base.front(stops))...)
    return map((a, b) -> a:b, starts, stops)
end

function _block_ranges(lens::AbstractVector{<:Integer})
    stops = cumsum(lens)
    return map((l, b) -> (b - l + 1):b, lens, stops)
end



"""
    blockdiag_operator(blocks...)
    blockdiag_operator(blocks::AbstractVector)

Returns a matrix-shaped operator with the given block-diagonal structure.

The blocks may be matrix-shaped operators or `AbstractMatrix` values and
need not be square, given either as individual arguments (best for a few
blocks of different type) or as a vector (best for many blocks of equal
type). Operators created by [`diagonal_operator`](@ref) combine into a
single diagonal operator, [`RowGramOperator`](@ref) blocks combine into
a single row-Gram operator of the block-diagonal of their factors.
"""
function blockdiag_operator end
export blockdiag_operator

const _BlockLike = Union{MatrixShapedOperator,AbstractMatrix{<:Number}}

blockdiag_operator(b1::_BlockLike) = asoperator(b1)

function blockdiag_operator(b1::_BlockLike, bs::Vararg{_BlockLike,N}) where N
    blocks = (b1, bs...)
    ms = map(b -> size(b, 1), blocks)
    ns = map(b -> size(b, 2), blocks)
    T = promote_type(map(eltype, blocks)...)
    fwd = _BlockDiagMul(map(asoperator, blocks), _block_ranges(ns))
    adj = _BlockDiagMul(map(b -> adjoint(asoperator(b)), blocks), _block_ranges(ms))
    return MatrixFreeOperator{T,false,false,false}(fwd, adj, (sum(ms), sum(ns)))
end

blockdiag_operator(b1::_DiagonalWrappedOperator) = b1

function blockdiag_operator(b1::_DiagonalWrappedOperator, bs::Vararg{_DiagonalWrappedOperator,N}) where N
    return diagonal_operator(vcat(parent(b1).diag, map(b -> parent(b).diag, bs)...))
end

blockdiag_operator(b1::RowGramOperator) = b1

function blockdiag_operator(b1::RowGramOperator, bs::Vararg{RowGramOperator,N}) where N
    return RowGramOperator(blockdiag_operator(gram_factor(b1), map(gram_factor, bs)...))
end


function blockdiag_operator(blocks::AbstractVector{<:_BlockLike})
    isempty(blocks) && throw(ArgumentError("blockdiag_operator requires at least one block"))
    length(blocks) == 1 && return asoperator(only(blocks))
    ms = map(b -> size(b, 1), blocks)
    ns = map(b -> size(b, 2), blocks)
    T = mapreduce(eltype, promote_type, blocks)
    fwd = _BlockDiagMul(map(asoperator, blocks), _block_ranges(ns))
    adj = _BlockDiagMul(map(b -> adjoint(asoperator(b)), blocks), _block_ranges(ms))
    return MatrixFreeOperator{T,false,false,false}(fwd, adj, (sum(ms), sum(ns)))
end

function blockdiag_operator(blocks::AbstractVector{<:_DiagonalWrappedOperator})
    isempty(blocks) && throw(ArgumentError("blockdiag_operator requires at least one block"))
    return diagonal_operator(reduce(vcat, map(b -> parent(b).diag, blocks)))
end

function blockdiag_operator(blocks::AbstractVector{<:RowGramOperator})
    isempty(blocks) && throw(ArgumentError("blockdiag_operator requires at least one block"))
    length(blocks) == 1 && return only(blocks)
    return RowGramOperator(blockdiag_operator(map(gram_factor, blocks)))
end
