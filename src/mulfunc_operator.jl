# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

"""
    AutoDiffOperators.mulfunc_operator(
        ::Type{OP},
        ::Type{T}, sz::Dims{2}, ovp, vop,
        ::Val{sym}, ::Val{herm}, ::Val{posdef}
    ) where {OP, T<:Real, sym, herm, posdef}

Generates a linear operator object of type `OP` that supports multiplication
and with (adjoint) vectors based on a multiplication function `ovp` and an
adjoint multiplication function `vop`.

An operator
`op = mulfunc_operator(OP, T, sz, ovp, vop, Val(sym), ::Val{herm}, ::Val{posdef})`
must show show following behavior:

```julia
op isa OP
eltype(op) == T
size(op) == sz
op * x_r == ovp(x_r)
x_l' * op == vop(x_l)
issymmetric(op) == sym
ishermitian(op) == herm
isposdef(op) = posdef
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
    first_j_A = firstindex(A, 2)
    # Enzyme doesn't like `Duplicated` with mixed views/non-views,
    # so allocate separate vectors for each thread instead of
    # an `Matrix{T}(undef, sz[2], nthreads())`. Try using UnsafeArrays
    # here instead of standard views?
    xs = [Vector{T}(undef, sz[2]) for i in 1:nthreads()]
    for x in xs
        fill!(x, zero(T))
    end
    @threads for j in Base.OneTo(sz[2])
        col = view(A, :, j-1+first_j_A)
        x = xs[threadid()]
        first_j_x = firstindex(x, 1)
        j_x = j-1+first_j_x
        x[j_x] = one(T)
        col .= ovp(x)
        x[j_x] = zero(T)
    end
    return A
end
