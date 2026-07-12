# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

module AutoDiffOperatorsSciMLOperatorsExt

using SciMLOperators

import AutoDiffOperators

using LinearAlgebra


# Adapts a plain multiplication function to the SciMLOperators out-of-place
# operator application signature:
struct _OpApplyFunc{F} <: Function
    f::F
end
(g::_OpApplyFunc)(v, u, p, t) = g.f(v)

function AutoDiffOperators.mulfunc_operator(
    ::Type{<:SciMLOperators.AbstractSciMLOperator},
    ::Type{T}, sz::Dims{2}, ovp, vop,
    ::Val{sym}, ::Val{herm}, ::Val{posdef}
) where {T<:Real, sym, herm, posdef}
    SciMLOperators.FunctionOperator(
        _OpApplyFunc(ovp), Vector{T}(undef, sz[2]), Vector{T}(undef, sz[1]);
        op_adjoint=_OpApplyFunc(vop),
        islinear=true, isconstant=true, isinplace=false, outofplace=true,
        issymmetric=sym, ishermitian=herm, isposdef=posdef
    )
end


function SciMLOperators.FunctionOperator(op::AutoDiffOperators.MulFuncOperator{T}) where T
    AutoDiffOperators.mulfunc_operator(
        SciMLOperators.AbstractSciMLOperator, T, size(op), op.ovp, op.vop,
        Val(issymmetric(op)), Val(ishermitian(op)), Val(isposdef(op))
    )
end

Base.convert(::Type{SciMLOperators.AbstractSciMLOperator}, op::AutoDiffOperators.MulFuncOperator) = SciMLOperators.FunctionOperator(op)


end # module AutoDiffOperatorsSciMLOperatorsExt
