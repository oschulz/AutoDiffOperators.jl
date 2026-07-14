# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

module AutoDiffOperatorsLinearMapsExt

using LinearMaps

import AutoDiffOperators

using LinearAlgebra


function AutoDiffOperators.mulfunc_operator(
    ::Type{<:Union{LinearMap,FunctionMap}},
    ::Type{T}, sz::Dims{2}, ovp, vop,
    ::Val{sym}, ::Val{herm}, ::Val{posdef}
) where {T<:Real, sym, herm, posdef}
    FunctionMap{T,false}(
        ovp, vop, sz...;
        isposdef=posdef, issymmetric=sym, ishermitian=herm
    )
end


function LinearMaps.FunctionMap{T}(op::AutoDiffOperators.MatrixFreeOperator{T}) where T
    FunctionMap{T,false}(
        op.ovp, op.vop, size(op)...;
        isposdef=isposdef(op), issymmetric=issymmetric(op), ishermitian=ishermitian(op)
    )
end


LinearMaps.FunctionMap(op::AutoDiffOperators.MatrixFreeOperator{T}) where T = LinearMaps.FunctionMap{T}(op)

LinearMaps.LinearMap{T}(op::AutoDiffOperators.MatrixFreeOperator{T}) where T = LinearMaps.FunctionMap{T}(op)
LinearMaps.LinearMap(op::AutoDiffOperators.MatrixFreeOperator{T}) where T = LinearMaps.LinearMap{T}(op)

Base.convert(::Type{LinearMaps.FunctionMap{T}}, op::AutoDiffOperators.MatrixFreeOperator{T}) where T = LinearMaps.FunctionMap{T}(op)
Base.convert(::Type{LinearMaps.FunctionMap}, op::AutoDiffOperators.MatrixFreeOperator) = LinearMaps.FunctionMap(op)
Base.convert(::Type{LinearMaps.LinearMap{T}}, op::AutoDiffOperators.MatrixFreeOperator{T}) where T = LinearMaps.LinearMap{T}(op)
Base.convert(::Type{LinearMaps.LinearMap}, op::AutoDiffOperators.MatrixFreeOperator) = LinearMaps.LinearMap(op)

Base.:(*)(A::LinearMaps.LinearMap{<:Number}, B::AutoDiffOperators.MatrixFreeOperator) = A * LinearMaps.LinearMap(B)
Base.:(*)(A::AutoDiffOperators.MatrixFreeOperator, B::LinearMaps.LinearMap{<:Number}) = LinearMaps.LinearMap(A) * B


end # module AutoDiffOperatorsLinearMapsExt
