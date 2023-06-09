# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

module AutoDiffOperatorsLinearMapsExt

@static if isdefined(Base, :get_extension)
    using LinearMaps
else
    using ..LinearMaps
end

import AutoDiffOperators

using LinearAlgebra


LinearMaps.LinearMap{T}(op::AutoDiffOperators.MatrixLikeOperator{T}) where T = LinearMaps.FunctionMap{T}(op)
LinearMaps.LinearMap(op::AutoDiffOperators.MatrixLikeOperator{T}) where T = LinearMaps.LinearMap{T}(op)

Base.convert(::Type{LinearMaps.LinearMap{T}}, op::AutoDiffOperators.MatrixLikeOperator{T}) where T = LinearMaps.LinearMap{T}(op)
Base.convert(::Type{LinearMaps.LinearMap}, op::AutoDiffOperators.MatrixLikeOperator{T}) where T = LinearMaps.LinearMap(op)


function LinearMaps.FunctionMap{T}(op::AutoDiffOperators.MatrixLikeOperator{T}) where T
    FunctionMap{T,false}(
        AutoDiffOperators._jvp_func(op), AutoDiffOperators._vjp_func(op), size(op)...;
        isposdef=isposdef(op), issymmetric=issymmetric(op), ishermitian=ishermitian(op)#
    )
end

LinearMaps.FunctionMap(op::AutoDiffOperators.MatrixLikeOperator{T}) where T = LinearMaps.FunctionMap{T}(op)

Base.convert(::Type{LinearMaps.FunctionMap{T}}, op::AutoDiffOperators.MatrixLikeOperator{T}) where T = LinearMaps.FunctionMap{T}(op)
Base.convert(::Type{LinearMaps.FunctionMap}, op::AutoDiffOperators.MatrixLikeOperator{T}) where T = LinearMaps.FunctionMap(op)


Base.:(*)(A::LinearMaps.LinearMap{<:Number}, B::AutoDiffOperators.MatrixLikeOperator) = A * LinearMaps.LinearMap(B)
Base.:(*)(A::AutoDiffOperators.MatrixLikeOperator, B::LinearMaps.LinearMap{<:Number}) = LinearMaps.LinearMap(A) * B

end # module LinearMaps
