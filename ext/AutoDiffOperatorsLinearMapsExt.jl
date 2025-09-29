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


end # module AutoDiffOperatorsLinearMapsExt
