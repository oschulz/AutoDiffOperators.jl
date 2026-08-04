# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

module AutoDiffOperatorsStaticArraysExt

using StaticArrays: StaticArrays, SArray, FieldArray

import AutoDiffOperators

AutoDiffOperators._similar_type(::Type{T}) where {T<:SArray} = T
AutoDiffOperators._similar_type(::Type{T}) where {T<:FieldArray} = T

# The result container keeps the static shape of the second (result)
# argument; the generic vcat-based inference fallback would sum the
# static lengths instead:
AutoDiffOperators._similar_type(::Type{T}, ::Type{U}) where {T<:AbstractVector,U<:Union{SArray,FieldArray}} =
    StaticArrays.similar_type(U, promote_type(eltype(T), eltype(U)))

end # module AutoDiffOperatorsStaticArraysExt
