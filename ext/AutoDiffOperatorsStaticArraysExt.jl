# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

module AutoDiffOperatorsStaticArraysExt

using StaticArrays: SArray, FieldArray

import AutoDiffOperators

AutoDiffOperators._similar_type(::Type{T}) where {T<:SArray} = T
AutoDiffOperators._similar_type(::Type{T}) where {T<:FieldArray} = T

end # module AutoDiffOperatorsStaticArraysExt
