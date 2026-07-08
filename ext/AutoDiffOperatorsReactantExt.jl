# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

module AutoDiffOperatorsReactantExt

using Reactant: TracedRNumber, AnyTracedRArray, AbstractConcreteArray

import AutoDiffOperators

# Reactant arrays are traced or device-resident, backend-specific AD
# preparation and function wrappers must not be used with them:
AutoDiffOperators._traced_array_kind(::AnyTracedRArray) = Val(:Reactant)
AutoDiffOperators._traced_array_kind(::AbstractConcreteArray) = Val(:Reactant)
AutoDiffOperators._traced_array_kind(::Array{<:TracedRNumber}) = Val(:Reactant)

end # module AutoDiffOperatorsReactantExt
