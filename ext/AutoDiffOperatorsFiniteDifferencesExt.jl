# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

module AutoDiffOperatorsFiniteDifferencesExt

using FiniteDifferences

import AutoDiffOperators
using ADTypes: AutoFiniteDifferences


const default_method = FiniteDifferences.central_fdm(5, 1)
AutoDiffOperators._adsel_finitedifferences(::Val{true}) = AutoFiniteDifferences(fdm = default_method)


end # module AutoDiffOperatorsFiniteDifferencesExt
