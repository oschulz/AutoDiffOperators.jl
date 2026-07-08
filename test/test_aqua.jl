# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import AutoDiffOperators
using ADTypes: AbstractADType

Test.@testset "Aqua tests" begin
    Aqua.test_all(
        AutoDiffOperators,
        # `convert(::Type{ADSelector}, ...)` is benign piracy: `ADSelector` is a
        # Union that contains the package-owned `WrappedADSelector`, so colliding
        # method definitions elsewhere would have to reference this package:
        piracies = (treat_as_own = [AbstractADType],)
    )
end # testset
