# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import AutoDiffOperators

Test.@testset "Package ambiguities" begin
    # Test.@test isempty(Test.detect_ambiguities(AutoDiffOperators))
end # testset

Test.@testset "Aqua tests" begin
    Aqua.test_all(
        AutoDiffOperators,
        ambiguities = false,
        # Piracy detection is triggered incorrectly by `Base.convert(::Type{ADSelector}, m)`:
        piracies = false
    )
end # testset
