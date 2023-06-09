# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import AutoDiffOperators

Test.@testset "Aqua tests" begin
    # ToDo: Fix ambiguities on Julia v1.6:
    Aqua.test_all(AutoDiffOperators, project_toml_formatting=VERSION≥v"1.7", ambiguities=VERSION≥v"1.9")
end # testset
