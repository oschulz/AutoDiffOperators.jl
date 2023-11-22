# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

import Test
import Aqua
import AutoDiffOperators

Test.@testset "Aqua tests" begin
    Aqua.test_all(AutoDiffOperators)
end # testset
