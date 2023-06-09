# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package AutoDiffOperators" begin
    include("test_aqua.jl")
    include("test_matrix_like_operator.jl")
    include("test_finitedifferences.jl")
    include("test_forwarddiff.jl")
    include("test_zygote.jl")
    include("test_enzyme.jl")
    include("test_linear_maps.jl")
    include("test_docs.jl")
    isempty(Test.detect_ambiguities(AutoDiffOperators))
end # testset
