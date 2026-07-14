# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

module AutoDiffOperatorsStatisticsExt

using Statistics

import AutoDiffOperators
using AutoDiffOperators: MatrixShapedOperator, MatrixShapedSum, UniformScalingOperator


# The generic mean would fold pairwise sums and require operator division,
# use `inv(n) * I` times a single sum instead:
function _mean_operator(ops)
    s = MatrixShapedSum(ops)
    T = eltype(s)
    return UniformScalingOperator(inv(T(length(ops))), size(s, 1)) * s
end

Statistics.mean(ops::AbstractVector{<:MatrixShapedOperator}) = _mean_operator(ops)
Statistics.mean(ops::Tuple{MatrixShapedOperator,Vararg{MatrixShapedOperator}}) = _mean_operator(ops)


end # module AutoDiffOperatorsStatisticsExt
