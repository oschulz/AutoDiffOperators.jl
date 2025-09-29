# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

module AutoDiffOperatorsEnzymeExt

using Enzyme

import AutoDiffOperators
using AutoDiffOperators: with_floatlike_contents
using AutoDiffOperators: _primal_return_type, _similar_type, _matrix_type, _JVPFunc, _VJPFunc

import ADTypes
using ADTypes: AutoEnzyme


function AutoDiffOperators._adsel_enzyme_forward(ad::AutoEnzyme{M, A}) where {M, A}
    mode = _enzyme_forward_withprimal(ad.mode)
    return AutoEnzyme(function_annotation = Enzyme.Const, mode = mode)
end

_enzyme_forward_withprimal(mode::Enzyme.ForwardMode{true}) = mode
_enzyme_forward_withprimal(::Nothing) = Enzyme.ForwardWithPrimal
_enzyme_forward_withprimal(mode) = throw(ArgumentError("Enzyme mode $mode is not a forward mode with primal"))

_enzyme_forward_mode(mode::Enzyme.ForwardMode) = mode
_enzyme_forward_mode(::Nothing) = Enzyme.Forward
_enzyme_forward_mode(mode) = throw(ArgumentError("Enzyme mode $mode is not a forward mode"))


function AutoDiffOperators._adsel_enzyme_reverse(ad::AutoEnzyme{M, A}) where {M, A}
    mode = _enzyme_reverse_withprimal(ad.mode)
    return AutoEnzyme(function_annotation = Enzyme.Const, mode = mode)
end

_enzyme_reverse_withprimal(mode::Enzyme.ReverseMode{true}) = mode
_enzyme_reverse_withprimal(::Nothing) = Enzyme.ReverseWithPrimal
_enzyme_reverse_withprimal(mode) = throw(ArgumentError("Enzyme mode $mode is not a reverse mode with primal"))

_enzyme_reverse_mode(mode::Enzyme.ReverseMode) = mode
_enzyme_reverse_mode(::Nothing) = Enzyme.Reverse
_enzyme_reverse_mode(mode) = throw(ArgumentError("Enzyme mode $mode is not a reverse mode"))

end # module AutoDiffOperatorsEnzymeExt
