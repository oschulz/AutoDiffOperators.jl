# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

"""
    AutoDiffOperators.FwdRevADSelector{Fwd<:ADSelector,Rev<:ADSelector} <: ADSelector

Represent an automatic differentiation backend that forwards
forward-mode and reverse-mode AD to two separate selectors
`fwd::ADSelector` and `rev::ADSelector`.

User code should not instantiate `AutoDiffOperators.FwdRevADSelector`
directly, but use `ADSelector(fwd, rev)` or
`ADSelector(fwd = fwd, rev = rev)` instead.
"""
struct FwdRevADSelector{Fwd<:ADSelector,Rev<:ADSelector} <: WrappedADSelector
    fwd::Fwd
    rev::Rev
end

ADSelector(fwd, rev) = FwdRevADSelector(ADSelector(fwd), ADSelector(rev))
ADSelector(fwd, ::Nothing) = fwd
ADSelector(::Nothing, rev) = rev

ADSelector(;fwd, rev) = ADSelector(fwd, rev)

forward_adtype(ad::FwdRevADSelector) = forward_adtype(ad.fwd)
reverse_adtype(ad::FwdRevADSelector) = reverse_adtype(ad.rev)
