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

forward_ad_selector(ad::FwdRevADSelector) = ad.fwd
reverse_ad_selector(ad::FwdRevADSelector) = ad.rev

with_gradient(f, x::AbstractVector{<:Number}, ad::FwdRevADSelector) = with_gradient(f, x, reverse_ad_selector(ad))

with_jvp(f, x::AbstractVector{<:Number}, z::AbstractVector{<:Number}, ad::FwdRevADSelector) = with_jvp(f, x, z, forward_ad_selector(ad))
with_vjp_func(f, x::AbstractVector{<:Number}, ad::FwdRevADSelector) = with_vjp_func(f, x, reverse_ad_selector(ad))
