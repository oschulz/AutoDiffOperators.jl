# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

"""
    DiffIfAD{Fwd<:ADSelector,Rev<:ADSelector} <: ADSelector

Uses [DifferentiationInterfac](https://github.com/gdalle/DifferentiationInterface.jl)
to interface with an AD-backend.

Constructor: `DiffIfAD(backend::ADTypes.AbstractADType)`
"""
struct DiffIfAD{B<:AbstractADType} <: WrappedADSelector
    backend::B
end
export DiffIfAD

forward_ad_selector(ad::DiffIfAD) = DiffIfAD(forward_ad_selector(ad.backend))
reverse_ad_selector(ad::DiffIfAD) = DiffIfAD(reverse_ad_selector(ad.backend))

# Support for struct values not implemented yet:
supports_structargs(ad::DiffIfAD) = false
