# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).


@deprecate ADModule(m::Symbol) ADSelector(m)
@deprecate ADModule(m::Module) ADSelector(m)


"""
    abstract type AutoDiffOperators.WrappedADSelector 

Supertype for AD selectors that wrap other AD selectors.
"""
abstract type WrappedADSelector end


"""
    const ADSelector = Union{
        ADTypes.AbstractADType,
        WrappedADSelector
    }

Instances speficy an automatic differentiation backend.

Either a subtype of
[`ADTypes.AbstractADType`]](https://github.com/SciML/ADTypes.jl),
or an AD-selector wrapper like [`AutoDiffOperators.FwdRevADSelector`](@ref).

AutoDiffOperators currently provides it's own implementations for following
AD-selectors: `AutoForwardDiff()`, `AutoFiniteDifferences()`, `AutoZygote()`
and `AutoEnzyme()`.

`ADSelector` (specifically `ADTypes.AbstractADType` ) instances for these
backends can be constructed directly from modules and modules names
(using `AutoDiffOperators` default backend parameters):

```julia
import ForwardDiff
ADSelector(ForwardDiff)
ADSelector(:ForwardDiff)
ADSelector(Val(:ForwardDiff))
convert(ADSelector, ForwardDiff)
convert(ADSelector, :ForwardDiff)
convert(ADSelector, Val(:ForwardDiff))
```

Some operations that specifically require forward-mode or reverse-mode
AD will only accept a subset of these backends though.

Alternatively,
[`DifferentiationInterface``](https://github.com/gdalle/DifferentiationInterface.jl)
can be used to interface with various AD-backends, by using
`DiffIfAD(backend::ADTypes.AbstractADType)` as the AD-selector.

# Implementation

The following functions must be specialized for subtypes of `ADSelector`:
[`with_jvp`](@ref), [`with_vjp_func`](@ref).

A default implementation is provided for [`with_gradient`](@ref), but
specialized implementations may often be more performant.

Selector types that delegate forward and/or reverse-mode AD to other selector
types resp AD-backends should also specialize [`forward_ad_selector`](@ref)
and [`reverse_ad_selector`](@ref).
"""
const ADSelector = Union{
    ADTypes.AbstractADType,
    WrappedADSelector
}
export ADSelector

@inline ADSelector(ad::ADSelector) = ad

@inline ADSelector(m::Symbol) = ADSelector(Val(m))
@inline ADSelector(m::Module) = ADSelector(Val(nameof(m)))

@inline Base.convert(::Type{ADSelector}, ::Val{m}) where m = ADSelector(m)
@inline Base.convert(::Type{ADSelector}, m::Symbol) = ADSelector(Val(m))
@inline Base.convert(::Type{ADSelector}, m::Module) = ADSelector(Val(nameof(m)))


"""
    forward_ad_selector(ad::ADSelector)::ADSelector

Returns the forward-mode AD backen selector for `ad`.

Returns `ad` itself by default. Also see [`FwdRevADSelector`](@ref).
"""
function forward_ad_selector end
export forward_ad_selector

forward_ad_selector(ad::ADSelector) = ad


"""
    reverse_ad_selector(ad::ADSelector)::ADSelector

Returns the reverse-mode AD backen selector for `ad`.

Returns `ad` itself by default. Also see [`FwdRevADSelector`](@ref).
"""
function reverse_ad_selector end
export reverse_ad_selector

reverse_ad_selector(ad::ADSelector) = ad
