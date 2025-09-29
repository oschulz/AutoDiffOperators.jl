# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).


"""
    abstract type AutoDiffOperators.WrappedADSelector 

Supertype for AD selectors that wrap other AD selectors.
"""
abstract type WrappedADSelector end


"""
    const ADSelector = Union{
        AbstractADType,
        WrappedADSelector
    }

Instances specify an automatic differentiation backend.

Either a subtype of
[`ADTypes.AbstractADType`](https://github.com/SciML/ADTypes.jl),
or an AD-selector wrapper like [`AutoDiffOperators.FwdRevADSelector`](@ref).

In addition to using instances of `AbstractADType` directly (e.g. 
`ADTypes.AutoForwardDiff()`), `ADSelector` (specifically
`AbstractADType`) instances for AD backends can be constructed directly from
the backend modules (using default backend parameters):

```julia
import ForwardDiff

ADTypes.AutoForwardDiff()
ADSelector(ForwardDiff)
convert(ADSelector, ForwardDifsf)
```

all construct an identical `AutoForwardDiff` object.

Separate AD backends for forward- and reverse-mode AD can be specified via
`ADSelector(fwd_adtype, rev_adtype)`, e.g.

```julia
import ForwardDiff, Mooncake

ADSelector(ADTypes.AutoForwardDiff(), ADTypes.AutoMooncake())
ADSelector(ADSelector(ForwardDiff), ADSelector(Mooncake))
ADSelector(ForwardDiff, Mooncake)
```

# Implementation

`ADSelector` instances can also be constructed from module names, though this
should be avoided in end-user code:

```julia
ADSelector(:ForwardDiff)
ADSelector(Val(:ForwardDiff))
convert(ADSelector, :ForwardDiff)
convert(ADSelector, Val(:ForwardDiff))
```

End-users should use module objects instead of module name, so that the
respective AD backend package must be part of their environment/dependencies.
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


# ToDo? ADSelector(::Val{:ChainRules}) = ADTypes.AutoChainRules()
ADSelector(::Val{:Diffractor}) = ADTypes.AutoDiffractor()
ADSelector(::Val{:Enzyme}) = ADTypes.AutoEnzyme()
ADSelector(::Val{:FastDifferentiation}) = ADTypes.AutoFastDifferentiation()
ADSelector(::Val{:FiniteDiff}) = ADTypes.AutoFiniteDiff()
ADSelector(::Val{:FiniteDifferences}) = _adsel_finitedifferences(Val(true))
ADSelector(::Val{:ForwardDiff}) = ADTypes.AutoForwardDiff()
ADSelector(::Val{:GTPSA}) = ADTypes.AutoGTPSA()
ADSelector(::Val{:Mooncake}) = ADTypes.AutoMooncake()
ADSelector(::Val{:ReverseDiff}) = ADTypes.AutoReverseDiff()
ADSelector(::Val{:Symbolics}) = ADTypes.AutoSymbolics()
ADSelector(::Val{:TaylorDiff}) = ADTypes.AutoTaylorDiff()
ADSelector(::Val{:Tracker}) = ADTypes.AutoTracker()
ADSelector(::Val{:Zygote}) = ADTypes.AutoZygote()

_adsel_finitedifferences(::Val) = throw(ErrorException("Package FiniteDifferences not loaded, try `import FiniteDifferences`"))


"""
    forward_adtype(ad::ADSelector)::ADTypes.AbstractADType

Returns the forward-mode AD backend selector for `ad`.

Returns `ad` itself by default if `ad` supports forward-mode automatic
differentation, or instance of `ADTypes.NoAutoDiff` if it does not.

May be specialized for some AD selector types, see [`FwdRevADSelector`](@ref),
for example.
"""
function forward_adtype end
export forward_adtype

forward_adtype(ad::ADSelector) = _default_forward_ad_selector(ad, ADTypes.mode(ad))

function _default_forward_ad_selector(
    ad::ADSelector,
    ::Union{ADTypes.ForwardMode, ADTypes.ForwardOrReverseMode, ADTypes.SymbolicMode}
)
    return ad
end

function _default_forward_ad_selector(::ADSelector, ::ADTypes.AbstractMode)
    return NoAutoDiff()
end

forward_adtype(ad::NoAutoDiff) = ad

# Enzyme requires special handling, to set Enzyme mode and function_annotation:
forward_adtype(ad::ADTypes.AutoEnzyme) = _adsel_enzyme_forward(ad)
_adsel_enzyme_forward(::Any) = throw(ErrorException("Package Enzyme not loaded, try `import Enzyme`"))

# Use ForwardDiff for forward-mode operations with Zygote, since Zygote loads ForwardDiff anyway:
forward_adtype(::ADTypes.AutoZygote) = ADTypes.AutoForwardDiff()

# Use AutoMooncakeForward as forward mode for Mooncake, AutoMooncake only uses
# Mooncake reverse mode:
forward_adtype(ad::ADTypes.AutoMooncake) = ADTypes.AutoMooncakeForward(ad.config)


"""
    valid_forward_adtype(ad::ADSelector)::ADTypes.AbstractADType

Similar to [`forward_adtype`](@ref), but throws an exception if `ad`
doesn't support forward-mode automatic differentiation instead of returning
a `NoAutoDiff`.
"""
function valid_forward_adtype end
export valid_forward_adtype

valid_forward_adtype(ad::ADSelector) = _valid_forward_adtype_impl(forward_adtype(ad), ad)

_valid_forward_adtype_impl(ad, ::ADSelector) = ad
function _valid_forward_adtype_impl(::NoAutoDiff, orig_ad::ADSelector)
    throw(ArgumentError("No forward-mode automatic differentiation available for $orig_ad"))
end


"""
    reverse_adtype(ad::ADSelector)::ADTypes.AbstractADType

Returns the reverse-mode AD backend selector for `ad`.

Returns `ad` itself by default if `ad` supports reverse-mode automatic
differentation, or instance of `ADTypes.NoAutoDiff` if it does not.

May be specialized for some AD selector types, see [`FwdRevADSelector`](@ref),
for example.
"""
function reverse_adtype end
export reverse_adtype

reverse_adtype(ad::ADSelector) = _default_reverse_ad_selector(ad, ADTypes.mode(ad))

# Forward-mode AD is often used for gradients with low-dimensional problems,
# so allow it for reverse-type operations:
function _default_reverse_ad_selector(
    ad::ADSelector,
    ::Union{ADTypes.ForwardMode, ADTypes.ReverseMode, ADTypes.ForwardOrReverseMode, ADTypes.SymbolicMode}
)
    return ad
end

function _default_reverse_ad_selector(::ADSelector, ::ADTypes.AbstractMode)
    return NoAutoDiff()
end

reverse_adtype(ad::NoAutoDiff) = ad

# Enzyme requires special handling, to set Enzyme mode and function_annotation:
reverse_adtype(ad::ADTypes.AutoEnzyme) = _adsel_enzyme_reverse(ad)
_adsel_enzyme_reverse(::Any) = throw(ErrorException("Package Enzyme not loaded, try `import Enzyme`"))


"""
    valid_reverse_adtype(ad::ADSelector)::ADTypes.AbstractADType

Similar to [`reverse_adtype`](@ref), but throws an exception if `ad`
doesn't support reverse-mode automatic differentiation instead of returning
a `NoAutoDiff`.
"""
function valid_reverse_adtype end
export valid_reverse_adtype

valid_reverse_adtype(ad::ADSelector) = _valid_reverse_adtype_impl(reverse_adtype(ad), ad)

_valid_reverse_adtype_impl(ad, ::ADSelector) = ad
function _valid_reverse_adtype_impl(::NoAutoDiff, orig_ad::ADSelector)
    throw(ArgumentError("No reverse-mode automatic differentiation available for $orig_ad"))
end
