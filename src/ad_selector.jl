# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).


"""
    struct ADModule{m}

Speficies and automatic differentiation backend via it's module name.

We recommend to use the AD-backend types defined in
[AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl)
and [ADTypes.jl}(https://github.com/SciML/ADTypes.jl) instead of `ADModule`.
The function [`convert_ad`](@ref) provides a conversion mechanism between all
of these different AD-backend-speficiers.

Examples:

```
ADModule(:ForwardDiff)
ADModule(:Zygote)
```

Can be converted to a `Val` for compatibily with approaches like
`LogDensityProblemsAD.ADgradient`
(see [LogDensityProblemsAD](https://github.com/tpapp/LogDensityProblemsAD.jl)):

```julia
Val(ADModule(:ForwardDiff)) == Val(:ForwardDiff)
```

Constructing an instance of `ADModule` will fail if the package hosting the
corresponding module is not loaded (either directly, or indirectly via
dependencies of other loaded packages).
    
`Module(ad::ADModule)` returns the `Module` object that corresponds
to `ad`.
"""
struct ADModule{m}
    function ADModule{m}() where m
        ad = new{m}()
        Module(ad) # will fail if m is not loaded
        return ad
    end
end
export ADModule

function Base.Module(@nospecialize ad::ADModule{m}) where m
    throw(ErrorException("Cannot get `Module($ad)`, $m is either not loaded (directly or indirectly) or not supported by AutoDiffOperators."))
end

ADModule(m::Symbol) = ADModule{m}()
ADModule(m::Module) = ADModule{nameof(m)}()

Base.Val(::Type{ADModule{m}}) where m = Val(m)
Base.convert(::Type{Val}, ad::ADModule) = Val(ad)


"""
    const ADSelector = Union{
        AbstractDifferentiation.AbstractBackend,
        ADTypes.AbstractADType,
        ADModule
    }

Instances speficy an automatic differentiation backend.

Unifies the AD-backend selector types defined in
[AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl)
and [ADTypes.jl}(https://github.com/SciML/ADTypes.jl), as well as
[`AutoDiffOperators.ADModule`](@ref).

AutoDiffOperators currently supports the following AD-backends, and its
functions will, in general, accept any subtype of `ADSelector` as
AD-selectors that match them:

* ForwardDiff
* Zygote

Some operations that specifically require forward-mode or reverse-mode
AD will only accept a subset of these backends though.

The following functions must be specialized for subtypes of `ADSelector`:
[`convert_ad`](@ref), [`with_jvp`](@ref) and [`with_vjp_func`](@ref).

Default implementations are provided for [`jacobian_matrix`](@ref) and
[`with_gradient`](@ref), but specialized implementations may often
be more performant.

Selector types that forward forward and reverse-mode ad to
other selector types should specialize [`forward_ad_selector`](@ref)
and [`reverse_ad_selector`](@ref).
"""
const ADSelector = Union{
    AbstractDifferentiation.AbstractBackend,
    ADTypes.AbstractADType,
    ADModule
}
export ADSelector


"""
    convert_ad(::Type{AbstractDifferentiation.AbstractBackend}, ad::ADSelector)
    convert_ad(::Type{ADTypes.AbstractADType}, ad::ADSelector)
    convert_ad(::Type{ADModule}, ad::ADSelector)

Converts AD-backend selector types between
[AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl),
[ADTypes.jl](https://github.com/SciML/ADTypes.jl) and
[`AutoDiffOperators.ADModule`](@ref).
"""
function convert_ad end
export convert_ad

convert_ad(::Type{ADTypes.AbstractADType}, ad::ADTypes.AbstractADType) = ad
convert_ad(::Type{AbstractDifferentiation.AbstractBackend}, ad::AbstractDifferentiation.AbstractBackend) = ad
convert_ad(::Type{ADModule}, ad::ADModule) = ad


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



"""
    AutoDiffOperators.FwdRevADSelector{Fwd<:ADSelector,Rev<:ADSelector} <: ADSelector

Represent an automatic differentiation backend that forwards
forward-mode and reverse-mode AD to two separate selectors
`fwd::ADSelector` and `rev::ADSelector`.

User code should not instantiate `AutoDiffOperators.FwdRevADSelector`
directly, but use `ADSelector(fwd, rev)` or
`ADSelector(fwd = fwd, rev = rev)` instead.
"""
struct FwdRevADSelector{Fwd<:ADSelector,Rev<:ADSelector}
    fwd::Fwd
    rev::Rev
end
export FwdRevADSelector

ADSelector(fwd::ADSelector, rev::ADSelector) = FwdRevADSelector(fwd, rev)
ADSelector(fwd::ADSelector, ::Nothing) = fwd
ADSelector(::Nothing, rev::ADSelector) = rev

ADSelector(;fwd::ADSelector, rev::ADSelector) = FwdRevADSelector(fwd, rev)

forward_ad_selector(ad::FwdRevADSelector) = ad.fwd
reverse_ad_selector(ad::FwdRevADSelector) = ad.fwd
