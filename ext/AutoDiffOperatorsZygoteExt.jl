# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

module AutoDiffOperatorsZygoteExt

@static if isdefined(Base, :get_extension)
    using Zygote
else
    using ..Zygote
end

import AutoDiffOperators
import AbstractDifferentiation
import ADTypes


Base.Module(::AutoDiffOperators.ADModule{:Zygote}) = Zygote

const AbstractDifferentiation_ZygoteBackend = AbstractDifferentiation.ReverseRuleConfigBackend{<:Zygote.ZygoteRuleConfig}

const ZygoteAD = Union{
    ADTypes.AutoZygote,
    AbstractDifferentiation_ZygoteBackend,
    AutoDiffOperators.ADModule{:Zygote}
}


AutoDiffOperators.forward_ad_selector(::ADTypes.AutoZygote) = AbstractDifferentiation.ForwardDiffBackend()

AutoDiffOperators.forward_ad_selector(::AbstractDifferentiation_ZygoteBackend) =
    AbstractDifferentiation.ForwardDiffBackend()

AutoDiffOperators.forward_ad_selector(::ZygoteAD) = AutoDiffOperators.ADModule{:ForwardDiff}()


function AutoDiffOperators.convert_ad(::Type{ADTypes.AbstractADType}, ad::AbstractDifferentiation_ZygoteBackend)
    ADTypes.AutoZygote()
end

function AutoDiffOperators.convert_ad(::Type{ADTypes.AbstractADType}, ::AutoDiffOperators.ADModule{:Zygote})
    ADTypes.AutoZygote()
end

function AutoDiffOperators.convert_ad(::Type{AbstractDifferentiation.AbstractBackend}, ad::ADTypes.AutoZygote)
    AbstractDifferentiation.ZygoteBackend()
end

function AutoDiffOperators.convert_ad(::Type{AbstractDifferentiation.AbstractBackend}, ::AutoDiffOperators.ADModule{:Zygote})
    AbstractDifferentiation.ZygoteBackend()
end

function AutoDiffOperators.convert_ad(::Type{AutoDiffOperators.ADModule}, ad::AbstractDifferentiation_ZygoteBackend)
    AutoDiffOperators.ADModule{:Zygote}()
end

function AutoDiffOperators.convert_ad(::Type{AutoDiffOperators.ADModule}, ad::ADTypes.AutoZygote)
    AutoDiffOperators.ADModule{:Zygote}()
end


function AutoDiffOperators.with_jvp(f, x, z, ad::ZygoteAD)
    fwd_ad = forward_ad_selector(ad)
    @assert !(fwd_ad isa ZygoteAD)
    AutoDiffOperators.with_jvp(f, x, z, fwd_ad)
end


function AutoDiffOperators.with_vjp_func(f::Function, x, ::ZygoteAD)
    y, pullback = Zygote.pullback(f, x)
    return y, only ∘ pullback
end


function AutoDiffOperators.jacobian_matrix(f, x, ::ZygoteAD)
    only(Zygote.jacobian(f, x))
end


end # module Zygote
