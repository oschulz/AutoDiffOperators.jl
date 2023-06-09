# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

module AutoDiffOperatorsForwardDiffExt

@static if isdefined(Base, :get_extension)
    using ForwardDiff
else
    using ..ForwardDiff
end

import AutoDiffOperators
import AbstractDifferentiation
import ADTypes


Base.Module(::AutoDiffOperators.ADModule{:ForwardDiff}) = ForwardDiff

const ForwardDiffAD = Union{
    AbstractDifferentiation.ForwardDiffBackend,
    ADTypes.AutoForwardDiff,
    AutoDiffOperators.ADModule{:ForwardDiff}
}

# ToDo: Convert AD parameters
function AutoDiffOperators.convert_ad(::Type{ADTypes.AbstractADType}, ad::AbstractDifferentiation.ForwardDiffBackend)
    ADTypes.AutoForwardDiff()
end

function AutoDiffOperators.convert_ad(::Type{ADTypes.AbstractADType}, ::AutoDiffOperators.ADModule{:ForwardDiff})
    ADTypes.AutoForwardDiff()
end

# ToDo: Convert AD parameters
function AutoDiffOperators.convert_ad(::Type{AbstractDifferentiation.AbstractBackend}, ad::ADTypes.AutoForwardDiff)
    AbstractDifferentiation.ForwardDiffBackend()
end

function AutoDiffOperators.convert_ad(::Type{AbstractDifferentiation.AbstractBackend}, ::AutoDiffOperators.ADModule{:ForwardDiff})
    AbstractDifferentiation.ForwardDiffBackend()
end

function AutoDiffOperators.convert_ad(::Type{AutoDiffOperators.ADModule}, ad::AbstractDifferentiation.ForwardDiffBackend)
    AutoDiffOperators.ADModule{:ForwardDiff}()
end

function AutoDiffOperators.convert_ad(::Type{AutoDiffOperators.ADModule}, ad::ADTypes.AutoForwardDiff)
    AutoDiffOperators.ADModule{:ForwardDiff}()
end


# ToDo: Use AD parameters
AutoDiffOperators.with_gradient(f, x::AbstractVector{<:Real}, ad::ForwardDiffAD) = f(x), ForwardDiff.gradient(f, x)

# ToDo: Use AD parameters
AutoDiffOperators.jacobian_matrix(f, x::AbstractVector{<:Real}, ad::ForwardDiffAD) = ForwardDiff.jacobian(f, x)


struct _JacVecProdTag{F, T} end

function _dual_along(f::F, x::AbstractVector{T1}, z::AbstractVector{T2}) where {F, T1, T2}
    T =  promote_type(T1, T2)
    T_Dual = _JacVecProdTag{F, T}
    # ToDo: use `StructArrays.StructArray`? Would add StructArrays to deps.
    f(ForwardDiff.Dual{T_Dual}.(x, z))
end

function AutoDiffOperators.with_jvp(f, x, z, ::ForwardDiffAD)
    dual_y = _dual_along(f, x, z)
    ForwardDiff.value.(dual_y), ForwardDiff.partials.(dual_y, 1)
end


function AutoDiffOperators.with_vjp_func(f, x, ad::ForwardDiffAD)
    f(x), AutoDiffOperators._FwdModeVJPFunc(f, x, ad)
end


# ToDo: Use AD parameters
function AutoDiffOperators.jacobian_matrix(f, x, ad::ForwardDiffAD)
    ForwardDiff.jacobian(f, x)
end


# ToDo: Use AD parameters
# ToDo: Use custom multi-threaded code instead of using `ForwardDiff.gradient!`?
function with_gradient(f, x::AbstractVector{<:Real}, ad::ForwardDiffAD)
    y = f(x)
    AutoDiffOperators._grad_sensitivity(y) # Check that y is a real number
    R = promote_type(eltype(x), typeof(y))

    grad_f_x = similar(f, x)
    result = ForwardDiff.DiffResults.MutableDiffResult(zero(R), (grad_f_x,))

    # chunk = ForwardDiff.Chunk(x)
    # config = ForwardDiff.GradientConfig(f, x, chunk)
    # ForwardDiff.gradient!(result, f, x, config)
    ForwardDiff.gradient!(result, f, x)
    @assert DiffResults.gradient(result) === grad_f_x

    # Ensure type stability:
    return convert(R, DiffResults.value(result))::R
end

end # module ForwardDiff
