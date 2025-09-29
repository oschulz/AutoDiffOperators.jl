using AutoDiffOperators
using Test

using LinearAlgebra
import DifferentiationInterface
import ForwardDiff, FiniteDifferences, Zygote, Enzyme

include("testutils.jl")

@testset "test DifferentiationInterface" begin
    _enzyme_v0_13 = isdefined(Enzyme, :ForwardWithPrimal)

    # DifferentiationInterface (up to at least v0.6.1) still seems to have trouble
    # with Enzyme v0.13:
    ad_modules = if !_enzyme_v0_13
        [ForwardDiff, FiniteDifferences, Zygote, Enzyme]
    else
        [ForwardDiff, FiniteDifferences, Zygote]
    end

    for ad_module in ad_modules
        @testset "DifferentiationInterface for $ad_module" begin
            ad = DiffIfAD(ADSelector(ad_module))

            @testset "fwd and rev sel for $ad" begin
                @test @inferred(forward_ad_selector(ad)).backend == forward_ad_selector(ad.backend)
                @test @inferred(reverse_ad_selector(ad)).backend == reverse_ad_selector(ad.backend)
            end

            test_adsel_functionality(ad)
        end
    end
end
