# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using Test

using ADTypes: ADTypes
import FiniteDifferences


@testset "test adselector" begin
    @test @inferred(ADSelector(Val(:Diffractor))) isa ADTypes.AutoDiffractor
    @test @inferred(ADSelector(Val(:Enzyme))) isa ADTypes.AutoEnzyme
    @test @inferred(ADSelector(Val(:FastDifferentiation))) isa ADTypes.AutoFastDifferentiation
    @test @inferred(ADSelector(Val(:FiniteDiff))) isa ADTypes.AutoFiniteDiff
    @test @inferred(ADSelector(Val(:FiniteDifferences))) isa ADTypes.AutoFiniteDifferences
    @test @inferred(ADSelector(Val(:ForwardDiff))) isa ADTypes.AutoForwardDiff
    @test @inferred(ADSelector(Val(:GTPSA))) isa ADTypes.AutoGTPSA
    @test @inferred(ADSelector(Val(:Mooncake))) isa ADTypes.AutoMooncake
    @test @inferred(ADSelector(Val(:ReverseDiff))) isa ADTypes.AutoReverseDiff
    @test @inferred(ADSelector(Val(:Symbolics))) isa ADTypes.AutoSymbolics
    @test @inferred(ADSelector(Val(:TaylorDiff))) isa ADTypes.AutoTaylorDiff
    @test @inferred(ADSelector(Val(:Tracker))) isa ADTypes.AutoTracker
    @test @inferred(ADSelector(Val(:Zygote))) isa ADTypes.AutoZygote
end
