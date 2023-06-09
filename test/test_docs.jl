# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using Test
using AutoDiffOperators
import Documenter

Documenter.DocMeta.setdocmeta!(
    AutoDiffOperators,
    :DocTestSetup,
    :(using AutoDiffOperators);
    recursive=true,
)
Documenter.doctest(AutoDiffOperators)
