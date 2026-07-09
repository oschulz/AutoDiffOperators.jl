# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using Test
using AutoDiffOperators
using LinearAlgebra
import Reactant, Enzyme

# CPU is available everywhere and sufficient to test tracing compatibility:
Reactant.set_default_backend("cpu")

# Top-level functions, Reactant traces them more reliably than closures:
_rct_f_scalar(x) = sum(abs2, x) / 2
_rct_f_vec(x) = exp.(0.3 .* x) .* x

_rct_grad_call(x) = gradient_func(_rct_f_scalar, ADSelector(Enzyme), x)(x)
_rct_valgrad_call(x) = valgrad_func(_rct_f_scalar, ADSelector(Enzyme), x)(x)
_rct_jvp_call(x, z) = jvp_func(_rct_f_vec, x, ADSelector(Enzyme))(z)
_rct_vjp_call(x, z) = with_vjp_func(_rct_f_vec, x, ADSelector(Enzyme))[2](z)

Test.@testset "test_reactant" begin
    x = randn(8)
    z = randn(8)
    J_ref = Diagonal(exp.(0.3 .* x) .* (1 .+ 0.3 .* x))

    xr = Reactant.to_rarray(x)
    zr = Reactant.to_rarray(z)

    grad_compiled = Reactant.@compile _rct_grad_call(xr)
    @test Array(grad_compiled(xr)) ≈ x

    valgrad_compiled = Reactant.@compile _rct_valgrad_call(xr)
    f_x, δx = valgrad_compiled(xr)
    @test Float64(f_x) ≈ sum(abs2, x) / 2
    @test Array(δx) ≈ x

    jvp_compiled = Reactant.@compile _rct_jvp_call(xr, zr)
    @test Array(jvp_compiled(xr, zr)) ≈ J_ref * z

    vjp_compiled = Reactant.@compile _rct_vjp_call(xr, zr)
    @test Array(vjp_compiled(xr, zr)) ≈ J_ref' * z
end
