# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using Test
using AutoDiffOperators
using MatrixShapedOperators: MulFuncOperator, MatrixShapedOperator
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
_rct_jacop(x) = with_jacobian(_rct_f_vec, x, MulFuncOperator, ADSelector(Enzyme))[2]
_rct_jacop_mul(x, z) = _rct_jacop(x) * z
_rct_jacop_adjmul(x, z) = _rct_jacop(x)' * z
_rct_jacop_matmul(x, Z) = _rct_jacop(x) * Z

# The abstract operator request yields an ADJacobian:
_rct_adjacop(x) = with_jacobian(_rct_f_vec, x, MatrixShapedOperator, ADSelector(Enzyme))[2]
_rct_adjacop_mul(x, z) = _rct_adjacop(x) * z
_rct_adjacop_adjmul(x, z) = _rct_adjacop(x)' * z
_rct_adjacop_matmul(x, Z) = _rct_adjacop(x) * Z

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

    Z = randn(8, 3)
    Zr = Reactant.to_rarray(Z)

    jacop_mul_compiled = Reactant.@compile _rct_jacop_mul(xr, zr)
    @test Array(jacop_mul_compiled(xr, zr)) ≈ J_ref * z

    jacop_adjmul_compiled = Reactant.@compile _rct_jacop_adjmul(xr, zr)
    @test Array(jacop_adjmul_compiled(xr, zr)) ≈ J_ref' * z

    jacop_matmul_compiled = Reactant.@compile _rct_jacop_matmul(xr, Zr)
    @test Array(jacop_matmul_compiled(xr, Zr)) ≈ J_ref * Z

    adjacop_mul_compiled = Reactant.@compile _rct_adjacop_mul(xr, zr)
    @test Array(adjacop_mul_compiled(xr, zr)) ≈ J_ref * z

    adjacop_adjmul_compiled = Reactant.@compile _rct_adjacop_adjmul(xr, zr)
    @test Array(adjacop_adjmul_compiled(xr, zr)) ≈ J_ref' * z

    adjacop_matmul_compiled = Reactant.@compile _rct_adjacop_matmul(xr, Zr)
    @test Array(adjacop_matmul_compiled(xr, Zr)) ≈ J_ref * Z
end
