# AutoDiffOperators.jl

This package provides multiplicative operators that act via automatic differentiation (AD).

AutoDiffOperators.jl uses AD-backend abstractions and supports a subset of the AD-backends specifiers in both [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl) and [ADTypes.jl](https://github.com/SciML/ADTypes.jl). Support for additional AD-backends is planned.

AD-backends are specified via subtypes of [`ADSelector`](@ref), separate backends for forward and reverse mode AD can be specified if desired. Different AD specifiers that refer to the same AD-backend can be converted into each other via [`convert_ad`](@ref).

The main functions are [`with_gradient`](@ref) and [`with_jacobian`](@ref). Explicit Jacobian matrices can be obtained via [`jacobian_matrix`](@ref). The central lower-level functions are [`with_jvp`](@ref) and [`with_vjp_func`](@ref).

Operators (as returned from [`with_jacobian`](@ref)) can be converted to a [LinearMaps.LinearMap](https://github.com/JuliaLinearAlgebra/LinearMaps.jl).

Different Julia packages require function and gradient calculation to be passed in a different fashion. AutoDiffOperators provides

* [`valgrad_func(f, ad::ADSelector)`](@ref): generates `f_∇f` with `y, δx = f_∇f(x)`.
* [`gradient_func(f, ad::ADSelector)`](@ref): generates `∇f` with `δx = ∇f(x)`.
* [`gradient!_func(f, ad::ADSelector)`](@ref): generates `∇f!` with `δx === ∇f!(δx, x)`.

to cover several popular options.
