# AutoDiffOperators.jl

This package provides multiplicative operators that act via automatic differentiation (AD).

AutoDiffOperators.jl uses AD-backend abstractions and supports a subset of the AD-backends specifiers in both [AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl) and [ADTypes.jl](https://github.com/SciML/ADTypes.jl). Support for additional AD-backends is planned.

AD-backends are specified via subtypes of [`ADSelector`](@ref), separate backends for forward and reverse mode AD can be specified if desired. Different AD specifiers that refer to the same AD-backend can be converted into each other via [`convert_ad`](@ref).

The main functions are [`with_gradient`](@ref) and [`with_jacobian`](@ref). The central lower-level functions are [`with_jvp`](@ref) and [`with_vjp_func`](@ref). Jacobian operators can be implicit (e.g. a [`LinearMap`/`FunctionMap`](https://github.com/JuliaLinearAlgebra/LinearMaps.jl) or similar) or explixit (i.e. a `Matrix`).

Different Julia packages require function and gradient calculation to be passed in a different fashion. AutoDiffOperators provides

* [`valgrad_func(f, ad::ADSelector)`](@ref): generates `f_∇f` with `y, δx = f_∇f(x)`.
* [`gradient_func(f, ad::ADSelector)`](@ref): generates `∇f` with `δx = ∇f(x)`.
* [`gradient!_func(f, ad::ADSelector)`](@ref): generates `∇f!` with `δx === ∇f!(δx, x)`.

to cover several popular options.

AutoDiffOperators currently supports the following automatic differentiation packages as backends:

* [FiniteDifferences](https://github.com/JuliaDiff/FiniteDifferences.jl), selected via `AutoDiffOperators.ADModule(:FiniteDifferences)` or `AbstractDifferentiation.FiniteDifferencesBackend()`.

* [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl), selected via `AutoDiffOperators.ADModule(:ForwardDiff)`, `AbstractDifferentiation.ForwardDiffBackend()` or `ADTypes.AutoForwardDiff()`.

* [Zygote](https://github.com/FluxML/Zygote.jl), selected via `AutoDiffOperators.ADModule(:Zygote)`, `AbstractDifferentiation.ZygoteBackend()` or `ADTypes.AutoZygote()`.

* [Enzyme](https://github.com/EnzymeAD/Enzyme.jl), selected via `AutoDiffOperators.ADModule(:Enzyme)` or `ADTypes.AutoEnzyme()`.

AutoDiffOperators may not support all options and funcionalities of these AD packages. Also, most of them have some limitations on which code constructs in the target function and which function argument types they support. Which backend(s) will perform best will depend on the target function and the argument size, as well as the application (`J*z` vs. `z'*J` and gradient calculation) and the compute device (CPU vs. GPU). Please see the documentation of the individual AD packages linked above for more details on their capabilities.
