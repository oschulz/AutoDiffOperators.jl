# AutoDiffOperators.jl

This package provides multiplicative operators that act via automatic differentiation (AD), as well as additional AD-related functionality.

AD-backends are specified via subtypes of [`ADSelector`](@ref), which includes [`ADTypes.AbstractADType`](https://github.com/SciML/ADTypes.jl). separate backends for forward and reverse mode AD can be specified if desired.

The main functions are [`with_gradient`](@ref) and [`with_jacobian`](@ref). The central lower-level functions are [`with_jvp`](@ref) and [`with_vjp_func`](@ref). Jacobian operators can be implicit (e.g. a [`LinearMap`/`FunctionMap`](https://github.com/JuliaLinearAlgebra/LinearMaps.jl) or similar) or explixit (i.e. a `Matrix`).

Different Julia packages require function and gradient calculation to be passed in a different fashion. AutoDiffOperators provides

* [`valgrad_func(f, ad::ADSelector)`](@ref): generates `f_∇f` with `y, δx = f_∇f(x)`.
* [`gradient_func(f, ad::ADSelector)`](@ref): generates `∇f` with `δx = ∇f(x)`.
* [`gradient!_func(f, ad::ADSelector)`](@ref): generates `∇f!` with `δx === ∇f!(δx, x)`.

to cover several popular options.

AutoDiffOperators natively supports the following automatic differentiation packages as backends:

* [FiniteDifferences](https://github.com/JuliaDiff/FiniteDifferences.jl), selected via `ADTypes.AutoFiniteDifferences()` or `ADSelector(FiniteDifferences)`.

* [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl), selected via `ADTypes.AutoForwardDiff()` or `ADSelector(ForwardDiff)`.

* [Zygote](https://github.com/FluxML/Zygote.jl), selected via `ADTypes.AutoZygote()` or `ADSelector(Zygote)`.

* [Enzyme](https://github.com/EnzymeAD/Enzyme.jl), selected via `ADTypes.AutoEnzyme()` or `ADSelector(Enzyme)`.

Alternatively,
[`DifferentiationInterface``](https://github.com/gdalle/DifferentiationInterface.jl)
can be used to interface with various AD-backends, by using
`DiffIfAD(backend::ADTypes.AbstractADType)` as the AD-selector.

AutoDiffOperators may not support all options and funcionalities of these AD packages. Also, most of them have some limitations on which code constructs in the target function and which function argument types they support. Which backend(s) will perform best will depend on the target function and the argument size, as well as the application (`J*z` vs. `z'*J` and gradient calculation) and the compute device (CPU vs. GPU). Please see the documentation of the individual AD packages linked above for more details on their capabilities.
