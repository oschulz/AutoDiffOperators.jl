# AutoDiffOperators.jl

This package provides multiplicative operators that act via automatic differentiation (AD), as well as additional AD-related functionality.

For Jacobians, this package provides the function [`with_jacobian`](@ref). It can return implicit Jacobian operators (e.g. a [`LinearMap`/`FunctionMap`](https://github.com/JuliaLinearAlgebra/LinearMaps.jl) or similar) or explicit Jacobian operators (i.e. a `DenseMatrix`). Lower-level functions [`with_jvp`](@ref), [`jvp_func`](@ref), [`with_vjp_func`](@ref) are provided as well.

In respect to gradients, different Julia algorithm packages require function and gradient calculation to be passed in a different fashion. AutoDiffOperators provides

* [`valgrad_func(f, ad::ADSelector)`](@ref): returns `f_∇f` with `y, δx = f_∇f(x)`.
* [`gradient_func(f, ad::ADSelector)`](@ref): returns `∇f` with `δx = ∇f(x)`.
* [`gradient!_func(f, ad::ADSelector)`](@ref): returns `∇f!` with `δx === ∇f!(δx, x)`.

to cover several popular options. AutoDiffOperators also provides the direct gradient functions

* [`with_gradient`](@ref), [`with_gradient!`](@ref), [`with_gradient!!`](@ref)
* [`only_gradient`](@ref), [`only_gradient!`](@ref), [`only_gradient!!`](@ref)

AD-backends are specified via subtypes of [`ADSelector`](@ref), which includes [`ADTypes.AbstractADType`](https://github.com/SciML/ADTypes.jl). In addition to using subtypes of `AbstractADType` directly, you can use `ADSelector(SomeADModule)` (e.g. `ADSelector(ForwardDiff)`) to select a backend with default options. Separate backends for forward and reverse mode AD can be specified via `ADSelector(fwd_adtype, rev_adtype)`.

Examples for valid `ADSelector` choices:

```julia
ADTypes.AutoForwardDiff()
ADSelector(ForwardDiff)
ADSelector(ADTypes.AutoForwardDiff(), ADTypes.AutoMooncake())
ADSelector(ADSelector(ForwardDiff), ADSelector(Mooncake))
ADSelector(ForwardDiff, Mooncake)
```

AutoDiffOperators uses [`DifferentiationInterface`](https://github.com/gdalle/DifferentiationInterface.jl) internally to interact with the various Julia AD backend packages, adding some specializations and optimizations for type stability and performance. Which backend(s) will perform best for a given use case will depend on the target function and the argument size, as well as the application (`J*z` vs. `z'*J` and gradient calculation) and the compute device (CPU vs. GPU). Please see the documentation of the individual AD backend packages for details.
