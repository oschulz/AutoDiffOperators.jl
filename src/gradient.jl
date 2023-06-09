# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).


"""
    with_gradient(f, x, ad::ADSelector)

Returns a tuple (f(x), ∇f(x)) with the gradient ∇f(x) of `f` at `x`.
"""
function with_gradient end
export with_gradient

with_gradient(f, x, ad::FwdRevADSelector) = with_gradient(f, x, reverse_ad_selector(ad))


_grad_sensitivity(y::Number) = one(y)
_grad_sensitivity(@nospecialize(::Complex)) = error("f(x) is a complex number, but with_gradient expects it to a real number")
_grad_sensitivity(@nospecialize(::T)) where T = error("f(x) is of type $(nameof(T)), but with_gradient expects it to a real number")

function with_gradient(f, x, ad::ADSelector)
    y, vjp = with_vjp_func(f, x, ad)
    y isa Real || throw(ArgumentError("with_gradient expects f(x) to return a real number"))
    grad_f_x = vjp(_grad_sensitivity(y))
    return y, grad_f_x
end


# ToDo: add `with_gradient!(f, δx, x, ad::ADSelector)`?
