# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).


"""
    AutoDiffOperators.with_floatlike_contents(A::AbstractArray)

If the elements of `A` are integer-like, convert them using `float`,
otherwise return `A` unchanged.
"""
function with_floatlike_contents end

with_floatlike_contents(A::AbstractArray{T}) where {T<:Real} = _with_floatlike_contents_impl(A, float(T))

_with_floatlike_contents_impl(A::AbstractArray{T}, ::Type{T}) where {T<:Real} = A
_with_floatlike_contents_impl(A::AbstractArray{T}, ::Type{U}) where {T<:Real,U<:Real} = float.(A)


"""
    AutoDiffOperators.similar_onehot(A::AbstractArray, ::Type{T}, n::Integer, i::Integer)

Return an array similar to `A`, but with `n` elements of type `T`, all set to
zero but the `i`-th element set to one.
"""
function similar_onehot end

function similar_onehot(A::AbstractArray{<:Number}, ::Type{T}, n::Integer, i::Integer) where {T<:Number}
    result = similar(A, T, (n,))
    fill!(result, zero(T))
    result[i] = one(T)
    return result
end


# _copyto!(y, x) = (y === x) ? y : copyto!(y, x)::typeof(y)

_primal_return_type(f::F, ::T) where {F, T} = Core.Compiler.return_type(f, Tuple{T})

function _concrete_return_vector_type(f::F, x::T) where {F, T}
    R = Core.Compiler.return_type(f, Tuple{T})
    return _concrete_return_vector_type_impl(R, f, x)
end
_concrete_return_vector_type_impl(::Type{R}, ::Any, ::Any) where {R<:AbstractVector{<:Real}} = R
_concrete_return_vector_type_impl(::Type{R}, f, x) where {R} = typeof(f(x))


_similar_type(::Type{T}) where {T<:AbstractVector} = Core.Compiler.return_type(similar, Tuple{T})

_similar_type(::Type{T},::Type{U}) where {T<:AbstractVector, U<:AbstractVector} = Core.Compiler.return_type(vcat, Tuple{T,U})

_matrix_type(::Type{T}) where {T<:AbstractVector} = Core.Compiler.return_type(_self_outer_prod, Tuple{T})
_self_outer_prod(x::AbstractVector) = x * x'

_matrix_type(::Type{T},::Type{U}) where {T<:AbstractVector, U<:AbstractVector} = Core.Compiler.return_type(_outer_prod, Tuple{T,U})
_outer_prod(x::AbstractVector, y::AbstractVector) = x * y'

#function _jacobian_matrix_type(::Type{F},::Type{T}) where {F,T<:AbstractVector}
#    return Core.Compiler.return_type(_pseudo_jacobian, Tuple{F,T})
#end
#_pseudo_jacobian(f, x::AbstractVector) = f(x) * x'


_oftype(::T, x::T) where T = x
_oftype(::T, x::U) where {T,U} = convert(T, x)::T

_oftype(::T, x::T) where {T<:AbstractArray} = x
function _oftype(::T, x::U) where {T<:AbstractArray,U<:AbstractArray}
    R = _similar_type(T)
    return convert(R, x)::R
end


_oftype!!(::T, x::T) where T = x
_oftype!!(y::T, x::U) where {T,U} = oftype(y, x)

_oftype!!(::T, x::T) where {T<:AbstractArray} = x
_oftype!!(y::T, x::U) where {T<:AbstractArray,U<:AbstractArray} = _oftype_array_impl!!(_is_immutable_type(T), y, x)
# T is immutable:
_oftype_array_impl!!(::Val{true}, y::T, x::U) where {T<:AbstractArray,U<:AbstractArray} = oftype(y, x)
# T is mutable:
_oftype_array_impl!!(::Val{false}, y::T, x::U) where {T<:AbstractArray,U<:AbstractArray} = copyto!(y, x)::T

_is_immutable_type(::Type{T}) where T = Val(isbitstype(T))



struct _WrappedFunction{Ty,Tx} <: Function
    f_fw::FunctionWrapper{Ty,Tuple{Tx}}

    _WrappedFunction{Ty,Tx}(f_fw::FunctionWrapper{Ty,Tx}) where {Ty,Tx} = new{Ty,Tx}(f_fw)
    _WrappedFunction{Ty,Tx}(f::F) where {Ty,Tx,F} = new{Ty,Tx}(FunctionWrapper{Ty,Tuple{Tx}}(f))
end


(f::_WrappedFunction{Ty,Tx})(x::Tx) where {Ty,Tx} = f.f_fw(x)
(f::_WrappedFunction{Ty,Tx})(x) where {Ty,Tx} = f.f_fw(convert(Tx, x))

# # Doesn't seem to fully decouple copy from original:
# Base.deepcopy(f::_WrappedFunction{Ty,Tx}) where {Ty,Tx} = _WrappedFunction{Ty,Tx}(deepcopy(f.f_fw.obj[]))
