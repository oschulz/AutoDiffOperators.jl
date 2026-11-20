# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).


"""
    AutoDiffOperators.with_floatlike_contents(A::AbstractArray)

If the elements of `A` are integer-like, convert them using `float`,
otherwise return `A` unchanged.

The elements must be real-valued numbers, though not necessarily `Real`
(e.g. tracing- and symbolic-number types), so realness is checked via
`real(T) === T` instead of dispatch. Complex and other non-real numbers
are rejected, real-AD conventions would silently give incorrect results
for them.
"""
function with_floatlike_contents end

with_floatlike_contents(A::AbstractArray{T}) where {T<:Real} = _with_floatlike_contents_impl(A, float(T))

function with_floatlike_contents(A::AbstractArray{T}) where {T<:Number}
    real(T) === T || throw(ArgumentError(
        "AutoDiffOperators only supports real-valued numbers, got array eltype $T"
    ))
    return _with_floatlike_contents_impl(A, float(T))
end

_with_floatlike_contents_impl(A::AbstractArray{T}, ::Type{T}) where {T<:Number} = A
_with_floatlike_contents_impl(A::AbstractArray{T}, ::Type{U}) where {T<:Number,U<:Number} = float.(A)


_primal_return_type(f::F, ::T) where {F, T} = Core.Compiler.return_type(f, Tuple{T})

function _concrete_return_vector_type(f::F, x::T) where {F, T}
    R = Core.Compiler.return_type(f, Tuple{T})
    return _concrete_return_vector_type_impl(R, f, x)
end
_concrete_return_vector_type_impl(::Type{R}, ::Any, ::Any) where {R<:AbstractVector{<:Real}} = R
_concrete_return_vector_type_impl(::Type{R}, f, x) where {R} = typeof(f(x))

function _concrete_return_realtype(f::F, x::T) where {F, T}
    R = Core.Compiler.return_type(f, Tuple{T})
    return _concrete_return_realtype_impl(R, f, x)
end
_concrete_return_realtype_impl(::Type{R}, ::Any, ::Any) where {R<:Real} = R
_concrete_return_realtype_impl(::Type{R}, f, x) where {R} = typeof(f(x))


_similar_type(::Type{T}) where {T<:AbstractVector} = Core.Compiler.return_type(similar, Tuple{T})

_similar_type(::Type{T},::Type{U}) where {T<:AbstractVector, U<:AbstractVector} = Core.Compiler.return_type(vcat, Tuple{T,U})

_matrix_type(::Type{T}) where {T<:AbstractVector} = Core.Compiler.return_type(_self_outer_prod, Tuple{T})
_self_outer_prod(x::AbstractVector) = x * x'

_matrix_type(::Type{T},::Type{U}) where {T<:AbstractVector, U<:AbstractVector} = Core.Compiler.return_type(_outer_prod, Tuple{T,U})
_outer_prod(x::AbstractVector, y::AbstractVector) = x * y'


_oftype(::T, x::T) where T = x
_oftype(::T, x::U) where {T,U} = convert(T, x)::T

_oftype(::T, x::T) where {T<:AbstractArray} = x
function _oftype(::T, x::U) where {T<:AbstractArray,U<:AbstractArray}
    R = _similar_type(T)
    return convert(R, x)::R
end


_is_immutable_type(::Type{T}) where T = Val(isbitstype(T))


# Classifies arrays for which backend-specific preparation and function
# wrappers must not be used: standard CPU arrays yield `nothing` (prepared AD),
# while traced or device-resident arrays (e.g. Reactant's, via a package
# extension) yield a `Val` marker selecting the unprepared, wrapper-free AD path:
_traced_array_kind(::AbstractArray) = nothing



struct _WrappedFunction{Ty,Tx} <: Function
    f_fw::FunctionWrapper{Ty,Tuple{Tx}}

    _WrappedFunction{Ty,Tx}(f_fw::FunctionWrapper{Ty,Tuple{Tx}}) where {Ty,Tx} = new{Ty,Tx}(f_fw)
    _WrappedFunction{Ty,Tx}(f::F) where {Ty,Tx,F} = new{Ty,Tx}(FunctionWrapper{Ty,Tuple{Tx}}(f))
end


(f::_WrappedFunction{Ty,Tx})(x::Tx) where {Ty,Tx} = f.f_fw(x)
(f::_WrappedFunction{Ty,Tx})(x) where {Ty,Tx} = f.f_fw(convert(Tx, x))

# # Doesn't seem to fully decouple copy from original:
# Base.deepcopy(f::_WrappedFunction{Ty,Tx}) where {Ty,Tx} = _WrappedFunction{Ty,Tx}(deepcopy(f.f_fw.obj[]))



@kwdef struct _CacheLikeUse
    nreaders::Int = nthreads()
    nwriters::Int = nthreads()
end


# `make_value()` must return a fresh object equivalent to `value`, that can be
# used concurrently with it. It may be called at any later time:
@inline _borrowable_object(usage::_CacheLikeUse, value::T, make_value::F) where {T,F} = _borrowable_object_impl(_is_immutable_type(T), usage, value, make_value)
@inline _borrowable_object_impl(::Val{true}, ::_CacheLikeUse, value::T, ::F) where {T,F} = value
@inline _borrowable_object_impl(::Val{false}, usage::_CacheLikeUse, value::T, make_value::F) where {T,F} = _CacheLikePool(value, make_value, usage.nwriters)

_borrow_maybewrite(obj) = obj, nothing
_return_borrowed(obj, ::Any, ::Nothing) = nothing


macro _borrow_maybewrite(objvar, expr)
    @assert objvar isa Symbol
    quote
        instance, handle = _borrow_maybewrite($(esc(objvar)))
        try
            let $(esc(objvar)) = instance
                $(esc(expr))
            end
        finally
            _return_borrowed($(esc(objvar)), instance, handle)
        end
    end
end


const _ConditionType = typeof(Threads.Condition())

# The pool owns `value` as its first instance. The lowest free instance is
# borrowed first, so serial use only ever sees `value` itself, further
# instances are made on demand when several are borrowed at the same time.
struct _CacheLikePool{T,F}
    make_value::F
    isvalid::BitVector
    isavail::BitVector
    instances::Vector{T}
    cond::_ConditionType
end

function _CacheLikePool(value::T, make_value::F, n_max::Int) where {T,F}
    n_max >= 1 || throw(ArgumentError("_CacheLikePool requires at least one instance"))
    instances = Vector{T}(undef, n_max)
    instances[1] = value
    isvalid = BitVector(undef, n_max)
    fill!(isvalid, false)
    isvalid[1] = true
    isavail = BitVector(undef, n_max)
    fill!(isavail, true)
    cond = Threads.Condition()
    return _CacheLikePool{T,F}(make_value, isvalid, isavail, instances, cond)
end

struct _MaybeWriteIdxHandle
    instance_idx::Int
end

function _borrow_maybewrite(obj::_CacheLikePool{T}) where {T}
    (;isvalid, isavail, instances, cond) = obj

    instance_idx::Int = -1
    instance_valid::Bool = false
    @lock cond begin
        success::Bool = false
        while !success
            found = findfirst(isavail)
            if found isa Nothing
                wait(cond)
            else
                instance_idx = found
                success = true
            end
        end

        isavail[instance_idx] = false
        instance_valid = isvalid[instance_idx]
    end

    instance::T = if instance_valid
        instances[instance_idx]
    else
        new_instance = try
            obj.make_value()
        catch
            @lock cond begin
                isavail[instance_idx] = true
                notify(cond; all = false)
            end
            rethrow()
        end
        # A new object of another type than `value` can't be used, its
        # slot stays disabled, so borrow another instance instead:
        new_instance isa T || return _borrow_maybewrite(obj)
        instances[instance_idx] = new_instance
        @lock cond isvalid[instance_idx] = true
        new_instance
    end

    return instance, _MaybeWriteIdxHandle(instance_idx)
end

function _return_borrowed(obj::_CacheLikePool{T}, instance::T, handle::_MaybeWriteIdxHandle) where {T}
    (;isvalid, isavail, instances, cond) = obj
    instance_idx = handle.instance_idx
    @lock cond begin
        if !isvalid[instance_idx]
            throw(ErrorException("Object handle returned to _CacheLikePool does reference an invalid object, invalid handle returned?"))
        elseif isavail[instance_idx]
            throw(ErrorException("Object handle $handle returned to _CacheLikePool is already marked as available ($isavail), wrong handle returned?"))
        elseif instances[instance_idx] !== instance
            throw(ErrorException("Object instance (type $(nameof(T))) returned to _CacheLikePool does not match borrowed instance for given handle, wrong object/handle returned?"))
        end
        isavail[instance_idx] = true
        notify(cond; all = false)
    end
    return nothing
end



struct _DIPrep{T}
    prep::T
end


# Makes equivalent DI preparations on demand, via `prepare(f, ad, args...)`.
# A closure would capture a type-valued `f` as a `DataType` on Julia v1.10,
# this keeps it as `Type{FT}`:
struct _DIPrepFactory{P,F,AD<:AbstractADType,Args<:Tuple}
    prepare::P
    f::F
    ad::AD
    args::Args
end

function _DIPrepFactory(prepare::P, ::Type{FT}, ad::AD, args::Args) where {P,FT,AD<:AbstractADType,Args<:Tuple}
    return _DIPrepFactory{P,Type{FT},AD,Args}(prepare, FT, ad, args)
end

(make_prep::_DIPrepFactory)() = make_prep.prepare(make_prep.f, make_prep.ad, make_prep.args...)
