# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using Test

using AutoDiffOperators: _CacheLikeUse, _MaybeWriteIdxHandle, _borrowable_object, _borrow_maybewrite, _return_borrowed, @_borrow_maybewrite
using AutoDiffOperators: _CacheLikePool

@testset "test_util" begin
    x = 42
    A = rand(5)
    usage = _CacheLikeUse()

    value = x
    @test @inferred(_borrowable_object(usage, value, () -> value)) === value
    @test @inferred(_borrow_maybewrite(_borrowable_object(usage, value, () -> value))) isa Tuple{typeof(value), Nothing}
    obj = _borrowable_object(usage, value, () -> value)
    @test obj isa typeof(value)
    instance, handle = _borrow_maybewrite(obj)
    @test instance === value
    @test handle === nothing
    @test @inferred(_return_borrowed(obj, instance, handle)) isa Nothing

    @test length(_CacheLikePool(A, () -> copy(A), 3).instances) == 3

    value = A
    n_made = Threads.Atomic{Int}(0)
    make_value = () -> (Threads.atomic_add!(n_made, 1); copy(value))
    @test @inferred(_borrowable_object(usage, value, make_value)) isa _CacheLikePool{typeof(value)}
    @test @inferred(_borrow_maybewrite(_borrowable_object(usage, value, make_value))) isa Tuple{typeof(value), _MaybeWriteIdxHandle}
    obj = _borrowable_object(usage, value, make_value)
    @test obj isa _CacheLikePool{typeof(value)}

    # Serial use borrows the original object itself, without making new ones:
    for _ in 1:3
        instance, handle = _borrow_maybewrite(obj)
        @test instance === value
        @test @inferred(_return_borrowed(obj, instance, handle)) isa Nothing
    end
    @test n_made[] == 0

    # Objects borrowed at the same time are distinct, made on demand and reused:
    let pool = _CacheLikePool(value, make_value, 3)
        for _ in 1:2
            borrowed = [_borrow_maybewrite(pool) for _ in 1:3]
            @test borrowed[1][1] === value
            @test allunique(objectid.(first.(borrowed)))
            @test all(==(value), first.(borrowed))
            foreach(b -> _return_borrowed(pool, b...), borrowed)
        end
        @test n_made[] == 2
    end

    # A failure to make a new object leaves the pool usable:
    let pool = _CacheLikePool(value, () -> error("can't make value"), 2)
        instance, handle = _borrow_maybewrite(pool)
        @test_throws ErrorException _borrow_maybewrite(pool)
        @test_throws ErrorException _borrow_maybewrite(pool)
        _return_borrowed(pool, instance, handle)
        @test _borrow_maybewrite(pool)[1] === value
    end

    # New objects of another type than the original are not used, borrowers
    # share the remaining ones:
    let n_rejected = Threads.Atomic{Int}(0)
        pool = _CacheLikePool(value, () -> (Threads.atomic_add!(n_rejected, 1); Float32.(value)), 3)
        instance, handle = _borrow_maybewrite(pool)
        waiting = Threads.@spawn _borrow_maybewrite(pool)
        @test timedwait(() -> n_rejected[] == 2, 30) === :ok
        @test !istaskdone(waiting)
        _return_borrowed(pool, instance, handle)
        @test timedwait(() -> istaskdone(waiting), 30) === :ok
        @test fetch(waiting)[1] === value
    end

    obj = _borrowable_object(usage, value, make_value)
    n = length(obj.instances)
    instances = [_borrow_maybewrite(obj) for i in 1:n]
    cmp_results = fill(false, 1000)
    tasks = [
        Threads.@spawn begin
            @_borrow_maybewrite obj begin
                cmp_results[i] = (obj == value)
                sleep(0.001 * rand())
            end
        end
        for i in eachindex(cmp_results)
    ]
    for (instance, handle) in instances
        @test instance == value
        _return_borrowed(obj, instance, handle)
    end
    for i in 1:30
        all(istaskdone, tasks) && break
        sleep(1)
    end
    @test all(istaskdone, tasks)
    @test !any(istaskfailed, tasks)
    @test all(cmp_results)

    if any(istaskfailed, tasks)
        println("Failed task exceptions:")
        for t in tasks
            if istaskfailed(t)
                println(t.result)
            end
        end
    end
end
