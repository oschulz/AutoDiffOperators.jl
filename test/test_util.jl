# This file is a part of AutoDiffOperators.jl, licensed under the MIT License (MIT).

using AutoDiffOperators
using Test

using AutoDiffOperators: _CacheLikeUse, _MaybeWriteIdxHandle, _borrowable_object, _borrow_maybewrite, _return_borrowed
using AutoDiffOperators: _CacheLikePool

@testset "test_util" begin
    x = 42
    A = rand(5)
    usage = _CacheLikeUse()

    value = x
    @test @inferred(_borrowable_object(usage, value)) === value
    @test @inferred(_borrow_maybewrite(_borrowable_object(usage, value))) isa Tuple{typeof(value), Nothing}
    obj = _borrowable_object(usage, value)
    @test obj isa typeof(value)
    instance, handle = _borrow_maybewrite(obj)
    @test instance === value
    @test handle === nothing
    @test @inferred(_return_borrowed(obj, instance, handle)) isa Nothing

    value = A
    @test @inferred(_borrowable_object(usage, value)) isa _CacheLikePool{typeof(value)}
    @test @inferred(_borrow_maybewrite(_borrowable_object(usage, value))) isa Tuple{typeof(value), _MaybeWriteIdxHandle}
    obj = _borrowable_object(usage, value)
    @test obj isa _CacheLikePool{typeof(value)}
    instance, handle = _borrow_maybewrite(obj)
    @test instance == value
    @test @inferred(_return_borrowed(obj, instance, handle)) isa Nothing

    obj = _borrowable_object(usage, value)
    n = length(obj.instances)
    instances = [_borrow_maybewrite(obj) for i in 1:n]
    cmp_results = fill(false, 1000)
    tasks = [
        Threads.@spawn begin
            local myinstance, myhandle = _borrow_maybewrite(obj)
            try
                cmp_results[i] = myinstance == value
                sleep(0.001 * rand())
            finally
                _return_borrowed(obj, myinstance, myhandle)
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
