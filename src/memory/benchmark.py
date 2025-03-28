#!/usr/bin/env python3
"""
Benchmark for the memory management system
"""
import gc
import random
import threading
import time

from src.memory.eviction import LFUEvictionPolicy, LRUEvictionPolicy
from src.memory.map import Memory
from src.memory.placement import BestFitPlacementPolicy
from src.memory.sync import HybridLockManager, SimpleLockPolicy

# Disable GC during benchmarks
gc.disable()


def format_result(operation, iterations, elapsed, per_op_time):
    """Format benchmark result"""
    return f"{operation:<25} | {iterations:>10,d} ops | {elapsed:>8.3f} sec | {per_op_time*1e6:>10.2f} µs/op"


def run_benchmark(name, func, iterations, *args, **kwargs):
    """Run benchmark and report results"""
    print(f"\nRunning benchmark: {name} ({iterations:,d} iterations)")
    gc.collect()  # Force collection before benchmark

    start_time = time.time()
    result = func(iterations, *args, **kwargs)
    elapsed = time.time() - start_time

    per_op = elapsed / iterations
    print(format_result(name, iterations, elapsed, per_op))
    return elapsed, per_op, result


def benchmark_allocation(mem, iterations, size_range=(100, 2000)):
    """Benchmark memory allocation and deallocation"""
    keys = []
    data = []

    # Generate test data
    for i in range(iterations):
        keys.append(f"test-{i}")
        size = random.randint(*size_range)
        data.append(b"x" * size)

    # Measure allocation
    start_time = time.time()
    for i in range(iterations):
        mem[keys[i]] = data[i]
    alloc_time = time.time() - start_time

    # Measure access
    start_time = time.time()
    for i in range(iterations):
        _ = mem[keys[i]]
    access_time = time.time() - start_time

    # Measure deallocation
    start_time = time.time()
    for i in range(iterations):
        del mem[keys[i]]
    dealloc_time = time.time() - start_time

    print(f"  Allocation: {alloc_time:.3f}s ({alloc_time/iterations*1e6:.2f} µs/op)")
    print(f"  Access:     {access_time:.3f}s ({access_time/iterations*1e6:.2f} µs/op)")
    print(
        f"  Deletion:   {dealloc_time:.3f}s ({dealloc_time/iterations*1e6:.2f} µs/op)"
    )

    return alloc_time, access_time, dealloc_time


def benchmark_concurrent_access(mem, iterations, num_threads=8):
    """Benchmark concurrent access patterns"""
    # Prep data
    key_space = 1000
    keys = [f"concurrent-{i}" for i in range(key_space)]
    for key in keys:
        mem[key] = b"x" * 1000

    # Thread function
    def worker(thread_id, results):
        start_time = time.time()
        ops_per_thread = iterations // num_threads

        for i in range(ops_per_thread):
            # Mix of reads and writes
            key = keys[random.randint(0, key_space - 1)]
            if i % 10 == 0:  # 10% writes
                mem[key] = b"y" * 1000
            else:  # 90% reads
                _ = mem[key]

        results[thread_id] = time.time() - start_time

    # Run concurrent threads
    threads = []
    results = [0] * num_threads

    start_time = time.time()
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i, results))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - start_time
    ops_per_second = iterations / elapsed

    print(f"  Threads:    {num_threads}")
    print(f"  Throughput: {ops_per_second:,.0f} ops/sec")
    print(f"  Latency:    {elapsed/iterations*1e6:.2f} µs/op")

    # Cleanup
    for key in keys:
        del mem[key]

    return elapsed, ops_per_second


def benchmark_fragmentation(mem, iterations, max_size=10 * 1024 * 1024):
    """Benchmark memory fragmentation behavior"""
    # Fill memory with varying size blocks
    keys = []
    sizes = []

    # Phase 1: Fill with random sizes
    total_alloc = 0
    while total_alloc < max_size * 0.9:
        size = random.randint(1000, 100000)
        if total_alloc + size > max_size:
            break

        key = f"frag-{len(keys)}"
        mem[key] = b"x" * size
        keys.append(key)
        sizes.append(size)
        total_alloc += size

    initial_count = len(keys)
    print(f"  Initial blocks: {initial_count}")

    # Phase 2: Delete random blocks to create fragmentation
    delete_indices = random.sample(range(len(keys)), len(keys) // 2)
    for idx in sorted(delete_indices, reverse=True):
        del mem[keys[idx]]
        del keys[idx]
        del sizes[idx]

    # Phase 3: Try to allocate large blocks
    success_count = 0
    start_time = time.time()
    for i in range(iterations):
        size = random.randint(5000, 50000)
        try:
            key = f"new-{i}"
            mem[key] = b"y" * size
            keys.append(key)
            success_count += 1
        except MemoryError:
            # Memory fragmentation prevented allocation
            pass

    elapsed = time.time() - start_time

    # Calculate fragmentation metrics
    stats = mem.get_stats()
    frag_ratio = stats["memory"]["fragmentation_ratio"]

    print(f"  Success rate: {success_count/iterations*100:.1f}%")
    print(f"  Fragmentation: {frag_ratio*100:.1f}%")

    # Cleanup
    for key in keys:
        try:
            del mem[key]
        except KeyError:
            pass

    return elapsed, success_count, frag_ratio


def main():
    """Run all benchmarks"""
    print("=" * 70)
    print("MEMORY SYSTEM BENCHMARKS")
    print("=" * 70)

    # Create memory with different configurations
    memory_size = 100 * 1024 * 1024  # 100MB

    configs = [
        (
            "Default",
            Memory(
                memory_size,
                lock_policy=HybridLockManager(),
                eviction_policy=LRUEvictionPolicy(),
                placement_policy=BestFitPlacementPolicy(),
            ),
        ),
        (
            "Simple Locking",
            Memory(
                memory_size,
                lock_policy=SimpleLockPolicy(),
                eviction_policy=LRUEvictionPolicy(),
                placement_policy=BestFitPlacementPolicy(),
            ),
        ),
        (
            "LFU Eviction",
            Memory(
                memory_size,
                lock_policy=HybridLockManager(),
                eviction_policy=LFUEvictionPolicy(),
                placement_policy=BestFitPlacementPolicy(),
            ),
        ),
    ]

    # Number of operations for each benchmark
    allocation_iterations = 50000
    access_iterations = 100000
    concurrent_iterations = 500000
    frag_iterations = 1000

    for name, mem in configs:
        print("\n" + "=" * 70)
        print(f"Configuration: {name}")
        print("-" * 70)

        # Run each benchmark
        print("\n• Allocation Benchmark")
        alloc_result = benchmark_allocation(mem, allocation_iterations)

        print("\n• Concurrent Access Benchmark")
        concurrent_result = benchmark_concurrent_access(mem, concurrent_iterations)

        print("\n• Fragmentation Benchmark")
        frag_result = benchmark_fragmentation(mem, frag_iterations)

        # Clean up
        mem.shutdown()

    # Re-enable GC
    gc.enable()
    print("\nBenchmarks completed!")


if __name__ == "__main__":
    main()
