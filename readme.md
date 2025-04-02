# Varbos

**Varbos** is a high-performance memory management system designed for applications that require efficient, thread-safe memory operations with configurable allocation and eviction strategies.

## Features

- **Direct Memory Access** with buffer protocol and NumPy integration
- **Policy-based Architecture** for customizable behavior:
  - Configurable memory allocation strategies
  - Pluggable cache eviction policies
  - Adaptive synchronization mechanisms
- **Thread-safe Operations** with optimized locking strategies
- **Background Maintenance** for memory defragmentation and cleanup
- **Performance Optimizations**:
  - Numba-accelerated core operations
  - Cache-efficient memory copying
  - Zero-copy views where possible
  - Parallel execution with adaptive strategies


## Architecture

Varbos uses a modular architecture with the following key components:

### Core Components

- **DirectMemory**: Low-level memory buffer with proper buffer protocol support
- **Memory**: Main interface for memory operations with policy-based behavior
- **NucleosManager**: Thread pool manager for background operations

### Policies

1. **Placement Policies**:
   - `BestFitPlacementPolicy`: Minimizes memory fragmentation
   - `BuddyAllocator`: High-performance power-of-2 size classes allocation

2. **Eviction Policies**:
   - `LRUEvictionPolicy`: Least Recently Used strategy
   - `LFUEvictionPolicy`: Least Frequently Used strategy

3. **Lock Policies**:
   - `SimpleLockPolicy`: Efficient reader-writer lock
   - `MultiLockStrategy`: Fine-grained lock strategy
   - `HybridLockManager`: Adaptive locking with optimistic reads

### Optimized Operations

- `parallel_memcpy`: Optimized memory copy with adaptive strategies
- `zero_memory`: Efficient memory zeroing
- `merge_free_blocks`: Memory defragmentation
- Various optimized search and fit algorithms

## Performance Characteristics

- **Memory Operations**: Optimized for both small and large data transfers
- **Concurrency**: Designed for high throughput in multi-threaded environments
- **Cache Efficiency**: Algorithms tuned for modern CPU cache hierarchies
- **Memory Overhead**: Minimal metadata overhead for efficient storage
- **Defragmentation**: Automatic background memory optimization

## Requirements

- Python 3.7+
- NumPy
- Numba