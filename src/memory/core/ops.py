import numpy as np
from numba import njit, prange


@njit(parallel=True)
def parallel_memcpy(src, dest, length):
    """Parallelized memory copy operation with enhanced performance"""
    # Determine optimal chunk size based on data size
    if length <= 256:
        # For small data, avoid parallelization overhead
        for j in range(length):
            dest[j] = src[j]
        return length

    # Process in chunks for better cache performance
    # Use power of 2 for chunk size to optimize cache access
    chunk_size = 1024
    while chunk_size * 4 > length and chunk_size > 64:
        chunk_size = chunk_size // 2

    # Number of chunks
    num_chunks = (length + chunk_size - 1) // chunk_size

    # Copy in parallel chunks
    for i in prange(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, length)

        # Process 8 elements at a time when possible
        main_end = start + ((end - start) // 8) * 8

        # Main unrolled loop for better performance
        for j in range(start, main_end, 8):
            dest[j] = src[j]
            dest[j + 1] = src[j + 1]
            dest[j + 2] = src[j + 2]
            dest[j + 3] = src[j + 3]
            dest[j + 4] = src[j + 4]
            dest[j + 5] = src[j + 5]
            dest[j + 6] = src[j + 6]
            dest[j + 7] = src[j + 7]

        # Handle remaining elements
        for j in range(main_end, end):
            dest[j] = src[j]

    return length


@njit
def combined_fit(free_blocks, size):
    """Find smallest block that fits the requested size (optimized)"""
    if len(free_blocks) == 0:
        return (-1, -1)  # No blocks available

    best_size = np.iinfo(np.int64).max
    best_addr = -1

    # Find smallest block that fits - optimize by checking min size block first
    for i in range(len(free_blocks)):
        block_size, addr = free_blocks[i]
        if block_size >= size and block_size < best_size:
            best_size = block_size
            best_addr = addr

            # Early exit if perfect fit found
            if block_size == size:
                break

    if best_addr == -1:
        return (-1, -1)  # No suitable block found

    return (best_size, best_addr)


@njit
def binary_search_first_fit(free_blocks, size_needed):
    """Find first block that fits size needed using binary search"""
    if len(free_blocks) == 0:
        return -1

    # Binary search for first block >= size_needed
    left = 0
    right = len(free_blocks) - 1
    result = -1

    while left <= right:
        mid = (left + right) // 2
        if free_blocks[mid][0] >= size_needed:  # block size >= needed
            result = mid
            right = mid - 1  # look for smaller blocks that also fit
        else:
            left = mid + 1

    return result


@njit
def binary_search_best_fit(free_blocks, size_needed):
    """Find best fit block for size needed (optimized)"""
    if len(free_blocks) == 0:
        return -1

    best_fit = -1
    min_waste = np.iinfo(np.int64).max

    # Skip blocks too small for faster selection
    for i in range(len(free_blocks)):
        block_size = free_blocks[i][0]
        if block_size >= size_needed:
            waste = block_size - size_needed
            if waste < min_waste:
                min_waste = waste
                best_fit = i

                # Early exit on perfect fit
                if waste == 0:
                    break

    return best_fit


@njit
def zero_memory(buffer, start, length):
    """
    Efficiently zero out a memory region

    Args:
        buffer: Memory buffer (supports buffer protocol)
        start: Start position
        length: Number of bytes to zero
    """
    # Handle small regions directly
    if length <= 64:
        for i in range(start, start + length):
            buffer[i] = 0
        return

    # For larger regions, use block clearing
    end = start + length
    step = 64  # Process 64 bytes at a time

    # Main loop for 64-byte blocks
    main_end = start + (length // step) * step
    for pos in range(start, main_end, step):
        for i in range(64):
            buffer[pos + i] = 0

    # Handle remaining bytes
    for pos in range(main_end, end):
        buffer[pos] = 0
