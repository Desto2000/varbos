import numpy as np
from numba import jit, njit, prange, types
from numba.typed import List


@njit(parallel=True, fastmath=True, cache=True)
def parallel_memcpy(src, dest, length):
    """Parallelized memory copy operation with adaptive optimization"""
    # Small data optimization - avoid parallelization overhead
    if length < 256:
        for j in range(length):
            dest[j] = src[j]
        return length

    # Medium data optimization - use unrolled loops
    elif length < 16384:  # 16KB threshold
        # Process in 8-byte chunks
        main_part = (length // 8) * 8
        for j in range(0, main_part, 8):
            dest[j] = src[j]
            dest[j + 1] = src[j + 1]
            dest[j + 2] = src[j + 2]
            dest[j + 3] = src[j + 3]
            dest[j + 4] = src[j + 4]
            dest[j + 5] = src[j + 5]
            dest[j + 6] = src[j + 6]
            dest[j + 7] = src[j + 7]

        # Handle remaining bytes
        for j in range(main_part, length):
            dest[j] = src[j]

        return length

    # Large data - use parallel processing with optimal chunk size
    else:
        # Optimize chunk size for cache efficiency (L1 cache ~32KB on many CPUs)
        chunk_size = 8192  # 8KB chunks

        # Number of chunks
        num_chunks = (length + chunk_size - 1) // chunk_size

        # Copy in parallel chunks
        for i in prange(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, length)

            # Use 8-byte chunks within each parallel section
            main_part = start + ((end - start) // 8) * 8

            # Unrolled copy loop for better performance
            for j in range(start, main_part, 8):
                dest[j] = src[j]
                dest[j + 1] = src[j + 1]
                dest[j + 2] = src[j + 2]
                dest[j + 3] = src[j + 3]
                dest[j + 4] = src[j + 4]
                dest[j + 5] = src[j + 5]
                dest[j + 6] = src[j + 6]
                dest[j + 7] = src[j + 7]

            # Handle remainder
            for j in range(main_part, end):
                dest[j] = src[j]

    return length


@njit(cache=True)
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


@njit(fastmath=True, cache=True)
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


# FIXED: Remove parallel=True to address the warning, since the loop has early exit
@njit(fastmath=True, cache=True)
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


# Optimized version of merge_free_blocks with consistent return type for Numba
@njit(fastmath=True, cache=True)
def merge_free_blocks_numba(blocks_array):
    """Merge adjacent free blocks with consistent return type for Numba"""
    if len(blocks_array) <= 1:
        return blocks_array

    # Sort blocks by address
    # Create a sorted copy since we can't sort in-place with numba
    sorted_indices = np.argsort(blocks_array[:, 1])
    blocks_by_address = np.empty_like(blocks_array)

    for i in range(len(sorted_indices)):
        idx = sorted_indices[i]
        blocks_by_address[i, 0] = blocks_array[idx, 0]  # size
        blocks_by_address[i, 1] = blocks_array[idx, 1]  # address

    # Merge adjacent blocks
    result_size = 0
    merged_blocks = np.empty_like(blocks_array)

    i = 0
    while i < len(blocks_by_address):
        current_size = blocks_by_address[i, 0]
        current_addr = blocks_by_address[i, 1]
        current_end = current_addr + current_size

        # Look for adjacent blocks
        j = i + 1
        while j < len(blocks_by_address):
            next_addr = blocks_by_address[j, 1]

            # If adjacent or overlapping
            if next_addr <= current_end:
                # Merge blocks
                next_size = blocks_by_address[j, 0]
                new_end = max(current_end, next_addr + next_size)
                current_size = new_end - current_addr
                current_end = new_end
                j += 1
            else:
                # No more adjacent blocks
                break

        merged_blocks[result_size, 0] = current_size
        merged_blocks[result_size, 1] = current_addr
        result_size += 1
        i = j

    return merged_blocks[:result_size]


# Python version to be called from Python code
def merge_free_blocks(blocks_by_address):
    """Python wrapper for merge_free_blocks_numba

    Takes a list of tuples and returns a list of tuples
    """
    if len(blocks_by_address) <= 1:
        return blocks_by_address

    # Convert list of tuples to numpy array
    blocks_array = np.array(blocks_by_address, dtype=np.int64)

    # Call numba optimized function
    result_array = merge_free_blocks_numba(blocks_array)

    # Convert back to list of tuples
    result = [
        (result_array[i, 0], result_array[i, 1]) for i in range(len(result_array))
    ]

    # Return sorted by size for best-fit allocation
    return sorted(result)


@njit(parallel=True, fastmath=True, cache=True)
def zero_memory(buffer, start, length):
    """
    Efficiently zero out a memory region

    Args:
        buffer: Memory buffer (supports buffer protocol)
        start: Start position
        length: Number of bytes to zero
    """
    # Handle small regions directly
    if length <= 128:
        for i in range(start, start + length):
            buffer[i] = 0
        return

    # For larger regions, use block clearing with parallelization
    end = start + length

    # Optimize for cache line size (typically 64 bytes)
    # Process in 64-byte chunks for better cache utilization
    chunk_size = 64

    # Calculate number of full chunks
    full_chunks = length // chunk_size

    # Process full chunks in parallel
    if full_chunks > 1:
        for chunk_idx in prange(full_chunks):
            chunk_start = start + chunk_idx * chunk_size
            for i in range(chunk_size):
                buffer[chunk_start + i] = 0

    # Handle remaining bytes
    remainder_start = start + full_chunks * chunk_size
    for i in range(remainder_start, end):
        buffer[i] = 0


# New function: optimized memset with value
@njit(parallel=True, fastmath=True, cache=True)
def memset(buffer, start, length, value=0):
    """
    Efficiently set memory region to a specific value

    Args:
        buffer: Memory buffer (supports buffer protocol)
        start: Start position
        length: Number of bytes to set
        value: Value to set (default 0)
    """
    # Handle small regions directly
    if length <= 128:
        for i in range(start, start + length):
            buffer[i] = value
        return

    # For larger regions, use block setting with parallelization
    end = start + length

    # Optimize for cache line size
    chunk_size = 64

    # Calculate number of full chunks
    full_chunks = length // chunk_size

    # Process full chunks in parallel
    if full_chunks > 1:
        for chunk_idx in prange(full_chunks):
            chunk_start = start + chunk_idx * chunk_size
            for i in range(chunk_size):
                buffer[chunk_start + i] = value

    # Handle remaining bytes
    remainder_start = start + full_chunks * chunk_size
    for i in range(remainder_start, end):
        buffer[i] = value
