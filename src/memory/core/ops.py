import numpy as np
from numba import njit, prange


@njit(parallel=True)
def parallel_memcpy(src, dest, length):
    """Parallelized memory copy operation"""
    # Process in chunks for better cache performance
    chunk_size = min(1024, length // 4)
    if chunk_size < 1:
        chunk_size = length

    # Number of chunks
    num_chunks = (length + chunk_size - 1) // chunk_size

    # Copy in parallel chunks
    for i in prange(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, length)
        for j in range(start, end):
            dest[j] = src[j]

    return length


@njit
def combined_fit(free_blocks, size):
    """Find smallest block that fits the requested size"""
    if len(free_blocks) == 0:
        return (-1, -1)  # No blocks available

    best_size = -1
    best_addr = -1

    # Manually find the smallest block that fits
    for block_size, addr in free_blocks:
        if block_size >= size:
            if best_size == -1 or block_size < best_size:
                best_size = block_size
                best_addr = addr

    if best_size == -1:
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
    """Find best fit block for size needed"""
    if len(free_blocks) == 0:
        return -1

    best_fit = -1
    min_waste = np.iinfo(np.int64).max

    # First find blocks that are big enough
    for i in range(len(free_blocks)):
        block_size = free_blocks[i][0]
        if block_size >= size_needed:
            waste = block_size - size_needed
            if waste < min_waste:
                min_waste = waste
                best_fit = i

    return best_fit
