import bisect
import threading

import numpy as np
from numba import njit

from src.memory.core import ops


class PlacementPolicy:
    """Interface for memory allocation strategies"""

    def initialize(self, memory_size):
        """Initialize memory blocks"""
        pass

    def allocate(self, size):
        """Allocate memory block and return (start, end)"""
        pass

    def deallocate(self, start, size):
        """Deallocate memory block"""
        pass

    def rebuild(self, allocated_blocks):
        """Rebuild/defragment memory"""
        pass

    def get_stats(self):
        """Get memory allocation statistics"""
        pass


@njit(fastmath=True, cache=True)
def find_best_fit_block(blocks_array, size_needed):
    """Find the smallest block that fits the requested size using Numba"""
    if len(blocks_array) == 0:
        return -1, -1  # No blocks available

    best_size = np.iinfo(np.int64).max
    best_idx = -1

    # Scan for best fit
    for i in range(len(blocks_array)):
        block_size = blocks_array[i, 0]
        if size_needed <= block_size < best_size:
            best_size = block_size
            best_idx = i

            # Early exit on perfect match
            if block_size == size_needed:
                break

    if best_idx == -1:
        return -1, -1

    # Return the block size and address
    return blocks_array[best_idx, 0], blocks_array[best_idx, 1]


@njit(fastmath=True, cache=True)
def compute_stats(blocks_array):
    """
    Compute memory statistics with Numba acceleration

    Args:
        blocks_array: Numpy array of shape (n, 2) with (size, address) pairs

    Returns:
        (total_free, largest_block, frag_ratio)
    """
    if len(blocks_array) == 0:
        return 0, 0, 0.0

    total_free = 0
    largest_block = 0

    for i in range(len(blocks_array)):
        size = blocks_array[i, 0]
        total_free += size
        largest_block = max(largest_block, size)

    # Calculate fragmentation ratio
    if total_free > 0:
        frag_ratio = 1.0 - (largest_block / total_free)
    else:
        frag_ratio = 0.0

    return total_free, largest_block, frag_ratio


class BestFitPlacementPolicy(PlacementPolicy):
    """Best-fit memory allocation strategy with Numba acceleration"""

    def __init__(self, fragmentation_threshold=0.3):
        self.free_blocks = []  # (size, start_address)
        self.fragmentation_threshold = fragmentation_threshold
        self.lock = threading.RLock()
        self.memory_size = 0
        self.allocations = 0
        self.deallocations = 0
        self.rebuilds = 0

        # Cache for numpy array representation of free blocks
        self._np_blocks_cache = None
        self._cache_valid = False

    def _update_np_cache(self):
        """Update numpy array cache of free blocks"""
        if not self._cache_valid and self.free_blocks:
            self._np_blocks_cache = np.array(self.free_blocks, dtype=np.int64)
            self._cache_valid = True
        elif not self.free_blocks:
            self._np_blocks_cache = np.empty((0, 2), dtype=np.int64)
            self._cache_valid = True

    def _invalidate_cache(self):
        """Invalidate numpy array cache"""
        self._cache_valid = False

    def get_memory_blocks(self):
        """Get all memory blocks (both free and allocated) for visualization purposes"""
        with self.lock:
            # Return a copy to avoid threading issues
            free_blocks = [
                (size, start, False) for size, start in self.free_blocks
            ]  # size, start, is_allocated
            return free_blocks

    def initialize(self, memory_size):
        with self.lock:
            self.memory_size = memory_size
            self.free_blocks = [(memory_size, 0)]
            self._invalidate_cache()

    def allocate(self, size):
        with self.lock:
            if size <= 0:
                raise ValueError("Allocation size must be positive")

            # Use Numba-optimized search if we have enough blocks
            if len(self.free_blocks) > 10:
                self._update_np_cache()
                best_size, best_addr = find_best_fit_block(self._np_blocks_cache, size)

                if best_size != -1:
                    # Found a suitable block
                    # Remove from free list
                    self.free_blocks.remove((best_size, best_addr))
                    self._invalidate_cache()

                    # If block is larger than needed, keep remainder in free list
                    if best_size > size:
                        remaining_size = best_size - size
                        remaining_address = best_addr + size
                        bisect.insort(
                            self.free_blocks, (remaining_size, remaining_address)
                        )

                    end_address = best_addr + size
                    self.allocations += 1
                    return best_addr, end_address

                return None  # No suitable block found
            else:
                # Original Python implementation for small lists
                suitable_blocks = [
                    (block_size, addr)
                    for block_size, addr in self.free_blocks
                    if block_size >= size
                ]
                if not suitable_blocks:
                    return None  # No suitable block found

                # Find smallest block that fits
                best_block = min(suitable_blocks, key=lambda x: x[0])
                block_size, start_address = best_block

                # Remove this block from free list
                self.free_blocks.remove(best_block)

                # If block is larger than needed, keep remainder in free list
                if block_size > size:
                    remaining_size = block_size - size
                    remaining_address = start_address + size
                    bisect.insort(self.free_blocks, (remaining_size, remaining_address))

                end_address = start_address + size
                self.allocations += 1
                return start_address, end_address

    def deallocate(self, start, size):
        with self.lock:
            # Add block to free list
            bisect.insort(self.free_blocks, (size, start))
            self.deallocations += 1
            self._invalidate_cache()

            # Try to merge adjacent blocks using Numba-optimized function
            self._merge_free_blocks_optimized()

    def _merge_free_blocks_optimized(self):
        """Merge adjacent free blocks using Numba acceleration"""
        if len(self.free_blocks) <= 1:
            return

        # Convert to numpy array for Numba processing
        blocks_array = np.array(self.free_blocks, dtype=np.int64)

        # Call Numba-optimized function
        merged_blocks_array = ops.merge_adjacent_blocks(blocks_array)

        # Convert back to Python list
        merged_blocks = [
            (merged_blocks_array[i, 0], merged_blocks_array[i, 1])
            for i in range(len(merged_blocks_array))
        ]

        # Update free blocks, sorted by size for best-fit allocation
        self.free_blocks = sorted(merged_blocks)
        self._invalidate_cache()

    def _check_fragmentation_optimized(self):
        """Check fragmentation using Numba acceleration"""
        if len(self.free_blocks) <= 1:
            return False

        # Convert to numpy array for Numba processing
        self._update_np_cache()

        # Call Numba-optimized function
        _, is_high_fragmentation = ops.check_fragmentation(
            self._np_blocks_cache, self.fragmentation_threshold
        )

        return is_high_fragmentation

    def rebuild(self, allocated_blocks):
        """
        Actually rebuild memory by reorganizing allocated blocks sequentially.

        Args:
            allocated_blocks: List of (start, end) tuples representing allocated memory
        """
        with self.lock:
            if not allocated_blocks:
                # Nothing to rebuild
                self.free_blocks = [(self.memory_size, 0)]
                self._invalidate_cache()
                return False  # No rebuild needed

            # Step 1: Save current block information for tracking changes
            old_locations = {
                (start, end): (start, end) for start, end in allocated_blocks
            }
            new_locations = {}

            # Step 2: Reset free blocks to one large block
            self.free_blocks = [(self.memory_size, 0)]
            self._invalidate_cache()

            # Step 3: Allocate new blocks sequentially to eliminate fragmentation
            current_position = 0
            for start, end in sorted(allocated_blocks, key=lambda x: x[0]):
                size = end - start

                # Allocate new space at the current position
                new_start = current_position
                new_end = new_start + size
                new_locations[(start, end)] = (new_start, new_end)

                # Move to next position
                current_position = new_end

            # Step 4: Create one large free block for the remaining space
            if current_position < self.memory_size:
                remaining_size = self.memory_size - current_position
                self.free_blocks = [(remaining_size, current_position)]
                self._invalidate_cache()

            self.rebuilds += 1
            return (
                old_locations,
                new_locations,
            )  # Return mapping for actual data movement

    def get_stats(self):
        with self.lock:
            self._update_np_cache()

            # Use Numba-accelerated stats computation for large block lists
            if len(self.free_blocks) > 10:
                total_free, largest_block, frag_ratio = compute_stats(
                    self._np_blocks_cache
                )
            else:
                # Fallback for small lists
                total_free = sum(size for size, _ in self.free_blocks)
                largest_block = max((size for size, _ in self.free_blocks), default=0)
                frag_ratio = 1 - (largest_block / total_free) if total_free > 0 else 0

            return {
                "free_bytes": total_free,
                "free_blocks": len(self.free_blocks),
                "largest_free_block": largest_block,
                "allocations": self.allocations,
                "deallocations": self.deallocations,
                "rebuilds": self.rebuilds,
                "fragmentation_ratio": frag_ratio,
            }
