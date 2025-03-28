import bisect
import threading

from numba import njit


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


class BestFitPlacementPolicy(PlacementPolicy):
    """Best-fit memory allocation strategy"""

    def __init__(self, fragmentation_threshold=0.3):
        self.free_blocks = []  # (size, start_address)
        self.fragmentation_threshold = fragmentation_threshold
        self.lock = threading.RLock()
        self.memory_size = 0
        self.allocations = 0
        self.deallocations = 0
        self.rebuilds = 0

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

    def allocate(self, size):
        with self.lock:
            if size <= 0:
                raise ValueError("Allocation size must be positive")

            # Find best-fit block
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
            return (start_address, end_address)

    def deallocate(self, start, size):
        with self.lock:
            # Add block to free list
            bisect.insort(self.free_blocks, (size, start))
            self.deallocations += 1

            # Try to merge adjacent blocks
            self._merge_free_blocks()

            # Check fragmentation
            self._check_fragmentation()

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
                return False  # No rebuild needed

            # Step 1: Save current block information for tracking changes
            old_locations = {
                (start, end): (start, end) for start, end in allocated_blocks
            }
            new_locations = {}

            # Step 2: Reset free blocks to one large block
            self.free_blocks = [(self.memory_size, 0)]

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

            self.rebuilds += 1
            return (
                old_locations,
                new_locations,
            )  # Return mapping for actual data movement

    def _merge_free_blocks(self):
        """Merge adjacent free blocks"""
        if len(self.free_blocks) <= 1:
            return

        # Sort blocks by address
        blocks_by_address = sorted(self.free_blocks, key=lambda x: x[1])

        # Merge adjacent blocks
        merged_blocks = []
        i = 0
        while i < len(blocks_by_address):
            current_size, current_addr = blocks_by_address[i]
            current_end = current_addr + current_size

            # Look for adjacent blocks
            j = i + 1
            while j < len(blocks_by_address):
                next_size, next_addr = blocks_by_address[j]

                # If adjacent or overlapping
                if next_addr <= current_end:
                    # Merge blocks
                    new_end = max(current_end, next_addr + next_size)
                    current_size = new_end - current_addr
                    current_end = new_end
                    j += 1
                else:
                    # No more adjacent blocks
                    break

            merged_blocks.append((current_size, current_addr))
            i = j

        # Update free blocks, sorted by size for best-fit allocation
        self.free_blocks = sorted(merged_blocks)

    def _check_fragmentation(self):
        """Check fragmentation and report if high"""
        if len(self.free_blocks) <= 1:
            return False

        # Calculate fragmentation metrics
        total_free = sum(size for size, _ in self.free_blocks)
        largest_block = max(size for size, _ in self.free_blocks)
        fragmentation_ratio = 1 - (largest_block / total_free) if total_free > 0 else 0

        # Return True if fragmentation is high
        return fragmentation_ratio > self.fragmentation_threshold

    def get_stats(self):
        with self.lock:
            total_free = sum(size for size, _ in self.free_blocks)
            return {
                "free_bytes": total_free,
                "free_blocks": len(self.free_blocks),
                "largest_free_block": max(
                    (size for size, _ in self.free_blocks), default=0
                ),
                "allocations": self.allocations,
                "deallocations": self.deallocations,
                "rebuilds": self.rebuilds,
                "fragmentation_ratio": (
                    1
                    - (
                        max((size for size, _ in self.free_blocks), default=0)
                        / total_free
                    )
                    if total_free > 0
                    else 0
                ),
            }


@njit
def find_best_bin(size, bin_sizes):
    """Find the best bin for a given size using binary search"""
    left, right = 0, len(bin_sizes) - 1
    best_fit = -1

    while left <= right:
        mid = (left + right) // 2
        if bin_sizes[mid] >= size:
            best_fit = mid
            right = mid - 1
        else:
            left = mid + 1

    return best_fit


class BuddyAllocator(PlacementPolicy):
    """High-performance buddy allocator with power-of-2 size classes"""

    def __init__(self, memory_size, min_block_size=64):
        self.memory_size = memory_size
        self.min_block_size = min_block_size

        # Calculate levels in the buddy system
        self.levels = 0
        size = memory_size
        while size >= min_block_size:
            size //= 2
            self.levels += 1

        # Free block lists for each level
        self.free_lists = [[] for _ in range(self.levels)]

        # Block size at each level
        self.level_sizes = [memory_size // (2**i) for i in range(self.levels)]

        # Initialize with one large block
        self.free_lists[0] = [0]  # Address of the initial block

        # Block metadata (allocated or free)
        self.block_status = {}  # (level, address) -> bool(is_allocated)

        # Performance tracking
        self.allocations = 0
        self.deallocations = 0
        self.splits = 0
        self.merges = 0

        # Lock for thread safety
        self.lock = threading.RLock()

    def allocate(self, size):
        """Allocate a block of memory"""
        with self.lock:
            # Round up to power of 2
            padded_size = max(self.min_block_size, 1 << (size - 1).bit_length())

            # Find the right level
            level = None
            for i, level_size in enumerate(self.level_sizes):
                if level_size >= padded_size:
                    level = i
                    break

            if level is None:
                return None  # Requested size too large

            # Try to allocate from this level
            addr = self._allocate_from_level(level)
            if addr is not None:
                self.allocations += 1
                return (addr, addr + padded_size)

            return None

    def _allocate_from_level(self, level):
        """Allocate from a specific level, splitting blocks if needed"""
        # Check if we have a free block at this level
        if self.free_lists[level]:
            addr = self.free_lists[level].pop(0)
            self.block_status[(level, addr)] = True  # Mark as allocated
            return addr

        # No free blocks at this level, try to split a block from a higher level
        if level > 0:
            parent_addr = self._allocate_from_level(level - 1)
            if parent_addr is not None:
                # Split the parent block
                self.splits += 1
                buddy_addr = parent_addr + self.level_sizes[level]

                # Add buddy to free list
                self.free_lists[level].append(buddy_addr)
                self.block_status[(level, buddy_addr)] = False  # Mark as free

                # Mark this block as allocated
                self.block_status[(level, parent_addr)] = True

                return parent_addr

        return None  # Couldn't allocate

    def deallocate(self, addr, size):
        """Deallocate a block of memory"""
        with self.lock:
            # Find the level of this block
            padded_size = max(self.min_block_size, 1 << (size - 1).bit_length())
            level = None
            for i, level_size in enumerate(self.level_sizes):
                if level_size == padded_size:
                    level = i
                    break

            if level is None:
                raise ValueError(f"Invalid block size: {size}")

            # Deallocate and potentially merge
            self._deallocate_at_level(addr, level)
            self.deallocations += 1

    def _deallocate_at_level(self, addr, level):
        """Deallocate at a specific level and merge if possible"""
        # Mark as free
        self.block_status[(level, addr)] = False

        # Try to merge with buddy
        if level < self.levels - 1:
            buddy_addr = self._find_buddy(addr, level)

            # If buddy is free, we can merge
            if buddy_addr in self.free_lists[level]:
                self.merges += 1

                # Remove buddy from free list
                self.free_lists[level].remove(buddy_addr)

                # Find parent address (always the lower address)
                parent_addr = min(addr, buddy_addr)

                # Deallocate at parent level (recursive merge)
                self._deallocate_at_level(parent_addr, level - 1)
            else:
                # No merge possible, add to free list
                self.free_lists[level].append(addr)
        else:
            # At lowest level, just add to free list
            self.free_lists[level].append(addr)

    def _find_buddy(self, addr, level):
        """Find the buddy address for a given block"""
        level_size = self.level_sizes[level]
        return addr ^ level_size

    def get_stats(self):
        """Get allocator statistics"""
        with self.lock:
            free_blocks = sum(len(lst) for lst in self.free_lists)
            free_bytes = sum(
                len(lst) * self.level_sizes[i] for i, lst in enumerate(self.free_lists)
            )

            largest_free = 0
            for i, lst in enumerate(self.free_lists):
                if lst:
                    largest_free = max(largest_free, self.level_sizes[i])

            return {
                "free_bytes": free_bytes,
                "free_blocks": free_blocks,
                "largest_free_block": largest_free,
                "allocations": self.allocations,
                "deallocations": self.deallocations,
                "splits": self.splits,
                "merges": self.merges,
                "fragmentation_ratio": (
                    1 - (largest_free / free_bytes) if free_bytes > 0 else 0
                ),
            }
