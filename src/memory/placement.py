import bisect
import threading

from src.memory.core.ops import combined_fit


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

            best_block = combined_fit(self.free_blocks, size)
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
