import queue
import threading

import numpy as np
import pyarrow as pa

from src.memory.core.fast_access import FastLookupTable
from src.memory.core.memory import DirectMemory
from src.memory.eviction import LRUEvictionPolicy
from src.memory.head import SimpleHeadPolicy
from src.memory.nucleos import NucleosManager
from src.memory.placement import BestFitPlacementPolicy, BuddyAllocator
from src.memory.sync import HybridLockManager


class Memory:
    """
    Unified memory system with policy-based behavior.
    """

    def __init__(
        self,
        memory_size,
        lock_policy=None,
        eviction_policy=None,
        placement_policy=None,
        head_policy=None,
    ):
        """
        Initialize the memory system.

        Args:
            memory_size: Size of memory in bytes
            lock_policy: Policy for synchronization
            eviction_policy: Policy for cache eviction
            placement_policy: Policy for memory allocation
        """
        # Initialize memory
        self.memory_size = memory_size
        self.mem = DirectMemory(self.memory_size)

        self.buffer_pool = pa.default_memory_pool()

        # Set up policies (with defaults if not provided)
        self.lock_policy = lock_policy or HybridLockManager()
        self.eviction_policy = eviction_policy or LRUEvictionPolicy()
        self.placement_policy = placement_policy or BuddyAllocator(self.memory_size)

        self.head = head_policy or SimpleHeadPolicy()

        # Initialize placement policy
        self.placement_policy.initialize(memory_size)

        # Key to memory location mapping
        self.lookup_table = FastLookupTable()  # key -> (start, end)

        # Start thread manager
        self.thread_manager = NucleosManager(self)

        # Statistics
        self.evictions = 0

    def __getitem__(self, key):
        """Get item from memory - priority on main thread speed"""
        if key not in self.lookup_table:
            self.eviction_policy.record_miss()
            raise KeyError(f"Key '{key}' not found")

        start, end = self.lookup_table[key]
        data = memoryview(self.mem[start:end])

        # Update access tracking (asynchronously)
        self.eviction_policy.on_access(key)

        return data

    def __setitem__(self, key, value):
        """Store item in memory - optimized for main thread performance"""
        if not isinstance(value, (bytes, memoryview, np.ndarray)):
            if isinstance(value, (bytearray)):
                # Optimize for bytearray by using memoryview
                value = memoryview(value)
            else:
                raise TypeError(
                    "Value must be bytes, memoryview, numpy array, or bytearray"
                )

        self.lock_policy.acquire_write(key)
        try:
            # If key exists, try to reuse the space
            if key in self.lookup_table:
                old_start, old_end = self.lookup_table[key]
                old_size = old_end - old_start

                # If new data fits in existing space, reuse it
                if len(value) <= old_size:
                    self.mem[old_start : old_start + len(value)] = value

                    # If we have leftover space, deallocate it in background
                    if len(value) < old_size:
                        leftover_start = old_start + len(value)
                        leftover_size = old_size - len(value)

                        # Update lookup table
                        self.lookup_table[key] = (old_start, leftover_start)

                        # Schedule background cleanup
                        self.thread_manager.schedule_task(
                            "free", (leftover_start, leftover_size)
                        )

                    # Update eviction policy
                    self.eviction_policy.on_access(key)
                    return

                # New data doesn't fit - deallocate old space (in background)
                self.thread_manager.schedule_task("free", (old_start, old_size))

            # Allocate new space
            result = self.placement_policy.allocate(len(value))

            # If allocation failed, try eviction and rebuilding
            if result is None:
                # Try immediate eviction
                self._evict_internal(1)

                # Try allocation again
                result = self.placement_policy.allocate(len(value))

                # If still fails, try rebuilding
                if result is None:
                    self._rebuild_internal()
                    result = self.placement_policy.allocate(len(value))

                    # If still no space, we're out of memory
                    if result is None:
                        raise MemoryError(
                            f"Not enough memory to store data ({len(value)} bytes)"
                        )

            # Write data to memory
            start, end = result
            self.mem[start:end] = value

            # Update lookup table
            self.lookup_table[key] = (start, end)

            # Update eviction policy
            self.eviction_policy.on_insert(key)

        finally:
            self.lock_policy.release_write(key)

    def __delitem__(self, key):
        """Delete item from memory"""
        self.lock_policy.acquire_write(key)
        try:
            if key not in self.lookup_table:
                raise KeyError(f"Key '{key}' not found")

            start, end = self.lookup_table[key]
            size = end - start

            # Remove from lookup table
            del self.lookup_table[key]

            # Update eviction policy
            self.eviction_policy.on_remove(key)

            # Schedule background deallocation
            self.thread_manager.schedule_task("free", (start, size))

        finally:
            self.lock_policy.release_write(key)

    def __contains__(self, key):
        """Check if key exists in memory"""
        self.lock_policy.acquire_read(key)
        try:
            return key in self.lookup_table
        finally:
            self.lock_policy.release_read(key)

    def get(self, key, default=None):
        """Get item with default value if not found"""
        try:
            return self[key]
        except KeyError:
            return default

    def _deallocate_internal(self, start, size):
        """Internal method to deallocate memory (called by thread manager)"""
        self.lock_policy.acquire_write(None)  # Global lock for memory operations
        try:
            self.placement_policy.deallocate(start, size)
        finally:
            self.lock_policy.release_write(None)

    def _rebuild_internal(self):
        """Internal method to rebuild memory (called by thread manager)"""
        self.lock_policy.acquire_write(None)  # Global lock
        try:
            # Get current allocated blocks
            allocated_blocks = [
                (start, end) for start, end in self.lookup_table.values()
            ]

            # Skip rebuild if no blocks
            if not allocated_blocks:
                return

            # Rebuild memory using placement policy
            result = self.placement_policy.rebuild(allocated_blocks)

            # If rebuild returned mapping info, move the actual data
            if result:
                old_locations, new_locations = result

                # Create reverse mapping from old locations to keys
                location_to_key = {
                    (start, end): key for key, (start, end) in self.lookup_table.items()
                }

                # Move data to new locations
                for old_loc, new_loc in new_locations.items():
                    old_start, old_end = old_loc
                    new_start, new_end = new_loc

                    # Only move if locations actually changed
                    if old_start != new_start:
                        # Get the key for this data
                        if old_loc in location_to_key:
                            key = location_to_key[old_loc]

                            # Read data from old location
                            data = self.mem[old_start:old_end]

                            # Write to new location
                            self.mem[new_start:new_end] = data

                            # Update lookup table
                            self.lookup_table[key] = new_loc

                print(f"Memory rebuilt: {len(allocated_blocks)} objects reorganized")

        finally:
            self.lock_policy.release_write(None)

    def _evict_internal(self, count=1):
        """Internal method to evict items (called by thread manager)"""
        if count <= 0:
            return 0

        self.lock_policy.acquire_write(None)  # Global lock
        try:
            # Get eviction candidates
            candidates = self.eviction_policy.get_eviction_candidates(count)
            if not candidates:
                return 0

            evicted = 0
            for key in candidates:
                if key in self.lookup_table:
                    # Get memory location
                    start, end = self.lookup_table[key]
                    size = end - start

                    # Remove from lookup table
                    del self.lookup_table[key]

                    # Update eviction policy
                    self.eviction_policy.on_remove(key)

                    # Deallocate memory
                    self.placement_policy.deallocate(start, size)

                    evicted += 1

            self.evictions += evicted
            return evicted

        finally:
            self.lock_policy.release_write(None)

    def clear(self):
        """Clear all items from memory"""
        self.lock_policy.acquire_write(None)  # Global lock
        try:
            # Save keys for policy updates
            keys = list(self.lookup_table.keys())

            # Clear lookup table
            self.lookup_table = {}

            # Reset placement policy
            self.placement_policy.initialize(self.memory_size)

            # Update eviction policy
            for key in keys:
                self.eviction_policy.on_remove(key)

        finally:
            self.lock_policy.release_write(None)

    def get_stats(self):
        """Get memory statistics"""
        self.lock_policy.acquire_read(None)
        try:
            placement_stats = self.placement_policy.get_stats()
            eviction_stats = self.eviction_policy.get_stats()
            thread_stats = self.thread_manager.get_stats()

            # Calculate combined stats
            total_bytes = self.memory_size
            free_bytes = placement_stats.get("free_bytes", 0)
            used_bytes = total_bytes - free_bytes

            return {
                "memory": {
                    "total_bytes": total_bytes,
                    "used_bytes": used_bytes,
                    "free_bytes": free_bytes,
                    "percent_used": (
                        (used_bytes / total_bytes) * 100 if total_bytes > 0 else 0
                    ),
                    "fragmentation_ratio": placement_stats.get(
                        "fragmentation_ratio", 0
                    ),
                    "free_blocks": placement_stats.get("free_blocks", 0),
                    "allocations": placement_stats.get("allocations", 0),
                    "deallocations": placement_stats.get("deallocations", 0),
                    "rebuilds": placement_stats.get("rebuilds", 0),
                },
                "cache": {
                    "policy_type": eviction_stats.get("policy_type", "unknown"),
                    "items": len(self.lookup_table),
                    "hits": eviction_stats.get("hits", 0),
                    "misses": eviction_stats.get("misses", 0),
                    "hit_ratio": eviction_stats.get("hit_ratio", 0),
                    "evictions": self.evictions,
                },
                "threads": thread_stats,
            }

        finally:
            self.lock_policy.release_read(None)

    def shutdown(self):
        """Shutdown the memory system"""
        self.thread_manager.shutdown()

    def __str__(self):
        stats = self.get_stats()

        return (
            f"MemorySystem({len(self.lookup_table)} items, "
            f"{stats['memory']['percent_used']:.1f}% used, "
            f"{stats['cache']['hit_ratio']*100:.1f}% hit ratio, "
            f"{stats['memory']['free_blocks']} free blocks)"
        )
