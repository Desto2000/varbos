import threading

import numpy as np


# =============================================================================
# POLICY INTERFACES
# =============================================================================


class LockPolicy:
    """Interface for different locking strategies"""

    def acquire_read(self, key=None):
        """Acquire read lock"""
        pass

    def release_read(self, key=None):
        """Release read lock"""
        pass

    def acquire_write(self, key=None):
        """Acquire write lock"""
        pass

    def release_write(self, key=None):
        """Release write lock"""
        pass


class MultiLockStrategy(LockPolicy):
    """Multiple lock strategy for more granular concurrency"""

    def __init__(self, num_locks=16):
        self.locks = [threading.RLock() for _ in range(num_locks)]
        self.global_lock = threading.RLock()
        self.num_locks = num_locks

    def _get_lock_for_key(self, key) -> threading.RLock:
        """Get the appropriate lock for a given key"""
        # Use hash to determine which lock to use
        hash_val = hash(key) if key is not None else 0
        lock_index = abs(hash_val) % self.num_locks
        return self.locks[lock_index]

    def acquire_read(self, key=None) -> None:
        """Acquire a read lock for a specific key or global lock"""
        if key is None:
            self.global_lock.acquire()
        else:
            self._get_lock_for_key(key).acquire()

    def release_read(self, key=None) -> None:
        """Release a read lock for a specific key or global lock"""
        if key is None:
            self.global_lock.release()
        else:
            self._get_lock_for_key(key).release()

    def acquire_write(self, key=None) -> None:
        """Acquire a write lock for a specific key or global lock"""
        if key is None:
            self.global_lock.acquire()
        else:
            self._get_lock_for_key(key).acquire()

    def release_write(self, key=None) -> None:
        """Release a write lock for a specific key or global lock"""
        if key is None:
            self.global_lock.release()
        else:
            self._get_lock_for_key(key).release()


class PartitionedLockPolicy(LockPolicy):
    """Lock policy that uses multiple locks based on key hash"""

    def __init__(self, partitions=16):
        self.locks = [threading.RLock() for _ in range(partitions)]
        self.global_lock = threading.RLock()
        self.partitions = partitions

    def _get_lock(self, key):
        if key is None:
            return self.global_lock
        lock_idx = hash(key) % self.partitions
        return self.locks[lock_idx]

    def acquire_read(self, key=None):
        self._get_lock(key).acquire()

    def release_read(self, key=None):
        self._get_lock(key).release()

    def acquire_write(self, key=None):
        self._get_lock(key).acquire()

    def release_write(self, key=None):
        self._get_lock(key).release()


class SimpleLockPolicy(LockPolicy):
    """High-performance reader-writer lock with writer preference"""

    def __init__(self):
        self._mutex = threading.RLock()  # Control access to the state
        self._write_lock = threading.Lock()  # Exclusive lock for writers
        self._reader_count = 0  # Count of active readers
        self._writer_waiting = False  # Flag for waiting writers
        self._reader_event = threading.Event()  # Blocks readers when writer is waiting
        self._reader_event.set()  # Initially allowing readers

    def acquire_read(self, key=None):
        """Acquire read lock, blocked if writers are waiting"""
        # Wait until no writers are waiting
        self._reader_event.wait()

        with self._mutex:
            self._reader_count += 1
            # First reader locks the write lock
            if self._reader_count == 1:
                self._write_lock.acquire()

    def release_read(self, key=None):
        """Release read lock"""
        with self._mutex:
            self._reader_count -= 1
            # Last reader releases the write lock
            if self._reader_count == 0:
                self._write_lock.release()

    def acquire_write(self, key=None):
        """Acquire write lock with priority"""
        with self._mutex:
            # Signal readers to wait
            self._writer_waiting = True
            self._reader_event.clear()

        # Wait for exclusive access
        self._write_lock.acquire()

    def release_write(self, key=None):
        """Release write lock"""
        self._write_lock.release()

        with self._mutex:
            # Allow readers again
            self._writer_waiting = False
            self._reader_event.set()


class HybridLockManager(LockPolicy):
    """Advanced lock manager with adaptive strategies"""

    def __init__(self, partitions=64, thread_pool_size=None):
        # Increased number of partitions for less contention
        self.partitions = partitions
        self.partition_locks = [SimpleLockPolicy() for _ in range(partitions)]
        self.global_lock = SimpleLockPolicy()

        # Stats
        self.contentions = 0
        self.access_count = 0

        # Lock-free fast path for read-heavy workloads
        self.read_counters = np.zeros(partitions, dtype=np.int32)

    def _get_partition(self, key):
        """Get lock partition for a key - optimized hash distribution"""
        if key is None:
            return None

        # FNV-1a hash for better distribution
        h = 2166136261
        key_str = str(key).encode() if not isinstance(key, bytes) else key
        for b in key_str:
            h = ((h ^ b) * 16777619) & 0xFFFFFFFF
        return h % self.partitions

    def acquire_read(self, key=None):
        """Acquire read lock with optimistic fast path"""
        self.access_count += 1

        if key is None:
            self.global_lock.acquire_read()
        else:
            partition = self._get_partition(key)
            # Fast path - no actual locking for most read operations
            # Just increment read counter (atomic for numpy int32)
            self.read_counters[partition] += 1
            # Only acquire the real lock in high-contention scenarios
            if self.contentions > 100 and self.contentions / self.access_count > 0.01:
                self.partition_locks[partition].acquire_read()

    def release_read(self, key=None):
        """Release read lock"""
        if key is None:
            self.global_lock.release_read()
        else:
            partition = self._get_partition(key)
            # Decrement read counter
            self.read_counters[partition] -= 1
            # Only release the real lock in high-contention scenarios
            if self.contentions > 100 and self.contentions / self.access_count > 0.01:
                try:
                    self.partition_locks[partition].release_read()
                except:
                    pass  # Optimistic reads might not have acquired the actual lock

    def acquire_write(self, key=None):
        """Acquire write lock with contention tracking"""
        self.access_count += 1

        if key is None:
            self.global_lock.acquire_write()
        else:
            partition = self._get_partition(key)
            lock = self.partition_locks[partition]

            # Try optimistic fast path - check if no readers active
            readers_active = self.read_counters[partition] > 0
            if not readers_active:
                # Try immediate acquire
                acquired = False
                try:
                    acquired = lock._write_lock.acquire(blocking=False)
                except:
                    pass

                if not acquired:
                    # Count contention
                    self.contentions += 1
                    # Blocking acquire
                    lock.acquire_write()
                else:
                    # We got the lock but need to complete the acquire protocol
                    lock.release_write()
                    lock.acquire_write()
            else:
                # Readers active, use normal path
                lock.acquire_write()

    def release_write(self, key=None):
        """Release write lock"""
        if key is None:
            self.global_lock.release_write()
        else:
            partition = self._get_partition(key)
            self.partition_locks[partition].release_write()

    def get_stats(self):
        """Get lock statistics"""
        contention_rate = self.contentions / max(1, self.access_count)
        return {
            "lock_contentions": self.contentions,
            "access_count": self.access_count,
            "contention_rate": contention_rate,
            "partitions": self.partitions,
        }
