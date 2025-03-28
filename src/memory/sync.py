import threading


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


class SimpleLockPolicy(LockPolicy):
    """Simple lock policy using a single RLock"""
    def __init__(self):
        self.lock = threading.RLock()

    def acquire_read(self, key=None):
        self.lock.acquire()

    def release_read(self, key=None):
        self.lock.release()

    def acquire_write(self, key=None):
        self.lock.acquire()

    def release_write(self, key=None):
        self.lock.release()

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
