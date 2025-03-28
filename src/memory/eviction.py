import threading
import time
from collections import OrderedDict


class EvictionPolicy:
    """Interface for cache eviction policies"""

    def on_access(self, key):
        """Called when an item is accessed"""
        pass

    def on_insert(self, key):
        """Called when an item is inserted"""
        pass

    def on_remove(self, key):
        """Called when an item is removed"""
        pass

    def get_eviction_candidates(self, count=1):
        """Get keys that should be evicted"""
        pass

    def get_stats(self):
        """Get policy statistics"""
        pass


class LRUEvictionPolicy(EvictionPolicy):
    """Least Recently Used eviction policy"""

    def __init__(self):
        self.access_times = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def on_access(self, key):
        with self.lock:
            self.access_times.pop(key, None)
            self.access_times[key] = time.time()
            self.hits += 1

    def on_insert(self, key):
        with self.lock:
            self.access_times.pop(key, None)
            self.access_times[key] = time.time()

    def on_remove(self, key):
        with self.lock:
            self.access_times.pop(key, None)

    def get_eviction_candidates(self, count=1):
        with self.lock:
            candidates = []
            for key in self.access_times:
                candidates.append(key)
                if len(candidates) >= count:
                    break
            return candidates

    def record_miss(self):
        with self.lock:
            self.misses += 1

    def get_stats(self):
        with self.lock:
            return {
                "policy_type": "LRU",
                "tracked_items": len(self.access_times),
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": (
                    self.hits / (self.hits + self.misses)
                    if (self.hits + self.misses) > 0
                    else 0
                ),
            }


class LFUEvictionPolicy(EvictionPolicy):
    """Least Frequently Used eviction policy"""

    def __init__(self):
        self.access_counts = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def on_access(self, key):
        with self.lock:
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            self.hits += 1

    def on_insert(self, key):
        with self.lock:
            self.access_counts[key] = 1

    def on_remove(self, key):
        with self.lock:
            self.access_counts.pop(key, None)

    def get_eviction_candidates(self, count=1):
        with self.lock:
            # Sort by access count (ascending)
            sorted_items = sorted(self.access_counts.items(), key=lambda x: x[1])
            return [key for key, _ in sorted_items[:count]]

    def record_miss(self):
        with self.lock:
            self.misses += 1

    def get_stats(self):
        with self.lock:
            return {
                "policy_type": "LFU",
                "tracked_items": len(self.access_counts),
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": (
                    self.hits / (self.hits + self.misses)
                    if (self.hits + self.misses) > 0
                    else 0
                ),
            }
