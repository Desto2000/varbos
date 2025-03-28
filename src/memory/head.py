class HeadPolicy:
    """Interface for cache management policies"""

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


class SimpleHeadPolicy(HeadPolicy):
    pass
