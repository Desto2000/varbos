import numpy as np


class FastLookupTable:
    """
    Optimized lookup table with O(1) access patterns and proper container methods
    """

    def __init__(self, initial_capacity=1024):
        self._keys = {}  # Dictionary mapping keys to indices
        self._values_array = np.zeros(
            (initial_capacity, 2), dtype=np.int64
        )  # Array of (start, end) tuples
        self._size = 0
        self._capacity = initial_capacity
        # Numba acceleration is disabled by default
        self._numba_enabled = False
        self._numba_dict = None

    def __contains__(self, key):
        """Check if key exists in the lookup table"""
        # We'll use the standard Python dictionary for lookups
        # as it's already very fast and avoids type issues
        return key in self._keys

    def __getitem__(self, key):
        """Get the (start, end) tuple for a key"""
        if key not in self._keys:
            raise KeyError(f"Key {key} not found in lookup table")
        idx = self._keys[key]
        return tuple(self._values_array[idx])

    def __setitem__(self, key, value):
        """Set or update a (start, end) tuple for a key"""
        # Ensure value is a 2-element tuple/list
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("Value must be a (start, end) tuple/list with 2 elements")

        # Insert or update
        if key not in self._keys:
            if self._size >= self._capacity:
                self._resize()
            self._keys[key] = self._size
            self._values_array[self._size] = value
            self._size += 1
        else:
            idx = self._keys[key]
            self._values_array[idx] = value

    def __delitem__(self, key):
        """Delete a key from the lookup table"""
        if key not in self._keys:
            raise KeyError(f"Key {key} not found in lookup table")

        # Get index of item to remove
        idx_to_remove = self._keys[key]

        # Remove from keys dictionary
        del self._keys[key]

        # If we're removing the last item, just decrement size
        if idx_to_remove == self._size - 1:
            self._size -= 1
            return

        # Otherwise, move the last item to fill the gap
        last_idx = self._size - 1

        # Find which key points to the last index
        last_key = None
        for k, idx in self._keys.items():
            if idx == last_idx:
                last_key = k
                break

        if last_key is not None:
            # Move the last item to the removed position
            self._values_array[idx_to_remove] = self._values_array[last_idx]
            # Update the key's index
            self._keys[last_key] = idx_to_remove

        # Decrement size
        self._size -= 1

    def __len__(self):
        """Return number of items in the lookup table"""
        return self._size

    def _resize(self):
        """Resize the values array when full"""
        new_capacity = self._capacity * 2
        new_array = np.zeros((new_capacity, 2), dtype=np.int64)
        new_array[: self._size] = self._values_array[: self._size]
        self._values_array = new_array
        self._capacity = new_capacity

    def items(self):
        """Return an iterator over (key, (start, end)) pairs"""
        for key, idx in self._keys.items():
            yield key, tuple(self._values_array[idx])

    def values(self):
        """Return an iterator over (start, end) tuples"""
        sorted_items = sorted([(i, k) for k, i in self._keys.items()])
        for idx, key in sorted_items:
            yield tuple(self._values_array[idx])

    def keys(self):
        """Return an iterator over keys"""
        return self._keys.keys()

    def clear(self):
        """Clear all entries"""
        self._keys.clear()
        self._size = 0
