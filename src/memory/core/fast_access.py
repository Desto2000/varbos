import numpy as np


class FastLookupTable:
    """
    Optimized lookup table with O(1) access patterns and proper container methods
    """

    def __init__(self, initial_capacity=1024):
        self.keys = {}  # Dictionary mapping keys to indices
        self.values_array = np.zeros(
            (initial_capacity, 2), dtype=np.int64
        )  # Array of (start, end) tuples
        self.size = 0
        self.capacity = initial_capacity

    def __contains__(self, key):
        """Check if key exists in the lookup table"""
        return key in self.keys

    def __getitem__(self, key):
        """Get the (start, end) tuple for a key"""
        if key not in self.keys:
            raise KeyError(f"Key {key} not found in lookup table")
        idx = self.keys[key]
        return tuple(self.values_array[idx])

    def __setitem__(self, key, value):
        """Set or update a (start, end) tuple for a key"""
        # Ensure value is a 2-element tuple/list
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError("Value must be a (start, end) tuple/list with 2 elements")

        # Insert or update
        if key not in self.keys:
            if self.size >= self.capacity:
                self._resize()
            self.keys[key] = self.size
            self.values_array[self.size] = value
            self.size += 1
        else:
            idx = self.keys[key]
            self.values_array[idx] = value

    def __delitem__(self, key):
        """Delete a key from the lookup table"""
        if key not in self.keys:
            raise KeyError(f"Key {key} not found in lookup table")

        # Get index of item to remove
        idx_to_remove = self.keys[key]

        # Remove from keys dictionary
        del self.keys[key]

        # If we're removing the last item, just decrement size
        if idx_to_remove == self.size - 1:
            self.size -= 1
            return

        # Otherwise, move the last item to fill the gap
        last_idx = self.size - 1

        # Find which key points to the last index
        last_key = None
        for k, idx in self.keys.items():
            if idx == last_idx:
                last_key = k
                break

        if last_key is not None:
            # Move the last item to the removed position
            self.values_array[idx_to_remove] = self.values_array[last_idx]
            # Update the key's index
            self.keys[last_key] = idx_to_remove

        # Decrement size
        self.size -= 1

    def __len__(self):
        """Return number of items in the lookup table"""
        return self.size

    def _resize(self):
        """Resize the values array when full"""
        new_capacity = self.capacity * 2
        new_array = np.zeros((new_capacity, 2), dtype=np.int64)
        new_array[: self.size] = self.values_array[: self.size]
        self.values_array = new_array
        self.capacity = new_capacity

    def items(self):
        """Return an iterator over (key, (start, end)) pairs"""
        for key, idx in self.keys.items():
            yield key, tuple(self.values_array[idx])

    def values(self):
        """Return an iterator over (start, end) tuples"""
        for idx in range(self.size):
            for key, i in self.keys.items():
                if i == idx:
                    yield tuple(self.values_array[idx])
                    break

    def keys(self):
        """Return an iterator over keys"""
        return self.keys.keys()

    def clear(self):
        """Clear all entries"""
        self.keys.clear()
        self.size = 0
