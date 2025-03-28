import numpy as np


class DirectMemory:
    """Direct memory access with proper buffer protocol support"""

    def __init__(self, size_bytes):
        self.size = size_bytes
        # Use a bytearray as the backing store
        self._buffer = bytearray(size_bytes)
        # Create view for faster operations
        self._view = memoryview(self._buffer)

    def __getitem__(self, key):
        """Get data from buffer - supports slice and integer indexing"""
        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else self.size
            if start < 0 or stop > self.size:
                raise IndexError(
                    f"Index out of bounds: [{start}:{stop}] for buffer size {self.size}"
                )
            return self._view[start:stop]
        elif isinstance(key, int):
            if key < 0 or key >= self.size:
                raise IndexError(
                    f"Index {key} out of bounds for buffer size {self.size}"
                )
            return self._view[key : key + 1]
        else:
            raise TypeError(f"Invalid index type: {type(key)}")

    def __setitem__(self, key, value):
        """Set data in buffer - supports slice and integer indexing"""
        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = (
                key.stop if key.stop is not None else min(start + len(value), self.size)
            )
            if start < 0 or stop > self.size:
                raise IndexError(
                    f"Index out of bounds: [{start}:{stop}] for buffer size {self.size}"
                )

            # Handle different value types
            if isinstance(value, (bytes, bytearray, memoryview)):
                self._view[start:stop] = value
            elif isinstance(value, np.ndarray):
                # Ensure the array is contiguous and has the right type
                if not value.flags.c_contiguous:
                    value = np.ascontiguousarray(value)
                # Copy the bytes
                self._view[start:stop] = value.tobytes()
            else:
                # Try to convert to bytes
                self._view[start:stop] = bytes(value)
        elif isinstance(key, int):
            if key < 0 or key >= self.size:
                raise IndexError(
                    f"Index {key} out of bounds for buffer size {self.size}"
                )
            if isinstance(value, int):
                self._buffer[key] = value
            else:
                self._view[key : key + 1] = value
        else:
            raise TypeError(f"Invalid index type: {type(key)}")

    def clear(self, start=0, size=None):
        """Clear a region of memory (set to zeros)"""
        if size is None:
            size = self.size - start
        self._view[start : start + size] = b"\0" * size
