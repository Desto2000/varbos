from typing import Optional

import numpy as np

from src.memory.core.ops import zero_memory


class DirectMemory:
    """Direct memory access with proper buffer protocol support"""

    def __init__(self, size_bytes):
        self.size: int = size_bytes
        # Use a bytearray as the backing store
        self._buffer: bytearray = bytearray(size_bytes)
        # Create view for faster operations
        self._view: memoryview = memoryview(self._buffer)

    def __buffer__(self):
        """Support for Python's buffer protocol"""
        return self._buffer

    def __len__(self) -> int:
        """Return the size of the memory buffer in bytes."""
        return self.size

    def __getitem__(self, key):
        """Get data from buffer - supports slice and integer indexing"""
        return self._view[key]

    def __setitem__(self, key, value):
        """Set data in buffer - supports slice and integer indexing"""
        start = key.start
        stop = key.stop
        # Handle different value types
        if isinstance(value, memoryview):
            self._view[start:stop] = value.cast("B")
        elif isinstance(value, np.ndarray):
            # Use zero-copy approach where possible
            if value.nbytes <= (stop - start):
                # Direct copy when data fits
                data_view = memoryview(value).cast("B")
                self._view[start : start + len(data_view)] = data_view
            else:
                # Fallback to standard copy
                self._view[start:stop] = value.tobytes()[: stop - start]

    def get_numpy_view(self, start=0, size=None, dtype=np.uint8):
        """Get zero-copy numpy view of memory region"""
        if size is None:
            size = self.size - start

        # Ensure we can create a view with the given dtype
        dtype_size = np.dtype(dtype).itemsize
        if size % dtype_size != 0:
            # Adjust size to be a multiple of the dtype size
            size = (size // dtype_size) * dtype_size

        if start < 0 or start + size > self.size:
            raise IndexError(f"Region [{start}:{start+size}] is out of bounds")

        # Create a view with the requested dtype
        return np.frombuffer(self._buffer[start : start + size], dtype=dtype)

    def clear(self, start=0, size=None):
        """Clear memory region - optimized for different sizes"""
        if size is None:
            size = self.size - start

        if size <= 0:
            return

        zero_memory(self._buffer, start, size)
