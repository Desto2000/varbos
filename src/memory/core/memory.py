import numpy as np


class DirectMemory:
    """Direct memory access with proper buffer protocol support"""

    def __init__(self, size_bytes):
        self.size = size_bytes
        # Use a bytearray as the backing store
        self._buffer = bytearray(size_bytes)
        # Create view for faster operations
        self._view = memoryview(self._buffer)
        # Cache numpy view for faster numpy operations
        self._np_view = None

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
            return self._view[key]
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
                # Use zero-copy approach where possible
                if value.nbytes <= (stop - start):
                    # Direct copy when data fits
                    data_view = memoryview(value).cast("B")
                    self._view[start : start + len(data_view)] = data_view
                else:
                    # Fallback to standard copy
                    self._view[start:stop] = value.tobytes()[: stop - start]
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

        if size <= 0:
            return

        # Optimize small regions by direct assignment
        if size <= 1024:
            self._view[start : start + size] = b"\0" * size
        else:
            # For large regions, use numpy's fast zero filling
            if self._np_view is None:
                self._np_view = np.frombuffer(self._buffer, dtype=np.uint8)
            self._np_view[start : start + size] = 0

    def get_numpy_view(self, start=0, size=None, dtype=np.uint8):
        """Get a numpy view of the memory region with specified dtype"""
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
