import ctypes
import mmap

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def optimized_copy(src, dest, length):
    """Optimized memory copy with cache-friendly access patterns"""
    # Process in chunks that are cache-line aligned (typically 64 bytes)
    chunk_size = 64

    # For small copies, do it directly
    if length <= chunk_size * 2:
        for i in range(length):
            dest[i] = src[i]
        return

    # For larger copies, use chunking for better cache utilization
    num_chunks = (length + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, length)

        # Prefetch next chunk data
        if i + 1 < num_chunks:
            next_start = (i + 1) * chunk_size
            # This acts as a prefetch hint for Numba
            _ = src[min(next_start, length - 1)]

        # Copy current chunk
        for j in range(start, end):
            dest[j] = src[j]


class DirectMemory:
    """High-performance direct memory access with zero-copy capabilities"""

    def __init__(self, size_bytes, use_mmap=False, mmap_path=None, huge_pages=False):
        self.size = size_bytes
        self.use_mmap = use_mmap
        self.huge_pages = huge_pages

        if use_mmap:
            # Memory-mapped file for larger-than-RAM data
            self.mmap_path = mmap_path or "memory_buffer.bin"
            self.mmap_file = open(self.mmap_path, "w+b")
            self.mmap_file.truncate(size_bytes)

            # Use huge pages if available (Linux only)
            flags = mmap.MAP_SHARED
            if huge_pages and hasattr(mmap, "MAP_HUGETLB"):
                flags |= mmap.MAP_HUGETLB

            self._buffer = mmap.mmap(
                self.mmap_file.fileno(),
                size_bytes,
                flags=flags,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        else:
            # Try to allocate aligned memory for better performance
            try:
                # First try using NumPy's aligned memory allocation
                self._buffer = np.zeros(size_bytes, dtype=np.uint8)
            except MemoryError:
                # Fall back to standard allocation
                self._buffer = bytearray(size_bytes)

        # Create multiple views for different data types
        self._view = memoryview(self._buffer)

        # Create typed views for zero-copy operations
        self._uint8_view = np.frombuffer(self._buffer, dtype=np.uint8)
        self._uint32_view = np.frombuffer(self._buffer, dtype=np.uint32)
        self._int64_view = np.frombuffer(self._buffer, dtype=np.int64)
        self._float32_view = np.frombuffer(self._buffer, dtype=np.float32)
        self._float64_view = np.frombuffer(self._buffer, dtype=np.float64)

        # Use ctypes for aligned memory access
        self._c_buffer = (ctypes.c_ubyte * size_bytes).from_buffer(self._buffer)

        # Performance tracking
        self.reads = 0
        self.writes = 0
        self.large_transfers = 0

    def __del__(self):
        """Clean up resources"""
        if self.use_mmap and hasattr(self, "_buffer"):
            self._buffer.close()
            if hasattr(self, "mmap_file"):
                self.mmap_file.close()

    def get_typed_view(self, start, size, dtype=np.uint8):
        """Get a typed view with proper alignment checks"""
        # Ensure the access is properly aligned
        if dtype == np.int32 or dtype == np.uint32 or dtype == np.float32:
            alignment = 4
        elif dtype == np.int64 or dtype == np.uint64 or dtype == np.float64:
            alignment = 8
        else:
            alignment = 1

        if start % alignment != 0:
            raise ValueError(
                f"Misaligned memory access: address {start} is not {alignment}-byte aligned"
            )

        if dtype == np.uint8:
            return self._uint8_view[start : start + size]
        elif dtype == np.uint32:
            return self._uint32_view[start // 4 : (start + size) // 4]
        elif dtype == np.int64:
            return self._int64_view[start // 8 : (start + size) // 8]
        elif dtype == np.float32:
            return self._float32_view[start // 4 : (start + size) // 4]
        elif dtype == np.float64:
            return self._float64_view[start // 8 : (start + size) // 8]
        else:
            # Use memoryview for other types
            return self._view[start : start + size]

    def __getitem__(self, key):
        """Get data with optimized paths for different access patterns"""
        self.reads += 1

        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else self.size

            if start < 0 or stop > self.size:
                raise IndexError(
                    f"Memory access out of bounds: [{start}:{stop}] for buffer size {self.size}"
                )

            size = stop - start

            # Optimize for size - use different strategies
            if size >= 1024 * 1024:  # Large block: use numpy's optimized methods
                self.large_transfers += 1
                return self._uint8_view[start:stop].copy()
            elif size >= 1024:  # Medium block: use memoryview
                return self._view[start:stop]
            else:  # Small block: direct access
                return bytes(self._view[start:stop])

        elif isinstance(key, int):
            if key < 0 or key >= self.size:
                raise IndexError(
                    f"Index {key} out of bounds for buffer size {self.size}"
                )
            return self._buffer[key]

        else:
            raise TypeError(f"Invalid index type: {type(key)}")

    def __setitem__(self, key, value):
        """Set data with optimized paths for different types"""
        self.writes += 1

        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = (
                key.stop if key.stop is not None else min(start + len(value), self.size)
            )

            if start < 0 or stop > self.size:
                raise IndexError(
                    f"Memory access out of bounds: [{start}:{stop}] for buffer size {self.size}"
                )

            size = stop - start

            # Optimize for different value types and sizes
            if isinstance(value, np.ndarray):
                # Fast path for numpy arrays
                if size >= 1024 * 1024:  # Very large transfers
                    self.large_transfers += 1
                    # For large arrays, optimize the copy
                    np.copyto(self._uint8_view[start:stop], value[:size].view(np.uint8))
                else:
                    # Direct copy for smaller arrays
                    if value.dtype == np.uint8:
                        np.copyto(self._uint8_view[start:stop], value[:size])
                    elif value.dtype == np.int32 and start % 4 == 0 and size % 4 == 0:
                        np.copyto(
                            self._uint32_view[start // 4 : stop // 4],
                            value[: size // 4],
                        )
                    elif value.dtype == np.int64 and start % 8 == 0 and size % 8 == 0:
                        np.copyto(
                            self._int64_view[start // 8 : stop // 8], value[: size // 8]
                        )
                    else:
                        # Fall back to byte copy
                        np.copyto(
                            self._uint8_view[start:stop], value.view(np.uint8)[:size]
                        )

            elif isinstance(value, (bytes, bytearray, memoryview)):
                if size >= 1024 * 1024:  # Large block
                    self.large_transfers += 1
                    # Use optimized copy for large transfers
                    src_view = np.frombuffer(value[:size], dtype=np.uint8)
                    optimized_copy(
                        src_view,
                        self._uint8_view[start:stop],
                        min(len(src_view), stop - start),
                    )
                else:
                    # Direct copy for smaller blocks
                    self._view[start:stop] = value[:size]

            else:
                # Try to convert to bytes
                try:
                    self._view[start:stop] = bytes(value)[:size]
                except TypeError:
                    raise TypeError(
                        f"Cannot convert {type(value)} to bytes for memory storage"
                    )

        elif isinstance(key, int):
            if key < 0 or key >= self.size:
                raise IndexError(
                    f"Index {key} out of bounds for buffer size {self.size}"
                )

            if isinstance(value, int):
                self._buffer[key] = value
            else:
                self._view[key : key + 1] = bytes([value])

        else:
            raise TypeError(f"Invalid index type: {type(key)}")

    def clear(self, start=0, size=None):
        """Efficiently clear memory region"""
        if size is None:
            size = self.size - start

        if start < 0 or start + size > self.size:
            raise IndexError(f"Memory access out of bounds for clear operation")

        if size >= 1024 * 1024:  # Large block
            # Use numpy's fast zero-fill
            self._uint8_view[start : start + size] = 0
        elif size >= 1024:  # Medium block
            # Use memset through ctypes for better performance
            ctypes.memset(ctypes.addressof(self._c_buffer) + start, 0, size)
        else:  # Small block
            # Direct memory view is fast enough for small blocks
            self._view[start : start + size] = b"\0" * size

    def copy_from(self, source, dest_offset=0, source_offset=0, size=None):
        """Zero-copy data transfer from another memory buffer"""
        if size is None:
            size = min(len(source) - source_offset, self.size - dest_offset)

        if dest_offset < 0 or dest_offset + size > self.size:
            raise IndexError(
                f"Destination range [{dest_offset}:{dest_offset+size}] out of bounds"
            )

        if source_offset < 0 or source_offset + size > len(source):
            raise IndexError(
                f"Source range [{source_offset}:{source_offset+size}] out of bounds"
            )

        # Handle different source types optimally
        if hasattr(source, "_uint8_view") and hasattr(source, "_view"):
            # Source is another EnhancedDirectMemory - use numpy views for speed
            np.copyto(
                self._uint8_view[dest_offset : dest_offset + size],
                source._uint8_view[source_offset : source_offset + size],
            )
        elif isinstance(source, np.ndarray):
            # Source is a numpy array
            np.copyto(
                self._uint8_view[dest_offset : dest_offset + size],
                source[source_offset : source_offset + size].view(np.uint8),
            )
        elif isinstance(source, (bytes, bytearray, memoryview)):
            # Source is a bytes-like object
            self._view[dest_offset : dest_offset + size] = source[
                source_offset : source_offset + size
            ]
        else:
            # Fall back to standard copy
            self._view[dest_offset : dest_offset + size] = bytes(
                source[source_offset : source_offset + size]
            )

        self.writes += 1
        if size >= 1024 * 1024:
            self.large_transfers += 1

    def get_stats(self):
        """Get memory access statistics"""
        return {
            "size": self.size,
            "type": "mmap" if self.use_mmap else "ram",
            "reads": self.reads,
            "writes": self.writes,
            "large_transfers": self.large_transfers,
        }
