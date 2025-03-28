import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from src.memory.eviction import LRUEvictionPolicy
from src.memory.map import Memory
from src.memory.placement import BestFitPlacementPolicy
from src.memory.sync import SimpleLockPolicy


class MemoryMapVisualizer:
    """Visualizes the actual memory layout showing allocations and fragmentation"""

    def __init__(self, memory_system):
        """Initialize with a memory system instance"""
        self.memory = memory_system
        self.history = []  # Store memory snapshots
        self.timestamps = []  # Timestamps for snapshots

        # Colors for visualization
        self.allocated_color = "#ff7f0e"  # Orange for allocated blocks
        self.free_color = "#1f77b4"  # Blue for free blocks

    def capture_memory_map(self):
        """Capture the current state of memory blocks"""
        self.memory.lock_policy.acquire_read(None)
        try:
            # Get all allocated blocks from lookup table
            allocated_blocks = []
            for key, (start, end) in self.memory.lookup_table.items():
                allocated_blocks.append(
                    (start, end - start, True, key)
                )  # start, size, is_allocated, key

            # Get placement policy's free blocks
            # Assuming we add a method to access the free blocks list
            free_blocks = []
            for size, start in self.memory.placement_policy.free_blocks:
                free_blocks.append(
                    (start, size, False, None)
                )  # start, size, is_allocated, key

            # Combine and sort by start address
            all_blocks = allocated_blocks + free_blocks
            all_blocks.sort(key=lambda x: x[0])

            # Record this state
            self.history.append(all_blocks.copy())
            self.timestamps.append(time.time())

            return all_blocks

        finally:
            self.memory.lock_policy.release_read(None)

    def visualize_current(self, title=None):
        """Visualize the current memory map"""
        blocks = self.capture_memory_map()
        fig, ax = self._create_memory_map_plot(blocks, title)
        return fig

    def visualize_history(self, max_plots=12, output_file=None):
        """Visualize memory map history as a series of plots"""
        n = len(self.history)
        if n == 0:
            return None

        # Determine grid size (roughly square)
        plots_to_show = min(n, max_plots)
        indices = list(range(0, n, max(1, n // plots_to_show)))
        if indices[-1] != n - 1:
            indices.append(n - 1)  # Always include the last state

        cols = int(np.ceil(np.sqrt(len(indices))))
        rows = int(np.ceil(len(indices) / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))
        if rows * cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])
        elif cols == 1:
            axes = np.array([[ax] for ax in axes])

        # Plot each selected memory state
        for i, idx in enumerate(indices):
            if i >= rows * cols:
                break

            r, c = i // cols, i % cols
            ax = axes[r, c]

            # Plot memory map for this state
            self._create_memory_map_plot(self.history[idx], ax=ax)

            # Add timestamp
            timestamp = self.timestamps[idx]
            elapsed = timestamp - self.timestamps[0]
            ax.set_title(f"State {idx} (+{elapsed:.2f}s)", fontsize=10)

        # Hide any unused subplots
        for i in range(len(indices), rows * cols):
            r, c = i // cols, i % cols
            axes[r, c].axis("off")

        # Add legend to the figure
        handles = [
            patches.Patch(color=self.allocated_color, label="Allocated"),
            patches.Patch(color=self.free_color, label="Free"),
        ]
        fig.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.suptitle("Memory Map Visualization - Fragmentation Over Time", fontsize=14)

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            print(f"Saved memory map visualization to {output_file}")

        return fig

    def _create_memory_map_plot(self, blocks, title=None, ax=None):
        """Create a visualization of memory blocks"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        else:
            fig = ax.figure

        memory_size = self.memory.memory_size

        # Create patches for visualization
        patches_list = []
        allocated_count = 0
        free_count = 0

        for start, size, is_allocated, key in blocks:
            # Create rectangle patch
            rect = patches.Rectangle(
                (start, 0),
                size,
                1,
                linewidth=1,
                edgecolor="black",
                facecolor=self.allocated_color if is_allocated else self.free_color,
                alpha=0.7,
            )
            patches_list.append(rect)

            # Count block types
            if is_allocated:
                allocated_count += 1
            else:
                free_count += 1

            # Add text label for large enough blocks
            if (
                size > memory_size * 0.05
            ):  # Only label blocks that are big enough to see
                text_x = start + size / 2
                if is_allocated:
                    label = f"{size/1024:.1f}K"
                    ax.text(text_x, 0.5, label, ha="center", va="center", fontsize=8)
                else:
                    label = f"Free\n{size/1024:.1f}K"
                    ax.text(text_x, 0.5, label, ha="center", va="center", fontsize=8)

        # Add patches to the plot
        for p in patches_list:
            ax.add_patch(p)

        # Set plot limits and labels
        ax.set_xlim(0, memory_size)
        ax.set_ylim(0, 1)
        ax.set_xticks(
            [0, memory_size / 4, memory_size / 2, 3 * memory_size / 4, memory_size]
        )
        ax.set_xticklabels(
            [
                "0",
                f"{memory_size/4/1024:.0f}K",
                f"{memory_size/2/1024:.0f}K",
                f"{3*memory_size/4/1024:.0f}K",
                f"{memory_size/1024:.0f}K",
            ]
        )
        ax.set_yticks([])

        # Add information about fragmentation
        frag_ratio = self.memory.get_stats()["memory"]["fragmentation_ratio"]
        allocation_info = (
            f"{allocated_count} allocated blocks, {free_count} free blocks"
        )
        frag_info = f"Fragmentation: {frag_ratio:.2f}"

        if title:
            ax.set_title(f"{title}\n{allocation_info}, {frag_info}")
        else:
            ax.set_title(f"{allocation_info}, {frag_info}")

        return fig, ax


def memory_fragmentation_test(memory, visualizer):
    """Run a test specifically designed to show memory fragmentation patterns"""
    print("Starting memory fragmentation test...")

    # Phase 1: Create a base set of allocations
    print("Phase 1: Initial allocations")
    block_sizes = [5000, 10000, 15000, 20000, 25000, 30000, 8000, 12000]
    keys = []

    for i, size in enumerate(block_sizes):
        key = f"base_{i}"
        memory[key] = b"x" * size
        keys.append(key)

    visualizer.capture_memory_map()

    # Phase 2: Delete some blocks to create fragmentation
    print("Phase 2: Creating fragmentation by deleting alternate blocks")
    for i in range(0, len(keys), 2):
        if i < len(keys):
            del memory[keys[i]]

    visualizer.capture_memory_map()

    # Phase 3: Allocate blocks that should cause fragmentation
    print("Phase 3: Allocating blocks that fit in fragments")
    new_sizes = [4000, 7000, 9000, 3000, 6000]
    for i, size in enumerate(new_sizes):
        key = f"frag_{i}"
        memory[key] = b"x" * size

    visualizer.capture_memory_map()

    # Phase 4: Create a challenging pattern
    print("Phase 4: Creating more complex fragmentation pattern")
    # Delete every third remaining original block
    for i in range(1, len(keys), 3):
        if i < len(keys):
            try:
                del memory[keys[i]]
            except KeyError:
                pass  # Already deleted

    visualizer.capture_memory_map()

    # Add some medium blocks
    medium_sizes = [15000, 18000, 7500, 12500]
    for i, size in enumerate(medium_sizes):
        key = f"medium_{i}"
        try:
            memory[key] = b"x" * size
        except MemoryError:
            print(f"Couldn't allocate block of size {size}")

    visualizer.capture_memory_map()

    # Phase 5: Force a rebuild by adding a block too large for any fragment
    print("Phase 5: Triggering memory rebuild")
    # First check fragmentation level
    stats = memory.get_stats()
    print(
        f"Fragmentation ratio before rebuild: {stats['memory']['fragmentation_ratio']:.2f}"
    )

    # Try to allocate a block that likely won't fit in fragmented memory
    # but should fit after defrag
    try:
        free_bytes = stats["memory"]["free_bytes"]
        # Try to allocate 80% of total free space
        allocation_size = int(free_bytes * 0.8)
        if allocation_size > 0:
            print(f"Attempting to allocate {allocation_size/1024:.1f}K block")
            memory["large_block"] = b"x" * allocation_size
            print("Large allocation succeeded")
        else:
            print("Not enough free space for large allocation")
    except MemoryError:
        print("Couldn't allocate large block - memory too fragmented")

        # Force a rebuild
        memory.thread_manager.schedule_task("rebuild", urgent=True)
        print("Forced memory rebuild")
        time.sleep(0.1)  # Give time for rebuild to complete

    visualizer.capture_memory_map()

    # Phase 6: After rebuild, check if we can allocate the large block
    stats = memory.get_stats()
    print(
        f"Fragmentation ratio after rebuild: {stats['memory']['fragmentation_ratio']:.2f}"
    )

    free_bytes = stats["memory"]["free_bytes"]
    if free_bytes > 10000:  # Only try if we have enough space
        try:
            # Try to allocate most of the free space
            allocation_size = int(free_bytes * 0.7)
            memory["post_rebuild_block"] = b"x" * allocation_size
            print(f"Successfully allocated {allocation_size/1024:.1f}K after rebuild")
        except MemoryError:
            print("Still couldn't allocate large block after rebuild")

    visualizer.capture_memory_map()

    print("Fragmentation test completed")


def run_fragmentation_test():
    """Run a test focused on memory fragmentation visualization"""
    print("=== Memory Fragmentation Visualization Test ===")
    print("Date: 2025-03-22 17:38:01")
    print("User: Desto2000")
    print("=" * 50)

    # Create a memory system that will show fragmentation clearly
    memory_size = 512 * 1024  # 512KB - small enough to show fragmentation clearly
    memory = Memory(
        memory_size=memory_size,
        eviction_policy=LRUEvictionPolicy(),
        lock_policy=SimpleLockPolicy(),
        # Set a high fragmentation threshold so it doesn't rebuild automatically
        placement_policy=BestFitPlacementPolicy(fragmentation_threshold=0.7),
    )

    # Create visualizer
    visualizer = MemoryMapVisualizer(memory)

    # Capture initial state
    print("Capturing initial memory state...")
    visualizer.capture_memory_map()

    # Run fragmentation test
    memory_fragmentation_test(memory, visualizer)

    # Generate visualization
    print("Generating memory map visualization...")
    visualizer.visualize_history(output_file="memory_fragmentation_map.png")

    # Print final stats
    stats = memory.get_stats()
    print("\nFinal Memory Statistics:")
    mem_stats = stats["memory"]
    print(f"- Total Memory: {memory_size/1024:.1f}KB")
    print(
        f"- Used: {mem_stats['used_bytes']/1024:.1f}KB ({mem_stats['percent_used']:.1f}%)"
    )
    print(f"- Free blocks: {mem_stats['free_blocks']}")
    print(f"- Fragmentation: {mem_stats['fragmentation_ratio']:.2f}")
    print(f"- Rebuilds: {mem_stats['rebuilds']}")

    # Clean shutdown
    memory.shutdown()

    print(
        "\nTest completed. Memory map visualization saved to memory_fragmentation_map.png"
    )


if __name__ == "__main__":
    # Set up matplotlib for non-GUI environments if needed
    import matplotlib
    import os

    if os.environ.get("DISPLAY") is None:
        matplotlib.use("Agg")  # Use non-interactive backend

    run_fragmentation_test()
