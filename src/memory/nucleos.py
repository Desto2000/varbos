import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor


class NucleosManager:
    """Highly optimized thread pool manager for memory operations"""

    def __init__(self, memory_instance, max_workers=None):
        self.memory = memory_instance
        # Use ThreadPoolExecutor for work distribution
        self.thread_pool = ThreadPoolExecutor(
            max_workers=max_workers or (threading.active_count() * 2)
        )
        self.maintenance_queue = queue.PriorityQueue()
        self.task_futures = {}
        self.shutdown_flag = False
        self.tasks_processed = 0
        self.maintenance_runs = 0
        self.lock = threading.RLock()  # Add lock for task_futures dictionary

        # Start maintenance thread
        self.maintenance_thread = threading.Thread(
            target=self._maintenance_worker, daemon=True
        )
        self.maintenance_thread.start()

        # Use thread local storage for per-thread caching
        self.thread_local = threading.local()

    def schedule_task(self, task_type, args=None, priority=5, timeout=None):
        """Schedule task with priority (lower = higher priority)"""
        if self.shutdown_flag:
            return None

        # Create cancellable future
        future = self.thread_pool.submit(self._execute_task, task_type, args)
        task_id = id(future)

        # Thread-safe update of task_futures
        with self.lock:
            self.task_futures[task_id] = future

        # Return future for caller to wait on if needed
        return future

    def _execute_task(self, task_type, args):
        """Execute a task with thread local context"""
        # Initialize thread local storage if needed
        if not hasattr(self.thread_local, "context"):
            self.thread_local.context = {"last_task": None, "stats": {}}

        start_time = time.time()
        result = None

        try:
            # Task routing with optimized paths
            if task_type == "free":
                start, size = args
                result = self.memory._deallocate_internal(start, size)
            elif task_type == "rebuild":
                result = self.memory._rebuild_internal()
            elif task_type == "evict":
                count = args if args is not None else 1
                result = self.memory._evict_internal(count)
            # Add more task types as needed

            # Track execution time for this task type
            exec_time = time.time() - start_time
            if task_type not in self.thread_local.context["stats"]:
                self.thread_local.context["stats"][task_type] = []
            stats = self.thread_local.context["stats"][task_type]
            stats.append(exec_time)

            # Keep only last 100 measurements per type
            if len(stats) > 100:
                stats.pop(0)

            with self.lock:
                self.tasks_processed += 1
            return result

        except Exception as e:
            print(f"Task execution error ({task_type}): {e}")
            return None  # Don't re-raise to prevent thread termination

    def _maintenance_worker(self):
        """Smart background maintenance"""
        last_gc_time = time.time()

        while not self.shutdown_flag:
            try:
                # Balance between responsiveness and CPU usage
                time.sleep(0.01)

                # Clean up completed futures periodically
                now = time.time()
                if now - last_gc_time > 5.0:  # Every 5 seconds
                    with self.lock:
                        # Make a copy to avoid modification during iteration
                        futures_items = list(self.task_futures.items())

                    completed = [k for k, v in futures_items if v.done()]

                    if completed:
                        with self.lock:
                            for task_id in completed:
                                if task_id in self.task_futures:
                                    del self.task_futures[task_id]
                    last_gc_time = now

                # Perform other maintenance work
                self._check_system_health()

            except Exception as e:
                print(f"Maintenance worker error: {e}")
                time.sleep(1)  # Sleep longer on error to avoid rapid cycling

    def _check_system_health(self):
        """Proactive system health monitoring"""
        try:
            # Check memory pressure
            stats = self.memory.get_stats()
            memory_stats = stats.get("memory", {})

            # Adaptive maintenance based on system state
            free_ratio = memory_stats.get("free_bytes", 0) / max(
                memory_stats.get("total_bytes", 1), 1
            )

            # Prioritize based on current conditions
            if free_ratio < 0.1:  # Critical memory pressure
                self.schedule_task("evict", 10, priority=1)
            elif free_ratio < 0.2:  # High memory pressure
                self.schedule_task("evict", 5, priority=2)

            # Check fragmentation ratio and schedule accordingly
            frag_ratio = memory_stats.get("fragmentation_ratio", 0)
            if frag_ratio > 0.5:  # Severe fragmentation
                self.schedule_task("rebuild", priority=2)
            elif frag_ratio > 0.3:  # Moderate fragmentation
                self.schedule_task("rebuild", priority=3)

            with self.lock:
                self.maintenance_runs += 1
        except Exception as e:
            print(f"Health check error: {e}")

    def shutdown(self):
        """Gracefully shut down thread manager"""
        self.shutdown_flag = True

        with self.lock:
            # Cancel all pending tasks
            for future in self.task_futures.values():
                future.cancel()

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=False)

    def get_stats(self):
        """Get thread manager statistics"""
        with self.lock:
            return {
                "tasks_processed": self.tasks_processed,
                "maintenance_runs": self.maintenance_runs,
                "queued_tasks": len(self.task_futures),
            }
