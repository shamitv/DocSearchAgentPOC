import concurrent.futures
import threading
import queue

class BoundedProcessPoolExecutor(concurrent.futures.ProcessPoolExecutor):
    """
    A ProcessPoolExecutor with a bounded queue for backpressure.
    The submitter is blocked if the queue is full until a slot becomes free.
    """
    def __init__(self, max_workers=None, max_queue_size=0, **kwargs):
        super().__init__(max_workers=max_workers, **kwargs)
        # If max_queue_size is 0, default to max_workers * 2
        self._queue_size = max_queue_size or (max_workers or 1) * 2
        self._semaphore = threading.Semaphore(self._queue_size)

    def submit(self, *args, **kwargs):
        self._semaphore.acquire()
        future = super().submit(*args, **kwargs)
        future.add_done_callback(lambda f: self._semaphore.release())
        return future
