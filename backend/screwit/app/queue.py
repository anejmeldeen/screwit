"""Job queue wiring (SPEC 3, 7 P4).

Two backends behind one interface:
- `inline` runs the job in-process (mock mode and tests; nothing to install).
- `rq` enqueues on Redis. There are two named queues: a GPU queue (reconstruction, run by an
  rq SimpleWorker with concurrency 1 so CUDA is never forked) and a CPU queue (validation, CAD).

The GPU worker is started with `python -m screwit.app.worker`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

GPU_QUEUE = "gpu"
CPU_QUEUE = "cpu"


class Queue(Protocol):
    def enqueue(self, queue_name: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        """Run or schedule func(*args, **kwargs); returns a job id."""


class InlineQueue:
    """Runs the job synchronously. Used in mock mode and tests."""

    def __init__(self) -> None:
        self._n = 0

    def enqueue(self, queue_name: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        func(*args, **kwargs)
        self._n += 1
        return f"inline-{queue_name}-{self._n}"


class ThreadQueue:
    """One background worker thread per queue name, concurrency 1 (dev live mode without Redis).

    Jobs run in the API process, so a long reconstruction never blocks the request that started it,
    and the GPU queue still runs one job at a time as SPEC 3 requires.
    """

    def __init__(self) -> None:
        import queue
        import threading

        self._queues: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        self._n = 0
        self._threading = threading
        self._queue_mod = queue

    def _worker(self, q) -> None:
        import logging

        while True:
            func, args, kwargs = q.get()
            try:
                func(*args, **kwargs)
            except Exception:  # noqa: BLE001 - a failed job must not kill the worker
                logging.getLogger("screwit.queue").exception("job failed")
            finally:
                q.task_done()

    def enqueue(self, queue_name: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        with self._lock:
            q = self._queues.get(queue_name)
            if q is None:
                q = self._queues[queue_name] = self._queue_mod.Queue()
                self._threading.Thread(target=self._worker, args=(q,), daemon=True,
                                       name=f"screwit-{queue_name}").start()
            self._n += 1
            q.put((func, args, kwargs))
            return f"thread-{queue_name}-{self._n}"


class RqQueue:
    """Enqueues on Redis via RQ. Imported lazily so the dev install needs no redis/rq."""

    def __init__(self, redis_url: str, job_timeout: int = 3600) -> None:
        import redis
        from rq import Queue as _RqQueue

        conn = redis.Redis.from_url(redis_url)
        self._gpu = _RqQueue(GPU_QUEUE, connection=conn, default_timeout=job_timeout)
        self._cpu = _RqQueue(CPU_QUEUE, connection=conn, default_timeout=job_timeout)

    def enqueue(self, queue_name: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        q = self._gpu if queue_name == GPU_QUEUE else self._cpu
        job = q.enqueue(func, *args, **kwargs)
        return job.id


def make_queue(backend: str, redis_url: str) -> Queue:
    if backend == "inline":
        return InlineQueue()
    if backend == "thread":
        return ThreadQueue()
    if backend == "rq":
        return RqQueue(redis_url)
    raise ValueError(f"unknown queue backend {backend!r}")
