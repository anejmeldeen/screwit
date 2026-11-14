"""GPU worker entrypoint: `python -m screwit.app.worker` (SPEC 3, 7 P4).

Runs an rq SimpleWorker on the GPU queue with concurrency 1 (no fork, so CUDA survives; SPEC 3.1).
The CPU queue (validation, CAD) is served by a separate ordinary worker on the same command with
`--cpu`.
"""

from __future__ import annotations

import sys

from screwit.app.queue import CPU_QUEUE, GPU_QUEUE
from screwit.app.settings import settings


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    queue_name = CPU_QUEUE if "--cpu" in argv else GPU_QUEUE

    import redis
    from rq import Queue, SimpleWorker

    conn = redis.Redis.from_url(settings().redis_url)
    # SimpleWorker never forks, so CUDA stays initialised across jobs (SPEC 3.1).
    worker = SimpleWorker([Queue(queue_name, connection=conn)], connection=conn)
    worker.work()


if __name__ == "__main__":
    main()
