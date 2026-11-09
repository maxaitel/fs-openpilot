"""Bounded asynchronous adapter for the existing evidence format."""

from __future__ import annotations

import copy
import queue
import threading
from pathlib import Path
from typing import Any

from isaac_eval.evidence import EvidenceWriter


class AsyncEvidenceWriter:
    """Write snapshots in order on one worker, with at most two queued records.

    Calls to append and close must come from the same producer thread. The
    existing writer owns serialization and hashing; close drains the queue and
    joins the worker before returning. Count and write_seconds are final after
    close. Worker errors are raised on the next append or close.
    """

    def __init__(self, directory: str | Path, metadata: dict[str, Any]):
        self._writer = EvidenceWriter(directory, metadata)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=2)
        self._stop = object()
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="evidence-writer", daemon=True)
        self._thread.start()

    @property
    def count(self) -> int:
        return self._writer.count

    @property
    def write_seconds(self) -> float:
        return self._writer.write_seconds

    def _raise_error(self) -> None:
        if self._error is not None:
            raise self._error

    def _run(self) -> None:
        try:
            while True:
                record = self._queue.get()
                if record is self._stop:
                    return
                self._writer.append(record)
        except BaseException as exc:
            self._error = exc

    def _put(self, item: Any) -> None:
        # Poll for worker failure while the bounded queue is full.
        while True:
            self._raise_error()
            try:
                self._queue.put(item, timeout=0.05)
                return
            except queue.Full:
                continue

    def append(self, record: dict[str, Any]) -> None:
        if self._closed:
            raise ValueError("evidence writer is closed")
        self._raise_error()
        snapshot = copy.deepcopy(record)
        self._put(snapshot)
        self._raise_error()

    def close(self) -> None:
        if not self._closed:
            try:
                self._put(self._stop)
            finally:
                self._thread.join()
                self._closed = True
                self._writer.close()
        self._raise_error()

    def __enter__(self) -> AsyncEvidenceWriter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
