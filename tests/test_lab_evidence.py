import copy
import threading

import numpy as np
import pytest

from isaac_eval.evidence import EvidenceWriter, finalize_evidence, read_evidence
from isaac_lab_fsop.evidence import AsyncEvidenceWriter


def record():
    return {
        "images": {"wide": np.zeros((4, 5, 3), dtype=np.uint8), "road": np.ones((4, 5, 3), dtype=np.uint8)},
        "prediction": {"plan": np.zeros((2, 2)), "nested": [{"value": np.array([1, 2])}]},
        "action": [0.0, 1.0],
    }


def test_async_snapshots_match_sync_format_and_completion(tmp_path, monkeypatch):
    metadata = {"pilot_only": False, "label": "test"}
    value = record()
    snapshots = []
    for index in range(3):
        value["images"]["wide"].fill(index)
        value["prediction"]["nested"][0]["value"][0] = index
        value["action"][0] = index
        snapshots.append(copy.deepcopy(value))

    sync = tmp_path / "sync"
    with EvidenceWriter(sync, metadata) as writer:
        for snapshot in snapshots:
            writer.append(snapshot)

    started, release = threading.Event(), threading.Event()
    original_append = EvidenceWriter.append

    def paused_append(writer, item):
        started.set()
        assert release.wait(5)
        original_append(writer, item)

    monkeypatch.setattr(EvidenceWriter, "append", paused_append)
    async_path = tmp_path / "async"
    with AsyncEvidenceWriter(async_path, metadata) as writer:
        for snapshot in snapshots:
            writer.append(snapshot)
        assert started.wait(5)
        for snapshot in snapshots:
            snapshot["images"]["wide"].fill(99)
            snapshot["prediction"]["nested"][0]["value"].fill(99)
            snapshot["action"].append(99)
        release.set()
    assert writer.count == 3
    assert writer.write_seconds > 0
    assert not writer._thread.is_alive()
    assert (sync / "metadata.json").read_bytes() == (async_path / "metadata.json").read_bytes()
    assert (sync / "manifest.jsonl").read_bytes() == (async_path / "manifest.jsonl").read_bytes()
    for index in range(3):
        assert (sync / f"frame_{index:06d}.npz").read_bytes() == (async_path / f"frame_{index:06d}.npz").read_bytes()

    summary = tmp_path / "summary.json"
    summary.write_text('{"result": "ok"}')
    finalize_evidence(sync, summary)
    finalize_evidence(async_path, summary)
    assert (sync / "completion.json").read_bytes() == (async_path / "completion.json").read_bytes()
    loaded = list(read_evidence(async_path))
    assert [entry["frame"] for _, _, entry in loaded] == [0, 1, 2]
    for index, (_, item, _) in enumerate(loaded):
        assert np.all(item["images"]["wide"] == index)
        assert item["prediction"]["nested"][0]["value"].tolist() == [index, 2]
        assert item["action"] == [index, 1.0]


@pytest.mark.parametrize("blocked_operation", ["append", "close"])
def test_worker_failure_reaches_blocked_producer_and_close(tmp_path, monkeypatch, blocked_operation):
    started, release = threading.Event(), threading.Event()

    def fail_append(_writer, _item):
        started.set()
        assert release.wait(5)
        raise OSError("injected write failure")

    monkeypatch.setattr(EvidenceWriter, "append", fail_append)
    writer = AsyncEvidenceWriter(tmp_path / "episode", {"pilot_only": True})
    writer.append(record())
    assert started.wait(5)
    writer.append(record())
    writer.append(record())
    assert writer._queue.full()
    result = []
    operation_started = threading.Event()

    def operation():
        operation_started.set()
        try:
            if blocked_operation == "append":
                writer.append(record())
            else:
                writer.close()
        except BaseException as exc:
            result.append(exc)

    producer = threading.Thread(target=operation, daemon=True)
    producer.start()
    assert operation_started.wait(5)
    producer.join(0.1)
    assert producer.is_alive(), "operation did not block on the full queue"
    release.set()
    producer.join(5)
    assert not producer.is_alive(), "producer deadlocked on full queue"
    assert len(result) == 1 and isinstance(result[0], OSError)
    assert str(result[0]) == "injected write failure"
    with pytest.raises(OSError, match="injected write failure"):
        writer.close()
    assert not writer._thread.is_alive()
    assert writer._writer._manifest.closed


def test_context_exit_propagates_worker_failure(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()

    def fail_append(_writer, _item):
        started.set()
        assert release.wait(5)
        raise OSError("context write failure")

    monkeypatch.setattr(EvidenceWriter, "append", fail_append)
    with pytest.raises(OSError, match="context write failure"):
        with AsyncEvidenceWriter(tmp_path / "episode", {"pilot_only": True}) as writer:
            writer.append(record())
            assert started.wait(5)
            release.set()
            writer._thread.join(5)
    assert not writer._thread.is_alive()
    assert writer._writer._manifest.closed
