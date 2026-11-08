import hashlib
import json
import zlib
from dataclasses import asdict

import pytest

from scripts.collect_skidpad_recovery import GuardSettings, RecoveryGuard, episode_specs


def test_guard_takes_over_and_releases_only_after_minimum_and_both_recover():
    guard = RecoveryGuard(GuardSettings())
    assert not guard.teacher(0.0, 0.6, 0.2)
    assert guard.teacher(0.05, 0.61, 0.0)
    assert guard.teacher(0.9, 0.0, 0.0)
    assert guard.teacher(1.1, 0.3, 0.0)
    assert guard.teacher(1.2, 0.0, 0.12)
    assert not guard.teacher(1.3, 0.29, 0.11)
    assert guard.teacher(1.4, 0.0, -0.21)
    assert guard.interventions == 2


def test_episode_seeds_are_unique_training_only_and_stable_when_extended():
    specs = episode_specs(60998, 11000)
    assert specs[:4] == episode_specs(60998, 4)
    assert len({seed for _, seed, _ in specs}) == len(specs)
    assert all(zlib.crc32(eid.encode()) % 1000 >= 200 for eid, _, _ in specs)
    assert not {61000, 61001, 61002, 71000, 71001, 71002, 71003, 71004} & {seed for _, seed, _ in specs}
    assert all(eid == f"recovery_{seed:06d}_skidpad" for eid, seed, _ in specs)
    assert [dr for _, _, dr in specs[:4]] == ["light", "heavy", "light", "heavy"]


def test_resume_rejects_a_different_checkpoint(tmp_path, monkeypatch):
    from scripts.collect_skidpad_recovery import main

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"new checkpoint")
    output = tmp_path / "output"
    eid, seed, _ = episode_specs(85000, 1)[0]
    episode = output / eid
    episode.mkdir(parents=True)
    (episode / "meta.json").write_text(json.dumps({"id": eid, "seed": seed, "collection": {
        "checkpoint_sha256": "different"}}))
    monkeypatch.setattr("sys.argv", ["collect", "--checkpoint", str(checkpoint), "--output", str(output),
                                    "--count", "1"])
    with pytest.raises(ValueError, match="settings or checkpoint mismatch"):
        main()


@pytest.mark.parametrize(('stored_reference', 'course_heading', 'accepted'), [
    (None, False, True),
    (None, True, False),
    ('course', False, False),
    ('course', True, True),
], ids=['legacy-body', 'body-to-course', 'course-to-body', 'same-course'])
def test_resume_preserves_heading_reference(tmp_path, monkeypatch, stored_reference, course_heading, accepted):
    from scripts.collect_skidpad_recovery import main

    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'checkpoint')
    output = tmp_path / 'output'
    eid, seed, dr = episode_specs(87000, 1)[0]
    episode = output / eid
    episode.mkdir(parents=True)
    collection = {'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  'guard': asdict(GuardSettings()), 'controller': 'pp', 'max_time': 90.0,
                  'venue_prob': 1.0, 'noise': False, 'dr': dr}
    if stored_reference is not None:
        collection['heading_reference'] = stored_reference
    meta = {'id': eid, 'seed': seed, 'track_id': 'skidpad', 'discipline': 'skidpad',
            'frames': 1, 'scene': {}, 'summary': {'result': 'pass'}, 'collection': collection}
    (episode / 'meta.json').write_text(json.dumps(meta))
    (episode / 'controls.npy').touch()
    argv = ['collect', '--checkpoint', str(checkpoint), '--output', str(output),
            '--start-seed', '87000', '--count', '1']
    if course_heading:
        argv.append('--course-heading')
    monkeypatch.setattr('sys.argv', argv)
    if accepted:
        main()
        rows = json.loads((output / 'index.json').read_text())['episodes']
        assert [row['id'] for row in rows] == [eid]
        assert json.loads((episode / 'meta.json').read_text()) == meta
    else:
        with pytest.raises(ValueError, match='settings or checkpoint mismatch'):
            main()
        assert not (output / 'index.json').exists()
