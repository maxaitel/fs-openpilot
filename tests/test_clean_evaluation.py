import json
from pathlib import Path

import pytest

from scripts import evaluate_skidpad_clean as evaluation


@pytest.fixture
def job(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'checkpoint')
    output = tmp_path / 'evaluation'
    monkeypatch.setattr('sys.argv', ['evaluate', '--checkpoint', str(checkpoint), '--output', str(output),
                                   '--seeds', '71003', '--levels', 'light', 'heavy', '--workers', '1'])
    return checkpoint, output


def install_probe(monkeypatch, mutate=None):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        folder = Path(command[command.index('--output') + 1])
        folder.mkdir(parents=True, exist_ok=True)
        data = {'checkpoint': command[command.index('--checkpoint') + 1], 'runs': [
            {'seed': 71003, 'dr': level, 'controller': 'pp', 'command_mode': 'recorded',
             'pass': True, 'doo': 0, 'oc': 0} for level in ('light', 'heavy')]}
        if mutate is not None:
            mutate(data)
        (folder / 'driving.json').write_text(json.dumps(data))

    monkeypatch.setattr(evaluation.subprocess, 'run', run)
    return calls


def test_successful_evaluation_resumes_only_completed_cases(job, monkeypatch):
    checkpoint, output = job
    calls = install_probe(monkeypatch)
    evaluation.main()
    completion = json.loads((output / '71003/protocol.json').read_text())
    root_protocol = json.loads((output / 'protocol.json').read_text())
    assert completion.items() >= root_protocol.items()
    assert completion['result_sha256']
    assert completion['sha256']
    assert completion['checkpoint'] == str(checkpoint)
    assert completion['command_mode'] == 'recorded'
    assert calls[0][calls[0].index('--controller') + 1] == 'pp'
    assert calls[0][calls[0].index('--command-mode') + 1] == 'recorded'
    evaluation.main()
    assert len(calls) == 1
    assert json.loads((output / 'driving.json').read_text())['summary']['clean'] == 2


@pytest.mark.parametrize(('field', 'value'), [('controller', 'op'), ('command_mode', 'zero')])
def test_rejects_wrong_control_protocol_before_marking_complete(job, monkeypatch, field, value):
    _, output = job
    install_probe(monkeypatch, lambda data: data['runs'][0].update({field: value}))
    with pytest.raises(ValueError, match='controller or command mode'):
        evaluation.main()
    assert not (output / '71003/protocol.json').exists()
    assert not (output / 'driving.json').exists()


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'extra', 'wrong_seed'])
def test_rejects_incomplete_or_duplicate_child_cases(job, monkeypatch, change):
    def mutate(data):
        if change == 'missing':
            data['runs'].pop()
        elif change == 'duplicate':
            data['runs'][1] = data['runs'][0].copy()
        elif change == 'extra':
            data['runs'].append(data['runs'][0].copy())
        else:
            data['runs'][0]['seed'] = 71004

    install_probe(monkeypatch, mutate)
    with pytest.raises(ValueError, match='exact requested cases'):
        evaluation.main()


@pytest.mark.parametrize('missing', ['protocol.json', '71003/protocol.json', '71003/driving.json'])
def test_rejects_orphan_results_or_completion_records(job, monkeypatch, missing):
    _, output = job
    calls = install_probe(monkeypatch)
    evaluation.main()
    (output / missing).unlink()
    with pytest.raises(ValueError, match='without a root protocol|Incomplete evaluation provenance'):
        evaluation.main()
    assert len(calls) == 1


@pytest.mark.parametrize(('field', 'value'), [('sha256', 'old checkpoint'), ('seeds', [71004]),
                                             ('levels', ['heavy']), ('controller', 'op'),
                                             ('command_mode', 'zero')])
def test_rejects_mismatched_child_provenance(job, monkeypatch, field, value):
    _, output = job
    calls = install_probe(monkeypatch)
    evaluation.main()
    path = output / '71003/protocol.json'
    completion = json.loads(path.read_text())
    completion[field] = value
    path.write_text(json.dumps(completion))
    with pytest.raises(ValueError, match='completion protocol mismatch'):
        evaluation.main()
    assert len(calls) == 1


def test_rejects_result_replaced_after_completion(job, monkeypatch):
    _, output = job
    install_probe(monkeypatch)
    evaluation.main()
    result = output / '71003/driving.json'
    data = json.loads(result.read_text())
    data['runs'][0]['doo'] = 20
    result.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='completion protocol mismatch'):
        evaluation.main()


def test_rejects_checkpoint_replaced_at_same_path(job, monkeypatch):
    checkpoint, _ = job
    calls = install_probe(monkeypatch)
    evaluation.main()
    checkpoint.write_bytes(b'different checkpoint')
    with pytest.raises(ValueError, match='different protocol'):
        evaluation.main()
    assert len(calls) == 1


def test_does_not_mark_complete_when_checkpoint_changes_during_run(job, monkeypatch):
    checkpoint, output = job
    install_probe(monkeypatch, lambda data: checkpoint.write_bytes(b'changed during run'))
    with pytest.raises(ValueError, match='Checkpoint changed during evaluation'):
        evaluation.main()
    assert not (output / '71003/protocol.json').exists()
    assert not (output / 'driving.json').exists()
