from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np
import torch

from fsop.data.native_replay import ARRAY_NAMES, NativeEpisode, NativeReplay, episode_paths
from fsop.geometry.yuv import rgb_to_model
from fsop.train.lab_train import RecoverySampler, dataset_verification, fresh_history, sha256
from joint_core import Capture

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'direct_command_trial'
RECOVERY = HERE.parent / 'direct_command_recovery'
DATA_TRIAL = HERE.parent / 'direct_command_recovery_trial'


def read_json(path):
    return json.loads(Path(path).read_text())


def isolated_report(script, args):
    with tempfile.TemporaryDirectory(prefix='joint-data-verification-') as directory:
        output = Path(directory) / 'report.json'
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join((str(Path.cwd()), str(Path.cwd() / 'src'), str(OLD), env.get('PYTHONPATH', '')))
        run = subprocess.run([sys.executable, str(script), *map(str, args), '--out', str(output)],
                             env=env, capture_output=True, text=True)
        if run.returncode:
            raise ValueError(f'independent verifier failed ({script}): {run.stderr[-4000:]} {run.stdout[-1000:]}')
        return read_json(output)


def membership(recipe):
    original, expanded = read_json(recipe['original_membership']), read_json(recipe['new_membership'])
    extras = [read_json(RECOVERY / 'cases' / (recipe['new_cases'][d]['id'] + '.json'))
              for d in ('skidpad', 'trackdrive')]
    cases = original['cases'] + extras
    if (original['train_cases'] != 32 or expanded != {**original, 'cases': cases, 'train_cases': 34}
            or len({case['id'] for case in cases}) != len(cases)):
        raise ValueError('expanded membership must add exactly two unique TRAIN cases')
    for case in extras:
        matches = [c for c in original['cases'] if c['id'] == case['id'].removesuffix('-direct-recovery-v1')]
        if len(matches) != 1 or case != {**matches[0], 'id': case['id']} or case['split'] != 'train':
            raise ValueError('new case differs from its matched TRAIN case')
    return {'original_sha256': sha256(recipe['original_membership']),
            'expanded_sha256': sha256(recipe['new_membership']), 'train_cases': 34}


def verified_receipts(recipe, device):
    if (sha256(recipe['manifest']) != recipe['manifest_sha256'] or
            sha256(recipe['base_checkpoint']) != recipe['base_checkpoint_sha256'] or
            sha256(recipe['collection_checkpoint']) != recipe['collection_checkpoint_sha256']):
        raise ValueError('manifest or checkpoint identity differs')
    old = dataset_verification(recipe['manifest'])
    independent = isolated_report('scripts/verify_lab_dataset.py',
                                  ['--manifest', recipe['manifest'], '--membership', recipe['original_membership']])
    if independent != read_json(old['path']):
        raise ValueError('original verification differs from independent current audit')
    receipts = {}
    for discipline in ('skidpad', 'trackdrive'):
        item = recipe['new_cases'][discipline]
        episode, receipt = Path(item['episode']).resolve(), Path(item['receipt']).resolve()
        if receipt.is_relative_to(episode):
            raise ValueError('receipt must be outside episode')
        stored = read_json(receipt)
        current = isolated_report(RECOVERY / 'verify_v2.py',
                                  ['--episode', episode, '--checkpoint', recipe['collection_checkpoint'], '--device', device])
        if current.get('passed') is not True or current.get('reasons') != [] or current != stored:
            raise ValueError('new recovery receipt differs from independent current replay')
        receipts[discipline] = {'path': str(receipt), 'sha256': sha256(receipt), 'report': stored}
    return old, independent, receipts


def close_episode(episode):
    for name in ARRAY_NAMES:
        array = getattr(episode, name, None)
        mmap = getattr(array, '_mmap', None)
        if mmap is not None and not mmap.closed:
            mmap.close()


def stable_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def legacy_identity(dataset, val, recipe):
    return {'manifest_path': str(Path(recipe['manifest']).resolve()),
            'manifest_sha256': sha256(recipe['manifest']),
            'recovery_recipe_path': str(Path(recipe['recovery_recipe']).resolve()),
            'recovery_recipe_sha256': sha256(recipe['recovery_recipe']),
            'train_index_sha256': stable_digest(dataset.index),
            'val_index_sha256': stable_digest(val.index),
            'episodes': [{'path': str(ep.path.resolve()), 'meta_sha256': stable_digest(ep.meta),
                          'disk_meta_sha256': sha256(ep.path / 'meta.json')} for ep in dataset.episodes]}


def legacy_report(recipe_path):
    recipe = read_json(recipe_path)
    episodes = []
    try:
        for path in episode_paths(manifest=recipe['manifest']):
            episode = NativeEpisode.__new__(NativeEpisode)
            episodes.append(episode)
            NativeEpisode.__init__(episode, path)
        train, val = NativeReplay(episodes, 'train'), NativeReplay(episodes, 'val')
        if (len(train), len(val)) != (7006, 2067):
            raise ValueError('original anchor counts differ')
        sampler = RecoverySampler(train, recipe['recovery_recipe'], recipe['recovery_gates'], 128, 16,
                                  recipe['base_checkpoint_sha256'], provenance_mode='per_case_v2')
        return {'recipe_file_sha256': sha256(recipe_path), 'identity': legacy_identity(train, val, recipe),
                'groups': sampler.groups, 'recovery_indices': sorted(sampler.recovery_indices),
                'protocol': sampler.protocol()}
    finally:
        for episode in episodes:
            close_episode(episode)


class LegacySpec:
    def __init__(self, dataset, val, recipe, report):
        if (report['recipe_file_sha256'] != sha256(HERE / 'recipe.json') or
                recipe != read_json(HERE / 'recipe.json') or
                report['identity'] != legacy_identity(dataset, val, recipe)):
            raise ValueError('legacy dataset or recipe identity differs')
        groups, protocol = report['groups'], report['protocol']
        recovery_recipe = read_json(recipe['recovery_recipe'])
        if [g['case_id'] for g in groups] != recovery_recipe['recovery_case_ids'] or not groups:
            raise ValueError('legacy recovery group identity differs')
        indices = []
        for group in groups:
            ei = group['episode']
            ep = dataset.episodes[ei]
            gate = Path(recipe['recovery_gates']) / (group['case_id'] + '.json')
            frames = group['frames']
            receipt = read_json(gate)
            if (receipt['recovery_frame_indices'] != frames or receipt['recovery_ticks'] != group['ticks'] or
                    receipt['takeover_tick'] != group['takeover_tick'] or
                    receipt['terminal_start_tick'] != group['terminal_start_tick'] or
                    group['collection_contract'] != recovery_recipe['collection_by_case'][group['case_id']] or
                    group['collection_checkpoint_sha256'] != group['collection_contract']['checkpoint_sha256'] or
                    ep.meta['id'] != group['case_id'] or ep.meta['split'] != 'train' or
                    group['gate_path'] != str(gate.resolve()) or group['gate_sha256'] != sha256(gate) or
                    len(frames) < 8 or len(frames) != len(group['indices']) or
                    len(frames) != len(group['ticks']) or
                    any(dataset.index[i] != (ei, frame) or int(ep.ticks[frame]) != tick
                        for i, frame, tick in zip(group['indices'], frames, group['ticks'], strict=True))):
                raise ValueError('legacy recovery group binding differs')
            indices.extend(group['indices'])
        if (len(indices) != len(set(indices)) or sorted(indices) != report['recovery_indices'] or
                protocol['recovery_episodes'] != [dict(g, anchor_probability=1 / (len(groups) * len(g['indices'])))
                                                  for g in groups] or
                protocol['recovery_episode_probability'] != 1 / len(groups) or
                protocol['recovery_anchor_count'] != len(indices)):
            raise ValueError('legacy recovery indices or uniform groups differ')
        other_groups = {}
        recovery_set = set(indices)
        for index, (ei, frame) in enumerate(dataset.index):
            if index not in recovery_set:
                other_groups.setdefault(ei, []).append(frame)
        other_count = len(dataset) - len(indices)
        expected_protocol = {'recipe_path': str(Path(recipe['recovery_recipe']).resolve()),
                             'recipe_sha256': sha256(recipe['recovery_recipe']), 'provenance_mode': 'per_case_v2',
                             'collection_policy_checkpoint_sha': None,
                             'optimization_init_sha256': recipe['base_checkpoint_sha256'],
                             'window_ticks': 40, 'control_hz': 20, 'with_replacement': True,
                             'recovery_draws_per_batch': 16, 'other_draws_per_batch': 112,
                             'recovery_episode_probability': 1 / len(groups),
                             'other_anchor_probability': 1 / other_count,
                             'recovery_anchor_count': len(indices), 'other_anchor_count': other_count,
                             'recovery_episodes': [dict(g, anchor_probability=1 / (len(groups) * len(g['indices'])))
                                                   for g in groups],
                             'other_episodes': [{'case_id': dataset.episodes[ei].meta['id'],
                                                 'episode': ei, 'frames': frames}
                                                for ei, frames in other_groups.items()]}
        if protocol != expected_protocol:
            raise ValueError('legacy recovery protocol differs')
        self.dataset, self.groups, self.recovery_indices = dataset, groups, set(indices)
        self._protocol = protocol

    def protocol(self):
        return self._protocol



class TrialSampler:
    def __init__(self, dataset, legacy, new_groups):
        self.dataset, self.legacy, self.new_groups = dataset, legacy, new_groups
        self.old_indices = set(legacy.recovery_indices)
        self.new_indices = {i for group in new_groups for i in group['indices']}
        if len(new_groups) != 2 or any(len(g['indices']) < 8 for g in new_groups) or self.old_indices & self.new_indices:
            raise ValueError('invalid recovery groups')
        self.other = np.array([i for i in range(len(dataset))
                               if i not in self.old_indices and i not in self.new_indices], dtype=np.int64)
        if not len(self.other):
            raise ValueError('ordinary pool is empty')
        self.counts = np.zeros(len(dataset), dtype=np.int64)

    def draw(self, rng):
        def group_draw(groups, count):
            choices = rng.integers(len(groups), size=count)
            return [groups[int(g)]['indices'][int(rng.integers(len(groups[int(g)]['indices'])))] for g in choices]
        result = np.concatenate((np.asarray(group_draw(self.legacy.groups, 16), dtype=np.int64),
                                 np.asarray(group_draw(self.new_groups, 16), dtype=np.int64),
                                 self.other[rng.integers(len(self.other), size=96)]))
        rng.shuffle(result)
        if (sum(int(i) in self.old_indices for i in result) != 16 or
                sum(int(i) in self.new_indices for i in result) != 16):
            raise ValueError('sampler violated 16/16/96 contract')
        np.add.at(self.counts, result, 1)
        return result

    def protocol(self):
        return {'legacy_gate': self.legacy.protocol(), 'new_groups': self.new_groups,
                'old_recovery_draws': 16, 'new_recovery_draws': 16, 'ordinary_draws': 96,
                'new_case_probability': 0.5, 'with_replacement': True, 'ordinary_indices': self.other.tolist()}

    def draw_counts(self):
        return {'total_draws': int(self.counts.sum()),
                'old_recovery_draws': int(sum(self.counts[i] for i in self.old_indices)),
                'new_recovery_draws': int(sum(self.counts[i] for i in self.new_indices)),
                'ordinary_draws': int(sum(self.counts[i] for i in self.other)),
                'episodes': [{'case_id': ep.meta['id'], 'anchors': [
                    {'frame': frame, 'draws': int(self.counts[i])}
                    for i, (episode, frame) in enumerate(self.dataset.index) if episode == ei]}
                    for ei, ep in enumerate(self.dataset.episodes) if ep.meta['split'] == 'train']}


class JointData(AbstractContextManager):
    def __init__(self, recipe, verification_device):
        self.episodes = []
        self.closed = False
        try:
            m = membership(recipe)
            old, independent, receipts = verified_receipts(recipe, verification_device)
            legacy_report_data = isolated_report(HERE / 'joint_data.py', ['--legacy-spec', '--recipe', HERE / 'recipe.json'])
            paths = episode_paths(manifest=recipe['manifest'])
            for path in paths:
                self._open(path)
            old_train, old_val = NativeReplay(self.episodes.copy(), 'train'), NativeReplay(self.episodes.copy(), 'val')
            if (len(old_train), len(old_val)) != (7006, 2067):
                raise ValueError('original anchor counts differ')
            legacy = LegacySpec(old_train, old_val, recipe, legacy_report_data)
            for discipline in ('skidpad', 'trackdrive'):
                self._open(recipe['new_cases'][discipline]['episode'])
            identities = {}
            for ep in self.episodes:
                track = str(ep.meta['track_id'])
                identity = (track, str(ep.meta['seed'])) if str(ep.meta['discipline']).lower() == 'skidpad' else (track,)
                if identity in identities and identities[identity] != ep.meta['split']:
                    raise ValueError('train/VAL identity overlap')
                identities[identity] = ep.meta['split']
            self.train, self.val = NativeReplay(self.episodes, 'train'), NativeReplay(self.episodes, 'val')
            if (self.train.index[:7006] != old_train.index or self.val.index != old_val.index or
                    (len(self.train), len(self.val)) != (7430, 2067)):
                raise ValueError('joint anchor membership differs')
            old_ids = [ep.meta['id'] for ep in old_train.episodes if ep.meta['split'] == 'train']
            expected = [c['id'] for c in read_json(recipe['original_membership'])['cases'] if c['split'] == 'train']
            if len(old_ids) != 32 or set(old_ids) != set(expected):
                raise ValueError('original training membership differs')
            groups = []
            for discipline, expected_count in (('skidpad', 219), ('trackdrive', 205)):
                item, report = recipe['new_cases'][discipline], receipts[discipline]['report']
                matches = [ei for ei, ep in enumerate(self.episodes) if ep.meta.get('id') == item['id']]
                if len(matches) != 1:
                    raise ValueError('new case identity missing or duplicated')
                ei = matches[0]
                ep = self.episodes[ei]
                indices = [i for i, (e, frame) in enumerate(self.train.index)
                           if e == ei and int(ep.ticks[frame]) in report['qualified_anchors_first_2s']]
                ticks = [int(ep.ticks[self.train.index[i][1]]) for i in indices]
                if ticks != report['qualified_anchors_first_2s'] or ep.meta['split'] != 'train' or len(ep.anchors) != expected_count:
                    raise ValueError('recovery anchors differ')
                groups.append({'case_id': item['id'], 'episode': ei, 'indices': indices, 'ticks': ticks})
            self.sampler = TrialSampler(self.train, legacy, groups)
            self.verification = {'original': old, 'original_independent': independent, 'membership': m,
                                 'new_receipts': receipts, 'original_train_count': 7006, 'added_train_count': 424,
                                 'train_count': 7430, 'val_count': 2067}
        except BaseException:
            self.close()
            raise

    def _open(self, path):
        episode = NativeEpisode.__new__(NativeEpisode)
        self.episodes.append(episode)
        NativeEpisode.__init__(episode, path)

    def close(self):
        if not self.closed:
            for episode in self.episodes:
                close_episode(episode)
            self.closed = True

    def __exit__(self, *args):
        self.close()


def wiring_indices(sampler):
    rng = np.random.default_rng(0)
    old_other = [i for i in sampler.other if i < len(sampler.legacy.dataset)]
    if len(old_other) < 16:
        raise ValueError('fewer than 16 old ordinary anchors')
    return (rng.choice(old_other, size=16, replace=False).tolist() +
            [i for group in sampler.new_groups for i in rng.choice(group['indices'], size=8, replace=False).tolist()])


def batch_input(model, dataset, ids, device, hook=True, blank=False):
    pairs = [dataset.index[int(i)] for i in ids]
    eps, frames = zip(*pairs)
    rows = []
    for ei, frame in pairs:
        ep = dataset.episodes[ei]
        tick = int(ep.ticks[frame])
        if not ep.teacher[tick]:
            raise ValueError('same-tick command is not a teacher control')
        rows.append((ep, frame, tick))
    blank_frame = rgb_to_model(np.zeros((256, 512, 3), dtype=np.uint8)) if blank else None
    blank_pair = np.concatenate((blank_frame, blank_frame)) if blank else None

    class BlankEpisode:
        def frame_pair(self, camera, frame):
            return blank_pair

    history_episodes = [BlankEpisode() for _ in dataset.episodes] if blank else dataset.episodes
    road = torch.from_numpy(np.stack([blank_pair if blank else ep.frame_pair('road', k) for ep, k, _ in rows])).to(device).float()
    wide = torch.from_numpy(np.stack([blank_pair if blank else ep.frame_pair('wide', k) for ep, k, _ in rows])).to(device).float()
    desire = torch.from_numpy(np.stack([np.array(ep.desires[k], copy=True) for ep, k, _ in rows])).to(device).float()
    traffic = torch.tensor([[1., 0.]], device=device).repeat(len(rows), 1)
    prop = torch.tensor(np.stack([ep.states[tick, 3:6] for ep, _, tick in rows]).copy(), device=device).float()
    target = torch.tensor(np.stack([ep.controls[tick] for ep, _, tick in rows]).copy(), device=device).float()
    with torch.no_grad():
        history = fresh_history(model, history_episodes, eps, frames, device, chunk_size=64,
                                amp=False, channels_last=False, history_context=8)
        hidden = model.base.encode(road, wide).hidden
    history, hidden = history.detach().clone(), hidden.detach().clone()
    if torch.is_inference(history) or torch.is_inference(hidden):
        raise ValueError('joint policy backward requires ordinary frozen encoder tensors')
    capture = Capture(model) if hook else None
    try:
        output = model.base.policy_raw(torch.cat((history, hidden[:, None]), 1), desire, traffic)
        latent = capture.take(len(rows)) if capture else None
        if hook and torch.is_grad_enabled() and any(p.requires_grad for p in model.base.policy.parameters()) and latent.grad_fn is None:
            raise ValueError('joint command latent lost policy gradient path')
        return latent, prop, target, tuple(value.detach().clone() for value in output)
    finally:
        if capture:
            capture.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy-spec', action='store_true', required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with args.out.open('x') as stream:
        json.dump(legacy_report(args.recipe), stream, allow_nan=False)
