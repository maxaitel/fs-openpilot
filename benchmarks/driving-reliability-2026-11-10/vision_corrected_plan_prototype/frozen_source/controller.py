"""Saved-input prototype. Coordinates are body x forward and y left."""
from dataclasses import dataclass, replace
from math import atan2, cos, exp, sin, sqrt
import numpy as np


@dataclass(frozen=True)
class Observation:
    timestamp: float
    plan_xy: tuple
    cones: tuple
    motion: tuple
    mission_kind: str


@dataclass(frozen=True)
class Track:
    identifier: int
    xy: tuple
    cone_class: int
    score: float
    last_seen: float
    orange_side: int = 0


def observed(record, mission_kind):
    pred, state = record['prediction'], record['state']
    return Observation(float(state['t']), tuple(map(tuple, np.asarray(pred['plan'])[:, :2].tolist())),
                       tuple((float(x), float(y), int(c), float(s)) for (x, y), c, s in
                             zip(pred['cones_xy'], pred['cones_cls'], pred['cones_score'])),
                       tuple(float(state[k]) for k in ('vx', 'vy', 'r')), mission_kind)


def warp(point, motion, dt):
    vx, vy, r = motion
    theta = r * dt
    a = sin(theta) / r if abs(r) > 1e-9 else dt
    b = (1 - cos(theta)) / r if abs(r) > 1e-9 else 0.
    dx, dy = a * vx - b * vy, b * vx + a * vy
    x, y = point[0] - dx, point[1] - dy
    return cos(theta) * x + sin(theta) * y, -sin(theta) * x + cos(theta) * y


def stations(path):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]


def point_at(path, station):
    if len(path) < 2:
        return None
    s = stations(path)
    if station > s[-1] + 1e-9:
        return None
    return np.array([np.interp(station, s, path[:, j]) for j in (0, 1)])


def prior_path(points, p):
    xy = np.asarray(points, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
        raise ValueError('invalid original plan XY')
    kept = [xy[0]]
    for row in xy[1:]:
        if np.linalg.norm(row - kept[-1]) >= p['minimum_segment_m']:
            kept.append(row)
    if len(kept) < 2:
        return np.empty((0, 2))
    xy = np.asarray(kept)
    s = stations(xy)
    end = min(p['prior_length_m'], s[-1])
    sample = np.r_[np.arange(0, end, p['path_spacing_m']), end]
    return np.stack([np.interp(sample, s, xy[:, j]) for j in (0, 1)], axis=1)


def project(point, path):
    vectors = np.diff(path, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    u = np.clip(np.sum((np.asarray(point) - path[:-1]) * vectors, axis=1) / lengths ** 2, 0., 1.)
    feet = path[:-1] + u[:, None] * vectors
    index = int(np.argmin(np.linalg.norm(feet - point, axis=1)))
    tangent = vectors[index] / lengths[index]
    normal = np.array([-tangent[1], tangent[0]])
    return float(stations(path)[index] + u[index] * lengths[index]), float(np.dot(np.asarray(point) - feet[index], normal))


def offset_path(path, offset):
    if len(path) < 2:
        return path.copy()
    tangent = np.gradient(path, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-9)
    return path + offset * np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)


def margins(path, obstacles, p):
    if len(path) < 2 or not len(obstacles):
        return np.empty(0)
    tangent = np.gradient(path, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-9)
    delta = np.asarray(obstacles)[None, :, :] - path[:, None, :]
    along = np.sum(delta * tangent[:, None, :], axis=2)
    across = -delta[:, :, 0] * tangent[:, None, 1] + delta[:, :, 1] * tangent[:, None, 0]
    qx = np.maximum(-p['rear_extent_m'] - along, along - p['front_extent_m'])
    qy = np.abs(across) - p['vehicle_width_m'] / 2
    signed = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)
    return np.min(signed, axis=0) - p['cone_radius_m'] - p['heuristic_clearance_pad_m']


def control(path, motion, supported, reach, obstacles, p):
    speed = max(motion[0], 0.)
    lookahead = p['lookahead_base_m'] + p['lookahead_speed_s'] * speed
    target = point_at(path, lookahead)
    available = max(0., reach - p['front_extent_m'] - p['heuristic_clearance_pad_m'])
    required = speed * p['assumed_response_s'] + speed ** 2 / (2 * p['brake_mps2'])
    steer, curvature = 0., 0.
    if target is not None:
        dx, dy = target[0] + p['rear_axle_offset_m'], target[1]
        curvature = 2 * dy / max(dx * dx + dy * dy, 1e-9)
        steer = float(np.clip(atan2(p['wheelbase_m'] * curvature, 1), -p['steer_limit_rad'], p['steer_limit_rad']))
    clearance = margins(path, obstacles, p)
    conflict = bool(len(clearance) and np.min(clearance) <= 0)
    braking = p['brake_mps2'] if speed > p['stationary_speed_mps'] else p['stop_hold_mps2']
    stop_speed = sqrt((p['brake_mps2'] * p['assumed_response_s']) ** 2 + 2 * p['brake_mps2'] * available) - p['brake_mps2'] * p['assumed_response_s']
    target_speed = min(p['speed_cap_mps'], sqrt(p['lateral_accel_mps2'] / max(abs(curvature), 1e-9)), stop_speed) if supported and target is not None else 0.
    accel = float(np.clip(p['speed_gain_per_s'] * (target_speed - speed), -p['brake_mps2'], p['acceleration_max_mps2']))
    if not supported or target is None or required > available or conflict:
        accel, target_speed = -braking, 0.
    return {'steer_rad': steer if supported else 0., 'geometric_steer_rad': steer, 'accel_mps2': accel,
            'target_speed_mps': target_speed, 'lookahead_m': lookahead,
            'target_xy': None if target is None else target.tolist(), 'required_stop_distance_m': required,
            'available_stop_distance_m': available, 'predicted_path_conflict': conflict,
            'predicted_per_cone_margins_m': clearance.tolist()}


class CorrectedPlan:
    def __init__(self, p):
        self.p = p
        self.reset()

    def reset(self):
        self.tracks, self.previous, self.next_id = [], None, 0

    def step(self, obs):
        p = self.p
        if not np.isfinite([obs.timestamp, *obs.motion]).all() or obs.mission_kind not in p['nominal_halfwidth_m']:
            raise ValueError('invalid measured input or mission kind')
        path = prior_path(obs.plan_xy, p)
        reset = False
        if self.previous is not None:
            dt = obs.timestamp - self.previous.timestamp
            if dt <= 0 or dt > p['max_motion_gap_s']:
                self.reset()
                reset = True
            else:
                self.tracks = [replace(t, xy=warp(t.xy, self.previous.motion, dt)) for t in self.tracks
                               if obs.timestamp - t.last_seen <= p['max_track_age_s']]
        self.previous = obs
        detections = sorted([d for d in obs.cones if np.isfinite(d).all() and
                             p['support_range_min_m'] <= np.hypot(d[0], d[1]) <= p['support_range_max_m'] and
                             d[0] >= 0 and d[3] >= p['support_score_min']], key=lambda d: (-d[3], d[0], d[1], d[2]))
        used, conflicts = set(), 0
        for x, y, cls, score in detections[:p['max_detections']]:
            cls = cls if cls in range(4) else 4
            nearest = sorted((np.hypot(t.xy[0] - x, t.xy[1] - y), i) for i, t in enumerate(self.tracks) if i not in used)
            if nearest and nearest[0][0] <= p['association_distance_m']:
                _, index = nearest[0]
                old = self.tracks[index]
                merged = cls if old.cone_class == cls or old.cone_class in (2, 3) and cls in (2, 3) else 4
                conflicts += int(merged == 4 and old.cone_class != 4)
                self.tracks[index] = replace(old, xy=(x, y), cone_class=merged, score=score, last_seen=obs.timestamp)
            else:
                index = len(self.tracks)
                self.tracks.append(Track(self.next_id, (x, y), cls, score, obs.timestamp))
                self.next_id += 1
            used.add(index)
        self.tracks = sorted(self.tracks, key=lambda t: (-t.last_seen, t.identifier))[:p['max_tracks']]
        residuals = []
        halfwidth = p['nominal_halfwidth_m'][obs.mission_kind]
        if len(path) >= 2:
            for i, track in enumerate(self.tracks):
                if track.cone_class == 4 or track.xy[0] < -p['max_behind_m']:
                    continue
                s, d = project(track.xy, path)
                side = 1 if track.cone_class == 0 else -1 if track.cone_class == 1 else track.orange_side
                if track.cone_class in (2, 3):
                    if side == 0 and abs(d) >= p['orange_side_min_m']:
                        side = 1 if d > 0 else -1
                        self.tracks[i] = replace(track, orange_side=side)
                    elif side * d < 0:
                        continue
                residual = d - side * halfwidth
                if not side or side * d < p['boundary_side_min_m'] or abs(residual) > p['max_boundary_residual_m']:
                    continue
                age = obs.timestamp - track.last_seen
                weight = track.score * exp(-age / p['max_track_age_s']) * exp(-s / p['station_weight_scale_m'])
                residuals.append({'track_id': track.identifier, 'station_m': s, 'normal_m': d,
                                  'side': side, 'residual_m': residual, 'weight': weight, 'age_s': age})
        offset = 0.
        if residuals:
            ordered = sorted(residuals, key=lambda r: (r['residual_m'], r['track_id']))
            cumulative = np.cumsum([r['weight'] for r in ordered])
            median = ordered[int(np.searchsorted(cumulative, cumulative[-1] / 2))]['residual_m']
            offset = float(np.clip(median, -p['max_offset_m'], p['max_offset_m']))
        corrected = offset_path(path, offset)
        supported = bool(residuals) and point_at(corrected, p['lookahead_base_m'] + p['lookahead_speed_s'] * max(obs.motion[0], 0)) is not None
        reach = max((r['station_m'] for r in residuals), default=0.)
        obstacles = [t.xy for t in self.tracks] + [(d[0], d[1]) for d in obs.cones if np.isfinite(d).all() and
                    d[3] >= p['obstacle_score_min'] and d[0] >= -p['max_behind_m'] and np.hypot(d[0], d[1]) <= p['prior_length_m']]
        return {'status': 'supported' if supported else 'unknown', 'offset_m': offset, 'support_reach_m': reach,
                'reason': 'boundary_offset' if supported else 'no_boundary_support' if not residuals else 'short_prior',
                'reset': reset, 'class_conflicts': conflicts, 'tracks': [t.__dict__ for t in self.tracks],
                'residuals': residuals, 'obstacles_xy': obstacles,
                'variants': {name: {'path': candidate.tolist(), 'control': control(candidate, obs.motion, supported, reach, obstacles, p)}
                             for name, candidate in (('uncorrected_pp', path), ('corrected_pp', corrected))}}
