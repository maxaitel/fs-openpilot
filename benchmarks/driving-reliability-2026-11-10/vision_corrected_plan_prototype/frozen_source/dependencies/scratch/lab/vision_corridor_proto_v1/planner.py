"""Bounded cone-only corridor prototype. All coordinates use x forward, y left."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum, Enum
from itertools import combinations
import json
from math import atan2, cos, hypot, isfinite, sin, sqrt
from pathlib import Path
from typing import Literal

Point = tuple[float, float]


class ConeClass(IntEnum):
    BLUE = 0
    YELLOW = 1
    ORANGE = 2
    BIG_ORANGE = 3
    UNKNOWN = 4


class RouteCommand(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    EXIT = "exit"


@dataclass(frozen=True)
class ConeDetection:
    x: float
    y: float
    cone_class: ConeClass
    score: float


@dataclass(frozen=True)
class Motion:
    vx: float
    vy: float
    yaw_rate: float


@dataclass(frozen=True)
class RouteEvent:
    timestamp: float
    command: RouteCommand
    time_resolution_s: float = 0.0


@dataclass(frozen=True)
class Observation:
    timestamp: float
    cones: tuple[ConeDetection, ...]
    motion: Motion
    route_history: tuple[RouteEvent, ...] = ()


@dataclass(frozen=True)
class Recipe:
    support_range_min_m: float
    support_range_max_m: float
    obstacle_score_min: float
    support_score_min: float
    association_distance_m: float
    max_track_age_s: float
    max_observation_gap_s: float
    max_motion_gap_s: float
    max_tracks: int
    max_detections: int
    max_gates: int
    beam_width: int
    max_path_gates: int
    gate_width_min_m: float
    gate_width_max_m: float
    gate_width_change_max_m: float
    gate_heading_max_rad: float
    edge_heading_max_rad: float
    heading_change_max_rad: float
    edge_length_min_m: float
    edge_length_max_m: float
    path_curvature_max_inv_m: float
    max_origin_extension_m: float
    max_origin_offset_m: float
    max_behind_m: float
    path_spacing_m: float
    minimum_path_gates: int
    ambiguity_target_distance_m: float
    ambiguity_heading_rad: float
    route_turn_min_rad: float
    candidate_length_slack_m: float
    length_cost_per_m: float
    heading_cost: float
    width_cost: float
    prior_cost: float
    score_cost: float
    age_cost: float
    vehicle_width_m: float
    wheelbase_m: float
    rear_axle_offset_m: float
    front_extent_m: float
    rear_extent_m: float
    cone_radius_m: float
    heuristic_clearance_pad_m: float
    lookahead_base_m: float
    lookahead_speed_s: float
    speed_cap_mps: float
    lateral_accel_mps2: float
    brake_mps2: float
    assumed_response_s: float
    acceleration_max_mps2: float
    speed_gain_per_s: float
    steer_limit_rad: float
    stop_hold_mps2: float
    stationary_speed_mps: float
    numeric_epsilon: float

    @classmethod
    def load(cls, path: Path | None = None) -> Recipe:
        source = path or Path(__file__).with_name("recipe.json")
        return cls(**json.loads(source.read_text())["planner"])


@dataclass(frozen=True)
class Track:
    identifier: int
    xy: Point
    cone_class: ConeClass
    score: float
    last_seen: float
    supported_at: float | None


@dataclass(frozen=True)
class Gate:
    left_id: int
    right_id: int
    midpoint: Point
    heading: float
    width: float
    color: Literal["opposite", "orange"]
    score: float
    age_s: float


@dataclass(frozen=True)
class ControlRequest:
    steer_rad: float
    accel_mps2: float
    target_speed_mps: float
    brake: bool
    lookahead_m: float
    target_xy: Point | None
    required_stop_distance_m: float
    available_stop_distance_m: float


@dataclass(frozen=True)
class Diagnostics:
    route: RouteCommand | None
    route_event_timestamp: float | None
    route_source: str
    rejected_detections: int
    track_count: int
    supported_track_count: int
    same_color_pairs_rejected: int
    gate_count: int
    candidate_count: int
    selected_gates: tuple[Gate, ...]
    predicted_clearance_m: float | None
    reason: str
    reset_reason: str | None
    observed_tracks: tuple[Track, ...]


@dataclass(frozen=True)
class CorridorResult:
    status: Literal["supported", "unknown"]
    path: tuple[Point, ...]
    support_length_m: float
    control: ControlRequest
    diagnostics: Diagnostics


def distance(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def angle_difference(a: float, b: float) -> float:
    return atan2(sin(a - b), cos(a - b))


def warp_point(point: Point, motion: Motion, dt: float) -> Point:
    """Exact stationary-point transport under a constant measured body twist."""
    theta = motion.yaw_rate * dt
    if abs(theta) < 1e-9:
        a, b = dt, motion.yaw_rate * dt * dt / 2
    else:
        a, b = sin(theta) / motion.yaw_rate, (1 - cos(theta)) / motion.yaw_rate
    tx = a * motion.vx - b * motion.vy
    ty = b * motion.vx + a * motion.vy
    x, y = point[0] - tx, point[1] - ty
    return cos(theta) * x + sin(theta) * y, -sin(theta) * x + cos(theta) * y


def path_length(path: tuple[Point, ...]) -> float:
    return sum(distance(a, b) for a, b in zip(path, path[1:]))


def point_at(path: tuple[Point, ...], station: float) -> Point | None:
    if not path or station < 0:
        return None
    for a, b in zip(path, path[1:]):
        length = distance(a, b)
        if station <= length and length > 0:
            t = station / length
            return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        station -= length
    return path[-1] if station <= 1e-9 else None


def path_curvature(path: tuple[Point, ...]) -> float:
    values = []
    for a, b, c in zip(path, path[1:], path[2:]):
        ab, bc, ac = distance(a, b), distance(b, c), distance(a, c)
        denominator = ab * bc * ac
        if denominator > 1e-9:
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            values.append(abs(2 * cross / denominator))
    return max(values, default=0.0)


def footprint_clearance(path: tuple[Point, ...], obstacles: tuple[Point, ...], p: Recipe) -> float | None:
    """Sampled reference-path rectangle clearance. It is not a dynamics rollout."""
    if not obstacles or len(path) < 2:
        return None
    clearance = float("inf")
    for a, b in zip(path, path[1:]):
        length = distance(a, b)
        if length <= p.numeric_epsilon:
            continue
        heading = atan2(b[1] - a[1], b[0] - a[0])
        steps = max(1, int(length / p.path_spacing_m) + 1)
        for i in range(steps + 1):
            t = i / steps
            x, y = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
            for ox, oy in obstacles:
                dx, dy = ox - x, oy - y
                u = cos(heading) * dx + sin(heading) * dy
                v = -sin(heading) * dx + cos(heading) * dy
                qx = max(-p.rear_extent_m - u, u - p.front_extent_m)
                qy = abs(v) - p.vehicle_width_m / 2
                signed = hypot(max(qx, 0), max(qy, 0)) + min(max(qx, qy), 0)
                clearance = min(clearance, signed - p.cone_radius_m - p.heuristic_clearance_pad_m)
    return clearance


def local_control(path: tuple[Point, ...], motion: Motion, p: Recipe, supported: bool) -> ControlRequest:
    speed = max(motion.vx, 0.0)
    lookahead = p.lookahead_base_m + p.lookahead_speed_s * speed
    required = speed * p.assumed_response_s + speed * speed / (2 * p.brake_mps2)
    length = path_length(path)
    available = max(length - p.front_extent_m - p.heuristic_clearance_pad_m, 0.0)
    target = point_at(path, lookahead)
    if not supported or target is None:
        braking = p.brake_mps2 if speed > p.stationary_speed_mps else p.stop_hold_mps2
        return ControlRequest(0.0, -braking, 0.0, True, lookahead, None, required, available)
    dx, dy = target[0] + p.rear_axle_offset_m, target[1]
    steer = atan2(2 * p.wheelbase_m * dy, max(dx * dx + dy * dy, p.numeric_epsilon))
    steer = max(-p.steer_limit_rad, min(p.steer_limit_rad, steer))
    curvature = max(path_curvature(path), abs(2 * dy / max(dx * dx + dy * dy, p.numeric_epsilon)))
    curve_speed = sqrt(p.lateral_accel_mps2 / max(curvature, p.numeric_epsilon))
    delay = p.assumed_response_s
    stop_speed = max(0.0, sqrt((p.brake_mps2 * delay) ** 2 + 2 * p.brake_mps2 * available) - p.brake_mps2 * delay)
    target_speed = min(p.speed_cap_mps, curve_speed, stop_speed)
    accel = max(-p.brake_mps2, min(p.acceleration_max_mps2, p.speed_gain_per_s * (target_speed - speed)))
    if required > available:
        accel = -p.brake_mps2
    return ControlRequest(steer, accel, target_speed, accel < 0, lookahead, target, required, available)


class CorridorPlanner:
    def __init__(self, recipe: Recipe | None = None):
        self.recipe = recipe or Recipe.load()
        self.reset()

    def reset(self) -> None:
        self._tracks: tuple[Track, ...] = ()
        self._last_observation: Observation | None = None
        self._previous_path: tuple[Point, ...] = ()
        self._route: RouteCommand | None = None
        self._route_timestamp: float | None = None
        self._next_id = 0

    def _route_update(self, obs: Observation) -> bool:
        events = sorted(obs.route_history, key=lambda item: item.timestamp)
        by_time: dict[float, RouteCommand] = {}
        for event in events:
            if not isfinite(event.timestamp) or event.timestamp > obs.timestamp or not isinstance(event.command, RouteCommand):
                return False
            if event.timestamp in by_time and by_time[event.timestamp] != event.command:
                return False
            by_time[event.timestamp] = event.command
            if self._route_timestamp == event.timestamp and self._route != event.command:
                return False
        for timestamp, command in by_time.items():
            if self._route_timestamp is None or timestamp > self._route_timestamp:
                self._route, self._route_timestamp = command, timestamp
        return True

    def _update_tracks(self, obs: Observation) -> tuple[int, bool]:
        p = self.recipe
        previous = self._last_observation
        tracks = list(self._tracks)
        if previous is not None:
            dt = obs.timestamp - previous.timestamp
            motion = Motion(*((a + b) / 2 for a, b in zip(
                (previous.motion.vx, previous.motion.vy, previous.motion.yaw_rate),
                (obs.motion.vx, obs.motion.vy, obs.motion.yaw_rate))))
            tracks = [replace(t, xy=warp_point(t.xy, motion, dt)) for t in tracks]
            self._previous_path = tuple(warp_point(xy, motion, dt) for xy in self._previous_path)
        tracks = [t for t in tracks if obs.timestamp - t.last_seen <= p.max_track_age_s
                  and t.xy[0] >= -p.max_behind_m and hypot(*t.xy) <= p.support_range_max_m]
        valid = []
        rejected = 0
        for cone in obs.cones:
            if (not all(isfinite(v) for v in (cone.x, cone.y, cone.score))
                    or not 0 <= cone.score <= 1 or not isinstance(cone.cone_class, ConeClass)
                    or cone.score < p.obstacle_score_min or cone.x < -p.max_behind_m
                    or hypot(cone.x, cone.y) > p.support_range_max_m):
                rejected += 1
                continue
            color = ConeClass.ORANGE if cone.cone_class == ConeClass.BIG_ORANGE else cone.cone_class
            valid.append(replace(cone, cone_class=color))
        valid.sort(key=lambda c: (-c.score, hypot(c.x, c.y), int(c.cone_class), c.x, c.y))
        overflow = len(valid) > p.max_detections
        for cone in valid[:p.max_detections]:
            xy = (cone.x, cone.y)
            trusted = (p.support_range_min_m <= hypot(*xy) <= p.support_range_max_m
                       and cone.score >= p.support_score_min and cone.cone_class != ConeClass.UNKNOWN)
            nearby = [(distance(t.xy, xy), i) for i, t in enumerate(tracks)
                      if distance(t.xy, xy) <= p.association_distance_m]
            if nearby:
                _, i = min(nearby)
                old = tracks[i]
                same_color = old.cone_class == cone.cone_class
                color = old.cone_class if same_color else ConeClass.UNKNOWN
                # Retained range-calibrated position survives weak nearfield observations.
                position = xy if trusted and same_color else old.xy
                tracks[i] = replace(old, xy=position, cone_class=color,
                                    score=max(old.score, cone.score), last_seen=obs.timestamp,
                                    supported_at=obs.timestamp if trusted and same_color else old.supported_at)
            else:
                tracks.append(Track(self._next_id, xy, cone.cone_class, cone.score,
                                    obs.timestamp, obs.timestamp if trusted else None))
                self._next_id += 1
        tracks.sort(key=lambda t: (hypot(*t.xy), t.identifier))
        overflow = overflow or len(tracks) > p.max_tracks
        self._tracks = tuple(tracks[:p.max_tracks])
        return rejected, overflow

    def _gates(self, now: float) -> tuple[tuple[Gate, ...], int, int, bool]:
        p = self.recipe
        supported = [t for t in self._tracks if t.supported_at is not None
                     and now - t.supported_at <= p.max_track_age_s
                     and now - t.last_seen <= p.max_observation_gap_s
                     and t.cone_class != ConeClass.UNKNOWN]
        gates = []
        same_color = 0
        for a, b in combinations(supported, 2):
            width = distance(a.xy, b.xy)
            if not p.gate_width_min_m <= width <= p.gate_width_max_m:
                continue
            if {a.cone_class, b.cone_class} == {ConeClass.BLUE, ConeClass.YELLOW}:
                left, right = (a, b) if a.cone_class == ConeClass.BLUE else (b, a)
                color = "opposite"
            elif a.cone_class == b.cone_class == ConeClass.ORANGE:
                left, right = (a, b) if a.xy[1] > b.xy[1] else (b, a)
                color = "orange"
            else:
                same_color += a.cone_class == b.cone_class
                continue
            dx, dy = right.xy[0] - left.xy[0], right.xy[1] - left.xy[1]
            heading = atan2(dx, -dy)
            if abs(heading) > p.gate_heading_max_rad:
                continue
            mid = ((a.xy[0] + b.xy[0]) / 2, (a.xy[1] + b.xy[1]) / 2)
            gates.append(Gate(left.identifier, right.identifier, mid, heading, width, color,
                              min(a.score, b.score), max(now - a.supported_at, now - b.supported_at)))
        gates.sort(key=lambda g: (hypot(*g.midpoint), g.left_id, g.right_id))
        return tuple(gates[:p.max_gates]), same_color, len(supported), len(gates) > p.max_gates

    def _paths(self, gates: tuple[Gate, ...], motion: Motion) -> list[tuple[float, tuple[Point, ...], tuple[Gate, ...]]]:
        p = self.recipe
        seeds = []
        for i, g in enumerate(gates):
            longitudinal = g.midpoint[0] * cos(g.heading) + g.midpoint[1] * sin(g.heading)
            lateral = -g.midpoint[0] * sin(g.heading) + g.midpoint[1] * cos(g.heading)
            if not -p.max_behind_m <= longitudinal <= p.max_origin_extension_m or abs(lateral) > p.max_origin_offset_m:
                continue
            seeds.append((i,))
        beam = seeds[:p.beam_width]
        completed = []
        for _ in range(p.max_path_gates - 1):
            extended = []
            for chain in beam:
                a = gates[chain[-1]]
                for i, b in enumerate(gates):
                    if i in chain or abs(a.width - b.width) > p.gate_width_change_max_m:
                        continue
                    length = distance(a.midpoint, b.midpoint)
                    if not p.edge_length_min_m <= length <= p.edge_length_max_m:
                        continue
                    heading = atan2(b.midpoint[1] - a.midpoint[1], b.midpoint[0] - a.midpoint[0])
                    if max(abs(angle_difference(heading, a.heading)), abs(angle_difference(heading, b.heading))) > p.edge_heading_max_rad:
                        continue
                    turn = abs(angle_difference(b.heading, a.heading))
                    if turn > p.heading_change_max_rad or turn / length > p.path_curvature_max_inv_m:
                        continue
                    if len(chain) > 1:
                        before = gates[chain[-2]]
                        old_heading = atan2(a.midpoint[1] - before.midpoint[1], a.midpoint[0] - before.midpoint[0])
                        if abs(angle_difference(heading, old_heading)) > p.heading_change_max_rad:
                            continue
                    extended.append(chain + (i,))
            if not extended:
                break
            def partial_cost(chain):
                gs = [gates[i] for i in chain]
                length = sum(distance(a.midpoint, b.midpoint) for a, b in zip(gs, gs[1:]))
                turns = sum(abs(angle_difference(a.heading, b.heading)) for a, b in zip(gs, gs[1:]))
                return p.length_cost_per_m * length + p.heading_cost * turns
            extended.sort(key=lambda c: (partial_cost(c), c))
            beam = extended[:p.beam_width]
            completed.extend(beam)
        candidates = []
        for chain in completed:
            gs = tuple(gates[i] for i in chain)
            if len(gs) < p.minimum_path_gates:
                continue
            a, b = gs[0].midpoint, gs[1].midpoint
            length = distance(a, b)
            tx, ty = (b[0] - a[0]) / length, (b[1] - a[1]) / length
            station = -(a[0] * tx + a[1] * ty)
            if station < -p.max_origin_extension_m or station > length:
                continue
            origin = (a[0] + station * tx, a[1] + station * ty)
            if hypot(*origin) > p.max_origin_offset_m:
                continue
            rest = tuple(g.midpoint for g in gs if (g.midpoint[0] - origin[0]) * tx + (g.midpoint[1] - origin[1]) * ty > p.numeric_epsilon)
            path = (origin,) + rest
            if len(path) < 2 or path_curvature(path) > p.path_curvature_max_inv_m:
                continue
            target = point_at(path, p.lookahead_base_m + p.lookahead_speed_s * max(motion.vx, 0))
            if target is None:
                continue
            turns = sum(abs(angle_difference(a.heading, b.heading)) for a, b in zip(gs, gs[1:]))
            width_change = sum(abs(a.width - b.width) for a, b in zip(gs, gs[1:]))
            cost = (p.length_cost_per_m * path_length(path) + p.heading_cost * turns
                    + p.width_cost * width_change + p.score_cost * sum(1 - g.score for g in gs) / len(gs)
                    + p.age_cost * max(g.age_s for g in gs))
            old = point_at(self._previous_path, p.lookahead_base_m + p.lookahead_speed_s * max(motion.vx, 0))
            if old is not None:
                cost += p.prior_cost * distance(target, old)
            candidates.append((cost, path, gs))
        return sorted(candidates, key=lambda item: (item[0], item[1]))

    def step(self, obs: Observation) -> CorridorResult:
        p = self.recipe
        reset_reason = None
        reason = "supported"
        finite = all(isfinite(x) for x in (obs.timestamp, obs.motion.vx, obs.motion.vy, obs.motion.yaw_rate))
        if not finite:
            self.reset()
            reason, reset_reason = "invalid_motion_or_time", "invalid_motion_or_time"
            obs = Observation(0.0, (), Motion(0.0, 0.0, 0.0))
        if self._last_observation is not None:
            dt = obs.timestamp - self._last_observation.timestamp
            if dt <= 0 or dt > p.max_motion_gap_s:
                reset_reason = "nonmonotonic_time" if dt <= 0 else "motion_gap"
                self.reset()
                reason = reset_reason
        route_valid = self._route_update(obs)
        rejected, track_overflow = self._update_tracks(obs)
        gates, same_color, support_count, gate_overflow = self._gates(obs.timestamp)
        candidates = self._paths(gates, obs.motion)
        obstacles = tuple(t.xy for t in self._tracks)
        chosen_path: tuple[Point, ...] = ()
        chosen_gates: tuple[Gate, ...] = ()
        clearance = None
        if reason == "supported":
            if not route_valid:
                reason = "invalid_route_history"
            elif track_overflow or gate_overflow:
                reason = "capacity_exceeded"
            elif not gates:
                reason = "no_supported_gates"
            elif not candidates:
                reason = "no_supported_nearfield_path"
            else:
                longest = max(path_length(item[1]) for item in candidates)
                eligible = [item for item in candidates if path_length(item[1]) >= longest - p.candidate_length_slack_m]
                lookahead = p.lookahead_base_m + p.lookahead_speed_s * max(obs.motion.vx, 0)
                best = eligible[0]
                def divergent(item):
                    a, b = point_at(item[1], lookahead), point_at(best[1], lookahead)
                    turn_a = angle_difference(item[2][-1].heading, item[2][0].heading)
                    turn_b = angle_difference(best[2][-1].heading, best[2][0].heading)
                    return distance(a, b) > p.ambiguity_target_distance_m or abs(angle_difference(turn_a, turn_b)) > p.ambiguity_heading_rad
                if any(divergent(item) for item in eligible[1:]):
                    if self._route in (RouteCommand.LEFT, RouteCommand.RIGHT):
                        sign = 1 if self._route == RouteCommand.LEFT else -1
                        eligible = [item for item in eligible if sign * angle_difference(item[2][-1].heading, item[2][0].heading) >= p.route_turn_min_rad]
                    elif self._route == RouteCommand.EXIT:
                        eligible = [item for item in eligible if item[2][-1].color == "orange"
                                    and abs(angle_difference(item[2][-1].heading, item[2][0].heading)) < p.route_turn_min_rad]
                    else:
                        eligible = []
                    if not eligible:
                        reason = "ambiguous_route"
                    else:
                        best = eligible[0]
                        if any(divergent(item) for item in eligible[1:]):
                            reason = "ambiguous_route"
                if reason == "supported":
                    _, chosen_path, chosen_gates = best
                    clearance = footprint_clearance(chosen_path, obstacles, p)
                    current = footprint_clearance(((0.0, 0.0), (p.numeric_epsilon * 2, 0.0)), obstacles, p)
                    if clearance is not None and current is not None:
                        clearance = min(clearance, current)
                    if clearance is None or clearance <= 0:
                        reason = "predicted_footprint_conflict"
                    elif path_length(chosen_path) < lookahead + p.front_extent_m:
                        reason = "insufficient_forward_support"
        supported = reason == "supported"
        control = local_control(chosen_path, obs.motion, p, supported)
        if supported:
            self._previous_path = chosen_path
        self._last_observation = obs
        diagnostics = Diagnostics(self._route, self._route_timestamp, "provided_external_events", rejected,
                                  len(self._tracks), support_count, same_color, len(gates), len(candidates),
                                  chosen_gates, clearance, reason, reset_reason, self._tracks)
        return CorridorResult("supported" if supported else "unknown", chosen_path, path_length(chosen_path), control, diagnostics)
