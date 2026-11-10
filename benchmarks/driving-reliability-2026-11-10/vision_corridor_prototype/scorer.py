"""Privileged offline scorer. Never imported by the inference planner or adapter."""
from math import atan2, cos, sin
from pathlib import Path

import numpy as np

from fsop.sim.expert import Expert, ExpertParams, Mission
from fsop.sim.vehicle import VehicleParams, VehicleState
from fsop.track.fixed_layouts import skidpad
from fsop.track.io import load_json
from planner import angle_difference, point_at, path_length


def world_to_ego(points, state):
    relative = np.asarray(points) - np.array([state.x, state.y])
    return relative @ np.array([[cos(state.yaw), -sin(state.yaw)], [sin(state.yaw), cos(state.yaw)]])


def sampled_footprint_margin(path, cone_xy, recipe):
    if len(path) < 2 or len(cone_xy) == 0:
        return None
    points, headings = [], []
    for a, b in zip(path, path[1:]):
        vector = np.asarray(b) - a
        length = np.linalg.norm(vector)
        if length <= recipe.numeric_epsilon:
            continue
        n = max(1, int(np.ceil(length / recipe.path_spacing_m)))
        points.extend(np.asarray(a)[None] + np.linspace(0, 1, n + 1)[:, None] * vector)
        headings.extend([np.arctan2(vector[1], vector[0])] * (n + 1))
    if not points:
        return None
    displacement = np.asarray(cone_xy)[None, :, :] - np.asarray(points)[:, None, :]
    h = np.asarray(headings)[:, None]
    along = np.cos(h) * displacement[:, :, 0] + np.sin(h) * displacement[:, :, 1]
    across = -np.sin(h) * displacement[:, :, 0] + np.cos(h) * displacement[:, :, 1]
    qx = np.maximum(-recipe.rear_extent_m - along, along - recipe.front_extent_m)
    qy = np.abs(across) - recipe.vehicle_width_m / 2
    signed = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)
    return float(np.min(signed) - recipe.cone_radius_m - recipe.heuristic_clearance_pad_m)


class ReferenceScorer:
    def __init__(self, metadata, recipe, protocol, states=None, controls=None, flags=None, cones=None, track_path=None, teacher_params=None):
        self.metadata = metadata
        self.recipe = recipe
        self.protocol = protocol
        self.states = states
        self.cones = cones
        self.progress = None
        self.reconstruction_max_abs = None
        self.teacher_verified = False
        self.mission = None
        self.expert = None
        discipline = metadata["case"]["discipline"]
        if states is None:
            params = teacher_params if teacher_params is not None else metadata["expert_params"]
            vehicle = metadata["vehicle_params"]
        else:
            params = metadata["teacher"]
            vehicle = metadata["vehicle"]
        if discipline == "skidpad":
            track = skidpad()
        elif track_path is not None and Path(track_path).exists():
            track = load_json(track_path)
            track.laps = 1
        else:
            track = None
        if track is not None:
            p = ExpertParams(**{**params, "k_lookahead": protocol["teacher_lookahead_speed_s"]})
            self.mission = Mission(track, p)
            self.expert = Expert(self.mission, VehicleParams(**vehicle))
        if states is not None and self.mission is not None:
            self.progress = []
            hint, s = self.mission.init_progress(*states[0, :2])
            for row in states:
                hint, s, _ = self.mission.update_progress(row[0], row[1], hint, s)
                self.progress.append(s)
            if flags is not None:
                cutoff = metadata["terminal_continuation"]["start_tick"]
                ids = np.flatnonzero(np.asarray(flags, dtype=bool)[:cutoff])
                residuals = [np.max(np.abs(np.asarray(self.expert.act(VehicleState(*states[i]), self.progress[i])) - controls[i])) for i in ids]
                if residuals:
                    self.reconstruction_max_abs = float(max(residuals))
                    self.teacher_verified = self.reconstruction_max_abs <= protocol["teacher_reconstruction_max_abs"]

    def bind_pilot_labels(self, records, diagnostic):
        labels = {row["frame"]: row for row in diagnostic["rows"]}
        residuals = []
        for record in records:
            label = labels[record["frame"]]
            if record["state"] != label["state"] or record["progress"] != label["progress"]:
                raise ValueError("pilot teacher label references a different recorded state")
            predicted = self.expert.act(VehicleState(**record["state"]), record["progress"]["s"])
            residuals.append(np.max(np.abs(np.asarray(predicted) - label["native_teacher_at_actual_state"])))
        self.reconstruction_max_abs = float(max(residuals))
        self.teacher_verified = self.reconstruction_max_abs <= self.protocol["teacher_reconstruction_max_abs"]
        if not self.teacher_verified:
            raise ValueError("pilot canonical teacher reconstruction failed")

    def _phase(self, state, tick, progress):
        if self.states is not None and tick >= self.metadata["terminal_continuation"]["start_tick"]:
            return "terminal"
        if self.metadata["case"]["discipline"] != "skidpad":
            return "trackdrive"
        crossings = self.metadata.get("summary", {}).get("route", {}).get("crossings", [])
        if len(crossings) == 5:
            names = ("entry", "right_1", "right_2", "left_1", "left_2", "exit")
            return names[sum(state.t >= x for x in crossings)]
        if self.mission is not None:
            seg = self.mission.track.generator["segments"]
            limits = [self.mission.s_arr[seg["right"][0]],
                      self.mission.s_arr[(seg["right"][0] + seg["right"][1]) // 2],
                      self.mission.s_arr[seg["left"][0]],
                      self.mission.s_arr[(seg["left"][0] + seg["left"][1]) // 2],
                      self.mission.s_arr[seg["exit"][0]]]
            return ("entry", "right_1", "right_2", "left_1", "left_2", "exit")[sum(progress >= x for x in limits)]
        return "unverified"

    def score(self, result, tick, record=None):
        if record is None:
            state = VehicleState(*self.states[tick])
            cone_xy = world_to_ego(self.cones[:, :2], state)
            progress = self.progress[tick] if self.progress is not None else None
            verified = self.teacher_verified
        else:
            state = VehicleState(**record["state"])
            cone_xy = world_to_ego(record["actual_cones_xyz"][:, :2], state)
            progress = record["progress"]["s"]
            verified = self.teacher_verified
        source = "canonical_mission" if verified else "measured_future_trajectory"
        teacher = None
        lookahead = result.control.lookahead_m
        target = result.control.target_xy
        if verified:
            teacher = self.expert.act(state, progress)
            reference_world = self.mission.point_at(progress + lookahead)
            tangent_world = self.mission.point_at(progress + lookahead + self.recipe.path_spacing_m) - reference_world
            reference = world_to_ego([reference_world], state)[0]
            heading = atan2(tangent_world[1], tangent_world[0]) - state.yaw
        elif self.states is not None:
            future = self.states[tick:, :2]
            length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(future, axis=0), axis=1))]
            if length[-1] >= lookahead:
                k = min(int(np.searchsorted(length, lookahead, side="right")), len(length) - 1)
                f = (lookahead - length[k - 1]) / max(length[k] - length[k - 1], self.recipe.numeric_epsilon)
                reference_world = future[k - 1] + f * (future[k] - future[k - 1])
                reference = world_to_ego([reference_world], state)[0]
                tangent = future[k] - future[k - 1]
                heading = atan2(tangent[1], tangent[0]) - state.yaw
            else:
                reference, heading = None, None
        else:
            reference, heading = None, None
        lateral_error, point_error, heading_error, wrong_branch = None, None, None, None
        if target is not None and reference is not None:
            delta = np.asarray(target) - reference
            lateral_error = float(-sin(heading) * delta[0] + cos(heading) * delta[1])
            point_error = float(np.linalg.norm(delta))
            a = point_at(result.path, max(0, lookahead - self.recipe.path_spacing_m))
            b = point_at(result.path, min(path_length(result.path), lookahead + self.recipe.path_spacing_m))
            if a is not None and b is not None:
                heading_error = angle_difference(atan2(b[1] - a[1], b[0] - a[0]), heading)
                wrong_branch = abs(heading_error) > self.protocol["branch_heading_error_rad"]
        correction_agrees = None
        if teacher is not None and abs(teacher[0]) > self.protocol["teacher_correction_min_rad"]:
            correction_agrees = result.control.steer_rad * teacher[0] > 0
        return {
            "phase": self._phase(state, tick, progress), "speed_mps": state.vx,
            "teacher_verified": verified, "reference_source": source,
            "reference_target_xy": None if reference is None else np.asarray(reference).tolist(),
            "lateral_error_m": lateral_error, "target_error_m": point_error,
            "heading_error_rad": heading_error, "wrong_branch_proxy": wrong_branch,
            "teacher_command": None if teacher is None else list(teacher),
            "command_error": None if teacher is None else [result.control.steer_rad - teacher[0], result.control.accel_mps2 - teacher[1]],
            "teacher_correction_direction_agrees": correction_agrees,
            "gt_sampled_footprint_margin_m": sampled_footprint_margin(result.path, cone_xy, self.recipe),
            "reference_limit": "sampled geometry on recorded states, not physical driving or a guaranteed swept-volume bound",
        }
