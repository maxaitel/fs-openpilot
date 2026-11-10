"""Audit recorded vision detections against cone positions on precontact TRAIN frames."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from fsop.data.targets import visible_cones, world_to_ego
from isaac_eval.evidence import read_evidence

cases = (("runs/direct_command_pilot3", 80), ("runs/direct_command_recovery_pilot2", 53))
for case_path, precontact_frames in cases:
    summary = json.loads((Path(case_path) / "summary.json").read_text())
    if summary["contact_events"][0]["control_frame"] != precontact_frames:
        raise ValueError("first contact changed")
    rows = {name: {"pred": 0, "gt_visible": 0, "matched_pred": 0,
                   "matched_visible_gt": 0, "class_mismatch": 0, "errors_m": []}
            for name in ("0-5m", "5-10m", "10-20m")}
    frames = 0
    for metadata, record, _ in read_evidence(Path(case_path) / "evidence"):
        if record["frame"] >= precontact_frames:
            break
        if metadata["case"]["split"] != "train" or metadata["case"]["id"] != "train-skidpad-82000-matched":
            raise ValueError("unexpected TRAIN case")
        state = record["state"]
        xy = world_to_ego(np.asarray(record["actual_cones_xyz"])[:, :2], state["x"], state["y"], state["yaw"])
        colors = np.array([{"blue": 0, "yellow": 1}.get(c, -1) for c in metadata["cone_colors"]])
        valid_gt = colors >= 0
        xy, colors = xy[valid_gt], colors[valid_gt]
        visible = visible_cones(xy, np.asarray(record["nominal_detection_calibration"]))
        pred = record["prediction"]
        p = np.asarray(pred["cones_xy"])
        pc = np.asarray(pred["cones_cls"])
        ps = np.asarray(pred["cones_score"])
        valid_pred = ((pc == 0) | (pc == 1)) & np.isfinite(p).all(axis=1) & np.isfinite(ps)
        p, pc = p[valid_pred], pc[valid_pred]
        if len(p) and len(xy):
            dist = np.linalg.norm(p[:, None, :] - xy[None, :, :], axis=2)
            pi, gi = linear_sum_assignment(np.where(dist < 0.75, dist, 1e6))
            keep = dist[pi, gi] < 0.75
            pi, gi = pi[keep], gi[keep]
        else:
            dist = np.empty((len(p), len(xy)))
            pi = gi = np.array([], dtype=int)
        pd = np.linalg.norm(p, axis=1)
        gd = np.linalg.norm(xy, axis=1)
        for name, low, high in (("0-5m", 0, 5), ("5-10m", 5, 10), ("10-20m", 10, 20)):
            row = rows[name]
            row["pred"] += int(((pd >= low) & (pd < high)).sum())
            row["gt_visible"] += int((visible & (gd >= low) & (gd < high)).sum())
            good = pc[pi] == colors[gi]
            in_pred = (pd[pi] >= low) & (pd[pi] < high)
            in_gt = visible[gi] & (gd[gi] >= low) & (gd[gi] < high)
            row["matched_pred"] += int((good & in_pred).sum())
            row["matched_visible_gt"] += int((good & in_gt).sum())
            row["class_mismatch"] += int(((~good) & in_pred).sum())
            row["errors_m"].extend(dist[pi[good & in_pred], gi[good & in_pred]].tolist())
        frames += 1
    if frames != precontact_frames:
        raise ValueError("missing precontact frames")
    for row in rows.values():
        errors = row.pop("errors_m")
        row["precision_to_any_gt"] = row["matched_pred"] / row["pred"] if row["pred"] else None
        row["frustum_recall"] = row["matched_visible_gt"] / row["gt_visible"] if row["gt_visible"] else None
        row["matched_error_p50_m"] = float(np.median(errors)) if errors else None
        row["matched_error_p95_m"] = float(np.quantile(errors, .95)) if errors else None
    print(json.dumps({"case": case_path, "frames": frames, "range": rows,
                      "limitation": "Frustum visibility ignores occlusion. The 0.75 m matching gate is not a safety margin. Correlated frames are not independent samples."}, indent=2))
