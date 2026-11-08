from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union


def layout_sha256(layout):
    return hashlib.sha256(
        json.dumps(layout, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def protected_area(env):
    track = env.track
    points = np.asarray(track.centerline, dtype=float)
    if len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Track centerline must contain finite points")
    if track.closed:
        points = np.vstack((points, points[0]))
    width = float(max(np.max(track.width_left), np.max(track.width_right))) + 0.8
    areas = [LineString(points).buffer(width, cap_style="round", join_style="round")]
    if hasattr(env, "mission"):
        route = np.array(
            [
                env.mission.point_at(float(s))
                for s in np.linspace(0, env.mission.s_end, max(2, int(env.mission.s_end / 0.5) + 1))
            ]
        )
        if route.ndim == 2 and route.shape[1] >= 2:
            areas.append(LineString(route[:, :2]).buffer(width, cap_style="round", join_style="round"))
    cones = np.asarray(env.cones_xy, dtype=float)
    if cones.ndim != 2 or cones.shape[1] != 2 or not np.isfinite(cones).all():
        raise ValueError("Cone positions must be finite XY pairs")
    areas.extend(Point(*p).buffer(0.8) for p in cones)
    return unary_union(areas)


def footprint(x, y, yaw, size):
    shape = box(-size[0] / 2, -size[1] / 2, size[0] / 2, size[1] / 2)
    return translate(rotate(shape, yaw, origin=(0, 0), use_radians=True), x, y)


def build_layout(env):
    """Return JSON-compatible exact placements. No draw touches env.rng."""
    protected = protected_area(env)
    seed_bytes = json.dumps(
        {"seed": int(env.seed), "track": env.track.id, "variant": "visual_stress_v1"}, sort_keys=True
    ).encode()
    seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    route = np.asarray(env.track.centerline, dtype=float)
    widths = max(float(np.max(env.track.width_left)), float(np.max(env.track.width_right)))
    objects = []
    occupied = []
    kinds = [
        ("person_business", (1.4, 1.4)),
        ("person_worker", (1.4, 1.4)),
        ("person_police", (1.4, 1.4)),
        ("barrel_blue", (0.8, 0.8)),
        ("barrel_yellow", (0.8, 0.8)),
        ("tires", (1.1, 1.1)),
        ("worktable", (2.0, 1.0)),
        ("equipment", (2.1, 1.0)),
        ("barrier", (2.5, 0.65)),
        ("wall", (3.5, 0.5)),
    ]
    for i in np.linspace(0, len(route) - 1, min(48, len(route)), dtype=int):
        p = route[i]
        d = route[(i + 1) % len(route)] - route[max(0, i - 1)]
        if np.linalg.norm(d) < 1e-8:
            continue
        heading = math.atan2(float(d[1]), float(d[0]))
        normal = np.array([-math.sin(heading), math.cos(heading)])
        for side in (-1, 1):
            for n in range(2):
                kind, size = kinds[(len(objects) + n + int(i)) % len(kinds)]
                offset = widths + 2.4 + n * 1.8 + float(rng.uniform(0, 1.0))
                xy = (
                    p
                    + side * offset * normal
                    + rng.uniform(-0.6, 0.6) * np.array([math.cos(heading), math.sin(heading)])
                )
                yaw = heading + (
                    math.pi / 2
                    if kind.startswith("person_") and side > 0
                    else -math.pi / 2
                    if kind.startswith("person_")
                    else 0
                )
                shape = footprint(float(xy[0]), float(xy[1]), yaw, size)
                clearance = shape.distance(protected)
                if clearance < 0.15 or any(shape.distance(other) < 0.2 for other in occupied):
                    continue
                occupied.append(shape)
                objects.append(
                    {
                        "kind": kind,
                        "position": [float(xy[0]), float(xy[1]), 0.0],
                        "yaw_rad": float(yaw),
                        "footprint_size": list(size),
                        "footprint_xy": [[float(x), float(y)] for x, y in list(shape.exterior.coords)[:-1]],
                        "clearance_m": float(clearance),
                    }
                )
    if not objects:
        raise ValueError("No stress objects fit outside the protected corridor")
    surface = []
    for kind, count in (("tar", 28), ("rubber", 18), ("paint", 12)):
        added = 0
        for _ in range(count * 30):
            if added == count:
                break
            k = int(rng.integers(len(route)))
            p = route[k] + rng.uniform(-widths - 7, widths + 7, size=2)
            direction = route[(k + 1) % len(route)] - route[max(0, k - 1)]
            heading = math.atan2(float(direction[1]), float(direction[0]))
            angle = heading + float(rng.uniform(-0.3, 0.3))
            if kind == "tar":
                size = [float(rng.uniform(2, 5)), float(rng.uniform(0.012, 0.026))]
            elif kind == "rubber":
                size = [float(rng.uniform(4, 9)), float(rng.uniform(0.035, 0.065))]
            else:
                size = [float(rng.uniform(1.8, 4)), float(rng.uniform(0.045, 0.075))]
            surface.append(
                {
                    "kind": kind,
                    "position": [float(p[0]), float(p[1])],
                    "yaw_rad": angle,
                    "size": size,
                    "variation": [float(rng.uniform(0.75, 1.0)) for _ in range(12)],
                }
            )
            added += 1
    bounds_points = np.vstack((route[:, :2], np.asarray(env.cones_xy, dtype=float)))
    low = bounds_points.min(axis=0) - 30.0
    high = bounds_points.max(axis=0) + 30.0
    fields = []
    for _ in range(max(1, int(np.prod(high - low) / 180))):
        fields.append(
            {
                "center": [float(v) for v in rng.uniform(low, high)],
                "radius_m": [float(v) for v in rng.uniform(4.0, 10.0, size=2)],
                "gain": float(rng.uniform(-0.20, 0.20)),
            }
        )
    weather = {
        "bounds_m": [low.tolist(), high.tolist()],
        "spacing_m": 1.5,
        "edge_fade_m": 12.0,
        "base_gain": 4.0,
        "gain_limits": [0.78, 1.22],
        "fields": fields,
    }
    return {
        "schema": "visual_stress_layout_v1",
        "seed": seed,
        "corridor_buffer_m": widths + 0.8,
        "cone_buffer_m": 0.8,
        "objects": objects,
        "surface": surface,
        "weather": weather,
    }
