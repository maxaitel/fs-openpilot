"""Whitelist adapters for the provided decoded detections and route inputs."""
from planner import ConeClass, ConeDetection, Motion, Observation, RouteCommand, RouteEvent

CHANNELS = {1: RouteCommand.LEFT, 2: RouteCommand.RIGHT, 7: RouteCommand.EXIT}


def detections(xy, classes, scores):
    if not len(xy) == len(classes) == len(scores):
        raise ValueError("cone arrays have different lengths")
    return tuple(ConeDetection(float(p[0]), float(p[1]), ConeClass(int(c)) if int(c) in range(4) else ConeClass.UNKNOWN, float(s))
                 for p, c, s in zip(xy, classes, scores))


def recorded_pilot_observation(record, threshold):
    state, predicted = record["state"], record["prediction"]
    now = float(state["t"])
    events = tuple(RouteEvent(now, command) for channel, command in CHANNELS.items()
                   if record["commands"]["applied"][channel] >= threshold)
    return Observation(now, detections(predicted["cones_xy"], predicted["cones_cls"], predicted["cones_score"]),
                       Motion(float(state["vx"]), float(state["vy"]), float(state["r"])), events)


def pooled_record_observation(record, bin_seconds, threshold):
    history = record["desire_history"]
    if len(history) != 25 or any(len(row) != 8 for row in history):
        raise ValueError("expected the provided 25 by 8 pooled desire history")
    now = float(record["time_s"])
    events = tuple(RouteEvent(round(now - (len(history) - 1 - i) * bin_seconds, 9), command, bin_seconds)
                   for i, row in enumerate(history) for channel, command in CHANNELS.items() if row[channel] >= threshold)
    return Observation(now, detections(record["cones_xy"], record["cones_cls"], record["cones_score"]),
                       Motion(*(float(x) for x in record["vx_vy_yaw_rate"])), events)
