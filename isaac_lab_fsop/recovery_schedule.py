"""Frozen phase recovery schedule and its control-state transitions."""

from dataclasses import dataclass

from fsop.sim.route_commands import ROUTE_DESIRE, RouteCommands


@dataclass(frozen=True)
class PhaseSchedule:
    mode: str
    phase: str
    lateral_threshold_m: float
    learner_window_seconds: float
    transition_lead_m: float

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Phase schedule must be an object")
        try:
            schedule = cls(**value)
        except TypeError as exc:
            raise ValueError("Invalid phase schedule fields") from exc
        if (schedule.mode != "phase_recovery_v1" or schedule.phase not in ("entry", "left_transition")
                or schedule.lateral_threshold_m != 0.35 or schedule.learner_window_seconds != 6
                or schedule.transition_lead_m != 12):
            raise ValueError("Phase schedule differs from the frozen protocol")
        return schedule


def first_left_decision(track, mission):
    decisions = RouteCommands.for_mission(track, mission, 20).decisions
    left = [s for s, desire in decisions if desire == ROUTE_DESIRE["left"]]
    if not left:
        raise ValueError("Track has no left route decision")
    return left[0]


class PhaseRecovery:
    def __init__(self, schedule, first_left_m):
        self.schedule = schedule
        self.activation_progress_m = first_left_m - schedule.transition_lead_m
        self.state = "learner" if schedule.phase == "entry" else "teacher_prefix"
        self.activation_tick = 0 if schedule.phase == "entry" else None
        self.activation_reason = "reset" if schedule.phase == "entry" else None
        self.takeover_tick = None
        self.takeover_reason = None

    def advance(self, tick, progress, lateral, prefix_clean=True):
        if self.state == "teacher_prefix" and progress >= self.activation_progress_m:
            self.state = "learner"
            self.activation_tick = tick
            self.activation_reason = "first_left_decision_lead"
            if not prefix_clean:
                self.state = "expert_recovery"
                self.takeover_tick = tick
                self.takeover_reason = "invalid_teacher_prefix"
        if self.state == "learner":
            if abs(lateral) >= self.schedule.lateral_threshold_m:
                self.state = "expert_recovery"
                self.takeover_tick = tick
                self.takeover_reason = "lateral_threshold"
            elif tick - self.activation_tick >= self.schedule.learner_window_seconds * 20:
                self.state = "expert_recovery"
                self.takeover_tick = tick
                self.takeover_reason = "no_corrective_trigger"
        return self.state


def phase_proposal(policy, obs, env, teacher):
    proposed = policy(obs, env)[:2]
    if teacher:
        reset = getattr(getattr(policy, "ctrl", None), "reset", None)
        if callable(reset):
            reset()
    return proposed


def phase_control(policy, expert, obs, env, state):
    """Observe every tick and execute the actor selected by the schedule."""
    teacher = state != "learner"
    proposed = phase_proposal(policy, obs, env, teacher)
    return expert.act() if teacher else proposed
