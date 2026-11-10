"""Design-only signatures for the direct-native-actuator experiment.

No training or evaluation implementation is included.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class MeasuredVehicle:
    vx_mps: float
    vy_mps: float
    yaw_rate_radps: float


@dataclass(frozen=True)
class ActuatorRequest:
    steer_rad: float
    accel_mps2: float


@dataclass(frozen=True)
class CommandContract:
    kind: Literal["direct_native_actuator_v1"]
    initializer_sha256: str
    native_manifest_sha256: str
    source_sha256: str
    recipe_sha256: str
    feature_shape: tuple[int, int] = (9, 512)
    route_shape: tuple[int, int] = (25, 8)
    latent_module: str = "base.policy.p_node_linear_20"
    latent_width: int = 256
    vehicle_order: tuple[str, ...] = ("vx", "vy", "r")
    vehicle_scales: tuple[float, ...] = (10.0, 2.0, 1.0)
    output_scales: tuple[float, float] = (0.1, 2.0)
    loss_scales: tuple[float, float] = (0.02, 0.5)
    retained_parameters: int = 31_154_802
    command_parameters: int = 33_538


@dataclass(frozen=True)
class _CommandBatch:
    """Private tensors validated from qualified same-tick native observations.

    latent is [B,256], vehicle is [B,3], and teacher_request is [B,2].
    All are FP32. Frozen policy consumes exact image and route history.
    The batch contains no global pose, lateral error, progress, or future plan.
    """
    latent: torch.Tensor
    vehicle: torch.Tensor
    teacher_request: torch.Tensor


class _CommandHead(nn.Module):
    """259 to 128 to 2, with SiLU between affine layers."""

    def __init__(self, contract: CommandContract):
        raise NotImplementedError("Declare exactly four trainable affine tensors.")

    def forward(self, batch: _CommandBatch) -> torch.Tensor:
        """Return [B,2] raw physical requests before limits and mission intervention."""
        raise NotImplementedError("Scale vehicle inputs, concatenate with detached latent, apply head, decode units.")


class NativeActuatorPolicy:
    """Own one full frozen model, the command head, and per-episode temporal queues."""

    @classmethod
    def load(cls, package: Path, *, device: str) -> "NativeActuatorPolicy":
        raise NotImplementedError("Validate distinct command contract and strict tensor inventory.")

    def reset(self) -> None:
        raise NotImplementedError("Reset TemporalState and evidence clock. Do not mutate weights.")

    def __call__(self, obs: Any, env: Any) -> tuple[float, float, dict]:
        """Use only camera, route pulse, traffic, and three declared inputs in the head.

        Existing mission intervention has its separate documented progress access.
        Return raw, bounded, and mission-applied requests in evidence.
        Preserve original plan and cone diagnostics as frozen evidence.
        """
        raise NotImplementedError("Run exact frozen encode and temporal policy, capture latent once, predict, intervene once.")


def _native_batch(recipe: Path, split: Literal["train", "val"], anchors: np.ndarray) -> _CommandBatch:
    """Verify membership once at the boundary and align image and raw request by tick."""
    raise NotImplementedError("Use NativeReplay qualification and fresh_history without a feature cache.")


def _command_loss(requests: torch.Tensor, batch: _CommandBatch, contract: CommandContract) -> torch.Tensor:
    """Mean SmoothL1 over two dimensionless command errors with beta one."""
    raise NotImplementedError("Compare raw requests to controls[tick], never to plan-derived actions.")


def audit(recipe: Path, out: Path) -> Path:
    """Return a source-bound CPU interface audit receipt or fail closed."""
    raise NotImplementedError("Verify identity, exact frozen parity, tick semantics, and observed inputs.")


def train(recipe: Path, out: Path) -> Path:
    """Return a full candidate package only after the fixed offline qualification."""
    raise NotImplementedError("Require audit receipt, run one fixed recipe, retain every checkpoint.")


def qualify(recipe: Path, package: Path, out: Path) -> Path:
    """Return physical qualification evidence for the already selected package."""
    raise NotImplementedError("Run CPU pilot, exposed development repeats, and both video audits.")
