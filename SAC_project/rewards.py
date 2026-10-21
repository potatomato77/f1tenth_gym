"""A decomposed tracking reward; independent of the simulator's default reward."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np


@dataclass(frozen=True)
class TrackingRewardConfig:
    progress_weight: float = 1.0
    lateral_weight: float = 4.0
    heading_weight: float = 1.0
    residual_weight: float = 0.1
    smoothness_weight: float = 0.05
    failure_cost: float = 20.0
    lateral_scale_m: float = 0.5
    heading_scale_rad: float = 0.5

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field.name} must be finite and non-negative")
        if self.lateral_scale_m <= 0 or self.heading_scale_rad <= 0:
            raise ValueError("reward error scales must be positive")


@dataclass(frozen=True)
class RewardResult:
    total: float
    terms: dict[str, float]
    raw_progress_m: float
    used_progress_m: float


def tracking_reward(
    *,
    previous_s: float,
    current_s: float,
    path_length: float,
    dt: float,
    speed_limit: float,
    lateral_error: float,
    heading_error: float,
    z: float,
    previous_z: float,
    failed: bool,
    config: TrackingRewardConfig = TrackingRewardConfig(),
) -> RewardResult:
    """Unwrap path progress and charge running costs per physical second.

    A nearest-segment jump cannot earn an arbitrary progress bonus: progress
    is bounded to twice the vehicle speed limit times dt, with both raw and
    used values retained. Collision and tracker failure share one terminal cost.
    """
    if not (dt > 0 and path_length > 0 and speed_limit > 0):
        raise ValueError("dt, path length and speed limit must be positive")
    progress = float(
        (current_s - previous_s + path_length / 2) % path_length - path_length / 2
    )
    used = float(np.clip(progress, -2 * speed_limit * dt, 2 * speed_limit * dt))
    terms = {
        "progress": config.progress_weight * used,
        "lateral": -dt
        * config.lateral_weight
        * (lateral_error / config.lateral_scale_m) ** 2,
        "heading": -dt
        * config.heading_weight
        * (heading_error / config.heading_scale_rad) ** 2,
        "residual": -dt * config.residual_weight * z**2,
        "smoothness": -dt * config.smoothness_weight * (z - previous_z) ** 2,
        "failure": -config.failure_cost if failed else 0.0,
    }
    return RewardResult(float(sum(terms.values())), terms, progress, used)
