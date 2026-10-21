"""Ordered state-and-map features for a rear-axle residual controller."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from SAC_project.baseline import PathProjection, PurePursuitController, PursuitCommand


@dataclass(frozen=True)
class FeatureConfig:
    preview_distances_m: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    yaw_rate_scale_radps: float = 10.0

    def __post_init__(self):
        distances = tuple(float(value) for value in self.preview_distances_m)
        if (
            not distances
            or not all(math.isfinite(value) and value > 0 for value in distances)
            or any(b <= a for a, b in zip(distances, distances[1:]))
        ):
            raise ValueError(
                "preview distances must be finite, positive and increasing"
            )
        if (
            not math.isfinite(self.yaw_rate_scale_radps)
            or self.yaw_rate_scale_radps <= 0
        ):
            raise ValueError("yaw-rate scale must be finite and positive")
        object.__setattr__(self, "preview_distances_m", distances)


@dataclass(frozen=True)
class FeatureFrame:
    observation: np.ndarray
    path_s_m: float
    heading_error_rad: float
    clipped_features: int


def projection_arclength(controller: PurePursuitController, projection: PathProjection):
    return float(
        controller.arclength[projection.segment]
        + projection.fraction * controller.lengths[projection.segment]
    )


def sample_reference(controller: PurePursuitController, arclength: float):
    """Sample the same closed polyline and speed interpolation as the baseline."""
    arclength = float(arclength % controller.arclength[-1])
    index = int(np.searchsorted(controller.arclength, arclength, side="right") - 1)
    fraction = (arclength - controller.arclength[index]) / controller.lengths[index]
    point = controller.points[index] + fraction * controller.vectors[index]
    projection = PathProjection(point, 0.0, 0.0, index, float(fraction))
    speed = np.clip(
        controller.reference_speed_at(projection),
        controller.action_low[1],
        controller.action_high[1],
    )
    return point, float(speed)


class FeatureExtractor:
    """Default layout: 13 state/history features + 4 * 3 preview features.

    Bounds are finite [-1, 1]. Values beyond the declared physical scales are
    clipped for the policy and counted; raw tracking metrics stay in info.
    """

    def __init__(self, controller: PurePursuitController, config=FeatureConfig()):
        self.controller = controller
        self.config = config
        vehicle = controller.vehicle
        self.speed_scale = max(abs(vehicle.v_min), abs(vehicle.v_max))
        self.steering_scale = max(abs(vehicle.s_min), abs(vehicle.s_max))
        self.lateral_scale = controller.max_reacquire_distance
        self.preview_scale = max(config.preview_distances_m) + self.lateral_scale
        names = [
            "rear_lateral_error",
            "sin_heading_error",
            "cos_heading_error",
            "actual_speed",
            "target_speed",
            "actual_steering",
            "yaw_rate",
            "sin_slip_angle",
            "cos_slip_angle",
            "base_steering",
            "previous_residual_z",
            "previous_command_steering",
            "previous_command_speed",
        ]
        for distance in config.preview_distances_m:
            names.extend(
                [
                    f"preview_{distance:g}m_local_x",
                    f"preview_{distance:g}m_local_y",
                    f"preview_{distance:g}m_target_speed",
                ]
            )
        self.names = tuple(names)

    def observe(
        self,
        state: np.ndarray,
        command: PursuitCommand,
        previous_z: float,
        previous_action: np.ndarray,
    ) -> FeatureFrame:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (7,) or not np.all(np.isfinite(state)):
            raise ValueError("features require a finite standard state of shape (7,)")
        projection = command.projection
        if projection is None:
            raise ValueError("baseline command must include its rear-axle projection")
        tangent = self.controller.vectors[projection.segment]
        heading_error = float(
            (state[4] - math.atan2(tangent[1], tangent[0]) + np.pi) % (2 * np.pi)
            - np.pi
        )
        path_s = projection_arclength(self.controller, projection)
        values = [
            command.rear_error / self.lateral_scale,
            math.sin(heading_error),
            math.cos(heading_error),
            state[3] / self.speed_scale,
            command.action[1] / self.speed_scale,
            state[2] / self.steering_scale,
            state[5] / self.config.yaw_rate_scale_radps,
            math.sin(state[6]),
            math.cos(state[6]),
            command.action[0] / self.steering_scale,
            previous_z,
            previous_action[0] / self.steering_scale,
            previous_action[1] / self.speed_scale,
        ]
        cosine, sine = math.cos(state[4]), math.sin(state[4])
        for distance in self.config.preview_distances_m:
            point, speed = sample_reference(self.controller, path_s + distance)
            dx, dy = point - command.rear_position
            values.extend(
                [
                    (cosine * dx + sine * dy) / self.preview_scale,
                    (-sine * dx + cosine * dy) / self.preview_scale,
                    speed / self.speed_scale,
                ]
            )
        values = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("non-finite residual observation")
        return FeatureFrame(
            np.clip(values, -1.0, 1.0).astype(np.float32),
            path_s,
            heading_error,
            int(np.count_nonzero(np.abs(values) > 1.0)),
        )
