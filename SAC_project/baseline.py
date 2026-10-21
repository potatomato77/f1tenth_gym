"""Rear-axle pure pursuit with exact targets and explicit physical limits.

Run from the repository root: ``python -m SAC_project.baseline --render``.
The controller accepts a CoG pose and returns [target steering, target speed].
Its geometry consistently uses the rear axle; the environment stays CoG-based.


# 速度曲线模式，默认倍率为 1
.\.venv\Scripts\python.exe -m SAC_project.baseline --speed-mode profile --laps 3

# 固定速度模式，目标速度 2 m/s
.\.venv\Scripts\python.exe -m SAC_project.baseline --speed-mode fixed --speed 2 --laps 1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.dynamic_models import (
    F1TENTH_VEHICLE_PARAMETERS,
    VehicleParameters,
)
from f1tenth_gym.envs.env_config import (
    EnvConfig,
    ObservationConfig,
    ResetConfig,
    SimulationConfig,
    TerminationConfig,
)
from f1tenth_gym.envs.lidar import LiDARConfig
from f1tenth_gym.envs.observation import ObservationType
from f1tenth_gym.envs.reset import ReferenceLine
from f1tenth_gym.envs.track import Track
from SAC_project.reporting import error_summary, generate_plots, load_telemetry


@dataclass(frozen=True)
class PathProjection:
    point: np.ndarray
    distance: float
    signed_distance: float
    segment: int
    fraction: float


@dataclass(frozen=True)
class PursuitCommand:
    action: np.ndarray  # float32 (2,): [target steering rad, target speed m/s]
    rear_position: np.ndarray
    target: np.ndarray | None
    target_distance: float | None
    rear_error: float
    raw_steering: float
    steering_clipped: bool
    speed_clipped: bool
    mode: str  # tracking, recovery, or stop
    reason: str = ""
    reference_speed: float = 0.0  # desired speed before physical clipping
    projection: PathProjection | None = None


class PurePursuitController:
    """Stateless, forward-only controller over a closed reference polyline.

    The chosen reference line is the path to be followed by the rear axle.
    A CoG tracking error can therefore differ from the rear-axle error.
    Vehicle parameters are fixed for a run; domain randomization is disabled
    in this baseline. A future randomized task must supply appropriate limits.
    """

    def __init__(
        self,
        waypoints: np.ndarray,
        vehicle: VehicleParameters = F1TENTH_VEHICLE_PARAMETERS,
        *,
        lookahead: float = 1.2,
        speed: float = 2.0,
        speed_mode: str = "fixed",
        speed_profile: np.ndarray | None = None,
        vgain: float = 1.0,
        max_reacquire_distance: float = 3.0,
    ):
        if not math.isfinite(lookahead) or lookahead <= 0:
            raise ValueError("lookahead must be finite and positive")
        if not math.isfinite(speed) or speed < 0:
            raise ValueError("speed must be finite and non-negative")
        if speed_mode not in ("fixed", "profile"):
            raise ValueError("speed_mode must be fixed or profile")
        if not math.isfinite(vgain) or vgain <= 0:
            raise ValueError("vgain must be finite and positive")
        if (
            not math.isfinite(max_reacquire_distance)
            or max_reacquire_distance <= lookahead
        ):
            raise ValueError("max_reacquire_distance must exceed lookahead")
        required = (
            vehicle.lf,
            vehicle.lr,
            vehicle.s_min,
            vehicle.s_max,
            vehicle.v_min,
            vehicle.v_max,
        )
        if not all(math.isfinite(value) for value in required):
            raise ValueError("vehicle geometry and action limits must be finite")
        if vehicle.lf <= 0 or vehicle.lr < 0:
            raise ValueError("vehicle requires lf > 0 and lr >= 0")
        if (
            not vehicle.s_min < 0 < vehicle.s_max
            or not vehicle.v_min <= 0 < vehicle.v_max
        ):
            raise ValueError("action limits must include zero and forward motion")

        points = np.array(waypoints, dtype=np.float64, copy=True)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError("waypoints must be a finite (N, 2) array")
        profile = None
        if speed_mode == "profile":
            profile = np.array(speed_profile, dtype=np.float64, copy=True)
            if (
                profile.shape != (len(points),)
                or not np.all(np.isfinite(profile))
                or np.any(profile < 0)
            ):
                raise ValueError(
                    "profile mode requires one finite non-negative speed per waypoint"
                )
        # Remove zero-length edges, including an explicit repeated closing point.
        if len(points):
            keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-10]
            points = points[keep]
            if profile is not None:
                profile = profile[keep]
        if len(points) > 1 and np.linalg.norm(points[-1] - points[0]) <= 1e-10:
            points = points[:-1]
            if profile is not None:
                profile = profile[:-1]
        if len(points) < 3:
            raise ValueError(
                "a closed path needs at least three distinct consecutive points"
            )

        self.points = points
        self.vectors = np.roll(points, -1, axis=0) - points
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        self.lengths_squared = self.lengths**2
        self.arclength = np.r_[0.0, np.cumsum(self.lengths)]
        self.vehicle = vehicle
        self.wheelbase = vehicle.lf + vehicle.lr
        self.lookahead = float(lookahead)
        self.speed = float(speed)
        self.speed_mode = speed_mode
        self.speed_profile = profile
        self.vgain = float(vgain)
        self.max_reacquire_distance = float(max_reacquire_distance)
        # Match Gym's float32 physical bounds; reverse commands are excluded.
        self.action_low = np.array([vehicle.s_min, 0.0], dtype=np.float32)
        self.action_high = np.array([vehicle.s_max, vehicle.v_max], dtype=np.float32)

    @classmethod
    def from_track(
        cls, track: Track, *, reference: str = "raceline", speed_mode="fixed", **kwargs
    ):
        if reference not in ("raceline", "centerline"):
            raise ValueError("reference must be raceline or centerline")
        line = getattr(track, reference)
        return cls(
            np.column_stack([line.xs, line.ys]),
            speed_mode=speed_mode,
            speed_profile=line.vxs if speed_mode == "profile" else None,
            **kwargs,
        )

    def reference_speed_at(self, projection: PathProjection) -> float:
        """Interpolate the selected line's speed at the rear-axle projection.

        The closing segment interpolates back to the first waypoint. The speed
        is sampled at the vehicle's position, not at its lookahead target.
        """
        if self.speed_mode == "fixed":
            return self.speed
        index = projection.segment
        next_index = (index + 1) % len(self.points)
        speed = (1.0 - projection.fraction) * self.speed_profile[
            index
        ] + projection.fraction * self.speed_profile[next_index]
        return float(speed * self.vgain)

    def project(self, position: np.ndarray) -> PathProjection:
        """Project onto every segment, including the closing edge."""
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError("position must be a finite (2,) array")
        offsets = position - self.points
        fractions = np.clip(
            np.einsum("ij,ij->i", offsets, self.vectors) / self.lengths_squared,
            0.0,
            1.0,
        )
        projections = self.points + fractions[:, None] * self.vectors
        differences = position - projections
        distances_squared = np.einsum("ij,ij->i", differences, differences)
        index = int(np.argmin(distances_squared))
        tangent = self.vectors[index] / self.lengths[index]
        # Positive error is left of the path's direction of travel.
        side = tangent[0] * differences[index, 1] - tangent[1] * differences[index, 0]
        distance = float(np.sqrt(distances_squared[index]))
        return PathProjection(
            projections[index].copy(),
            distance,
            math.copysign(distance, float(side)),
            index,
            float(fractions[index]),
        )

    def _circle_target(self, rear: np.ndarray, projection: PathProjection):
        """Find the first exact circle intersection in cyclic path order."""
        offsets = self.points - rear
        a = self.lengths_squared
        b = 2.0 * np.einsum("ij,ij->i", offsets, self.vectors)
        c = np.einsum("ij,ij->i", offsets, offsets) - self.lookahead**2
        discriminant = b**2 - 4.0 * a * c
        tolerance = 1e-12 * np.maximum(1.0, b**2 + np.abs(4.0 * a * c))
        roots = (
            -b[:, None]
            + np.array([-1.0, 1.0]) * np.sqrt(np.maximum(discriminant, 0.0))[:, None]
        ) / (2.0 * a[:, None])
        valid = (
            (discriminant >= -tolerance)[:, None]
            & (roots >= -1e-10)
            & (roots <= 1.0 + 1e-10)
        )
        roots = np.clip(roots, 0.0, 1.0)
        start = projection.segment + projection.fraction
        progress = (np.arange(len(self.points))[:, None] + roots - start) % len(
            self.points
        )
        progress = np.where(valid, progress, np.inf)
        flat_index = int(np.argmin(progress))
        if not np.isfinite(progress.flat[flat_index]):
            return None
        segment, root = np.unravel_index(flat_index, roots.shape)
        return self.points[segment] + roots[segment, root] * self.vectors[segment]

    def _recovery_target(self, projection: PathProjection) -> np.ndarray:
        """Advance along the path when the rear axle is outside the circle."""
        start_s = (
            self.arclength[projection.segment]
            + projection.fraction * self.lengths[projection.segment]
        )
        target_s = (start_s + self.lookahead) % self.arclength[-1]
        index = int(np.searchsorted(self.arclength, target_s, side="right") - 1)
        fraction = (target_s - self.arclength[index]) / self.lengths[index]
        return self.points[index] + fraction * self.vectors[index]

    def limit_action(self, action: np.ndarray) -> np.ndarray:
        """Physical [steering, speed] limits, also reusable after residual addition."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite [steering, speed] vector")
        return np.clip(action, self.action_low, self.action_high).astype(np.float32)

    def plan(self, cog_pose: np.ndarray) -> PursuitCommand:
        """Convert the CoG pose once; use the rear axle for all control geometry."""
        pose = np.asarray(cog_pose, dtype=np.float64)
        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            raise ValueError("cog_pose must be finite [x, y, yaw]")
        heading = np.array([np.cos(pose[2]), np.sin(pose[2])])
        rear = pose[:2] - self.vehicle.lr * heading
        projection = self.project(rear)
        target = None
        mode, reason = "tracking", ""
        if projection.distance > self.max_reacquire_distance:
            reason = "path_too_far"
        elif projection.distance >= self.lookahead:
            target = self._recovery_target(projection)
            mode = "recovery"
        else:
            target = self._circle_target(rear, projection)
            if target is None:
                reason = "no_circle_intersection"

        distance, raw_steering = None, 0.0
        if target is not None:
            offset = target - rear
            distance_squared = float(offset @ offset)
            distance = float(np.sqrt(distance_squared))
            if distance_squared <= 1e-16 or float(offset @ heading) <= 1e-9:
                reason = "target_not_ahead"
            else:
                lateral = -heading[1] * offset[0] + heading[0] * offset[1]
                # Use the actual chord length, including recovery targets.
                raw_steering = float(
                    np.arctan(2.0 * self.wheelbase * lateral / distance_squared)
                )
        if reason:
            return PursuitCommand(
                self.limit_action([0.0, 0.0]),
                rear,
                target,
                distance,
                projection.signed_distance,
                0.0,
                False,
                False,
                "stop",
                reason,
                reference_speed=0.0,
                projection=projection,
            )
        reference_speed = self.reference_speed_at(projection)
        action = self.limit_action([raw_steering, reference_speed])
        return PursuitCommand(
            action,
            rear,
            target,
            distance,
            projection.signed_distance,
            raw_steering,
            bool(
                raw_steering < self.action_low[0] or raw_steering > self.action_high[0]
            ),
            bool(reference_speed > self.action_high[1]),
            mode,
            reference_speed=reference_speed,
            projection=projection,
        )


def build_config(
    *,
    map_name="Spielberg",
    laps=3,
    max_steps=100_000,
    render=False,
    reference="raceline",
) -> EnvConfig:
    """ST/RK4 at 100 Hz; discard the out lap and count only complete laps."""
    return EnvConfig(
        map_name=map_name,
        simulation_config=SimulationConfig(
            max_laps=laps, count_partial_first_lap=False
        ),
        observation_config=ObservationConfig(
            type=ObservationType.FEATURES, features=("std_state",)
        ),
        reset_config=ResetConfig(reference_line=ReferenceLine[reference.upper()]),
        lidar_config=LiDARConfig(enabled=False),
        termination_config=TerminationConfig(max_episode_steps=max_steps),
        render_enabled=render,
    )


def run_baseline(
    *,
    map_name="Spielberg",
    seed=12345,
    laps=3,
    max_steps=100_000,
    lookahead=1.2,
    speed=2.0,
    speed_mode="profile",
    vgain=1.0,
    max_reacquire_distance=3.0,
    reference="raceline",
    render=False,
    make_plots=True,
    output_dir: Path,
) -> dict:
    """Run one episode; write post-step states and the commands that produced them."""
    cfg = build_config(
        map_name=map_name,
        laps=laps,
        max_steps=max_steps,
        render=render,
        reference=reference,
    )
    env = gym.make(
        "f1tenth_gym:f1tenth-v0", config=cfg, render_mode="human" if render else None
    )
    try:
        controller = PurePursuitController.from_track(
            env.unwrapped.track,
            reference=reference,
            vehicle=cfg.params,
            lookahead=lookahead,
            speed=speed,
            speed_mode=speed_mode,
            vgain=vgain,
            max_reacquire_distance=max_reacquire_distance,
        )
        obs, info = env.reset(seed=seed)
        state = obs["agent_0"]["std_state"]
        command = controller.plan(state[[0, 1, 4]])
        if render:
            getattr(env.unwrapped.track, reference).render_waypoints(
                env.unwrapped.renderer
            )
            marker = None

            def draw_target(renderer):
                nonlocal marker
                if command.target is not None:
                    points = command.target[None, :]
                    if marker is None:
                        marker = renderer.get_points_renderer(
                            points, color=(228, 26, 28), size=10
                        )
                    else:
                        marker.update(points)

            env.unwrapped.add_render_callback(draw_target)
            env.render()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save the exact cleaned, closed reference used by this controller.
        with (output_dir / "reference_path.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["s", "x", "y", "reference_speed"])
            for index, arclength in enumerate(controller.arclength):
                vertex = index % len(controller.points)
                vertex_speed = (
                    controller.speed
                    if speed_mode == "fixed"
                    else controller.speed_profile[vertex] * controller.vgain
                )
                writer.writerow([arclength, *controller.points[vertex], vertex_speed])
        fields = [
            "step",
            "sim_time",
            "cog_x",
            "cog_y",
            "yaw",
            "rear_x",
            "rear_y",
            "actual_speed",
            "actual_steering",
            "rear_error",
            "cog_error",
            "reference_x",
            "reference_y",
            "position_error_norm",
            "cog_position_error_norm",
            "heading_error",
            "target_x",
            "target_y",
            "target_distance",
            "command_steering",
            "command_speed",
            "reference_speed",
            "speed_error",
            "reference_speed_error",
            "steering_error",
            "raw_steering",
            "steering_clipped",
            "speed_clipped",
            "mode",
            "collision",
            "lap_count",
        ]
        lap_times = []
        steering_clips = speed_clips = steps = completed_laps = 0
        collided = terminated = truncated = False
        stop_reason = ""
        with (output_dir / "trajectory.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for steps in range(1, max_steps + 1):
                applied = command
                obs, _, terminated, truncated, info = env.step(applied.action[None, :])
                state = obs["agent_0"]["std_state"]
                command = controller.plan(state[[0, 1, 4]])
                cog_projection = controller.project(state[:2])
                rear_projection = command.projection
                tangent = controller.vectors[rear_projection.segment]
                reference_yaw = math.atan2(tangent[1], tangent[0])
                heading_error = (float(state[4]) - reference_yaw + np.pi) % (
                    2 * np.pi
                ) - np.pi
                steering_clips += int(applied.steering_clipped)
                speed_clips += int(applied.speed_clipped)
                collided |= bool(info["collisions"][0])
                lap_count = int(info["lap_counts"][0])
                if lap_count > completed_laps:
                    lap_times.append(float(info["lap_times"][0]))
                    completed_laps = lap_count
                writer.writerow(
                    {
                        "step": steps,
                        "sim_time": float(info["sim_time"]),
                        "cog_x": float(state[0]),
                        "cog_y": float(state[1]),
                        "yaw": float(state[4]),
                        "rear_x": command.rear_position[0],
                        "rear_y": command.rear_position[1],
                        "actual_speed": float(state[3]),
                        "actual_steering": float(state[2]),
                        "rear_error": command.rear_error,
                        "cog_error": cog_projection.signed_distance,
                        "reference_x": rear_projection.point[0],
                        "reference_y": rear_projection.point[1],
                        "position_error_norm": rear_projection.distance,
                        "cog_position_error_norm": cog_projection.distance,
                        "heading_error": heading_error,
                        "target_x": (
                            applied.target[0] if applied.target is not None else None
                        ),
                        "target_y": (
                            applied.target[1] if applied.target is not None else None
                        ),
                        "target_distance": applied.target_distance,
                        "command_steering": float(applied.action[0]),
                        "command_speed": float(applied.action[1]),
                        "reference_speed": applied.reference_speed,
                        "speed_error": float(state[3]) - float(applied.action[1]),
                        "reference_speed_error": (
                            float(state[3]) - applied.reference_speed
                        ),
                        "steering_error": float(state[2]) - float(applied.action[0]),
                        "raw_steering": applied.raw_steering,
                        "steering_clipped": applied.steering_clipped,
                        "speed_clipped": applied.speed_clipped,
                        "mode": applied.mode,
                        "collision": bool(info["collisions"][0]),
                        "lap_count": lap_count,
                    }
                )
                if render:
                    env.render()
                if applied.mode == "stop" or command.mode == "stop":
                    stop_reason = (
                        applied.reason if applied.mode == "stop" else command.reason
                    )
                if terminated or truncated or stop_reason:
                    break

        telemetry = load_telemetry(output_dir / "trajectory.csv")
        summary = {
            "success": completed_laps == laps and not collided and not stop_reason,
            "map": str(map_name),
            "seed": seed,
            "reference": reference,
            "tracking_point": "rear_axle",
            "speed_mode": speed_mode,
            "speed_mps": speed if speed_mode == "fixed" else None,
            "vgain": vgain,
            "speed_profile_source": (
                f"{reference}.vxs" if speed_mode == "profile" else None
            ),
            "speed_profile_interpolation": (
                "linear at rear-axle projection" if speed_mode == "profile" else None
            ),
            "lookahead_m": lookahead,
            "max_reacquire_distance_m": max_reacquire_distance,
            "wheelbase_m": controller.wheelbase,
            "lr_m": cfg.params.lr,
            "timestep_s": cfg.simulation_config.timestep,
            "max_steps": max_steps,
            "requested_laps": laps,
            "completed_laps": completed_laps,
            "full_lap_times_s": lap_times,
            "sim_time_including_out_lap_s": float(info["sim_time"]),
            "steps": steps,
            "collided": collided,
            "terminated": terminated,
            "truncated": truncated,
            "stop_reason": stop_reason,
            "rear_error": error_summary(telemetry["rear_error"], "m"),
            "cog_error": error_summary(telemetry["cog_error"], "m"),
            "position_error_norm": error_summary(telemetry["position_error_norm"], "m"),
            "speed_error": error_summary(telemetry["speed_error"], "mps"),
            "reference_speed_error": error_summary(
                telemetry["reference_speed_error"], "mps"
            ),
            "heading_error": error_summary(telemetry["heading_error"], "rad"),
            "steering_error": error_summary(telemetry["steering_error"], "rad"),
            "reference_speed_range_mps": [
                float(np.min(telemetry["reference_speed"])),
                float(np.max(telemetry["reference_speed"])),
            ],
            "command_speed_range_mps": [
                float(np.min(telemetry["command_speed"])),
                float(np.max(telemetry["command_speed"])),
            ],
            "actual_speed_mean_mps": float(np.mean(telemetry["actual_speed"])),
            "actual_speed_max_mps": float(np.max(telemetry["actual_speed"])),
            "error_scope": "all post-step samples, including out lap; reference is a polyline",
            "metric_definitions": {
                "position_error_norm": "Euclidean distance from rear axle to its closest reference-path point; not lookahead distance",
                "speed_error": "post-step actual speed minus the clipped target applied over that step",
                "reference_speed_error": "post-step actual speed minus the unclipped desired speed applied over that step",
                "heading_error": "wrapped body yaw minus reference-segment yaw at the post-step rear-axle projection",
                "steering_error": "post-step actual steering minus the clipped steering target applied over that step",
                "time_alignment": "row time is the interval end; commands are held targets from the interval start",
            },
            "steering_clip_fraction": steering_clips / steps,
            "speed_clip_fraction": speed_clips / steps,
            "plots": [],
        }
        if make_plots:
            summary["plots"] = generate_plots(output_dir, telemetry, summary)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return summary
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", dest="map_name", default="Spielberg")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--laps", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--lookahead", type=float, default=1.2)
    parser.add_argument(
        "--speed-mode",
        choices=("fixed", "profile"),
        default="profile",
        help="profile interpolates the selected reference line's vxs",
    )
    parser.add_argument(
        "--speed", type=float, default=2.0, help="fixed-mode speed [m/s]"
    )
    parser.add_argument(
        "--vgain", type=float, default=1.0, help="profile speed multiplier"
    )
    parser.add_argument("--max-reacquire-distance", type=float, default=3.0)
    parser.add_argument(
        "--reference", choices=("raceline", "centerline"), default="raceline"
    )
    parser.add_argument(
        "--render", action="store_true", help="open the native desktop renderer"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--no-plots",
        dest="make_plots",
        action="store_false",
        help="write CSV and JSON without generating PNG reports",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        args.output_dir = (
            Path(__file__).parent / "runs" / f"baseline_{stamp}_seed{args.seed}"
        )
    summary = run_baseline(**vars(args))
    print(json.dumps(summary, indent=2))
    print(f"Results: {args.output_dir.resolve()}")
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
