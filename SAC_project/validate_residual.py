"""Verify zero-residual equivalence and random-action interfaces, without training.

Run ``python -m SAC_project.validate_residual --speed-mode profile --laps 3``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from SAC_project.baseline import PurePursuitController, build_config
from SAC_project.residual_env import make_residual_env


STATE_NAMES = (
    "x_m",
    "y_m",
    "steering_rad",
    "speed_mps",
    "yaw_rad",
    "yaw_rate_radps",
    "slip_rad",
)
REWARD_NAMES = ("progress", "lateral", "heading", "residual", "smoothness", "failure")


def plot_equivalence(output_dir: Path, equivalent: bool):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    data = np.genfromtxt(
        output_dir / "zero_residual.csv", delimiter=",", names=True, ndmin=1
    )
    figure = Figure(figsize=(13, 9), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    axes[0, 0].plot(data["base_x"], data["base_y"], label="Pure pursuit")
    axes[0, 0].plot(data["residual_x"], data["residual_y"], "--", label="Zero residual")
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].set(
        xlabel="World X [m]", ylabel="World Y [m]", title="Actual CoG trajectories"
    )
    axes[0, 1].plot(data["sim_time"], data["base_speed"], label="Pure pursuit")
    axes[0, 1].plot(
        data["sim_time"], data["residual_speed"], "--", label="Zero residual"
    )
    axes[0, 1].set(
        xlabel="Simulation time [s]", ylabel="Speed [m/s]", title="Actual speeds"
    )
    axes[1, 0].plot(
        data["sim_time"], data["position_difference_norm"], label="Position difference"
    )
    axes[1, 0].set(
        xlabel="Simulation time [s]",
        ylabel="Difference norm [m]",
        title="CoG position difference",
    )
    axes[1, 1].plot(
        data["sim_time"],
        data["steering_command_difference"],
        label="Absolute command difference",
    )
    axes[1, 1].set(
        xlabel="Simulation time [s]",
        ylabel="Difference [rad]",
        title="Steering command difference",
    )
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=9)
    figure.suptitle(f"Zero-residual equivalence: {'PASS' if equivalent else 'FAIL'}")
    figure.savefig(output_dir / "zero_residual_equivalence.png", dpi=160)
    figure.clear()


def validate(
    *,
    output_dir: Path,
    map_name="Spielberg",
    seed=12345,
    speed_mode="profile",
    speed=2.0,
    vgain=1.0,
    lookahead=1.2,
    laps=3,
    max_steps=100_000,
    random_steps=256,
    tolerance=1e-6,
) -> dict:
    """Run two independent native simulators with identical configs and seeds."""
    from stable_baselines3.common.env_checker import check_env

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_config(map_name=map_name, laps=laps, max_steps=max_steps)
    base = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
    residual = None
    try:
        # Share immutable map geometry, not simulator state or random generators.
        residual = make_residual_env(
            env_config=cfg.with_updates(map_name=base.unwrapped.track),
            speed_mode=speed_mode,
            speed=speed,
            vgain=vgain,
            lookahead=lookahead,
        )
        check_env(residual, warn=True)
        contract = residual.describe()
        controller = PurePursuitController.from_track(
            base.unwrapped.track,
            vehicle=cfg.params,
            speed_mode=speed_mode,
            speed=speed,
            vgain=vgain,
            lookahead=lookahead,
        )
        observation, base_info = base.reset(seed=seed)
        vector, info = residual.reset(seed=seed)
        initial = observation["agent_0"]["std_state"].copy()
        initial_equal = np.array_equal(initial, info["residual"]["std_state"])
        maximum_state_error = np.zeros(7)
        maximum_action_error = np.zeros(2)
        first_mismatch = None
        all_states_exact = initial_equal
        equivalent = initial_equal and residual.observation_space.contains(vector)
        command = controller.plan(initial[[0, 1, 4]])
        full_lap_times = []
        fields = [
            "step",
            "sim_time",
            "base_x",
            "base_y",
            "residual_x",
            "residual_y",
            "base_speed",
            "residual_speed",
            "position_difference_norm",
            "steering_command_difference",
            "speed_command_difference",
            "maximum_state_difference",
            "base_rear_error",
            "residual_rear_error",
            "residual_reward",
            "lap_count",
            "collision",
        ] + [f"reward_{name}" for name in REWARD_NAMES]
        with (output_dir / "zero_residual.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for step in range(1, max_steps + 1):
                applied = command
                observation, _, terminated, truncated, base_info = base.step(
                    applied.action[None, :]
                )
                vector, reward, r_terminated, r_truncated, info = residual.step(
                    np.zeros(1, dtype=np.float32)
                )
                state = observation["agent_0"]["std_state"]
                command = controller.plan(state[[0, 1, 4]])
                diagnostic = info["residual"]
                state_error = np.abs(
                    state.astype(float) - diagnostic["std_state"].astype(float)
                )
                action_error = np.abs(
                    applied.action.astype(float)
                    - diagnostic["physical_action"].astype(float)
                )
                maximum_state_error = np.maximum(maximum_state_error, state_error)
                maximum_action_error = np.maximum(maximum_action_error, action_error)
                all_states_exact &= np.array_equal(state, diagnostic["std_state"])
                stopped = applied.mode == "stop" or command.mode == "stop"
                matches = (
                    np.max(state_error) <= tolerance
                    and np.max(action_error) <= tolerance
                    and (bool(terminated or stopped), bool(truncated))
                    == (r_terminated, r_truncated)
                    and np.array_equal(base_info["collisions"], info["collisions"])
                    and np.array_equal(base_info["lap_counts"], info["lap_counts"])
                    and np.allclose(
                        base_info["lap_times"],
                        info["lap_times"],
                        atol=tolerance,
                        rtol=0,
                    )
                    and base_info["sim_time"] == info["sim_time"]
                    and residual.observation_space.contains(vector)
                    and np.isfinite(reward)
                )
                equivalent &= bool(matches)
                if int(base_info["lap_counts"][0]) > len(full_lap_times):
                    full_lap_times.append(float(base_info["lap_times"][0]))
                writer.writerow(
                    dict(
                        zip(
                            fields,
                            [
                                step,
                                float(base_info["sim_time"]),
                                state[0],
                                state[1],
                                *diagnostic["std_state"][:2],
                                state[3],
                                diagnostic["std_state"][3],
                                float(np.linalg.norm(state_error[:2])),
                                *action_error,
                                float(state_error.max()),
                                command.rear_error,
                                diagnostic["rear_lateral_error_m"],
                                reward,
                                int(base_info["lap_counts"][0]),
                                bool(base_info["collisions"][0]),
                                *[
                                    diagnostic["reward_terms"][name]
                                    for name in REWARD_NAMES
                                ],
                            ],
                        )
                    )
                )
                if not matches:
                    first_mismatch = step
                    break
                if terminated or truncated or stopped:
                    break
        baseline_success = bool(
            int(base_info["lap_counts"][0]) >= laps
            and not bool(base_info["collisions"][0])
            and not stopped
        )
        report = {
            "equivalent": bool(equivalent),
            "all_states_bitwise_equal": bool(all_states_exact),
            "initial_states_bitwise_equal": bool(initial_equal),
            "first_mismatch_step": first_mismatch,
            "tolerance": tolerance,
            "compared_steps": step,
            "maximum_state_error": dict(zip(STATE_NAMES, maximum_state_error.tolist())),
            "maximum_action_error": dict(
                zip(("steering_rad", "speed_mps"), maximum_action_error.tolist())
            ),
            "map": str(map_name),
            "seed": seed,
            "speed_mode": speed_mode,
            "requested_full_laps": laps,
            "completed_full_laps": int(base_info["lap_counts"][0]),
            "last_full_lap_time_s": float(base_info["lap_times"][0]),
            "full_lap_times_s": full_lap_times,
            "sim_time_s": float(base_info["sim_time"]),
            "baseline_success": baseline_success,
            "collision": bool(base_info["collisions"][0]),
            "terminated": bool(terminated or stopped),
            "truncated": bool(truncated),
            "sb3_checker": "passed",
            "sb3_version": importlib.metadata.version("stable-baselines3"),
            "initial_standard_state": initial.tolist(),
            "contract": contract,
            "note": "Task rewards intentionally differ from the native survival reward; no optimizer or policy training is run.",
        }
        rng = np.random.default_rng(seed)
        residual.reset(seed=seed)
        clipped = 0
        effective_max = 0.0
        random_valid = True
        random_count = 0
        with (output_dir / "random_residual.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "step",
                    "sim_time",
                    "z",
                    "effective_residual_rad",
                    "command_steering",
                    "command_speed",
                    "reward",
                    "terminated",
                    "truncated",
                ]
                + [f"reward_{name}" for name in REWARD_NAMES]
            )
            for random_count in range(1, random_steps + 1):
                action = rng.uniform(-1, 1, size=1).astype(np.float32)
                vector, reward, ended, timed_out, info = residual.step(action)
                diagnostic = info["residual"]
                physical = diagnostic["physical_action"]
                random_valid &= bool(
                    residual.observation_space.contains(vector)
                    and np.isfinite(reward)
                    and np.all(physical >= controller.action_low)
                    and np.all(physical <= controller.action_high)
                    and np.array_equal(action, diagnostic["policy_action"])
                )
                clipped += int(diagnostic["steering_clipped"])
                effective_max = max(
                    effective_max, abs(diagnostic["effective_steering_residual_rad"])
                )
                writer.writerow(
                    [
                        random_count,
                        info["sim_time"],
                        action[0],
                        diagnostic["effective_steering_residual_rad"],
                        *physical,
                        reward,
                        ended,
                        timed_out,
                        *[diagnostic["reward_terms"][name] for name in REWARD_NAMES],
                    ]
                )
                if ended or timed_out:
                    break
        report["random_action_check"] = {
            "interface_valid": bool(random_valid),
            "steps": random_count,
            "steering_clip_count": clipped,
            "maximum_effective_residual_rad": effective_max,
        }
    finally:
        base.close()
        if residual is not None:
            residual.close()
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot_equivalence(output_dir, report["equivalent"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", dest="map_name", default="Spielberg")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--speed-mode", choices=("fixed", "profile"), default="profile")
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--vgain", type=float, default=1.0)
    parser.add_argument("--lookahead", type=float, default=1.2)
    parser.add_argument("--laps", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--random-steps", type=int, default=256)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.random_steps < 0:
        parser.error("random-steps must be non-negative")
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.output_dir = (
            Path(__file__).parent
            / "runs"
            / f"residual_validation_{stamp}_{args.speed_mode}_seed{args.seed}"
        )
    report = validate(**vars(args))
    print(json.dumps(report, indent=2))
    print(f"Results: {args.output_dir.resolve()}")
    return (
        0
        if report["equivalent"] and report["random_action_check"]["interface_valid"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
