"""Evaluate a saved residual SAC policy against the zero-residual baseline.

    python -m SAC_project.evaluate --run-dir SAC_project/runs/sac_... --episodes 5

Add ``--render`` on a desktop machine to watch the same run in the native window.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import time

import numpy as np

from stable_baselines3 import SAC

from SAC_project.rl_common import (
    EnvSettings,
    add_env_arguments,
    build_env,
    command_line,
    make_run_dir,
    run_metadata,
    settings_from_args,
    timestamp,
    write_json,
)

REWARD_NAMES = ("progress", "lateral", "heading", "residual", "smoothness", "failure")


def add_render_overlays(env, reference: str) -> None:
    """Draw the reference line and live lookahead point, as baseline.py does."""
    renderer = env.unwrapped.renderer
    if renderer is None:
        raise RuntimeError("render requested but the environment has no renderer")
    getattr(env.unwrapped.track, reference).render_waypoints(renderer)
    marker = None

    def draw(renderer_):
        nonlocal marker
        command = getattr(env, "_command", None)
        if command is None or command.target is None:
            return
        points = command.target[None, :]
        if marker is None:
            marker = renderer_.get_points_renderer(
                points, color=(228, 26, 28), size=10
            )
        else:
            marker.update(points)

    env.unwrapped.add_render_callback(draw)


def _lap_count(info: dict) -> int:
    counts = info.get("lap_counts")
    return int(counts[0]) if counts is not None else 0


def _lap_time(info: dict) -> float | None:
    times = info.get("lap_times")
    return float(times[0]) if times is not None else None


def run_episode(
    env,
    model,
    *,
    seed: int,
    render: bool,
    trajectory_path: Path | None = None,
) -> dict:
    observation, info = env.reset(seed=seed)
    if render:
        env.render()
    steps = 0
    rear = []
    heading = []
    abs_z = []
    clip_count = 0
    reward_total = 0.0
    term_sums = defaultdict(float)
    latency = []
    lap_times = []
    previous_lap_count = _lap_count(info)
    rows = [] if trajectory_path is not None else None

    while True:
        steps += 1
        if model is None:
            action = np.zeros(1, dtype=np.float32)
            latency_s = None
        else:
            started = time.perf_counter()
            action, _ = model.predict(observation, deterministic=True)
            latency_s = time.perf_counter() - started
            latency.append(latency_s)
        observation, reward, terminated, truncated, info = env.step(action)
        diagnostic = info["residual"]
        rear.append(float(diagnostic["rear_lateral_error_m"]))
        heading.append(float(diagnostic["heading_error_rad"]))
        z = float(np.asarray(diagnostic["policy_action"]).reshape(-1)[0])
        abs_z.append(abs(z))
        clip_count += int(diagnostic["steering_clipped"])
        reward_total += float(reward)
        for name in REWARD_NAMES:
            term_sums[name] += float(diagnostic["reward_terms"][name])
        lap_count = _lap_count(info)
        if lap_count > previous_lap_count:
            lap_time = _lap_time(info)
            if lap_time is not None:
                lap_times.append(lap_time)
            previous_lap_count = lap_count
        if rows is not None:
            state = diagnostic["std_state"]
            base_action = diagnostic["base_action"]
            physical = diagnostic["physical_action"]
            rows.append(
                {
                    "step": steps,
                    "sim_time": float(info["sim_time"]),
                    "x": float(state[0]),
                    "y": float(state[1]),
                    "yaw": float(state[4]),
                    "speed": float(state[3]),
                    "z": z,
                    "base_steering": float(base_action[0]),
                    "base_speed": float(base_action[1]),
                    "steering": float(physical[0]),
                    "speed_command": float(physical[1]),
                    "rear_error_m": rear[-1],
                    "heading_error_rad": heading[-1],
                    "steering_clipped": int(diagnostic["steering_clipped"]),
                    "reward": float(reward),
                    "latency_ms": (
                        latency_s * 1000.0 if latency_s is not None else float("nan")
                    ),
                    **{
                        f"reward_{name}": float(diagnostic["reward_terms"][name])
                        for name in REWARD_NAMES
                    },
                    "lap_count": lap_count,
                    "collision": int(bool(info["collisions"][0])),
                }
            )
        if render:
            env.render()
        if terminated or truncated:
            break

    if trajectory_path is not None and rows:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        with trajectory_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    residual = info["residual"]
    end_reason = residual.get("end_reason", "")
    rear_array = np.asarray(rear, dtype=np.float64)
    heading_array = np.asarray(heading, dtype=np.float64)
    latency_ms = np.asarray(latency, dtype=np.float64) * 1000.0
    return {
        "seed": seed,
        "success": bool(info.get("is_success")),
        "end_reason": end_reason,
        "collision": end_reason == "collision",
        "tracking_stop": end_reason == "tracking_stop",
        "timeout": end_reason == "time_limit",
        "steps": steps,
        "sim_time_s": float(info["sim_time"]),
        "rear_rms_m": float(np.sqrt(np.mean(rear_array**2))),
        "rear_p95_m": float(np.percentile(np.abs(rear_array), 95)),
        "heading_rms_rad": float(np.sqrt(np.mean(heading_array**2))),
        "mean_abs_z": float(np.mean(abs_z)),
        "max_abs_z": float(np.max(abs_z)),
        "clip_fraction": clip_count / max(steps, 1),
        "episode_return": reward_total,
        "reward_terms_mean": {
            name: term_sums[name] / max(steps, 1) for name in REWARD_NAMES
        },
        "complete_laps": len(lap_times),
        "full_lap_times_s": lap_times,
        "last_full_lap_time_s": lap_times[-1] if lap_times else None,
        "inference_latency_ms_mean": (
            float(np.mean(latency_ms)) if latency_ms.size else None
        ),
        "inference_latency_ms_p95": (
            float(np.percentile(latency_ms, 95)) if latency_ms.size else None
        ),
    }


def _mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def summarize(label: str, episodes: list[dict]) -> dict:
    completed_laps = [
        lap for episode in episodes for lap in episode["full_lap_times_s"]
    ]
    return {
        "label": label,
        "episodes": len(episodes),
        "success_rate": _mean([float(e["success"]) for e in episodes]),
        "collision_rate": _mean([float(e["collision"]) for e in episodes]),
        "tracking_stop_rate": _mean(
            [float(e["tracking_stop"]) for e in episodes]
        ),
        "timeout_rate": _mean([float(e["timeout"]) for e in episodes]),
        "rear_rms_m_mean": _mean([e["rear_rms_m"] for e in episodes]),
        "rear_p95_m_mean": _mean([e["rear_p95_m"] for e in episodes]),
        "heading_rms_rad_mean": _mean([e["heading_rms_rad"] for e in episodes]),
        "mean_abs_z": _mean([e["mean_abs_z"] for e in episodes]),
        "max_abs_z": _mean([e["max_abs_z"] for e in episodes]),
        "clip_fraction_mean": _mean([e["clip_fraction"] for e in episodes]),
        "episode_return_mean": _mean([e["episode_return"] for e in episodes]),
        "episode_steps_mean": _mean([float(e["steps"]) for e in episodes]),
        "full_lap_time_mean_s": _mean(completed_laps),
        "full_lap_time_min_s": min(completed_laps) if completed_laps else None,
        "complete_laps_total": len(completed_laps),
        "inference_latency_ms_mean": _mean(
            [e["inference_latency_ms_mean"] for e in episodes]
        ),
        "inference_latency_ms_p95": _mean(
            [e["inference_latency_ms_p95"] for e in episodes]
        ),
        "episodes_detail": episodes,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_env_arguments(parser)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--device", choices=("cpu", "auto", "cuda"), default="cpu"
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--trajectory", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if args.run_dir is None and args.model is None:
        parser.error("--run-dir or --model is required")
    return args


def resolve_model_path(path: Path) -> Path:
    if path.exists():
        return path
    candidate = path.with_suffix(".zip")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"model not found: {path}")


def main(argv=None) -> int:
    args = parse_args(argv)
    base = EnvSettings()
    model_path = args.model
    if args.run_dir is not None:
        config_path = args.run_dir / "train_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"training config not found: {config_path}")
        training_config = json.loads(config_path.read_text(encoding="utf-8"))
        base = EnvSettings.from_dict(training_config["env"])
        if model_path is None:
            model_path = args.run_dir / "sac_final.zip"
    model_path = resolve_model_path(model_path)
    settings = settings_from_args(args, base)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
    elif args.run_dir is not None:
        output_dir = make_run_dir(
            "evaluation", args.run_dir, f"evaluation_{timestamp()}"
        )
    else:
        output_dir = make_run_dir(
            "evaluation",
            Path(__file__).parent / "runs",
            f"evaluation_{timestamp()}",
        )

    model = SAC.load(str(model_path), env=None, device=args.device)
    model.policy.set_training_mode(False)

    episodes = {"zero_residual": [], "policy": []}
    env = build_env(settings, render=args.render)
    try:
        if args.render:
            add_render_overlays(env, settings.reference)
        for index in range(args.episodes):
            seed = args.seed + index
            print(f"[{index + 1}/{args.episodes}] zero residual, seed {seed}")
            episodes["zero_residual"].append(
                run_episode(
                    env,
                    None,
                    seed=seed,
                    render=args.render,
                    trajectory_path=(
                        output_dir / "trajectories" / f"zero_residual_seed{seed}.csv"
                        if args.trajectory
                        else None
                    ),
                )
            )
            print(f"[{index + 1}/{args.episodes}] policy, seed {seed}")
            episodes["policy"].append(
                run_episode(
                    env,
                    model,
                    seed=seed,
                    render=args.render,
                    trajectory_path=(
                        output_dir / "trajectories" / f"policy_seed{seed}.csv"
                        if args.trajectory
                        else None
                    ),
                )
            )
    finally:
        env.close()

    summaries = {
        label: summarize(label, results) for label, results in episodes.items()
    }
    comparison = {}
    baseline = summaries["zero_residual"]
    policy = summaries["policy"]
    for metric in (
        "success_rate",
        "rear_rms_m_mean",
        "rear_p95_m_mean",
        "heading_rms_rad_mean",
        "full_lap_time_mean_s",
        "episode_return_mean",
    ):
        left = baseline[metric]
        right = policy[metric]
        comparison[metric] = {
            "zero_residual": left,
            "policy": right,
            "delta_policy_minus_zero": (
                right - left if left is not None and right is not None else None
            ),
        }

    report = run_metadata(
        settings,
        command=command_line(),
        model=str(model_path),
        evaluation={
            "episodes": args.episodes,
            "seed": args.seed,
            "device": str(model.device),
            "render": args.render,
        },
        results=summaries,
        comparison=comparison,
    )
    write_json(output_dir / "summary.json", report)

    fields = [
        "label",
        "seed",
        "success",
        "end_reason",
        "steps",
        "sim_time_s",
        "complete_laps",
        "last_full_lap_time_s",
        "rear_rms_m",
        "rear_p95_m",
        "heading_rms_rad",
        "mean_abs_z",
        "max_abs_z",
        "clip_fraction",
        "episode_return",
        "inference_latency_ms_mean",
        "inference_latency_ms_p95",
    ]
    with (output_dir / "episodes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, results in episodes.items():
            for result in results:
                writer.writerow(
                    {
                        "label": label,
                        **{field: result.get(field) for field in fields[1:]},
                    }
                )

    print(json.dumps({"comparison": comparison}, indent=2))
    print(f"Results: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
