"""Evaluate a saved residual SAC policy against the zero-residual baseline.

    python -m SAC_project.evaluate --run-dir SAC_project/runs/sac_... --episodes 5

Add ``--render`` on a desktop machine to watch the same run in the native window.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
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


def parse_float_list(text: str) -> list[float]:
    try:
        values = [float(part) for part in text.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values or not all(math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("expected a non-empty finite list")
    return values


def build_cases(args) -> list[dict]:
    """Fixed-seed nominal runs, or the Cartesian product of start offsets."""
    if args.lateral_offsets is None and args.heading_offsets_deg is None:
        return [
            {
                "index": index,
                "seed": args.seed + index,
                "lateral_m": 0.0,
                "heading_deg": 0.0,
            }
            for index in range(args.episodes)
        ]
    lateral = args.lateral_offsets or [0.0]
    heading = args.heading_offsets_deg or [0.0]
    cases = []
    for lateral_m in lateral:
        for heading_deg in heading:
            index = len(cases)
            cases.append(
                {
                    "index": index,
                    "seed": args.seed + index,
                    "lateral_m": float(lateral_m),
                    "heading_deg": float(heading_deg),
                }
            )
    return cases


def nominal_initial_state(env, seed: int) -> np.ndarray:
    _, info = env.reset(seed=seed)
    return np.array(info["residual"]["std_state"], dtype=np.float64, copy=True)


def perturb_initial_state(
    env, state: np.ndarray, lateral_m: float, heading_deg: float
) -> np.ndarray:
    """Shift the CoG along the local path normal and rotate the heading."""
    state = np.array(state, dtype=np.float64, copy=True)
    command = env.controller.plan(state[[0, 1, 4]])
    projection = command.projection
    if projection is None:
        raise ValueError("cannot perturb a start state without a path projection")
    tangent = np.asarray(
        env.controller.vectors[projection.segment], dtype=np.float64
    )
    norm = float(np.linalg.norm(tangent))
    if norm == 0:
        raise ValueError("reference path has a zero-length segment")
    tangent /= norm
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    state[0] += lateral_m * normal[0]
    state[1] += lateral_m * normal[1]
    state[4] = (
        state[4] + math.radians(heading_deg) + math.pi
    ) % (2 * math.pi) - math.pi
    return state.astype(np.float32)


def run_episode(
    env,
    model,
    *,
    seed: int,
    render: bool,
    trajectory_path: Path | None = None,
    initial_state: np.ndarray | None = None,
    case: dict | None = None,
) -> dict:
    if initial_state is None:
        observation, info = env.reset(seed=seed)
    else:
        observation, info = env.reset(
            seed=seed,
            options={"states": np.asarray(initial_state, dtype=np.float32)[None, :]},
        )
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
    spawn_collision = bool(
        initial_state is not None and steps <= 1 and end_reason == "collision"
    )
    rear_array = np.asarray(rear, dtype=np.float64)
    heading_array = np.asarray(heading, dtype=np.float64)
    latency_ms = np.asarray(latency, dtype=np.float64) * 1000.0
    return {
        "seed": seed,
        "case_index": int(case["index"]) if case else 0,
        "lateral_offset_m": float(case["lateral_m"]) if case else 0.0,
        "heading_offset_deg": float(case["heading_deg"]) if case else 0.0,
        "initial_state": (
            None
            if initial_state is None
            else np.asarray(initial_state, dtype=float).tolist()
        ),
        "success": bool(info.get("is_success")),
        "spawn_collision": spawn_collision,
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
    valid = [episode for episode in episodes if not episode.get("spawn_collision")]
    invalid_starts = len(episodes) - len(valid)
    completed_laps = [
        lap for episode in valid for lap in episode["full_lap_times_s"]
    ]
    return {
        "label": label,
        "episodes": len(valid),
        "invalid_starts": invalid_starts,
        "success_rate": _mean([float(e["success"]) for e in valid]),
        "collision_rate": _mean([float(e["collision"]) for e in valid]),
        "tracking_stop_rate": _mean(
            [float(e["tracking_stop"]) for e in valid]
        ),
        "timeout_rate": _mean([float(e["timeout"]) for e in valid]),
        "rear_rms_m_mean": _mean([e["rear_rms_m"] for e in valid]),
        "rear_p95_m_mean": _mean([e["rear_p95_m"] for e in valid]),
        "heading_rms_rad_mean": _mean([e["heading_rms_rad"] for e in valid]),
        "mean_abs_z": _mean([e["mean_abs_z"] for e in valid]),
        "max_abs_z": _mean([e["max_abs_z"] for e in valid]),
        "clip_fraction_mean": _mean([e["clip_fraction"] for e in valid]),
        "episode_return_mean": _mean([e["episode_return"] for e in valid]),
        "episode_steps_mean": _mean([float(e["steps"]) for e in valid]),
        "full_lap_time_mean_s": _mean(completed_laps),
        "full_lap_time_min_s": min(completed_laps) if completed_laps else None,
        "complete_laps_total": len(completed_laps),
        "inference_latency_ms_mean": _mean(
            [e["inference_latency_ms_mean"] for e in valid]
        ),
        "inference_latency_ms_p95": _mean(
            [e["inference_latency_ms_p95"] for e in valid]
        ),
        "episodes_detail": episodes,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_env_arguments(parser)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="number of fixed-seed nominal runs when no offsets are given",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--lateral-offsets",
        type=parse_float_list,
        help="comma-separated CoG lateral offsets in metres; "
        "evaluates the Cartesian product with heading offsets",
    )
    parser.add_argument(
        "--heading-offsets-deg",
        type=parse_float_list,
        help="comma-separated heading offsets in degrees",
    )
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

    perturbation_mode = (
        args.lateral_offsets is not None or args.heading_offsets_deg is not None
    )
    cases = build_cases(args)
    if perturbation_mode:
        print(
            f"Using perturbation grid: {len(cases)} cases; --episodes is ignored"
        )

    episodes = {"zero_residual": [], "policy": []}
    env = build_env(settings, render=args.render)
    try:
        if args.render:
            add_render_overlays(env, settings.reference)
        base_state = None
        if perturbation_mode:
            base_state = nominal_initial_state(env, args.seed)
            print(f"Nominal start state: {base_state.tolist()}")
        for case in cases:
            seed = case["seed"]
            description = (
                f"lateral {case['lateral_m']:+.3f} m, "
                f"heading {case['heading_deg']:+.2f} deg"
            )
            if perturbation_mode:
                initial_state = perturb_initial_state(
                    env, base_state, case["lateral_m"], case["heading_deg"]
                )
            else:
                initial_state = None
            position = f"{case['index'] + 1}/{len(cases)}"
            print(f"[{position}] zero residual, {description}, seed {seed}")
            episodes["zero_residual"].append(
                run_episode(
                    env,
                    None,
                    seed=seed,
                    render=args.render,
                    trajectory_path=(
                        output_dir
                        / "trajectories"
                        / f"zero_residual_case{case['index']:02d}_seed{seed}.csv"
                        if args.trajectory
                        else None
                    ),
                    initial_state=initial_state,
                    case=case,
                )
            )
            print(f"[{position}] policy, {description}, seed {seed}")
            episodes["policy"].append(
                run_episode(
                    env,
                    model,
                    seed=seed,
                    render=args.render,
                    trajectory_path=(
                        output_dir
                        / "trajectories"
                        / f"policy_case{case['index']:02d}_seed{seed}.csv"
                        if args.trajectory
                        else None
                    ),
                    initial_state=initial_state,
                    case=case,
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

    case_comparison = []
    for baseline_episode, policy_episode in zip(
        episodes["zero_residual"], episodes["policy"]
    ):
        case_comparison.append(
            {
                "case_index": baseline_episode["case_index"],
                "seed": baseline_episode["seed"],
                "lateral_offset_m": baseline_episode["lateral_offset_m"],
                "heading_offset_deg": baseline_episode["heading_offset_deg"],
                "zero_success": baseline_episode["success"],
                "policy_success": policy_episode["success"],
                "zero_spawn_collision": baseline_episode["spawn_collision"],
                "policy_spawn_collision": policy_episode["spawn_collision"],
                "zero_end_reason": baseline_episode["end_reason"],
                "policy_end_reason": policy_episode["end_reason"],
                "zero_rear_rms_m": baseline_episode["rear_rms_m"],
                "policy_rear_rms_m": policy_episode["rear_rms_m"],
                "zero_rear_p95_m": baseline_episode["rear_p95_m"],
                "policy_rear_p95_m": policy_episode["rear_p95_m"],
                "zero_lap_time_s": baseline_episode["last_full_lap_time_s"],
                "policy_lap_time_s": policy_episode["last_full_lap_time_s"],
            }
        )

    report = run_metadata(
        settings,
        command=command_line(),
        model=str(model_path),
        evaluation={
            "episodes": len(cases),
            "seed": args.seed,
            "device": str(model.device),
            "render": args.render,
            "lateral_offsets_m": args.lateral_offsets,
            "heading_offsets_deg": args.heading_offsets_deg,
            "perturbation_mode": perturbation_mode,
            "base_initial_state": (
                None if base_state is None else base_state.tolist()
            ),
        },
        results=summaries,
        comparison=comparison,
        case_comparison=case_comparison,
    )
    write_json(output_dir / "summary.json", report)

    fields = [
        "label",
        "seed",
        "case_index",
        "lateral_offset_m",
        "heading_offset_deg",
        "spawn_collision",
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
    if args.lateral_offsets is not None or args.heading_offsets_deg is not None:
        print("\nPer-case results (zero -> policy):")
        for case in case_comparison:
            marker = (
                "  [invalid start]"
                if case["zero_spawn_collision"] or case["policy_spawn_collision"]
                else ""
            )
            print(
                f"  lateral {case['lateral_offset_m']:+.3f} m, "
                f"heading {case['heading_offset_deg']:+.2f} deg: "
                f"success {int(case['zero_success'])} -> "
                f"{int(case['policy_success'])}, "
                f"RMS {case['zero_rear_rms_m']:.4f} -> "
                f"{case['policy_rear_rms_m']:.4f} m{marker}"
            )
    print(f"Results: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
