"""Aggregate multi-seed SAC evaluation results and draw comparison figures.

Example:
    python -m SAC_project.analyze_results \
      --run-dir SAC_project/runs/sac_local_500k_seed12345 \
      --run-dir SAC_project/runs/sac_local_500k_seed42 \
      --run-dir SAC_project/runs/sac_local_500k_seed2026 \
      --output-dir SAC_project/results/3seed_500k
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_COLOR = "#4C72B0"
POLICY_COLOR = "#DD8452"
SEED_COLORS = ("#4C72B0", "#DD8452", "#55A868")
TRAIN_PANELS = (
    ("rollout/ep_rew_mean", "Episode return"),
    ("residual/success", "Training success rate"),
    ("residual/rear_rms_m", "Rear lateral RMS [m]"),
    ("residual/mean_abs_z", "Mean |residual z|"),
)
SUMMARY_METRICS = (
    ("success_rate", "Success rate", "percent"),
    ("collision_rate", "Collision rate", "percent"),
    ("rear_rms_m_mean", "Rear RMS [m]", "m"),
    ("rear_p95_m_mean", "Rear P95 [m]", "m"),
    ("heading_rms_rad_mean", "Heading RMS [rad]", "rad"),
    ("full_lap_time_mean_s", "Full lap time [s]", "s"),
    ("episode_return_mean", "Episode return", "number"),
    ("mean_abs_z", "Mean abs(z)", "number"),
    ("clip_fraction_mean", "Steering clip fraction", "percent"),
)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latest_evaluation(run_dir: Path) -> Path:
    candidates = sorted(Path(run_dir).glob("evaluation_*"))
    if not candidates:
        raise FileNotFoundError(f"no evaluation_* directory in {run_dir}")
    return candidates[-1]


def load_seed(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    config = read_json(run_dir / "train_config.json")
    evaluation = latest_evaluation(run_dir)
    summary = read_json(evaluation / "summary.json")
    progress = read_csv(run_dir / "logs" / "progress.csv")
    return {
        "run_dir": run_dir,
        "seed": int(config["train"]["seed"]),
        "evaluation": evaluation,
        "summary": summary,
        "progress": progress,
    }


def numeric_series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for row in rows:
        try:
            x = float(row["time/total_timesteps"])
            y = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return np.asarray(xs), np.asarray(ys)


def seed_values(seeds: list[dict], label: str, key: str) -> list[float]:
    values = []
    for seed in seeds:
        value = seed["summary"]["results"][label].get(key)
        if value is not None:
            values.append(float(value))
    return values


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def paired_cases(seeds: list[dict]):
    for seed in seeds:
        for case in seed["summary"]["case_comparison"]:
            yield seed["seed"], case


def format_value(value, kind: str) -> str:
    if value is None:
        return "-"
    if kind == "percent":
        return f"{100.0 * value:.1f}%"
    if kind == "m":
        return f"{value:.4f} m"
    if kind == "rad":
        return f"{value:.4f} rad"
    if kind == "s":
        return f"{value:.3f} s"
    return f"{value:.3f}"


def format_seed(value, kind: str) -> str:
    if value is None:
        return "-"
    if kind == "percent":
        return f"{100.0 * value:.1f}%"
    if kind == "m":
        return f"{value:.4f}"
    if kind == "rad":
        return f"{value:.4f}"
    if kind == "s":
        return f"{value:.3f}"
    return f"{value:.3f}"


def plot_learning_curves(seeds: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for axis, (key, label) in zip(axes.flat, TRAIN_PANELS):
        curves = []
        for index, seed in enumerate(seeds):
            x, y = numeric_series(seed["progress"], key)
            if x.size == 0:
                continue
            axis.plot(
                x,
                y,
                color=SEED_COLORS[index % len(SEED_COLORS)],
                alpha=0.35,
                linewidth=1.2,
                label=f"seed {seed['seed']}",
            )
            curves.append((x, y))
        if curves:
            min_x = min(curve[0].min() for curve in curves)
            max_x = max(curve[0].max() for curve in curves)
            grid = np.linspace(min_x, max_x, 200)
            stack = np.full((len(curves), grid.size), np.nan)
            for index, (x, y) in enumerate(curves):
                mask = (grid >= x.min()) & (grid <= x.max())
                stack[index, mask] = np.interp(grid[mask], x, y)
            mean = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0)
            axis.plot(grid, mean, color="black", linewidth=2.0, label="mean")
            axis.fill_between(
                grid, mean - std, mean + std, color="black", alpha=0.12, linewidth=0
            )
        axis.set_xlabel("Environment steps")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
        if key == "residual/success":
            axis.set_ylim(-0.02, 1.02)
        axis.legend(fontsize=8)
    figure.suptitle("Training curves (stochastic rollouts, 3 seeds)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_metrics(seeds: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    labels = ("Zero residual", "SAC")
    for axis, (key, label, kind) in zip(axes.flat, SUMMARY_METRICS[:6]):
        baseline = seed_values(seeds, "zero_residual", key)
        policy = seed_values(seeds, "policy", key)
        baseline_mean = float(np.mean(baseline)) if baseline else 0.0
        policy_mean = float(np.mean(policy)) if policy else 0.0
        baseline_std = float(np.std(baseline)) if baseline else 0.0
        policy_std = float(np.std(policy)) if policy else 0.0
        bars = axis.bar(
            labels,
            [baseline_mean, policy_mean],
            yerr=[baseline_std, policy_std],
            color=[BASE_COLOR, POLICY_COLOR],
            capsize=4,
            width=0.6,
        )
        axis.set_title(label)
        axis.grid(True, axis="y", alpha=0.25)
        if kind == "percent":
            axis.set_ylim(0, 1.05)
        for bar, value in zip(bars, (baseline_mean, policy_mean)):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                format_value(value, kind),
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.tick_params(axis="x", labelsize=8)
    figure.suptitle(
        "Deterministic evaluation, 9 perturbation cases (mean ± std over training seeds)"
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def robustness_by_lateral(seeds: list[dict]):
    lateral_values = sorted(
        {
            case["lateral_offset_m"]
            for seed in seeds
            for case in seed["summary"]["case_comparison"]
        }
    )
    result = {}
    for method in ("zero_residual", "policy"):
        success_curves = []
        rms_curves = []
        for seed in seeds:
            case_success = []
            case_rms = []
            for lateral in lateral_values:
                cases = [
                    case
                    for case in seed["summary"]["case_comparison"]
                    if abs(case["lateral_offset_m"] - lateral) < 1e-9
                ]
                case_success.append(
                    float(np.mean([case[f"{'zero' if method == 'zero_residual' else 'policy'}_success"] for case in cases]))
                )
                case_rms.append(
                    float(np.mean([case[f"{'zero' if method == 'zero_residual' else 'policy'}_rear_rms_m"] for case in cases]))
                )
            success_curves.append(case_success)
            rms_curves.append(case_rms)
        result[method] = {
            "lateral": lateral_values,
            "success": np.asarray(success_curves),
            "rms": np.asarray(rms_curves),
        }
    return result


def plot_robustness(seeds: list[dict], output: Path) -> None:
    data = robustness_by_lateral(seeds)
    lateral = np.asarray(data["zero_residual"]["lateral"])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for axis, metric, ylabel in zip(
        axes, ("success", "rms"), ("Success rate", "Rear lateral RMS [m]")
    ):
        for method, label, color in (
            ("zero_residual", "Zero residual", BASE_COLOR),
            ("policy", "SAC", POLICY_COLOR),
        ):
            curves = data[method][metric]
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            axis.plot(lateral, mean, marker="o", color=color, label=label)
            axis.fill_between(
                lateral, mean - std, mean + std, color=color, alpha=0.15, linewidth=0
            )
            for row in curves:
                axis.scatter(lateral, row, color=color, alpha=0.35, s=18)
        axis.set_xlabel("Initial lateral offset [m]")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend()
        if metric == "success":
            axis.set_ylim(-0.02, 1.02)
    figure.suptitle("Robustness to initial lateral offset (3 heading offsets, 3 seeds)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_heatmap(seeds: list[dict], output: Path) -> None:
    lateral_values = sorted(
        {
            case["lateral_offset_m"]
            for seed in seeds
            for case in seed["summary"]["case_comparison"]
        }
    )
    heading_values = sorted(
        {
            case["heading_offset_deg"]
            for seed in seeds
            for case in seed["summary"]["case_comparison"]
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for axis, method, title in (
        (axes[0], "zero_success", "Zero residual"),
        (axes[1], "policy_success", "SAC"),
    ):
        counts = np.zeros((len(heading_values), len(lateral_values)))
        for _, case in paired_cases(seeds):
            i = heading_values.index(case["heading_offset_deg"])
            j = lateral_values.index(case["lateral_offset_m"])
            counts[i, j] += int(bool(case[method]))
        fraction = counts / len(seeds)
        image = axis.imshow(
            fraction, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto"
        )
        for i in range(len(heading_values)):
            for j in range(len(lateral_values)):
                axis.text(
                    j,
                    i,
                    f"{int(counts[i, j])}/{len(seeds)}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )
        axis.set_xticks(range(len(lateral_values)))
        axis.set_xticklabels([f"{value:+.2f}" for value in lateral_values])
        axis.set_yticks(range(len(heading_values)))
        axis.set_yticklabels([f"{value:+.1f}°" for value in heading_values])
        axis.set_xlabel("Lateral offset [m]")
        axis.set_ylabel("Heading offset [deg]")
        axis.set_title(title)
    figure.colorbar(image, ax=axes, shrink=0.8, label="Success fraction")
    figure.suptitle("Success across perturbation grid (count over seeds)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def find_reference_path(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None
    candidates = sorted(
        Path("SAC_project/runs").glob("baseline_*_seed*/reference_path.csv")
    )
    return candidates[-1] if candidates else None


def plot_trajectory(
    seeds: list[dict],
    output: Path,
    reference_path: Path | None,
    lateral_target: float,
    heading_target: float,
) -> str | None:
    first = seeds[0]
    cases = first["summary"]["case_comparison"]
    if not cases:
        return None
    representative = min(
        cases,
        key=lambda case: abs(case["lateral_offset_m"] - lateral_target)
        + abs(case["heading_offset_deg"] - heading_target),
    )
    evaluation = first["evaluation"]
    case_index = representative["case_index"]
    case_seed = representative["seed"]
    baseline_file = (
        evaluation
        / "trajectories"
        / f"zero_residual_case{case_index:02d}_seed{case_seed}.csv"
    )
    policy_file = (
        evaluation
        / "trajectories"
        / f"policy_case{case_index:02d}_seed{case_seed}.csv"
    )
    if not baseline_file.exists() or not policy_file.exists():
        return None
    baseline = read_csv(baseline_file)
    policy = read_csv(policy_file)

    def columns(rows):
        return {
            key: np.asarray([float(row[key]) for row in rows if row.get(key) not in (None, "")])
            for key in ("x", "y", "sim_time", "rear_error_m", "z")
        }

    base = columns(baseline)
    sac = columns(policy)
    collision = np.flatnonzero(
        np.asarray([float(row.get("collision", 0)) for row in baseline]) > 0
    )
    reference = read_csv(reference_path) if reference_path is not None else None

    def draw_paths(axis):
        if reference is not None:
            axis.plot(
                [float(row["x"]) for row in reference],
                [float(row["y"]) for row in reference],
                color="black",
                linestyle=":",
                linewidth=1.0,
                label="Reference",
            )
        axis.plot(
            base["x"],
            base["y"],
            color=BASE_COLOR,
            linewidth=2.0,
            label="Zero residual",
            zorder=3,
        )
        axis.plot(
            sac["x"],
            sac["y"],
            color=POLICY_COLOR,
            linewidth=2.0,
            linestyle="--",
            label="SAC",
            zorder=2,
        )
        axis.scatter(
            base["x"][0], base["y"][0], color="black", marker="o", s=25, zorder=5
        )
        if collision.size:
            index = int(collision[0])
            axis.scatter(
                base["x"][index],
                base["y"][index],
                color=BASE_COLOR,
                marker="X",
                s=100,
                zorder=6,
                label="Baseline collision",
            )
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("World X [m]")
        axis.set_ylabel("World Y [m]")
        axis.grid(True, alpha=0.25)

    figure, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    draw_paths(axes[0])
    axes[0].set_title(
        f"Full track: lateral {representative['lateral_offset_m']:+.2f} m, "
        f"heading {representative['heading_offset_deg']:+.1f} deg"
    )
    axes[0].legend(fontsize=8)

    zoom_seconds = 6.0
    base_mask = base["sim_time"] <= zoom_seconds
    sac_mask = sac["sim_time"] <= zoom_seconds
    draw_paths(axes[1])
    x_values = np.concatenate([base["x"][base_mask], sac["x"][sac_mask]])
    y_values = np.concatenate([base["y"][base_mask], sac["y"][sac_mask]])
    if x_values.size and y_values.size:
        pad = 1.5
        axes[1].set_xlim(x_values.min() - pad, x_values.max() + pad)
        axes[1].set_ylim(y_values.min() - pad, y_values.max() + pad)
    axes[1].set_title(f"Start zoom (first {zoom_seconds:g} s)")

    axis = axes[2]
    axis.plot(
        base["sim_time"],
        base["rear_error_m"],
        color=BASE_COLOR,
        linewidth=2.0,
        label="Zero residual",
    )
    axis.plot(
        sac["sim_time"],
        sac["rear_error_m"],
        color=POLICY_COLOR,
        linewidth=2.0,
        linestyle="--",
        label="SAC",
    )
    if collision.size:
        axis.axvline(
            base["sim_time"][int(collision[0])],
            color=BASE_COLOR,
            linestyle=":",
            label="Baseline collision",
        )
    axis.set_xlim(0, min(12.0, float(sac["sim_time"][-1])))
    axis.set_xlabel("Simulation time [s]")
    axis.set_ylabel("Rear lateral error [m]")
    axis.set_title("Tracking error")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return f"case {case_index}, seed {case_seed}"


def plot_failure_breakdown(seeds: list[dict], output: Path) -> None:
    categories = ("lap_complete", "collision", "tracking_stop", "time_limit")
    labels = ("Success", "Collision", "Tracking stop", "Timeout")
    colors = ("#55A868", "#C44E52", "#8172B2", "#CCB974")
    counts = {"zero_residual": {key: 0 for key in categories}, "policy": {key: 0 for key in categories}}
    for _, case in paired_cases(seeds):
        for method, prefix in (("zero_residual", "zero"), ("policy", "policy")):
            reason = case[f"{prefix}_end_reason"]
            if reason in counts[method]:
                counts[method][reason] += 1
    figure, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
    left = np.zeros(2)
    for category, label, color in zip(categories, labels, colors):
        values = np.asarray(
            [counts["zero_residual"][category], counts["policy"][category]]
        )
        axis.barh(
            ["Zero residual", "SAC"],
            values,
            left=left,
            color=color,
            label=label,
        )
        left += values
    axis.set_xlabel("Episode count")
    axis.set_title("Termination reasons across all perturbation cases and seeds")
    axis.grid(True, axis="x", alpha=0.25)
    axis.legend(fontsize=8, ncol=4, loc="lower center")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_summary(seeds: list[dict], output_dir: Path, trajectory_note: str | None) -> None:
    rows = []
    for key, label, kind in SUMMARY_METRICS:
        baseline = seed_values(seeds, "zero_residual", key)
        policy = seed_values(seeds, "policy", key)
        baseline_mean, baseline_std = mean_std(baseline)
        policy_mean, policy_std = mean_std(policy)
        row = {
            "metric": key,
            "label": label,
            "zero_mean": baseline_mean,
            "zero_std": baseline_std,
            "policy_mean": policy_mean,
            "policy_std": policy_std,
        }
        for seed in seeds:
            row[f"seed_{seed['seed']}"] = seed["summary"]["results"]["policy"].get(key)
        rows.append(row)
    with (output_dir / "metrics_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 3-seed SAC results",
        "",
        "Protocol: deterministic policy, 9 perturbation cases "
        "(lateral -0.3/0/+0.1 m x heading -5/0/+5 deg), "
        "fixed nominal start state, 3 training seeds, 500k steps each.",
        "",
        "## Aggregate over perturbation cases",
        "",
        "| Metric | Zero residual | SAC (mean ± std over seeds) |",
        "|---|---:|---:|",
    ]
    for key, label, kind in SUMMARY_METRICS:
        baseline = seed_values(seeds, "zero_residual", key)
        policy = seed_values(seeds, "policy", key)
        baseline_mean, _ = mean_std(baseline)
        policy_mean, policy_std = mean_std(policy)
        if kind == "percent":
            baseline_text = f"{100.0 * baseline_mean:.1f}%"
            policy_text = f"{100.0 * policy_mean:.1f}% ± {100.0 * policy_std:.1f}%"
        elif kind in ("m", "rad", "s"):
            unit = {"m": "m", "rad": "rad", "s": "s"}[kind]
            baseline_text = f"{baseline_mean:.4f} {unit}"
            policy_text = f"{policy_mean:.4f} ± {policy_std:.4f} {unit}"
        else:
            baseline_text = f"{baseline_mean:.3f}"
            policy_text = f"{policy_mean:.3f} ± {policy_std:.3f}"
        lines.append(f"| {label} | {baseline_text} | {policy_text} |")

    lines.extend(["", "## Per seed", ""])
    header = "| Metric | " + " | ".join(f"seed {s['seed']}" for s in seeds) + " |"
    lines.extend([header, "|---|" + "---:|" * len(seeds)])
    for key, label, kind in SUMMARY_METRICS:
        values = " | ".join(
            format_seed(seed["summary"]["results"]["policy"].get(key), kind)
            for seed in seeds
        )
        lines.append(f"| {label} | {values} |")

    lines.extend(
        [
            "",
            "## Nominal case (lateral 0, heading 0)",
            "",
            "| Seed | Zero success | SAC success | Zero RMS [m] | SAC RMS [m] "
            "| Zero lap [s] | SAC lap [s] |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in seeds:
        case = next(
            case
            for case in seed["summary"]["case_comparison"]
            if abs(case["lateral_offset_m"]) < 1e-9
            and abs(case["heading_offset_deg"]) < 1e-9
        )
        zero_lap = case["zero_lap_time_s"]
        policy_lap = case["policy_lap_time_s"]
        lines.append(
            f"| {seed['seed']} | "
            f"{'yes' if case['zero_success'] else 'no'} | "
            f"{'yes' if case['policy_success'] else 'no'} | "
            f"{case['zero_rear_rms_m']:.4f} | {case['policy_rear_rms_m']:.4f} | "
            f"{'-' if zero_lap is None else f'{zero_lap:.3f}'} | "
            f"{'-' if policy_lap is None else f'{policy_lap:.3f}'} |"
        )

    rms_deltas = []
    p95_deltas = []
    lap_deltas = []
    wins = ties = losses = 0
    for seed in seeds:
        for case in seed["summary"]["case_comparison"]:
            rms_deltas.append(case["policy_rear_rms_m"] - case["zero_rear_rms_m"])
            p95_deltas.append(case["policy_rear_p95_m"] - case["zero_rear_p95_m"])
            if (
                case["zero_lap_time_s"] is not None
                and case["policy_lap_time_s"] is not None
            ):
                lap_deltas.append(
                    case["policy_lap_time_s"] - case["zero_lap_time_s"]
                )
            if case["policy_success"] and not case["zero_success"]:
                wins += 1
            elif case["zero_success"] and not case["policy_success"]:
                losses += 1
            else:
                ties += 1

    def delta_text(values, unit):
        if not values:
            return "n/a"
        return f"{np.mean(values):+.4f} ± {np.std(values):.4f} {unit}"

    lines.extend(
        [
            "",
            "## Paired changes (policy - baseline)",
            "",
            f"- Rear RMS: {delta_text(rms_deltas, 'm')} over {len(rms_deltas)} cases",
            f"- Rear P95: {delta_text(p95_deltas, 'm')} over {len(p95_deltas)} cases",
            f"- Lap time: {delta_text(lap_deltas, 's')} over {len(lap_deltas)} "
            "cases where both completed",
            f"- Success: {wins} rescues, {ties} ties, {losses} losses",
        ]
    )

    rescue_seeds = []
    matched_seeds = []
    for seed in seeds:
        seed_cases = seed["summary"]["case_comparison"]
        minus_cases = [
            case
            for case in seed_cases
            if abs(case["lateral_offset_m"] + 0.3) < 1e-9
        ]
        if minus_cases and all(case["policy_success"] for case in minus_cases):
            rescue_seeds.append(seed["seed"])
        seed_wins = sum(
            1
            for case in seed_cases
            if case["policy_success"] and not case["zero_success"]
        )
        seed_losses = sum(
            1
            for case in seed_cases
            if case["zero_success"] and not case["policy_success"]
        )
        if seed_wins == 0 and seed_losses == 0:
            matched_seeds.append(seed["seed"])
    baseline_success = np.mean(
        seed_values(seeds, "zero_residual", "success_rate")
    )
    policy_success_values = seed_values(seeds, "policy", "success_rate")
    policy_success = np.mean(policy_success_values)
    policy_success_std = np.std(policy_success_values)

    lines.extend(
        [
            "",
            "## Key findings",
            "",
            f"- Aggregate success: {100.0 * baseline_success:.1f}% -> "
            f"{100.0 * policy_success:.1f}% ± {100.0 * policy_success_std:.1f}% "
            "(mean ± std over seeds).",
            f"- Paired outcomes: {wins} rescues, {ties} ties, {losses} losses.",
            f"- Seeds rescuing every lateral -0.3 m case: "
            f"{rescue_seeds if rescue_seeds else 'none'}.",
            f"- Seeds with no rescue but no regression: "
            f"{matched_seeds if matched_seeds else 'none'}.",
            f"- Rear RMS change: {delta_text(rms_deltas, 'm')}; "
            f"P95 change: {delta_text(p95_deltas, 'm')}.",
            "- Protocol limits: one map, 500k steps, initial lateral/heading offsets "
            "only; friction, actuator delay and domain randomization are untested.",
            "- Lap times use successful episodes only; failed episodes remain in "
            "success and tracking-error statistics.",
            "",
            "## Figures",
            "",
            "- `fig_learning_curves.png`: stochastic training curves per seed.",
            "- `fig_metrics_comparison.png`: deterministic aggregate metrics with seed error bars.",
            "- `fig_robustness_lateral.png`: success and RMS versus initial lateral offset.",
            "- `fig_success_heatmap.png`: success counts over the lateral x heading grid.",
            "- `fig_trajectory_compare.png`: representative baseline crash versus SAC recovery"
            + (f" ({trajectory_note})." if trajectory_note else "."),
            "- `fig_failure_breakdown.png`: termination reasons.",
        ]
    )
    (output_dir / "metrics_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="training run directory; repeat once per seed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("SAC_project/results/3seed_500k"),
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        help="optional reference_path.csv for the trajectory panel",
    )
    parser.add_argument("--representative-lateral", type=float, default=-0.3)
    parser.add_argument("--representative-heading-deg", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    seeds = [load_seed(path) for path in args.run_dir]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = find_reference_path(args.reference_path)

    plot_learning_curves(seeds, args.output_dir / "fig_learning_curves.png")
    plot_metrics(seeds, args.output_dir / "fig_metrics_comparison.png")
    plot_robustness(seeds, args.output_dir / "fig_robustness_lateral.png")
    plot_heatmap(seeds, args.output_dir / "fig_success_heatmap.png")
    trajectory_note = plot_trajectory(
        seeds,
        args.output_dir / "fig_trajectory_compare.png",
        reference,
        args.representative_lateral,
        args.representative_heading_deg,
    )
    plot_failure_breakdown(seeds, args.output_dir / "fig_failure_breakdown.png")
    write_summary(seeds, args.output_dir, trajectory_note)

    print(f"Seeds: {', '.join(str(seed['seed']) for seed in seeds)}")
    for seed in seeds:
        print(
            f"  seed {seed['seed']}: {seed['evaluation']} "
            f"({seed['summary']['results']['policy']['success_rate']:.3f} success)"
        )
    print(f"Results: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
