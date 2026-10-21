"""Metrics and headless PNG reports for recorded baseline episodes.

Plot from saved telemetry with ``python -m SAC_project.reporting RUN_DIRECTORY``.
FigureCanvasAgg avoids GUI backends on both Windows and Linux training hosts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


NUMERIC_FIELDS = (
    "sim_time",
    "rear_x",
    "rear_y",
    "cog_x",
    "cog_y",
    "actual_speed",
    "actual_steering",
    "command_speed",
    "command_steering",
    "reference_speed",
    "rear_error",
    "cog_error",
    "position_error_norm",
    "cog_position_error_norm",
    "speed_error",
    "reference_speed_error",
    "heading_error",
    "steering_error",
    "lap_count",
)


def load_telemetry(path: Path) -> np.ndarray:
    """Read numeric report fields without parsing text modes or empty targets."""
    return np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        usecols=NUMERIC_FIELDS,
        dtype=np.float64,
        encoding="utf-8",
        ndmin=1,
    )


def error_summary(values: np.ndarray, unit: str) -> dict:
    """Aggregate every recorded sample; no startup or collision filtering."""
    values = np.asarray(values, dtype=np.float64)
    return {
        f"rms_{unit}": float(np.sqrt(np.mean(values**2))),
        f"mae_{unit}": float(np.mean(np.abs(values))),
        f"p95_abs_{unit}": float(np.percentile(np.abs(values), 95)),
        f"max_abs_{unit}": float(np.max(np.abs(values))),
        f"mean_{unit}": float(np.mean(values)),
    }


def generate_plots(output_dir: Path, data: np.ndarray, summary: dict) -> list[str]:
    """Save plots of the same samples and applied targets used by the metrics."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output_dir = Path(output_dir)
    time = data["sim_time"]
    generated = []
    clipped = not np.allclose(
        data["reference_speed"], data["command_speed"], rtol=0, atol=1e-6
    )

    def new_figure(figsize):
        figure = Figure(figsize=figsize, layout="constrained")
        FigureCanvasAgg(figure)
        return figure

    def save(figure, filename):
        figure.savefig(output_dir / filename, dpi=160)
        figure.clear()
        generated.append(filename)

    def finish_time_axis(axis, ylabel):
        axis.set_xlabel("Simulation time [s]")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        if summary["collided"]:
            axis.axvline(time[-1], color="crimson", linestyle=":", label="Collision")
        axis.legend(loc="best", fontsize=8)

    def speed_tracking(axis):
        axis.plot(
            time,
            data["reference_speed"],
            color="tab:orange",
            linestyle="--",
            linewidth=1.5,
            label="Desired speed",
        )
        if clipped:
            axis.plot(
                time,
                data["command_speed"],
                color="black",
                linestyle=":",
                linewidth=1.2,
                label="Applied target (clipped)",
            )
        axis.plot(time, data["actual_speed"], linewidth=1.1, label="Actual speed")
        axis.set_title("Desired and actual speed")
        finish_time_axis(axis, "Speed [m/s]")

    def speed_error(axis):
        axis.plot(
            time, data["speed_error"], linewidth=1.0, label="Actual - applied target"
        )
        if clipped:
            axis.plot(
                time,
                data["reference_speed_error"],
                linestyle="--",
                linewidth=1.0,
                label="Actual - desired (before clipping)",
            )
        axis.axhline(0, color="gray", linewidth=0.7)
        axis.set_title(
            f"Speed error | RMSE {summary['speed_error']['rms_mps']:.3f} m/s"
        )
        finish_time_axis(axis, "Speed error [m/s]")

    def position_error(axis):
        axis.plot(
            time,
            data["position_error_norm"],
            linewidth=1.0,
            label="Rear axle (controlled)",
        )
        axis.plot(
            time,
            data["cog_position_error_norm"],
            linewidth=1.0,
            alpha=0.8,
            label="CoG (diagnostic)",
        )
        axis.set_ylim(bottom=0)
        axis.set_title(
            "Distance to closest reference-path point"
            f" | rear RMSE {summary['position_error_norm']['rms_m']:.3f} m"
        )
        finish_time_axis(axis, "Position error norm [m]")

    def lateral_error(axis):
        axis.plot(time, data["rear_error"], linewidth=1.0, label="Rear axle")
        axis.plot(time, data["cog_error"], linewidth=1.0, alpha=0.8, label="CoG")
        axis.axhline(0, color="gray", linewidth=0.7)
        axis.set_title("Signed path error (positive = left)")
        finish_time_axis(axis, "Signed distance [m]")

    def steering_tracking(axis):
        axis.plot(
            time,
            np.rad2deg(data["command_steering"]),
            color="tab:orange",
            linestyle="--",
            linewidth=1.2,
            label="Applied steering target",
        )
        axis.plot(
            time,
            np.rad2deg(data["actual_steering"]),
            linewidth=1.0,
            label="Actual steering",
        )
        axis.set_title("Target and actual steering")
        finish_time_axis(axis, "Steering [deg]")

    def heading_error(axis):
        axis.plot(
            time,
            np.rad2deg(data["heading_error"]),
            linewidth=1.0,
            label="Body yaw - path yaw",
        )
        axis.axhline(0, color="gray", linewidth=0.7)
        axis.set_title("Heading error (wrapped to [-180, 180) deg)")
        finish_time_axis(axis, "Heading error [deg]")

    panels = (
        speed_tracking,
        speed_error,
        position_error,
        lateral_error,
        steering_tracking,
        heading_error,
    )
    figure = new_figure((15, 11))
    axes = figure.subplots(3, 2)
    for axis, draw in zip(axes.flat, panels):
        draw(axis)
    status = "SUCCESS" if summary["success"] else "INCOMPLETE"
    figure.suptitle(
        f"Pure pursuit | {summary['speed_mode']} speed | seed {summary['seed']}"
        f" | {summary['completed_laps']}/{summary['requested_laps']} full laps | {status}\n"
        "All recorded steps, including startup and out lap; commands held over each step",
        fontsize=13,
    )
    save(figure, "performance.png")

    for filename, draw in (
        ("speed_tracking.png", speed_tracking),
        ("position_error.png", position_error),
        ("speed_error.png", speed_error),
    ):
        after_startup = time >= 2.0
        if filename == "speed_error.png" and np.count_nonzero(after_startup) >= 2:
            # Keep the full startup transient, then magnify the smaller errors.
            # Only the displayed axes change; all summary metrics keep all data.
            figure = new_figure((11, 7.5))
            full_axis, zoom_axis = figure.subplots(2, 1)
            draw(full_axis)
            draw(zoom_axis)
            zoom_errors = data["speed_error"][after_startup]
            if clipped:
                zoom_errors = np.r_[
                    zoom_errors, data["reference_speed_error"][after_startup]
                ]
            low, high = float(zoom_errors.min()), float(zoom_errors.max())
            margin = max((high - low) * 0.1, 0.01)
            zoom_axis.set_ylim(low - margin, high + margin)
            zoom_axis.set_xlim(2.0, time[-1])
            zoom_axis.set_title(
                "Speed error after 2 s (display zoom; metrics retain all samples)"
            )
        else:
            figure = new_figure((11, 4.5))
            draw(figure.subplots())
        save(figure, filename)

    reference = np.genfromtxt(
        output_dir / "reference_path.csv", delimiter=",", names=True, ndmin=1
    )
    figure = new_figure((9, 8))
    axis = figure.subplots()
    axis.plot(
        reference["x"], reference["y"], "k--", linewidth=1.4, label="Reference path"
    )
    axis.plot(data["rear_x"], data["rear_y"], linewidth=1.0, label="Rear axle")
    axis.plot(data["cog_x"], data["cog_y"], linewidth=0.8, alpha=0.7, label="CoG")
    axis.scatter(
        data["rear_x"][0], data["rear_y"][0], color="green", s=30, label="First sample"
    )
    axis.scatter(
        data["rear_x"][-1],
        data["rear_y"][-1],
        color="crimson",
        s=30,
        label="Collision" if summary["collided"] else "Last sample",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("World X [m]")
    axis.set_ylabel("World Y [m]")
    axis.set_title("Reference and actual trajectories")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    save(figure, "trajectory.png")
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    summary_path = args.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    data = load_telemetry(args.output_dir / "trajectory.csv")
    summary["plots"] = generate_plots(args.output_dir, data, summary)
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Plots: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
