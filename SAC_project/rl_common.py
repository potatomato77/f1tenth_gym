"""Shared environment settings and run metadata for the SAC scripts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from SAC_project.features import FeatureConfig
from SAC_project.residual_env import ResidualConfig, make_residual_env
from SAC_project.rewards import TrackingRewardConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

ENV_KEYS = (
    "map_name",
    "laps",
    "max_steps",
    "reference",
    "lookahead",
    "speed_mode",
    "speed",
    "vgain",
    "max_reacquire_distance",
    "max_steering_residual_rad",
    "preview_distances_m",
    "yaw_rate_scale_radps",
)

REWARD_KEYS = (
    "progress_weight",
    "lateral_weight",
    "heading_weight",
    "residual_weight",
    "smoothness_weight",
    "failure_cost",
    "lateral_scale_m",
    "heading_scale_rad",
)


@dataclass(frozen=True)
class EnvSettings:
    """Everything that defines the residual task; frozen for one training run."""

    map_name: str = "Spielberg"
    laps: int = 3
    max_steps: int = 100_000
    reference: str = "raceline"
    lookahead: float = 1.2
    speed_mode: str = "profile"
    speed: float = 2.0
    vgain: float = 1.0
    max_reacquire_distance: float = 3.0
    max_steering_residual_rad: float = 0.05
    preview_distances_m: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    yaw_rate_scale_radps: float = 10.0
    reward: TrackingRewardConfig = field(default_factory=TrackingRewardConfig)

    def residual_config(self) -> ResidualConfig:
        return ResidualConfig(
            max_steering_residual_rad=self.max_steering_residual_rad
        )

    def feature_config(self) -> FeatureConfig:
        return FeatureConfig(
            preview_distances_m=self.preview_distances_m,
            yaw_rate_scale_radps=self.yaw_rate_scale_radps,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EnvSettings":
        values = {key: data[key] for key in ENV_KEYS if key in data}
        if "preview_distances_m" in values:
            values["preview_distances_m"] = tuple(
                float(value) for value in values["preview_distances_m"]
            )
        return cls(
            reward=TrackingRewardConfig(**data.get("reward", {})),
            **values,
        )


def build_env(settings: EnvSettings, *, render: bool = False):
    """Construct a fresh residual environment; safe to call in a subprocess."""
    return make_residual_env(
        map_name=settings.map_name,
        laps=settings.laps,
        max_steps=settings.max_steps,
        reference=settings.reference,
        lookahead=settings.lookahead,
        speed_mode=settings.speed_mode,
        speed=settings.speed,
        vgain=settings.vgain,
        max_reacquire_distance=settings.max_reacquire_distance,
        render=render,
        residual_config=settings.residual_config(),
        feature_config=settings.feature_config(),
        reward_config=settings.reward,
    )


def add_env_arguments(parser: argparse.ArgumentParser) -> None:
    """Add task arguments without defaults so callers can layer them over a config."""
    parser.add_argument("--map", dest="map_name", default=argparse.SUPPRESS)
    parser.add_argument("--laps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--reference",
        choices=("raceline", "centerline"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--lookahead", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--speed-mode",
        choices=("profile", "fixed"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--speed", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--vgain", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--max-reacquire-distance", type=float, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--max-residual-rad", type=float, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--preview-distances",
        dest="preview_distances_m",
        default=argparse.SUPPRESS,
        help="comma-separated preview distances in metres, e.g. 1,2,4,8",
    )
    parser.add_argument(
        "--yaw-rate-scale", dest="yaw_rate_scale_radps", type=float,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--progress-weight", type=float, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--lateral-weight", type=float, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--heading-weight", type=float, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--residual-weight", type=float, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--smoothness-weight", type=float, default=argparse.SUPPRESS
    )
    parser.add_argument("--failure-cost", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--lateral-scale-m", dest="lateral_scale_m", type=float,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--heading-scale-rad", dest="heading_scale_rad", type=float,
        default=argparse.SUPPRESS,
    )


def settings_from_args(
    args: argparse.Namespace, base: EnvSettings | None = None
) -> EnvSettings:
    """Apply only the task arguments explicitly present in ``args``."""
    base = base or EnvSettings()
    values = {}
    for key in ENV_KEYS:
        if not hasattr(args, key):
            continue
        value = getattr(args, key)
        if key == "preview_distances_m" and isinstance(value, str):
            value = tuple(float(part) for part in value.split(",") if part.strip())
        values[key] = value
    reward_values = {
        key: getattr(args, key) for key in REWARD_KEYS if hasattr(args, key)
    }
    if reward_values:
        values["reward"] = replace(base.reward, **reward_values)
    return replace(base, **values)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def make_run_dir(prefix: str, root: Path, name: str | None = None) -> Path:
    directory = Path(root) / (name or f"{prefix}_{timestamp()}")
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def write_json(path: Path, data: dict) -> None:
    Path(path).write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _git(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_metadata() -> dict:
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
    }


def version_metadata() -> dict:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in (
        "stable-baselines3",
        "torch",
        "gymnasium",
        "numpy",
        "jax",
        "jaxlib",
        "f1tenth-gym",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return versions


def run_metadata(
    settings: EnvSettings,
    *,
    command: list[str],
    env_description: dict | None = None,
    **extra,
) -> dict:
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": command,
        "working_directory": str(Path.cwd()),
        "git": git_metadata(),
        "versions": version_metadata(),
        "env": settings.to_dict(),
        "env_description": env_description or {},
        **extra,
    }


def describe_env(env) -> dict:
    return env.describe()


def command_line() -> list[str]:
    return [sys.executable, *sys.argv]
