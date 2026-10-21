"""Train SAC on the residual steering task.

Smoke test (pipeline only, no learning claim):

    python -m SAC_project.train --smoke

Formal training should run headless on the Linux server, for example:

    python -m SAC_project.train --total-timesteps 500000 --n-envs 8 --vec-env subproc
"""

from __future__ import annotations

import argparse
from collections import deque
import functools
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)

from SAC_project.rl_common import (
    EnvSettings,
    add_env_arguments,
    build_env,
    command_line,
    make_run_dir,
    run_metadata,
    settings_from_args,
    write_json,
)


REWARD_NAMES = ("progress", "lateral", "heading", "residual", "smoothness", "failure")


class ResidualMetricsCallback(BaseCallback):
    """Aggregate task-level episode outcomes that return logging cannot show."""

    KEYS = (
        "success",
        "collision",
        "tracking_stop",
        "timeout",
        "rear_rms_m",
        "rear_p95_m",
        "heading_rms_rad",
        "mean_abs_z",
        "max_abs_z",
        "clip_fraction",
        "episode_return",
        "steps",
    )

    def __init__(self, n_envs: int, window: int = 50, verbose: int = 0):
        super().__init__(verbose)
        self.n_envs = n_envs
        self.window = window
        self.history = {key: deque(maxlen=window) for key in self.KEYS}
        self.term_history = {
            name: deque(maxlen=window) for name in REWARD_NAMES
        }
        self.episodes = 0
        self._accumulators = [self._empty() for _ in range(n_envs)]

    @staticmethod
    def _empty() -> dict:
        return {
            "steps": 0,
            "rear_sq": 0.0,
            "rear_abs": [],
            "heading_sq": 0.0,
            "abs_z": 0.0,
            "max_abs_z": 0.0,
            "clips": 0,
            "return": 0.0,
            "terms": {name: 0.0 for name in REWARD_NAMES},
        }

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        rewards = self.locals.get("rewards")
        if infos is None or dones is None:
            return True
        for rank, info in enumerate(infos):
            accumulator = self._accumulators[rank]
            if rewards is not None:
                accumulator["return"] += float(rewards[rank])
            residual = info.get("residual")
            if residual:
                accumulator["steps"] += 1
                rear = float(residual["rear_lateral_error_m"])
                accumulator["rear_sq"] += rear * rear
                accumulator["rear_abs"].append(abs(rear))
                heading = float(residual["heading_error_rad"])
                accumulator["heading_sq"] += heading * heading
                z = abs(float(np.asarray(residual["policy_action"]).reshape(-1)[0]))
                accumulator["abs_z"] += z
                accumulator["max_abs_z"] = max(accumulator["max_abs_z"], z)
                accumulator["clips"] += int(residual["steering_clipped"])
                for name in REWARD_NAMES:
                    accumulator["terms"][name] += float(
                        residual["reward_terms"][name]
                    )
            if dones[rank]:
                self._record(rank, info, accumulator)
                self._accumulators[rank] = self._empty()
        return True

    def _record(self, rank: int, info: dict, accumulated: dict) -> None:
        residual = info.get("residual", {})
        steps = max(accumulated["steps"], 1)
        end_reason = residual.get("end_reason", "")
        values = {
            "success": 1.0 if info.get("is_success") else 0.0,
            "collision": 1.0 if end_reason == "collision" else 0.0,
            "tracking_stop": 1.0 if end_reason == "tracking_stop" else 0.0,
            "timeout": 1.0 if end_reason == "time_limit" else 0.0,
            "rear_rms_m": math.sqrt(accumulated["rear_sq"] / steps),
            "rear_p95_m": (
                float(np.percentile(accumulated["rear_abs"], 95))
                if accumulated["rear_abs"]
                else 0.0
            ),
            "heading_rms_rad": math.sqrt(accumulated["heading_sq"] / steps),
            "mean_abs_z": accumulated["abs_z"] / steps,
            "max_abs_z": accumulated["max_abs_z"],
            "clip_fraction": accumulated["clips"] / steps,
            "episode_return": accumulated["return"],
            "steps": float(steps),
        }
        self.episodes += 1
        for key, value in values.items():
            self.history[key].append(value)
        for name in REWARD_NAMES:
            self.term_history[name].append(accumulated["terms"][name] / steps)
        prefix = "residual"
        self.logger.record(f"{prefix}/episodes", self.episodes)
        for key, values_window in self.history.items():
            self.logger.record(
                f"{prefix}/{key}"
                if key != "episode_return"
                else "rollout/episode_return",
                float(np.mean(values_window)),
            )
        for name, values_window in self.term_history.items():
            self.logger.record(
                f"{prefix}/reward_{name}_mean", float(np.mean(values_window))
            )
        if self.verbose >= 2:
            print(
                f"episode {self.episodes} env {rank}: {end_reason or 'ended'}, "
                f"return {values['episode_return']:.3f}, "
                f"rear RMS {values['rear_rms_m']:.4f} m"
            )


class CheckpointCallback(BaseCallback):
    """Save model (and optionally replay buffer) at a fixed env-step interval."""

    def __init__(
        self,
        save_freq: int,
        save_dir: Path,
        name_prefix: str = "sac_checkpoint",
        save_replay: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir = Path(save_dir)
        self.name_prefix = name_prefix
        self.save_replay = save_replay
        self._last_save = 0

    def _on_step(self) -> bool:
        if self.save_freq <= 0:
            return True
        if self.num_timesteps - self._last_save < self.save_freq:
            return True
        stem = self.save_dir / f"{self.name_prefix}_{self.num_timesteps}_steps"
        self.model.save(stem)
        if self.save_replay:
            self.model.save_replay_buffer(stem.with_name(stem.name + "_replay"))
        self._last_save = self.num_timesteps
        if self.verbose:
            print(f"checkpoint saved: {stem}.zip")
        return True


def create_env(settings: EnvSettings):
    return build_env(settings)


def build_vec_env(
    settings: EnvSettings, n_envs: int, backend: str
) -> VecMonitor:
    factories = [
        functools.partial(create_env, settings) for _ in range(n_envs)
    ]
    if backend == "subproc":
        environment = SubprocVecEnv(factories)
    else:
        environment = DummyVecEnv(factories)
    return VecMonitor(environment)


def parse_hidden_sizes(text: str) -> list[int]:
    sizes = [int(part) for part in text.split(",") if part.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("hidden sizes must be positive integers")
    return sizes


def parse_ent_coef(text: str) -> str | float:
    if text == "auto":
        return text
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError("entropy coefficient must be non-negative")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_env_arguments(parser)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument(
        "--vec-env", choices=("dummy", "subproc"), default="dummy"
    )
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=parse_ent_coef, default="auto")
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument(
        "--policy-hidden-sizes", type=parse_hidden_sizes, default=[256, 256]
    )
    parser.add_argument("--checkpoint-freq", type=int, default=0)
    parser.add_argument(
        "--checkpoint-replay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also save the replay buffer at each checkpoint (large)",
    )
    parser.add_argument(
        "--save-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save the final replay buffer for resume",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-replay", type=Path)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent / "runs")
    parser.add_argument("--run-name")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="force a 64-step pipeline check with tiny buffers and CPU",
    )
    args = parser.parse_args(argv)
    if args.smoke:
        args.total_timesteps = 64
        args.learning_starts = 16
        args.batch_size = 16
        args.buffer_size = 256
        args.n_envs = 1
        args.vec_env = "dummy"
        args.device = "cpu"
        args.checkpoint_freq = 32
        args.log_interval = 1
    if args.n_envs < 1:
        parser.error("n-envs must be positive")
    if args.total_timesteps < 1:
        parser.error("total-timesteps must be positive")
    if args.batch_size < 1 or args.buffer_size < args.batch_size:
        parser.error("buffer-size must be at least batch-size")
    if args.total_timesteps <= args.learning_starts:
        parser.error("total-timesteps must exceed learning-starts to optimize")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    settings = settings_from_args(args)
    prefix = "sac_smoke" if args.smoke else "sac"
    run_name = args.run_name or f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
    run_dir = make_run_dir(prefix, args.output_root, run_name)

    probe = build_env(settings)
    try:
        description = probe.describe()
    finally:
        probe.close()

    config = run_metadata(
        settings,
        command=command_line(),
        env_description=description,
        train={
            "total_timesteps": args.total_timesteps,
            "seed": args.seed,
            "n_envs": args.n_envs,
            "vec_env": args.vec_env,
            "learning_starts": args.learning_starts,
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "gamma": args.gamma,
            "tau": args.tau,
            "learning_rate": args.learning_rate,
            "ent_coef": args.ent_coef,
            "train_freq": args.train_freq,
            "gradient_steps": args.gradient_steps,
            "policy_hidden_sizes": args.policy_hidden_sizes,
            "device": args.device,
            "smoke": args.smoke,
            "resume": str(args.resume) if args.resume else None,
        },
    )
    config_path = run_dir / "train_config.json"
    write_json(config_path, config)

    environment = build_vec_env(settings, args.n_envs, args.vec_env)
    logger = configure(str(run_dir / "logs"), ["stdout", "csv"])
    callbacks = [
        ResidualMetricsCallback(n_envs=args.n_envs, window=50, verbose=args.verbose)
    ]
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_dir=run_dir,
                save_replay=args.checkpoint_replay,
                verbose=args.verbose,
            )
        )

    started = time.perf_counter()
    try:
        if args.resume:
            model_path = args.resume
            if not model_path.exists() and model_path.with_suffix(".zip").exists():
                model_path = model_path.with_suffix(".zip")
            model = SAC.load(str(model_path), env=environment, device=args.device)
            if args.resume_replay:
                model.load_replay_buffer(str(args.resume_replay))
            if (
                model.observation_space.shape != environment.observation_space.shape
                or model.action_space.shape != environment.action_space.shape
            ):
                raise SystemExit(
                    "resumed model spaces do not match the current residual task"
                )
        else:
            model = SAC(
                "MlpPolicy",
                environment,
                learning_rate=args.learning_rate,
                buffer_size=args.buffer_size,
                learning_starts=args.learning_starts,
                batch_size=args.batch_size,
                tau=args.tau,
                gamma=args.gamma,
                train_freq=args.train_freq,
                gradient_steps=args.gradient_steps,
                ent_coef=args.ent_coef,
                policy_kwargs={"net_arch": args.policy_hidden_sizes},
                seed=args.seed,
                device=args.device,
                verbose=args.verbose,
            )
        model.set_logger(logger)
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=args.log_interval,
            reset_num_timesteps=not bool(args.resume),
            progress_bar=args.progress_bar,
        )
        duration = time.perf_counter() - started
        final_model = run_dir / "sac_final"
        model.save(final_model)
        if args.save_replay:
            model.save_replay_buffer(str(run_dir / "replay_buffer"))
        logger.record("time/training_seconds", duration)
        logger.record("time/final_timesteps", float(model.num_timesteps))
        logger.dump(step=model.num_timesteps)
        config["result"] = {
            "num_timesteps": int(model.num_timesteps),
            "training_seconds": duration,
            "device": str(model.device),
            "model": str(final_model.with_suffix(".zip")),
            "replay_buffer": (
                str(run_dir / "replay_buffer.pkl") if args.save_replay else None
            ),
        }
        write_json(config_path, config)
    finally:
        environment.close()

    print(f"\nModel: {final_model}.zip")
    print(f"Config: {config_path}")
    print(f"Training seconds: {duration:.2f}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
