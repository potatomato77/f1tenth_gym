"""Single-car Gymnasium residual task over the unchanged mutable simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.action import LongitudinalActionType, SteerActionType
from f1tenth_gym.envs.env_config import EnvConfig
from f1tenth_gym.envs.wrappers import SingleAgentWrapper
from SAC_project.baseline import PurePursuitController, build_config
from SAC_project.features import FeatureConfig, FeatureExtractor
from SAC_project.rewards import TrackingRewardConfig, tracking_reward


@dataclass(frozen=True)
class ResidualConfig:
    max_steering_residual_rad: float = 0.05

    def __post_init__(self):
        if (
            not math.isfinite(self.max_steering_residual_rad)
            or self.max_steering_residual_rad <= 0
        ):
            raise ValueError("maximum steering residual must be finite and positive")


class ResidualEnv(gym.Wrapper):
    """Accept normalized residual z, return ordered state-and-map features.

    Wrap a SingleAgentWrapper whose observation contains std_state. One policy
    step is exactly one native physics step. The controller is cached from the
    current state, never from the previous step. No training library is imported.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        controller: PurePursuitController,
        residual_config: ResidualConfig = ResidualConfig(),
        feature_config: FeatureConfig = FeatureConfig(),
        reward_config: TrackingRewardConfig = TrackingRewardConfig(),
    ):
        super().__init__(env)
        cfg = env.unwrapped.env_config
        if (
            env.action_space.shape != (2,)
            or "std_state" not in env.observation_space.spaces
        ):
            raise ValueError(
                "wrap the native environment in SingleAgentWrapper first; require std_state"
            )
        if cfg.domain_randomization_config.enabled:
            raise ValueError(
                "this first residual task requires fixed vehicle parameters (DR disabled)"
            )
        if (
            cfg.control_config.longitudinal_mode is not LongitudinalActionType.SPEED
            or cfg.control_config.steering_mode is not SteerActionType.STEERING_ANGLE
        ):
            raise ValueError(
                "residual task requires target-speed and target-steering-angle control"
            )
        if not cfg.termination_config.terminate_on_collision:
            raise ValueError("residual task requires collision termination")
        for name in ("lf", "lr", "s_min", "s_max", "v_min", "v_max"):
            if getattr(controller.vehicle, name) != getattr(cfg.params, name):
                raise ValueError(f"controller and environment disagree on {name}")
        self.controller = controller
        self.residual_config = residual_config
        self.reward_config = reward_config
        self.features = FeatureExtractor(controller, feature_config)
        self.feature_names = self.features.names
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -1.0, 1.0, shape=(len(self.feature_names),), dtype=np.float32
        )
        self.dt = float(env.unwrapped.timestep)
        self._needs_reset = True
        self._state = None
        self._command = None
        self._frame = None
        self._previous_z = 0.0
        self._previous_action = np.zeros(2, dtype=np.float32)

    @property
    def baseline_action(self):
        if self._needs_reset:
            raise gym.error.ResetNeeded("reset before requesting a baseline action")
        return self._command.action.copy()

    def describe(self) -> dict:
        """Serializable action, observation and reward contract for saved runs."""
        return {
            "residual": asdict(self.residual_config),
            "features": asdict(self.features.config),
            "reward": asdict(self.reward_config),
            "feature_names": list(self.feature_names),
            "feature_scales": {
                "lateral_m": self.features.lateral_scale,
                "speed_mps": self.features.speed_scale,
                "steering_rad": self.features.steering_scale,
                "yaw_rate_radps": self.features.config.yaw_rate_scale_radps,
                "preview_position_m": self.features.preview_scale,
            },
            "baseline": {
                "speed_mode": self.controller.speed_mode,
                "fixed_speed_mps": self.controller.speed,
                "vgain": self.controller.vgain,
                "lookahead_m": self.controller.lookahead,
                "max_reacquire_distance_m": self.controller.max_reacquire_distance,
            },
            "physical_action_low": self.controller.action_low.tolist(),
            "physical_action_high": self.controller.action_high.tolist(),
            "timestep_s": self.dt,
            "action_repeat": 1,
        }

    def _observe_state(self, observation):
        self._state = np.array(observation["std_state"], dtype=np.float32, copy=True)
        self._command = self.controller.plan(self._state[[0, 1, 4]])
        self._frame = self.features.observe(
            self._state, self._command, self._previous_z, self._previous_action
        )

    def _state_info(self):
        return {
            "std_state": self._state.copy(),
            "next_base_action": self._command.action.copy(),
            "rear_position": self._command.rear_position.copy(),
            "rear_lateral_error_m": self._command.rear_error,
            "position_error_norm_m": self._command.projection.distance,
            "heading_error_rad": self._frame.heading_error_rad,
            "path_s_m": self._frame.path_s_m,
            "next_tracking_mode": self._command.mode,
            "next_stop_reason": self._command.reason,
            "feature_clip_count": self._frame.clipped_features,
        }

    def reset(self, *, seed=None, options=None):
        self._needs_reset = True
        observation, info = self.env.reset(seed=seed, options=options)
        self._previous_z = 0.0
        self._previous_action = np.zeros(2, dtype=np.float32)
        self._observe_state(observation)
        self._needs_reset = False
        info = dict(info)
        info["is_success"] = False
        info["residual"] = {
            **self._state_info(),
            "base_action": self._command.action.copy(),
            "policy_action": np.zeros(1, dtype=np.float32),
            "physical_action": self._previous_action.copy(),
            "reward_terms": {},
            "progress_raw_m": 0.0,
            "progress_used_m": 0.0,
        }
        return self._frame.observation.copy(), info

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded(
                "reset before stepping, including after episode end"
            )
        action = np.asarray(action, dtype=np.float64)
        if (
            action.shape != (1,)
            or not np.all(np.isfinite(action))
            or np.any(np.abs(action) > 1)
        ):
            raise ValueError("policy action must be a finite (1,) residual in [-1, 1]")
        z = float(np.float32(action[0]))
        previous_z = self._previous_z
        previous_s = self._frame.path_s_m
        base = self._command
        requested_residual = self.residual_config.max_steering_residual_rad * z
        raw_steering = float(base.action[0]) + requested_residual
        if base.mode == "stop":
            # A residual cannot override the baseline's invalid-target stop.
            physical = base.action.copy()
        else:
            physical = self.controller.limit_action([raw_steering, base.action[1]])
        self._needs_reset = True
        observation, base_reward, base_terminated, base_truncated, info = self.env.step(
            physical
        )
        self._previous_z = z
        self._previous_action = physical.copy()
        self._observe_state(observation)
        collision = bool(info["collisions"][0])
        tracking_stop = base.mode == "stop" or self._command.mode == "stop"
        terminated = bool(base_terminated or tracking_stop)
        truncated = bool(base_truncated)
        reward = tracking_reward(
            previous_s=previous_s,
            current_s=self._frame.path_s_m,
            path_length=float(self.controller.arclength[-1]),
            dt=self.dt,
            speed_limit=self.features.speed_scale,
            lateral_error=self._command.rear_error,
            heading_error=self._frame.heading_error_rad,
            z=z,
            previous_z=previous_z,
            failed=collision or tracking_stop,
            config=self.reward_config,
        )
        max_laps = self.env.unwrapped.env_config.simulation_config.max_laps
        success = bool(
            max_laps is not None
            and int(info["lap_counts"][0]) >= max_laps
            and not collision
            and not tracking_stop
        )
        reason = (
            "collision"
            if collision
            else "tracking_stop"
            if tracking_stop
            else "lap_complete"
            if success
            else "base_termination"
            if terminated
            else "time_limit"
            if truncated
            else ""
        )
        self._needs_reset = terminated or truncated
        info = dict(info)
        info["is_success"] = success
        info["residual"] = {
            **self._state_info(),
            "base_action": base.action.copy(),
            "base_raw_steering_rad": base.raw_steering,
            "policy_action": action.astype(np.float32),
            "requested_steering_residual_rad": requested_residual,
            "effective_steering_residual_rad": float(physical[0])
            - float(base.action[0]),
            "physical_action": physical.copy(),
            "steering_clipped": bool(
                base.mode != "stop"
                and (
                    raw_steering < self.controller.action_low[0]
                    or raw_steering > self.controller.action_high[0]
                )
            ),
            "base_steering_clipped": base.steering_clipped,
            "base_speed_clipped": base.speed_clipped,
            "residual_ignored": base.mode == "stop",
            "tracking_stop": tracking_stop,
            "end_reason": reason,
            "base_terminated": bool(base_terminated),
            "base_truncated": bool(base_truncated),
            "base_reward": float(base_reward),
            "reward_terms": reward.terms.copy(),
            "progress_raw_m": reward.raw_progress_m,
            "progress_used_m": reward.used_progress_m,
            "progress_clipped": reward.raw_progress_m != reward.used_progress_m,
        }
        return self._frame.observation.copy(), reward.total, terminated, truncated, info


def make_residual_env(
    *,
    env_config: EnvConfig | None = None,
    map_name="Spielberg",
    laps=3,
    max_steps=100_000,
    reference="raceline",
    lookahead=1.2,
    speed_mode="profile",
    speed=2.0,
    vgain=1.0,
    max_reacquire_distance=3.0,
    render=False,
    residual_config=ResidualConfig(),
    feature_config=FeatureConfig(),
    reward_config=TrackingRewardConfig(),
) -> ResidualEnv:
    """Construct the same native env as the baseline, then add task wrappers."""
    cfg = (
        env_config
        if env_config is not None
        else build_config(
            map_name=map_name,
            laps=laps,
            max_steps=max_steps,
            reference=reference,
            render=render,
        )
    )
    base = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=cfg,
        render_mode="human" if render else None,
    )
    try:
        controller = PurePursuitController.from_track(
            base.unwrapped.track,
            vehicle=cfg.params,
            reference=reference,
            lookahead=lookahead,
            speed=speed,
            speed_mode=speed_mode,
            vgain=vgain,
            max_reacquire_distance=max_reacquire_distance,
        )
        return ResidualEnv(
            SingleAgentWrapper(base),
            controller=controller,
            residual_config=residual_config,
            feature_config=feature_config,
            reward_config=reward_config,
        )
    except Exception:
        base.close()
        raise
