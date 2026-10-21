"""Typed configuration structures for the simplified F1TENTH gym environment."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Callable, Optional

import math
import warnings

import numpy as np

from .integrators import IntegratorType
from .dynamic_models import (
    DynamicModel,
    PARAMETER_ORDER,
    VehicleParameters,
    F1TENTH_VEHICLE_PARAMETERS,
)
from .action import LongitudinalActionType, SteerActionType
from .observation import ObservationType
from .reset import ReferenceLine, ResetStrategy
from .contact import ContactConfig
from .lidar import LiDARConfig
from .collision_models import CollisionCheckMode
from .termination import AgentTerminationMode

if TYPE_CHECKING:
    from .track import Track

# Type aliases for callables used throughout the simulator
DynamicsFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
StandardStateFn = Callable[[np.ndarray], np.ndarray]
IntegratorFn = Callable[[DynamicsFn, np.ndarray, np.ndarray, float, np.ndarray], np.ndarray]


class LoopCounterMode(IntEnum):
    """Mode for counting completed laps.

    FRENET_BASED: Frenet progress along the centerline (the default).
    WINDING_ANGLE: Cumulative angle around the centroid; convex tracks only.
    """

    FRENET_BASED = 2
    WINDING_ANGLE = 3


class RewardMode(IntEnum):
    """How the per-step scalar reward is computed (see ``RewardConfig``).

    - ``SURVIVAL``: reward = timestep (the default).
    - ``PROGRESS``: weighted arclength, speed, survival, collision. Needs Frenet.
    - ``CUSTOM``: reward = reward_fn(obs, action, info, terminated, truncated).
    """

    SURVIVAL = 0
    PROGRESS = 1
    CUSTOM = 2


@dataclass(frozen=True)
class ControlConfig:
    """Configuration for vehicle control inputs.

    Attributes:
        longitudinal_mode: How longitudinal control is interpreted (speed or acceleration).
        steering_mode: How steering control is interpreted (angle or angular velocity).
        steer_delay_steps: Number of timesteps to delay steering commands.
        throttle_delay_steps: Number of timesteps to delay longitudinal commands
            (models drivetrain/ESC lag; steering already had its own delay).
        steer_noise_std: Std of Gaussian noise added to the steering command
            each step (actuator/servo noise).
        accl_noise_std: Std of Gaussian noise added to the longitudinal command
            each step.
        steer_kp: Gain of the STEERING_ANGLE P controller
            (``sv = clip(kp * error, -sv_max, sv_max)``). ``None`` derives
            ``10 * sv_max / (s_max - s_min)`` from the vehicle limits;
            ``<= 0`` selects the legacy bang-bang relay.
    """

    longitudinal_mode: LongitudinalActionType = LongitudinalActionType.SPEED
    steering_mode: SteerActionType = SteerActionType.STEERING_ANGLE
    steer_delay_steps: int = 0
    throttle_delay_steps: int = 0
    steer_noise_std: float = 0.0
    accl_noise_std: float = 0.0
    steer_kp: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.longitudinal_mode, LongitudinalActionType):
            raise TypeError("longitudinal_mode must be a LongitudinalActionType")
        if not isinstance(self.steering_mode, SteerActionType):
            raise TypeError("steering_mode must be a SteerActionType")
        # the delay fields index ring buffers, so coerce rather than accept 2.7
        object.__setattr__(self, "steer_delay_steps", int(self.steer_delay_steps))
        object.__setattr__(self, "throttle_delay_steps", int(self.throttle_delay_steps))
        if self.steer_delay_steps < 0:
            raise ValueError(f"steer_delay_steps must be >= 0, got {self.steer_delay_steps}")
        if self.throttle_delay_steps < 0:
            raise ValueError(f"throttle_delay_steps must be >= 0, got {self.throttle_delay_steps}")
        if self.steer_noise_std < 0:
            raise ValueError(f"steer_noise_std must be >= 0, got {self.steer_noise_std}")
        if self.accl_noise_std < 0:
            raise ValueError(f"accl_noise_std must be >= 0, got {self.accl_noise_std}")
        if self.steer_kp is not None and not math.isfinite(self.steer_kp):
            # NaN defeats both guards in pid_steer (`kp <= 0` is False for NaN),
            # so the steering angle comes back NaN from the first step.
            raise ValueError(f"steer_kp must be finite, got {self.steer_kp}")

    def with_updates(self, **changes: Any) -> "ControlConfig":
        return replace(self, **changes)


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for physics simulation.

    Attributes:
        timestep: Simulation timestep in seconds.
        integrator_timestep: Integration timestep (can be smaller than timestep).
        integrator: Numerical integration method (Euler or RK4).
        dynamics_model: Vehicle dynamics model to use.
        loop_counter: Method for counting completed laps.
        compute_frenet_frame: Whether to compute Frenet frame coordinates.
        max_laps: Maximum laps before episode ends (None for infinite).
        count_partial_first_lap: Whether the first finish-line crossing completes
            a lap. True counts every crossing, so a car spawned mid-track scores a
            short first lap. False treats the spawn-to-line stretch as an out lap:
            the first crossing only starts the timer, so every lap time is a full
            circuit and ``max_laps`` always means that many complete laps.
    """

    timestep: float = 0.01
    integrator_timestep: float = 0.01
    integrator: IntegratorType = IntegratorType.RK4
    dynamics_model: DynamicModel = DynamicModel.ST
    loop_counter: LoopCounterMode = LoopCounterMode.FRENET_BASED
    compute_frenet_frame: bool = True
    max_laps: Optional[int] = 1
    count_partial_first_lap: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "count_partial_first_lap", bool(self.count_partial_first_lap))
        if self.timestep <= 0:
            raise ValueError(f"timestep must be > 0, got {self.timestep}")
        if self.integrator_timestep <= 0:
            raise ValueError(f"integrator_timestep must be > 0, got {self.integrator_timestep}")
        if self.max_laps is not None and self.max_laps < 1:
            raise ValueError(f"max_laps must be >= 1 or None, got {self.max_laps}")

    def with_updates(self, **changes: Any) -> "SimulationConfig":
        updated = replace(self, **changes)
        if updated.loop_counter is LoopCounterMode.FRENET_BASED and not updated.compute_frenet_frame:
            updated = replace(updated, compute_frenet_frame=True)
        return updated


@dataclass(frozen=True)
class ObservationConfig:
    """Configuration for environment observations.

    Attributes:
        type: Observation format type.
        features: Specific features to include (None for all).
    """

    type: ObservationType = ObservationType.DEFAULT
    features: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        # Every other type uses a fixed preset, so silently ignoring `features`
        # would hide a config mistake (the user probably meant type=FEATURES).
        if self.features is not None and self.type is not ObservationType.FEATURES:
            raise ValueError(
                f"observation `features` only applies to ObservationType.FEATURES, "
                f"but type={self.type.name}. Use ObservationType.FEATURES to select a "
                f"custom field subset, or drop `features`."
            )

    def with_updates(self, **changes: Any) -> "ObservationConfig":
        return replace(self, **changes)


@dataclass(frozen=True)
class ResetConfig:
    """Configuration for episode reset behavior.

    Attributes:
        strategy: Reset strategy for initial agent positions.
        min_dist: Minimum spacing between agents' spawn points, in metres.
            ``None`` = the strategy's default (1.5 for RL, 0.5 for map).
        max_dist: Maximum spacing between agents' spawn points, in metres.
            ``None`` = the strategy's default (2.5 for RL, 1.0 for map).
        shuffle: Override whether agent spawn order is shuffled. ``None`` uses
            the strategy default (STATIC strategies don't shuffle).
        move_laterally: Override whether spawns are offset laterally off the
            reference line. ``None`` uses the strategy default.
        reference_line: Which line the RL_* strategies spawn on. ``None`` =
            ``ReferenceLine.RACELINE`` (today's behaviour); ``CENTERLINE``
            gives ``ey == 0`` at spawn. Not applicable to MAP strategies.
        start_width: Length of the grid strategies' spawn window along the
            reference line, in absolute metres (default 1.0). The window is
            clamped to at least one waypoint, so scaled maps stay legal.
            Only applicable to the two GRID strategies.
    """

    strategy: ResetStrategy = ResetStrategy.RL_GRID_STATIC
    min_dist: Optional[float] = None
    max_dist: Optional[float] = None
    shuffle: Optional[bool] = None
    move_laterally: Optional[bool] = None
    reference_line: Optional[ReferenceLine] = None
    start_width: Optional[float] = None

    def __post_init__(self) -> None:
        if self.min_dist is not None and self.min_dist < 0:
            raise ValueError(f"min_dist must be >= 0, got {self.min_dist}")
        if self.max_dist is not None and self.max_dist <= 0:
            raise ValueError(f"max_dist must be > 0, got {self.max_dist}")
        if (
            self.min_dist is not None
            and self.max_dist is not None
            and self.min_dist >= self.max_dist
        ):
            raise ValueError(
                f"min_dist ({self.min_dist}) must be < max_dist ({self.max_dist})"
            )
        if self.reference_line is not None:
            if not isinstance(self.reference_line, ReferenceLine):
                raise TypeError("reference_line must be a ReferenceLine")
            # Only _rl_reset_factory consumes this key, so allow-list RL_*: a
            # second MAP strategy would die on an unexpected kwarg in gym.make.
            if self.strategy not in (
                ResetStrategy.RL_GRID_STATIC,
                ResetStrategy.RL_RANDOM_STATIC,
                ResetStrategy.RL_GRID_RANDOM,
                ResetStrategy.RL_RANDOM_RANDOM,
            ):
                raise ValueError("reference_line only applies to the RL_* strategies")
        if self.start_width is not None:
            if self.start_width <= 0:
                raise ValueError(f"start_width must be > 0, got {self.start_width}")
            if self.strategy not in (
                ResetStrategy.RL_GRID_STATIC,
                ResetStrategy.RL_GRID_RANDOM,
            ):
                raise ValueError("start_width only applies to the GRID strategies")

    def reset_kwargs(self) -> dict:
        """Non-None reset params, to forward to ``make_reset_fn``."""
        keys = ("min_dist", "max_dist", "shuffle", "move_laterally", "reference_line", "start_width")
        return {k: getattr(self, k) for k in keys if getattr(self, k) is not None}

    def with_updates(self, **changes: Any) -> "ResetConfig":
        return replace(self, **changes)


@dataclass(frozen=True)
class RenderConfig:
    """Configuration for rendering pacing and frame output.

    Rendering is decoupled from stepping: the environment does not own the
    step loop (the user does), so pacing and frame emission are governed by a
    small render clock driven from ``F110Env.render()``. See ``RenderClock``.

    Attributes:
        render_fps: Target fixed frame rate. In human modes this caps redraws
            to at most ``render_fps`` per wall-clock second (so stepping the
            dynamics faster than real time does not force more frames). In
            rgb_array mode it sets the distinct-frame cadence in *sim* time
            (a fresh frame is grabbed every ``1/render_fps`` sim-seconds; the
            cached frame is returned in between). This is NOT the RecordVideo
            container fps -- that stays ``round(1/timestep)`` for real-time
            playback (see ``F110Env``).
        real_time_factor: Sim-seconds simulated per wall-clock second in human
            modes. ``1.0`` = real time, ``5.0`` = 5x faster, ``float("inf")``
            = no pacing (free-run). Ignored in rgb_array mode (never paces).
            Togglable at runtime via ``F110Env.set_real_time_factor``.
        window_size: Square render/frame size in pixels (also the recorded
            rgb_array / video resolution).
        focus_on: Agent id the camera follows (e.g. ``"agent_0"``); ``None`` =
            map view.
        vehicle_palette: Per-agent car colours (hex strings, cycled by index).
        show_wheels: Draw the wheels on each car.
        render_map_img: Draw the occupancy-map image under the scene.
        car_thickness: Car outline thickness in pixels.
        bigger_car_when_map_centered: Scale cars up in the zoomed-out map view.
        show_lap_info: Overlay the lap-time / lap-count label on the frame.

    Rendering uses the OpenGL backend with the native Windows desktop or an
    X display on Linux (real or virtual via ``xvfb``); see ``make_renderer`` for
    headless/Colab setup.
    """

    render_fps: int = 60
    real_time_factor: float = 1.0
    window_size: int = 800
    focus_on: Optional[str] = "agent_0"
    vehicle_palette: tuple[str, ...] = (
        "#FD3754", "#377eb8", "#984ea3", "#e41a1c", "#ff7f00",
        "#a65628", "#f781bf", "#888888", "#a6cee3", "#b2df8a",
    )
    show_wheels: bool = True
    render_map_img: bool = True
    car_thickness: int = 1
    bigger_car_when_map_centered: bool = True
    show_lap_info: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "render_fps", int(self.render_fps))
        object.__setattr__(self, "real_time_factor", float(self.real_time_factor))
        object.__setattr__(self, "window_size", int(self.window_size))
        object.__setattr__(self, "car_thickness", int(self.car_thickness))
        if self.render_fps <= 0:
            raise ValueError(f"render_fps must be > 0, got {self.render_fps}")
        if not (self.real_time_factor > 0):
            raise ValueError(
                f"real_time_factor must be > 0 (or float('inf')), got {self.real_time_factor}"
            )
        if self.window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {self.window_size}")

    def with_updates(self, **changes: Any) -> "RenderConfig":
        return replace(self, **changes)


@dataclass(frozen=True)
class TerminationConfig:
    """Configuration for episode termination and truncation.

    Attributes:
        max_episode_steps: If set, the episode is truncated (``truncated=True``)
            once this many steps have elapsed since reset. ``None`` = no limit.
        terminate_on_collision: Whether a collision ends the episode
            (``terminated=True``).
        agent_mode: How per-agent collision and lap-completion status is reduced
            to the scalar ``terminated`` result.
    """

    max_episode_steps: Optional[int] = None
    terminate_on_collision: bool = True
    agent_mode: AgentTerminationMode = AgentTerminationMode.EGO

    def __post_init__(self) -> None:
        if self.max_episode_steps is not None and self.max_episode_steps < 1:
            raise ValueError(
                f"max_episode_steps must be >= 1 or None, got {self.max_episode_steps}"
            )
        if not isinstance(self.agent_mode, AgentTerminationMode):
            raise TypeError("agent_mode must be an AgentTerminationMode")

    def with_updates(self, **changes: Any) -> "TerminationConfig":
        return replace(self, **changes)


@dataclass(frozen=True)
class RewardConfig:
    """Configuration for the per-step scalar reward.

    Whatever the mode, ``info`` always carries the raw signals (``progress`` =
    per-agent forward Frenet arclength this step, ``collisions`` = per-agent
    flags) so external code (e.g. f1tenth_learning) can compute any reward.

    Attributes:
        mode: RewardMode (SURVIVAL / PROGRESS / CUSTOM). Default SURVIVAL keeps
            ``reward = timestep``.
        progress_weight: reward per metre of forward arclength (PROGRESS mode).
        velocity_weight: reward per m/s of ego speed (PROGRESS mode).
        timestep_weight: survival bonus per step, scaled by timestep (PROGRESS).
        collision_penalty: subtracted on a step where the ego is colliding
            (PROGRESS mode); must be >= 0.
        reward_fn: for CUSTOM mode, a callable
            ``(obs, action, info, terminated, truncated) -> float``.
    """

    mode: RewardMode = RewardMode.SURVIVAL
    progress_weight: float = 1.0
    velocity_weight: float = 0.0
    timestep_weight: float = 0.0
    collision_penalty: float = 0.0
    reward_fn: Optional[Callable] = None

    def __post_init__(self) -> None:
        if self.collision_penalty < 0:
            raise ValueError(f"collision_penalty must be >= 0, got {self.collision_penalty}")
        if self.mode is RewardMode.CUSTOM and self.reward_fn is None:
            raise ValueError("RewardMode.CUSTOM requires reward_fn to be set")
        if self.reward_fn is not None and not callable(self.reward_fn):
            raise TypeError("reward_fn must be callable")

    def with_updates(self, **changes: Any) -> "RewardConfig":
        return replace(self, **changes)


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """Per-episode randomization of vehicle parameters, applied at ``reset()``.

    The range is two ordinary :class:`VehicleParameters` giving per-field bounds
    in ABSOLUTE units; a field whose ``low`` and ``high`` agree is not randomized.
    Spaces come from :meth:`widest_params`, a fixed superset of every episode.

    Attributes:
        enabled: Whether to randomize at each reset.
        low: Per-field lower bound. Required when ``enabled``.
        high: Per-field upper bound. Required when ``enabled``.
    """

    enabled: bool = False
    low: Optional[VehicleParameters] = None
    high: Optional[VehicleParameters] = None

    def __post_init__(self) -> None:
        for name in ("low", "high"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, VehicleParameters):
                raise TypeError(f"{name} must be a VehicleParameters instance")
        if self.enabled and (self.low is None or self.high is None):
            raise ValueError(
                "domain randomization needs both low and high VehicleParameters "
                "when enabled (a field with low == high is simply not randomized)"
            )
        if self.low is None or self.high is None:
            return
        for name in PARAMETER_ORDER:
            lo, hi = getattr(self.low, name), getattr(self.high, name)
            lo_finite, hi_finite = math.isfinite(lo), math.isfinite(hi)
            if lo_finite != hi_finite:
                raise ValueError(
                    f"low.{name} and high.{name} must both be finite or both not, "
                    f"got {lo!r} and {hi!r}"
                )
            if lo_finite and lo > hi:
                raise ValueError(
                    f"low.{name} must be <= high.{name}, got {lo!r} > {hi!r}"
                )

    def randomized_fields(self) -> tuple[str, ...]:
        """Names of the fields that actually vary, i.e. where ``low != high``.

        A field left at ``nan`` on both sides (the whole multi-body block, on the
        small-scale presets) does NOT count as varying, even though ``nan != nan``.
        """

        def varies(lo: float, hi: float) -> bool:
            if math.isnan(lo) and math.isnan(hi):
                return False
            return lo != hi

        if not self.enabled or self.low is None or self.high is None:
            return ()
        return tuple(
            name
            for name in PARAMETER_ORDER
            if varies(getattr(self.low, name), getattr(self.high, name))
        )

    def bounds_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """``(low, high)`` as float64 arrays in ``PARAMETER_ORDER``, for sampling."""
        return (
            np.array([getattr(self.low, n) for n in PARAMETER_ORDER], dtype=np.float64),
            np.array([getattr(self.high, n) for n in PARAMETER_ORDER], dtype=np.float64),
        )

    def widest_params(self, base: VehicleParameters) -> VehicleParameters:
        """``base`` with each randomized *limit* field at its widest extreme.

        Spaces built from the result are a fixed valid superset of every
        randomized episode, as gymnasium requires. With DR disabled this returns
        ``base`` unchanged.
        """
        if not self.enabled or self.low is None or self.high is None:
            return base
        varying = set(self.randomized_fields())
        changes = {n: getattr(self.low, n) for n in _WIDEN_AT_LOW if n in varying}
        changes.update({n: getattr(self.high, n) for n in _WIDEN_AT_HIGH if n in varying})
        return base.with_updates(**changes) if changes else base

    def with_updates(self, **changes: Any) -> "DomainRandomizationConfig":
        return replace(self, **changes)


# Which extreme of a randomized range widens the space built from it. Params
# in neither table (m, mu, I, h, ...) don't feed any space bound.
_WIDEN_AT_LOW = ("v_min", "s_min", "sv_min", "lf", "lr")   # shorter wheelbase -> higher yaw-rate cap
_WIDEN_AT_HIGH = ("v_max", "s_max", "sv_max", "a_max")


@dataclass(frozen=True)
class EnvConfig:
    """Main configuration for the F1TENTH environment.

    Attributes:
        seed: Optional seed making a whole run deterministic: the first
            unseeded ``reset()`` behaves as ``reset(seed=seed)``, and every
            later unseeded reset continues that stream. An explicit
            ``reset(seed=...)`` always wins. ``None`` (default) seeds from OS
            entropy. Caution: vectorized sub-envs sharing one config share the
            seed and will produce identical rollouts — seed per sub-env there.
        map_name: Track name, path, or Track instance.
        map_scale: Scale factor for the map.
        params: Vehicle physical parameters.
        num_agents: Number of agents in the environment.
        ego_index: Index of the ego agent (0-indexed).
        control_config: Control input configuration.
        simulation_config: Physics simulation configuration.
        observation_config: Observation space configuration.
        reset_config: Episode reset configuration.
        lidar_config: LiDAR sensor configuration.
        render_config: Rendering pacing / frame-output configuration.
        termination_config: Episode termination / truncation configuration.
        reward_config: Per-step reward configuration.
        domain_randomization_config: Per-episode vehicle-param randomization.
        collision_check: Collision detection mode. Defaults to
            ``SEGMENT_CONTACT``, which detects and resolves contact. ``NONE``
            disables collision detection and response.
        render_enabled: Whether rendering is enabled.
    """

    seed: Optional[int] = None
    map_name: "Track | str" = "Spielberg"
    map_scale: float = 1.0
    params: VehicleParameters = F1TENTH_VEHICLE_PARAMETERS
    num_agents: int = 1
    ego_index: int = 0
    control_config: ControlConfig = field(default_factory=ControlConfig)
    simulation_config: SimulationConfig = field(default_factory=SimulationConfig)
    observation_config: ObservationConfig = field(default_factory=ObservationConfig)
    reset_config: ResetConfig = field(default_factory=ResetConfig)
    lidar_config: LiDARConfig = field(default_factory=LiDARConfig)
    contact_config: ContactConfig = field(default_factory=ContactConfig)
    render_config: RenderConfig = field(default_factory=RenderConfig)
    termination_config: TerminationConfig = field(default_factory=TerminationConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    domain_randomization_config: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )
    collision_check: CollisionCheckMode = CollisionCheckMode.SEGMENT_CONTACT
    render_enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.params, VehicleParameters):
            raise TypeError("params must be a VehicleParameters instance")

        if self.seed is not None:
            object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "map_scale", float(self.map_scale))
        object.__setattr__(self, "num_agents", int(self.num_agents))
        object.__setattr__(self, "ego_index", int(self.ego_index))
        object.__setattr__(self, "render_enabled", bool(self.render_enabled))

        # Validate numeric constraints
        if self.map_scale <= 0:
            raise ValueError(f"map_scale must be > 0, got {self.map_scale}")
        if self.num_agents < 1:
            raise ValueError(f"num_agents must be >= 1, got {self.num_agents}")
        if not (0 <= self.ego_index < self.num_agents):
            raise ValueError(
                f"ego_index must be in range [0, num_agents), got {self.ego_index} with num_agents={self.num_agents}"
            )

        control_cfg = self.control_config
        if not isinstance(control_cfg, ControlConfig):
            raise TypeError("control must be a ControlConfig instance")

        simulation_cfg = self.simulation_config
        if not isinstance(simulation_cfg, SimulationConfig):
            raise TypeError("simulation must be a SimulationConfig instance")

        observation_cfg = self.observation_config
        if not isinstance(observation_cfg, ObservationConfig):
            raise TypeError("observation must be an ObservationConfig instance")

        reset_cfg = self.reset_config
        if not isinstance(reset_cfg, ResetConfig):
            raise TypeError("reset must be a ResetConfig instance")

        lidar_cfg = self.lidar_config
        if not isinstance(lidar_cfg, LiDARConfig):
            raise TypeError("lidar must be a LiDARConfig instance")

        contact_cfg = self.contact_config
        if not isinstance(contact_cfg, ContactConfig):
            raise TypeError("contact must be a ContactConfig instance")

        # Coerce before the identity-based simulator dispatch.
        try:
            collision_check = CollisionCheckMode(self.collision_check)
        except ValueError as exc:
            raise ValueError(f"collision_check must be a CollisionCheckMode: {exc}") from exc
        object.__setattr__(self, "collision_check", collision_check)
        if simulation_cfg.dynamics_model is DynamicModel.MB:
            raise ValueError(
                "DynamicModel.MB is unsupported in this version of the gym. "
                "Use DynamicModel.ST, or DynamicModel.KS for a cheaper approximation."
            )
        if (
            collision_check is CollisionCheckMode.SEGMENT_CONTACT
            and simulation_cfg.dynamics_model is DynamicModel.KS
        ):
            warnings.warn(
                "CollisionCheckMode.SEGMENT_CONTACT is not accurate on diagonal "
                "contact under DynamicModel.KS: KS carries no slip angle or yaw "
                "rate, so the corrected velocity is projected onto its kinematic "
                "course. Use ST when accurate angled contact matters.",
                UserWarning,
                stacklevel=2,
            )

        render_cfg = self.render_config
        if not isinstance(render_cfg, RenderConfig):
            raise TypeError("render must be a RenderConfig instance")

        termination_cfg = self.termination_config
        if not isinstance(termination_cfg, TerminationConfig):
            raise TypeError("termination must be a TerminationConfig instance")

        reward_cfg = self.reward_config
        if not isinstance(reward_cfg, RewardConfig):
            raise TypeError("reward must be a RewardConfig instance")
        if reward_cfg.mode is RewardMode.PROGRESS and not simulation_cfg.compute_frenet_frame:
            raise ValueError(
                "RewardMode.PROGRESS needs the Frenet frame; set "
                "simulation_config.compute_frenet_frame=True"
            )

        dr_cfg = self.domain_randomization_config
        if not isinstance(dr_cfg, DomainRandomizationConfig):
            raise TypeError("domain_randomization must be a DomainRandomizationConfig instance")
        inactive_dr_fields = set(dr_cfg.randomized_fields()).intersection(
            PARAMETER_ORDER[20:]
        )
        if inactive_dr_fields:
            names = ", ".join(sorted(inactive_dr_fields))
            raise ValueError(
                "Domain randomization includes unsupported MB-only fields: " + names
            )

        if (
            simulation_cfg.loop_counter is LoopCounterMode.FRENET_BASED
            and not simulation_cfg.compute_frenet_frame
        ):
            simulation_cfg = simulation_cfg.with_updates(compute_frenet_frame=True)

        object.__setattr__(self, "control_config", control_cfg)
        object.__setattr__(self, "simulation_config", simulation_cfg)
        object.__setattr__(self, "observation_config", observation_cfg)
        object.__setattr__(self, "reset_config", reset_cfg)
        object.__setattr__(self, "lidar_config", lidar_cfg)
        object.__setattr__(self, "contact_config", contact_cfg)
        object.__setattr__(self, "render_config", render_cfg)
        object.__setattr__(self, "termination_config", termination_cfg)
        object.__setattr__(self, "reward_config", reward_cfg)
        object.__setattr__(self, "domain_randomization_config", dr_cfg)

    def with_updates(self, **changes: Any) -> "EnvConfig":
        return replace(self, **changes)
