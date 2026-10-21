"""Residual-task contracts, with actual physics and no downloaded maps."""

import csv

import gymnasium as gym
import numpy as np
import pytest

from f1tenth_gym.envs.dynamic_models import F1TENTH_VEHICLE_PARAMETERS as VEHICLE
from f1tenth_gym.envs.env_config import ControlConfig, DomainRandomizationConfig
from f1tenth_gym.envs.track import Track
from SAC_project.baseline import PurePursuitController, build_config
from SAC_project.features import FeatureExtractor, sample_reference
from SAC_project.residual_env import make_residual_env
from SAC_project.rewards import tracking_reward


def circle_track():
    angle = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    return Track.from_refline(
        6 * np.cos(angle), 6 * np.sin(angle), 2.0 + 0.2 * np.sin(angle)
    )


@pytest.fixture(scope="module")
def track():
    return circle_track()


@pytest.fixture
def env(track):
    instance = make_residual_env(map_name=track, max_steps=150, laps=1)
    yield instance
    instance.close()


@pytest.mark.parametrize("speed_mode", ["fixed", "profile"])
@pytest.mark.parametrize("delay", [0, 2])
def test_zero_residual_matches_actions_full_states_and_timeout(
    track, speed_mode, delay
):
    cfg = build_config(map_name=track, laps=1, max_steps=150).with_updates(
        control_config=ControlConfig(
            steer_delay_steps=delay, throttle_delay_steps=delay
        )
    )
    base = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
    residual = make_residual_env(env_config=cfg, speed_mode=speed_mode)
    controller = PurePursuitController.from_track(
        track, vehicle=cfg.params, speed_mode=speed_mode
    )
    options = {
        "states": np.array(
            [[6, VEHICLE.lr, 0.02, 2, np.pi / 2, 0.1, 0.01]], dtype=np.float32
        )
    }
    try:
        obs, _ = base.reset(seed=123, options=options)
        vector, info = residual.reset(seed=123, options=options)
        np.testing.assert_array_equal(
            obs["agent_0"]["std_state"], info["residual"]["std_state"]
        )
        for step in range(150):
            command = controller.plan(obs["agent_0"]["std_state"][[0, 1, 4]])
            obs, _, terminated, truncated, base_info = base.step(
                command.action[None, :]
            )
            vector, reward, r_terminated, r_truncated, info = residual.step(
                np.zeros(1, dtype=np.float32)
            )
            diagnostic = info["residual"]
            np.testing.assert_array_equal(command.action, diagnostic["physical_action"])
            np.testing.assert_array_equal(
                obs["agent_0"]["std_state"], diagnostic["std_state"]
            )
            np.testing.assert_array_equal(base_info["collisions"], info["collisions"])
            np.testing.assert_array_equal(base_info["lap_counts"], info["lap_counts"])
            assert (terminated, truncated) == (r_terminated, r_truncated)
            assert residual.observation_space.contains(vector)
            assert reward == pytest.approx(sum(diagnostic["reward_terms"].values()))
        assert not terminated and truncated
        with pytest.raises(gym.error.ResetNeeded):
            residual.step(np.zeros(1, dtype=np.float32))
    finally:
        base.close()
        residual.close()


@pytest.mark.parametrize("z", [-1.0, 1.0])
def test_residual_direction_speed_preservation_and_history(env, z):
    initial, info = env.reset(seed=123)
    base_action = info["residual"]["base_action"]
    vector, _, _, _, info = env.step(np.array([z], dtype=np.float32))
    physical = info["residual"]["physical_action"]
    assert physical[0] - base_action[0] == pytest.approx(z * 0.05, abs=1e-7)
    assert physical[1] == base_action[1]
    assert vector[10] == z
    assert vector[11] == pytest.approx(physical[0] / env.features.steering_scale)
    reset_vector, reset_info = env.reset(seed=123)
    np.testing.assert_array_equal(initial, reset_vector)
    np.testing.assert_array_equal(reset_vector[10:13], 0)
    assert reset_info["residual"]["progress_raw_m"] == 0


def test_residual_and_physical_action_are_separately_recorded_when_clipped(track):
    cfg = build_config(map_name=track, max_steps=3).with_updates(
        params=VEHICLE.with_updates(s_min=-0.1, s_max=0.06)
    )
    env = make_residual_env(env_config=cfg)
    try:
        env.reset(
            seed=1,
            options={"poses": np.array([[6, VEHICLE.lr, np.pi / 2]], dtype=np.float32)},
        )
        _, _, _, _, info = env.step(np.ones(1, dtype=np.float32))
        diagnostic = info["residual"]
        assert diagnostic["steering_clipped"]
        assert diagnostic["policy_action"][0] == 1.0
        assert diagnostic["requested_steering_residual_rad"] == 0.05
        assert diagnostic["effective_steering_residual_rad"] < 0.05
        assert diagnostic["physical_action"][0] == np.float32(0.06)
    finally:
        env.close()


@pytest.mark.parametrize(
    "action", [0.0, [0, 0], [[0]], [np.nan], [np.inf], [1.01], [-1.01]]
)
def test_invalid_actions_do_not_advance_physics(env, action):
    env.reset(seed=123)
    before = env.unwrapped.sim_time
    with pytest.raises(ValueError):
        env.step(action)
    assert env.unwrapped.sim_time == before
    _, _, _, _, info = env.step(np.zeros(1, dtype=np.float32))
    assert info["sim_time"] == pytest.approx(0.01)


def test_observations_and_info_are_independent_copies(env):
    first, first_info = env.reset(seed=123)
    snapshot = first_info["residual"]["std_state"].copy()
    baseline = env.baseline_action
    baseline[:] = 999
    first[:] = 999
    first_info["residual"]["std_state"][:] = 999
    second, _, _, _, second_info = env.step(np.zeros(1, dtype=np.float32))
    saved_second = second_info["residual"]["std_state"].copy()
    assert env.observation_space.contains(second)
    assert np.max(np.abs(second_info["residual"]["physical_action"])) < 999
    env.step(np.zeros(1, dtype=np.float32))
    np.testing.assert_array_equal(second_info["residual"]["std_state"], saved_second)
    _, reset_info = env.reset(seed=123)
    np.testing.assert_array_equal(reset_info["residual"]["std_state"], snapshot)


def test_invalid_tracker_target_stops_and_ignores_residual(env):
    env.reset(seed=123, options={"poses": np.array([[0, 0, 0]], dtype=np.float32)})
    _, _, terminated, truncated, info = env.step(np.ones(1, dtype=np.float32))
    diagnostic = info["residual"]
    assert terminated and not truncated
    assert diagnostic["residual_ignored"]
    assert diagnostic["tracking_stop"]
    assert diagnostic["end_reason"] == "tracking_stop"
    assert diagnostic["reward_terms"]["failure"] == -20
    np.testing.assert_array_equal(diagnostic["physical_action"], [0, 0])


def test_actual_wall_collision_terminates_and_costs_once():
    track = circle_track()
    row = int((0.75 - track.spec.origin[1]) / track.spec.resolution)
    track.occupancy_map[row:, :] = 0  # New track, before any geometry is cached.
    env = make_residual_env(map_name=track, max_steps=100)
    try:
        env.reset(
            seed=1,
            options={
                "states": np.array(
                    [[6, VEHICLE.lr, 0, 2, np.pi / 2, 0, 0]], dtype=np.float32
                )
            },
        )
        for _ in range(100):
            _, _, terminated, truncated, info = env.step(np.zeros(1, dtype=np.float32))
            if terminated or truncated:
                break
        assert bool(info["collisions"][0])
        assert terminated and not truncated
        assert info["residual"]["end_reason"] == "collision"
        assert info["residual"]["reward_terms"]["failure"] == -20
        with pytest.raises(gym.error.ResetNeeded):
            env.step(np.zeros(1, dtype=np.float32))
    finally:
        env.close()


def test_features_use_rear_projection_and_vehicle_local_preview_coordinates():
    path = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float)
    controller = PurePursuitController(
        path, speed_mode="profile", speed_profile=[2, 4, 6, 3]
    )
    state = np.array([2 + VEHICLE.lr, 0.2, 0.03, 3, 0, 0.1, 0.02])
    command = controller.plan(state[[0, 1, 4]])
    extractor = FeatureExtractor(controller)
    frame = extractor.observe(state, command, 0, np.zeros(2))
    assert len(frame.observation) == 25
    assert frame.path_s_m == pytest.approx(2)
    assert frame.observation[0] == pytest.approx(0.2 / 3)
    previews = frame.observation[13:].reshape(4, 3)
    np.testing.assert_allclose(previews[:, 0], np.array([1, 2, 4, 8]) / 11)
    np.testing.assert_allclose(previews[:, 1], -0.2 / 11)
    np.testing.assert_allclose(previews[:, 2], np.array([2.3, 2.4, 2.6, 3.0]) / 20)
    point, speed = sample_reference(controller, controller.arclength[-1] + 2)
    np.testing.assert_allclose(point, [2, 0])
    assert speed == pytest.approx(2.2)

    yaw = 1.2
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    translated = PurePursuitController(
        path @ rotation.T + [40, -12], speed_mode="profile", speed_profile=[2, 4, 6, 3]
    )
    moved = state.copy()
    moved[:2] = rotation @ state[:2] + [40, -12]
    moved[4] += yaw
    moved_command = translated.plan(moved[[0, 1, 4]])
    moved_frame = FeatureExtractor(translated).observe(
        moved, moved_command, 0, np.zeros(2)
    )
    np.testing.assert_allclose(frame.observation, moved_frame.observation, atol=1e-7)


def test_observation_clipping_is_counted_without_changing_raw_state():
    path = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float)
    controller = PurePursuitController(path)
    state = np.array([VEHICLE.lr, 0, 0, 0, 0, 100, 0], dtype=float)
    frame = FeatureExtractor(controller).observe(
        state, controller.plan(state[[0, 1, 4]]), 0, np.zeros(2)
    )
    assert frame.clipped_features == 1
    assert frame.observation[6] == 1
    assert state[5] == 100


def test_reward_has_hand_calculated_components_and_one_failure_cost():
    kwargs = dict(
        previous_s=0,
        current_s=0.08,
        path_length=100,
        dt=0.01,
        speed_limit=20,
        lateral_error=0.1,
        heading_error=0,
        z=0.4,
        previous_z=0.2,
    )
    result = tracking_reward(**kwargs, failed=False)
    assert result.terms["lateral"] == pytest.approx(-0.0016)
    assert result.terms["residual"] == pytest.approx(-0.00016)
    assert result.terms["smoothness"] == pytest.approx(-0.00002)
    assert result.total == pytest.approx(0.07822)
    failed = tracking_reward(**kwargs, failed=True)
    assert failed.total == pytest.approx(result.total - 20)


@pytest.mark.parametrize(
    "previous,current,expected",
    [(99.97, 0.02, 0.05), (0.02, 99.97, -0.05), (0, 10, 0.4)],
)
def test_progress_wraps_in_both_directions_and_bounds_projection_jumps(
    previous, current, expected
):
    result = tracking_reward(
        previous_s=previous,
        current_s=current,
        path_length=100,
        dt=0.01,
        speed_limit=20,
        lateral_error=0,
        heading_error=0,
        z=0,
        previous_z=0,
        failed=False,
    )
    assert result.used_progress_m == pytest.approx(expected)
    assert result.total == pytest.approx(expected)


def test_different_vehicle_randomization_requires_a_future_adapter(track):
    cfg = build_config(map_name=track).with_updates(
        domain_randomization_config=DomainRandomizationConfig(
            enabled=True, low=VEHICLE, high=VEHICLE.with_updates(lr=0.18)
        )
    )
    with pytest.raises(ValueError, match="fixed vehicle"):
        make_residual_env(env_config=cfg)


def test_sb3_environment_checker(env):
    checker = pytest.importorskip("stable_baselines3.common.env_checker")
    checker.check_env(env, warn=True)


def test_validation_report_preserves_timeout_and_reward_components(track, tmp_path):
    pytest.importorskip("stable_baselines3")
    from SAC_project.validate_residual import REWARD_NAMES, validate

    report = validate(
        output_dir=tmp_path, map_name=track, laps=1, max_steps=4, random_steps=3
    )
    assert report["equivalent"] and report["all_states_bitwise_equal"]
    assert report["truncated"] and not report["terminated"]
    assert not report["baseline_success"]
    assert report["full_lap_times_s"] == []
    for filename, reward_field in [
        ("zero_residual.csv", "residual_reward"),
        ("random_residual.csv", "reward"),
    ]:
        with (tmp_path / filename).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        for row in rows:
            assert None not in row
            assert float(row[reward_field]) == pytest.approx(
                sum(float(row[f"reward_{name}"]) for name in REWARD_NAMES)
            )
    assert (tmp_path / "zero_residual_equivalence.png").is_file()
