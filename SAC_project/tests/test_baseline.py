"""Geometry and lifecycle gates for the independent SAC-project baseline."""

import csv
import json

import numpy as np
import pytest

from f1tenth_gym.envs.dynamic_models import F1TENTH_VEHICLE_PARAMETERS as VEHICLE
from f1tenth_gym.envs.track import Track
from SAC_project.baseline import PurePursuitController, run_baseline
from SAC_project.reporting import error_summary


SPARSE_PATH = np.array([[0, 0], [2, 1], [4, 0], [4, -2], [0, -2]], dtype=float)


def test_exact_intersection_is_used_instead_of_segment_start():
    controller = PurePursuitController(SPARSE_PATH, lookahead=1.0)
    command = controller.plan([VEHICLE.lr, 0.0, 0.0])
    expected = np.array([2.0, 1.0]) / np.sqrt(5.0)
    np.testing.assert_allclose(command.target, expected, atol=1e-12)
    assert command.target_distance == pytest.approx(1.0)
    assert command.raw_steering == pytest.approx(
        np.arctan(2 * controller.wheelbase * expected[1])
    )
    assert command.mode == "tracking"


@pytest.mark.parametrize("yaw", [0.0, 0.7, np.pi / 2, -2.3])
def test_world_rotation_and_translation_preserve_control(yaw):
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    translation = np.array([10.0, -3.0])
    controller = PurePursuitController(
        SPARSE_PATH @ rotation.T + translation, lookahead=1.0
    )
    cog = translation + VEHICLE.lr * rotation[:, 0]
    command = controller.plan([*cog, yaw])
    expected_target = rotation @ (np.array([2.0, 1.0]) / np.sqrt(5.0)) + translation
    np.testing.assert_allclose(command.rear_position, translation, atol=1e-12)
    np.testing.assert_allclose(command.target, expected_target, atol=1e-12)
    assert command.raw_steering == pytest.approx(
        np.arctan(2 * controller.wheelbase / np.sqrt(5.0))
    )


def test_closing_edge_is_projected_and_followed_without_duplicate_endpoint():
    path = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    controller = PurePursuitController(path, lookahead=0.5)
    rear = np.array([-0.1, 1.0])
    yaw = -np.pi / 2
    cog = rear + VEHICLE.lr * np.array([np.cos(yaw), np.sin(yaw)])
    command = controller.plan([*cog, yaw])
    np.testing.assert_allclose(controller.project(rear).point, [0.0, 1.0])
    assert controller.project(rear).distance == pytest.approx(0.1)
    assert command.rear_error == pytest.approx(-0.1)
    np.testing.assert_allclose(command.target, [0, 1.0 - np.sqrt(0.25 - 0.01)])
    assert command.target_distance == pytest.approx(0.5)


def test_duplicate_points_do_not_create_zero_length_geometry():
    path = np.vstack([SPARSE_PATH[0], SPARSE_PATH, SPARSE_PATH[0], SPARSE_PATH[0]])
    controller = PurePursuitController(path, lookahead=1.0)
    command = controller.plan([VEHICLE.lr, 0.0, 0.0])
    np.testing.assert_allclose(command.target, np.array([2, 1]) / np.sqrt(5.0))
    assert np.all(np.isfinite(command.action))


def test_recovery_uses_actual_distance_and_a_point_ahead_of_projection():
    controller = PurePursuitController(
        np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        lookahead=1.0,
    )
    command = controller.plan([0.5 + VEHICLE.lr, -2.0, 0.0])
    np.testing.assert_allclose(command.target, [1.5, 0.0])
    assert command.mode == "recovery"
    assert command.target_distance == pytest.approx(np.sqrt(5.0))
    expected_steering = np.arctan(2 * controller.wheelbase * 2.0 / 5.0)
    assert command.raw_steering == pytest.approx(expected_steering)
    assert command.action[0] == pytest.approx(expected_steering)


@pytest.mark.parametrize("side, bound", [(1, 0.2), (-1, -0.1)])
def test_asymmetric_steering_and_speed_limits(side, bound):
    path = SPARSE_PATH * [1, side]
    vehicle = VEHICLE.with_updates(s_min=-0.1, s_max=0.2, v_max=1.5)
    controller = PurePursuitController(path, vehicle, lookahead=1.0, speed=2.0)
    command = controller.plan([vehicle.lr, 0.0, 0.0])
    assert command.action.dtype == np.float32
    assert command.action.shape == (2,)
    assert command.action[0] == pytest.approx(bound)
    assert command.action[1] == pytest.approx(1.5)
    assert command.steering_clipped
    assert command.speed_clipped


def test_limits_can_be_reused_after_residual_addition():
    controller = PurePursuitController(SPARSE_PATH)
    np.testing.assert_allclose(controller.limit_action([10, -1]), [VEHICLE.s_max, 0])
    np.testing.assert_allclose(
        controller.limit_action([-10, 100]), [VEHICLE.s_min, VEHICLE.v_max]
    )


def test_profile_speed_is_interpolated_at_rear_projection_not_lookahead():
    path = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    controller = PurePursuitController(
        path,
        lookahead=1.0,
        speed_mode="profile",
        speed_profile=[2, 6, 8, 4],
        vgain=0.5,
    )
    command = controller.plan([2.5 + VEHICLE.lr, 0, 0])
    np.testing.assert_allclose(command.target, [3.5, 0])
    assert command.reference_speed == pytest.approx(1.5)
    assert command.action[1] == pytest.approx(1.5)


def test_profile_wraps_and_stays_aligned_after_duplicate_removal():
    path = np.array([[0, 0], [0, 0], [10, 0], [10, 10], [0, 10], [0, 0]])
    controller = PurePursuitController(
        path,
        speed_mode="profile",
        speed_profile=[2, 99, 6, 8, 4, 2],
    )
    yaw = -np.pi / 2
    rear = np.array([0, 5])
    cog = rear + VEHICLE.lr * np.array([np.cos(yaw), np.sin(yaw)])
    command = controller.plan([*cog, yaw])
    assert command.reference_speed == pytest.approx(3.0)
    np.testing.assert_array_equal(controller.speed_profile, [2, 6, 8, 4])


def test_profile_reference_is_preserved_when_speed_is_clipped():
    vehicle = VEHICLE.with_updates(v_max=3.0)
    controller = PurePursuitController(
        SPARSE_PATH,
        vehicle,
        speed_mode="profile",
        speed_profile=np.full(len(SPARSE_PATH), 4.0),
        vgain=2.0,
    )
    command = controller.plan([vehicle.lr, 0, 0])
    assert command.reference_speed == pytest.approx(8.0)
    assert command.action[1] == pytest.approx(3.0)
    assert command.speed_clipped


@pytest.mark.parametrize(
    "kwargs",
    [
        {"speed_mode": "unknown"},
        {"speed_mode": "profile"},
        {"speed_mode": "profile", "speed_profile": [1, 2]},
        {"speed_mode": "profile", "speed_profile": [1, 2, 3, 4, np.nan]},
        {"speed_mode": "profile", "speed_profile": [1, 2, 3, 4, -1]},
        {"vgain": 0},
        {"vgain": np.inf},
    ],
)
def test_invalid_speed_profiles_are_rejected(kwargs):
    with pytest.raises(ValueError):
        PurePursuitController(SPARSE_PATH, **kwargs)


def test_metric_aggregation_uses_all_samples_with_signed_bias():
    stats = error_summary(np.array([-3.0, 0.0, 4.0]), "mps")
    assert stats["rms_mps"] == pytest.approx(np.sqrt(25 / 3))
    assert stats["mae_mps"] == pytest.approx(7 / 3)
    assert stats["mean_mps"] == pytest.approx(1 / 3)
    assert stats["max_abs_mps"] == 4.0


def test_far_from_path_stops_instead_of_commanding_forward_motion():
    command = PurePursuitController(SPARSE_PATH).plan([50, 50, 0])
    assert command.mode == "stop"
    assert command.reason == "path_too_far"
    np.testing.assert_array_equal(command.action, [0, 0])


def test_goal_behind_vehicle_stops():
    controller = PurePursuitController(
        np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
        lookahead=1.0,
    )
    command = controller.plan([-VEHICLE.lr, 0.0, np.pi])
    assert command.mode == "stop"
    assert command.reason == "target_not_ahead"
    np.testing.assert_array_equal(command.action, [0, 0])


def test_path_entirely_inside_circle_has_no_spurious_intersection():
    controller = PurePursuitController(
        np.array([[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]]),
        lookahead=1.0,
    )
    command = controller.plan([VEHICLE.lr, 0, 0])
    assert command.reason == "no_circle_intersection"
    np.testing.assert_array_equal(command.action, [0, 0])


@pytest.mark.parametrize(
    "points",
    [np.zeros((4, 2)), np.zeros((2, 2)), np.array([[0, 0], [1, 1], [2, np.nan]])],
)
def test_invalid_paths_are_rejected(points):
    with pytest.raises(ValueError):
        PurePursuitController(points)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lookahead": 0},
        {"speed": np.nan},
        {"speed": -1},
        {"max_reacquire_distance": 1.0},
    ],
)
def test_invalid_controller_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        PurePursuitController(SPARSE_PATH, **kwargs)


def test_invalid_pose_or_action_is_rejected_without_mutating_controller():
    controller = PurePursuitController(SPARSE_PATH)
    with pytest.raises(ValueError):
        controller.plan([0, 0, np.nan])
    with pytest.raises(ValueError):
        controller.limit_action([0, np.inf])
    command = controller.plan([VEHICLE.lr, 0, 0])
    command.action[:] = 123
    assert np.max(controller.plan([VEHICLE.lr, 0, 0]).action) < 123


def test_timeout_report_and_csv_are_written_without_map_download(tmp_path):
    angle = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    track = Track.from_refline(
        6 * np.cos(angle), 6 * np.sin(angle), np.full_like(angle, 2.0)
    )
    report = run_baseline(map_name=track, laps=1, max_steps=2, output_dir=tmp_path)
    assert not report["success"]
    assert report["truncated"]
    assert not report["terminated"]
    assert report["completed_laps"] == 0
    assert report["full_lap_times_s"] == []
    assert report["sim_time_including_out_lap_s"] == pytest.approx(0.02)
    assert json.loads((tmp_path / "summary.json").read_text())["steps"] == 2
    with (tmp_path / "trajectory.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert None not in rows[0]
    assert rows[-1]["sim_time"] == "0.02"
    assert float(rows[-1]["command_speed"]) == 2.0
    for row in rows:
        position_error = np.hypot(
            float(row["rear_x"]) - float(row["reference_x"]),
            float(row["rear_y"]) - float(row["reference_y"]),
        )
        assert float(row["position_error_norm"]) == pytest.approx(position_error)
        assert float(row["speed_error"]) == pytest.approx(
            float(row["actual_speed"]) - float(row["command_speed"])
        )
        assert -np.pi <= float(row["heading_error"]) < np.pi
    for filename in report["plots"]:
        assert (tmp_path / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert len(report["plots"]) == 5
    assert (tmp_path / "reference_path.csv").is_file()
    repeated = run_baseline(
        map_name=track, laps=1, max_steps=2, output_dir=tmp_path / "repeat"
    )
    assert repeated == report
    assert (tmp_path / "trajectory.csv").read_bytes() == (
        tmp_path / "repeat" / "trajectory.csv"
    ).read_bytes()


def test_logged_profile_target_belongs_to_applied_step_and_fixed_mode_is_constant(
    tmp_path,
):
    angle = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    track = Track.from_refline(
        6 * np.cos(angle), 6 * np.sin(angle), 3.0 + np.sin(angle)
    )
    report = run_baseline(
        map_name=track, laps=1, max_steps=8, make_plots=False, output_dir=tmp_path
    )
    with (tmp_path / "trajectory.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    controller = PurePursuitController.from_track(track, speed_mode="profile")
    speeds = [float(row["command_speed"]) for row in rows]
    assert np.ptp(speeds) > 1e-5
    for previous, current in zip(rows[:-1], rows[1:]):
        command = controller.plan(
            [float(previous[key]) for key in ("cog_x", "cog_y", "yaw")]
        )
        assert float(current["command_speed"]) == float(command.action[1])
        assert float(current["reference_speed"]) == command.reference_speed
    errors = np.array([float(row["speed_error"]) for row in rows])
    assert report["speed_error"]["rms_mps"] == pytest.approx(
        np.sqrt(np.mean(errors**2))
    )
    assert report["plots"] == []

    fixed_dir = tmp_path / "fixed"
    fixed = run_baseline(
        map_name=track,
        speed_mode="fixed",
        speed=2.0,
        laps=1,
        max_steps=2,
        make_plots=False,
        output_dir=fixed_dir,
    )
    assert fixed["speed_mps"] == 2.0
    assert fixed["command_speed_range_mps"] == [2.0, 2.0]


def test_clipped_and_unclipped_speed_errors_are_distinct_in_csv(tmp_path):
    angle = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    track = Track.from_refline(
        6 * np.cos(angle), 6 * np.sin(angle), np.full_like(angle, 2.0)
    )
    report = run_baseline(
        map_name=track,
        speed_mode="fixed",
        speed=30.0,
        laps=1,
        max_steps=2,
        make_plots=False,
        output_dir=tmp_path,
    )
    with (tmp_path / "trajectory.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            assert float(row["reference_speed"]) == 30.0
            assert float(row["command_speed"]) == VEHICLE.v_max
            assert float(row["reference_speed_error"]) == pytest.approx(
                float(row["speed_error"]) - (30.0 - VEHICLE.v_max)
            )
    assert report["speed_clip_fraction"] == 1.0
