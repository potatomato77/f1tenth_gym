"""Perturbation-evaluation helpers."""

import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from SAC_project.evaluate import (
    build_cases,
    parse_float_list,
    perturb_initial_state,
)


def test_parse_float_list():
    assert parse_float_list("-0.3,0,0.1") == [-0.3, 0.0, 0.1]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_float_list("")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_float_list("nan")


def test_build_cases_nominal():
    args = argparse.Namespace(
        lateral_offsets=None, heading_offsets_deg=None, seed=10, episodes=3
    )
    cases = build_cases(args)
    assert [case["seed"] for case in cases] == [10, 11, 12]
    assert all(case["lateral_m"] == 0.0 for case in cases)


def test_build_cases_grid():
    args = argparse.Namespace(
        lateral_offsets=[-0.3, 0.0],
        heading_offsets_deg=[-5.0, 5.0],
        seed=100,
        episodes=99,
    )
    cases = build_cases(args)
    assert [(case["lateral_m"], case["heading_deg"]) for case in cases] == [
        (-0.3, -5.0),
        (-0.3, 5.0),
        (0.0, -5.0),
        (0.0, 5.0),
    ]
    assert [case["seed"] for case in cases] == [100, 101, 102, 103]


def test_perturb_initial_state_uses_left_normal_and_wraps_yaw():
    class FakeController:
        vectors = np.array([[1.0, 0.0]], dtype=np.float64)

        def plan(self, pose):
            return SimpleNamespace(projection=SimpleNamespace(segment=0))

    env = SimpleNamespace(controller=FakeController())
    state = np.zeros(7, dtype=np.float64)

    left = perturb_initial_state(env, state, 0.2, 90.0)
    np.testing.assert_allclose(left[:2], [0.0, 0.2], atol=1e-6)
    np.testing.assert_allclose(left[4], np.pi / 2, atol=1e-6)

    right = perturb_initial_state(env, state, -0.1, -90.0)
    np.testing.assert_allclose(right[:2], [0.0, -0.1], atol=1e-6)
    np.testing.assert_allclose(right[4], -np.pi / 2, atol=1e-6)
