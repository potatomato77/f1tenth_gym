"""Configuration contracts shared by train.py and evaluate.py."""

import argparse
from dataclasses import replace

from SAC_project.rl_common import (
    EnvSettings,
    add_env_arguments,
    settings_from_args,
)


def test_env_settings_round_trip():
    settings = replace(
        EnvSettings(),
        map_name="oval",
        speed_mode="fixed",
        speed=3.5,
        max_steering_residual_rad=0.04,
        preview_distances_m=(1.0, 2.0),
        reward=replace(EnvSettings().reward, lateral_weight=2.5),
    )
    restored = EnvSettings.from_dict(settings.to_dict())
    assert restored == settings


def test_cli_overrides_only_supplied_values():
    parser = argparse.ArgumentParser()
    add_env_arguments(parser)
    args = parser.parse_args(
        ["--map", "oval", "--lateral-weight", "2.5", "--preview-distances", "1,3"]
    )
    settings = settings_from_args(args)
    assert settings.map_name == "oval"
    assert settings.reward.lateral_weight == 2.5
    assert settings.reward.heading_weight == EnvSettings().reward.heading_weight
    assert settings.preview_distances_m == (1.0, 3.0)


def test_cli_arguments_layer_over_a_base_config():
    base = replace(EnvSettings(), map_name="base", speed=4.0)
    parser = argparse.ArgumentParser()
    add_env_arguments(parser)
    args = parser.parse_args(["--map", "override"])
    settings = settings_from_args(args, base)
    assert settings.map_name == "override"
    assert settings.speed == 4.0
