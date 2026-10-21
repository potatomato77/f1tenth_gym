![Python 3.12+](https://github.com/f1tenth/f1tenth_gym/actions/workflows/ci.yml/badge.svg)
![Docker](https://github.com/f1tenth/f1tenth_gym/actions/workflows/docker.yml/badge.svg)
![Code Style](https://github.com/f1tenth/f1tenth_gym/actions/workflows/lint.yml/badge.svg)

# The F1TENTH Gym environment

> **Fork note:** This repository is a fork of
> [f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) (MIT License).
> The upstream project provides the simulator, vehicle dynamics and base
> Gymnasium environment. This fork adds a residual SAC steering study under
> [SAC_project/](SAC_project/README.md): a pure-pursuit baseline, residual
> environment and reward, SAC training/evaluation pipeline, and native Windows
> rendering adaptations.

## What this fork adds

The upstream project provides the simulator, vehicle dynamics and base
Gymnasium environment. This fork adds a residual SAC steering study in
[SAC_project/](SAC_project/README.md):

- a rear-axle pure pursuit baseline with reproducible lap and tracking metrics;
- a residual steering wrapper with fixed 25-D features, bounded action contract
  and decomposed reward;
- zero-residual equivalence and random-action interface validation;
- SB3 SAC training, checkpoint/resume, and baseline-versus-policy evaluation;
- initial lateral/heading perturbation grids with invalid-start detection;
- a 3-seed, 500k-step perturbation study with figures in
  [SAC_project/results/3seed_500k/](SAC_project/results/3seed_500k/);
- native Windows rendering adaptations and additional tests.

The simulator, examples, documentation and upstream tests remain from the
f1tenth_gym authors; see [LICENSE](LICENSE).

This is the repository of the F1TENTH Gym environment.

This project is still under heavy developement.

You can find the [documentation](https://f1tenth-gym.readthedocs.io/en/latest/) of the environment here.

## Quickstart

### Using uv (recommended)

We recommend using [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management:

```bash
git clone https://github.com/f1tenth/f1tenth_gym.git
cd f1tenth_gym
uv sync --frozen
```

Then you can run a quick waypoint follow example by:
```bash
uv run python examples/waypoint_follow.py
```

For a complete device-native PPO update with no map download, run:

```bash
uv run --extra train python examples/jax_ppo_training.py \
  --smoke-test --device cpu --num-envs 4 --rollout-steps 2 \
  --total-timesteps 8 --update-epochs 1 --minibatches 2 --no-save
```

The regular training job defaults to the GPU, the Spielberg track, wall
contact, and per-environment domain randomization. See the
[RL guide](https://f1tenth-gym.readthedocs.io/en/latest/rl.html) for the native
batch API and optional SBX interoperability.

### CPU-only machines

The default sync installs GPU JAX. Without CUDA, skip the ~3 GB of GPU wheels:

```bash
uv sync --frozen --no-group gpu
```

Then run examples with:
```bash
uv run python examples/waypoint_follow.py
```

### Using Docker

A Dockerfile is also provided with support for the GUI with nvidia-docker (nvidia GPU required):
```bash
docker build -t f1tenth_gym_container -f Dockerfile .
docker run --gpus all -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix f1tenth_gym_container
```

## Citing
If you find this Gym environment useful, please consider citing:

```
@inproceedings{okelly2020f1tenth,
  title={F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
  author={O’Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
  booktitle={NeurIPS 2019 Competition and Demonstration Track},
  pages={77--89},
  year={2020},
  organization={PMLR}
}
```
