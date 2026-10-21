# 3-seed SAC results

Protocol: deterministic policy, 9 perturbation cases (lateral -0.3/0/+0.1 m x heading -5/0/+5 deg), fixed nominal start state, 3 training seeds, 500k steps each.

## Aggregate over perturbation cases

| Metric | Zero residual | SAC (mean ± std over seeds) |
|---|---:|---:|
| Success rate | 66.7% | 88.9% ± 15.7% |
| Collision rate | 33.3% | 11.1% ± 15.7% |
| Rear RMS [m] | 0.1370 m | 0.0898 ± 0.0157 m |
| Rear P95 [m] | 0.2561 m | 0.1942 ± 0.0134 m |
| Heading RMS [rad] | 0.0846 rad | 0.0743 ± 0.0110 rad |
| Full lap time [s] | 45.5533 s | 45.4333 ± 0.0287 s |
| Episode return | 876.250 | 1175.393 ± 206.926 |
| Mean abs(z) | 0.000 | 0.465 ± 0.046 |
| Steering clip fraction | 0.0% | 0.0% ± 0.0% |

## Per seed

| Metric | seed 12345 | seed 42 | seed 2026 |
|---|---:|---:|---:|
| Success rate | 100.0% | 100.0% | 66.7% |
| Collision rate | 0.0% | 0.0% | 33.3% |
| Rear RMS [m] | 0.0774 | 0.0802 | 0.1119 |
| Rear P95 [m] | 0.1754 | 0.2056 | 0.2016 |
| Heading RMS [rad] | 0.0662 | 0.0670 | 0.0899 |
| Full lap time [s] | 45.437 | 45.467 | 45.397 |
| Episode return | 1323.277 | 1320.140 | 882.761 |
| Mean abs(z) | 0.499 | 0.400 | 0.497 |
| Steering clip fraction | 0.0% | 0.0% | 0.0% |

## Nominal case (lateral 0, heading 0)

| Seed | Zero success | SAC success | Zero RMS [m] | SAC RMS [m] | Zero lap [s] | SAC lap [s] |
|---|---:|---:|---:|---:|---:|---:|
| 12345 | yes | yes | 0.0929 | 0.0768 | 45.550 | 45.440 |
| 42 | yes | yes | 0.0929 | 0.0797 | 45.550 | 45.460 |
| 2026 | yes | yes | 0.0929 | 0.0590 | 45.550 | 45.400 |

## Paired changes (policy - baseline)

- Rear RMS: -0.0472 ± 0.0532 m over 27 cases
- Rear P95: -0.0619 ± 0.0498 m over 27 cases
- Lap time: -0.1172 ± 0.0258 s over 18 cases where both completed
- Success: 6 rescues, 21 ties, 0 losses

## Key findings

- Aggregate success: 66.7% -> 88.9% ± 15.7% (mean ± std over seeds).
- Paired outcomes: 6 rescues, 21 ties, 0 losses.
- Seeds rescuing every lateral -0.3 m case: [12345, 42].
- Seeds with no rescue but no regression: [2026].
- Rear RMS change: -0.0472 ± 0.0532 m; P95 change: -0.0619 ± 0.0498 m.
- Protocol limits: one map, 500k steps, initial lateral/heading offsets only; friction, actuator delay and domain randomization are untested.
- Lap times use successful episodes only; failed episodes remain in success and tracking-error statistics.

## Figures

- `fig_learning_curves.png`: stochastic training curves per seed.
- `fig_metrics_comparison.png`: deterministic aggregate metrics with seed error bars.
- `fig_robustness_lateral.png`: success and RMS versus initial lateral offset.
- `fig_success_heatmap.png`: success counts over the lateral x heading grid.
- `fig_trajectory_compare.png`: representative baseline crash versus SAC recovery (case 1, seed 12346).
- `fig_failure_breakdown.png`: termination reasons.
