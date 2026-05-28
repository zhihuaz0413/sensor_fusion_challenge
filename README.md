# Sensor Fusion Challenge

This is a small Python project for exploring how a wheeled robot can combine
IMU and wheel-odometry measurements to estimate its 2D position and velocity.
It includes tunable sensor noise and drift, several fusion strategies, and
scripts for comparing accuracy and update speed.

The robot follows analytical trajectories, the virtual sensors add configurable
noise and bias, and each estimator plugs into the same simple `SensorFusion`
interface.

## What Is Included

| Area | Details |
|---|---|
| Robot motion | Six analytical trajectories: `straight`, `circle`, `sine`, `zigzag`, `figure8`, `stopgo` |
| Sensors | Virtual 2D accelerometer and wheel-velocity odometry |
| Noise presets | `ideal`, `low`, `medium`, `high`, including a cheap drifting IMU scenario |
| Fusion modes | Kalman filter, augmented Kalman filter, IMU-only, odometry-only, complementary filter |
| Reports | CSV, Markdown/PDF report, heatmaps, static plots, optional animations |
| Tests | Pytest unit tests plus a broader benchmark/invariant suite |

Fusion modes:

| Mode | Class | Purpose |
|---|---|---|
| `kf` | `KalmanFusion2D` | 6-state Kalman filter: `[px, py, vx, vy, bax, bay]` |
| `kf_augmented` | `AugmentedKalmanFusion2D` | EKF variant that also estimates wheel scale |
| `imu_only` | `ImuDeadReckoning` | Shows accelerometer drift from double integration |
| `odometry_only` | `WheelDeadReckoning` | Wheel-velocity baseline |
| `complementary` | `ComplementaryFusion` | Lightweight fixed-gain IMU/wheel blend |

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Unit tests
pytest -q

# Fast benchmark smoke test
python test_sensor_fusion.py --quick --no-pdf

# Full benchmark and report
python test_sensor_fusion.py

# Visual comparison for one scenario
python run_simulation.py --trajectory zigzag --noise high \
    --save outputs/compare_zigzag_high.png
```

Generated files are written to `outputs/`.

## Common Commands

Run the full benchmark suite:

```bash
python test_sensor_fusion.py
```

Useful benchmark options:

```bash
python test_sensor_fusion.py --quick
python test_sensor_fusion.py --no-pdf
python test_sensor_fusion.py --imu-rates 500 1000 2000 5000
python test_sensor_fusion.py --help
```

Visualize selected modes:

```bash
python run_simulation.py --trajectory sine --noise medium --animate
python run_simulation.py --modes kf complementary --trajectory circle --noise high
python run_simulation.py --modes kf kf_augmented --trajectory sine --noise high
python run_simulation.py --trajectory figure8 --animate --save outputs/figure8.gif
```

By default, `run_simulation.py` compares `kf`, `imu_only`, `odometry_only`,
and `complementary`. The augmented filter remains available with
`--modes kf_augmented` when you want to inspect the wheel-scale variant.

Run tests:

```bash
pytest -q
```

## Project Layout

```text
sensor_fusion_challenge/
├── config.py                  # Shared dataclasses and noise presets
├── simulation.py              # Core run_simulation(...) harness
├── run_simulation.py          # CLI visualization tool
├── test_sensor_fusion.py      # Benchmark/report generator
├── requirements.txt
├── robot/
│   ├── robot.py               # Robot geometry and drawing helpers
│   └── trajectory.py          # Analytical trajectory catalogue
├── sensors/
│   └── sensors.py             # IMU and wheel-odometry sensor models
├── sensor_fusion/
│   ├── __init__.py            # FusionMode and make_fusion(...)
│   ├── kf_fusion.py           # KF and augmented KF implementations
│   └── dummpy_fusion.py       # Baseline and complementary estimators
└── tests/
    └── test_fusion.py
```


## Sensor Noise

Noise is configured in `config.py` with the `NoiseLevel` enum and helper
functions:

```python
from config import NoiseLevel, imu_config_for, wheel_config_for

imu_cfg = imu_config_for(NoiseLevel.HIGH, rate_hz=1000.0)
wheel_cfg = wheel_config_for("medium", rate_hz=50.0)
```

Preset summary:

| Level | IMU accel std | IMU initial bias std | IMU bias random walk | Wheel velocity std | Wheel scale error |
|---|---:|---:|---:|---:|---:|
| `ideal` | 0.00 | 0.00 | 0.000 | 0.00 | 0.000 |
| `low` | 0.02 | 0.01 | 0.001 | 0.03 | 0.005 |
| `medium` | 0.10 | 0.05 | 0.010 | 0.08 | 0.010 |
| `high` | 1.00 | 0.80 | 0.200 | 0.20 | 0.050 |

All units are SI: acceleration in `m/s^2`, velocity in `m/s`, and scale error
as a unitless multiplier.

## Programmatic Usage

```python
from config import NoiseLevel, SimulationConfig, imu_config_for, wheel_config_for
from robot import TRAJECTORY_CATALOG
from sensor_fusion import FusionMode
from simulation import run_simulation

result = run_simulation(
    "demo",
    sim_config=SimulationConfig(duration_s=8.0, seed=7),
    imu_config=imu_config_for(NoiseLevel.HIGH, rate_hz=1000.0),
    wheel_config=wheel_config_for(NoiseLevel.HIGH, rate_hz=50.0),
    trajectory=TRAJECTORY_CATALOG["sine"],
    fusion_mode=FusionMode.KF,
)

print(result.rms_position_error_m)
print(result.mean_update_time_us)
```

To construct a filter directly:

```python
from sensor_fusion import ComplementaryNoise, FusionMode, make_fusion

fusion = make_fusion(
    FusionMode.COMPLEMENTARY,
    complementary_noise=ComplementaryNoise(wheel_weight=0.2),
)
```

## Benchmark Expectations

The benchmark script checks that:

- ideal-noise scenarios stay near ground truth;
- the Kalman filter beats IMU-only dead reckoning under high IMU drift;
- position error degrades monotonically as sensor noise increases;
- p99 update latency stays below the 1 ms budget for a 1 kHz IMU;
- a paced wall-clock loop misses fewer than 1% of deadlines.

On the current development machine, the quick benchmark reports p99 latency on
the order of tens of microseconds for the Kalman filters and only a few
microseconds for the baseline filters, leaving substantial headroom under the
1 ms IMU deadline.

## Design Notes

- The primary estimator is a 6-state Kalman filter over 2D position,
  velocity, and IMU acceleration bias.
- IMU measurements drive the high-rate prediction step, which keeps the
  estimator aligned with the 1 kHz sensor stream.
- Wheel odometry provides a lower-rate velocity correction that bounds the
  drift from IMU-only double integration.
- The `high` noise preset intentionally models a cheap IMU with large initial
  bias and bias random walk.
- The augmented EKF adds wheel-scale estimation, but those extra states are
  not always observable enough to outperform the simpler 6-state filter on
  every trajectory.
- The harness is estimator-agnostic: new fusion methods can implement the
  same `SensorFusion` interface and be evaluated with the existing benchmark,
  plots, and reports.

## Limitations

This is a simplified harness rather than a full robot state estimator.
The simulation uses world-frame 2D acceleration and velocity; it does not model
yaw, body-frame IMU readings, angular velocity, wheel slip, contacts, or
collisions.

## Dependencies

- NumPy - array math for the filter and simulation
- Matplotlib - plots, animations, and reports
- Pytest - unit tests
