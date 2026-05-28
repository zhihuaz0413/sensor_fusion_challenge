"""Simulation harness wiring the robot, virtual sensors, and a fusion estimator.

The harness is intentionally agnostic about *which* fusion algorithm
it's running.  Any object implementing :class:`SensorFusion`
(``KalmanFusion2D``, ``ImuDeadReckoning``, ``WheelDeadReckoning``,
``ComplementaryFusion``, …) can be plugged in and benchmarked against
any :class:`Trajectory` shape under any :class:`ImuConfig` /
:class:`WheelOdomConfig` noise profile.

The result is a :class:`SimulationResult` that carries everything the
plotting / scoring code needs (truth, estimate, bias, per-step timing)
in pre-allocated NumPy arrays.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from config import ImuConfig, SimulationConfig, WheelOdomConfig
from robot import SineWaveTrajectory, Trajectory
from sensor_fusion import (
    AugmentedFusionNoise,
    ComplementaryNoise,
    FusionMode,
    FusionNoise,
    KalmanFusion2D,
    SensorFusion,
    make_fusion,
)
from sensors import ImuSensor, WheelOdometrySensor

__all__ = ["SimulationResult", "run_simulation"]


@dataclass(frozen=True)
class SimulationResult:
    """Per-step truth, estimate, and timing arrays for one fusion run."""

    name: str
    time_s: np.ndarray              # (N,)
    truth_position: np.ndarray      # (N, 2)
    truth_velocity: np.ndarray      # (N, 2)
    estimate_position: np.ndarray   # (N, 2)
    estimate_velocity: np.ndarray   # (N, 2)
    estimate_bias: np.ndarray       # (N, 2) - zeros for estimators that don't model bias
    rms_position_error_m: float
    final_position_error_m: float
    mean_update_time_us: float


def _resolve_fusion(
    fusion: SensorFusion | None,
    fusion_factory: Callable[[], SensorFusion] | None,
    fusion_mode: FusionMode | str | None,
    kf_noise: FusionNoise | None,
    augmented_noise: AugmentedFusionNoise | None,
    complementary_noise: ComplementaryNoise | None,
) -> SensorFusion:
    if fusion is not None:
        return fusion
    if fusion_factory is not None:
        return fusion_factory()
    if fusion_mode is not None:
        return make_fusion(
            fusion_mode,
            kf_noise=kf_noise,
            augmented_noise=augmented_noise,
            complementary_noise=complementary_noise,
        )
    # No explicit selection: keep the historical default.
    return KalmanFusion2D(kf_noise)


def run_simulation(
    name: str,
    *,
    sim_config: SimulationConfig | None = None,
    imu_config: ImuConfig | None = None,
    wheel_config: WheelOdomConfig | None = None,
    trajectory: Trajectory | None = None,
    fusion: SensorFusion | None = None,
    fusion_factory: Callable[[], SensorFusion] | None = None,
    fusion_mode: FusionMode | str | None = None,
    kf_noise: FusionNoise | None = None,
    augmented_noise: AugmentedFusionNoise | None = None,
    complementary_noise: ComplementaryNoise | None = None,
) -> SimulationResult:
    """Run one fusion scenario and return its result.

    All keyword arguments are optional and fall back to sensible defaults:

    - ``sim_config``       → :class:`SimulationConfig()`
    - ``imu_config``       → :class:`ImuConfig()`
    - ``wheel_config``     → :class:`WheelOdomConfig()`
    - ``trajectory``       → :class:`SineWaveTrajectory()`
    - ``fusion`` / ``fusion_factory`` / ``fusion_mode`` →
      :class:`KalmanFusion2D()`

    Three ways to pick the estimator, in priority order:

    1. ``fusion`` - an already-built :class:`SensorFusion` instance.
       The harness calls :meth:`SensorFusion.initialise` on it.
    2. ``fusion_factory`` - a zero-arg callable returning a fresh
       instance.  Best for repeated runs so each starts clean.
    3. ``fusion_mode`` - a :class:`FusionMode` (or its string value
       ``"kf"`` / ``"imu_only"`` / ``"odometry_only"`` /
       ``"complementary"``).  ``kf_noise`` / ``complementary_noise``
       feed the relevant tuning to :func:`make_fusion`.
    """
    sim_cfg = sim_config or SimulationConfig()
    imu_cfg = imu_config or ImuConfig()
    wheel_cfg = wheel_config or WheelOdomConfig()
    traj = trajectory or SineWaveTrajectory()
    fusion_obj = _resolve_fusion(
        fusion, fusion_factory, fusion_mode,
        kf_noise, augmented_noise, complementary_noise,
    )

    rng = np.random.default_rng(sim_cfg.seed)
    imu = ImuSensor(imu_cfg, rng)
    wheel_odom = WheelOdometrySensor(wheel_cfg, rng)

    imu_dt = 1.0 / imu_cfg.rate_hz
    wheel_period_steps = max(
        1, int(round(imu_cfg.rate_hz / wheel_cfg.rate_hz))
    )
    n_steps = int(round(sim_cfg.duration_s * imu_cfg.rate_hz)) + 1

    t0_state = traj.state_at(0.0)
    first_wheel = wheel_odom.measure(t0_state)
    fusion_obj.initialise(t0_state.position, first_wheel)

    times = np.empty(n_steps)
    truth_pos = np.empty((n_steps, 2))
    truth_vel = np.empty((n_steps, 2))
    est_pos = np.empty((n_steps, 2))
    est_vel = np.empty((n_steps, 2))
    est_bias = np.empty((n_steps, 2))
    update_times_ns = np.empty(n_steps, dtype=np.int64)

    for k in range(n_steps):
        t = k * imu_dt
        state = traj.state_at(t)

        if k == 0:
            # The initial estimate was seeded above from the first wheel
            # sample. Log it at t=0 before propagating to the next IMU tick.
            update_times_ns[k] = 0
        else:
            accel_meas = imu.measure(state, imu_dt)

            start_ns = time.perf_counter_ns()
            fusion_obj.predict(accel_meas, imu_dt)
            if k % wheel_period_steps == 0:
                fusion_obj.update_wheel_velocity(wheel_odom.measure(state))
            end_ns = time.perf_counter_ns()
            update_times_ns[k] = end_ns - start_ns

        times[k] = t
        truth_pos[k] = state.position
        truth_vel[k] = state.velocity
        est_pos[k] = fusion_obj.position
        est_vel[k] = fusion_obj.velocity
        est_bias[k] = fusion_obj.accel_bias

    pos_error = np.linalg.norm(est_pos - truth_pos, axis=1)
    return SimulationResult(
        name=name,
        time_s=times,
        truth_position=truth_pos,
        truth_velocity=truth_vel,
        estimate_position=est_pos,
        estimate_velocity=est_vel,
        estimate_bias=est_bias,
        rms_position_error_m=float(np.sqrt(np.mean(pos_error**2))),
        final_position_error_m=float(pos_error[-1]),
        mean_update_time_us=float(
            np.mean(update_times_ns[1:]) / 1000.0 if n_steps > 1 else 0.0
        ),
    )
