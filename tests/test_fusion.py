from __future__ import annotations

import numpy as np

from config import ImuConfig, SimulationConfig, WheelOdomConfig
from robot import StraightLineTrajectory
from sensor_fusion import FusionNoise, KalmanFusion2D
from simulation import run_simulation


def test_filter_runs_and_tracks_position() -> None:
    result = run_simulation(
        "test",
        sim_config=SimulationConfig(duration_s=3.0, seed=1),
        imu_config=ImuConfig(
            rate_hz=1000.0,
            accel_noise_std=0.03,
            bias_initial_std=0.02,
        ),
        wheel_config=WheelOdomConfig(rate_hz=100.0, velocity_noise_std=0.03),
        kf_noise=FusionNoise(
            accel_noise_std=0.05,
            accel_bias_rw_std=0.01,
            wheel_velocity_noise_std=0.05,
        ),
    )
    assert result.rms_position_error_m < 0.25
    assert result.mean_update_time_us < 1000.0  # comfortably below 1 ms per IMU tick on normal laptops


def test_prediction_with_zero_acceleration_keeps_velocity_constant() -> None:
    fusion = KalmanFusion2D()
    fusion.initialise(np.array([0.0, 0.0]), np.array([1.0, -2.0]))
    fusion.predict(np.array([0.0, 0.0]), 0.1)
    np.testing.assert_allclose(fusion.velocity, [1.0, -2.0], atol=1e-12)
    np.testing.assert_allclose(fusion.position, [0.1, -0.2], atol=1e-12)


def test_simulation_estimates_match_logged_timestamps() -> None:
    result = run_simulation(
        "timing",
        sim_config=SimulationConfig(duration_s=0.003, seed=1),
        imu_config=ImuConfig(
            rate_hz=1000.0,
            accel_noise_std=0.0,
            bias_initial_std=0.0,
            bias_random_walk_std=0.0,
        ),
        wheel_config=WheelOdomConfig(
            rate_hz=1000.0,
            velocity_noise_std=0.0,
            scale_error=0.0,
        ),
        trajectory=StraightLineTrajectory(speed=1.0),
    )

    np.testing.assert_allclose(result.time_s, [0.0, 0.001, 0.002, 0.003])
    np.testing.assert_allclose(
        result.estimate_position,
        result.truth_position,
        atol=1e-12,
    )
