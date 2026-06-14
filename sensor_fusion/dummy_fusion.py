"""Baseline (non-Bayesian) sensor-fusion estimators.

These are intentionally naive so they can serve as comparison points
for any real fusion algorithm (e.g. :class:`KalmanFusion2D` in
:mod:`sensor_fusion.kf_fusion`).  All three implement
:class:`SensorFusion`, so they can be dropped into the same simulation
harness with no other code changes.

- :class:`ImuDeadReckoning`: doubly-integrates IMU acceleration with zero
  correction from wheel odometry.  Drifts badly under bias / noise and
  is the canonical "why fusion?" exhibit.
- :class:`WheelDeadReckoning`: ignores the IMU and uses wheel velocity
  alone.  Doesn't drift on flat ground but lags during accelerations the
  wheel-odom sample rate cannot resolve.
- :class:`ComplementaryFusion`: a textbook complementary filter on
  velocity - integrates IMU between wheel updates and exponentially
  pulls velocity back toward each wheel reading.  O(1) per update with
  no matrix algebra, so it scales to multi-kHz IMU rates trivially.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kf_fusion import SensorFusion

__all__ = [
    "ImuDeadReckoning",
    "WheelDeadReckoning",
    "ComplementaryNoise",
    "ComplementaryFusion",
]


class ImuDeadReckoning(SensorFusion):
    """IMU-only double integration; wheel odometry is ignored.

    Used as a baseline to show how badly an un-corrected accelerometer
    drifts, especially under the ``HIGH`` / cheap-MEMS noise preset.
    """

    def __init__(self) -> None:
        self._position = np.zeros(2, dtype=float)
        self._velocity = np.zeros(2, dtype=float)

    @property
    def position(self) -> np.ndarray:
        return self._position.copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity.copy()

    def initialise(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self._position[:] = position
        self._velocity[:] = velocity

    def predict(self, accel_meas: np.ndarray, dt: float) -> None:
        a = np.asarray(accel_meas, dtype=float)
        self._position = (
            self._position + self._velocity * dt + 0.5 * a * dt * dt
        )
        self._velocity = self._velocity + a * dt

    def update_wheel_velocity(self, velocity_meas: np.ndarray) -> None:
        # Pure IMU-only: wheel updates are ignored on purpose.
        return None


class WheelDeadReckoning(SensorFusion):
    """Wheel-only dead reckoning; the IMU is consumed only for its ``dt``.

    Treats the most recent wheel-odometry velocity as the current
    velocity and integrates it forward.  This has no drift on a slip-free
    surface but lags any acceleration that happens between wheel samples.
    """

    def __init__(self) -> None:
        self._position = np.zeros(2, dtype=float)
        self._velocity = np.zeros(2, dtype=float)

    @property
    def position(self) -> np.ndarray:
        return self._position.copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity.copy()

    def initialise(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self._position[:] = position
        self._velocity[:] = velocity

    def predict(self, accel_meas: np.ndarray, dt: float) -> None:
        # Wheel velocity is treated as ground truth between updates; we
        # only need dt here to advance position.
        self._position = self._position + self._velocity * dt

    def update_wheel_velocity(self, velocity_meas: np.ndarray) -> None:
        self._velocity[:] = np.asarray(velocity_meas, dtype=float)


@dataclass(frozen=True)
class ComplementaryNoise:
    """Tuning for :class:`ComplementaryFusion`.

    Attributes:
        wheel_weight: Blend factor used on each wheel-odometry update::

            v <- (1 - wheel_weight) * v_imu + wheel_weight * v_wheel

            A value of ``0.0`` reduces to :class:`ImuDeadReckoning`,
            ``1.0`` reduces to :class:`WheelDeadReckoning`, and values
            between blend the two.  ``0.2`` is a reasonable default
            when wheel odometry is roughly twice as trusted as IMU
            integration over the wheel-update interval.
    """

    wheel_weight: float = 0.2


class ComplementaryFusion(SensorFusion):
    """Complementary filter blending IMU integration with wheel odometry.

    Between wheel updates the filter runs an IMU dead-reckoning step.
    On each wheel update the velocity is exponentially relaxed toward
    the wheel reading; position is left untouched at the update step so
    no jumps appear in the trajectory.
    """

    def __init__(self, noise: ComplementaryNoise | None = None) -> None:
        self.noise = noise or ComplementaryNoise()
        self._position = np.zeros(2, dtype=float)
        self._velocity = np.zeros(2, dtype=float)
        self._validate_weight(self.noise.wheel_weight)

    @staticmethod
    def _validate_weight(w: float) -> None:
        if not (0.0 <= w <= 1.0):
            raise ValueError(
                "ComplementaryNoise.wheel_weight must be in [0, 1], "
                f"got {w!r}"
            )

    @property
    def position(self) -> np.ndarray:
        return self._position.copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity.copy()

    def initialise(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self._position[:] = position
        self._velocity[:] = velocity

    def predict(self, accel_meas: np.ndarray, dt: float) -> None:
        a = np.asarray(accel_meas, dtype=float)
        self._position = (
            self._position + self._velocity * dt + 0.5 * a * dt * dt
        )
        self._velocity = self._velocity + a * dt

    def update_wheel_velocity(self, velocity_meas: np.ndarray) -> None:
        w = self.noise.wheel_weight
        v_wheel = np.asarray(velocity_meas, dtype=float)
        self._velocity = (1.0 - w) * self._velocity + w * v_wheel
