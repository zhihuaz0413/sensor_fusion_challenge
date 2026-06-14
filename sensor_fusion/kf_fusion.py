"""Fast 2D Kalman-filter sensor fusion.

This module also defines the abstract :class:`SensorFusion` interface
that the baselines in :mod:`sensor_fusion.dummy_fusion` implement, so
any estimator in the package can be benchmarked against the same
ground-truth + sensor harness with no code changes upstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

__all__ = [
    "SensorFusion",
    "FusionNoise",
    "KalmanFusion2D",
    "AugmentedFusionNoise",
    "AugmentedKalmanFusion2D",
]


class SensorFusion(ABC):
    """Common interface for 2D sensor-fusion estimators.

    All concrete estimators expose the same five operations:

    - :meth:`initialise` to seed the state with an initial pose / velocity,
    - :meth:`predict` to advance the state by ``dt`` using an IMU
      acceleration sample,
    - :meth:`update_wheel_velocity` to apply a wheel-odometry correction,
    - the read-only :attr:`position` and :attr:`velocity` properties for
      the current best-estimate state.

    :attr:`accel_bias` is a non-abstract convenience that defaults to a
    zero vector for estimators that do not model bias.
    """

    @abstractmethod
    def initialise(self, position: np.ndarray, velocity: np.ndarray) -> None:
        """Seed the state with an initial position and velocity."""

    @abstractmethod
    def predict(self, accel_meas: np.ndarray, dt: float) -> None:
        """Advance the state by ``dt`` seconds using the IMU acceleration."""

    @abstractmethod
    def update_wheel_velocity(self, velocity_meas: np.ndarray) -> None:
        """Correct the state using a wheel-odometry velocity measurement."""

    @property
    @abstractmethod
    def position(self) -> np.ndarray:
        """Current estimated 2D position, shape ``(2,)``."""

    @property
    @abstractmethod
    def velocity(self) -> np.ndarray:
        """Current estimated 2D velocity, shape ``(2,)``."""

    @property
    def accel_bias(self) -> np.ndarray:
        """Estimated IMU acceleration bias; defaults to zero."""
        return np.zeros(2, dtype=float)


@dataclass(frozen=True)
class FusionNoise:
    """Filter noise parameters.

    Attributes:
        accel_noise_std: Standard deviation of white acceleration noise
            assumed by the filter (m/s²).  Drives the process-noise
            entries in ``Q`` for position and velocity.
        accel_bias_rw_std: Standard deviation of the IMU bias random
            walk per √s.  Drives the process-noise entries in ``Q`` for
            the bias state.
        wheel_velocity_noise_std: Standard deviation of the wheel-odom
            velocity measurement noise (m/s).  Used to build ``R``.
        bias_initial_std: Standard deviation of the *initial* IMU bias
            seen at startup (m/s²).  Used to seed ``P[bax, bax]`` and
            ``P[bay, bay]`` in :meth:`KalmanFusion2D.initialise`, so the
            filter is appropriately under-confident in its zero-bias
            prior at the start of each run.
    """

    accel_noise_std: float = 0.08
    accel_bias_rw_std: float = 0.01
    wheel_velocity_noise_std: float = 0.05
    bias_initial_std: float = 0.45  # ≈ √0.2 — preserves the pre-existing prior


class KalmanFusion2D(SensorFusion):
    """Linear Kalman filter for 2D position/velocity estimation.

    State vector:
        [px, py, vx, vy, bax, bay]

    The IMU acceleration is used as a control input. The acceleration bias is
    estimated as part of the state and is subtracted during prediction.
    """

    def __init__(self, noise: FusionNoise | None = None):
        self.noise = noise or FusionNoise()
        self.x = np.zeros(6, dtype=float)
        self.P = np.diag([1.0, 1.0, 1.0, 1.0, 0.1, 0.1]).astype(float)
        self.I = np.eye(6, dtype=float)

        # Reused arrays to reduce allocations in high-rate loop.
        self.F = np.eye(6, dtype=float)
        self.B = np.zeros((6, 2), dtype=float)
        self.Q = np.zeros((6, 6), dtype=float)
        self.H_vel = np.zeros((2, 6), dtype=float)
        self.H_vel[0, 2] = 1.0
        self.H_vel[1, 3] = 1.0
        self.R_vel = np.eye(2, dtype=float) * self.noise.wheel_velocity_noise_std**2

    @property
    def position(self) -> np.ndarray:
        return self.x[0:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:4].copy()

    @property
    def accel_bias(self) -> np.ndarray:
        return self.x[4:6].copy()

    def initialise(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self.x[:] = 0.0
        self.x[0:2] = position
        self.x[2:4] = velocity
        bias_var = self.noise.bias_initial_std ** 2
        self.P = np.diag([0.05, 0.05, 0.1, 0.1, bias_var, bias_var]).astype(float)

    def predict(self, accel_meas: np.ndarray, dt: float) -> None:
        """Prediction step using IMU acceleration at high rate."""
        dt2 = dt * dt

        # x_k+1 = F x_k + B a_measured
        self.F[:] = self.I
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.F[0, 4] = -0.5 * dt2
        self.F[1, 5] = -0.5 * dt2
        self.F[2, 4] = -dt
        self.F[3, 5] = -dt

        self.B.fill(0.0)
        self.B[0, 0] = 0.5 * dt2
        self.B[1, 1] = 0.5 * dt2
        self.B[2, 0] = dt
        self.B[3, 1] = dt

        self.x = self.F @ self.x + self.B @ accel_meas

        accel_var = self.noise.accel_noise_std**2
        bias_var = self.noise.accel_bias_rw_std**2
        self.Q.fill(0.0)
        # Integrated acceleration uncertainty.
        self.Q[0, 0] = 0.25 * dt2 * dt2 * accel_var
        self.Q[1, 1] = 0.25 * dt2 * dt2 * accel_var
        self.Q[2, 2] = dt2 * accel_var
        self.Q[3, 3] = dt2 * accel_var
        # Bias random walk.
        self.Q[4, 4] = dt * bias_var
        self.Q[5, 5] = dt * bias_var

        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_wheel_velocity(self, velocity_meas: np.ndarray) -> None:
        """Correction step using wheel odometry velocity measurement."""
        z = velocity_meas
        y = z - self.H_vel @ self.x
        S = self.H_vel @ self.P @ self.H_vel.T + self.R_vel
        K = self.P @ self.H_vel.T @ np.linalg.inv(S)
        self.x = self.x + K @ y

        # Joseph form improves numerical stability.
        KH = K @ self.H_vel
        self.P = (self.I - KH) @ self.P @ (self.I - KH).T + K @ self.R_vel @ K.T


@dataclass(frozen=True)
class AugmentedFusionNoise:
    """Filter noise parameters for :class:`AugmentedKalmanFusion2D`.

    Adds two extra knobs over :class:`FusionNoise`:

    - ``scale_initial_std``: how uncertain we are about the wheel scale
      error at startup (dimensionless; e.g. 0.05 means we expect ~5 %
      scale error a priori).
    - ``scale_rw_std``: random-walk std-dev of the wheel scale per √s.
      Should be small — wheel scale changes slowly, mostly with tyre
      pressure, payload, surface, etc.
    """

    accel_noise_std: float = 0.08
    accel_bias_rw_std: float = 0.01
    wheel_velocity_noise_std: float = 0.05
    bias_initial_std: float = 0.45
    scale_initial_std: float = 0.05
    scale_rw_std: float = 1e-3


class AugmentedKalmanFusion2D(SensorFusion):
    """Kalman filter with online wheel-scale estimation.

    State vector:
        ``[px, py, vx, vy, bax, bay, sx, sy]``

    Where ``sx, sy`` are wheel scale errors per axis.  The wheel
    measurement model is::

        z_v = (1 + s) * v_truth + N(0, R)

    which is nonlinear in ``s * v``.  The predict step stays linear (IMU
    acceleration as control input, with bias subtracted via ``F``); the
    wheel update step linearises ``h(x) = (1+s)·v`` around the current
    state estimate (EKF-style).

    Compared to :class:`KalmanFusion2D`, this filter can absorb the
    multiplicative wheel scale error (`WheelOdomConfig.scale_error`)
    into a dedicated state instead of letting it leak into the bias
    state or the position estimate.
    """

    def __init__(self, noise: AugmentedFusionNoise | None = None):
        self.noise = noise or AugmentedFusionNoise()
        self.x = np.zeros(8, dtype=float)
        self.I = np.eye(8, dtype=float)

        # Reused buffers.
        self.F = np.eye(8, dtype=float)
        self.B = np.zeros((8, 2), dtype=float)
        self.Q = np.zeros((8, 8), dtype=float)
        self.H = np.zeros((2, 8), dtype=float)
        self.R = np.eye(2, dtype=float) * self.noise.wheel_velocity_noise_std**2
        self.P = np.zeros((8, 8), dtype=float)

    # State accessors --------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self.x[0:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:4].copy()

    @property
    def accel_bias(self) -> np.ndarray:
        return self.x[4:6].copy()

    @property
    def wheel_scale(self) -> np.ndarray:
        """Currently estimated wheel-scale error per axis (dimensionless)."""
        return self.x[6:8].copy()

    # API --------------------------------------------------------------
    def initialise(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self.x[:] = 0.0
        self.x[0:2] = position
        # The first wheel reading is already biased by (1 + scale_error);
        # we treat it as our best initial velocity estimate (s ≈ 0 prior).
        self.x[2:4] = velocity
        bias_var = self.noise.bias_initial_std ** 2
        scale_var = self.noise.scale_initial_std ** 2
        self.P = np.diag([
            0.05, 0.05, 0.1, 0.1, bias_var, bias_var, scale_var, scale_var,
        ]).astype(float)

    def predict(self, accel_meas: np.ndarray, dt: float) -> None:
        """Predict step — identical to KF6 plus pass-through on scale states."""
        dt2 = dt * dt

        self.F[:] = self.I
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.F[0, 4] = -0.5 * dt2
        self.F[1, 5] = -0.5 * dt2
        self.F[2, 4] = -dt
        self.F[3, 5] = -dt
        # Scale states are constant in expectation (slow random walk).

        self.B.fill(0.0)
        self.B[0, 0] = 0.5 * dt2
        self.B[1, 1] = 0.5 * dt2
        self.B[2, 0] = dt
        self.B[3, 1] = dt

        self.x = self.F @ self.x + self.B @ accel_meas

        accel_var = self.noise.accel_noise_std ** 2
        bias_var = self.noise.accel_bias_rw_std ** 2
        scale_var = self.noise.scale_rw_std ** 2
        self.Q.fill(0.0)
        self.Q[0, 0] = 0.25 * dt2 * dt2 * accel_var
        self.Q[1, 1] = 0.25 * dt2 * dt2 * accel_var
        self.Q[2, 2] = dt2 * accel_var
        self.Q[3, 3] = dt2 * accel_var
        self.Q[4, 4] = dt * bias_var
        self.Q[5, 5] = dt * bias_var
        self.Q[6, 6] = dt * scale_var
        self.Q[7, 7] = dt * scale_var

        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_wheel_velocity(self, velocity_meas: np.ndarray) -> None:
        """Correction step using a wheel-odometry velocity measurement.

        ``h(x) = ((1 + sx)·vx, (1 + sy)·vy)`` is linearised around the
        current state to give an EKF Jacobian ``H``.
        """
        vx, vy = self.x[2], self.x[3]
        sx, sy = self.x[6], self.x[7]

        h = np.array([(1.0 + sx) * vx, (1.0 + sy) * vy])
        y = np.asarray(velocity_meas, dtype=float) - h

        self.H.fill(0.0)
        self.H[0, 2] = 1.0 + sx
        self.H[0, 6] = vx
        self.H[1, 3] = 1.0 + sy
        self.H[1, 7] = vy

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y

        KH = K @ self.H
        self.P = (self.I - KH) @ self.P @ (self.I - KH).T + K @ self.R @ K.T
