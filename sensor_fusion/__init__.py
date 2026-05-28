"""Sensor-fusion estimators for the 2D wheeled-robot testbed.

The package exposes:

- the common :class:`SensorFusion` interface (in
  :mod:`sensor_fusion.kf_fusion`),
- one Bayesian implementation - :class:`KalmanFusion2D` - plus its
  :class:`FusionNoise` tuning dataclass,
- three baseline implementations (:class:`ImuDeadReckoning`,
  :class:`WheelDeadReckoning`, :class:`ComplementaryFusion`) in
  :mod:`sensor_fusion.dummpy_fusion`,
- and a named "mode" selector :class:`FusionMode` + the
  :func:`make_fusion` factory that lets callers pick any of the above
  by name.

Typical use::

    from sensor_fusion import FusionMode, make_fusion

    for mode in FusionMode:
        fusion = make_fusion(mode)
        ...

or, in one go via the simulation harness::

    from simulation import run_simulation
    result = run_simulation("kf-circle", fusion_mode="kf", ...)
"""

from __future__ import annotations

from enum import Enum
from typing import Callable

from .dummpy_fusion import (
    ComplementaryFusion,
    ComplementaryNoise,
    ImuDeadReckoning,
    WheelDeadReckoning,
)
from .kf_fusion import (
    AugmentedFusionNoise,
    AugmentedKalmanFusion2D,
    FusionNoise,
    KalmanFusion2D,
    SensorFusion,
)

__all__ = [
    "SensorFusion",
    "FusionNoise",
    "KalmanFusion2D",
    "AugmentedFusionNoise",
    "AugmentedKalmanFusion2D",
    "ImuDeadReckoning",
    "WheelDeadReckoning",
    "ComplementaryNoise",
    "ComplementaryFusion",
    "FusionMode",
    "FUSION_MODES",
    "make_fusion",
]


class FusionMode(str, Enum):
    """Named sensor-fusion strategies, all implementing :class:`SensorFusion`.

    Members:
        KF: Linear Kalman filter with online IMU bias estimation
            (:class:`KalmanFusion2D`).  6-state model.
        KF_AUGMENTED: Extended Kalman filter with online IMU bias *and*
            wheel-scale estimation (:class:`AugmentedKalmanFusion2D`).
            8-state model; absorbs the multiplicative wheel scale error
            that would otherwise leak into the bias state.
        IMU_ONLY: Pure IMU dead reckoning; wheel updates ignored
            (:class:`ImuDeadReckoning`).  The drift baseline.
        ODOMETRY_ONLY: Wheel-velocity dead reckoning; IMU consumed only
            for the time step (:class:`WheelDeadReckoning`).
        COMPLEMENTARY: Textbook complementary filter blending IMU
            integration with wheel odometry
            (:class:`ComplementaryFusion`).
    """

    KF = "kf"
    KF_AUGMENTED = "kf_augmented"
    IMU_ONLY = "imu_only"
    ODOMETRY_ONLY = "odometry_only"
    COMPLEMENTARY = "complementary"


def _coerce_mode(mode: "FusionMode | str") -> FusionMode:
    return mode if isinstance(mode, FusionMode) else FusionMode(mode)


def make_fusion(
    mode: "FusionMode | str" = FusionMode.KF,
    *,
    kf_noise: FusionNoise | None = None,
    augmented_noise: AugmentedFusionNoise | None = None,
    complementary_noise: ComplementaryNoise | None = None,
) -> SensorFusion:
    """Construct a fresh fusion estimator for ``mode``.

    Tuning arguments are mode-specific and ignored when irrelevant:

    - ``kf_noise`` is only used for :attr:`FusionMode.KF`.
    - ``augmented_noise`` is only used for :attr:`FusionMode.KF_AUGMENTED`.
    - ``complementary_noise`` is only used for
      :attr:`FusionMode.COMPLEMENTARY`.
    """
    m = _coerce_mode(mode)
    if m is FusionMode.KF:
        return KalmanFusion2D(kf_noise)
    if m is FusionMode.KF_AUGMENTED:
        return AugmentedKalmanFusion2D(augmented_noise)
    if m is FusionMode.IMU_ONLY:
        return ImuDeadReckoning()
    if m is FusionMode.ODOMETRY_ONLY:
        return WheelDeadReckoning()
    if m is FusionMode.COMPLEMENTARY:
        return ComplementaryFusion(complementary_noise)
    raise ValueError(f"Unknown fusion mode: {mode!r}")


#: A zero-arg factory per mode, handy for parameter sweeps and benchmarks.
FUSION_MODES: dict[FusionMode, Callable[[], SensorFusion]] = {
    mode: (lambda m=mode: make_fusion(m)) for mode in FusionMode
}
