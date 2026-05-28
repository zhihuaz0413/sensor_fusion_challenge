"""Analytical 2D trajectories for fusion / SLAM testing.

Each trajectory is a small object that exposes::

    state_at(t: float) -> RobotState

and returns the ground-truth planar state (position, velocity,
acceleration) at simulated time ``t``.  Position is in metres,
velocities in m/s, accelerations in m/s².  All accelerations are
analytical so an IMU model gets a self-consistent ground-truth signal
even at 1 kHz sampling.

The catalogue covers the shapes that commonly trip up dead reckoning /
SLAM front-ends:

- :class:`StraightLineTrajectory` - zero acceleration baseline, the
  "if this drifts you have a bias problem" case.
- :class:`CircleTrajectory` - constant centripetal acceleration, exercises
  yaw integration and the circular dead-reckoning failure mode.
- :class:`SineWaveTrajectory` - forward motion plus a lateral sinusoid,
  the original default trajectory used by the fusion demo.
- :class:`ZigzagTrajectory` - triangle-wave lateral motion built from
  the first few odd harmonics; sharp-but-smooth turning behaviour.
- :class:`FigureEightTrajectory` - Gerono lemniscate; self-intersecting
  path, good loop-closure stress test.
- :class:`StopAndGoTrajectory` - a straight line with a stationary
  pause in the middle; exposes integrators that don't notice zero
  motion.

`TrajectoryGenerator` is kept as a backward-compatible alias for
:class:`SineWaveTrajectory`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

__all__ = [
    "RobotState",
    "Trajectory",
    "StraightLineTrajectory",
    "CircleTrajectory",
    "SineWaveTrajectory",
    "ZigzagTrajectory",
    "FigureEightTrajectory",
    "StopAndGoTrajectory",
    "TrajectoryGenerator",
    "TRAJECTORY_CATALOG",
]


@dataclass(frozen=True)
class RobotState:
    """Ground-truth robot state in 2D."""

    t: float
    position: np.ndarray  # shape: (2,)
    velocity: np.ndarray  # shape: (2,)
    acceleration: np.ndarray  # shape: (2,)


class Trajectory(ABC):
    """Abstract base for an analytical 2D trajectory."""

    @abstractmethod
    def state_at(self, t: float) -> RobotState: ...

    @property
    def name(self) -> str:
        return type(self).__name__


def _as_state(t: float, p, v, a) -> RobotState:
    return RobotState(
        t=t,
        position=np.asarray(p, dtype=float),
        velocity=np.asarray(v, dtype=float),
        acceleration=np.asarray(a, dtype=float),
    )


@dataclass(frozen=True)
class StraightLineTrajectory(Trajectory):
    """Constant-velocity straight line.

    Accelerations are exactly zero, so this is the "is my IMU bias
    estimator working?" sanity case - any drift you see is bias.
    """

    speed: float = 1.0
    heading_rad: float = 0.0
    origin: tuple[float, float] = (0.0, 0.0)

    def state_at(self, t: float) -> RobotState:
        c, s = math.cos(self.heading_rad), math.sin(self.heading_rad)
        vx, vy = self.speed * c, self.speed * s
        px = self.origin[0] + vx * t
        py = self.origin[1] + vy * t
        return _as_state(t, (px, py), (vx, vy), (0.0, 0.0))


@dataclass(frozen=True)
class CircleTrajectory(Trajectory):
    """Uniform circular motion at constant angular speed.

    Acceleration is purely centripetal; SLAM front-ends that assume
    straight-line motion between scans will struggle here.
    """

    radius: float = 2.0
    angular_speed: float = 0.5  # rad/s; positive = counter-clockwise
    center: tuple[float, float] = (0.0, 0.0)
    phase_rad: float = 0.0

    def state_at(self, t: float) -> RobotState:
        w = self.angular_speed
        r = self.radius
        theta = w * t + self.phase_rad
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        px = self.center[0] + r * cos_t
        py = self.center[1] + r * sin_t
        vx = -r * w * sin_t
        vy = r * w * cos_t
        ax = -r * w * w * cos_t
        ay = -r * w * w * sin_t
        return _as_state(t, (px, py), (vx, vy), (ax, ay))


@dataclass(frozen=True)
class SineWaveTrajectory(Trajectory):
    """Forward motion + a lateral sinusoidal oscillation.

    Identical to the original `TrajectoryGenerator` defaults; kept here
    as a named member of the catalogue.
    """

    forward_speed: float = 0.8
    forward_amplitude: float = 0.2
    forward_freq: float = 0.8  # rad/s
    lateral_amplitude: float = 0.8
    lateral_freq: float = 0.5  # rad/s

    def state_at(self, t: float) -> RobotState:
        v = self.forward_speed
        Af, Wf = self.forward_amplitude, self.forward_freq
        Al, Wl = self.lateral_amplitude, self.lateral_freq

        px = v * t + Af * math.sin(Wf * t)
        py = Al * math.sin(Wl * t)

        vx = v + Af * Wf * math.cos(Wf * t)
        vy = Al * Wl * math.cos(Wl * t)

        ax = -Af * Wf * Wf * math.sin(Wf * t)
        ay = -Al * Wl * Wl * math.sin(Wl * t)

        return _as_state(t, (px, py), (vx, vy), (ax, ay))


@dataclass(frozen=True)
class ZigzagTrajectory(Trajectory):
    """Constant forward speed plus a (band-limited) triangle-wave lateral.

    The lateral motion is the truncated Fourier series of a triangle
    wave so position / velocity / acceleration stay analytical and
    differentiable everywhere.  ``harmonics`` controls how sharp the
    corners are; more harmonics → closer to an ideal triangle (and
    larger acceleration spikes near the corners).
    """

    forward_speed: float = 1.0
    amplitude: float = 1.0
    period_s: float = 4.0
    harmonics: int = 7

    def state_at(self, t: float) -> RobotState:
        w0 = 2.0 * math.pi / self.period_s
        A = self.amplitude
        coef = 8.0 * A / (math.pi**2)

        py = 0.0
        vy = 0.0
        ay = 0.0
        for k in range(self.harmonics):
            n = 2 * k + 1
            wn = n * w0
            sign = (-1.0) ** k
            inv_n2 = 1.0 / (n * n)
            py += coef * sign * math.sin(wn * t) * inv_n2
            vy += coef * sign * wn * math.cos(wn * t) * inv_n2
            ay += -coef * sign * (wn * wn) * math.sin(wn * t) * inv_n2

        px = self.forward_speed * t
        return _as_state(t, (px, py), (self.forward_speed, vy), (0.0, ay))


@dataclass(frozen=True)
class FigureEightTrajectory(Trajectory):
    """Lemniscate of Gerono - a smooth, self-intersecting figure-of-eight.

    Useful for loop-closure stress tests because the path crosses itself
    once per cycle.
    """

    scale: float = 2.0
    angular_speed: float = 0.3

    def state_at(self, t: float) -> RobotState:
        A = self.scale
        w = self.angular_speed
        wt = w * t
        sw, cw = math.sin(wt), math.cos(wt)
        s2w, c2w = math.sin(2.0 * wt), math.cos(2.0 * wt)

        px = A * sw
        py = 0.5 * A * s2w
        vx = A * w * cw
        vy = A * w * c2w
        ax = -A * w * w * sw
        ay = -2.0 * A * w * w * s2w
        return _as_state(t, (px, py), (vx, vy), (ax, ay))


@dataclass(frozen=True)
class StopAndGoTrajectory(Trajectory):
    """Straight-line drive with a stationary pause in the middle.

    The velocity profile is a smooth raised-cosine ramp:

        v(t) = v_max * 0.5 * (1 - cos(2π * phase(t)))

    where ``phase(t)`` linearly traverses [0, 1] during the *driving*
    intervals before and after the stop, so the position is
    continuously differentiable and there are no acceleration impulses.
    Good for catching IMU integrators that secretly add drift while the
    robot is supposedly stationary.
    """

    cruise_speed: float = 1.0
    drive_duration_s: float = 4.0
    stop_duration_s: float = 2.0
    heading_rad: float = 0.0

    def _phase(self, t: float) -> tuple[float, float]:
        """Return (speed, distance_travelled) at time t along the 1-D axis."""
        d = self.drive_duration_s
        s = self.stop_duration_s

        if t <= 0.0:
            return 0.0, 0.0

        if t < d:
            ramp = 0.5 * (1.0 - math.cos(math.pi * t / d))
            v = self.cruise_speed * ramp
            # Distance = ∫₀ᵗ v dt
            dist = self.cruise_speed * (
                0.5 * t - (d / (2.0 * math.pi)) * math.sin(math.pi * t / d)
            )
            return v, dist

        first_segment_dist = self.cruise_speed * 0.5 * d
        if t < d + s:
            return 0.0, first_segment_dist

        tau = t - (d + s)
        if tau < d:
            ramp = 0.5 * (1.0 - math.cos(math.pi * tau / d))
            v = self.cruise_speed * ramp
            extra = self.cruise_speed * (
                0.5 * tau - (d / (2.0 * math.pi)) * math.sin(math.pi * tau / d)
            )
            return v, first_segment_dist + extra

        return 0.0, 2.0 * first_segment_dist

    def state_at(self, t: float) -> RobotState:
        v_along, dist = self._phase(t)

        # Analytical acceleration along the drive axis.
        d = self.drive_duration_s
        s = self.stop_duration_s
        if 0.0 < t < d:
            a_along = (
                self.cruise_speed * math.pi / (2.0 * d) * math.sin(math.pi * t / d)
            )
        elif d <= t < d + s:
            a_along = 0.0
        elif d + s <= t < 2 * d + s:
            tau = t - (d + s)
            a_along = (
                self.cruise_speed * math.pi / (2.0 * d) * math.sin(math.pi * tau / d)
            )
        else:
            a_along = 0.0

        c, sh = math.cos(self.heading_rad), math.sin(self.heading_rad)
        return _as_state(
            t,
            (dist * c, dist * sh),
            (v_along * c, v_along * sh),
            (a_along * c, a_along * sh),
        )


# Backward-compatible alias so the rest of the repo can keep using
# `TrajectoryGenerator` for the default smooth trajectory.
TrajectoryGenerator = SineWaveTrajectory


TRAJECTORY_CATALOG: dict[str, Trajectory] = {
    "straight": StraightLineTrajectory(speed=1.0),
    "circle":   CircleTrajectory(radius=2.0, angular_speed=0.5),
    "sine":     SineWaveTrajectory(),
    "zigzag":   ZigzagTrajectory(forward_speed=1.0, amplitude=1.0, period_s=4.0),
    "figure8":  FigureEightTrajectory(scale=2.0, angular_speed=0.3),
    "stopgo":   StopAndGoTrajectory(),
}
