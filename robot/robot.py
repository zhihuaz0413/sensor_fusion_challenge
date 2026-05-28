"""Wheeled-robot chassis model.

The actual configuration dataclasses (`RobotShape`, `RobotConfig`) live
in :mod:`config` so the whole project has a single source of truth for
tunable parameters.  The trajectory model (`Trajectory`, `RobotState`,
concrete trajectory classes) lives in :mod:`robot.trajectory`.

This module wires those two halves together via the `Robot` class - a
thin composition of a `RobotConfig` (geometry + drive-train) and a
swappable `Trajectory` (how the robot moves).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

# Make the project-root `config.py` importable regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import RobotConfig, RobotShape  # noqa: E402

from .trajectory import (  # noqa: E402
    RobotState,
    SineWaveTrajectory,
    Trajectory,
    TrajectoryGenerator,
)

__all__ = [
    "RobotConfig",
    "RobotShape",
    "RobotState",
    "Trajectory",
    "TrajectoryGenerator",
    "Robot",
]


class Robot:
    """A wheeled robot composed of a `RobotConfig` and a `Trajectory`.

    `config` carries the chassis geometry (`RobotShape`) and the
    drive-train numbers (`wheel_base_m`, `wheel_radius_m`).  `trajectory`
    is any object implementing the `Trajectory` interface; if omitted,
    the default `SineWaveTrajectory` is used so old code that simply
    wrote ``Robot()`` keeps behaving the way it always did.

    For backwards compatibility a `RobotShape` may also be passed
    directly as the `config` argument; it is upgraded to a `RobotConfig`
    with default wheel base / radius.
    """

    def __init__(
        self,
        config: RobotConfig | RobotShape | None = None,
        trajectory: Trajectory | None = None,
    ) -> None:
        if config is None:
            self.config = RobotConfig()
        elif isinstance(config, RobotShape):
            self.config = RobotConfig(shape=config)
        else:
            self.config = config

        self.trajectory = (
            trajectory if trajectory is not None else SineWaveTrajectory()
        )

    # Convenience accessors so callers don't always have to dot through
    # `robot.config.shape.body_length` etc.
    @property
    def shape(self) -> RobotShape:
        return self.config.shape

    @property
    def wheel_base_m(self) -> float:
        return self.config.wheel_base_m

    @property
    def wheel_radius_m(self) -> float:
        return self.config.wheel_radius_m

    def state_at(self, t: float) -> RobotState:
        return self.trajectory.state_at(t)

    def states_at(self, times: Iterable[float]) -> list[RobotState]:
        return [self.trajectory.state_at(float(t)) for t in times]

    def yaw_at(self, t: float) -> float:
        return self.yaw_from_velocity(self.trajectory.state_at(t).velocity)

    @staticmethod
    def yaw_from_velocity(velocity: np.ndarray) -> float:
        vx, vy = float(velocity[0]), float(velocity[1])
        if vx == 0.0 and vy == 0.0:
            return 0.0
        return math.atan2(vy, vx)

    def body_corners(self, center: np.ndarray, yaw: float) -> np.ndarray:
        """World-frame corners of the chassis rectangle (shape ``(4, 2)``)."""
        hl = 0.5 * self.shape.body_length
        hw = 0.5 * self.shape.body_width
        local = np.array(
            [
                [hl, hw],
                [hl, -hw],
                [-hl, -hw],
                [-hl, hw],
            ]
        )
        return self._to_world(local, center, yaw)

    def wheel_corners(
        self, center: np.ndarray, yaw: float, side: int
    ) -> np.ndarray:
        """Corners of one wheel rectangle.

        `side` is ``+1`` for the left wheel and ``-1`` for the right wheel.
        """
        if side not in (-1, 1):
            raise ValueError("`side` must be +1 (left) or -1 (right)")
        cy = side * self.shape.wheel_offset_y
        hl = 0.5 * self.shape.wheel_length
        hw = 0.5 * self.shape.wheel_width
        local = np.array(
            [
                [hl, cy + hw],
                [hl, cy - hw],
                [-hl, cy - hw],
                [-hl, cy + hw],
            ]
        )
        return self._to_world(local, center, yaw)

    def heading_arrow_tip(
        self, center: np.ndarray, yaw: float, pad: float = 0.05
    ) -> np.ndarray:
        """End point of an arrow drawn from the body centre along ``yaw``."""
        length = 0.5 * self.shape.body_length + pad
        return center + length * np.array([math.cos(yaw), math.sin(yaw)])

    @staticmethod
    def _to_world(
        local: np.ndarray, center: np.ndarray, yaw: float
    ) -> np.ndarray:
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        return (local @ rot.T) + center
