"""Wheeled-robot model: chassis geometry + a catalogue of trajectories.

Public API:

- `Robot`, `RobotShape` come from :mod:`robot.robot`.
- `RobotState`, the abstract `Trajectory` base, and the concrete
  trajectory classes / catalogue come from :mod:`robot.trajectory`.

Callers can use either the package-level imports
(``from robot import Robot, CircleTrajectory``) or the submodule
imports (``from robot.trajectory import CircleTrajectory``).
"""

from .robot import Robot, RobotConfig, RobotShape
from .trajectory import (
    TRAJECTORY_CATALOG,
    CircleTrajectory,
    FigureEightTrajectory,
    RobotState,
    SineWaveTrajectory,
    StopAndGoTrajectory,
    StraightLineTrajectory,
    Trajectory,
    TrajectoryGenerator,
    ZigzagTrajectory,
)

__all__ = [
    "Robot",
    "RobotConfig",
    "RobotShape",
    "RobotState",
    "Trajectory",
    "TrajectoryGenerator",
    "StraightLineTrajectory",
    "CircleTrajectory",
    "SineWaveTrajectory",
    "ZigzagTrajectory",
    "FigureEightTrajectory",
    "StopAndGoTrajectory",
    "TRAJECTORY_CATALOG",
]
