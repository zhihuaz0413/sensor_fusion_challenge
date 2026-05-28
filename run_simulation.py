"""Visualise the robot + sensors + fusion estimators on a chosen trajectory.

Each selected fusion mode gets its own subplot showing the analytical
ground-truth path, the estimated path, and the robot chassis at its final
pose; a bottom strip plots position error vs time so the modes can be ranked
at a glance.

CLI examples::

    python run_simulation.py
    python run_simulation.py --trajectory circle --noise high
    python run_simulation.py --modes kf complementary
    python run_simulation.py --trajectory zigzag --save outputs/compare.png
    python run_simulation.py --animate --save outputs/compare.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from config import (
    NoiseLevel,
    SimulationConfig,
    imu_config_for,
    wheel_config_for,
)
from robot import TRAJECTORY_CATALOG, Robot
from sensor_fusion import ComplementaryNoise, FusionMode
from simulation import SimulationResult, run_simulation


MODE_COLORS = {
    FusionMode.KF:            "#1f77b4",
    FusionMode.KF_AUGMENTED:  "#17becf",
    FusionMode.IMU_ONLY:      "#d62728",
    FusionMode.ODOMETRY_ONLY: "#2ca02c",
    FusionMode.COMPLEMENTARY: "#9467bd",
}

DEFAULT_MODES = [
    FusionMode.KF,
    FusionMode.IMU_ONLY,
    FusionMode.ODOMETRY_ONLY,
    FusionMode.COMPLEMENTARY,
]


def _grid_shape(n: int) -> tuple[int, int]:
    """Return ``(nrows, ncols)`` that fits ``n`` subplots roughly square."""
    if n <= 1:
        return 1, 1
    if n == 2:
        return 1, 2
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 2, 3
    return 3, 3


def _resolve_modes(names: list[str] | None) -> list[FusionMode]:
    if not names:
        return DEFAULT_MODES
    return [FusionMode(n) for n in names]


def _run_scenarios(
    trajectory_name: str,
    noise: NoiseLevel,
    modes: list[FusionMode],
    duration_s: float,
    seed: int,
    imu_rate_hz: float,
    wheel_rate_hz: float,
) -> dict[FusionMode, SimulationResult]:
    sim_cfg = SimulationConfig(duration_s=duration_s, seed=seed)
    imu_cfg = imu_config_for(noise, rate_hz=imu_rate_hz)
    wheel_cfg = wheel_config_for(noise, rate_hz=wheel_rate_hz)
    trajectory = TRAJECTORY_CATALOG[trajectory_name]

    results: dict[FusionMode, SimulationResult] = {}
    for mode in modes:
        results[mode] = run_simulation(
            f"{mode.value}/{trajectory_name}/{noise.value}",
            sim_config=sim_cfg,
            imu_config=imu_cfg,
            wheel_config=wheel_cfg,
            trajectory=trajectory,
            fusion_mode=mode,
            complementary_noise=ComplementaryNoise(wheel_weight=0.2),
        )
    return results


def _global_xy_limits(
    results: dict[FusionMode, SimulationResult]
) -> tuple[float, float, float, float]:
    """Common axis limits so every panel uses the same camera."""
    chunks = []
    for r in results.values():
        chunks.append(r.truth_position)
        chunks.append(r.estimate_position)
    all_pos = np.concatenate(chunks, axis=0)
    xmin, ymin = all_pos.min(axis=0)
    xmax, ymax = all_pos.max(axis=0)
    pad = max(0.5, 0.1 * max(xmax - xmin, ymax - ymin))
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad


def _add_robot_patches(
    ax: plt.Axes,
    *,
    facecolor: str = "#2f6fb6",
    edgecolor: str = "#0a3a6e",
    alpha: float = 1.0,
) -> tuple[patches.Polygon, patches.Polygon, patches.Polygon]:
    """Add empty body + two wheels to ``ax`` and return the patch handles."""
    body = patches.Polygon(
        np.zeros((4, 2)), closed=True,
        facecolor=facecolor, edgecolor=edgecolor, lw=1.0,
        alpha=alpha, zorder=4,
    )
    left = patches.Polygon(
        np.zeros((4, 2)), closed=True,
        facecolor="#1f1f1f", edgecolor="black", lw=0.6,
        alpha=alpha, zorder=5,
    )
    right = patches.Polygon(
        np.zeros((4, 2)), closed=True,
        facecolor="#1f1f1f", edgecolor="black", lw=0.6,
        alpha=alpha, zorder=5,
    )
    ax.add_patch(body)
    ax.add_patch(left)
    ax.add_patch(right)
    return body, left, right


def _set_robot_pose(
    robot: Robot,
    body: patches.Polygon,
    left: patches.Polygon,
    right: patches.Polygon,
    pos: np.ndarray,
    yaw: float,
) -> None:
    body.set_xy(robot.body_corners(pos, yaw))
    left.set_xy(robot.wheel_corners(pos, yaw, +1))
    right.set_xy(robot.wheel_corners(pos, yaw, -1))


def build_static_figure(
    results: dict[FusionMode, SimulationResult],
    trajectory_name: str,
    noise: NoiseLevel,
    robot: Robot,
) -> plt.Figure:
    """Return a multi-panel figure with one (x, y) subplot per fusion mode."""
    modes = list(results)
    nrows, ncols = _grid_shape(len(modes))

    fig = plt.figure(
        figsize=(5.2 * ncols, 4.2 * nrows + 2.8),
        layout="constrained",
    )
    gs = fig.add_gridspec(
        nrows + 1, ncols,
        height_ratios=[*([3] * nrows), 2],
    )

    xmin, xmax, ymin, ymax = _global_xy_limits(results)

    for i, mode in enumerate(modes):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        result = results[mode]

        ax.plot(
            result.truth_position[:, 0], result.truth_position[:, 1],
            color="k", lw=1.6, label="truth",
        )
        ax.plot(
            result.estimate_position[:, 0], result.estimate_position[:, 1],
            color=MODE_COLORS[mode], lw=1.5, alpha=0.9, label=mode.value,
        )

        # Robot chassis at the trajectory's final pose.
        body, lw_, rw_ = _add_robot_patches(ax)
        final_pos = result.truth_position[-1]
        final_yaw = Robot.yaw_from_velocity(result.truth_velocity[-1])
        _set_robot_pose(robot, body, lw_, rw_, final_pos, final_yaw)

        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"{mode.value}  —  RMS={result.rms_position_error_m:.3f} m, "
            f"final={result.final_position_error_m:.3f} m"
        )
        ax.legend(loc="best", fontsize=8)

    # Bottom strip: position error vs time across all modes.
    ax_err = fig.add_subplot(gs[-1, :])
    for mode in modes:
        result = results[mode]
        err = np.linalg.norm(
            result.estimate_position - result.truth_position, axis=1
        )
        ax_err.plot(
            result.time_s, err,
            color=MODE_COLORS[mode], lw=1.4, label=mode.value,
        )
    ax_err.set_xlabel("t [s]")
    ax_err.set_ylabel("‖estimate − truth‖ [m]")
    ax_err.set_yscale("symlog", linthresh=1e-3)
    ax_err.grid(True, which="both", alpha=0.3)
    ax_err.legend(loc="best", fontsize=9, ncols=min(4, len(modes)))
    ax_err.set_title("Position error over time")

    first = next(iter(results.values()))
    fig.suptitle(
        f"Trajectory: {trajectory_name}   |   Noise: {noise.value}   |   "
        f"Duration: {first.time_s[-1]:.1f} s",
        fontsize=13,
    )
    return fig


def build_animated_figure(
    results: dict[FusionMode, SimulationResult],
    trajectory_name: str,
    noise: NoiseLevel,
    robot: Robot,
    fps: int,
) -> tuple[plt.Figure, animation.FuncAnimation]:
    """Return ``(fig, anim)`` with one animated panel per fusion mode."""
    modes = list(results)
    nrows, ncols = _grid_shape(len(modes))

    first = next(iter(results.values()))
    full_times = first.time_s
    full_dt = float(full_times[1] - full_times[0])
    step = max(1, int(round(1.0 / (fps * full_dt))))
    frame_indices = np.arange(0, len(full_times), step)

    xmin, xmax, ymin, ymax = _global_xy_limits(results)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.2 * ncols, 4.2 * nrows + 1.2),
        squeeze=False,
        layout="constrained",
    )
    panels: list[dict] = []
    for i, mode in enumerate(modes):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        result = results[mode]

        # Faint full ground-truth as a reference background.
        ax.plot(
            result.truth_position[:, 0], result.truth_position[:, 1],
            color="k", lw=0.7, alpha=0.3,
        )
        (truth_line,) = ax.plot([], [], color="k", lw=1.5, label="truth")
        (est_line,) = ax.plot([], [], color=MODE_COLORS[mode], lw=1.6, label=mode.value)
        body, lw_, rw_ = _add_robot_patches(ax)

        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"{mode.value}  —  RMS={result.rms_position_error_m:.3f} m"
        )
        ax.legend(loc="lower right", fontsize=8)

        text = ax.text(
            0.02, 0.97, "", transform=ax.transAxes,
            ha="left", va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=3),
        )

        panels.append({
            "result": result,
            "truth_line": truth_line,
            "est_line": est_line,
            "body": body,
            "left": lw_,
            "right": rw_,
            "text": text,
        })

    # Hide unused subplots when len(modes) < nrows * ncols.
    for j in range(len(modes), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(
        f"Trajectory: {trajectory_name}   |   Noise: {noise.value}",
        fontsize=13,
    )

    def init():
        artists = []
        for p in panels:
            p["truth_line"].set_data([], [])
            p["est_line"].set_data([], [])
            p["body"].set_xy(np.zeros((4, 2)))
            p["left"].set_xy(np.zeros((4, 2)))
            p["right"].set_xy(np.zeros((4, 2)))
            p["text"].set_text("")
            artists.extend([
                p["truth_line"], p["est_line"],
                p["body"], p["left"], p["right"], p["text"],
            ])
        return artists

    def update(frame: int):
        i = int(frame_indices[frame]) + 1  # inclusive slice end
        artists = []
        for p in panels:
            res = p["result"]
            p["truth_line"].set_data(res.truth_position[:i, 0], res.truth_position[:i, 1])
            p["est_line"].set_data(res.estimate_position[:i, 0], res.estimate_position[:i, 1])
            pos = res.truth_position[i - 1]
            yaw = Robot.yaw_from_velocity(res.truth_velocity[i - 1])
            _set_robot_pose(robot, p["body"], p["left"], p["right"], pos, yaw)
            err = float(np.linalg.norm(res.estimate_position[i - 1] - pos))
            p["text"].set_text(
                f"t = {res.time_s[i - 1]:5.2f} s\nerr = {err:.3f} m"
            )
            artists.extend([
                p["truth_line"], p["est_line"],
                p["body"], p["left"], p["right"], p["text"],
            ])
        return artists

    anim = animation.FuncAnimation(
        fig, update,
        frames=len(frame_indices),
        init_func=init,
        interval=1000.0 / fps,
        blit=False,
        repeat=False,
    )
    return fig, anim


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trajectory", choices=list(TRAJECTORY_CATALOG), default="sine",
        help="Which ground-truth trajectory to drive (default: sine).",
    )
    parser.add_argument(
        "--noise", choices=[lvl.value for lvl in NoiseLevel], default="medium",
        help="Sensor noise preset (default: medium).",
    )
    parser.add_argument(
        "--modes", nargs="+", choices=[m.value for m in FusionMode], default=None,
        help=(
            "Fusion modes to compare. Defaults to "
            "kf imu_only odometry_only complementary; kf_augmented is available "
            "when requested explicitly."
        ),
    )
    parser.add_argument(
        "--duration", type=float, default=12.0,
        help="Simulated duration in seconds (default: 12).",
    )
    parser.add_argument(
        "--seed", type=int, default=7, help="RNG seed (default: 7).",
    )
    parser.add_argument(
        "--imu-rate", type=float, default=1000.0,
        help="IMU sample rate in Hz (default: 1000).",
    )
    parser.add_argument(
        "--wheel-rate", type=float, default=50.0,
        help="Wheel-odom sample rate in Hz (default: 50).",
    )
    parser.add_argument(
        "--animate", action="store_true",
        help="Build an animation instead of a static figure.",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Animation frame rate (default: 30).",
    )
    parser.add_argument(
        "--save", type=Path, default=None,
        help="Output path; e.g. outputs/compare.png or outputs/compare.gif. "
             "If omitted, the figure is shown interactively.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    noise = NoiseLevel(args.noise)
    modes = _resolve_modes(args.modes)

    print(
        f"Trajectory: {args.trajectory}   noise: {noise.value}   "
        f"modes: {[m.value for m in modes]}"
    )
    print(f"Running {len(modes)} simulation(s)...")
    results = _run_scenarios(
        trajectory_name=args.trajectory,
        noise=noise,
        modes=modes,
        duration_s=args.duration,
        seed=args.seed,
        imu_rate_hz=args.imu_rate,
        wheel_rate_hz=args.wheel_rate,
    )

    for mode, r in results.items():
        print(
            f"  {mode.value:>14}  rms={r.rms_position_error_m:.4f} m  "
            f"final={r.final_position_error_m:.4f} m  "
            f"per-step={r.mean_update_time_us:.2f} us"
        )

    robot = Robot()
    if args.animate:
        fig, anim = build_animated_figure(
            results, args.trajectory, noise, robot, fps=args.fps,
        )
    else:
        fig = build_static_figure(results, args.trajectory, noise, robot)
        anim = None

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        if anim is not None:
            suffix = args.save.suffix.lower()
            writer = (
                animation.PillowWriter(fps=args.fps) if suffix == ".gif"
                else animation.FFMpegWriter(fps=args.fps)
            )
            print(f"Saving animation to {args.save} ...")
            anim.save(str(args.save), writer=writer, dpi=120)
        else:
            print(f"Saving figure to {args.save} ...")
            fig.savefig(args.save, dpi=160, bbox_inches="tight")
        print("Done.")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
