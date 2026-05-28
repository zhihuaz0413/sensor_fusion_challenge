#!/usr/bin/env python3
"""Comprehensive sensor-fusion test + report generator.

Runs every :class:`sensor_fusion.FusionMode` against four batteries of
scenarios and writes a Markdown report to
``outputs/test_sensor_fusion_report.md`` plus a CSV with the raw
numbers.  The script returns exit code ``0`` if every invariant check
passes and ``1`` otherwise so it can be wired into CI.

Sections:

1. **Noise-level sweep** - all modes × all :class:`NoiseLevel` presets on
   a baseline ``sine`` trajectory.  Catches the canonical "good IMU vs
   cheap IMU" failure mode the challenge cares about.
2. **Trajectory sweep** - all modes × every shape in
   :data:`TRAJECTORY_CATALOG` at ``MEDIUM`` noise.  Catches motion-shape-
   specific failure modes (e.g. ``imu_only`` on ``stopgo``).
3. **Frequency sweep** - vary the IMU sample rate and the wheel-odom
   sample rate; verify the fusion still tracks at off-nominal cadences.
4. **Real-time test** - a tight latency-percentile measurement plus a
   sustained wall-clock loop at 1 kHz that counts missed deadlines.

Usage::

    python test_sensor_fusion.py                          # full run
    python test_sensor_fusion.py --quick                  # smaller grid
    python test_sensor_fusion.py --report path/report.md  # custom output
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from config import (
    IMU_PRESETS,
    WHEEL_PRESETS,
    NoiseLevel,
    SimulationConfig,
    imu_config_for,
    wheel_config_for,
)
from robot import TRAJECTORY_CATALOG
from sensor_fusion import (
    ComplementaryNoise,
    FusionMode,
    FusionNoise,
    make_fusion,
)
from sensors import ImuSensor, WheelOdometrySensor
from simulation import SimulationResult, run_simulation


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = REPO_ROOT / "outputs" / "test_sensor_fusion_report.md"
DEFAULT_CSV = REPO_ROOT / "outputs" / "test_sensor_fusion_results.csv"
DEFAULT_PDF = REPO_ROOT / "outputs" / "test_sensor_fusion_report.pdf"

COMPLEMENTARY_TUNING = ComplementaryNoise(wheel_weight=0.2)


# ─────────────────────────────────────────────────── data structures ──


@dataclass
class CheckResult:
    """One invariant check, with a human-readable verdict line."""

    name: str
    ok: bool
    detail: str


@dataclass
class Report:
    """Everything collected during a run, ready to be rendered."""

    noise_sweep: dict[tuple[FusionMode, NoiseLevel], SimulationResult] = field(
        default_factory=dict
    )
    trajectory_sweep: dict[tuple[FusionMode, str], SimulationResult] = field(
        default_factory=dict
    )
    imu_rate_sweep: dict[tuple[FusionMode, float], SimulationResult] = field(
        default_factory=dict
    )
    wheel_rate_sweep: dict[tuple[FusionMode, float], SimulationResult] = field(
        default_factory=dict
    )
    realtime_latency: dict[FusionMode, dict[str, float]] = field(default_factory=dict)
    wallclock_realtime: dict[FusionMode, dict[str, float]] = field(default_factory=dict)
    # KF run with FusionNoise tuned per noise-level preset (sine trajectory).
    tuned_kf_noise_sweep: dict[NoiseLevel, SimulationResult] = field(
        default_factory=dict
    )
    checks: list[CheckResult] = field(default_factory=list)


# ─────────────────────────────────────────────────────────── sweeps ──


def _run(
    name: str,
    *,
    mode: FusionMode,
    trajectory_name: str,
    noise: NoiseLevel,
    imu_rate_hz: float,
    wheel_rate_hz: float,
    duration_s: float,
    seed: int = 7,
) -> SimulationResult:
    return run_simulation(
        name,
        sim_config=SimulationConfig(duration_s=duration_s, seed=seed),
        imu_config=imu_config_for(noise, rate_hz=imu_rate_hz),
        wheel_config=wheel_config_for(noise, rate_hz=wheel_rate_hz),
        trajectory=TRAJECTORY_CATALOG[trajectory_name],
        fusion_mode=mode,
        complementary_noise=COMPLEMENTARY_TUNING,
    )


def run_noise_sweep(
    report: Report, *, duration_s: float, imu_hz: float, wheel_hz: float
) -> None:
    print("[1/4] noise-level sweep")
    for level in NoiseLevel:
        for mode in FusionMode:
            r = _run(
                f"noise/{mode.value}/{level.value}",
                mode=mode,
                trajectory_name="sine",
                noise=level,
                imu_rate_hz=imu_hz,
                wheel_rate_hz=wheel_hz,
                duration_s=duration_s,
            )
            report.noise_sweep[(mode, level)] = r


def tune_kf_noise_for(level: NoiseLevel) -> FusionNoise:
    """Build a `FusionNoise` matched to the sensor preset for ``level``.

    The KF's internal noise parameters should reflect *actual* sensor
    noise.  Otherwise the filter weights the IMU vs wheel correction
    against the wrong covariances and a hand-tuned constant-gain
    complementary filter can beat it.

    We floor the parameters at small positive values so the filter
    doesn't go degenerate at `NoiseLevel.IDEAL` (zero process noise →
    infinite confidence in the model).
    """
    imu = IMU_PRESETS[level]
    wheel = WHEEL_PRESETS[level]
    return FusionNoise(
        accel_noise_std=max(imu.accel_noise_std, 1e-3),
        accel_bias_rw_std=max(imu.bias_random_walk_std, 1e-4),
        wheel_velocity_noise_std=max(wheel.velocity_noise_std, 1e-3),
        bias_initial_std=max(imu.bias_initial_std, 1e-3),
    )


def run_kf_tuning_sweep(
    report: Report, *, duration_s: float, imu_hz: float, wheel_hz: float
) -> None:
    """Re-run the KF at each NoiseLevel with FusionNoise tuned to that level."""
    print("[1b] KF tuning sweep (FusionNoise matched to each noise preset)")
    for level in NoiseLevel:
        kf_noise = tune_kf_noise_for(level)
        report.tuned_kf_noise_sweep[level] = run_simulation(
            f"tuned_kf/sine/{level.value}",
            sim_config=SimulationConfig(duration_s=duration_s, seed=7),
            imu_config=imu_config_for(level, rate_hz=imu_hz),
            wheel_config=wheel_config_for(level, rate_hz=wheel_hz),
            trajectory=TRAJECTORY_CATALOG["sine"],
            fusion_mode=FusionMode.KF,
            kf_noise=kf_noise,
        )


def run_trajectory_sweep(
    report: Report, *, duration_s: float, imu_hz: float, wheel_hz: float
) -> None:
    print("[2/4] trajectory sweep")
    for traj_name in TRAJECTORY_CATALOG:
        for mode in FusionMode:
            r = _run(
                f"traj/{mode.value}/{traj_name}",
                mode=mode,
                trajectory_name=traj_name,
                noise=NoiseLevel.MEDIUM,
                imu_rate_hz=imu_hz,
                wheel_rate_hz=wheel_hz,
                duration_s=duration_s,
            )
            report.trajectory_sweep[(mode, traj_name)] = r


def run_frequency_sweep(
    report: Report,
    *,
    duration_s: float,
    imu_rates: list[float],
    wheel_rates: list[float],
    base_imu_hz: float,
    base_wheel_hz: float,
) -> None:
    print("[3/4] frequency sweep (IMU rate + wheel-odom rate)")
    for imu_hz in imu_rates:
        for mode in FusionMode:
            r = _run(
                f"imu_rate/{mode.value}/{imu_hz:.0f}",
                mode=mode,
                trajectory_name="sine",
                noise=NoiseLevel.MEDIUM,
                imu_rate_hz=imu_hz,
                wheel_rate_hz=base_wheel_hz,
                duration_s=duration_s,
            )
            report.imu_rate_sweep[(mode, imu_hz)] = r

    for wheel_hz in wheel_rates:
        for mode in FusionMode:
            r = _run(
                f"wheel_rate/{mode.value}/{wheel_hz:.0f}",
                mode=mode,
                trajectory_name="sine",
                noise=NoiseLevel.MEDIUM,
                imu_rate_hz=base_imu_hz,
                wheel_rate_hz=wheel_hz,
                duration_s=duration_s,
            )
            report.wheel_rate_sweep[(mode, wheel_hz)] = r


# ─────────────────────────────────────────────────── real-time test ──


def _precompute_measurements(
    *,
    trajectory_name: str,
    n_steps: int,
    dt: float,
    wheel_period: int,
    noise: NoiseLevel,
    imu_rate_hz: float,
    wheel_rate_hz: float,
    seed: int = 7,
) -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    imu = ImuSensor(imu_config_for(noise, rate_hz=imu_rate_hz), rng)
    wheel = WheelOdometrySensor(
        wheel_config_for(noise, rate_hz=wheel_rate_hz), rng
    )
    trajectory = TRAJECTORY_CATALOG[trajectory_name]

    accels = np.empty((n_steps, 2))
    wheel_meas: list[np.ndarray] = []
    for k in range(n_steps):
        t = k * dt
        s = trajectory.state_at(t)
        accels[k] = imu.measure(s, dt)
        if k % wheel_period == 0:
            wheel_meas.append(wheel.measure(s))
    return accels, wheel_meas


def latency_percentiles(
    mode: FusionMode,
    *,
    imu_rate_hz: float = 1000.0,
    wheel_rate_hz: float = 50.0,
    duration_s: float = 4.0,
    noise: NoiseLevel = NoiseLevel.MEDIUM,
    trajectory_name: str = "sine",
) -> dict[str, float]:
    """Tight-loop measurement of `predict` (+ occasional wheel update) cost.

    Sensors are pre-computed so the timing isolates the fusion cost.
    GC is disabled during the loop to avoid Python's cyclic-GC pauses
    showing up as latency spikes.
    """
    dt = 1.0 / imu_rate_hz
    wheel_period = max(1, int(round(imu_rate_hz / wheel_rate_hz)))
    n_steps = int(round(duration_s * imu_rate_hz))

    accels, wheels = _precompute_measurements(
        trajectory_name=trajectory_name,
        n_steps=n_steps,
        dt=dt,
        wheel_period=wheel_period,
        noise=noise,
        imu_rate_hz=imu_rate_hz,
        wheel_rate_hz=wheel_rate_hz,
    )

    trajectory = TRAJECTORY_CATALOG[trajectory_name]
    s0 = trajectory.state_at(0.0)
    fusion = make_fusion(mode, complementary_noise=COMPLEMENTARY_TUNING)
    fusion.initialise(s0.position, s0.velocity)

    # Warm up so JIT / page-cache effects don't bias the first few samples.
    for k in range(min(200, n_steps)):
        fusion.predict(accels[k], dt)
        if k % wheel_period == 0:
            fusion.update_wheel_velocity(wheels[k // wheel_period])
    # Reset the filter so the timed pass starts from a clean state.
    fusion = make_fusion(mode, complementary_noise=COMPLEMENTARY_TUNING)
    fusion.initialise(s0.position, s0.velocity)

    step_ns = np.empty(n_steps, dtype=np.int64)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        j = 0
        for k in range(n_steps):
            start = time.perf_counter_ns()
            fusion.predict(accels[k], dt)
            if k % wheel_period == 0:
                fusion.update_wheel_velocity(wheels[j])
                j += 1
            step_ns[k] = time.perf_counter_ns() - start
    finally:
        if gc_was_enabled:
            gc.enable()

    step_us = step_ns / 1000.0
    budget_us = dt * 1e6
    return {
        "imu_rate_hz": imu_rate_hz,
        "budget_us": budget_us,
        "mean_us": float(np.mean(step_us)),
        "median_us": float(np.median(step_us)),
        "p95_us": float(np.percentile(step_us, 95)),
        "p99_us": float(np.percentile(step_us, 99)),
        "max_us": float(np.max(step_us)),
        "missed": int(np.sum(step_us > budget_us)),
        "n_steps": float(n_steps),
    }


def wallclock_realtime(
    mode: FusionMode,
    *,
    imu_rate_hz: float = 1000.0,
    wheel_rate_hz: float = 50.0,
    duration_s: float = 2.0,
    noise: NoiseLevel = NoiseLevel.MEDIUM,
    trajectory_name: str = "sine",
) -> dict[str, float]:
    """Sustained real-time loop: run at wall-clock IMU rate, count misses.

    Each iteration busy-waits for the next deadline; if the previous
    iteration overran, the overshoot is recorded as a missed deadline.
    Busy-waiting (vs. ``time.sleep``) is used because Python's
    sleep resolution on Linux is ~1 ms, which is the same as the IMU
    period at 1 kHz.
    """
    dt_ns = int(round(1e9 / imu_rate_hz))
    dt = 1.0 / imu_rate_hz
    wheel_period = max(1, int(round(imu_rate_hz / wheel_rate_hz)))
    n_steps = int(round(duration_s * imu_rate_hz))

    accels, wheels = _precompute_measurements(
        trajectory_name=trajectory_name,
        n_steps=n_steps,
        dt=dt,
        wheel_period=wheel_period,
        noise=noise,
        imu_rate_hz=imu_rate_hz,
        wheel_rate_hz=wheel_rate_hz,
    )

    trajectory = TRAJECTORY_CATALOG[trajectory_name]
    s0 = trajectory.state_at(0.0)
    fusion = make_fusion(mode, complementary_noise=COMPLEMENTARY_TUNING)
    fusion.initialise(s0.position, s0.velocity)

    missed = 0
    max_overshoot_ns = 0
    j = 0
    start_ns = time.perf_counter_ns()
    deadline_ns = start_ns
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for k in range(n_steps):
            deadline_ns += dt_ns
            fusion.predict(accels[k], dt)
            if k % wheel_period == 0:
                fusion.update_wheel_velocity(wheels[j])
                j += 1
            now_ns = time.perf_counter_ns()
            if now_ns > deadline_ns:
                missed += 1
                if now_ns - deadline_ns > max_overshoot_ns:
                    max_overshoot_ns = now_ns - deadline_ns
            else:
                # Busy-wait until the deadline.
                while time.perf_counter_ns() < deadline_ns:
                    pass
    finally:
        if gc_was_enabled:
            gc.enable()
    end_ns = time.perf_counter_ns()

    elapsed_s = (end_ns - start_ns) / 1e9
    achieved_rate_hz = n_steps / elapsed_s
    return {
        "imu_rate_hz": imu_rate_hz,
        "duration_s": duration_s,
        "n_steps": n_steps,
        "elapsed_s": elapsed_s,
        "achieved_rate_hz": achieved_rate_hz,
        "missed": missed,
        "miss_rate_ppm": (missed / n_steps) * 1e6,
        "max_overshoot_us": max_overshoot_ns / 1000.0,
    }


def run_realtime_tests(report: Report, *, duration_s: float, imu_hz: float) -> None:
    print(f"[4/4] real-time test @ {imu_hz:.0f} Hz IMU")
    for mode in FusionMode:
        report.realtime_latency[mode] = latency_percentiles(
            mode, imu_rate_hz=imu_hz, duration_s=duration_s
        )
    for mode in FusionMode:
        report.wallclock_realtime[mode] = wallclock_realtime(
            mode, imu_rate_hz=imu_hz, duration_s=duration_s / 2
        )


# ─────────────────────────────────────────────── invariant checking ──


def check_invariants(report: Report) -> None:
    add = report.checks.append

    # 1. IDEAL noise → every mode is essentially perfect (≤ 5 cm).
    for mode in FusionMode:
        r = report.noise_sweep[(mode, NoiseLevel.IDEAL)]
        ok = r.rms_position_error_m < 0.05
        add(CheckResult(
            f"{mode.value} RMS ≤ 5 cm on sine+IDEAL",
            ok,
            f"got {r.rms_position_error_m:.4f} m",
        ))

    # 2. HIGH noise → KF beats imu_only by ≥ 5× on every trajectory.
    for traj_name in TRAJECTORY_CATALOG:
        # Re-use trajectory sweep results, which run at MEDIUM noise.  For
        # this check we need HIGH-noise numbers; pull them from the noise
        # sweep when the trajectory happens to be sine, otherwise run a
        # one-off sim so the check stays meaningful for every shape.
        if traj_name == "sine":
            kf = report.noise_sweep[(FusionMode.KF, NoiseLevel.HIGH)]
            imu = report.noise_sweep[(FusionMode.IMU_ONLY, NoiseLevel.HIGH)]
        else:
            kf = _run(
                f"check/HIGH/{traj_name}/kf",
                mode=FusionMode.KF,
                trajectory_name=traj_name,
                noise=NoiseLevel.HIGH,
                imu_rate_hz=1000.0,
                wheel_rate_hz=50.0,
                duration_s=4.0,
            )
            imu = _run(
                f"check/HIGH/{traj_name}/imu_only",
                mode=FusionMode.IMU_ONLY,
                trajectory_name=traj_name,
                noise=NoiseLevel.HIGH,
                imu_rate_hz=1000.0,
                wheel_rate_hz=50.0,
                duration_s=4.0,
            )
        ratio = (
            imu.rms_position_error_m / kf.rms_position_error_m
            if kf.rms_position_error_m > 0 else float("inf")
        )
        ok = ratio >= 5.0
        add(CheckResult(
            f"kf beats imu_only by ≥5× on {traj_name}+HIGH",
            ok,
            f"kf={kf.rms_position_error_m:.4f} m  "
            f"imu={imu.rms_position_error_m:.4f} m  ratio={ratio:.1f}×",
        ))

    # 2b. Tuned KF on HIGH noise beats the default-tuned KF.
    if report.tuned_kf_noise_sweep:
        default_kf_high = report.noise_sweep[
            (FusionMode.KF, NoiseLevel.HIGH)
        ].rms_position_error_m
        tuned_kf_high = report.tuned_kf_noise_sweep[
            NoiseLevel.HIGH
        ].rms_position_error_m
        improvement = (default_kf_high - tuned_kf_high) / default_kf_high
        ok = tuned_kf_high <= default_kf_high  # at least as good
        add(CheckResult(
            "tuned kf ≤ default kf on sine+HIGH",
            ok,
            f"tuned={tuned_kf_high:.4f} m  default={default_kf_high:.4f} m  "
            f"improvement={improvement * 100:+.1f}%",
        ))

    # 3. KF degrades monotonically with noise level on sine.
    rms_by_level = [
        report.noise_sweep[(FusionMode.KF, lvl)].rms_position_error_m
        for lvl in (NoiseLevel.IDEAL, NoiseLevel.LOW, NoiseLevel.MEDIUM, NoiseLevel.HIGH)
    ]
    ok = all(a <= b + 1e-6 for a, b in zip(rms_by_level, rms_by_level[1:]))
    add(CheckResult(
        "kf RMS is monotonic in noise level on sine",
        ok,
        " ≤ ".join(f"{x:.4f}" for x in rms_by_level),
    ))

    # 4. Real-time: p99 per-step latency < IMU deadline budget.
    for mode, stats in report.realtime_latency.items():
        ok = stats["p99_us"] < stats["budget_us"]
        add(CheckResult(
            f"{mode.value} p99 latency < IMU budget "
            f"({stats['imu_rate_hz']:.0f} Hz)",
            ok,
            f"p99={stats['p99_us']:.2f} µs  budget={stats['budget_us']:.0f} µs",
        ))

    # 5. Real-time: wall-clock loop misses < 1 % of deadlines.
    for mode, stats in report.wallclock_realtime.items():
        miss_rate = stats["missed"] / stats["n_steps"]
        ok = miss_rate < 0.01
        add(CheckResult(
            f"{mode.value} wall-clock miss rate < 1 %",
            ok,
            f"{stats['missed']}/{stats['n_steps']} "
            f"({miss_rate * 100:.3f} %)  "
            f"max_overshoot={stats['max_overshoot_us']:.1f} µs",
        ))


# ─────────────────────────────────────────────────────── plotting ──

# A4 portrait dimensions in inches (210 × 297 mm).
A4_PORTRAIT = (8.27, 11.69)

# Margins as fractions of A4 portrait for content placement.
_A4_LEFT = 0.10
_A4_RIGHT = 0.92


def _heatmap_axes_position(nrows: int, ncols: int) -> tuple[float, float, float, float]:
    """Return ``(left, bottom, width, height)`` for a heatmap on A4 portrait.

    Sized so that the cells stay roughly square and the heatmap occupies
    the upper-middle portion of the page, leaving headroom for a section
    title above and breathing room below.
    """
    page_w, page_h = A4_PORTRAIT
    cell_in = 0.85  # inches per cell (gives readable cell labels at 9 pt)
    heatmap_w = min(page_w * 0.78, cell_in * ncols + 1.6)  # +1.6" for ticks/colorbar
    heatmap_h = min(page_h * 0.40, cell_in * nrows + 1.2)  # +1.2" for ticks/title
    left = (page_w - heatmap_w) / 2 / page_w
    width = heatmap_w / page_w
    # Place top of axes ~85% up the page (below section title).
    top_frac = 0.84
    bottom = top_frac - (heatmap_h / page_h)
    height = heatmap_h / page_h
    return left, bottom, width, height


def _build_heatmap_figure(
    data: np.ndarray,
    *,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    value_label: str = "error [m]",
    log_scale: bool = True,
    cmap: str = "YlOrRd",
):
    """Build an A4-portrait heatmap page with the section title at the top."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    data = np.asarray(data, dtype=float)
    vmin = max(float(data.min()), 1e-4)
    vmax = max(float(data.max()), vmin * 1.01)
    norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else Normalize(vmin=vmin, vmax=vmax)

    nrows, ncols = data.shape
    fig = plt.figure(figsize=A4_PORTRAIT)
    fig.text(
        0.5, 0.94, title,
        ha="center", va="top", fontsize=14, fontweight="bold",
    )

    left, bottom, width, height = _heatmap_axes_position(nrows, ncols)
    ax = fig.add_axes([left, bottom, width, height])
    im = ax.imshow(data, aspect="equal", cmap=cmap, norm=norm)
    ax.set_xticks(range(ncols), col_labels, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(nrows), row_labels, fontsize=9)

    midpoint = float(np.sqrt(vmin * vmax)) if log_scale else 0.5 * (vmin + vmax)
    for i in range(nrows):
        for j in range(ncols):
            v = data[i, j]
            text = f"{v:.3f}" if v >= 0.001 else f"{v:.4f}"
            ax.text(
                j, i, text, ha="center", va="center",
                color="white" if v > midpoint else "black",
                fontsize=8,
            )

    # Colorbar on the right edge of the heatmap, matching height.
    cbar_ax = fig.add_axes([left + width + 0.015, bottom, 0.018, height])
    fig.colorbar(im, cax=cbar_ax, label=value_label)

    # Per-mode summary table directly under the heatmap (best / worst).
    summary_y = bottom - 0.04
    row_best = data.argmin(axis=1)
    row_worst = data.argmax(axis=1)
    fig.text(
        _A4_LEFT, summary_y, "Per-mode summary",
        fontsize=10, fontweight="bold", va="top",
    )
    summary_y -= 0.025
    for i in range(nrows):
        b = data[i, row_best[i]]
        w = data[i, row_worst[i]]
        ratio = (w / b) if b > 0 else float("inf")
        line = (
            f"{row_labels[i]:>14}  best={b:.4f} m @ {col_labels[row_best[i]]:<10}  "
            f"worst={w:.4f} m @ {col_labels[row_worst[i]]:<10}  ratio={ratio:.1f}×"
        )
        fig.text(
            _A4_LEFT, summary_y, line,
            fontsize=8, family="monospace", va="top",
        )
        summary_y -= 0.022

    return fig


def _build_realtime_figure(report: "Report"):
    """A4-portrait page: latency bars on top, wall-clock miss bars below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(FusionMode)
    labels = [m.value for m in modes]
    x = np.arange(len(modes))

    means = [report.realtime_latency[m]["mean_us"] for m in modes]
    p95s = [report.realtime_latency[m]["p95_us"] for m in modes]
    p99s = [report.realtime_latency[m]["p99_us"] for m in modes]
    maxs = [report.realtime_latency[m]["max_us"] for m in modes]
    budget_us = report.realtime_latency[modes[0]]["budget_us"]

    fig = plt.figure(figsize=A4_PORTRAIT)
    fig.text(
        0.5, 0.94, "4. Real-time performance",
        ha="center", va="top", fontsize=14, fontweight="bold",
    )

    ax = fig.add_axes([_A4_LEFT, 0.50, _A4_RIGHT - _A4_LEFT, 0.36])
    width = 0.2
    ax.bar(x - 1.5 * width, means, width, label="mean", color="#1f77b4")
    ax.bar(x - 0.5 * width, p95s,  width, label="p95",  color="#2ca02c")
    ax.bar(x + 0.5 * width, p99s,  width, label="p99",  color="#ff7f0e")
    ax.bar(x + 1.5 * width, maxs,  width, label="max",  color="#d62728")
    ax.axhline(
        budget_us, color="k", lw=1.2, ls="--",
        label=f"IMU budget ({budget_us:.0f} µs)",
    )
    ax.set_yscale("log")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("per-step latency [µs] (log)")
    ax.set_title("Per-step latency (tight loop, GC off)", fontsize=11)
    ax.legend(fontsize=8, ncols=2, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    ax2 = fig.add_axes([_A4_LEFT, 0.10, _A4_RIGHT - _A4_LEFT, 0.30])
    miss_pct = []
    for m in modes:
        wc = report.wallclock_realtime[m]
        n = max(1, int(wc["n_steps"]))
        miss_pct.append(100.0 * wc["missed"] / n)
    bars = ax2.bar(x, miss_pct, color="#9467bd")
    ax2.set_xticks(x, labels, rotation=15, ha="right")
    ax2.set_ylabel("missed deadlines [%]")
    ax2.set_title(
        f"Wall-clock loop @ {report.wallclock_realtime[modes[0]]['imu_rate_hz']:.0f} Hz "
        f"(lower is better)", fontsize=11,
    )
    if all(p == 0 for p in miss_pct):
        ax2.set_ylim(0, 0.5)
    for bar, p in zip(bars, miss_pct):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{p:.3f}%",
            ha="center", va="bottom", fontsize=9,
        )
    ax2.grid(True, axis="y", alpha=0.3)
    return fig


def _build_title_page(report: "Report", args, n_sims: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=A4_PORTRAIT)
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(
        0.5, 0.94, "Sensor Fusion Test Report",
        ha="center", fontsize=22, fontweight="bold", transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.89,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        ha="center", fontsize=10, color="#555", transform=ax.transAxes,
    )

    lines = [
        ("Fusion modes",        ", ".join(m.value for m in FusionMode)),
        ("Trajectories",        ", ".join(TRAJECTORY_CATALOG)),
        ("Noise levels",        ", ".join(lvl.value for lvl in NoiseLevel)),
        ("Duration / scenario", f"{args.duration:.1f} s"),
        ("Baseline IMU rate",   f"{args.imu_hz:.0f} Hz"),
        ("Baseline wheel rate", f"{args.wheel_hz:.0f} Hz"),
        ("IMU rate sweep",      ", ".join(f"{int(r)} Hz" for r in args.imu_rates)),
        ("Wheel rate sweep",    ", ".join(f"{int(r)} Hz" for r in args.wheel_rates)),
        ("Total simulations",   f"{n_sims}"),
    ]
    y = 0.78
    for label, value in lines:
        ax.text(0.10, y, f"{label}:", fontweight="bold", fontsize=11,
                transform=ax.transAxes)
        ax.text(0.40, y, value, fontsize=11, transform=ax.transAxes)
        y -= 0.038

    ax.text(
        0.5, 0.36,
        "Sections:\n"
        "  1. Noise-level sweep — mode × noise on sine\n"
        "  2. Trajectory sweep — mode × trajectory at MEDIUM noise\n"
        "  3. Frequency sweep — mode × IMU rate / wheel rate\n"
        "  4. Real-time performance — per-step latency + wall-clock miss rate\n"
        "  5. Invariant checks — pass/fail summary",
        fontsize=9.5, family="monospace", color="#333",
        ha="center", va="top", transform=ax.transAxes,
    )

    passed = sum(1 for c in report.checks if c.ok)
    total = len(report.checks)
    all_pass = passed == total and total > 0
    color = "#0a7f3f" if all_pass else "#b00020"
    ax.text(
        0.5, 0.15,
        f"{passed}/{total} invariant checks passed",
        ha="center", fontsize=15, fontweight="bold", color=color,
        transform=ax.transAxes,
        bbox=dict(facecolor="white", edgecolor=color, lw=2,
                  boxstyle="round,pad=0.6"),
    )
    return fig


def _build_checks_page(report: "Report"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=A4_PORTRAIT)
    ax = fig.add_subplot(111)
    ax.axis("off")

    passed = sum(1 for c in report.checks if c.ok)
    total = len(report.checks)
    color = "#0a7f3f" if passed == total else "#b00020"

    ax.text(0.5, 0.96, "5. Invariant Checks", ha="center",
            fontsize=16, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.92, f"{passed}/{total} passed", ha="center",
            fontsize=11, color=color, transform=ax.transAxes)

    y = 0.87
    line_height = 0.039
    for c in report.checks:
        marker = "PASS" if c.ok else "FAIL"
        mc = "#0a7f3f" if c.ok else "#b00020"
        ax.text(0.05, y, f"[{marker}]", color=mc, fontsize=9,
                fontweight="bold", family="monospace", transform=ax.transAxes)
        ax.text(0.14, y, c.name, fontsize=9, transform=ax.transAxes)
        ax.text(0.14, y - 0.016, c.detail, fontsize=7.5, color="#555",
                transform=ax.transAxes)
        y -= line_height
        if y < 0.04:
            ax.text(0.5, 0.02,
                    f"(... {total - report.checks.index(c)} more checks truncated)",
                    ha="center", fontsize=8, color="#999",
                    transform=ax.transAxes)
            break
    return fig


def _collect_sweep_arrays(report: Report) -> dict[str, dict]:
    """Pre-compute the per-section data arrays consumed by both the PNG
    saver and the PDF builder, so they stay in lock-step."""
    sections = {}
    sections["noise_rms"] = dict(
        data=np.array([
            [report.noise_sweep[(m, lvl)].rms_position_error_m for lvl in NoiseLevel]
            for m in FusionMode
        ]),
        row_labels=[m.value for m in FusionMode],
        col_labels=[lvl.value for lvl in NoiseLevel],
        title="1. Position RMS error — mode × noise (sine trajectory)",
    )
    sections["noise_final"] = dict(
        data=np.array([
            [report.noise_sweep[(m, lvl)].final_position_error_m for lvl in NoiseLevel]
            for m in FusionMode
        ]),
        row_labels=[m.value for m in FusionMode],
        col_labels=[lvl.value for lvl in NoiseLevel],
        title="1. Final position error — mode × noise (sine trajectory)",
    )
    # KF tuning comparison: untuned KF vs KF tuned per noise level vs
    # complementary filter, all on sine.
    if report.tuned_kf_noise_sweep:
        comp_row = [
            report.noise_sweep[(FusionMode.COMPLEMENTARY, lvl)].rms_position_error_m
            for lvl in NoiseLevel
        ]
        kf_untuned_row = [
            report.noise_sweep[(FusionMode.KF, lvl)].rms_position_error_m
            for lvl in NoiseLevel
        ]
        kf_tuned_row = [
            report.tuned_kf_noise_sweep[lvl].rms_position_error_m
            for lvl in NoiseLevel
        ]
        sections["kf_tuning"] = dict(
            data=np.array([kf_untuned_row, kf_tuned_row, comp_row]),
            row_labels=["kf (default)", "kf (tuned)", "complementary"],
            col_labels=[lvl.value for lvl in NoiseLevel],
            title="1b. KF tuning effect — position RMS (sine trajectory)",
        )
    traj_names = list(TRAJECTORY_CATALOG)
    sections["trajectory"] = dict(
        data=np.array([
            [report.trajectory_sweep[(m, t)].rms_position_error_m for t in traj_names]
            for m in FusionMode
        ]),
        row_labels=[m.value for m in FusionMode],
        col_labels=traj_names,
        title="2. Position RMS error — mode × trajectory (MEDIUM noise)",
    )
    imu_rates = sorted({k[1] for k in report.imu_rate_sweep})
    sections["imu_rate"] = dict(
        data=np.array([
            [report.imu_rate_sweep[(m, hz)].rms_position_error_m for hz in imu_rates]
            for m in FusionMode
        ]),
        row_labels=[m.value for m in FusionMode],
        col_labels=[f"{int(hz)} Hz" for hz in imu_rates],
        title="3. Position RMS error — mode × IMU rate (sine, MEDIUM)",
    )
    wheel_rates = sorted({k[1] for k in report.wheel_rate_sweep})
    sections["wheel_rate"] = dict(
        data=np.array([
            [report.wheel_rate_sweep[(m, hz)].rms_position_error_m for hz in wheel_rates]
            for m in FusionMode
        ]),
        row_labels=[m.value for m in FusionMode],
        col_labels=[f"{int(hz)} Hz" for hz in wheel_rates],
        title="3. Position RMS error — mode × wheel-odom rate (sine, MEDIUM)",
    )
    return sections


def render_artifacts(
    report: Report,
    args,
    output_dir: Path,
    *,
    write_png: bool = True,
    pdf_path: Path | None = None,
) -> dict[str, str]:
    """Build every figure once; emit PNGs and/or a multipage PDF.

    Returns ``{section_key: png_filename}`` for the markdown report's
    image-reference inlining (empty if ``write_png=False``).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_dir.mkdir(parents=True, exist_ok=True)
    sections = _collect_sweep_arrays(report)
    n_sims = (
        len(report.noise_sweep) + len(report.trajectory_sweep)
        + len(report.imu_rate_sweep) + len(report.wheel_rate_sweep)
    )

    pdf = PdfPages(pdf_path) if pdf_path is not None else None
    png_paths: dict[str, str] = {}
    try:
        # Title page (PDF only) — full A4 page.
        if pdf is not None:
            fig = _build_title_page(report, args, n_sims)
            pdf.savefig(fig)
            plt.close(fig)

        # Sections 1-3: heatmaps.
        for key, spec in sections.items():
            fig = _build_heatmap_figure(
                spec["data"],
                row_labels=spec["row_labels"],
                col_labels=spec["col_labels"],
                title=spec["title"],
            )
            if write_png:
                png = output_dir / f"test_sensor_fusion_{key}.png"
                # `bbox_inches='tight'` trims the A4 whitespace for compact
                # PNGs that look fine when embedded in the markdown report.
                fig.savefig(png, dpi=150, bbox_inches="tight")
                png_paths[key] = png.name
            if pdf is not None:
                # No bbox_inches: the PDF keeps the full A4 page geometry.
                pdf.savefig(fig)
            plt.close(fig)

        # Section 4: real-time bars.
        fig = _build_realtime_figure(report)
        if write_png:
            png = output_dir / "test_sensor_fusion_realtime.png"
            fig.savefig(png, dpi=150, bbox_inches="tight")
            png_paths["realtime"] = png.name
        if pdf is not None:
            pdf.savefig(fig)
        plt.close(fig)

        # Section 5: invariant checks (PDF only — markdown lists them inline).
        if pdf is not None:
            fig = _build_checks_page(report)
            pdf.savefig(fig)
            plt.close(fig)

            d = pdf.infodict()
            d["Title"] = "Sensor Fusion Test Report"
            d["Author"] = "test_sensor_fusion.py"
            d["Subject"] = "Comprehensive fusion-mode benchmark report"
            d["CreationDate"] = datetime.now()
    finally:
        if pdf is not None:
            pdf.close()

    return png_paths


# Back-compat shim so any external caller still using `render_plots` keeps
# working; new code should call `render_artifacts` directly.
def render_plots(report: Report, output_dir: Path) -> dict[str, str]:
    return render_artifacts(
        report, _MinimalArgs(), output_dir, write_png=True, pdf_path=None
    )


class _MinimalArgs:
    """Stand-in for the argparse Namespace fields read by the title page."""

    duration = 0.0
    imu_hz = 0.0
    wheel_hz = 0.0
    imu_rates: list[float] = []
    wheel_rates: list[float] = []


# ───────────────────────────────────────────────────────── reports ──


def _heatmap_table(
    title: str,
    *,
    row_label: str,
    col_label: str,
    rows: list,
    cols: list,
    cell: callable,
    fmt: str = "{:.4f}",
) -> str:
    def _label(x) -> str:
        return x.value if hasattr(x, "value") else str(x)

    lines = [f"### {title}", ""]
    header = (
        f"| {row_label} \\ {col_label} | "
        + " | ".join(_label(c) for c in cols)
        + " |"
    )
    sep = "|" + "---|" * (len(cols) + 1)
    lines.append(header)
    lines.append(sep)
    for row in rows:
        vals = [fmt.format(cell(row, c)) for c in cols]
        lines.append(f"| {_label(row)} | " + " | ".join(vals) + " |")
    lines.append("")
    return "\n".join(lines)


def render_report(
    report: Report,
    args: argparse.Namespace,
    plot_paths: dict[str, str] | None = None,
) -> str:
    plot_paths = plot_paths or {}

    def img(key: str, alt: str) -> str:
        if key in plot_paths:
            return f"![{alt}]({plot_paths[key]})\n"
        return ""

    parts: list[str] = []
    parts.append("# Sensor Fusion Test Report")
    parts.append("")
    parts.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    parts.append("")
    parts.append("Tested fusion modes: " + ", ".join(
        f"`{m.value}`" for m in FusionMode
    ))
    parts.append("")

    # 1. Noise sweep
    parts.append("## 1. Noise-level sweep (sine trajectory)")
    parts.append("")
    parts.append(
        f"Duration {args.duration:.1f} s, IMU {args.imu_hz:.0f} Hz, "
        f"wheel {args.wheel_hz:.0f} Hz, seed 7."
    )
    parts.append("")
    parts.append(_heatmap_table(
        "Position RMS error [m]",
        row_label="mode", col_label="noise",
        rows=list(FusionMode), cols=list(NoiseLevel),
        cell=lambda m, lvl: report.noise_sweep[(m, lvl)].rms_position_error_m,
    ))
    parts.append(img("noise_rms", "Noise sweep RMS heatmap"))
    parts.append(_heatmap_table(
        "Final position error [m]",
        row_label="mode", col_label="noise",
        rows=list(FusionMode), cols=list(NoiseLevel),
        cell=lambda m, lvl: report.noise_sweep[(m, lvl)].final_position_error_m,
    ))
    parts.append(img("noise_final", "Noise sweep final-error heatmap"))

    # 1b. KF tuning effect
    if report.tuned_kf_noise_sweep:
        parts.append("## 1b. KF tuning effect")
        parts.append("")
        parts.append(
            "The default `KalmanFusion2D` uses `FusionNoise(0.08, 0.01, 0.05)` "
            "regardless of the actual sensor noise — a hand-picked guess.  "
            "Below the KF is re-run with `FusionNoise` matched to each "
            "`NoiseLevel`'s real `ImuConfig` / `WheelOdomConfig`.  The tuned "
            "KF should at least match — and on noisy presets meaningfully "
            "beat — the constant-gain complementary filter."
        )
        parts.append("")
        rows_kt = ["kf (default)", "kf (tuned)", "complementary"]

        def _kt(row: str, lvl: NoiseLevel) -> float:
            if row == "kf (default)":
                return report.noise_sweep[(FusionMode.KF, lvl)].rms_position_error_m
            if row == "kf (tuned)":
                return report.tuned_kf_noise_sweep[lvl].rms_position_error_m
            return report.noise_sweep[
                (FusionMode.COMPLEMENTARY, lvl)
            ].rms_position_error_m

        parts.append(_heatmap_table(
            "Position RMS error [m]",
            row_label="variant", col_label="noise",
            rows=rows_kt, cols=list(NoiseLevel),
            cell=_kt,
        ))
        parts.append(img("kf_tuning", "KF tuning comparison heatmap"))

    # 2. Trajectory sweep
    parts.append("## 2. Trajectory sweep (MEDIUM noise)")
    parts.append("")
    parts.append(_heatmap_table(
        "Position RMS error [m]",
        row_label="mode", col_label="trajectory",
        rows=list(FusionMode), cols=list(TRAJECTORY_CATALOG),
        cell=lambda m, t: report.trajectory_sweep[(m, t)].rms_position_error_m,
    ))
    parts.append(img("trajectory", "Trajectory sweep heatmap"))

    # 3. Frequency sweep
    parts.append("## 3. Frequency sweep (sine, MEDIUM noise)")
    parts.append("")
    imu_rates = sorted({k[1] for k in report.imu_rate_sweep})
    parts.append(_heatmap_table(
        f"Position RMS error [m] vs IMU rate (wheel = {args.wheel_hz:.0f} Hz)",
        row_label="mode", col_label="IMU Hz",
        rows=list(FusionMode), cols=imu_rates,
        cell=lambda m, hz: report.imu_rate_sweep[(m, hz)].rms_position_error_m,
    ))
    parts.append(img("imu_rate", "IMU-rate sweep heatmap"))
    wheel_rates = sorted({k[1] for k in report.wheel_rate_sweep})
    parts.append(_heatmap_table(
        f"Position RMS error [m] vs wheel-odom rate (IMU = {args.imu_hz:.0f} Hz)",
        row_label="mode", col_label="wheel Hz",
        rows=list(FusionMode), cols=wheel_rates,
        cell=lambda m, hz: report.wheel_rate_sweep[(m, hz)].rms_position_error_m,
    ))
    parts.append(img("wheel_rate", "Wheel-rate sweep heatmap"))

    # 4. Real-time
    parts.append("## 4. Real-time performance")
    parts.append("")
    parts.append(img("realtime", "Real-time latency and miss-rate bars"))
    parts.append("### Per-step latency (tight loop, GC disabled)")
    parts.append("")
    parts.append("| mode | mean µs | p95 µs | p99 µs | max µs | budget µs | misses |")
    parts.append("|---|---|---|---|---|---|---|")
    for mode in FusionMode:
        s = report.realtime_latency[mode]
        parts.append(
            f"| {mode.value} | {s['mean_us']:.2f} | {s['p95_us']:.2f} | "
            f"{s['p99_us']:.2f} | {s['max_us']:.2f} | "
            f"{s['budget_us']:.0f} | {int(s['missed'])} |"
        )
    parts.append("")
    parts.append("### Wall-clock real-time loop (busy-wait pacing)")
    parts.append("")
    parts.append("| mode | duration s | achieved Hz | misses / N | miss rate ppm | max overshoot µs |")
    parts.append("|---|---|---|---|---|---|")
    for mode in FusionMode:
        s = report.wallclock_realtime[mode]
        parts.append(
            f"| {mode.value} | {s['duration_s']:.2f} | "
            f"{s['achieved_rate_hz']:.1f} | "
            f"{int(s['missed'])}/{int(s['n_steps'])} | "
            f"{s['miss_rate_ppm']:.0f} | "
            f"{s['max_overshoot_us']:.1f} |"
        )
    parts.append("")

    # 5. Invariant checks
    parts.append("## 5. Invariant checks")
    parts.append("")
    passed = sum(1 for c in report.checks if c.ok)
    parts.append(f"**{passed}/{len(report.checks)} checks passed.**")
    parts.append("")
    parts.append("| status | check | detail |")
    parts.append("|---|---|---|")
    for c in report.checks:
        parts.append(
            f"| {'PASS' if c.ok else 'FAIL'} | {c.name} | {c.detail} |"
        )
    parts.append("")
    return "\n".join(parts)


def write_csv(report: Report, path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "section", "mode", "trajectory", "noise",
            "imu_rate_hz", "wheel_rate_hz",
            "rms_m", "final_m", "us_per_step",
        ])
        for (m, lvl), r in report.noise_sweep.items():
            w.writerow(["noise", m.value, "sine", lvl.value, "", "",
                        r.rms_position_error_m, r.final_position_error_m,
                        r.mean_update_time_us])
        for (m, t), r in report.trajectory_sweep.items():
            w.writerow(["trajectory", m.value, t, "medium", "", "",
                        r.rms_position_error_m, r.final_position_error_m,
                        r.mean_update_time_us])
        for (m, hz), r in report.imu_rate_sweep.items():
            w.writerow(["imu_rate", m.value, "sine", "medium", hz, "",
                        r.rms_position_error_m, r.final_position_error_m,
                        r.mean_update_time_us])
        for (m, hz), r in report.wheel_rate_sweep.items():
            w.writerow(["wheel_rate", m.value, "sine", "medium", "", hz,
                        r.rms_position_error_m, r.final_position_error_m,
                        r.mean_update_time_us])


def print_summary(report: Report) -> None:
    print()
    print("=" * 72)
    print(" SUMMARY")
    print("=" * 72)
    print()

    # Noise table on sine
    print(" RMS [m] on sine, by mode and noise level:")
    header = "  " + " " * 16 + "  ".join(f"{lvl.value:>9}" for lvl in NoiseLevel)
    print(header)
    for mode in FusionMode:
        row = "  " + f"{mode.value:<14}"
        for lvl in NoiseLevel:
            r = report.noise_sweep[(mode, lvl)]
            row += f"  {r.rms_position_error_m:9.4f}"
        print(row)
    print()

    # KF tuning comparison
    if report.tuned_kf_noise_sweep:
        print(" KF tuning effect on sine (RMS [m]):")
        header = "  " + " " * 16 + "  ".join(f"{lvl.value:>9}" for lvl in NoiseLevel)
        print(header)
        for label, getter in [
            ("kf (default)",
             lambda lvl: report.noise_sweep[(FusionMode.KF, lvl)].rms_position_error_m),
            ("kf (tuned)",
             lambda lvl: report.tuned_kf_noise_sweep[lvl].rms_position_error_m),
            ("complementary",
             lambda lvl: report.noise_sweep[
                 (FusionMode.COMPLEMENTARY, lvl)
             ].rms_position_error_m),
        ]:
            row = "  " + f"{label:<14}"
            for lvl in NoiseLevel:
                row += f"  {getter(lvl):9.4f}"
            print(row)
        print()

    # Real-time table
    print(" Real-time: tight-loop p99 + wall-clock miss rate:")
    print("  " + f"{'mode':<14}{'p99 µs':>10}{'budget µs':>12}{'misses':>10}{'overshoot µs':>15}")
    for mode in FusionMode:
        lat = report.realtime_latency[mode]
        wc = report.wallclock_realtime[mode]
        print("  " +
              f"{mode.value:<14}{lat['p99_us']:>10.2f}{lat['budget_us']:>12.0f}"
              f"{wc['missed']:>10d}{wc['max_overshoot_us']:>15.1f}")
    print()

    # Invariant checks
    passed = sum(1 for c in report.checks if c.ok)
    total = len(report.checks)
    print(f" Invariant checks: {passed}/{total} passed")
    for c in report.checks:
        marker = " PASS" if c.ok else " FAIL"
        print(f"  {marker}  {c.name}  ({c.detail})")
    print()


# ───────────────────────────────────────────────────────────── main ──


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT,
        help=f"Markdown report path (default: {DEFAULT_REPORT.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--pdf", type=Path, default=DEFAULT_PDF,
        help=f"Multipage PDF report path (default: {DEFAULT_PDF.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV,
        help=f"CSV results path (default: {DEFAULT_CSV.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--duration", type=float, default=8.0,
        help="Per-scenario simulated duration in seconds (default: 8).",
    )
    parser.add_argument(
        "--imu-hz", type=float, default=1000.0,
        help="Baseline IMU rate (default: 1000).",
    )
    parser.add_argument(
        "--wheel-hz", type=float, default=50.0,
        help="Baseline wheel-odometry rate (default: 50).",
    )
    parser.add_argument(
        "--imu-rates", type=float, nargs="+",
        default=[250.0, 500.0, 1000.0, 2000.0],
        help="IMU rates to sweep (default: 250 500 1000 2000).",
    )
    parser.add_argument(
        "--wheel-rates", type=float, nargs="+",
        default=[10.0, 25.0, 50.0, 100.0],
        help="Wheel rates to sweep (default: 10 25 50 100).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Halve duration and trim frequency sweeps for a fast smoke run.",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating standalone PNG heatmaps next to the report.",
    )
    parser.add_argument(
        "--no-pdf", action="store_true",
        help="Skip the multipage PDF report.",
    )
    parser.add_argument(
        "--no-md", action="store_true",
        help="Skip the Markdown report (PDF and CSV still written).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.quick:
        args.duration = max(2.0, args.duration / 2)
        args.imu_rates = [500.0, 1000.0]
        args.wheel_rates = [25.0, 50.0]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Running sensor-fusion test suite "
        f"(duration {args.duration:.1f} s, base IMU {args.imu_hz:.0f} Hz, "
        f"base wheel {args.wheel_hz:.0f} Hz)"
    )
    start = time.perf_counter()
    report = Report()
    run_noise_sweep(
        report, duration_s=args.duration,
        imu_hz=args.imu_hz, wheel_hz=args.wheel_hz,
    )
    run_kf_tuning_sweep(
        report, duration_s=args.duration,
        imu_hz=args.imu_hz, wheel_hz=args.wheel_hz,
    )
    run_trajectory_sweep(
        report, duration_s=args.duration,
        imu_hz=args.imu_hz, wheel_hz=args.wheel_hz,
    )
    run_frequency_sweep(
        report, duration_s=args.duration,
        imu_rates=args.imu_rates, wheel_rates=args.wheel_rates,
        base_imu_hz=args.imu_hz, base_wheel_hz=args.wheel_hz,
    )
    run_realtime_tests(
        report, duration_s=args.duration, imu_hz=args.imu_hz,
    )
    check_invariants(report)
    elapsed = time.perf_counter() - start
    print(f"Done in {elapsed:.2f} s; writing report ...")

    pdf_path = None if args.no_pdf else args.pdf
    if pdf_path is not None:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

    plot_paths = render_artifacts(
        report, args, args.report.parent,
        write_png=not args.no_plots,
        pdf_path=pdf_path,
    )

    if not args.no_md:
        args.report.write_text(render_report(report, args, plot_paths))
    write_csv(report, args.csv)
    print_summary(report)
    if pdf_path is not None:
        print(f"PDF:     {pdf_path}")
    if not args.no_md:
        print(f"Report:  {args.report}")
    print(f"CSV:     {args.csv}")
    if plot_paths:
        print(f"Plots:   {args.report.parent}/test_sensor_fusion_*.png "
              f"({len(plot_paths)} files)")

    return 0 if all(c.ok for c in report.checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
