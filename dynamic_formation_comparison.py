"""Scene-3-only geometry and center-point ablation plotting helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STAGES = (
    (0, 140, "巡航楔形", "#edf4fb"),
    (140, 260, "第一次转换", "#fff6e8"),
    (260, 340, "宽域搜索", "#eef8f2"),
    (340, 460, "第二次转换", "#fff6e8"),
    (460, 600, "防护队形", "#f3f0fa"),
)


def _set_academic_style():
    """Use a compact, conventional matplotlib style suitable for PPT figures."""
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS"],
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
    })


def _decorate_stages(ax, *, shade=True):
    for start, end, label, color in STAGES:
        if shade:
            ax.axvspan(start, end, color=color, alpha=0.62, zorder=0)
        ax.text(
            (start + end) / 2.0,
            1.025,
            f"{label}\n{start}–{end} s",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#222222",
        )
    for boundary in (140, 260, 340, 460):
        ax.axvline(
            boundary,
            color="#9a9a9a",
            linewidth=0.75,
            linestyle="--",
            alpha=0.8,
            zorder=1,
        )


def build_center_only_friendly_series(friendly_series):
    """Collapse each four-aircraft state to one moving formation-center point."""
    center_series = []

    for friendlies in friendly_series:
        positions = np.array(
            [[f["X"], f["Y"], f["Z"]] for f in friendlies], dtype=float
        )
        velocities = np.array(
            [[f["VX"], f["VY"], f["VZ"]] for f in friendlies], dtype=float
        )
        center = np.mean(positions, axis=0)
        velocity = np.mean(velocities, axis=0)
        formation_type = (
            friendlies[0].get("FormationType", "Unknown")
            if friendlies
            else "Unknown"
        )

        center_series.append([{
            "Aircraft_ID": 0,
            "Name": "Formation_Center_Point",
            "Role": "CenterPoint",
            "AircraftType": "formation_center",
            "Value": 1.0,
            "Vulnerability": 1.0,
            "BaselineVulnerability": 1.0,
            "Maneuverability": 1.0,
            "BaselineManeuverability": 1.0,
            "SensorRange": 230.0,
            "WeaponRange": 140.0,
            "ECMCapability": 0.5,
            "CapabilityState": "Healthy",
            "DamageLevel": 0.0,
            "CapabilityEvent": "",
            "FormationType": formation_type,
            "FormationMode": "center_only_baseline",
            "FormationModeDescription": "Formation-center point baseline",
            "Type": "FriendlyFighter",
            "X": float(center[0]),
            "Y": float(center[1]),
            "Z": float(center[2]),
            "VX": float(velocity[0]),
            "VY": float(velocity[1]),
            "VZ": float(velocity[2]),
        }])

    return center_series


def save_formation_geometry_timeseries(friendly_series, save_dir):
    """Plot actual lateral width and 3-D formation radius over time."""
    if not friendly_series:
        raise ValueError("friendly_series is empty.")

    times = np.arange(len(friendly_series), dtype=float)
    lateral_width = []
    formation_radius = []

    for friendlies in friendly_series:
        positions = np.array(
            [[f["X"], f["Y"], f["Z"]] for f in friendlies], dtype=float
        )
        center = np.mean(positions, axis=0)
        lateral_width.append(float(np.ptp(positions[:, 1])))
        formation_radius.append(
            float(np.max(np.linalg.norm(positions - center, axis=1)))
        )

    lateral_width = np.asarray(lateral_width)
    formation_radius = np.asarray(formation_radius)
    marker_step = max(1, int(round(len(times) / 30)))

    _set_academic_style()
    fig, ax_left = plt.subplots(figsize=(9.0, 3.25), dpi=220)
    fig.subplots_adjust(left=0.10, right=0.89, bottom=0.20, top=0.76)
    ax_right = ax_left.twinx()

    _decorate_stages(ax_left, shade=False)
    line_width, marker_size = 1.35, 2.5
    width_line = ax_left.plot(
        times,
        lateral_width,
        color="#214be0",
        linewidth=line_width,
        marker="o",
        markersize=marker_size,
        markevery=marker_step,
        label=r"横向宽度 $W_f(t)$",
        zorder=3,
    )[0]
    radius_line = ax_right.plot(
        times,
        formation_radius,
        color="#14813b",
        linewidth=line_width,
        linestyle="--",
        marker="s",
        markersize=marker_size,
        markevery=marker_step,
        label=r"编队半径 $R_f(t)$",
        zorder=3,
    )[0]

    ax_left.set_xlim(0, 600)
    ax_left.set_ylim(0, 200)
    ax_right.set_ylim(0, 120)
    ax_left.set_xticks(np.arange(0, 601, 100))
    ax_left.set_yticks(np.arange(0, 201, 50))
    ax_right.set_yticks(np.arange(0, 121, 30))
    ax_left.set_xlabel("时间 $t$ (s)")
    ax_left.set_ylabel(r"横向宽度 $W_f(t)$ (km)", color="#214be0")
    ax_right.set_ylabel(r"编队半径 $R_f(t)$ (km)", color="#14813b")
    ax_left.tick_params(axis="y", colors="#214be0")
    ax_right.tick_params(axis="y", colors="#14813b")
    ax_left.grid(True, color="#d8d8d8", linewidth=0.65, alpha=0.75)
    ax_left.legend(
        [width_line, radius_line],
        [width_line.get_label(), radius_line.get_label()],
        loc="upper left",
        frameon=False,
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    png_path = save_dir / "Figure_Formation_Geometry_TimeSeries.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(png_path)


def save_structure_vs_center_comparison(
    structure_records,
    center_records,
    save_dir,
):
    """Plot scene-3 structure-aware and center-only threat scores."""
    if len(structure_records) != len(center_records):
        raise ValueError("Structure-aware and center-only record lengths differ.")

    times = np.array([record["time"] for record in structure_records], dtype=float)
    structure_scores = np.stack([
        np.asarray(record["scores"], dtype=float)
        for record in structure_records
    ])
    center_scores = np.stack([
        np.asarray(record["scores"], dtype=float)
        for record in center_records
    ])
    if structure_scores.shape != center_scores.shape:
        raise ValueError("Structure-aware and center-only score shapes differ.")

    score_delta = structure_scores - center_scores
    transition_mask = times >= 140.0
    sensitivity = np.mean(np.abs(score_delta[transition_mask]), axis=0)
    representative_idx = int(np.argmax(sensitivity))
    representative_id = representative_idx + 1
    target_delta = score_delta[:, representative_idx]
    max_delta_idx = int(np.argmax(np.abs(target_delta)))
    max_abs_delta = float(abs(target_delta[max_delta_idx]))
    marker_step = max(1, int(round(len(times) / 30)))

    _set_academic_style()
    fig, ax = plt.subplots(figsize=(9.0, 3.75), dpi=220)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.27, top=0.74)

    _decorate_stages(ax, shade=True)
    ax.plot(
        times,
        structure_scores[:, representative_idx],
        color="#df2b24",
        linewidth=1.45,
        marker="o",
        markersize=2.7,
        markevery=marker_step,
        label="编队结构感知方法",
        zorder=3,
    )
    ax.plot(
        times,
        center_scores[:, representative_idx],
        color="#1d4ed8",
        linewidth=1.35,
        linestyle="--",
        marker="s",
        markersize=2.4,
        markevery=marker_step,
        label="编队中心点基线",
        zorder=3,
    )

    ax.set_xlim(0, 600)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(0, 601, 100))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("时间 $t$ (s)")
    ax.set_ylabel(r"综合威胁度 $S_j^{total}$")
    ax.grid(True, color="#d8d8d8", linewidth=0.65, alpha=0.75)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title(
        f"代表目标 T{representative_id} 综合威胁度时序对比",
        fontsize=10,
        pad=34,
    )
    fig.text(
        0.10,
        0.075,
        "• 在敌方目标、数据干扰与动态队形一致条件下，仅改变编队结构建模方式，"
        "代表目标威胁度随队形变化产生差异。",
        ha="left",
        va="center",
        fontsize=8.8,
        color="#123d88",
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    png_path = save_dir / "Figure_Structure_vs_Center_Threat_Comparison.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "png_path": str(png_path),
        "representative_target": f"T{representative_id}",
        "max_abs_delta": max_abs_delta,
        "max_delta_target": f"T{representative_id}",
        "max_delta_time": int(times[max_delta_idx]),
    }
