# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ds_iff_2026 import IFFConfig, LowAltitudeIFFRecognizer, RouteProfile
from iff_scene import generate_iff_time_series, state_to_observation


LABEL_COLORS = {
    "FR": "#1f77b4",
    "AC": "#2ca02c",
    "ST": "#ff7f0e",
    "FO": "#d62728",
    "Theta": "#7f7f7f",
    None: "#7f7f7f",
}

LABEL_MARKERS = {
    "FR": "o",
    "AC": "^",
    "ST": "s",
    "FO": "X",
    None: ".",
}

TAIL_LEN_DEFAULT = 50


# 生成 IFF 可视化所需的目标、编队和识别结果时序。
def build_iff_visual_records(num_steps=601, window_size=3):
    config = IFFConfig(
        route=RouteProfile(height_m=1000.0, speed_kmh=600.0, heading_deg=290.0),
        window_size=window_size,
    )
    recognizer = LowAltitudeIFFRecognizer(config)
    target_series, friendly_series = generate_iff_time_series(num_steps=num_steps, config=config)
    windows = defaultdict(lambda: deque(maxlen=config.window_size))

    records = []
    for t_idx, states in enumerate(target_series):
        targets = []
        for state in states:
            obs = state_to_observation(state)
            windows[obs.target_id].append(obs)
            result = recognizer.identify(list(windows[obs.target_id]))

            targets.append(
                {
                    "target_id": result.target_id,
                    "name": result.name,
                    "truth": obs.truth,
                    "label": result.label,
                    "mass": result.mass,
                    "deltas": result.deltas,
                    "mean_conflict": result.diagnostics.get("mean_conflict", 0.0),
                    "x": safe_float(state.get("X")),
                    "y": safe_float(state.get("Y")),
                    "z": safe_float(state.get("Z")),
                    "vx": safe_float(state.get("VX")),
                    "vy": safe_float(state.get("VY")),
                    "vz": safe_float(state.get("VZ")),
                }
            )

        friendlies = []
        for f in friendly_series[t_idx]:
            friendlies.append(
                {
                    "id": int(f.get("Aircraft_ID", len(friendlies))),
                    "name": f.get("Name", "Friendly"),
                    "role": f.get("Role", ""),
                    "x": safe_float(f.get("X")),
                    "y": safe_float(f.get("Y")),
                    "z": safe_float(f.get("Z")),
                    "vx": safe_float(f.get("VX")),
                    "vy": safe_float(f.get("VY")),
                    "vz": safe_float(f.get("VZ")),
                }
            )

        records.append({"time": int(t_idx), "targets": targets, "friendlies": friendlies})

    return records, config


# 将输入安全转换为有限浮点数。
def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        value = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(value) or np.isinf(value):
        return default
    return value


# 提取指定对象在当前帧之前的一段轨迹尾迹。
def trajectory(records, group, obj_idx, frame_idx, tail_len):
    start = max(0, frame_idx - tail_len + 1)
    xs, ys, zs = [], [], []
    for record in records[start : frame_idx + 1]:
        items = record.get(group, [])
        if obj_idx >= len(items):
            continue
        item = items[obj_idx]
        x, y, z = item.get("x"), item.get("y"), item.get("z")
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
            xs.append(x)
            ys.append(y)
            zs.append(z)
    return np.asarray(xs), np.asarray(ys), np.asarray(zs)


# 计算全部动画帧的固定三维坐标轴范围。
def compute_axis_limits(records, frame_indices=None):
    if frame_indices is None:
        frame_indices = range(len(records))
    pts = []
    for idx in frame_indices:
        record = records[idx]
        for group in ("targets", "friendlies"):
            for item in record.get(group, []):
                x, y, z = item.get("x"), item.get("y"), item.get("z")
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                    pts.append((x, y, z))

    arr = np.asarray(pts, dtype=float)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if arr.size == 0:
        return None

    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    centers = (mins + maxs) / 2.0
    xy_range = max(maxs[0] - mins[0], maxs[1] - mins[1], 1.0)
    z_range = max(maxs[2] - mins[2], 10.0)
    pad_xy = xy_range * 0.12
    pad_z = max(2.0, z_range * 0.20)
    return {
        "xlim": (centers[0] - xy_range / 2 - pad_xy, centers[0] + xy_range / 2 + pad_xy),
        "ylim": (centers[1] - xy_range / 2 - pad_xy, centers[1] + xy_range / 2 + pad_xy),
        "zlim": (max(0.0, mins[2] - pad_z), maxs[2] + pad_z),
    }


# 将预计算坐标轴范围应用到三维坐标轴。
def apply_axis_limits(ax, limits):
    if limits is None:
        return
    ax.set_xlim(*limits["xlim"])
    ax.set_ylim(*limits["ylim"])
    ax.set_zlim(*limits["zlim"])


# 在三维场景中绘制最小风险返场通道参考线。
def draw_route_reference(ax, config, limits):
    route = config.route
    if limits is None:
        span = 450.0
        center_x, center_y = 0.0, 0.0
    else:
        x0, x1 = limits["xlim"]
        y0, y1 = limits["ylim"]
        span = max(x1 - x0, y1 - y0) * 0.75
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0

    heading = np.radians(route.heading_deg)
    direction = np.array([np.cos(heading), np.sin(heading)])
    p0 = np.array([center_x, center_y]) - direction * span / 2.0
    p1 = np.array([center_x, center_y]) + direction * span / 2.0
    z = route.height_m / 1000.0
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        [z, z],
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.55,
        label="IFF corridor center",
    )


# 绘制动画中的单个三维帧和右侧信息面板。
def draw_frame(ax3d, ax_text, records, frame_idx, config, tail_len, limits):
    ax3d.clear()
    ax_text.clear()
    record = records[frame_idx]
    time_value = record["time"]

    draw_route_reference(ax3d, config, limits)

    for idx, _friendly in enumerate(record.get("friendlies", [])):
        xs, ys, zs = trajectory(records, "friendlies", idx, frame_idx, tail_len)
        if len(xs):
            ax3d.plot(xs, ys, zs, color="#1f77b4", linewidth=1.8, alpha=0.58)

    for idx, target in enumerate(record.get("targets", [])):
        xs, ys, zs = trajectory(records, "targets", idx, frame_idx, tail_len)
        color = LABEL_COLORS.get(target.get("label"), LABEL_COLORS[None])
        if len(xs):
            ax3d.plot(xs, ys, zs, color=color, linewidth=1.5, alpha=0.62)

    for friendly in record.get("friendlies", []):
        x, y, z = friendly["x"], friendly["y"], friendly["z"]
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            continue
        ax3d.scatter(x, y, z, s=65, c="#1f77b4", marker="^", depthshade=True)
        ax3d.text(x, y, z + 1.2, friendly.get("role", "F"), fontsize=8, color="#1f77b4")

    for target in record.get("targets", []):
        x, y, z = target["x"], target["y"], target["z"]
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            continue
        label = target.get("label")
        color = LABEL_COLORS.get(label, LABEL_COLORS[None])
        marker = LABEL_MARKERS.get(label, ".")
        mass = target.get("mass", {})
        confidence = max(float(mass.get(k, 0.0)) for k in ("FR", "AC", "ST", "FO"))
        size = 45.0 + 135.0 * confidence
        ax3d.scatter(x, y, z, s=size, c=color, marker=marker, edgecolors="black", linewidths=0.45)
        ax3d.text(
            x,
            y,
            z + 0.9,
            f"T{target['target_id'] + 1}:{label}",
            fontsize=8,
            color=color,
        )
        draw_velocity_arrow(ax3d, target, color)

    apply_axis_limits(ax3d, limits)
    ax3d.set_title(f"IFF 3D real-time identity recognition | t = {time_value}s", fontsize=12)
    ax3d.set_xlabel("X / km")
    ax3d.set_ylabel("Y / km")
    ax3d.set_zlabel("Z / km")
    ax3d.grid(True, alpha=0.30)
    ax3d.view_init(elev=22, azim=-58)

    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        build_info_text(record, config),
        va="top",
        ha="left",
        fontsize=8.8,
        family="monospace",
    )


# 绘制目标当前速度方向箭头。
def draw_velocity_arrow(ax, target, color):
    vx, vy, vz = target.get("vx"), target.get("vy"), target.get("vz")
    if not (np.isfinite(vx) and np.isfinite(vy) and np.isfinite(vz)):
        return
    vec = np.asarray([vx, vy, vz], dtype=float)
    norm = np.linalg.norm(vec)
    if norm <= 1e-9:
        return
    direction = vec / norm
    length = float(np.clip(norm / 0.340 * 5.0, 2.0, 13.0))
    start = np.asarray([target["x"], target["y"], target["z"]], dtype=float) + direction * 1.2
    arrow = direction * length
    ax.quiver(
        start[0],
        start[1],
        start[2],
        arrow[0],
        arrow[1],
        arrow[2],
        color=color,
        linewidth=0.55,
        arrow_length_ratio=0.15,
        alpha=0.72,
    )


# 构建右侧面板中的识别概率和偏差信息文本。
def build_info_text(record, config):
    lines = [
        f"Time: {record['time']} s",
        f"Route: H={config.route.height_m:.0f}m V={config.route.speed_kmh:.0f}km/h C={config.route.heading_deg:.0f}deg",
        "",
        "Legend: FR blue | AC green | ST orange | FO red",
        "",
        "Target identity masses:",
    ]
    for item in record.get("targets", []):
        mass = item.get("mass", {})
        deltas = item.get("deltas", {})
        truth = item.get("truth") or "-"
        lines.append(
            f"T{item['target_id'] + 1:02d} {item['label']:<2} truth={truth:<2} "
            f"FR={mass.get('FR', 0.0):.2f} AC={mass.get('AC', 0.0):.2f} "
            f"ST={mass.get('ST', 0.0):.2f} FO={mass.get('FO', 0.0):.2f}"
        )
        lines.append(
            f"    dH={fmt_optional(deltas.get('H1')):>6}m "
            f"dV={fmt_optional(deltas.get('V')):>6} "
            f"dC={fmt_optional(deltas.get('C')):>6} "
            f"K={item.get('mean_conflict', 0.0):.2f} {item['name'][:22]}"
        )
    return "\n".join(lines)


# 将可选数值格式化为短字符串。
def fmt_optional(value):
    if value is None:
        return "None"
    return f"{float(value):.1f}"


# 将指定帧渲染为 PIL 图像对象。
def render_frame_to_image(fig, ax3d, ax_text, records, frame_idx, config, tail_len, limits):
    draw_frame(ax3d, ax_text, records, frame_idx, config, tail_len, limits)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(rgba).convert("RGB")


# 将可视化帧序列导出为 GIF 文件。
def export_gif(records, config, output_path, frame_indices, fps=8, dpi=95, tail_len=TAIL_LEN_DEFAULT):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15.8, 8.6), dpi=dpi)
    ax3d = fig.add_axes([0.055, 0.13, 0.61, 0.80], projection="3d")
    ax_text = fig.add_axes([0.70, 0.12, 0.28, 0.82])
    limits = compute_axis_limits(records, frame_indices)

    frames = []
    duration_ms = int(1000 / fps)
    total = len(frame_indices)
    for n, idx in enumerate(frame_indices, start=1):
        print(f"[GIF] rendering frame {n}/{total}, t={records[idx]['time']}s")
        img = render_frame_to_image(fig, ax3d, ax_text, records, idx, config, tail_len, limits)
        frames.append(img.convert("P", palette=Image.ADAPTIVE))

    if not frames:
        raise RuntimeError("No frames to export")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    plt.close(fig)
    print(f"GIF saved to: {output_path.resolve()}")


# 将可视化帧序列导出为 MP4 文件。
def export_mp4(records, config, output_path, frame_indices, fps=12, dpi=95, tail_len=TAIL_LEN_DEFAULT):
    from matplotlib.animation import FFMpegWriter

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15.8, 8.6), dpi=dpi)
    ax3d = fig.add_axes([0.055, 0.13, 0.61, 0.80], projection="3d")
    ax_text = fig.add_axes([0.70, 0.12, 0.28, 0.82])
    limits = compute_axis_limits(records, frame_indices)
    writer = FFMpegWriter(fps=fps, metadata={"title": "IFF 3D animation"}, bitrate=1800)

    total = len(frame_indices)
    with writer.saving(fig, str(output_path), dpi=dpi):
        for n, idx in enumerate(frame_indices, start=1):
            print(f"[MP4] rendering frame {n}/{total}, t={records[idx]['time']}s")
            draw_frame(ax3d, ax_text, records, idx, config, tail_len, limits)
            writer.grab_frame()
    plt.close(fig)
    print(f"MP4 saved to: {output_path.resolve()}")


# 解析命令行参数。
def parse_args():
    parser = argparse.ArgumentParser(description="Export IFF 3D dynamic identity-recognition visualization.")
    parser.add_argument("--format", choices=["gif", "mp4"], default="gif")
    parser.add_argument("--step", type=int, default=10, help="Sample one animation frame every N seconds.")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=601)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--tail-len", type=int, default=TAIL_LEN_DEFAULT)
    parser.add_argument("--dpi", type=int, default=95)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


# 执行动画导出主流程。
def main():
    args = parse_args()
    if args.step <= 0:
        raise ValueError("--step must be positive")
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive")

    records, config = build_iff_visual_records(num_steps=args.num_steps, window_size=args.window_size)
    frame_indices = list(range(0, len(records), args.step))
    if frame_indices[-1] != len(records) - 1:
        frame_indices.append(len(records) - 1)

    out_dir = Path(__file__).resolve().parent / "results"
    output = args.output
    if output is None:
        suffix = "gif" if args.format == "gif" else "mp4"
        output = out_dir / f"iff_3d_iff_animation.{suffix}"

    if args.format == "gif":
        export_gif(records, config, output, frame_indices, fps=args.fps, dpi=args.dpi, tail_len=args.tail_len)
    else:
        export_mp4(records, config, output, frame_indices, fps=args.fps, dpi=args.dpi, tail_len=args.tail_len)


if __name__ == "__main__":
    main()
