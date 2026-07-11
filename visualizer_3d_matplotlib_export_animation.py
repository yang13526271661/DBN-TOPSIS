# -*- coding: utf-8 -*-
"""
PNG 图标版三维编队威胁评估可视化（6类目标版本，放大图标+图标图例+性能优化）

使用前请确认：
1) 已运行主程序生成 results_fig/visual_data.json：
   python .\\DBN_AR_Generate_DS_06_team_DS_typefusion_with_visual_export.py

2) 工程目录下存在 icons 文件夹，并包含：
   icons/friendly_fighter.png
   icons/enemy_fighter.png
   icons/bomber.png
   icons/uav.png
   icons/helicopter.png
   icons/missile.png

3) 运行：
   python .\\visualizer_3d_matplotlib.py

说明：
- 本版本已经去掉 recon.png 和 fuel.png，不再要求这两个图标文件。
- 如果 visual_data.json 中意外出现 Recon/Fuel 类型，会自动用 enemy_fighter.png 作为兜底图标，不会报错。
"""

from __future__ import annotations

import json
from pathlib import Path as FilePath

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw

from matplotlib.widgets import Slider, Button
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d  # noqa: F401


# =========================
# 全局配置
# =========================
DATA_PATH = FilePath("results_fig") / "visual_data.json"
ICON_DIR = FilePath("icons")

TAIL_LEN_DEFAULT = 50
# 速度箭头参数
# 箭头含义：目标当前速度方向；箭头长度按目标速度大小缩放。
# 这里不用把速度向量直接乘很大倍数，而是先归一化方向，再给一个受限的长度，
# 避免箭头过粗、过长、遮挡目标。
VELOCITY_ARROW_COLOR = "red"
# 箭头总长度 = 横线长度 + 小箭头头部长度。
# 用“横线长短”表达速度大小，而不是把整个箭头画得很粗很大。
VELOCITY_ARROW_MIN_LEN = 3.0       # km，低速目标也能看见
VELOCITY_ARROW_MAX_LEN = 13.0      # km，防止高速目标箭头过长
VELOCITY_ARROW_LEN_PER_MACH = 4.2  # 每 1 Mach 增加的箭头长度
VELOCITY_ARROW_WIDTH = 0.45        # 箭杆细一点，避免遮挡目标
VELOCITY_ARROW_HEAD_RATIO = 0.12   # 箭头头部小一点，主要靠横线长度表达速度
VELOCITY_ARROW_ALPHA = 0.78
VELOCITY_ARROW_START_GAP = 2.0     # km，箭头从目标前方一点开始画，避免盖住图标

TEXT_Z_OFFSET = 2.0

# 只保留当前场景实际用到的 6 类图标
ICON_FILES = {
    "FriendlyFighter": "friendly_fighter.png",
    "Missile": "missile.png",
    "Fighter": "enemy_fighter.png",
    "Bomber": "bomber.png",
    "Heli": "helicopter.png",
    "UAV": "uav.png",
    "Unknown": "enemy_fighter.png",
}

# 图标在三维世界中的实际显示尺寸，单位 km。
# 这版不再用屏幕贴图坐标，而是把 PNG 作为小平面放到真实三维坐标位置上。
# 如果图标太大/太小，只改这里即可。
ICON_WORLD_SIZES = {
    # 单位是 km。数值越大，三维图标越大。
    # 当前值比上一版略大，但仍避免 T1/T2/T5 附近互相遮挡。
    "FriendlyFighter": 20.0,
    "Missile": 14.0,
    "Fighter": 20.0,
    "Bomber": 25.0,
    "Heli": 18.0,
    "UAV": 16.0,
    "Unknown": 17.0,
}

# 为了提高绘图速度，PNG 会被缩放到较小分辨率后再贴到 3D 平面上。
ICON_RENDER_PIXELS = 30

ICON_CACHE = {}
MISSING_ICON_WARNING_PRINTED = False

# 轨迹颜色仍按类型区分
TYPE_COLORS = {
    "FriendlyFighter": "tab:blue",
    "Missile": "red",
    "Fighter": "tab:orange",
    "Bomber": "tab:purple",
    "Heli": "tab:green",
    "UAV": "tab:gray",
    "Unknown": "black",
}


# =========================
# 数据读入
# =========================
def load_data(path: FilePath = DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path.resolve()}。\n"
            f"请先运行主程序生成 results_fig/visual_data.json。"
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    if not records:
        raise ValueError("visual_data.json 中 records 为空。")

    return data, records


# =========================
# 工具函数
# =========================
def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def get_value(obj: dict, *keys, default=np.nan):
    for k in keys:
        if k in obj:
            return obj.get(k)
    return default


def xyz_of(obj: dict):
    return (
        safe_float(get_value(obj, "x", "X")),
        safe_float(get_value(obj, "y", "Y")),
        safe_float(get_value(obj, "z", "Z")),
    )


def vxyz_of(obj: dict):
    return (
        safe_float(get_value(obj, "vx", "VX")),
        safe_float(get_value(obj, "vy", "VY")),
        safe_float(get_value(obj, "vz", "VZ")),
    )


def get_time_values(records):
    return [int(r.get("time", 0)) for r in records]


def find_record_index(records, t):
    times = get_time_values(records)
    t = int(t)
    if t in times:
        return times.index(t)
    return int(np.argmin(np.abs(np.array(times) - t)))


def fmt_rank(rank):
    if not rank:
        return ""
    return " > ".join(str(x) for x in rank)


def normalize_type(tp: str) -> str:
    """把未知/不在当前图标集合中的类型，映射到 Unknown，避免缺图标时报错。"""
    if tp in ICON_FILES:
        return tp
    return "Unknown"


def get_enemy_display_type(enemy: dict) -> str:
    """
    当前可视化显示类型：
    如果 D-S 做了类型修正，则优先显示 fused_type；
    否则显示 sensor_type / Type / type。
    """
    ds_action = enemy.get("ds_action", "")
    if ds_action in ("discount_sensor_type_by_DS", "temporal_window_conflict_by_DS") and enemy.get("fused_type"):
        return normalize_type(str(enemy.get("fused_type")))

    tp = str(
        enemy.get("sensor_type")
        or enemy.get("Type")
        or enemy.get("type")
        or enemy.get("fused_type")
        or "Unknown"
    )
    return normalize_type(tp)


def fallback_icon_image(tp: str):
    """Create a small transparent RGBA icon when external PNG assets are absent."""
    size = ICON_RENDER_PIXELS
    pad = max(3, size // 10)
    center = size / 2.0
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    fill_by_type = {
        "FriendlyFighter": (45, 121, 245, 235),
        "Missile": (220, 38, 38, 235),
        "Fighter": (245, 130, 32, 235),
        "Bomber": (126, 63, 181, 235),
        "Heli": (42, 160, 84, 235),
        "UAV": (105, 112, 122, 235),
        "Unknown": (40, 40, 40, 235),
    }
    fill = fill_by_type.get(tp, fill_by_type["Unknown"])
    outline = (20, 20, 20, 210)
    highlight = (255, 255, 255, 210)

    if tp == "Missile":
        body = [
            (center, pad),
            (size - pad - 1, size - pad - 1),
            (center, size - pad - 4),
            (pad, size - pad - 1),
        ]
        draw.polygon(body, fill=fill, outline=outline)
        draw.line((center, pad + 4, center, size - pad - 5), fill=highlight, width=1)
    elif tp == "Bomber":
        body = [
            (center, pad),
            (size - pad, center + 1),
            (center + 5, center + 4),
            (center + 4, size - pad),
            (center - 4, size - pad),
            (center - 5, center + 4),
            (pad, center + 1),
        ]
        draw.polygon(body, fill=fill, outline=outline)
        draw.rectangle((center - 3, pad + 4, center + 3, size - pad - 4), fill=highlight)
    elif tp == "Heli":
        draw.ellipse((center - 5, center - 7, center + 5, center + 8), fill=fill, outline=outline)
        draw.line((pad, center - 10, size - pad, center - 10), fill=outline, width=2)
        draw.line((center, pad, center, center - 3), fill=outline, width=2)
        draw.line((center, center + 8, center, size - pad), fill=outline, width=2)
        draw.line((center, size - pad, center + 8, size - pad + 1), fill=outline, width=2)
    elif tp == "UAV":
        body = [
            (center, pad),
            (size - pad, center),
            (center + 4, center + 3),
            (center + 2, size - pad),
            (center - 2, size - pad),
            (center - 4, center + 3),
            (pad, center),
        ]
        draw.polygon(body, fill=fill, outline=outline)
        draw.line((pad + 4, center, size - pad - 4, center), fill=highlight, width=1)
    else:
        body = [
            (center, pad),
            (center + 5, center + 5),
            (size - pad, center + 7),
            (center + 4, center + 10),
            (center + 3, size - pad),
            (center, size - pad - 4),
            (center - 3, size - pad),
            (center - 4, center + 10),
            (pad, center + 7),
            (center - 5, center + 5),
        ]
        draw.polygon(body, fill=fill, outline=outline)
        draw.line((center, pad + 4, center, size - pad - 5), fill=highlight, width=1)

    return np.asarray(img, dtype=float) / 255.0


# =========================
# PNG 图标函数：三维平面贴图版
# =========================
def load_icon_image(tp: str):
    """
    读取 PNG 图标，并缩小成 RGBA 数组。

    注意：这里不再使用 AnnotationBbox 屏幕贴图，而是后面用 plot_surface
    把图标作为一个小平面放到真实三维坐标位置。
    """
    tp = normalize_type(tp)
    if tp in ICON_CACHE:
        return ICON_CACHE[tp]

    filename = ICON_FILES.get(tp, ICON_FILES["Unknown"])
    icon_path = ICON_DIR / filename

    if not icon_path.exists():
        global MISSING_ICON_WARNING_PRINTED
        if not MISSING_ICON_WARNING_PRINTED:
            print(
                "[WARN] icons/*.png not found. "
                "Using built-in fallback icons for this export. "
                "Create an icons folder to use custom PNG aircraft icons."
            )
            MISSING_ICON_WARNING_PRINTED = True
        arr = fallback_icon_image(tp)
        ICON_CACHE[tp] = arr
        return arr

    if not icon_path.exists():
        required = "\n".join([f"  - {ICON_DIR / v}" for k, v in ICON_FILES.items() if k != "Unknown"])
        raise FileNotFoundError(
            f"找不到图标文件：{icon_path.resolve()}\n\n"
            f"请确认 icons 文件夹下至少包含以下 PNG 文件：\n{required}\n\n"
            f"注意：Windows 资源管理器可能隐藏 .png 后缀，文件实际名称必须是 xxx.png。"
        )

    img = Image.open(icon_path).convert("RGBA")
    img = img.resize((ICON_RENDER_PIXELS, ICON_RENDER_PIXELS), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=float) / 255.0

    ICON_CACHE[tp] = arr
    return arr


def add_icon3d(
    ax,
    tp: str,
    x: float,
    y: float,
    z: float,
    vx: float = 1.0,
    vy: float = 0.0,
    zoom_scale: float = 1.0,
    z_offset: float = 0.12,
):
    """
    在真实三维坐标处绘制 PNG 图标。

    实现方式：
    - 读取透明 PNG；
    - 构造一个位于目标 (x, y, z) 附近的小平面；
    - 小平面放在 XY 平面内，中心位于目标位置；
    - 根据目标水平速度方向 vx/vy 旋转图标，使图标机头大致指向飞行方向；
    - 用 ax.plot_surface(..., facecolors=img) 贴图。

    这样图标就是真正处在三维数据坐标上，不会再挤到图中央，也不会因投影坐标错误消失。
    """
    tp = normalize_type(tp)
    img = load_icon_image(tp)

    base_size = ICON_WORLD_SIZES.get(tp, ICON_WORLD_SIZES["Unknown"])
    size = base_size * zoom_scale

    # 计算图标朝向。默认 PNG 的“机头”朝图片上方，因此把图片竖直方向映射到速度方向。
    fwd = np.array([safe_float(vx, 0.0), safe_float(vy, 0.0)], dtype=float)
    norm = np.linalg.norm(fwd)
    if norm < 1e-8:
        fwd = np.array([1.0, 0.0], dtype=float)
    else:
        fwd = fwd / norm

    # 图片横向方向，与前向垂直。
    right = np.array([fwd[1], -fwd[0]], dtype=float)

    h, w = img.shape[:2]
    u = np.linspace(-size / 2.0, size / 2.0, w)
    # v 的第一行对应图片顶部，让图片顶部代表前向方向。
    v = np.linspace(size / 2.0, -size / 2.0, h)
    U, V = np.meshgrid(u, v)

    X = x + U * right[0] + V * fwd[0]
    Y = y + U * right[1] + V * fwd[1]
    Z = np.full_like(X, z + z_offset)

    ax.plot_surface(
        X,
        Y,
        Z,
        rstride=1,
        cstride=1,
        facecolors=img,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

# =========================
# 轨迹与坐标轴
# =========================
def trajectory(records, group: str, idx: int, frame_idx: int, tail_len: int):
    start = max(0, frame_idx - tail_len + 1)
    xs, ys, zs = [], [], []

    for r in records[start:frame_idx + 1]:
        group_list = r.get(group, [])
        if idx >= len(group_list):
            continue

        obj = group_list[idx]
        x, y, z = xyz_of(obj)

        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
            xs.append(x)
            ys.append(y)
            zs.append(z)

    return np.array(xs), np.array(ys), np.array(zs)


def all_positions_for_axes(records, frame_idx: int, tail_len: int):
    start = max(0, frame_idx - tail_len + 1)
    pts = []

    for r in records[start:frame_idx + 1]:
        for group in ("friendlies", "enemies"):
            for obj in r.get(group, []):
                x, y, z = xyz_of(obj)
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                    pts.append((x, y, z))

                xt = safe_float(get_value(obj, "x_true", "X_true"))
                yt = safe_float(get_value(obj, "y_true", "Y_true"))
                zt = safe_float(get_value(obj, "z_true", "Z_true"))
                if np.isfinite(xt) and np.isfinite(yt) and np.isfinite(zt):
                    pts.append((xt, yt, zt))

    return pts


def set_axes_equal(ax, pts):
    arr = np.array(pts, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return

    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if len(arr) == 0:
        return

    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    centers = (mins + maxs) / 2.0

    xy_range = max(maxs[0] - mins[0], maxs[1] - mins[1], 1.0)
    z_range = max(maxs[2] - mins[2], 8.0)

    pad_xy = xy_range * 0.12
    pad_z = max(4.0, z_range * 0.35)

    ax.set_xlim(centers[0] - xy_range / 2 - pad_xy, centers[0] + xy_range / 2 + pad_xy)
    ax.set_ylim(centers[1] - xy_range / 2 - pad_xy, centers[1] + xy_range / 2 + pad_xy)
    ax.set_zlim(max(0.0, mins[2] - pad_z), maxs[2] + pad_z)


# =========================
# 右侧文字信息
# =========================
def build_info_text(record: dict):
    lines = []
    t = int(record.get("time", 0))

    lines.append(f"Time: {t}s")
    lines.append("")

    lines.append("[Formation threat rank]")
    lines.append(f"Total: {fmt_rank(record.get('total_rank', []))}")
    lines.append(f"Form : {fmt_rank(record.get('form_rank', []))}")
    lines.append(f"Agg  : {fmt_rank(record.get('agg_rank', []))}")
    lines.append("")

    enemies = sorted(
        record.get("enemies", []),
        key=lambda e: safe_float(e.get("total_score"), 0.0),
        reverse=True,
    )

    lines.append("[Top targets]")
    for e in enemies[:5]:
        score = safe_float(e.get("total_score"), 0.0)
        tp = get_enemy_display_type(e)
        sensor = e.get("sensor_type") or e.get("Type") or e.get("type")
        fused = e.get("fused_type")
        ds_action = e.get("ds_action", "")
        suffix = f"({tp})"
        if ds_action == "discount_sensor_type_by_DS":
            suffix += f"  DS:{sensor}->{fused}"
        if e.get("has_type_jump") or ds_action == "temporal_window_conflict_by_DS":
            suffix += f"  JUMP:{safe_float(e.get('jump_score'), 0.0):.2f}"
        lines.append(f"{e.get('label')}: {score:.4f} {suffix}")
    lines.append("")

    lines.append("[Single-aircraft local ranks]")
    single_ranks = record.get("single_ranks", {})
    if isinstance(single_ranks, dict) and single_ranks:
        for k, v in single_ranks.items():
            lines.append(f"{k}: {fmt_rank(v)}")
    else:
        pair = np.array(record.get("pair_scores", []), dtype=float)
        labels = [e.get("label", f"T{j+1}") for j, e in enumerate(record.get("enemies", []))]
        for i, f in enumerate(record.get("friendlies", [])):
            if pair.ndim == 2 and i < pair.shape[0]:
                order = np.argsort(pair[i])[::-1]
                rank = " > ".join(labels[j] for j in order)
                lines.append(f"{f.get('label', 'F'+str(i+1))}: {rank}")

    lines.append("")

    event_lines = []
    for e in record.get("enemies", []):
        if e.get("ds_action") == "discount_sensor_type_by_DS":
            event_lines.append(
                f"D-S corrected {e.get('label')}: {e.get('sensor_type')} -> {e.get('fused_type')}"
            )
        if e.get("has_type_jump") or e.get("ds_action") == "temporal_window_conflict_by_DS":
            event_lines.append(
                f"Type jump {e.get('label')}: score={safe_float(e.get('jump_score'), 0.0):.2f}, "
                f"window K={safe_float(e.get('window_conflict'), 0.0):.2f}"
            )
        if e.get("is_missing"):
            event_lines.append(f"AR restoring {e.get('label')}")

    if event_lines:
        lines.append("[Events]")
        lines.extend(event_lines[:8])

    return "\n".join(lines)


def draw_type_legend_box(fig):
    """在图窗口左上角绘制固定图例框：图标和文字都包含在方框内。"""
    ax_leg = fig.add_axes([0.045, 0.67, 0.15, 0.24])
    ax_leg.set_xlim(0.0, 1.0)
    ax_leg.set_ylim(0.0, 1.0)
    ax_leg.axis("off")

    bg = FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02",
        facecolor="white",
        edgecolor="0.35",
        linewidth=1.1,
        alpha=0.92,
        transform=ax_leg.transAxes,
        zorder=0,
    )
    ax_leg.add_patch(bg)
    ax_leg.text(0.08, 0.93, "[Type icons]", fontsize=10, va="top", ha="left")

    legend_items = [
        ("FriendlyFighter", "Friendly fighter"),
        ("Missile", "Missile"),
        ("Fighter", "Enemy fighter"),
        ("Bomber", "Bomber"),
        ("Heli", "Helicopter"),
        ("UAV", "UAV"),
    ]

    y_top = 0.80
    dy = 0.125
    x_img0, x_img1 = 0.08, 0.18
    x_text = 0.24

    for idx, (tp, label) in enumerate(legend_items):
        y = y_top - idx * dy
        try:
            img = load_icon_image(tp)
            ax_leg.imshow(img, extent=(x_img0, x_img1, y - 0.04, y + 0.04), aspect='auto', zorder=1)
        except Exception:
            pass
        ax_leg.text(x_text, y, label, fontsize=9.5, va='center', ha='left', zorder=2)

    return ax_leg


# =========================
# 绘图主函数
# =========================
def draw_frame(
    ax3d,
    ax_text,
    records,
    frame_idx: int,
    tail_len: int = TAIL_LEN_DEFAULT,
    show_velocity: bool = True,
    show_truth: bool = True,
    show_legend: bool = True,
):
    ax3d.clear()
    ax_text.clear()

    record = records[frame_idx]
    t = int(record.get("time", 0))

    # 我方轨迹尾迹
    for i, _ in enumerate(record.get("friendlies", [])):
        xs, ys, zs = trajectory(records, "friendlies", i, frame_idx, tail_len)
        if len(xs) > 0:
            ax3d.plot(xs, ys, zs, color=TYPE_COLORS["FriendlyFighter"], alpha=0.60, linewidth=2.2)

    # 敌方轨迹尾迹
    for j, e in enumerate(record.get("enemies", [])):
        tp = get_enemy_display_type(e)
        color = TYPE_COLORS.get(tp, TYPE_COLORS["Unknown"])
        xs, ys, zs = trajectory(records, "enemies", j, frame_idx, tail_len)
        if len(xs) > 0:
            ax3d.plot(xs, ys, zs, color=color, alpha=0.62, linewidth=1.9)

    # 当前我方编队图标
    friendlies = record.get("friendlies", [])
    fx, fy, fz = [], [], []

    for f in friendlies:
        x, y, z = xyz_of(f)
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            continue

        fx.append(x)
        fy.append(y)
        fz.append(z)

        vx, vy, _ = vxyz_of(f)
        add_icon3d(ax3d, "FriendlyFighter", x, y, z, vx=vx, vy=vy, zoom_scale=1.0)
        ax3d.text(x, y, z + TEXT_Z_OFFSET, f.get("label", "F"), fontsize=9)

    # 连接成编队轮廓
    if len(fx) >= 4:
        order = [0, 1, 3, 2, 0]
        ax3d.plot(
            [fx[k] for k in order],
            [fy[k] for k in order],
            [fz[k] for k in order],
            color=TYPE_COLORS["FriendlyFighter"],
            alpha=0.92,
            linewidth=2.4,
        )

    # 当前敌方目标图标
    enemies = record.get("enemies", [])
    for e in enemies:
        x, y, z = xyz_of(e)
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            continue

        tp = get_enemy_display_type(e)
        score = safe_float(e.get("total_score"), 0.0)

        # 威胁度越大，图标只轻微放大，避免相互遮挡
        zoom_scale = 0.92 + 0.20 * score
        vx, vy, _ = vxyz_of(e)
        add_icon3d(ax3d, tp, x, y, z, vx=vx, vy=vy, zoom_scale=zoom_scale)

        label = e.get("label", "T")
        ax3d.text(x, y, z + TEXT_Z_OFFSET, label, fontsize=9)

        if show_velocity:
            vx, vy, vz = vxyz_of(e)
            if np.isfinite(vx) and np.isfinite(vy) and np.isfinite(vz):
                v_vec = np.array([vx, vy, vz], dtype=float)
                v_norm = np.linalg.norm(v_vec)

                if v_norm > 1e-8:
                    # 速度单位为 km/s；0.340 km/s 约等于 Mach 1。
                    speed_mach = v_norm / 0.340

                    # 箭头长度表示速度大小：低速目标短，高速目标长；
                    # 同时设置上下限，避免太短看不清或太长遮挡目标。
                    arrow_len = VELOCITY_ARROW_MIN_LEN + VELOCITY_ARROW_LEN_PER_MACH * speed_mach
                    arrow_len = float(np.clip(
                        arrow_len,
                        VELOCITY_ARROW_MIN_LEN,
                        VELOCITY_ARROW_MAX_LEN
                    ))

                    direction = v_vec / v_norm

                    # 箭头从目标图标前方一点开始画，避免箭头头部/箭杆压住图标。
                    start = np.array([x, y, z], dtype=float) + direction * VELOCITY_ARROW_START_GAP

                    # 箭头方向与速度方向一致；长度由速度大小决定。
                    # 低速：  —>
                    # 高速：  ————>
                    arrow_vec = direction * arrow_len

                    ax3d.quiver(
                        start[0], start[1], start[2],
                        arrow_vec[0], arrow_vec[1], arrow_vec[2],
                        color=VELOCITY_ARROW_COLOR,
                        linewidth=VELOCITY_ARROW_WIDTH,
                        arrow_length_ratio=VELOCITY_ARROW_HEAD_RATIO,
                        alpha=VELOCITY_ARROW_ALPHA,
                    )

    # 真实点位（用于缺失/插补参考）
    if show_truth:
        for e in enemies:
            xt = safe_float(get_value(e, "x_true", "X_true"))
            yt = safe_float(get_value(e, "y_true", "Y_true"))
            zt = safe_float(get_value(e, "z_true", "Z_true"))

            if np.isfinite(xt) and np.isfinite(yt) and np.isfinite(zt):
                ax3d.scatter(xt, yt, zt, c="black", s=12, marker=".", alpha=0.28, depthshade=True)

    pts = all_positions_for_axes(records, frame_idx, tail_len)
    set_axes_equal(ax3d, pts)

    ax3d.set_title(f"3D formation threat assessment scene  |  t = {t}s", fontsize=12)
    ax3d.set_xlabel("X / km")
    ax3d.set_ylabel("Y / km")
    ax3d.set_zlabel("Z / km")
    ax3d.grid(True, alpha=0.35)
    ax3d.view_init(elev=22, azim=-58)


    ax_text.axis("off")
    ax_text.text(
        0.0, 1.0,
        build_info_text(record),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )



# =========================
# 动画 / 视频导出
# =========================
def compute_global_axis_limits(records, frame_indices=None):
    """
    为导出动画预先计算固定坐标范围，避免视频播放时坐标轴随时间跳动。
    """
    pts = []
    if frame_indices is None:
        frame_indices = range(len(records))

    for idx in frame_indices:
        r = records[idx]
        for group in ("friendlies", "enemies"):
            for obj in r.get(group, []):
                x, y, z = xyz_of(obj)
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                    pts.append((x, y, z))

    arr = np.array(pts, dtype=float)
    arr = arr[np.all(np.isfinite(arr), axis=1)]

    if len(arr) == 0:
        return None

    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    centers = (mins + maxs) / 2.0

    xy_range = max(maxs[0] - mins[0], maxs[1] - mins[1], 1.0)
    z_range = max(maxs[2] - mins[2], 8.0)

    pad_xy = xy_range * 0.10
    pad_z = max(4.0, z_range * 0.28)

    return {
        "xlim": (centers[0] - xy_range / 2 - pad_xy, centers[0] + xy_range / 2 + pad_xy),
        "ylim": (centers[1] - xy_range / 2 - pad_xy, centers[1] + xy_range / 2 + pad_xy),
        "zlim": (max(0.0, mins[2] - pad_z), maxs[2] + pad_z),
    }


def apply_axis_limits(ax3d, limits):
    if limits is None:
        return
    ax3d.set_xlim(*limits["xlim"])
    ax3d.set_ylim(*limits["ylim"])
    ax3d.set_zlim(*limits["zlim"])


def render_frame_to_image(fig, ax3d, ax_text, records, frame_idx, fixed_limits):
    """
    绘制某一帧，并把当前 Matplotlib 画布转换为 PIL Image。
    """
    draw_frame(
        ax3d,
        ax_text,
        records,
        frame_idx,
        tail_len=TAIL_LEN_DEFAULT,
        show_velocity=True,
        show_truth=False,
        show_legend=False,
    )

    apply_axis_limits(ax3d, fixed_limits)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(rgba).convert("RGB")
    return img


def export_gif(records, output_path, frame_indices, fps=8, dpi=100):
    """
    导出 GIF 动图。
    不依赖 ffmpeg，Windows 下最稳。
    """
    output_path = FilePath(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15.8, 8.6), dpi=dpi)
    ax3d = fig.add_axes([0.055, 0.14, 0.60, 0.80], projection="3d")
    ax_text = fig.add_axes([0.71, 0.15, 0.26, 0.79])

    draw_type_legend_box(fig)
    fixed_limits = compute_global_axis_limits(records, frame_indices)

    pil_frames = []
    duration_ms = int(1000 / fps)

    total = len(frame_indices)
    for n, idx in enumerate(frame_indices, start=1):
        t = records[idx].get("time", idx)
        print(f"[GIF] rendering frame {n}/{total}, t={t}s")
        img = render_frame_to_image(fig, ax3d, ax_text, records, idx, fixed_limits)
        # 转调色板可显著减小 GIF 体积
        pil_frames.append(img.convert("P", palette=Image.ADAPTIVE))

    if not pil_frames:
        raise RuntimeError("没有可导出的帧。")

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )

    plt.close(fig)
    print(f"\nGIF saved to: {output_path.resolve()}")


def export_mp4(records, output_path, frame_indices, fps=12, dpi=100):
    """
    导出 MP4 视频。
    需要本机已安装 ffmpeg，并且 Matplotlib 能找到 ffmpeg。
    """
    from matplotlib.animation import FFMpegWriter

    output_path = FilePath(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15.8, 8.6), dpi=dpi)
    ax3d = fig.add_axes([0.055, 0.14, 0.60, 0.80], projection="3d")
    ax_text = fig.add_axes([0.71, 0.15, 0.26, 0.79])

    draw_type_legend_box(fig)
    fixed_limits = compute_global_axis_limits(records, frame_indices)

    writer = FFMpegWriter(
        fps=fps,
        metadata={"title": "3D formation threat assessment"},
        bitrate=1800,
    )

    total = len(frame_indices)
    with writer.saving(fig, str(output_path), dpi=dpi):
        for n, idx in enumerate(frame_indices, start=1):
            t = records[idx].get("time", idx)
            print(f"[MP4] rendering frame {n}/{total}, t={t}s")
            draw_frame(
                ax3d,
                ax_text,
                records,
                idx,
                tail_len=TAIL_LEN_DEFAULT,
                show_velocity=True,
                show_truth=False,
                show_legend=False,
            )
            apply_axis_limits(ax3d, fixed_limits)
            writer.grab_frame()

    plt.close(fig)
    print(f"\nMP4 saved to: {output_path.resolve()}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Export 3D formation threat assessment visualization to GIF or MP4."
    )
    parser.add_argument(
        "--format",
        choices=["gif", "mp4"],
        default="gif",
        help="导出格式。gif 不需要 ffmpeg；mp4 需要 ffmpeg。默认 gif。",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="每隔多少秒/帧取一帧导出。默认 5。数值越小越流畅但越慢、文件越大。",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="导出帧率。GIF 默认建议 6~10，MP4 可设 10~15。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径。默认保存到 results_fig/threat_assessment_animation.gif 或 .mp4。",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="导出分辨率。默认 100。想减小文件可设 80。",
    )

    args = parser.parse_args()

    _, records = load_data(DATA_PATH)
    if args.step <= 0:
        raise ValueError("--step 必须大于 0")

    frame_indices = list(range(0, len(records), args.step))
    if frame_indices[-1] != len(records) - 1:
        frame_indices.append(len(records) - 1)

    if args.output is None:
        suffix = "gif" if args.format == "gif" else "mp4"
        args.output = str(FilePath("results_fig") / f"threat_assessment_animation.{suffix}")

    if args.format == "gif":
        export_gif(records, args.output, frame_indices, fps=args.fps, dpi=args.dpi)
    else:
        try:
            export_mp4(records, args.output, frame_indices, fps=args.fps, dpi=args.dpi)
        except Exception as e:
            print("\nMP4 导出失败，通常是因为本机没有安装 ffmpeg。")
            print(f"错误信息：{e}")
            print("你可以先导出 GIF：")
            print("python .\\visualizer_3d_matplotlib_export_animation.py --format gif --step 5 --fps 8")
            raise


if __name__ == "__main__":
    main()
