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

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    if hasattr(Image, "LANCZOS"):
        RESAMPLE_LANCZOS = Image.LANCZOS
    else:
        RESAMPLE_LANCZOS = Image.ANTIALIAS


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
VELOCITY_ARROW_COLOR = "0.25"
# 箭头总长度 = 横线长度 + 小箭头头部长度。
# 用“横线长短”表达速度大小，而不是把整个箭头画得很粗很大。
VELOCITY_ARROW_MIN_LEN = 3.0       # km，低速目标也能看见
VELOCITY_ARROW_MAX_LEN = 13.0      # km，防止高速目标箭头过长
VELOCITY_ARROW_LEN_PER_MACH = 4.2  # 每 1 Mach 增加的箭头长度
VELOCITY_ARROW_WIDTH = 0.28        # 箭杆细一点，避免遮挡目标
VELOCITY_ARROW_HEAD_RATIO = 0.08   # 箭头头部小一点，主要靠横线长度表达速度
VELOCITY_ARROW_ALPHA = 0.42
VELOCITY_ARROW_START_GAP = 8.0     # km，箭头从目标前方一点开始画，避免盖住图标

FRIENDLY_LABEL_Z_OFFSET = 2.0
ENEMY_LABEL_Z_OFFSET = 0.7

# 只保留当前场景实际用到的 6 类图标
ICON_FILES = {
    "FriendlyFighter": "friendly_fighter.png",
    "Missile": "missile.png",
    "Fighter": "enemy_fighter.png",
    "Bomber": "bomber.png",
    "Heli": "helicopter.png",
    "UAV": "uav.png",
    "Recon": "recon.png",
    "Fuel": "fuel.png",
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
    "Recon": 18.0,
    "Fuel": 23.0,
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
    "Recon": "tab:cyan",
    "Fuel": "tab:brown",
    "Unknown": "black",
}

TRAIL_COLORS = {
    **TYPE_COLORS,
    "Missile": "#b00020",
    "Fighter": "#e6862a",
    "Recon": "#00a6b2",
    "Fuel": "#8c564b",
    "Unknown": "0.25",
}

DEFAULT_TRAIL_STYLE = {"alpha": 0.55, "linewidth": 1.8, "linestyle": "-"}
SCENE_B_FEINT_TRAIL_STYLE = {
    "color": "#d97706",
    "alpha": 0.92,
    "linewidth": 2.8,
    "linestyle": "-",
}
TRAIL_STYLES = {
    "Missile": {"alpha": 0.95, "linewidth": 2.8, "linestyle": "-"},
    "Fighter": {"alpha": 0.48, "linewidth": 1.6, "linestyle": "-"},
    "Bomber": {"alpha": 0.52, "linewidth": 1.7, "linestyle": "-"},
    "UAV": {"alpha": 0.50, "linewidth": 1.5, "linestyle": "-"},
    "Heli": {"alpha": 0.50, "linewidth": 1.5, "linestyle": "-"},
    "Recon": {"alpha": 0.46, "linewidth": 1.4, "linestyle": "-"},
    "Fuel": {"alpha": 0.46, "linewidth": 1.4, "linestyle": "-"},
}

TYPE_DISPLAY_NAMES = {
    "FriendlyFighter": "Friendly fighter",
    "Missile": "Missile",
    "Fighter": "Enemy fighter",
    "Bomber": "Bomber",
    "Heli": "Helicopter",
    "UAV": "UAV",
    "Recon": "Recon",
    "Fuel": "Fuel",
    "Unknown": "Unknown",
}


FRIENDLY_ROLE_STYLES = {
    "Leader": {
        "display": "Leader",
        "abbr": "L",
        "color": "#D62728",
        "marker": "D",
        "linewidth": 3.2,
        "linestyle": "-",
        "zoom": 1.18,
    },
    "LeftWing": {
        "display": "Left",
        "abbr": "LW",
        "color": "#1F77B4",
        "marker": "o",
        "linewidth": 2.3,
        "linestyle": "-",
        "zoom": 1.0,
    },
    "RightWing": {
        "display": "Right",
        "abbr": "RW",
        "color": "#2CA02C",
        "marker": "^",
        "linewidth": 2.3,
        "linestyle": "-",
        "zoom": 1.0,
    },
    "RearGuard": {
        "display": "Rear",
        "abbr": "RG",
        "color": "#9467BD",
        "marker": "s",
        "linewidth": 2.5,
        "linestyle": "--",
        "zoom": 1.05,
    },
    "Member": {
        "display": "Homogeneous member",
        "abbr": "",
        "color": "tab:blue",
        "marker": "o",
        "linewidth": 2.2,
        "linestyle": "-",
        "zoom": 1.0,
    },
}


def get_friendly_role(friendly: dict):
    role = str(friendly.get("role") or friendly.get("Role") or "Member")
    return role if role in FRIENDLY_ROLE_STYLES else "Member"


def get_friendly_role_style(friendly: dict):
    return FRIENDLY_ROLE_STYLES[get_friendly_role(friendly)]


def has_heterogeneous_friendly_roles(record: dict):
    return any(get_friendly_role(f) != "Member" for f in record.get("friendlies", []))


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
    优先显示场景设定/真实类型，避免 D-S 的 fused_type 把图标改成另一类目标。
    D-S 修正结果仍然在右侧信息栏中展示。
    """
    tp = str(
        enemy.get("display_type")
        or enemy.get("true_type")
        or enemy.get("sensor_type")
        or enemy.get("Type")
        or enemy.get("type")
        or enemy.get("fused_type")
        or "Unknown"
    )
    return normalize_type(tp)


def uses_full_history_trail(enemy: dict) -> bool:
    """Scene-B T2 keeps its complete approach-turn-departure trajectory."""
    return enemy.get("small_scene_id") == "B_M2"


def freeze_enemy_display_types(records):
    """
    Freeze one visual type per target label for the whole animation.

    Sensor and D-S fused types may change during misidentification intervals.  The
    icon should represent the scenario/true platform type, so we resolve it once
    and store it as display_type on every frame.
    """
    type_by_label = {}

    for record in records:
        for enemy in record.get("enemies", []):
            label = enemy.get("label")
            if not label or label in type_by_label:
                continue

            raw_tp = (
                enemy.get("true_type")
                or enemy.get("sensor_type")
                or enemy.get("Type")
                or enemy.get("type")
                or enemy.get("fused_type")
                or "Unknown"
            )
            type_by_label[label] = normalize_type(str(raw_tp))

    for record in records:
        for enemy in record.get("enemies", []):
            label = enemy.get("label")
            if label in type_by_label:
                enemy["display_type"] = type_by_label[label]

    return type_by_label


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
        "Recon": (0, 166, 178, 235),
        "Fuel": (140, 86, 75, 235),
        "Unknown": (40, 40, 40, 235),
    }
    fill = fill_by_type.get(tp, fill_by_type["Unknown"])
    outline = (20, 20, 20, 210)
    highlight = (255, 255, 255, 210)

    if tp == "Missile":
        body = [
            (center, pad),
            (center + 4, pad + 7),
            (center + 3, size - pad - 9),
            (center + 8, size - pad - 3),
            (center + 2, size - pad - 5),
            (center, size - pad),
            (center - 2, size - pad - 5),
            (center - 8, size - pad - 3),
            (center - 3, size - pad - 9),
            (center - 4, pad + 7),
        ]
        draw.polygon(body, fill=fill, outline=outline)
        draw.line((center, pad + 6, center, size - pad - 7), fill=highlight, width=1)
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
    elif tp == "Recon":
        draw.ellipse((pad + 2, center - 6, size - pad - 2, center + 6), fill=fill, outline=outline)
        draw.ellipse((center - 4, center - 4, center + 4, center + 4), fill=highlight, outline=outline)
        draw.line((center, pad, center, center - 6), fill=outline, width=2)
        draw.line((center, center + 6, center, size - pad), fill=outline, width=2)
    elif tp == "Fuel":
        body = [
            (center, pad),
            (size - pad, center + 2),
            (center + 4, center + 5),
            (center + 3, size - pad),
            (center - 3, size - pad),
            (center - 4, center + 5),
            (pad, center + 2),
        ]
        draw.polygon(body, fill=fill, outline=outline)
        draw.rectangle((center - 2, pad + 5, center + 2, size - pad - 5), fill=highlight)
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

    if tp == "Missile":
        arr = fallback_icon_image(tp)
        ICON_CACHE[tp] = arr
        return arr

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
    img = img.resize((ICON_RENDER_PIXELS, ICON_RENDER_PIXELS), Image.LANCZOS)
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

    lines.append("[All target threat scores]")
    for e in enemies:
        score = safe_float(e.get("total_score"), 0.0)
        tp = get_enemy_display_type(e)
        sensor = e.get("sensor_type") or e.get("Type") or e.get("type")
        fused = e.get("fused_type")
        ds_action = e.get("ds_action", "")
        suffix = f"({tp})"
        if ds_action == "discount_sensor_type_by_DS":
            suffix += f"  DS:{sensor}->{fused}"
        frf = e.get("formation_risk_factor")
        if frf is not None:
            suffix += f"  FRF={safe_float(frf, 0.0):.2f}"
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
                friendly_label = f.get("label", "F" + str(i + 1))
                role = get_friendly_role(f)
                if role != "Member":
                    friendly_label += f"[{FRIENDLY_ROLE_STYLES[role]['abbr']}]"
                lines.append(f"{friendly_label}: {rank}")

    lines.append("")

    capability_members = [
        f for f in record.get("friendlies", []) if f.get("capability_event")
    ]
    if capability_members:
        lines.append("[Friendly capability state]")
        for f in capability_members:
            role = get_friendly_role(f)
            role_abbr = FRIENDLY_ROLE_STYLES[role]["abbr"]
            state = f.get("capability_state", "Healthy")
            vulnerability = safe_float(f.get("vulnerability"), 1.0)
            maneuverability = safe_float(f.get("maneuverability"), 1.0)
            weight = safe_float(f.get("aggregation_weight"), 0.0)
            lines.append(
                f"{f.get('label', 'F')}[{role_abbr}] {state}: "
                f"V={vulnerability:.2f} M={maneuverability:.2f} w={weight:.3f}"
            )
        lines.append("")

    event_lines = []
    for f in record.get("friendlies", []):
        state = f.get("capability_state", "Healthy")
        if state in ("Degrading", "Degraded"):
            damage_level = 100.0 * safe_float(f.get("damage_level"), 0.0)
            event_lines.append(
                f"{f.get('label', 'F')} capability {state}: {damage_level:.0f}%"
            )

    for e in record.get("enemies", []):
        if e.get("ds_action") == "discount_sensor_type_by_DS":
            event_lines.append(
                f"D-S corrected {e.get('label')}: {e.get('sensor_type')} -> {e.get('fused_type')}"
            )
        if e.get("intent_switch"):
            configured = e.get("configured_intent") or "Unknown"
            current = e.get("intent_gt") or "Unknown"
            predicted = e.get("predicted_intent") or "Unknown"
            phase = e.get("intent_phase") or "Unknown"
            turn_percent = 100.0 * safe_float(e.get("turn_progress"), 0.0)
            event_lines.append(
                f"Intent {e.get('label')}: GT={current}, Pred={predicted}"
            )
            if current != configured:
                phase += f" -> {configured}"
            event_lines.append(f"Phase={phase}, turn={turn_percent:.0f}%")
        if e.get("is_missing"):
            event_lines.append(f"AR restoring {e.get('label')}")

    if event_lines:
        lines.append("[Events]")
        lines.extend(event_lines[:8])

    return "\n".join(lines)


def draw_type_legend_box(fig, record=None):
    """在图窗口左上角绘制固定图例框：图标和文字都包含在方框内。"""
    if record is not None:
        enemies = record.get("enemies", [])
        friendlies = record.get("friendlies", [])
        heterogeneous = has_heterogeneous_friendly_roles(record)
        friendly_row_count = len(friendlies) if heterogeneous else 1
        row_count = friendly_row_count + len(enemies)
        box_h = min(0.47, max(0.25, 0.055 * (row_count + 1.6)))
        ax_leg = fig.add_axes([0.012, 0.95 - box_h, 0.185, box_h])
    else:
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

    if record is not None:
        ax_leg.text(0.06, 0.95, "[Roles / targets]", fontsize=9.0, va="top", ha="left")

        legend_items = []
        if heterogeneous:
            for f in friendlies:
                role = get_friendly_role(f)
                aircraft_type = str(f.get("aircraft_type") or f.get("AircraftType") or "fighter")
                aircraft_type = aircraft_type.replace("_", " ")
                label = f"{f.get('label', 'F')}: {role} | {aircraft_type}"
                legend_items.append(("FriendlyFighter", label, role))
        else:
            legend_items.append(("FriendlyFighter", "Friendly: homogeneous", "Member"))

        for e in enemies:
            raw_tp = str(
                e.get("display_type")
                or e.get("true_type")
                or e.get("sensor_type")
                or e.get("Type")
                or e.get("type")
                or e.get("fused_type")
                or "Unknown"
            )
            icon_tp = normalize_type(raw_tp)
            type_name = "Helicopter" if icon_tp == "Heli" else icon_tp
            intent = e.get("intent_gt")
            if intent:
                configured_intent = e.get("configured_intent")
                if e.get("intent_switch") and configured_intent and configured_intent != intent:
                    intent_text = f"{intent}->{configured_intent}"
                else:
                    intent_text = str(intent)
                legend_items.append((icon_tp, f"{e.get('label', 'T')} {type_name} | {intent_text}", None))
            else:
                legend_items.append((icon_tp, f"{e.get('label', 'T')} {type_name}", None))

        y_top = 0.84
        dy = min(0.088, 0.76 / max(1, len(legend_items) - 1))
        x_img0, x_img1 = 0.055, 0.155
        x_text = 0.20

        for idx, (tp, label, role) in enumerate(legend_items):
            y = y_top - idx * dy
            try:
                img = load_icon_image(tp)
                ax_leg.imshow(img, extent=(x_img0, x_img1, y - 0.032, y + 0.032), aspect='auto', zorder=1)
            except Exception:
                pass

            if role and role != "Member":
                role_style = FRIENDLY_ROLE_STYLES[role]
                ax_leg.scatter(
                    [(x_img0 + x_img1) / 2.0],
                    [y],
                    s=105,
                    marker=role_style["marker"],
                    facecolors="none",
                    edgecolors=role_style["color"],
                    linewidths=1.8,
                    zorder=3,
                )

            ax_leg.text(x_text, y, label, fontsize=7.5, va='center', ha='left', zorder=2)

        return ax_leg

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
    show_title: bool = True,
):
    ax3d.clear()
    if ax_text is not None:
        ax_text.clear()

    record = records[frame_idx]
    t = int(record.get("time", 0))

    # 我方轨迹尾迹
    for i, friendly in enumerate(record.get("friendlies", [])):
        role_style = get_friendly_role_style(friendly)
        xs, ys, zs = trajectory(records, "friendlies", i, frame_idx, tail_len)
        if len(xs) > 0:
            ax3d.plot(
                xs,
                ys,
                zs,
                color=role_style["color"],
                alpha=0.72,
                linewidth=role_style["linewidth"],
                linestyle=role_style["linestyle"],
            )

    # 敌方轨迹尾迹
    enemy_trail_items = list(enumerate(record.get("enemies", [])))
    enemy_trail_items.sort(
        key=lambda item: 1 if get_enemy_display_type(item[1]) == "Missile" else 0
    )
    for j, e in enemy_trail_items:
        tp = get_enemy_display_type(e)
        full_history = uses_full_history_trail(e)
        style = (
            SCENE_B_FEINT_TRAIL_STYLE
            if full_history
            else TRAIL_STYLES.get(tp, DEFAULT_TRAIL_STYLE)
        )
        color = style.get("color", TRAIL_COLORS.get(tp, TYPE_COLORS["Unknown"]))
        history_length = frame_idx + 1 if full_history else tail_len
        xs, ys, zs = trajectory(records, "enemies", j, frame_idx, history_length)
        if len(xs) > 0:
            ax3d.plot(
                xs,
                ys,
                zs,
                color=color,
                alpha=style["alpha"],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
            )

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

        role = get_friendly_role(f)
        role_style = FRIENDLY_ROLE_STYLES[role]
        vx, vy, _ = vxyz_of(f)
        add_icon3d(
            ax3d,
            "FriendlyFighter",
            x,
            y,
            z,
            vx=vx,
            vy=vy,
            zoom_scale=role_style["zoom"],
        )

        if role != "Member":
            ax3d.scatter(
                [x],
                [y],
                [z],
                s=210,
                marker=role_style["marker"],
                facecolors="none",
                edgecolors=role_style["color"],
                linewidths=2.2,
                depthshade=False,
            )
            capability_state = f.get("capability_state", "Healthy")
            if capability_state in ("Degrading", "Degraded"):
                ax3d.scatter(
                    [x],
                    [y],
                    [z],
                    s=285,
                    marker="x",
                    color="#b91c1c",
                    linewidths=2.0,
                    depthshade=False,
                )
                friendly_label = f"{f.get('label', 'F')} [{role_style['abbr']}|DEG]"
            else:
                friendly_label = f"{f.get('label', 'F')} [{role_style['abbr']}]"
            ax3d.text(
                x,
                y,
                z + FRIENDLY_LABEL_Z_OFFSET,
                friendly_label,
                fontsize=9,
                color=role_style["color"],
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": role_style["color"],
                    "linewidth": 0.8,
                    "alpha": 0.78,
                },
            )
        else:
            ax3d.text(
                x,
                y,
                z + FRIENDLY_LABEL_Z_OFFSET,
                f.get("label", "F"),
                fontsize=9,
            )

    # 连接成编队轮廓
    if len(fx) >= 4:
        formation_type = str(record.get("formation_type", "Wedge"))
        if "WideSearchLine" in formation_type or "LineAbreast" in formation_type:
            link_segments = [(1, 0), (0, 2), (0, 3)]
        elif "ProtectiveBox" in formation_type or "DefensiveSpread" in formation_type:
            link_segments = [(1, 0), (2, 0), (0, 3), (1, 2)]
        else:
            link_segments = [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)]

        for a, b in link_segments:
            ax3d.plot(
                [fx[a], fx[b]],
                [fy[a], fy[b]],
                [fz[a], fz[b]],
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

        add_icon3d(ax3d, tp, x, y, z, vx=vx, vy=vy, zoom_scale=zoom_scale)

        label = e.get("label", "T")
        ax3d.text(x, y, z + ENEMY_LABEL_Z_OFFSET, label, fontsize=9)

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

    if show_title:
        formation_type = record.get("formation_type", "")
        title_suffix = f"  |  formation = {formation_type}" if formation_type else ""
        ax3d.set_title(f"3D formation threat assessment scene  |  t = {t}s{title_suffix}", fontsize=12)
    else:
        ax3d.set_title("")
    ax3d.set_xlabel("X / km")
    ax3d.set_ylabel("Y / km")
    ax3d.set_zlabel("Z / km")
    ax3d.grid(True, alpha=0.35)
    ax3d.view_init(elev=22, azim=-58)


    if ax_text is not None:
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


def export_scene_png(records, output_path, frame_idx, dpi=300, fixed_limits=None):
    """Export one scene-only frame without legends, rankings, or event text."""
    output_path = FilePath(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.2, 6.1), dpi=dpi)
    ax3d = fig.add_axes([0.03, 0.06, 0.92, 0.92], projection="3d")

    draw_frame(
        ax3d,
        None,
        records,
        frame_idx,
        tail_len=TAIL_LEN_DEFAULT,
        show_velocity=True,
        show_truth=False,
        show_legend=False,
        show_title=False,
    )
    if fixed_limits is None:
        fixed_limits = compute_global_axis_limits(records)
    apply_axis_limits(ax3d, fixed_limits)

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    plt.close(fig)
    print(f"Scene PNG saved to: {output_path.resolve()}")


def export_scene_gif(
    records,
    output_path,
    frame_indices,
    fps=6,
    dpi=100,
    fixed_limits=None,
):
    """Export a compact 3-D-scene-only GIF without legends or rank panels."""
    output_path = FilePath(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6.4, 6.0), dpi=dpi, facecolor="white")
    ax3d = fig.add_axes([0.02, 0.04, 0.94, 0.93], projection="3d")
    if fixed_limits is None:
        fixed_limits = compute_global_axis_limits(records)

    pil_frames = []
    duration_ms = int(1000 / fps)
    total = len(frame_indices)

    for n, idx in enumerate(frame_indices, start=1):
        t = int(records[idx].get("time", idx))
        print(f"[Scene GIF] rendering frame {n}/{total}, t={t}s")
        draw_frame(
            ax3d,
            None,
            records,
            idx,
            tail_len=TAIL_LEN_DEFAULT,
            show_velocity=True,
            show_truth=False,
            show_legend=False,
            show_title=False,
        )
        apply_axis_limits(ax3d, fixed_limits)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        image = Image.fromarray(rgba).convert("RGB")
        pil_frames.append(image.convert("P", palette=Image.ADAPTIVE))

    if not pil_frames:
        plt.close(fig)
        raise RuntimeError("No frames are available for the scene-only GIF.")

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    plt.close(fig)
    print(f"Scene-only GIF saved to: {output_path.resolve()}")


def select_uniform_time_frames(records, start_time, end_time, max_frames):
    """Select uniformly distributed record indices, including both endpoints."""
    eligible = [
        idx
        for idx, record in enumerate(records)
        if start_time <= int(record.get("time", 0)) <= end_time
    ]
    if not eligible:
        raise ValueError(
            f"No visualization records found between {start_time}s and {end_time}s."
        )

    sample_count = min(max_frames, len(eligible))
    positions = np.linspace(0, len(eligible) - 1, sample_count, dtype=int)
    return [eligible[pos] for pos in np.unique(positions)]


def combine_stage_gifs(gif_paths, output_path, default_duration_ms=167):
    """Append stage GIFs while removing duplicated boundary frames."""
    output_path = FilePath(output_path)
    combined_frames = []
    frame_durations = []

    for gif_index, gif_path in enumerate(gif_paths):
        with Image.open(gif_path) as source:
            for frame_index in range(source.n_frames):
                if gif_index > 0 and frame_index == 0:
                    continue
                source.seek(frame_index)
                duration = int(source.info.get("duration", default_duration_ms))
                frame = source.convert("RGB").convert(
                    "P", palette=Image.ADAPTIVE
                )
                combined_frames.append(frame.copy())
                frame_durations.append(duration)

    if not combined_frames:
        raise RuntimeError("No frames are available for the combined formation GIF.")

    combined_frames[0].save(
        output_path,
        save_all=True,
        append_images=combined_frames[1:],
        duration=frame_durations,
        loop=0,
        optimize=False,
    )
    print(f"Combined formation GIF saved to: {output_path.resolve()}")
    return output_path


def export_formation_stage_gifs(records, output_dir, fps=6, dpi=100, max_frames=21):
    """Export the three presentation stages of scene 3 as scene-only GIFs."""
    stages = (
        (0, 140, "Formation_Stage1_CruiseWedge_000_140.gif"),
        (140, 340, "Formation_Stage2_WideSearch_140_340.gif"),
        (340, 600, "Formation_Stage3_ProtectiveFormation_340_600.gif"),
    )
    output_dir = FilePath(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_limits = compute_global_axis_limits(records)
    outputs = []

    for start_time, end_time, filename in stages:
        frame_indices = select_uniform_time_frames(
            records,
            start_time=start_time,
            end_time=end_time,
            max_frames=max_frames,
        )
        output_path = output_dir / filename
        export_scene_gif(
            records,
            output_path=output_path,
            frame_indices=frame_indices,
            fps=fps,
            dpi=dpi,
            fixed_limits=fixed_limits,
        )
        outputs.append(output_path)

    combined_path = combine_stage_gifs(
        outputs,
        output_dir / "Formation_All_Stages_000_600.gif",
        default_duration_ms=int(1000 / fps),
    )
    outputs.append(combined_path)
    return outputs


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
    ax3d = fig.add_axes([0.20, 0.14, 0.47, 0.80], projection="3d")
    ax_text = fig.add_axes([0.71, 0.15, 0.26, 0.79])

    draw_type_legend_box(fig, records[frame_indices[0]])
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
    ax3d = fig.add_axes([0.20, 0.14, 0.47, 0.80], projection="3d")
    ax_text = fig.add_axes([0.71, 0.15, 0.26, 0.79])

    draw_type_legend_box(fig, records[frame_indices[0]])
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
        "--data",
        type=str,
        default=str(DATA_PATH),
        help="visual_data.json 路径。默认 results_fig/visual_data.json。",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="场景编号。设置后自动读取 results_fig/<scenario>/visual_data.json。",
    )
    parser.add_argument(
        "--formation-stage-gifs",
        action="store_true",
        help="Scene 3 only: export three compact formation-stage GIFs without rank panels.",
    )
    parser.add_argument(
        "--stage-max-frames",
        type=int,
        default=21,
        help="Maximum frames in each formation-stage GIF. Default: 21.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="每隔多少秒/帧取一帧导出。默认 5。数值越小越流畅但越慢、文件越大。",
    )
    parser.add_argument(
        "--start-time",
        type=int,
        default=None,
        help="First simulation second to include in the exported animation.",
    )
    parser.add_argument(
        "--end-time",
        type=int,
        default=None,
        help="Last simulation second to include in the exported animation.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Uniformly sample at most this many frames from the requested time range.",
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

    data_path = FilePath("results_fig") / args.scenario / "visual_data.json" if args.scenario else FilePath(args.data)
    print(f"Loading visualization data from: {data_path.resolve()}")
    meta, records = load_data(data_path)
    display_types = freeze_enemy_display_types(records)
    print("Frozen display types:", ", ".join(f"{k}={v}" for k, v in sorted(display_types.items())))
    if args.step <= 0:
        raise ValueError("--step 必须大于 0")
    if args.stage_max_frames <= 0:
        raise ValueError("--stage-max-frames must be greater than 0")

    if args.formation_stage_gifs:
        experiment_id = str(meta.get("experiment_id", args.scenario or ""))
        formation_mode = str(meta.get("formation_mode", ""))
        if (
            experiment_id != "dynamic_formation"
            and formation_mode != "dynamic_homogeneous"
        ):
            raise ValueError(
                "--formation-stage-gifs requires scene 3 dynamic_formation data."
            )
        output_dir = data_path.parent / "formation_stage_gifs"
        outputs = export_formation_stage_gifs(
            records,
            output_dir=output_dir,
            fps=args.fps,
            dpi=args.dpi,
            max_frames=args.stage_max_frames,
        )
        print("\nThree scene-only formation GIFs have been generated:")
        for output_path in outputs:
            print(f"  -> {output_path.resolve()}")
        return

    if (
        args.start_time is not None
        and args.end_time is not None
        and args.start_time > args.end_time
    ):
        raise ValueError("--start-time must not be greater than --end-time")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be greater than 0")

    eligible_indices = [
        idx
        for idx, record in enumerate(records)
        if (args.start_time is None or int(record.get("time", 0)) >= args.start_time)
        and (args.end_time is None or int(record.get("time", 0)) <= args.end_time)
    ]
    if not eligible_indices:
        raise ValueError("No visualization records fall inside the requested time range")

    if args.max_frames is not None:
        sample_count = min(args.max_frames, len(eligible_indices))
        sample_positions = np.linspace(
            0,
            len(eligible_indices) - 1,
            num=sample_count,
            dtype=int,
        )
        frame_indices = [eligible_indices[pos] for pos in np.unique(sample_positions)]
    else:
        frame_indices = eligible_indices[::args.step]
        if frame_indices[-1] != eligible_indices[-1]:
            frame_indices.append(eligible_indices[-1])

    if args.output is None:
        suffix = "gif" if args.format == "gif" else "mp4"
        if args.start_time is None and args.end_time is None:
            range_suffix = ""
        else:
            first_time = int(records[eligible_indices[0]].get("time", 0))
            last_time = int(records[eligible_indices[-1]].get("time", 0))
            range_suffix = f"_{first_time:03d}_{last_time:03d}"
        args.output = str(
            data_path.parent / f"threat_assessment_animation{range_suffix}.{suffix}"
        )

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
