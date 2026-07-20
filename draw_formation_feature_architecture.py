from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Polygon


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results_fig"
OUTPUT_DIR.mkdir(exist_ok=True)

NAVY = "#073b82"
BLUE = "#0d5db8"
MID_BLUE = "#2f79c5"
LIGHT_BLUE = "#eaf3ff"
PALE_BLUE = "#f6faff"
RED = "#cf2027"
GREEN = "#18864b"
GRAY = "#555b66"
LIGHT_GRAY = "#f2f3f5"
TEXT = "#132238"


def configure_font():
    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    installed = {font_manager.FontProperties(fname=p).get_name() for p in font_manager.findSystemFonts()}
    for name in preferred:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "stix"


def rounded_box(ax, x, y, w, h, edge=BLUE, face="white", lw=1.5, radius=0.10):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.015,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(box)
    return box


def section_header(ax, x, y, w, number, title):
    rounded_box(ax, x, y, w, 0.43, edge=NAVY, face=NAVY, lw=0, radius=0.20)
    ax.text(
        x + w / 2, y + 0.215, f"{number}.  {title}",
        ha="center", va="center", color="white", fontsize=16, fontweight="bold",
    )


def fighter(ax, x, y, scale=0.23, color=BLUE, angle=0, edge="white", zorder=5):
    pts = [
        (0.00, 0.62), (0.10, 0.22), (0.43, 0.02), (0.40, -0.10),
        (0.10, -0.03), (0.13, -0.40), (0.00, -0.54), (-0.13, -0.40),
        (-0.10, -0.03), (-0.40, -0.10), (-0.43, 0.02), (-0.10, 0.22),
    ]
    import numpy as np

    p = np.asarray(pts) * scale
    a = np.deg2rad(angle)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    p = p @ rot.T
    p[:, 0] += x
    p[:, 1] += y
    ax.add_patch(Polygon(p, closed=True, facecolor=color, edgecolor=edge, linewidth=0.7, zorder=zorder))


def missile(ax, x, y, scale=0.24, color=RED, angle=0, zorder=5):
    pts = [
        (0.00, 0.70), (0.10, 0.40), (0.09, -0.15), (0.25, -0.36),
        (0.08, -0.30), (0.00, -0.58), (-0.08, -0.30), (-0.25, -0.36),
        (-0.09, -0.15), (-0.10, 0.40),
    ]
    import numpy as np

    p = np.asarray(pts) * scale
    a = np.deg2rad(angle)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    p = p @ rot.T
    p[:, 0] += x
    p[:, 1] += y
    ax.add_patch(Polygon(p, closed=True, facecolor=color, edgecolor="white", linewidth=0.7, zorder=zorder))


def drone(ax, x, y, scale=0.18, color=RED):
    ax.plot([x - scale, x + scale], [y, y], color=color, lw=2.2, zorder=5)
    ax.plot([x, x], [y - scale * 0.65, y + scale * 0.65], color=color, lw=2.2, zorder=5)
    for dx, dy in [(-scale, 0), (scale, 0), (0, -scale * 0.65), (0, scale * 0.65)]:
        ax.add_patch(Circle((x + dx, y + dy), scale * 0.22, facecolor="white", edgecolor=color, lw=1.3, zorder=5))


def flow_arrow(ax, x1, x2, y=4.75):
    ax.add_patch(FancyArrowPatch(
        (x1, y), (x2, y), arrowstyle="-|>", mutation_scale=25,
        linewidth=2.1, color=BLUE,
    ))


def bullet(ax, x, y, text, color=TEXT, size=11.2, weight="normal"):
    ax.add_patch(Circle((x, y + 0.015), 0.027, facecolor=BLUE, edgecolor="none"))
    ax.text(x + 0.10, y, text, ha="left", va="center", color=color, fontsize=size, fontweight=weight)


def draw():
    configure_font()
    fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Title band
    ax.add_patch(Polygon([(0, 8.34), (16, 8.34), (16, 9), (0, 9)], closed=True, facecolor=NAVY, edgecolor="none"))
    ax.text(
        0.35, 8.67,
        "研究内容二：编队威胁评估方法（1）—— 编队相对态势建模与多尺度特征构建",
        ha="left", va="center", color="white", fontsize=23, fontweight="bold",
    )

    # Main panel geometry
    y0, h = 1.36, 6.74
    x1, w1 = 0.16, 3.55
    x2, w2 = 4.03, 3.12
    x3, w3 = 7.47, 3.83
    x4, w4 = 11.62, 4.22

    for x, w in [(x1, w1), (x2, w2), (x3, w3), (x4, w4)]:
        rounded_box(ax, x, y0, w, h, edge=NAVY, face="white", lw=1.4, radius=0.16)

    section_header(ax, x1 + 0.17, 7.48, w1 - 0.34, "1", "敌我动态态势输入")
    section_header(ax, x2 + 0.17, 7.48, w2 - 0.34, "2", "编队状态计算")
    section_header(ax, x3 + 0.17, 7.48, w3 - 0.34, "3", "单机局部相对特征构建")
    section_header(ax, x4 + 0.17, 7.48, w4 - 0.34, "4", "编队整体相对特征构建")

    flow_arrow(ax, x1 + w1 + 0.04, x2 - 0.06)
    flow_arrow(ax, x2 + w2 + 0.04, x3 - 0.06)
    flow_arrow(ax, x3 + w3 + 0.04, x4 - 0.06)

    # Panel 1: enemy input
    rounded_box(ax, x1 + 0.18, 5.10, w1 - 0.36, 2.05, edge=BLUE, face=PALE_BLUE, lw=1.15)
    ax.text(x1 + 0.43, 6.88, "A. 敌方目标状态  $E_j$", color=RED, fontsize=14.5, fontweight="bold", va="center")
    fighter(ax, x1 + 0.78, 6.20, scale=0.42, color=RED, angle=-8)
    missile(ax, x1 + 1.70, 6.18, scale=0.36, color=RED, angle=-5)
    drone(ax, x1 + 2.60, 6.22, scale=0.22, color=RED)
    ax.text(x1 + 0.62, 5.58, r"位置 $p_j^E$，速度 $v_j^E$", fontsize=11.5, color=TEXT, va="center")
    ax.text(x1 + 0.62, 5.30, r"目标类型 $Type_j$，干扰强度 $G_j$", fontsize=11.5, color=TEXT, va="center")

    # Panel 1: friendly input
    rounded_box(ax, x1 + 0.18, 2.20, w1 - 0.36, 2.66, edge=BLUE, face=PALE_BLUE, lw=1.15)
    ax.text(x1 + 0.43, 4.59, "B. 我方编队状态  $F_i$", color=GREEN, fontsize=14.5, fontweight="bold", va="center")
    formation_pts = [(x1 + 0.74, 3.95), (x1 + 1.55, 4.24), (x1 + 2.31, 3.99), (x1 + 2.88, 4.32)]
    for a, b in zip(formation_pts, formation_pts[1:] + formation_pts[:1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=MID_BLUE, lw=1.0, ls=(0, (3, 2)))
    for px, py in formation_pts:
        fighter(ax, px, py, scale=0.30, color=BLUE)
    ax.text(x1 + 0.45, 3.43, r"位置 $p_i^F$，速度 $v_i^F$", fontsize=10.4, color=TEXT, va="center")
    ax.text(x1 + 0.45, 3.16, r"角色 $Role_i$，成员价值 $Value_i$", fontsize=10.4, color=TEXT, va="center")
    ax.text(x1 + 0.45, 2.89, r"易损性 $Vulnerability_i$", fontsize=9.8, color=TEXT, va="center")
    ax.text(x1 + 0.45, 2.64, r"机动性 $Maneuverability_i$", fontsize=9.8, color=TEXT, va="center")
    ax.text(
        x1 + 0.45, 2.34,
        "以上能力属性仅修正聚合权重，不作为 DBN 节点",
        fontsize=8.0, color=GRAY, va="center",
    )
    rounded_box(ax, x1 + 0.34, 1.51, w1 - 0.68, 0.54, edge="#b7c0ca", face=LIGHT_GRAY, lw=0.9)
    ax.text(
        x1 + w1 / 2, 1.78,
        "支持固定同构、动态同构、动态异构\n及成员能力退化编队模式",
        ha="center", va="center", fontsize=9.7, color=TEXT, fontweight="bold", linespacing=1.25,
    )

    # Panel 2: formation state calculation
    cx = x2 + w2 / 2
    cy = 5.76
    ax.add_patch(Circle((cx, cy), 1.18, fill=False, edgecolor=BLUE, lw=1.4, ls=(0, (4, 3))))
    ax.text(cx + 0.47, cy + 0.98, r"编队空间包络 $R_F$", color=BLUE, fontsize=10.5, fontweight="bold")
    member_pts = [(cx, cy + 0.63), (cx - 0.72, cy), (cx + 0.72, cy), (cx, cy - 0.65)]
    for px, py in member_pts:
        ax.plot([cx, px], [cy, py], color=MID_BLUE, lw=0.9, ls=(0, (3, 2)))
        fighter(ax, px, py, scale=0.30, color=BLUE)
    ax.add_patch(Circle((cx, cy), 0.045, facecolor="black", edgecolor="none", zorder=6))
    ax.text(cx + 0.11, cy - 0.13, r"$p_F$", fontsize=12.5, color=TEXT)
    ax.add_patch(FancyArrowPatch((cx, cy), (cx + 1.02, cy + 0.46), arrowstyle="->", mutation_scale=14, lw=1.2, color=TEXT))
    ax.text(cx + 0.79, cy + 0.58, r"$v_F$", fontsize=12.5, color=TEXT)

    rounded_box(ax, x2 + 0.36, 2.62, w2 - 0.72, 1.75, edge="#9aa6b2", face="white", lw=0.9)
    formulas = [
        r"$p_F=\frac{1}{N_F}\sum_i p_i^F$",
        r"$v_F=\frac{1}{N_F}\sum_i v_i^F$",
        r"$R_F=\max_i\,\Vert p_i^F-p_F\Vert$",
    ]
    for k, formula in enumerate(formulas):
        ax.text(cx, 4.05 - 0.53 * k, formula, ha="center", va="center", fontsize=14.0, color=TEXT)
    rounded_box(ax, x2 + 0.38, 1.66, w2 - 0.76, 0.58, edge="#b7c0ca", face=LIGHT_GRAY, lw=0.9)
    ax.text(cx, 1.95, "刻画编队中心、平均速度\n与当前空间尺度", ha="center", va="center", fontsize=9.5, color=TEXT, fontweight="bold", linespacing=1.2)

    # Panel 3: target-to-aircraft local relative features
    tx, ty = x3 + 0.84, 6.45
    fx, fy = x3 + 2.70, 5.28
    fighter(ax, tx, ty, scale=0.38, color=RED, angle=-55)
    fighter(ax, fx, fy, scale=0.40, color=BLUE, angle=0)
    ax.text(tx + 0.20, ty + 0.22, r"$E_j$", fontsize=15, color=TEXT)
    ax.text(fx + 0.24, fy + 0.22, r"$F_i$", fontsize=15, color=TEXT)
    ax.plot([tx + 0.14, fx - 0.16], [ty - 0.08, fy + 0.08], color=RED, lw=1.2, ls=(0, (2, 2)))
    ax.add_patch(FancyArrowPatch((tx + 0.18, ty - 0.12), (fx - 0.42, fy + 0.32), arrowstyle="->", mutation_scale=15, lw=1.6, color=GREEN))
    ax.text(x3 + 1.72, 5.90, r"$r_{ij}$", color=TEXT, fontsize=14)
    ax.text(x3 + 1.42, 5.52, r"$v_{ij}$", color=GREEN, fontsize=14)

    rounded_box(ax, x3 + 0.83, 4.36, w3 - 1.66, 0.76, edge="#9aa6b2", face="white", lw=0.9)
    ax.text(x3 + w3 / 2, 4.84, r"$r_{ij}=p_j^E-p_i^F$", ha="center", va="center", fontsize=14.5, color=TEXT)
    ax.text(x3 + w3 / 2, 4.52, r"$v_{ij}=v_j^E-v_i^F$", ha="center", va="center", fontsize=14.5, color=TEXT)
    ax.text(x3 + w3 / 2, 4.07, "局部（目标—单机）动态相对特征", ha="center", va="center", fontsize=13.5, color=BLUE, fontweight="bold")

    local_left = [r"相对距离 $D_{ij}$", r"径向接近速度 $V_{c,ij}$", r"接近角 $C_{ij}$", r"航路捷径 $S_{ij}$"]
    local_right = [r"相对高度差 $H_{ij}$", r"威胁边界时间 $TTC_{ij}$", r"目标类型 $Type_j$", r"干扰强度 $G_j$"]
    yy = 3.66
    for i in range(4):
        bullet(ax, x3 + 0.40, yy - i * 0.42, local_left[i], size=9.2)
        bullet(ax, x3 + 2.04, yy - i * 0.42, local_right[i], size=9.2)

    rounded_box(ax, x3 + 0.52, 1.57, w3 - 1.04, 0.58, edge=NAVY, face=NAVY, lw=0, radius=0.20)
    ax.text(x3 + w3 / 2, 1.86, r"输出：单机局部特征向量  $x_{ij}^{pair}$", ha="center", va="center", color="white", fontsize=11.2, fontweight="bold")

    # Panel 4: target-to-formation features
    c4x, c4y = x4 + 2.15, 5.96
    ax.add_patch(Circle((c4x, c4y), 1.02, fill=False, edgecolor=BLUE, lw=1.4, ls=(0, (4, 3))))
    ax.text(c4x + 0.53, c4y + 0.87, r"空间包络 $R_F$", color=BLUE, fontsize=10.2, fontweight="bold")
    target_x, target_y = x4 + 0.63, 6.79
    fighter(ax, target_x, target_y, scale=0.38, color=RED, angle=-55)
    ax.text(target_x + 0.20, target_y + 0.15, r"$E_j$", fontsize=14, color=TEXT)
    member4 = [(c4x, c4y + 0.52), (c4x - 0.66, c4y), (c4x + 0.66, c4y), (c4x, c4y - 0.55)]
    for px, py in member4:
        fighter(ax, px, py, scale=0.27, color=BLUE)
        ax.plot([c4x, px], [c4y, py], color=MID_BLUE, lw=0.8, ls=(0, (3, 2)))
    ax.add_patch(Circle((c4x, c4y), 0.043, facecolor="black", edgecolor="none", zorder=6))
    ax.text(c4x + 0.08, c4y - 0.12, r"$p_F$", fontsize=11.5)
    nearest = member4[1]
    ax.plot([target_x + 0.12, nearest[0] - 0.12], [target_y - 0.08, nearest[1] + 0.08], color=RED, lw=1.1, ls=(0, (2, 2)))
    ax.scatter([nearest[0] - 0.12], [nearest[1] + 0.08], s=28, marker="*", color=GREEN, zorder=7)
    ax.text(x4 + 0.72, 6.11, "最近成员", color=GREEN, fontsize=10.3, fontweight="bold")

    ax.text(x4 + w4 / 2, 4.64, "整体（目标—编队）动态相对特征", ha="center", va="center", fontsize=13.5, color=BLUE, fontweight="bold")
    form_left = [r"编队边界距离 $D_{boundary}$", r"最近成员距离 $D_{min}$", r"最小航路捷径 $S_{min}$", r"最小到达时间 $TTC_{min}$"]
    form_right = [r"最大接近速度 $V_{c,max}$", r"价值加权覆盖比例 $CoverRatio$", r"编队中心接近角 $C_j^F$", r"编队高度差 $H_j^F$"]
    yy = 4.22
    for i in range(4):
        bullet(ax, x4 + 0.35, yy - i * 0.43, form_left[i], size=8.8)
        bullet(ax, x4 + 2.20, yy - i * 0.43, form_right[i], size=8.8)

    rounded_box(ax, x4 + 0.62, 2.14, w4 - 1.24, 0.58, edge=NAVY, face=NAVY, lw=0, radius=0.20)
    ax.text(x4 + w4 / 2, 2.43, r"输出：编队整体特征向量  $x_j^{form}$", ha="center", va="center", color="white", fontsize=11.2, fontweight="bold")

    rounded_box(ax, x4 + 0.26, 1.48, w4 - 0.52, 0.54, edge="#aeb5bd", face=LIGHT_GRAY, lw=0.9)
    ax.text(
        x4 + w4 / 2, 1.75,
        "结构解释量：长机暴露度 / 僚机屏护度 / 队形风险因子\n（仅用于解释与可视化）",
        fontsize=8.9, color=TEXT, fontweight="bold", va="center", ha="center", linespacing=1.15,
    )

    # Bottom conclusion band
    rounded_box(ax, 0.18, 0.18, 15.64, 0.87, edge=NAVY, face="#f7fbff", lw=1.4, radius=0.12)
    ax.add_patch(Circle((0.72, 0.615), 0.23, facecolor=NAVY, edgecolor="none"))
    ax.text(0.72, 0.615, "i", ha="center", va="center", color="white", fontsize=18, fontweight="bold")
    ax.text(
        1.12, 0.615,
        "面向多机编队，将原始单目标威胁指标扩展为“目标—单机”与“目标—编队”两类动态相对威胁特征，"
        "为后续双分支 DBN–TOPSIS 融合评估提供输入。",
        ha="left", va="center", fontsize=14.3, color=NAVY, fontweight="bold",
    )

    png_path = OUTPUT_DIR / "formation_threat_feature_architecture.png"
    svg_path = OUTPUT_DIR / "formation_threat_feature_architecture.svg"
    fig.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    draw()
