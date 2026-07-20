"""Scene-4-only result figures for healthy leader-wingman heterogeneity."""

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

ROLE_LABELS = ("F1\n长机", "F2\n左僚机", "F3\n右僚机", "F4\n后卫机")


def _set_style():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS"],
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
    })


def _shade_stages(ax):
    for start, end, label, color in STAGES:
        ax.axvspan(start, end, color=color, alpha=0.55, zorder=0)
        ax.text(
            (start + end) / 2,
            1.02,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7.6,
        )
    for boundary in (140, 260, 340, 460):
        ax.axvline(boundary, color="#999999", linestyle="--", linewidth=0.7)


def _stack(records, key):
    return np.stack([np.asarray(record[key], dtype=float) for record in records])


def _rank_positions(scores):
    order = np.argsort(-scores, axis=1)
    positions = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    positions[rows, order] = np.arange(1, scores.shape[1] + 1)
    return positions


def _spearman_from_positions(a, b):
    n = a.shape[1]
    if n <= 1:
        return np.ones(a.shape[0], dtype=float)
    squared_difference = np.sum((a - b) ** 2, axis=1)
    return 1.0 - 6.0 * squared_difference / (n * (n * n - 1.0))


def _components(record, target_idx, lambdas, tau, beta):
    pair = np.asarray(record["pair_scores"], dtype=float)[:, target_idx]
    weights = np.asarray(record["friendly_weights"], dtype=float)
    c_max = float(np.max(pair))
    c_avg = float(np.sum(weights * pair))
    c_soft = float(tau * np.log(np.sum(weights * np.exp(pair / tau)) + 1e-10))
    c_agg = float(
        lambdas[0] * c_max
        + lambdas[1] * c_avg
        + lambdas[2] * c_soft
    )
    c_form = float(np.asarray(record["form_scores"], dtype=float)[target_idx])
    c_total = float(beta * c_form + (1.0 - beta) * c_agg)

    recorded_agg = float(np.asarray(record["agg_scores"], dtype=float)[target_idx])
    recorded_total = float(np.asarray(record["scores"], dtype=float)[target_idx])
    if not np.isclose(c_agg, recorded_agg, atol=1e-9, rtol=1e-8):
        raise ValueError("Recomputed aggregation score does not match assessment record.")
    if not np.isclose(c_total, recorded_total, atol=1e-9, rtol=1e-8):
        raise ValueError("Recomputed total score does not match assessment record.")

    return np.array([c_max, c_avg, c_soft, c_agg, c_form, c_total])


def _save_control_consistency(metrics, output_path):
    fig, ax = plt.subplots(figsize=(7.0, 2.35), dpi=220)
    ax.axis("off")
    rows = [
        ["编队成员位置 MAE", f"{metrics['geometry_mae']:.3e}", "应接近 0"],
        ["单机局部威胁矩阵 MAE", f"{metrics['pair_mae']:.3e}", "应接近 0"],
        ["编队整体威胁分支 MAE", f"{metrics['form_mae']:.3e}", "应接近 0"],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["控制指标", "计算结果", "一致性要求"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.46, 0.26, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#8ba4bf")
        cell.set_linewidth(0.75)
        if row == 0:
            cell.set_facecolor("#0b438c")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f8fbff" if row % 2 else "white")
    ax.set_title("两组实验输入与未加权分支一致性验证", pad=8)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_weighted_contributions(
    hom_record,
    het_record,
    target_idx,
    time_value,
    output_path,
):
    hom_pair = np.asarray(hom_record["pair_scores"], dtype=float)[:, target_idx]
    het_pair = np.asarray(het_record["pair_scores"], dtype=float)[:, target_idx]
    hom_weights = np.asarray(hom_record["friendly_weights"], dtype=float)
    het_weights = np.asarray(het_record["friendly_weights"], dtype=float)
    hom_contribution = hom_weights * hom_pair
    het_contribution = het_weights * het_pair

    x = np.arange(len(hom_weights))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.3, 4.0), dpi=220)
    bars_hom = ax.bar(
        x - width / 2,
        hom_contribution,
        width,
        color="#2455a4",
        label="同构编队",
    )
    bars_het = ax.bar(
        x + width / 2,
        het_contribution,
        width,
        color="#ed8b00",
        label="异构编队",
    )
    for bars, values in (
        (bars_hom, hom_contribution),
        (bars_het, het_contribution),
    ):
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(ROLE_LABELS)
    ax.set_ylabel(r"加权平均分支贡献 $w_i C_{ij}$")
    ax.set_title(f"目标 T{target_idx + 1} 在 t={time_value:.0f} s 的成员加权贡献")
    ax.grid(axis="y", color="#dddddd", linewidth=0.65, alpha=0.8)
    ax.legend(frameon=False, loc="upper right")
    ax.set_axisbelow(True)
    ax.text(
        0.01,
        -0.22,
        "异构权重：F1=0.333，F2=0.208，F3=0.208，F4=0.250；"
        "柱值之和等于加权平均项。",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#174a8b",
    )
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_aggregation_branches(
    hom_values,
    het_values,
    target_idx,
    time_value,
    output_path,
):
    labels = (
        r"$C^{max}$ 最大项",
        r"$C^{avg}$ 加权平均",
        r"$C^{soft}$ 加权软最大",
        r"$S^{agg}$ 聚合分支",
        r"$S^{form}$ 编队整体分支",
        r"$S^{total}$ 综合威胁度",
    )
    y = np.arange(len(labels))
    delta = het_values - hom_values
    colors = [
        "#ed8b00" if value > 1e-12 else "#9ca3af"
        for value in delta
    ]
    fig, ax = plt.subplots(figsize=(7.7, 4.25), dpi=220)
    bars = ax.barh(y, delta, height=0.58, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    max_delta = max(float(np.max(np.abs(delta))), 1e-6)
    ax.set_xlim(-max_delta * 0.22, max_delta * 1.32)
    ax.axvline(0.0, color="#444444", linewidth=0.8)
    ax.set_xlabel(r"异构相对同构的变化量 $\Delta=Hetero-Homo$")
    ax.set_title(
        f"目标 T{target_idx + 1} 在 t={time_value:.0f} s 的异构影响传递",
        pad=10,
    )
    ax.grid(axis="x", color="#dddddd", linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, delta):
        x_text = value + max_delta * 0.025 if value >= 0 else value - max_delta * 0.025
        ax.text(
            x_text,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.4f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8.4,
        )
    ax.text(
        0.01,
        -0.18,
        rf"综合威胁度：{hom_values[-1]:.4f} $\rightarrow$ {het_values[-1]:.4f}；"
        r"$S^{total}=0.7S^{form}+0.3S^{agg}$。",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#174a8b",
    )
    fig.subplots_adjust(left=0.23, bottom=0.21)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_target_sensitivity(
    mean_abs_delta,
    max_abs_delta,
    selected_idx,
    output_path,
):
    target_count = len(mean_abs_delta)
    y = np.arange(target_count)
    height = 0.34
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=220)
    bars_mean = ax.barh(
        y + height / 2,
        mean_abs_delta,
        color="#2455a4",
        height=height,
        label="全时段平均绝对变化",
    )
    bars_max = ax.barh(
        y - height / 2,
        max_abs_delta,
        color="#ed8b00",
        height=height,
        label="最大瞬时绝对变化",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([f"T{i + 1}" for i in y])
    ax.invert_yaxis()
    ax.set_xlabel(r"综合威胁度绝对变化 $|\Delta S_j^{total}|$")
    ax.set_title("成员异构对各目标综合威胁度的响应幅度")
    ax.grid(axis="x", color="#dddddd", linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    max_value = max(float(np.max(max_abs_delta)), 1e-6)
    ax.set_xlim(0, max_value * 1.25)
    for bars, values in ((bars_mean, mean_abs_delta), (bars_max, max_abs_delta)):
        for bar, value in zip(bars, values):
            ax.text(
                value + max_value * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}",
                va="center",
                fontsize=7.7,
            )
    ax.get_yticklabels()[selected_idx].set_color("#d62828")
    ax.get_yticklabels()[selected_idx].set_weight("bold")
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_total_score_delta(times, score_delta, selected_idx, output_path):
    fig, ax = plt.subplots(figsize=(9.0, 3.7), dpi=220)
    _shade_stages(ax)
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, score_delta.shape[1]))
    for target_idx in range(score_delta.shape[1]):
        selected = target_idx == selected_idx
        ax.plot(
            times,
            score_delta[:, target_idx],
            color="#d62828" if selected else colors[target_idx],
            linewidth=2.0 if selected else 0.95,
            alpha=1.0 if selected else 0.72,
            label=f"T{target_idx + 1}",
            zorder=3 if selected else 2,
        )
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    max_value = max(float(np.max(np.abs(score_delta))), 1e-6)
    ax.set_ylim(-max_value * 1.18, max_value * 1.18)
    ax.set_xlim(float(times[0]), float(times[-1]))
    ax.set_xlabel("时间 t (s)")
    ax.set_ylabel(r"综合威胁度差值 $\Delta S_j^{total}$")
    ax.set_title("动态异构编队相对同构编队的逐时刻威胁度变化", pad=28)
    ax.grid(True, color="#dddddd", linewidth=0.65, alpha=0.72)
    ax.legend(frameon=False, ncol=4, loc="upper right")
    fig.subplots_adjust(top=0.76, bottom=0.18)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _stage_name(time_value):
    for start, end, label, _ in STAGES:
        if start <= time_value <= end:
            return label
    return ""


def _top_three(scores):
    return " > ".join(f"T{i + 1}" for i in np.argsort(scores)[::-1][:3])


def _save_rank_consistency(
    times,
    rho,
    hom_scores,
    het_scores,
    output_path,
):
    hom_positions = _rank_positions(hom_scores)
    het_positions = _rank_positions(het_scores)
    rank_delta = het_positions - hom_positions
    score_delta = het_scores - hom_scores

    event_indices = []
    candidates = [
        int(np.argmin(rho)),
        int(np.argmax(np.max(np.abs(score_delta), axis=1))),
    ]
    candidates.extend(int(idx) for idx in np.argsort(rho))
    for idx in candidates:
        if idx not in event_indices:
            event_indices.append(idx)
        if len(event_indices) == 3:
            break

    fig = plt.figure(figsize=(10.4, 4.3), dpi=220)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.35, 1.05), wspace=0.22)
    ax = fig.add_subplot(grid[0, 0])
    ax_table = fig.add_subplot(grid[0, 1])

    image = ax.imshow(
        rank_delta.T,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-2,
        vmax=2,
        extent=(float(times[0]), float(times[-1]), hom_scores.shape[1] + 0.5, 0.5),
    )
    ax.set_xlabel("时间 t (s)")
    ax.set_ylabel("敌方目标")
    ax.set_yticks(np.arange(1, hom_scores.shape[1] + 1))
    ax.set_yticklabels([f"T{i + 1}" for i in range(hom_scores.shape[1])])
    for boundary in (140, 260, 340, 460):
        ax.axvline(boundary, color="#333333", linestyle="--", linewidth=0.65)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("名次变化（负值表示优先级上升）", fontsize=8.5)

    rows = []
    for idx in sorted(event_indices):
        changed_targets = []
        for target_idx, difference in enumerate(rank_delta[idx]):
            if difference < 0:
                changed_targets.append(f"T{target_idx + 1}↑")
            elif difference > 0:
                changed_targets.append(f"T{target_idx + 1}↓")
        rows.append([
            f"{int(times[idx])} s",
            f"{rho[idx]:.3f}",
            _top_three(hom_scores[idx]),
            _top_three(het_scores[idx]),
            " ".join(changed_targets) or "无",
        ])
    ax_table.axis("off")
    table = ax_table.table(
        cellText=rows,
        colLabels=["时刻", r"$\rho$", "同构 Top-3", "异构 Top-3", "名次变化"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.15, 0.13, 0.25, 0.25, 0.23],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.1)
    table.scale(1.0, 1.62)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#8ba4bf")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#0b438c")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f8fbff" if row % 2 else "white")
    fig.text(
        0.29,
        0.96,
        f"目标名次变化分布（平均 ρ={np.mean(rho):.3f}，最小 ρ={np.min(rho):.3f}）",
        ha="center",
        va="top",
        fontsize=10.5,
    )
    fig.text(
        0.79,
        0.96,
        "排序变化关键时刻",
        ha="center",
        va="top",
        fontsize=10.5,
    )
    fig.subplots_adjust(top=0.83, bottom=0.14, left=0.07, right=0.98)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_heterogeneous_result_figures(
    homogeneous_records,
    heterogeneous_records,
    homogeneous_friendlies,
    heterogeneous_friendlies,
    save_dir,
    beta=0.7,
    lambdas=(0.5, 0.3, 0.2),
    tau=0.1,
):
    """Validate and save five independent scene-4 result figures."""
    if len(homogeneous_records) != len(heterogeneous_records):
        raise ValueError("Homogeneous and heterogeneous record lengths differ.")

    times_hom = np.array([record["time"] for record in homogeneous_records])
    times_het = np.array([record["time"] for record in heterogeneous_records])
    if not np.array_equal(times_hom, times_het):
        raise ValueError("Homogeneous and heterogeneous timelines differ.")

    hom_scores = _stack(homogeneous_records, "scores")
    het_scores = _stack(heterogeneous_records, "scores")
    hom_pair = _stack(homogeneous_records, "pair_scores")
    het_pair = _stack(heterogeneous_records, "pair_scores")
    hom_form = _stack(homogeneous_records, "form_scores")
    het_form = _stack(heterogeneous_records, "form_scores")
    if hom_scores.shape != het_scores.shape or hom_pair.shape != het_pair.shape:
        raise ValueError("Homogeneous and heterogeneous result shapes differ.")

    hom_positions = np.asarray([
        [[f["X"], f["Y"], f["Z"]] for f in step]
        for step in homogeneous_friendlies
    ], dtype=float)
    het_positions = np.asarray([
        [[f["X"], f["Y"], f["Z"]] for f in step]
        for step in heterogeneous_friendlies
    ], dtype=float)
    metrics = {
        "geometry_mae": float(np.mean(np.abs(hom_positions - het_positions))),
        "pair_mae": float(np.mean(np.abs(hom_pair - het_pair))),
        "form_mae": float(np.mean(np.abs(hom_form - het_form))),
    }

    score_delta = het_scores - hom_scores
    flat_index = int(np.argmax(np.abs(score_delta)))
    time_idx, target_idx = np.unravel_index(flat_index, score_delta.shape)
    time_value = float(times_hom[time_idx])
    mean_abs_delta = np.mean(np.abs(score_delta), axis=0)
    max_abs_delta_by_target = np.max(np.abs(score_delta), axis=0)
    mean_sensitive_idx = int(np.argmax(mean_abs_delta))
    max_sensitive_idx = int(np.argmax(max_abs_delta_by_target))

    hom_components = _components(
        homogeneous_records[time_idx], target_idx, lambdas, tau, beta
    )
    het_components = _components(
        heterogeneous_records[time_idx], target_idx, lambdas, tau, beta
    )

    hom_positions_rank = _rank_positions(hom_scores)
    het_positions_rank = _rank_positions(het_scores)
    rho = _spearman_from_positions(hom_positions_rank, het_positions_rank)

    _set_style()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "control": save_dir / "Figure_Heterogeneous_Control_Consistency.png",
        "contribution": save_dir / "Figure_Heterogeneous_Weighted_Contribution.png",
        "aggregation": save_dir / "Figure_Heterogeneous_Aggregation_Branches.png",
        "sensitivity": save_dir / "Figure_Heterogeneous_Target_Sensitivity.png",
        "delta_timeseries": save_dir / "Figure_Heterogeneous_TotalScore_Delta_TimeSeries.png",
        "ranking": save_dir / "Figure_Heterogeneous_Rank_Consistency.png",
    }

    _save_control_consistency(metrics, paths["control"])
    _save_weighted_contributions(
        homogeneous_records[time_idx],
        heterogeneous_records[time_idx],
        target_idx,
        time_value,
        paths["contribution"],
    )
    _save_aggregation_branches(
        hom_components,
        het_components,
        target_idx,
        time_value,
        paths["aggregation"],
    )
    _save_target_sensitivity(
        mean_abs_delta,
        max_abs_delta_by_target,
        max_sensitive_idx,
        paths["sensitivity"],
    )
    _save_total_score_delta(
        times_hom,
        score_delta,
        target_idx,
        paths["delta_timeseries"],
    )
    _save_rank_consistency(
        times_hom,
        rho,
        hom_scores,
        het_scores,
        paths["ranking"],
    )

    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "selected_target": f"T{target_idx + 1}",
        "selected_time": int(time_value),
        "mean_sensitive_target": f"T{mean_sensitive_idx + 1}",
        "max_abs_total_delta": float(abs(score_delta[time_idx, target_idx])),
        "mean_spearman": float(np.mean(rho)),
        "min_spearman": float(np.min(rho)),
        **metrics,
    }
