"""Scene-4-only validation plots for a healthy heterogeneous formation."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dynamics import build_pairwise_target


ROLE_IDS = ("F1", "F2", "F3", "F4")
ROLE_NAMES = ("F1 长机", "F2 左僚机", "F3 右僚机", "F4 后卫机")
ROLE_COLORS = ("#d73027", "#2878b5", "#2ca25f", "#7b61a8")
FORMATION_STAGES = (
    (0, 140, "巡航楔形", "#eaf2fb"),
    (140, 260, "第一次转换", "#fff4df"),
    (260, 340, "宽域搜索", "#eaf7f1"),
    (340, 460, "第二次转换", "#fff4df"),
    (460, 600, "防护队形", "#f1edfb"),
)


def _set_style():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS"],
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.0,
        "axes.linewidth": 0.9,
    })


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
    s_agg = float(
        lambdas[0] * c_max
        + lambdas[1] * c_avg
        + lambdas[2] * c_soft
    )
    s_form = float(np.asarray(record["form_scores"], dtype=float)[target_idx])
    s_total = float(beta * s_form + (1.0 - beta) * s_agg)

    if not np.isclose(
        s_agg,
        float(np.asarray(record["agg_scores"], dtype=float)[target_idx]),
        atol=1e-9,
        rtol=1e-8,
    ):
        raise ValueError("Recomputed aggregation score does not match record.")
    if not np.isclose(
        s_total,
        float(np.asarray(record["scores"], dtype=float)[target_idx]),
        atol=1e-9,
        rtol=1e-8,
    ):
        raise ValueError("Recomputed total score does not match record.")

    return {
        "C_max": c_max,
        "C_avg": c_avg,
        "C_soft": c_soft,
        "S_agg": s_agg,
        "S_form": s_form,
        "S_total": s_total,
    }


def _physical_reference_member(pair_features):
    """Choose the member reached first by the target's relative trajectory."""
    candidates = []
    for member_idx, feature in enumerate(pair_features):
        ttc = float(feature["TTC"])
        closing_speed = float(feature["ClosingSpeed"])
        if np.isfinite(ttc) and ttc < 1e5 and closing_speed > 1e-8:
            candidates.append(member_idx)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda idx: (
            float(pair_features[idx]["TTC"]),
            float(pair_features[idx]["Shortcut"]),
            float(pair_features[idx]["Distance"]),
        ),
    )


def _build_event_diagnostics(
    homogeneous_records,
    heterogeneous_records,
    target_series,
    friendly_series,
    lambdas,
    tau,
    beta,
    sample_step=5,
    start_time=75.0,
):
    rows = []
    times = np.asarray([record["time"] for record in heterogeneous_records], dtype=float)
    for record_idx in range(0, len(times), max(int(sample_step), 1)):
        time_value = float(times[record_idx])
        if time_value < start_time:
            continue
        hom_record = homogeneous_records[record_idx]
        het_record = heterogeneous_records[record_idx]
        n_targets = len(target_series[record_idx])
        for target_idx in range(n_targets):
            target_state = target_series[record_idx][target_idx]
            pair_features = [
                build_pairwise_target(target_state, friendly_state)
                for friendly_state in friendly_series[record_idx]
            ]
            physical_idx = _physical_reference_member(pair_features)
            if physical_idx is None:
                continue

            local_scores = np.asarray(het_record["pair_scores"], dtype=float)[:, target_idx]
            model_order = np.argsort(-local_scores)
            model_idx = int(model_order[0])
            second_idx = int(model_order[1]) if len(model_order) > 1 else model_idx
            hom_components = _components(hom_record, target_idx, lambdas, tau, beta)
            het_components = _components(het_record, target_idx, lambdas, tau, beta)
            reference_feature = pair_features[physical_idx]
            target_id = target_state.get("Target_ID", target_idx + 1)
            try:
                target_number = int(target_id)
                target_name = f"T{target_number}"
            except (TypeError, ValueError):
                target_name = str(target_id)

            row = {
                "record_idx": record_idx,
                "time": time_value,
                "target_idx": target_idx,
                "target": target_name,
                "target_type": str(target_state.get("Type", "Unknown")),
                "target_intent": str(target_state.get("IntentGT", "Unknown")),
                "physical_member_idx": physical_idx,
                "physical_member": ROLE_IDS[physical_idx],
                "model_member_idx": model_idx,
                "model_member": ROLE_IDS[model_idx],
                "top1_match": int(model_idx == physical_idx),
                "top2_match": int(physical_idx in model_order[:2]),
                "model_margin": float(local_scores[model_idx] - local_scores[second_idx]),
                "local_spread": float(np.max(local_scores) - np.min(local_scores)),
                "reference_distance_km": float(reference_feature["Distance"]),
                "reference_ttc_s": float(reference_feature["TTC"]),
                "reference_closing_speed_km_s": float(reference_feature["ClosingSpeed"]),
                "reference_shortcut_km": float(reference_feature["Shortcut"]),
            }
            for member_idx in range(4):
                row[f"C_F{member_idx + 1}"] = float(local_scores[member_idx])
                row[f"distance_F{member_idx + 1}_km"] = float(
                    pair_features[member_idx]["Distance"]
                )
                row[f"ttc_F{member_idx + 1}_s"] = float(pair_features[member_idx]["TTC"])
            for key in ("C_avg", "C_soft", "S_agg", "S_total"):
                row[f"hom_{key}"] = hom_components[key]
                row[f"het_{key}"] = het_components[key]
                row[f"delta_{key}"] = het_components[key] - hom_components[key]
            rows.append(row)
    if not rows:
        raise ValueError("No active target-member approach samples were found.")
    return rows


def _save_member_confusion(rows, output_path):
    matrix = np.zeros((4, 4), dtype=int)
    for row in rows:
        matrix[row["physical_member_idx"], row["model_member_idx"]] += 1
    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals > 0,
    ) * 100.0

    top1 = float(np.mean([row["top1_match"] for row in rows]))
    top2 = float(np.mean([row["top2_match"] for row in rows]))
    median_spread = float(np.median([row["local_spread"] for row in rows]))

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=220)
    image = ax.imshow(percentages, cmap="Blues", vmin=0.0, vmax=100.0)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(ROLE_IDS)
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(ROLE_NAMES)
    ax.set_xlabel("模型判定的主要受威胁成员")
    ax.set_ylabel("相对运动几何参考成员")
    ax.set_title("异构编队成员威胁识别混淆矩阵", pad=12)
    for row_idx in range(4):
        for col_idx in range(4):
            count = matrix[row_idx, col_idx]
            text_value = "-" if row_totals[row_idx, 0] == 0 else f"{percentages[row_idx, col_idx]:.1f}%\n(n={count})"
            ax.text(
                col_idx,
                row_idx,
                text_value,
                ha="center",
                va="center",
                fontsize=9.2,
                color="white" if percentages[row_idx, col_idx] >= 52 else "#1f1f1f",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
    colorbar.set_label("按几何参考成员归一化 / %")
    fig.text(
        0.5,
        0.02,
        f"有效接近样本 N={len(rows)}    Top-1一致率={top1:.1%}    "
        f"Top-2覆盖率={top2:.1%}    局部威胁中位极差={median_spread:.4f}",
        ha="center",
        fontsize=9.0,
        color="#174a8b",
    )
    fig.subplots_adjust(left=0.18, right=0.88, top=0.88, bottom=0.18)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return matrix, top1, top2, median_spread


def _save_correction_boxplot(rows, output_path):
    grouped = [
        np.asarray(
            [row["delta_S_agg"] * 1000.0 for row in rows if row["physical_member_idx"] == idx],
            dtype=float,
        )
        for idx in range(4)
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.5), dpi=220)
    for idx, values in enumerate(grouped, start=1):
        if values.size == 0:
            continue
        box = ax.boxplot(
            [values],
            positions=[idx],
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "white", "linewidth": 1.7},
            whiskerprops={"color": ROLE_COLORS[idx - 1]},
            capprops={"color": ROLE_COLORS[idx - 1]},
        )
        box["boxes"][0].set_facecolor(ROLE_COLORS[idx - 1])
        box["boxes"][0].set_alpha(0.86)
        jitter = np.linspace(-0.13, 0.13, min(values.size, 28))
        sampled = values[np.linspace(0, values.size - 1, len(jitter), dtype=int)]
        ax.scatter(idx + jitter, sampled, s=10, color="#303030", alpha=0.32, zorder=3)
        ax.text(
            idx,
            np.nanmax(values) + max(np.ptp(values), 0.001) * 0.08,
            f"n={values.size}\n中位数={np.median(values):+.3f}",
            ha="center",
            va="bottom",
            fontsize=8.2,
        )
    ax.axhline(0.0, color="#4d4d4d", linewidth=0.9, linestyle="--")
    ax.set_xticks(np.arange(1, 5))
    ax.set_xticklabels(ROLE_NAMES)
    ax.set_ylabel(r"异构修正量  $\Delta S^{agg}$  ($\times 10^{-3}$)")
    ax.set_title("不同受威胁成员下的异构聚合修正方向", pad=12)
    ax.grid(axis="y", color="#dddddd", linewidth=0.65, alpha=0.85)
    ax.set_axisbelow(True)
    fig.text(
        0.5,
        0.02,
        "正值表示异构任务价值使聚合威胁上调，负值表示下调；箱体反映全部有效接近样本，而非预设单点。",
        ha="center",
        fontsize=8.8,
        color="#174a8b",
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.87, bottom=0.20)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return grouped


def _select_key_event(rows):
    matched = [row for row in rows if row["top1_match"]]
    near_term = [
        row
        for row in matched
        if row["reference_distance_km"] <= 300.0
        and row["reference_ttc_s"] <= 300.0
    ]
    candidates = near_term or matched or rows
    return max(
        candidates,
        key=lambda row: (abs(row["delta_S_total"]), row["model_margin"]),
    )


def _save_key_event_chain(
    key_event,
    homogeneous_records,
    heterogeneous_records,
    output_path,
):
    record_idx = key_event["record_idx"]
    target_idx = key_event["target_idx"]
    hom_record = homogeneous_records[record_idx]
    het_record = heterogeneous_records[record_idx]
    local_scores = np.asarray(het_record["pair_scores"], dtype=float)[:, target_idx]
    hom_weights = np.asarray(hom_record["friendly_weights"], dtype=float)
    het_weights = np.asarray(het_record["friendly_weights"], dtype=float)
    contribution_delta = (het_weights - hom_weights) * local_scores
    propagation_names = (r"$\Delta C_{avg}$", r"$\Delta C_{soft}$", r"$\Delta S_{agg}$", r"$\Delta S_{total}$")
    propagation = np.asarray([
        key_event["delta_C_avg"],
        key_event["delta_C_soft"],
        key_event["delta_S_agg"],
        key_event["delta_S_total"],
    ])

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), dpi=220)
    x = np.arange(4)
    bars = axes[0].bar(x, local_scores, color=ROLE_COLORS, width=0.62)
    physical_idx = key_event["physical_member_idx"]
    bars[physical_idx].set_edgecolor("#111111")
    bars[physical_idx].set_linewidth(2.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ROLE_IDS)
    axes[0].set_ylim(0.0, 1.03)
    axes[0].set_ylabel("局部威胁度 $C_{ij}$")
    axes[0].set_title(
        f"1  单机局部威胁识别（参考/判定：{ROLE_IDS[physical_idx]}）"
    )
    for idx, value in enumerate(local_scores):
        axes[0].text(idx, value + 0.018, f"{value:.3f}", ha="center", fontsize=8.0)

    colors = ["#d73027" if value >= 0 else "#2878b5" for value in contribution_delta]
    axes[1].bar(x, contribution_delta * 1000.0, color=colors, width=0.62)
    axes[1].axhline(0.0, color="#444444", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ROLE_IDS)
    axes[1].set_ylabel(r"成员贡献变化 ($\times 10^{-3}$)")
    axes[1].set_title("2  异构任务价值加权")
    for idx, value in enumerate(contribution_delta * 1000.0):
        axes[1].text(idx, value, f"{value:+.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8.0)

    prop_colors = ["#2f60ad", "#4b8b3b", "#ed8b00", "#b2182b"]
    axes[2].bar(np.arange(4), propagation * 1000.0, color=prop_colors, width=0.62)
    axes[2].axhline(0.0, color="#444444", linewidth=0.8)
    axes[2].set_xticks(np.arange(4))
    axes[2].set_xticklabels(propagation_names)
    axes[2].set_ylabel(r"输出变化 ($\times 10^{-3}$)")
    axes[2].set_title("3  聚合链路传播")
    for idx, value in enumerate(propagation * 1000.0):
        axes[2].text(idx, value, f"{value:+.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8.0)

    for ax in axes:
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
    fig.suptitle(
        f"关键事件聚合链路：{key_event['target']}，t={key_event['time']:.0f} s",
        fontsize=13,
        y=0.99,
    )
    fig.text(
        0.5,
        0.02,
        f"参考成员距离={key_event['reference_distance_km']:.1f} km，"
        f"进入威胁边界时间={key_event['reference_ttc_s']:.1f} s；"
        "局部识别 → 角色加权 → 聚合分支 → 综合威胁，逐级展示异构信息如何进入结果。",
        ha="center",
        fontsize=8.8,
        color="#174a8b",
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.82, bottom=0.20, wspace=0.28)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _shade_stages(ax):
    for start, end, label, color in FORMATION_STAGES:
        ax.axvspan(start, end, color=color, alpha=0.72, zorder=0)
        ax.text(
            0.5 * (start + end),
            1.015,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8.0,
            color="#333333",
        )
    for boundary in (140, 260, 340, 460):
        ax.axvline(boundary, color="#a8a8a8", linestyle="--", linewidth=0.7)


def _save_rank_stability(times, hom_scores, het_scores, output_path):
    hom_positions = _rank_positions(hom_scores)
    het_positions = _rank_positions(het_scores)
    rho = _spearman_from_positions(hom_positions, het_positions)
    hom_order = np.argsort(-hom_scores, axis=1)
    het_order = np.argsort(-het_scores, axis=1)
    top1_retention = float(np.mean(hom_order[:, 0] == het_order[:, 0]))
    top3_retention = float(np.mean([
        set(hom_order[idx, :3]) == set(het_order[idx, :3])
        for idx in range(len(times))
    ]))
    delta = het_scores - hom_scores
    metrics = {
        "mean_spearman": float(np.mean(rho)),
        "min_spearman": float(np.min(rho)),
        "top1_retention": top1_retention,
        "top3_retention": top3_retention,
        "median_abs_score_delta": float(np.median(np.abs(delta))),
        "max_abs_score_delta": float(np.max(np.abs(delta))),
    }

    fig = plt.figure(figsize=(9.2, 4.4), dpi=220)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.85, 1.0), wspace=0.16)
    ax = fig.add_subplot(grid[0, 0])
    ax_info = fig.add_subplot(grid[0, 1])
    _shade_stages(ax)
    ax.plot(times, rho, color="#174a8b", linewidth=1.7)
    min_idx = int(np.argmin(rho))
    ax.scatter(times[min_idx], rho[min_idx], color="#d73027", s=28, zorder=4)
    ax.annotate(
        f"最低 ρ={rho[min_idx]:.3f}\nt={times[min_idx]:.0f} s",
        xy=(times[min_idx], rho[min_idx]),
        xytext=(12, -34),
        textcoords="offset points",
        fontsize=8.2,
        color="#b2182b",
        arrowprops={"arrowstyle": "->", "color": "#b2182b", "lw": 0.8},
    )
    ax.set_xlim(float(times[0]), float(times[-1]))
    ax.set_ylim(max(-0.05, float(np.min(rho)) - 0.08), 1.02)
    ax.set_xlabel("时间 t (s)")
    ax.set_ylabel("Spearman 排序相关系数 ρ")
    ax.set_title("异构修正前后的全目标排序一致性", pad=24)
    ax.grid(color="#dddddd", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)

    ax_info.axis("off")
    ax_info.set_xlim(0.0, 1.0)
    ax_info.set_ylim(0.0, 1.0)
    ax_info.set_title("排序稳定性指标", pad=12, color="#0b438c", fontweight="bold")
    labels = (
        ("平均 Spearman ρ", f"{metrics['mean_spearman']:.3f}"),
        ("最低 Spearman ρ", f"{metrics['min_spearman']:.3f}"),
        ("Top-1 保持率", f"{metrics['top1_retention']:.1%}"),
        ("Top-3 集合保持率", f"{metrics['top3_retention']:.1%}"),
        ("威胁度中位绝对变化", f"{metrics['median_abs_score_delta']:.4f}"),
        ("威胁度最大绝对变化", f"{metrics['max_abs_score_delta']:.4f}"),
    )
    y = 0.91
    for label, value in labels:
        ax_info.text(
            0.04, y, label, fontsize=9.0, color="#333333", va="center",
            transform=ax_info.transAxes,
        )
        ax_info.text(
            0.96, y, value, fontsize=10.4, color="#0b438c", va="center",
            ha="right", fontweight="bold", transform=ax_info.transAxes,
        )
        ax_info.plot([0.04, 0.96], [y - 0.07, y - 0.07], color="#d9e2ef", lw=0.8, transform=ax_info.transAxes)
        y -= 0.145
    fig.text(
        0.5,
        0.02,
        "高排序一致性表示异构信息对威胁度进行可解释的局部修正，同时保持全局排序的连续性与稳定性。",
        ha="center",
        fontsize=8.8,
        color="#174a8b",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.18)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return metrics


def _write_rows_csv(rows, output_path):
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(summary, output_path):
    with open(output_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value"))
        for key, value in summary.items():
            writer.writerow((key, value))


def save_heterogeneous_result_figures(
    homogeneous_records,
    heterogeneous_records,
    homogeneous_friendlies,
    heterogeneous_friendlies,
    target_series,
    save_dir,
    beta=0.7,
    lambdas=(0.5, 0.3, 0.2),
    tau=0.1,
):
    """Save four scene-4 validation figures and their traceable data tables."""
    if len(homogeneous_records) != len(heterogeneous_records):
        raise ValueError("Homogeneous and heterogeneous record lengths differ.")
    if len(target_series) != len(heterogeneous_records):
        raise ValueError("Target trajectory and assessment record lengths differ.")

    times_hom = np.asarray([record["time"] for record in homogeneous_records], dtype=float)
    times_het = np.asarray([record["time"] for record in heterogeneous_records], dtype=float)
    if not np.array_equal(times_hom, times_het):
        raise ValueError("Homogeneous and heterogeneous timelines differ.")

    hom_scores = _stack(homogeneous_records, "scores")
    het_scores = _stack(heterogeneous_records, "scores")
    hom_pair = _stack(homogeneous_records, "pair_scores")
    het_pair = _stack(heterogeneous_records, "pair_scores")
    hom_form = _stack(homogeneous_records, "form_scores")
    het_form = _stack(heterogeneous_records, "form_scores")
    hom_positions = np.asarray([
        [[f["X"], f["Y"], f["Z"]] for f in step]
        for step in homogeneous_friendlies
    ], dtype=float)
    het_positions = np.asarray([
        [[f["X"], f["Y"], f["Z"]] for f in step]
        for step in heterogeneous_friendlies
    ], dtype=float)
    control_metrics = {
        "geometry_mae": float(np.mean(np.abs(hom_positions - het_positions))),
        "pair_mae": float(np.mean(np.abs(hom_pair - het_pair))),
        "form_mae": float(np.mean(np.abs(hom_form - het_form))),
    }

    rows = _build_event_diagnostics(
        homogeneous_records,
        heterogeneous_records,
        target_series,
        heterogeneous_friendlies,
        lambdas,
        tau,
        beta,
    )
    key_event = _select_key_event(rows)

    _set_style()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "member_confusion": save_dir / "Figure_Heterogeneous_MemberRecognition.png",
        "correction_direction": save_dir / "Figure_Heterogeneous_CorrectionByRole.png",
        "key_event_chain": save_dir / "Figure_Heterogeneous_KeyEventChain.png",
        "rank_stability": save_dir / "Figure_Heterogeneous_RankStability.png",
        "event_data_csv": save_dir / "Table_Heterogeneous_EventDiagnostics.csv",
        "summary_csv": save_dir / "Table_Heterogeneous_ValidationSummary.csv",
    }

    _, top1, top2, median_spread = _save_member_confusion(rows, paths["member_confusion"])
    grouped = _save_correction_boxplot(rows, paths["correction_direction"])
    _save_key_event_chain(
        key_event,
        homogeneous_records,
        heterogeneous_records,
        paths["key_event_chain"],
    )
    rank_metrics = _save_rank_stability(times_hom, hom_scores, het_scores, paths["rank_stability"])

    role_medians = {
        f"delta_Sagg_median_{ROLE_IDS[idx]}": (
            float(np.median(values / 1000.0)) if values.size else float("nan")
        )
        for idx, values in enumerate(grouped)
    }
    summary = {
        "valid_approach_samples": len(rows),
        "member_top1_accuracy": top1,
        "member_top2_coverage": top2,
        "median_local_score_spread": median_spread,
        "key_event_target": key_event["target"],
        "key_event_time_s": key_event["time"],
        "key_event_physical_member": key_event["physical_member"],
        "key_event_model_member": key_event["model_member"],
        "key_event_delta_S_total": key_event["delta_S_total"],
        **role_medians,
        **rank_metrics,
        **control_metrics,
    }
    _write_rows_csv(rows, paths["event_data_csv"])
    _write_summary_csv(summary, paths["summary_csv"])

    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "valid_samples": len(rows),
        "top1_accuracy": top1,
        "top2_coverage": top2,
        "median_local_spread": median_spread,
        "key_event": {
            "target": key_event["target"],
            "time": int(key_event["time"]),
            "physical_member": key_event["physical_member"],
            "model_member": key_event["model_member"],
            "delta_total": key_event["delta_S_total"],
        },
        "role_delta_medians": role_medians,
        **rank_metrics,
        **control_metrics,
    }
