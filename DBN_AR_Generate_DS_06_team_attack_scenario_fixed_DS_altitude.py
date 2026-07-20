import copy
import argparse
import os
import numpy as np
import pandas as pd

import plot_utils
from assessment_pipeline import (
    run_formation_dynamic_assessment,
    spearman_rank_correlation,
    summarize_multiple_missing,
    summarize_multiple_misidentification,
)
from dbn_topsis import DBN_TOPSIS_Fusion_Assessment
from dynamics import (
    FORMATION_MODE_DESCRIPTIONS,
    FRIENDLY_DEGRADATION_EVENT,
    generate_friendly_series,
)
from preprocessing import (
    apply_ar_imputation_to_multiple,
    introduce_multiple_missing_blocks,
)
from ds_assessment import introduce_multiple_misidentification
from scenario import (
    create_attack_targets,
    get_available_scenarios,
    get_missing_configs,
    get_misidentification_configs,
    get_scenario_debug_time,
    get_scenario_info,
    get_scenario_timeline,
)

EXPERIMENT_PRESETS = {
    "typical_fixed": {
        "scenario": "typical_attack",
        "formation_mode": "fixed_homogeneous",
        "name": "E1 Typical fixed homogeneous formation",
    },
    "scene_A_saturation": {
        "scenario": "saturation_attack",
        "formation_mode": "fixed_homogeneous",
        "name": "A High-intensity saturation breakthrough",
    },
    "scene_B_deception": {
        "scenario": "deception_interference",
        "formation_mode": "fixed_homogeneous",
        "name": "B Electronic suppression and deception breakthrough",
    },
    "scene_C_border": {
        "scenario": "border_patrol",
        "formation_mode": "fixed_homogeneous",
        "name": "C Border patrol with single-missile warning",
    },
    "dynamic_formation": {
        "scenario": "typical_attack",
        "formation_mode": "dynamic_homogeneous",
        "name": "E3 Dynamic homogeneous formation",
    },
    "dynamic_heterogeneous": {
        "scenario": "typical_attack",
        "formation_mode": "dynamic_heterogeneous",
        "name": "E4 Dynamic heterogeneous leader-wingman formation",
    },
    "altitude_missile": {
        "scenario": "altitude_missile_test",
        "formation_mode": "dynamic_heterogeneous",
        "name": "E5 Dynamic heterogeneous formation with high-low missile comparison",
    },
    "member_degradation": {
        "scenario": "typical_attack",
        "formation_mode": "dynamic_heterogeneous_degraded",
        "name": "E6 Dynamic threat assessment with F2 capability degradation",
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DBN-TOPSIS assessment for a configured big scenario.")
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENT_PRESETS),
        default=None,
        help="Paper experiment preset. Overrides --scenario and --formation-mode when set.",
    )
    parser.add_argument(
        "--scenario",
        choices=get_available_scenarios(),
        default="typical_attack",
        help="Big scenario id defined in scenario_catalog.py.",
    )
    parser.add_argument(
        "--formation-mode",
        choices=sorted(FORMATION_MODE_DESCRIPTIONS),
        default="fixed_homogeneous",
        help="Friendly formation mode for ablation experiments.",
    )
    parser.add_argument(
        "--output-root",
        default="results_fig",
        help="Root directory for generated figures, tables, and visual_data.json.",
    )
    parser.add_argument(
        "--visual-step",
        type=int,
        default=5,
        help="Seconds between records in visual_data.json. Use 1 for exact intent-switch frames.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show matplotlib windows after saving figures. By default figures are closed after export.",
    )
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="Scene 3 only: save only the geometry and structure-versus-center comparison figures.",
    )
    parser.add_argument(
        "--heterogeneous-results-only",
        action="store_true",
        help="Scene 4 only: save healthy heterogeneous leader-wingman result figures.",
    )
    args = parser.parse_args()
    if args.visual_step <= 0:
        parser.error("--visual-step must be greater than 0")

    if args.experiment:
        preset = EXPERIMENT_PRESETS[args.experiment]
        scenario_id = preset["scenario"]
        formation_mode = preset["formation_mode"]
        experiment_id = args.experiment
        experiment_name = preset["name"]
    else:
        scenario_id = args.scenario
        formation_mode = args.formation_mode
        experiment_id = f"{scenario_id}__{formation_mode}"
        experiment_name = f"{scenario_id} | {formation_mode}"

    if args.comparison_only and experiment_id != "dynamic_formation":
        parser.error("--comparison-only is only available with --experiment dynamic_formation")
    if args.heterogeneous_results_only and experiment_id != "dynamic_heterogeneous":
        parser.error(
            "--heterogeneous-results-only is only available with "
            "--experiment dynamic_heterogeneous"
        )

    scenario_info = get_scenario_info(scenario_id)
    save_dir = os.path.join(args.output_root, experiment_id)

    print(f"\n======== 当前大场景: {scenario_id} | {scenario_info.get('name', '')} ========")
    print(f"Experiment: {experiment_id} | {experiment_name}")
    print(f"Formation mode: {formation_mode} | {FORMATION_MODE_DESCRIPTIONS.get(formation_mode, '')}")
    print(scenario_info.get("description", ""))

    # 使用异构球坐标系模型初始化目标
    spherical_targets = create_attack_targets(scenario_id)
    num_targets = len(spherical_targets)
    print(spherical_targets)

    model = DBN_TOPSIS_Fusion_Assessment()
    
    # [步骤 1]: 运行物理动力学模型，生成 75s-600s 完整的 Ground Truth 轨迹
    print("正在运行异构目标动力学模型，生成逐秒飞行姿态数据 (75s - 600s)...")
    full_time_series = []
    for t in range(0, 601):
        step_data = []
        for target_model in spherical_targets:
            step_data.append(target_model.get_state(t))
            target_model.update(dt=1.0)
        full_time_series.append(step_data)

    # [新增步骤]: 生成我方带角色的动态异构编队时间序列
    print("\n正在生成我方带角色的动态异构编队队形数据...")
    friendly_series = generate_friendly_series(
        len(full_time_series),
        dt=1.0,
        formation_mode=formation_mode,
    )
    print(f"  -> 我方编队飞机数量: {len(friendly_series[0])}")
    if formation_mode == "dynamic_heterogeneous_degraded":
        event = FRIENDLY_DEGRADATION_EVENT
        print(
            "  -> Friendly capability event: "
            f"{event['label']} degrades progressively from "
            f"{event['start_time']:.0f}s to {event['end_time']:.0f}s; "
            f"final Vulnerability={event['final_vulnerability']:.2f}, "
            f"Maneuverability={event['final_maneuverability']:.2f}."
        )

    # [步骤 2]: 模拟复杂的复合电磁干扰场景 (实施“开局 EMP 致盲攻击”)
    print("\n[突发战况] 防空雷达遭遇敌方高强度、全频段 EMP 覆盖干扰：")
    missing_configs = get_missing_configs(scenario_id)

    inaccurate_time_series = copy.deepcopy(full_time_series)

    inaccurate_time_series = introduce_multiple_missing_blocks(inaccurate_time_series, missing_configs, start_time=0) 

    for cfg in missing_configs:
        t_name = spherical_targets[cfg['target_idx']].name
        print(f"  -> 目标 {t_name} 在 {cfg['start']}s - {cfg['end']}s 期间丢失 [{cfg['feature']}] 数据！")
    
    
    # [步骤 3]: 修改 inaccurate_time_series 中目标的类型
    print("\n[突发情况]己方传感器无法准确识别敌方目标类型：")
    # 修改 T1 和 T2 的类型为 "UAV"，修改 T6 的类型为 "Heli"
    misidentification_configs = get_misidentification_configs(scenario_id)

    inaccurate_time_series = introduce_multiple_misidentification(inaccurate_time_series, misidentification_configs, start_time=0)

    for cfg in misidentification_configs:
        t_name = spherical_targets[cfg['target_idx']].name
        t_type = spherical_targets[cfg['target_idx']].type
        print(f"  -> 目标 {t_name} 的 Type  [{t_type}] 在 {cfg['start']}s - {cfg['end']}s 期间被误识别为 [{cfg['misidentification']}]")

    # [步骤 4]: 注入 AR(p) 预测填补模块进行多维修复
    print("\n正在启动 AR(p) 算法阵列，对所有丢失数据进行并行自回归填补...")
    ar_time_series = copy.deepcopy(inaccurate_time_series)
    ar_time_series = apply_ar_imputation_to_multiple(ar_time_series, missing_configs)

    # [步骤 5]: 在三个平行宇宙（完整、无AR复合缺维、有AR修复）中同时执行态势评估
    print("\n======== 开始多管线 DBN-TOPSIS 复合对比评估 ========")
    if not (args.comparison_only or args.heterogeneous_results_only):
        full_records = run_formation_dynamic_assessment(
            model=model,
            time_series=full_time_series,
            friendly_series=friendly_series,
            start_time=0,
            use_DS=False,
            allow_missing=False,
            beta=0.7
        )

        A_records = run_formation_dynamic_assessment(
            model=model,
            time_series=inaccurate_time_series,
            friendly_series=friendly_series,
            start_time=0,
            use_DS=False,
            allow_missing=True,
            beta=0.7
        )

        B_records = run_formation_dynamic_assessment(
            model=model,
            time_series=ar_time_series,
            friendly_series=friendly_series,
            start_time=0,
            use_DS=False,
            allow_missing=False,
            beta=0.7
        )

        C_records = run_formation_dynamic_assessment(
            model=model,
            time_series=inaccurate_time_series,
            friendly_series=friendly_series,
            start_time=0,
            use_DS=True,
            allow_missing=True,
            beta=0.7,
            type_correction_configs=misidentification_configs
        )

    D_records = run_formation_dynamic_assessment(
        model=model,
        time_series=ar_time_series,
        friendly_series=friendly_series,
        start_time=0,
        use_DS=True,
        allow_missing=False,
        beta=0.7,
        type_correction_configs=misidentification_configs
    )

    # Scene 3 only: compare the structure-aware four-aircraft model with a
    # moving formation-center point under identical target and disturbance data.
    if experiment_id == "dynamic_formation":
        from dynamic_formation_comparison import (
            build_center_only_friendly_series,
            save_formation_geometry_timeseries,
            save_structure_vs_center_comparison,
        )

        print("\n======== 场景3：编队结构感知方法 vs 编队中心点基线 ========")
        geometry_png = save_formation_geometry_timeseries(
            friendly_series=friendly_series,
            save_dir=save_dir,
        )
        center_only_friendly_series = build_center_only_friendly_series(
            friendly_series
        )
        center_only_records = run_formation_dynamic_assessment(
            model=model,
            time_series=ar_time_series,
            friendly_series=center_only_friendly_series,
            start_time=0,
            use_DS=True,
            allow_missing=False,
            beta=0.7,
            type_correction_configs=misidentification_configs,
        )
        comparison_outputs = save_structure_vs_center_comparison(
            structure_records=D_records,
            center_records=center_only_records,
            save_dir=save_dir,
        )
        print(f"  -> Geometry figure: {geometry_png}")
        print(f"  -> 对比图: {comparison_outputs['png_path']}")
        print(
            "  -> 代表目标: "
            f"{comparison_outputs['representative_target']}; "
            f"最大 |ΔS|={comparison_outputs['max_abs_delta']:.4f} "
            f"({comparison_outputs['max_delta_target']}, "
            f"t={comparison_outputs['max_delta_time']}s)"
        )
        if args.comparison_only:
            print("  -> comparison-only 完成，跳过其余通用图表和可视化数据导出。")
            raise SystemExit(0)

    # Scene 4 only: compare identical dynamic geometry with equal and
    # leader-wingman task-value weights. Other experiments do not enter here.
    if experiment_id == "dynamic_heterogeneous" and args.heterogeneous_results_only:
        from dynamic_heterogeneous_results import save_heterogeneous_result_figures

        print("\n======== Scene 4: homogeneous vs heterogeneous formation results ========")
        homogeneous_friendly_series = generate_friendly_series(
            len(full_time_series),
            dt=1.0,
            formation_mode="dynamic_homogeneous",
        )
        homogeneous_records = run_formation_dynamic_assessment(
            model=model,
            time_series=ar_time_series,
            friendly_series=homogeneous_friendly_series,
            start_time=0,
            use_DS=True,
            allow_missing=False,
            beta=0.7,
            type_correction_configs=misidentification_configs,
        )
        heterogeneous_outputs = save_heterogeneous_result_figures(
            homogeneous_records=homogeneous_records,
            heterogeneous_records=D_records,
            homogeneous_friendlies=homogeneous_friendly_series,
            heterogeneous_friendlies=friendly_series,
            target_series=ar_time_series,
            save_dir=save_dir,
        )
        for name, output_path in heterogeneous_outputs["paths"].items():
            print(f"  -> {name}: {output_path}")
        print(
            "  -> member recognition: "
            f"N={heterogeneous_outputs['valid_samples']}, "
            f"Top-1={heterogeneous_outputs['top1_accuracy']:.1%}, "
            f"Top-2={heterogeneous_outputs['top2_coverage']:.1%}, "
            f"median local spread={heterogeneous_outputs['median_local_spread']:.4f}"
        )
        key_event = heterogeneous_outputs["key_event"]
        print(
            "  -> key event: "
            f"{key_event['target']} at t={key_event['time']}s, "
            f"physical={key_event['physical_member']}, "
            f"model={key_event['model_member']}, "
            f"delta total={key_event['delta_total']:+.6f}"
        )
        print(
            "  -> control MAE: "
            f"geometry={heterogeneous_outputs['geometry_mae']:.3e}, "
            f"pair={heterogeneous_outputs['pair_mae']:.3e}, "
            f"form={heterogeneous_outputs['form_mae']:.3e}"
        )
        print(
            "  -> rank Spearman: "
            f"mean={heterogeneous_outputs['mean_spearman']:.4f}, "
            f"min={heterogeneous_outputs['min_spearman']:.4f}, "
            f"Top-1 retention={heterogeneous_outputs['top1_retention']:.1%}, "
            f"Top-3 retention={heterogeneous_outputs['top3_retention']:.1%}"
        )
        print("  -> heterogeneous-results-only complete; generic exports skipped.")
        raise SystemExit(0)

    # ================= 以上是你代码的 步骤 5 =================

    # ================= 为论文核心表格直接生成排版数据 =================
    print("\n" + "★"*80)
    print(">>> 【表 3：本文融合框架在 75s (受欺骗干扰期间) 的目标状态概率与威胁度】 <<<")
    print(f"{'Target ID':<10} | {'P(H)':<8} | {'P(M)':<8} | {'P(L)':<8} | {'Threat Degree'}")
    print("-" * 65)
    target_time = 75
    for i in range(num_targets):
        p_h = D_records[target_time]['posteriors'][i][0]
        p_m = D_records[target_time]['posteriors'][i][1]
        p_l = D_records[target_time]['posteriors'][i][2]
        score = D_records[target_time]['scores'][i]
        print(f"T{i+1:<9} | {p_h:<8.4f} | {p_m:<8.4f} | {p_l:<8.4f} | {score:.4f}")

    debug_time = get_scenario_debug_time(scenario_id)

    print("\n" + "★"*80)
    print(f">>> 【{debug_time}s 时刻：敌方目标对编队内各单机的威胁度矩阵】 <<<")
    print("行：我方飞机 Friendly_Fighter_i；列：敌方目标 Tj")
    print("-" * 80)

    pair_mat = D_records[debug_time]["pair_scores"]

    header = "Aircraft".ljust(18) + " | " + " | ".join([f"T{j+1:^7}" for j in range(pair_mat.shape[1])])
    print(header)
    print("-" * len(header))

    for i in range(pair_mat.shape[0]):
        row = f"Friendly_Fighter_{i+1}".ljust(18) + " | "
        row += " | ".join([f"{pair_mat[i, j]:.4f}".center(7) for j in range(pair_mat.shape[1])])
        print(row)

    timeline_for_intent = get_scenario_timeline(scenario_id)
    switch_target_indices = []
    for j in range(num_targets):
        has_switch = any(
            t < len(full_time_series) and full_time_series[t][j].get("IntentSwitch", False)
            for t in timeline_for_intent
        )
        if has_switch:
            switch_target_indices.append(j)

    if switch_target_indices:
        print("\n" + "*" * 100)
        print(">>> [Intent switch timeline] <<<")
        print("Time | Target | ConfiguredIntent | Current IntentGT")
        print("-" * 70)
        for t in timeline_for_intent:
            if t >= len(full_time_series):
                continue
            for j in switch_target_indices:
                state = full_time_series[t][j]
                print(
                    f"{t:<4} | "
                    f"T{j + 1:<5} | "
                    f"{str(state.get('ConfiguredIntent', 'Unknown')):<16} | "
                    f"{str(state.get('IntentGT', 'Unknown'))}"
                )
        print("*" * 100 + "\n")

    print("\n>>> 单机视角下，每架飞机认为最危险的敌方目标：")
    for i in range(pair_mat.shape[0]):
        local_rank = np.argsort(pair_mat[i])[::-1] + 1
        print(f"Friendly_Fighter_{i+1}: " + " > ".join([f"T{r}" for r in local_rank]))

    print("★"*80 + "\n")


    debug_time = get_scenario_debug_time(scenario_id)

    print("\n" + "★"*80)
    print(f">>> 【{debug_time}s 时刻：编队分支、单机聚合分支与最终威胁度对比】 <<<")
    print(f"{'Target':<8} | {'Form Score':<12} | {'Agg Score':<12} | {'Total Score':<12}")
    print("-" * 60)

    form_scores = D_records[debug_time]["form_scores"]
    agg_scores = D_records[debug_time]["agg_scores"]
    total_scores = D_records[debug_time]["scores"]

    for j in range(len(total_scores)):
        print(f"T{j+1:<7} | {form_scores[j]:<12.4f} | {agg_scores[j]:<12.4f} | {total_scores[j]:<12.4f}")

    print("\n排序对比：")
    print("Form Rank : " + " > ".join([f"T{r}" for r in np.argsort(form_scores)[::-1] + 1]))
    print("Agg Rank  : " + " > ".join([f"T{r}" for r in np.argsort(agg_scores)[::-1] + 1]))
    print("Total Rank: " + " > ".join([f"T{r}" for r in np.argsort(total_scores)[::-1] + 1]))
    print("★"*80 + "\n")


    # ================= 编队视角五类意图后验概率检查 =================
    # 更稳妥：按 record["time"] 找记录，而不是直接用 D_records[90]
    debug_record = next(record for record in D_records if record["time"] == debug_time)

    print("\n" + "★"*100)
    print(f">>> 【{debug_time}s 时刻：编队视角五类意图后验概率 P(E)】 <<<")

    intent_names_cn = debug_record.get("intent_names_cn", [
        "攻击突防", "电子干扰", "侦察监视", "佯攻欺骗", "护航规避"
    ])

    header = (
        f"{'Target':<8} | "
        + " | ".join([f"{name:<10}" for name in intent_names_cn])
        + " | Pred Intent | GT"
    )
    print(header)
    print("-" * len(header))

    intent_probs = debug_record["form_intent_posteriors"]

    for j in range(intent_probs.shape[0]):
        probs = intent_probs[j]
        pred_idx = int(np.argmax(probs))
        pred_name = intent_names_cn[pred_idx]

        gt = debug_record["formation_targets"][j].get("IntentGT", "Unknown")

        row = f"T{j+1:<7} | " + " | ".join([f"{p:<10.4f}" for p in probs])
        row += f" | {pred_name:<10} | {gt}"
        print(row)

    print("★"*100 + "\n")


    # ================= 相对运动指标检查：验证编队指标是否真的按运动目标重新计算 =================
    print("\n" + "★"*80)
    print(f">>> 【{debug_time}s 时刻：相对运动指标检查】 <<<")
    print("Target | FormType | D_center | D_min | D_leader | Heading | S_min | TTC_min | LeaderExp | Shielding | FRF")
    print("-" * 90)
    print("\n" + "★"*80)
    print(f">>> 【{debug_time}s 时刻：相对运动指标检查】 <<<")
    print("Target | FormType | D_center | D_min | D_leader | Heading | S_min | TTC_min | LeaderExp | Shielding | FRF")
    print("-" * 90)

    for j, ft in enumerate(debug_record["formation_targets"]):
        print(
            f"T{j+1:<5} | "
            f"{str(ft.get('FormationType', '')):<9} | "
            f"{ft['D_center']:<8.2f} | "
            f"{ft['D_min']:<6.2f} | "
            f"{ft.get('D_leader', np.nan):<8.2f} | "
            f"{ft['Heading']:<7.2f} | "
            f"{ft['S_min']:<6.2f} | "
            f"{ft['TTC_min']:<8.2f} | "
            f"{ft.get('RelativeLeaderExposure', np.nan):<9.3f} | "
            f"{ft.get('Shielding', np.nan):<9.3f} | "
            f"{ft.get('FormationRiskFactor', np.nan):<6.3f}"
        )

    print("★"*80 + "\n")


    # ================= 单机威胁差异度检查：验证不同我方飞机的单机威胁是否被拉开 =================
    pair_mat = debug_record["pair_scores"]

    print("\n" + "★"*80)
    print(f">>> 【{debug_time}s 时刻：单机威胁差异度检查】 <<<")
    print("说明：max-min 越大，说明敌方目标对不同编队成员的威胁差异越明显。")
    print("-" * 80)

    for j in range(pair_mat.shape[1]):
        diff = np.max(pair_mat[:, j]) - np.min(pair_mat[:, j])
        print(f"T{j+1}: max-min = {diff:.6f}")

    print("★"*80 + "\n")


    # ================= D-S 类型空间融合诊断检查 =================
    timeline = get_scenario_timeline(scenario_id)
    debug_times = [
        t for t in timeline
        if any(cfg["start"] <= t <= cfg["end"] for cfg in misidentification_configs)
    ]
    if not debug_times:
        debug_times = [debug_time]

    print("\n" + "★"*100)
    print(">>> 【D-S 类型空间融合诊断检查】 <<<")
    print("Time | Target | SensorType | FusedType | K_type | K_after | Action")
    print("-" * 100)

    for ds_debug_time in debug_times:
        ds_debug_record = next(record for record in D_records if record["time"] == ds_debug_time)

        for j in range(num_targets):
            info = ds_debug_record.get("ds_k", {}).get(j, {})
            print(
                f"{ds_debug_time:<4} | "
                f"T{j+1:<5} | "
                f"{str(info.get('sensor_type', None)):<10} | "
                f"{str(info.get('fused_type', None)):<9} | "
                f"{info.get('K_type', np.nan):<7.4f} | "
                f"{info.get('K_type_after', np.nan):<7.4f} | "
                f"{str(info.get('ds_action', None))}"
            )

    print("★"*100 + "\n")


    print("\n" + "★"*100)
    print(">>> 【表 4：动态异构编队场景下异常干扰区间内的敌方目标整体威胁排序对比】 <<<")
    print(f"{'Time(s)':<8} | {'无干扰对照组 (真实排序)':<40} | {'传统 DBN-TOPSIS (无纠偏)':<40} | {'动态异构编队 AR-DS-DBN-TOPSIS (稳健排序)'}")
    print("-" * 130)
    
    # 定义需要打印的所有关键时刻
    timeline = get_scenario_timeline(scenario_id)
    
    for t in timeline:
        # 添加区间分割线，方便查看
        if t == 90:
            print(f"{'':<8} | {'--- 连续数据缺失区间 (80s - 140s) ---':<40} | {'':<40} |")
        elif t == 210:
            print(f"{'':<8} | {'--- 目标类型误识区间 (200s - 300s) ---':<40} | {'':<40} |")
        elif t == 375:
            print(f"{'':<8} | {'--- 干扰结束，常态演化区间 ---':<40} | {'':<40} |")
            
        # 提取三种情况的排序
        ref_rank  = " > ".join([f"T{r}" for r in full_records[t]['rank']]) # 真实的无干扰基准
        trad_rank = " > ".join([f"T{r}" for r in A_records[t]['rank']])    # 传统方案 (受干扰不修复)
        our_rank  = " > ".join([f"T{r}" for r in D_records[t]['rank']])    # 本文方案 (受干扰且修复)
        
        print(f"{t:<8} | {ref_rank:<40} | {trad_rank:<40} | {our_rank}")
        
    print("★"*100 + "\n")

    # ================= 计算 Spearman deviation：D_rho = 1 - rho =================
    # 这里直接使用上面表格打印的 timeline；若论文表格只保留部分时刻，可修改 spearman_times。
    spearman_times = timeline

    ref_by_time = {rec['time']: rec for rec in full_records}
    dbn_by_time = {rec['time']: rec for rec in A_records}
    ards_by_time = {rec['time']: rec for rec in D_records}

    spearman_deviation_rows = []
    for t in spearman_times:
        rho_dbn, _, _ = spearman_rank_correlation(ref_by_time[t]['rank'], dbn_by_time[t]['rank'])
        rho_ards, _, _ = spearman_rank_correlation(ref_by_time[t]['rank'], ards_by_time[t]['rank'])
        spearman_deviation_rows.append({
            'Time/s': t,
            'DBN-TOPSIS': round(1.0 - rho_dbn, 4),
            'AR-DS-DBN-TOPSIS': round(1.0 - rho_ards, 4)
        })

    spearman_deviation_df = pd.DataFrame(spearman_deviation_rows)

    print("\n" + "★"*80)
    print(">>> 【Spearman deviation 排序偏差逐时刻结果：D_rho = 1 - rho】 <<<")
    print(spearman_deviation_df.to_string(index=False))
    print("★"*80 + "\n")


    # ================= 接你原来的绘图代码 =================
    # 恢复这两行代码，计算用于画最后一张图的数据
    summary_df1 = summarize_multiple_missing(full_records, A_records, B_records, missing_configs)
    summary_df2 = summarize_multiple_misidentification(full_records, A_records, C_records, misidentification_configs)
    # 下面是你原来的画图代码...
    print("\n======== 开始生成论文图表 ========")
    import matplotlib.pyplot as plt
    # 1. 创建结果存放文件夹
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"已自动创建文件夹: {save_dir}/")

    # ================= 新增：导出三维可视化所需数据 =================
    # 说明：
    # 1) visual_data.json 给 Streamlit/Plotly 可视化界面使用；
    # 2) 每隔 VIS_STEP 秒保存一次，避免文件过大；
    # 3) 位置使用 AR 修复后的 ar_time_series，作为算法实际输入轨迹；
    # 4) 同时保留 full_time_series 中的真实轨迹字段，方便后续对比显示。
    import json

    VIS_STEP = args.visual_step
    D_by_time = {record["time"]: record for record in D_records}

    capability_rows = []
    for record in D_records:
        t = int(record["time"])
        weight_diagnostics = record.get("friendly_weight_diagnostics", [])
        for i, friendly in enumerate(friendly_series[t]):
            diagnostic = weight_diagnostics[i] if i < len(weight_diagnostics) else {}
            capability_rows.append({
                "Time": t,
                "Aircraft": f"F{i + 1}",
                "Role": friendly.get("Role", ""),
                "CapabilityState": friendly.get("CapabilityState", "Healthy"),
                "DamageLevel": friendly.get("DamageLevel", 0.0),
                "Value": friendly.get("Value", 1.0),
                "Vulnerability": friendly.get("Vulnerability", 1.0),
                "BaselineVulnerability": friendly.get("BaselineVulnerability", 1.0),
                "Maneuverability": friendly.get("Maneuverability", 1.0),
                "BaselineManeuverability": friendly.get("BaselineManeuverability", 1.0),
                "DegradationFactor": diagnostic.get("degradation_factor", 0.0),
                "AggregationWeight": record["friendly_weights"][i],
            })

    capability_csv_path = os.path.join(save_dir, "Table_Friendly_Capability_Weights.csv")
    pd.DataFrame(capability_rows).to_csv(
        capability_csv_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Friendly capability and aggregation weights saved to: {capability_csv_path}")

    def _safe_float(x):
        try:
            if x is None:
                return None
            x = float(x)
            if np.isnan(x) or np.isinf(x):
                return None
            return x
        except Exception:
            return None

    def _safe_list(arr):
        return [_safe_float(v) for v in np.asarray(arr, dtype=float).reshape(-1)]

    def _rank_to_names(rank):
        return [f"T{int(r)}" for r in np.asarray(rank, dtype=int).reshape(-1)]

    def _active_configs(configs, t):
        return [
            {
                "target_idx": int(cfg.get("target_idx", -1)),
                "target": f"T{int(cfg.get('target_idx', -1)) + 1}",
                "target_scene": cfg.get("target_scene", None),
                "feature": cfg.get("feature"),
                "start": int(cfg.get("start", -1)),
                "end": int(cfg.get("end", -1)),
                "misidentification": cfg.get("misidentification", None),
            }
            for cfg in configs
            if int(cfg.get("start", 10**9)) <= t <= int(cfg.get("end", -10**9))
        ]

    visual_records = []

    for t in range(0, len(ar_time_series), VIS_STEP):
        if t not in D_by_time:
            continue

        rec = D_by_time[t]
        enemy_states = ar_time_series[t]       # 算法实际看到/修复后的状态
        enemy_truth = full_time_series[t]      # 无干扰真实状态
        friendlies = friendly_series[t]

        total_scores = np.asarray(rec["scores"], dtype=float)
        form_scores_v = np.asarray(rec["form_scores"], dtype=float)
        agg_scores_v = np.asarray(rec["agg_scores"], dtype=float)
        pair_scores_v = np.asarray(rec["pair_scores"], dtype=float)
        friendly_weights_v = np.asarray(rec.get("friendly_weights", []), dtype=float)
        friendly_weight_diagnostics_v = rec.get("friendly_weight_diagnostics", [])
        formation_targets_v = rec.get("formation_targets", [])

        enemies_vis = []
        for j, enemy in enumerate(enemy_states):
            truth = enemy_truth[j]
            ds_info = rec.get("ds_k", {}).get(j, {})
            intent_probs = np.asarray(rec["form_intent_posteriors"][j], dtype=float)
            intent_names = rec.get("intent_names", [])
            predicted_intent = (
                intent_names[int(np.argmax(intent_probs))]
                if len(intent_names) == len(intent_probs)
                else "Unknown"
            )

            enemies_vis.append({
                "id": int(j + 1),
                "label": f"T{j + 1}",
                "name": enemy.get("Name", f"T{j + 1}"),
                "small_scene_id": enemy.get("SmallSceneID", None),
                "small_scene_label": enemy.get("SmallSceneLabel", None),
                "threat_level_gt": enemy.get("ThreatLevelGT", None),
                "recommended_decision": enemy.get("RecommendedDecision", None),
                "decision_reason": enemy.get("DecisionReason", None),
                "core_features": enemy.get("CoreFeatures", None),
                "attack_role": enemy.get("AttackRole", None),
                "sensor_type": enemy.get("Type", None),
                "true_type": truth.get("Type", None),
                "fused_type": ds_info.get("fused_type", enemy.get("Type", None)),
                "ds_action": ds_info.get("ds_action", "None"),
                "intent_gt": truth.get("IntentGT", enemy.get("IntentGT", None)),
                "configured_intent": truth.get("ConfiguredIntent", enemy.get("ConfiguredIntent", None)),
                "intent_switch": bool(truth.get("IntentSwitch", enemy.get("IntentSwitch", False))),
                "intent_phase": truth.get("IntentPhase", enemy.get("IntentPhase", None)),
                "turn_progress": _safe_float(truth.get("TurnProgress", enemy.get("TurnProgress", 0.0))),
                "intent_switch_time": _safe_float(truth.get("IntentSwitchTime", enemy.get("IntentSwitchTime", None))),
                "predicted_intent": predicted_intent,
                "intent_probabilities": {
                    name: _safe_float(probability)
                    for name, probability in zip(intent_names, intent_probs)
                },
                "x": _safe_float(enemy.get("X")),
                "y": _safe_float(enemy.get("Y")),
                "z": _safe_float(enemy.get("Z")),
                "x_true": _safe_float(truth.get("X")),
                "y_true": _safe_float(truth.get("Y")),
                "z_true": _safe_float(truth.get("Z")),
                "vx": _safe_float(enemy.get("VX")),
                "vy": _safe_float(enemy.get("VY")),
                "vz": _safe_float(enemy.get("VZ")),
                "total_score": _safe_float(total_scores[j]),
                "form_score": _safe_float(form_scores_v[j]),
                "agg_score": _safe_float(agg_scores_v[j]),
                "formation_risk_factor": _safe_float(formation_targets_v[j].get("FormationRiskFactor")) if j < len(formation_targets_v) else None,
                "leader_exposure": _safe_float(formation_targets_v[j].get("RelativeLeaderExposure")) if j < len(formation_targets_v) else None,
                "shielding": _safe_float(formation_targets_v[j].get("Shielding")) if j < len(formation_targets_v) else None,
                "d_leader": _safe_float(formation_targets_v[j].get("D_leader")) if j < len(formation_targets_v) else None,
                "formation_closing_speed": _safe_float(formation_targets_v[j].get("VC_form")) if j < len(formation_targets_v) else None,
            })

        friendlies_vis = []
        for i, f in enumerate(friendlies):
            friendlies_vis.append({
                "id": int(i + 1),
                "label": f"F{i + 1}",
                "name": f.get("Name", f"Friendly_Fighter_{i + 1}"),
                "role": f.get("Role", ""),
                "aircraft_type": f.get("AircraftType", ""),
                "value": _safe_float(f.get("Value", 1.0)),
                "vulnerability": _safe_float(f.get("Vulnerability", 1.0)),
                "maneuverability": _safe_float(f.get("Maneuverability", 1.0)),
                "baseline_vulnerability": _safe_float(
                    f.get("BaselineVulnerability", f.get("Vulnerability", 1.0))
                ),
                "baseline_maneuverability": _safe_float(
                    f.get("BaselineManeuverability", f.get("Maneuverability", 1.0))
                ),
                "capability_state": f.get("CapabilityState", "Healthy"),
                "damage_level": _safe_float(f.get("DamageLevel", 0.0)),
                "capability_event": f.get("CapabilityEvent", ""),
                "aggregation_weight": (
                    _safe_float(friendly_weights_v[i])
                    if i < len(friendly_weights_v)
                    else None
                ),
                "degradation_factor": (
                    _safe_float(friendly_weight_diagnostics_v[i].get("degradation_factor"))
                    if i < len(friendly_weight_diagnostics_v)
                    else None
                ),
                "sensor_range": _safe_float(f.get("SensorRange", None)),
                "weapon_range": _safe_float(f.get("WeaponRange", None)),
                "ecm_capability": _safe_float(f.get("ECMCapability", None)),
                "formation_type": f.get("FormationType", ""),
                "x": _safe_float(f.get("X")),
                "y": _safe_float(f.get("Y")),
                "z": _safe_float(f.get("Z")),
                "vx": _safe_float(f.get("VX")),
                "vy": _safe_float(f.get("VY")),
                "vz": _safe_float(f.get("VZ")),
                "local_rank": _rank_to_names(np.argsort(pair_scores_v[i])[::-1] + 1),
                "local_scores": _safe_list(pair_scores_v[i]),
            })

        visual_records.append({
            "time": int(t),
            "formation_type": friendlies[0].get("FormationType", "Unknown") if friendlies else "Unknown",
            "friendlies": friendlies_vis,
            "enemies": enemies_vis,
            "total_rank": _rank_to_names(rec["rank"]),
            "form_rank": _rank_to_names(np.argsort(form_scores_v)[::-1] + 1),
            "agg_rank": _rank_to_names(np.argsort(agg_scores_v)[::-1] + 1),
            "total_scores": _safe_list(total_scores),
            "form_scores": _safe_list(form_scores_v),
            "agg_scores": _safe_list(agg_scores_v),
            "pair_scores": [[_safe_float(v) for v in row] for row in pair_scores_v],
            "events": {
                "emp_active": len(_active_configs(missing_configs, t)) > 0,
                "misid_active": len(_active_configs(misidentification_configs, t)) > 0,
                "friendly_degradation_active": any(
                    float(f.get("DamageLevel", 0.0)) > 0.0 for f in friendlies
                ),
                "missing_configs": _active_configs(missing_configs, t),
                "misidentification_configs": _active_configs(misidentification_configs, t),
            }
        })

    visual_payload = {
        "meta": {
            "description": "3D formation threat-assessment visualization data",
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "scenario_id": scenario_id,
            "scenario_name": scenario_info.get("name", ""),
            "scenario_description": scenario_info.get("description", ""),
            "small_scenes": scenario_info.get("small_scenes", []),
            "formation_mode": formation_mode,
            "formation_mode_description": FORMATION_MODE_DESCRIPTIONS.get(formation_mode, ""),
            "friendly_degradation_event": (
                FRIENDLY_DEGRADATION_EVENT
                if formation_mode == "dynamic_heterogeneous_degraded"
                else None
            ),
            "time_step_seconds": VIS_STEP,
            "num_records": len(visual_records),
            "num_friendlies": len(friendly_series[0]),
            "num_enemies": len(ar_time_series[0]),
        },
        "records": visual_records,
    }

    visual_json_path = os.path.join(save_dir, "visual_data.json")
    with open(visual_json_path, "w", encoding="utf-8") as f:
        json.dump(visual_payload, f, ensure_ascii=False, indent=2)

    print(f"三维可视化数据已保存到 {visual_json_path}")
    print(f"可视化导出命令: python .\\visualizer_3d_matplotlib_export_animation.py --scenario {experiment_id}")
    # ============================================================

    # 保存 Spearman deviation 逐时刻结果。
    spearman_deviation_df.to_csv(os.path.join(save_dir, 'Table_Spearman_Deviation.csv'), index=False, encoding='utf-8-sig')
    print(f"Spearman deviation 逐时刻结果已保存到 {save_dir}/Table_Spearman_Deviation.csv")

    # 提取有 DS 修复的实验 C 的 K 值记录
    # 把 C_records 换成经过 AR 修复的 D_records
    k_records = {r['time']: r['ds_k'] for r in D_records if 'ds_k' in r}

    # 2. 依次绘图并保存到指定文件夹
    print("正在绘制并保存图表...")
    missing_target_indices = sorted({cfg["target_idx"] for cfg in missing_configs})
    misid_target_indices = sorted({cfg["target_idx"] for cfg in misidentification_configs})
    plot_missing_targets = missing_target_indices[:2] if missing_target_indices else [0]
    ar_plot_target_idx = missing_target_indices[0] if missing_target_indices else 0
    ds_plot_target_idx = misid_target_indices[0] if misid_target_indices else 0
    
    # 【修改 2】强制对齐时间轴，解决 Figure 2 错位问题
    plot_utils.plot_emp_interference(full_time_series, target_indices=plot_missing_targets, feature='X', missing_configs=missing_configs, start_time=0)
    plt.savefig(os.path.join(save_dir, 'Figure1_EMP_Interference.png'), dpi=300, bbox_inches='tight')
    
    # 【修改 1】传入受损轨迹 A宇宙，预测轨迹 B宇宙
    plot_utils.plot_ar_imputation(full_time_series, inaccurate_time_series, ar_time_series, target_idx=ar_plot_target_idx, feature='X', missing_configs=missing_configs, start_time=0)
    plt.savefig(os.path.join(save_dir, 'Figure2_AR_Imputation.png'), dpi=300, bbox_inches='tight')
    
    plot_utils.plot_ds_conflict_diagnosis(k_records, target_idx=ds_plot_target_idx, spoof_configs=misidentification_configs)
    plt.savefig(os.path.join(save_dir, 'Figure3_DS_Diagnosis.png'), dpi=300, bbox_inches='tight')
    
    plot_utils.plot_threat_scores_baseline(full_records)
    plt.savefig(os.path.join(save_dir, 'Figure4_Baseline_Threat.png'), dpi=300, bbox_inches='tight')
    
    # 把 A 和 C 换成真正使用了 AR 填补的 B 和 D
    plot_utils.plot_ds_restoration(full_records, B_records, D_records, target_idx=ds_plot_target_idx, spoof_configs=misidentification_configs)
    plt.savefig(os.path.join(save_dir, 'Figure5_DS_Restoration.png'), dpi=300, bbox_inches='tight')
    
    plot_utils.plot_performance_metrics(summary_df1, summary_df2)
    plt.savefig(os.path.join(save_dir, 'Figure6_Performance_Metrics.png'), dpi=300, bbox_inches='tight')

    print(f"所有高清图表已自动保存到 {save_dir}/ 文件夹下！")
    
    # 最后一次性展示所有窗口 (如果不想在屏幕上弹出，只需注释掉下面这行即可)
    if args.show_plots:
        plt.show()
    else:
        plt.close('all')
