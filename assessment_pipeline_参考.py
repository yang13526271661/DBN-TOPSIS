import numpy as np
import pandas as pd

from ds_assessment import DS_Assessment
from dynamics import (
    build_formation_target,
    build_pairwise_target,
    normalize_scores,
    topsis_closeness,
)

# ================= 5. 评估流程控制器 =================
def run_dynamic_assessment(model, time_series, start_time=0, use_DS=False, allow_missing=False):
    priors = np.tile(model.prior_L, (len(time_series[0]), 1))
    records = []
    ds_k_records = {} # 核心修改：记录K值用于画图

    for t_idx, current_targets in enumerate(time_series):
        current_time = start_time + t_idx
        posteriors = np.zeros((len(current_targets), model.states_L))
        ds_k_records[current_time] = {}

        for i, target in enumerate(current_targets):
            evidence = model.fuzzify_target_data(target, allow_missing=allow_missing)

            if use_DS:
                DS = DS_Assessment()
                # 接收三个返回值
                evidence, k_tot, k_diag = DS.conflict_check(model=model, 
                                                            ev=evidence, 
                                                            target=target, 
                                                            diagnostic_item='Type', # 注意这里修正为 'Type'
                                                            K_diff_th=0.05,
                                                            drift_method=True)
                ds_k_records[current_time][i] = {'K_total': k_tot, 'K_diag': k_diag}

            posteriors[i] = model.bayesian_inference(evidence, priors[i])

        scores = model.topsis_evaluation(posteriors)
        rank = np.argsort(scores)[::-1] + 1

        record_entry = {
            'time': current_time,
            'scores': scores.copy(),
            'posteriors': posteriors.copy(),
            'rank': rank.copy()
        }
        if use_DS:
            record_entry['ds_k'] = ds_k_records[current_time] # 存入记录中
            
        records.append(record_entry)

        for i in range(len(current_targets)):
            priors[i] = np.dot(posteriors[i], model.transition_matrix)

    return records

def run_formation_dynamic_assessment(
    model,
    time_series,
    friendly_series,
    start_time=0,
    use_DS=True,
    allow_missing=True,
    beta=0.7,
    lambdas=(0.5, 0.3, 0.2),
    tau=0.1
):
    """
    同构飞机编队威胁评估主流程。

    输出仍然保留 scores/posteriors/rank 三个字段，
    这样你后面的表格、Spearman、绘图代码可以少改。
    """

    ds_model = DS_Assessment() if use_DS else None

    num_steps = len(time_series)
    num_enemy = len(time_series[0])
    num_friendly = len(friendly_series[0])

    pair_priors = np.tile(model.prior_L, (num_friendly, num_enemy, 1))
    form_priors = np.tile(model.prior_L, (num_enemy, 1))

    records = []

    for t_idx in range(num_steps):
        current_time = start_time + t_idx
        enemies = time_series[t_idx]
        friendlies = friendly_series[t_idx]

        fixed_id_evidences = {}
        ds_k_info = {}

        # 每个敌方目标只做一次 D-S 类型空间融合修正
        for j, enemy in enumerate(enemies):
            formation_target = build_formation_target(enemy, friendlies)
            form_ev = model.fuzzify_target_data(formation_target, allow_missing=allow_missing)

            if use_DS:
                corrected_id_ev, ds_info = ds_model.ds_correct_id_evidence_by_type_fusion(
                    model=model,
                    raw_enemy_state=enemy,
                    sensor_reliability=0.70,
                    discounted_reliability=0.30,
                    conflict_discount_th=0.45
                )

                fixed_id_evidences[j] = corrected_id_ev
                ds_k_info[j] = {
                    "sensor_type": ds_info["sensor_type"],
                    "fused_type": ds_info["fused_type"],
                    "K_type": ds_info["K_type"],
                    "K_type_after": ds_info["K_type_after"],
                    "ds_action": ds_info["ds_action"],
                }
            else:
                fixed_id_evidences[j] = form_ev.get("ID", None)

        # 单机级分支：得到 N_friendly × N_enemy 的威胁矩阵
        pair_posteriors = np.zeros((num_friendly, num_enemy, 3), dtype=float)
        pair_scores = np.zeros((num_friendly, num_enemy), dtype=float)

        for i, friendly in enumerate(friendlies):
            prob_matrix_i = []
            local_factors_i = []

            for j, enemy in enumerate(enemies):
                pair_target = build_pairwise_target(enemy, friendly)

                # ===== 单机局部几何修正因子 =====
                # 目的：在 DBN-TOPSIS 已经接近饱和时，增强“某个目标更靠近某架飞机、
                # 更快接近某架飞机、TTC 更小”的局部差异。
                ttc = pair_target.get("TTC", 1e6)
                dist = pair_target.get("Distance", 1e6)
                vc = pair_target.get("ClosingSpeed", 0.0)

                ttc_factor = 1.0 / (1.0 + ttc / 120.0)
                dist_factor = 1.0 / (1.0 + dist / 120.0)
                # vc_factor = max(vc, 0.0) / 0.340
                vc_factor = vc / (vc + 0.340 + 1e-10)

                local_factor = (
                    0.35 * ttc_factor
                    + 0.20 * dist_factor
                    + 0.10 * vc_factor
                )
                local_factors_i.append(local_factor)
                # =================================

                pair_ev = model.fuzzify_target_data(
                    pair_target,
                    allow_missing=allow_missing
                )

                if fixed_id_evidences.get(j, None) is not None:
                    pair_ev["ID"] = fixed_id_evidences[j]

                posterior = model.bayesian_inference(
                    pair_ev,
                    pair_priors[i, j]
                )

                pair_posteriors[i, j] = posterior
                pair_priors[i, j] = posterior @ model.transition_matrix

                prob_matrix_i.append(posterior)

            prob_matrix_i = np.array(prob_matrix_i)
            raw_pair_scores = topsis_closeness(prob_matrix_i)
            local_factors_i = np.array(local_factors_i, dtype=float)
            
            # 局部几何双向修正
            local_delta = local_factors_i - np.mean(local_factors_i)
            gamma_local = 1.5  # 修正强度
            local_adjust_factor = np.exp(gamma_local * local_delta)  # 指数型扰动因子

            pair_scores[i, :] = raw_pair_scores * local_adjust_factor
            pair_scores[i, :] = np.clip(pair_scores[i, :], 0.0, 1.0)

            # 注意：这里不归一化，因为后面 c_max/c_avg/c_soft 会统一归一化。
            # pair_scores[i, :] = raw_pair_scores * local_factors_i
            
        # 编队整体分支
        form_posteriors = np.zeros((num_enemy, 3), dtype=float)
        form_targets_debug = []

        for j, enemy in enumerate(enemies):
            formation_target = build_formation_target(enemy, friendlies)
            form_targets_debug.append(formation_target)

            form_ev = model.fuzzify_target_data(formation_target, allow_missing=allow_missing)

            if fixed_id_evidences.get(j, None) is not None:
                form_ev["ID"] = fixed_id_evidences[j]

            posterior = model.bayesian_inference(form_ev, form_priors[j])
            form_posteriors[j] = posterior
            form_priors[j] = posterior @ model.transition_matrix

        form_scores = topsis_closeness(form_posteriors)
        # form_scores = normalize_scores(form_scores_raw)

        # # 编队结构修正：覆盖比例越大、TTC越小，整体威胁越高
        # structure_factors = []
        # for ft in form_targets_debug:
        #     cover = ft.get("CoverRatio", 0.0)
        #     ttc = ft.get("TTC_min", 1e6)

        #     ttc_factor = 1.0 / (1.0 + ttc / 100.0)
        #     cover_factor = cover

        #     factor = 1.0 + 0.3 * cover_factor + 0.3 * ttc_factor
        #     structure_factors.append(factor)

        # structure_factors = np.array(structure_factors)
        # form_scores = normalize_scores(form_scores * structure_factors)

        # 单机威胁矩阵聚合
        c_max = np.max(pair_scores, axis=0)
        c_avg = np.mean(pair_scores, axis=0)
        c_soft = tau * np.log(np.mean(np.exp(pair_scores / tau), axis=0) + 1e-10)

        # c_max = normalize_scores(c_max)
        # c_avg = normalize_scores(c_avg)
        # c_soft = normalize_scores(c_soft)

        lambda_max, lambda_avg, lambda_soft = lambdas
        agg_scores = (
            lambda_max * c_max
            + lambda_avg * c_avg
            + lambda_soft * c_soft
        )
        # agg_scores = normalize_scores(agg_scores)

        # 最终整体编队威胁度
        total_scores = beta * form_scores + (1.0 - beta) * agg_scores

        rank = np.argsort(total_scores)[::-1] + 1

        record_entry = {
            "time": current_time,

            # 为了兼容你原来的表格和 Spearman 代码
            "scores": total_scores.copy(),
            "posteriors": form_posteriors.copy(),
            "rank": rank.copy(),

            # 新增输出
            "pair_scores": pair_scores.copy(),
            "pair_posteriors": pair_posteriors.copy(),
            "form_scores": form_scores.copy(),
            "agg_scores": agg_scores.copy(),
            "formation_targets": form_targets_debug,
        }

        if use_DS:
            record_entry["ds_k"] = ds_k_info

        records.append(record_entry)

    return records



def extract_target_series(records, target_idx):
    return pd.DataFrame({
        'time': [record['time'] for record in records],
        'score': [record['scores'][target_idx] for record in records],
        'p_high': [record['posteriors'][target_idx, 0] for record in records]
    })

def build_comparison_table(reference_records, no_ar_records, ar_records, target_idx, display_times):
    ref_df = extract_target_series(reference_records, target_idx).set_index('time')
    no_ar_df = extract_target_series(no_ar_records, target_idx).set_index('time')
    ar_df = extract_target_series(ar_records, target_idx).set_index('time')

    rows = []
    for t in display_times:
        rows.append({
            '时刻/s': t,
            '完整数据-T1威胁度': round(ref_df.loc[t, 'score'], 4),
            '缺维无AR-T1威胁度': round(no_ar_df.loc[t, 'score'], 4),
            '缺维有AR-T1威胁度': round(ar_df.loc[t, 'score'], 4),
            '无AR相对偏差': round(abs(no_ar_df.loc[t, 'score'] - ref_df.loc[t, 'score']), 4),
            '有AR相对偏差': round(abs(ar_df.loc[t, 'score'] - ref_df.loc[t, 'score']), 4)
        })
    return pd.DataFrame(rows)
def summarize_multiple_missing(reference_records, no_ar_records, ar_records, missing_configs):
    min_start = min(cfg['start'] for cfg in missing_configs)
    max_end = max(cfg['end'] for cfg in missing_configs)

    # 【核心修改】：计算全序列(整个威胁度)的平均位置匹配率
    def calc_full_rank_match(ref_records, test_records):
        match_scores = []
        for ref, test in zip(ref_records, test_records):
            if min_start <= ref['time'] <= max_end:
                ref_rank = np.array(ref['rank'])
                test_rank = np.array(test['rank'])
                # 计算 7 个目标中有几个排在了完全正确的位置上
                # 例如 7 个对了 5 个，这一秒的准确率就是 5/7
                accuracy = np.sum(ref_rank == test_rank) / len(ref_rank)
                match_scores.append(accuracy)
        return np.mean(match_scores) if match_scores else 0.0

    no_ar_rank_match = calc_full_rank_match(reference_records, no_ar_records)
    ar_rank_match = calc_full_rank_match(reference_records, ar_records)

    target_maes_no_ar = []
    target_maes_ar = []
    
    unique_missing_targets = sorted(set(
    (cfg['target_idx'], cfg['start'], cfg['end']) for cfg in missing_configs
))

    for t_idx, t_start, t_end in unique_missing_targets:
        ref_scores = np.array([r['scores'][t_idx] for r in reference_records if t_start <= r['time'] <= t_end])
        no_ar_scores = np.array([r['scores'][t_idx] for r in no_ar_records if t_start <= r['time'] <= t_end])
        ar_scores = np.array([r['scores'][t_idx] for r in ar_records if t_start <= r['time'] <= t_end])

        target_maes_no_ar.append(np.mean(np.abs(no_ar_scores - ref_scores)))
        target_maes_ar.append(np.mean(np.abs(ar_scores - ref_scores)))

    mean_no_ar_mae = np.mean(target_maes_no_ar)
    mean_ar_mae = np.mean(target_maes_ar)

    return pd.DataFrame([
        {
            '方案': '复合缺维且无 AR (退化评估)',
            '受损目标平均威胁度误差(MAE)': round(mean_no_ar_mae, 6),
            f'全局排序一致率({min_start}s-{max_end}s)': f"{no_ar_rank_match:.2%}"
        },
        {
            '方案': '复合缺维且使用 AR(p) 填补',
            '受损目标平均威胁度误差(MAE)': round(mean_ar_mae, 6),
            f'全局排序一致率({min_start}s-{max_end}s)': f"{ar_rank_match:.2%}"
        }
    ])

def summarize_multiple_misidentification(reference_records, no_ds_records, ds_records, missing_configs):
    min_start = min(cfg['start'] for cfg in missing_configs)
    max_end = max(cfg['end'] for cfg in missing_configs)

    # 【核心修改】：计算全序列(整个威胁度)的平均位置匹配率
    def calc_full_rank_match(ref_records, test_records):
        match_scores = []
        for ref, test in zip(ref_records, test_records):
            if min_start <= ref['time'] <= max_end:
                ref_rank = np.array(ref['rank'])
                test_rank = np.array(test['rank'])
                accuracy = np.sum(ref_rank == test_rank) / len(ref_rank)
                match_scores.append(accuracy)
        return np.mean(match_scores) if match_scores else 0.0

    no_ds_rank_match = calc_full_rank_match(reference_records, no_ds_records)
    ds_rank_match = calc_full_rank_match(reference_records, ds_records)

    target_maes_no_ds = []
    target_maes_ds = []
    
    for cfg in missing_configs:
        t_idx = cfg['target_idx']
        t_start = cfg['start']
        t_end = cfg['end']
        
        ref_scores = np.array([r['scores'][t_idx] for r in reference_records if t_start <= r['time'] <= t_end])
        no_ds_scores = np.array([r['scores'][t_idx] for r in no_ds_records if t_start <= r['time'] <= t_end])
        ds_scores = np.array([r['scores'][t_idx] for r in ds_records if t_start <= r['time'] <= t_end])
        
        target_maes_no_ds.append(np.mean(np.abs(no_ds_scores - ref_scores)))
        target_maes_ds.append(np.mean(np.abs(ds_scores - ref_scores)))

    mean_no_ds_mae = np.mean(target_maes_no_ds)
    mean_ds_mae = np.mean(target_maes_ds)

    return pd.DataFrame([
        {
            '方案': '识别错误且无 DS (退化评估)',
            '受损目标平均威胁度误差(MAE)': round(mean_no_ds_mae, 6),
            f'全局排序一致率({min_start}s-{max_end}s)': f"{no_ds_rank_match:.2%}"
        },
        {
            '方案': '识别错误且使用 DS',
            '受损目标平均威胁度误差(MAE)': round(mean_ds_mae, 6),
            f'全局排序一致率({min_start}s-{max_end}s)': f"{ds_rank_match:.2%}"
        }
    ])


# ================= 5.1 排序一致性计算：Spearman 秩相关系数 =================
def rank_order_to_positions(rank_order):
    """
    将“按威胁度从高到低排列的目标编号”转换为“每个目标对应的名次”。

    例如：rank_order = [1, 2, 5, 4, 6, 7, 3]
    表示 T1 排第1、T2排第2、T5排第3、T4排第4、T6排第5、T7排第6、T3排第7。
    返回：positions = [1, 2, 7, 4, 3, 5, 6]
    即 positions[i-1] 表示目标 Ti 的名次。
    """
    rank_order = np.asarray(rank_order, dtype=int)
    positions = np.empty(len(rank_order), dtype=int)
    for pos, target_id in enumerate(rank_order, start=1):
        positions[target_id - 1] = pos
    return positions


def spearman_rank_correlation(reference_rank, method_rank):
    """
    计算某一时刻 method_rank 相对于 reference_rank 的 Spearman 秩相关系数。

    reference_rank 和 method_rank 均为目标编号排序，例如：
    [1, 2, 5, 4, 6, 7, 3]

    公式：rho = 1 - 6 * sum(d_i^2) / (N * (N^2 - 1))
    其中 d_i = r_i^method - r_i^base。
    """
    ref_pos = rank_order_to_positions(reference_rank)
    method_pos = rank_order_to_positions(method_rank)

    if len(ref_pos) != len(method_pos):
        raise ValueError("reference_rank 与 method_rank 的目标数量不一致。")

    n = len(ref_pos)
    if n < 2:
        return 1.0, 0, np.zeros(n, dtype=int)

    d = method_pos - ref_pos
    sum_d2 = int(np.sum(d ** 2))
    rho = 1.0 - (6.0 * sum_d2) / (n * (n ** 2 - 1))
    return float(rho), sum_d2, d


def rank_to_string(rank_order):
    """将排序数组转换成论文表格中的字符串形式。"""
    return " > ".join([f"T{int(r)}" for r in rank_order])


def calculate_spearman_series(reference_records, method_records, time_points, method_name):
    """
    在多个采样时刻计算 Spearman 排序一致性。

    reference_records: 无干扰基准记录，例如 full_records
    method_records: 待评价方法记录，例如 A_records 或 D_records
    time_points: 需要统计的时刻列表
    method_name: 方法名称，用于输出表格
    """
    ref_by_time = {rec['time']: rec for rec in reference_records}
    method_by_time = {rec['time']: rec for rec in method_records}

    rows = []
    for t in time_points:
        if t not in ref_by_time or t not in method_by_time:
            continue

        ref_rank = ref_by_time[t]['rank']
        method_rank = method_by_time[t]['rank']
        rho, sum_d2, d = spearman_rank_correlation(ref_rank, method_rank)

        rows.append({
            'Time/s': t,
            'Method': method_name,
            'Baseline Rank': rank_to_string(ref_rank),
            'Method Rank': rank_to_string(method_rank),
            'd_i': ", ".join(map(str, d.tolist())),
            'Sum d_i^2': sum_d2,
            'Spearman rho': round(rho, 4),
            'Spearman deviation (1-rho)': round(1.0 - rho, 4)
        })

    return pd.DataFrame(rows)


def summarize_spearman_results(spearman_df, interference_times=None):
    """
    汇总 Spearman 平均值。
    默认输出：所有采样节点平均值；如果给出 interference_times，则额外输出干扰区间平均值。
    """
    rows = []
    for method_name, sub_df in spearman_df.groupby('Method'):
        rows.append({
            'Interval': 'All selected nodes',
            'Method': method_name,
            'Num nodes': len(sub_df),
            'Mean Spearman rho': round(sub_df['Spearman rho'].mean(), 4),
            'Mean Spearman deviation (1-rho)': round(sub_df['Spearman deviation (1-rho)'].mean(), 4)
        })

        if interference_times is not None:
            int_df = sub_df[sub_df['Time/s'].isin(interference_times)]
            if len(int_df) > 0:
                rows.append({
                    'Interval': 'Interference intervals',
                    'Method': method_name,
                    'Num nodes': len(int_df),
                    'Mean Spearman rho': round(int_df['Spearman rho'].mean(), 4),
                    'Mean Spearman deviation (1-rho)': round(int_df['Spearman deviation (1-rho)'].mean(), 4)
                })

    return pd.DataFrame(rows)
