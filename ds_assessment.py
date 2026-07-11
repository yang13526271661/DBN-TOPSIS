from typing import Dict, List, Tuple

import numpy as np

# ================= 4. D-S 冲突识别以处理失真问题与辅助函数 =================
def introduce_multiple_misidentification(series, type_modify_configs, start_time=0):
    """
    模拟“识别不准”场景：在指定时间段内，把指定目标的 feature 改为 modify
    """
    for cfg in type_modify_configs:
        start_idx = cfg['start'] - start_time
        end_idx = cfg['end'] - start_time
        t_idx = cfg['target_idx']
        feature = cfg['feature']
        new_type = cfg['misidentification']

        # 确保索引在合法范围内
        for i in range(max(0, start_idx), min(len(series), end_idx + 1)):
            series[i][t_idx][feature] = new_type

    return series

class DS_Assessment:
    def __init__(self):
        pass

    def _normalize(self, x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        s = np.sum(x)
        return x / s if s > eps else np.ones_like(x) / len(x)

    def _likelihood_to_mass(self, support: np.ndarray, ig: float = 0.05) -> Dict[int, float]:
        """
        把对威胁状态 {H,M,L} 的支持度映射为 D-S 质量函数。
        仅保留 {H},{M},{L},Θ 四类焦元，便于做冲突诊断。
        """
        support = np.clip(np.asarray(support, dtype=float), 0.0, None)
        support = self._normalize(support)
        singleton_mass = [support[0] * (1.0 - 2 * ig), support[1] * (1.0 - 2 * ig), support[0] * ig + support[1] * ig * 0.5,
                            support[2] * (1.0 - 2 * ig), 0.0, support[2] * ig + support[1] * ig * 0.5, np.sum(support) * ig]
        # singleton_mass = (1.0 - ig) * support
        return {
            1: float(singleton_mass[0]),  # {H}
            2: float(singleton_mass[1]),  # {M}
            3: float(singleton_mass[2]),  # {H,M}
            4: float(singleton_mass[3]),  # {L}
            5: float(singleton_mass[4]),  # {H,L}
            6: float(singleton_mass[5]),  # {M,L}
            7: float(singleton_mass[6])   # Θ
        }

    def build_ds_masses(self, evidence: Dict[str, np.ndarray], ignorance: float = 0.1) -> Dict[str, Dict[int, float]]:
        """为当前目标的每个指标构建 D-S 质量函数。"""
        masses = {}
        keys = ['V', 'D', 'S', 'ID', 'C', 'G', 'H']
        for key in keys:
            if key not in evidence:
                continue
            masses[key] = self._likelihood_to_mass(evidence[key], ignorance)
        return masses

    def ds_fuse_two(self, m1: Dict[int, float], m2: Dict[int, float]) -> Tuple[Dict[int, float], float]:
        """
        Dempster 组合：返回 (组合后的质量函数, 两证据之间的冲突系数 K)
        焦元只考虑 {H},{M},{L},Θ。
        """
        combined_raw = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0}
        conflict = 0.0

        for A, mA in m1.items():
            if mA == 0:
                continue
            for B, mB in m2.items():
                if mB == 0:
                    continue
                inter = A & B
                if inter == 0:
                    conflict += mA * mB
                else:
                    combined_raw[inter] = combined_raw.get(inter, 0.0) + mA * mB

        norm = 1.0 - conflict
        if norm <= 1e-12:
            # 完全冲突时，退回全未知
            return {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0}, 1.0

        combined = {k: v / norm for k, v in combined_raw.items()}
        return combined, float(conflict)

    def ds_fuse_all(self, masses: Dict[str, Dict[int, float]]) -> Tuple[Dict[int, float], float, List[Tuple[str, float]]]:
        """
        在给定顺序下依次组合全部证据，返回：
        1) 该顺序下的最终质量函数
        2) 该顺序下的累计冲突强度 K(order) = 1 - Π(1-K_i)
        3) 该顺序下每一步产生的局部 K 记录
        """
        local_conflicts: List[Tuple[str, float]] = []
        survival = 1.0
        first_key = list(masses.keys())[0]
        current_mass = masses[first_key]

        for name, mass in masses.items():
            if name != first_key:
                current_mass, k_local = self.ds_fuse_two(current_mass, mass)
                survival *= (1.0 - k_local)
                local_conflicts.append((mass, float(k_local)))

        k_total = 1.0 - survival
        # print(f"对于当前时刻当前敌机，质量矩阵为：{masses}，总冲突因子为：{k_total}")
        return current_mass, float(k_total), local_conflicts

    def ds_precheck(self, evidence: Dict[str, np.ndarray], drift_method=False):
        """
        在 bayesian_inference 之前调用：
        - 计算当前时刻、当前目标的 D-S 冲突系数 K
        - 用留一法检查可能存在的不准确数据
        """
        if drift_method:
            ev_stacked = np.stack(list(evidence.values()))
            ev_mean = np.mean(ev_stacked, axis=0)
            ev_diff = {k: np.linalg.norm(v - ev_mean, ord=1) for k, v in evidence.items()}

            eps = 1e-8
            inv_sum = sum(1.0 / (d+eps) for d in ev_diff.values())
            ev_weight = {k: (1.0 / (d+eps)) / inv_sum for k, d in ev_diff.items()}

            weighted_ev_sub = {k: ev_weight[k] * evidence[k] for k in evidence}
            weighted_ev = sum(weighted_ev_sub.values())
            evidence = {key: weighted_ev for key in evidence}

        masses = self.build_ds_masses(evidence)
        fusion_mass, k_total, local_conflicts = self.ds_fuse_all(masses)
        weight_map = {1:100, 2:60, 3:80, 4:20, 5:60, 6:40, 7:60}
        overall_threat = sum(fusion_mass[k] * weight_map[k] for k in weight_map)
        return {
            'masses': masses,
            'fusion_mass': fusion_mass,
            'confliction': float(k_total),
            'local_conflicts': local_conflicts,
            'global_leave_one_out': True,
            'k_total_definition': 'mean_over_all_permutations',
            'overall_threat': overall_threat
        }

    def conflict_check(self, model, ev, target, diagnostic_item='ID', K_diff_th=0.05, drift_method=False):
        key_mapping = {
            'Type': 'ID', 'Jamming': 'G', 'Height': 'H', 
            'Speed': 'V', 'Distance': 'D', 'Heading': 'C', 'Shortcut': 'S'
        }
        diagnostic_item = key_mapping.get(diagnostic_item, diagnostic_item)
        ev_diagnostic = copy.deepcopy(ev)
        ev_diagnostic.pop(diagnostic_item, None)

        ds_result = self.ds_precheck(ev, drift_method)
        ds_result_diagnostic = self.ds_precheck(ev_diagnostic, drift_method)

        K_tot = ds_result['confliction']
        K_diag = ds_result_diagnostic['confliction']
        K_diff = K_tot - K_diag

        if K_diff > K_diff_th:
            # 发现欺骗，直接赋予 [1,1,1] 剥离该假情报的影响
            ev_fixed = copy.deepcopy(ev)
            ev_fixed[diagnostic_item] = np.array([1.0, 0.0, 0.0])
        else:
            ev_fixed = ev

        return ev_fixed, K_tot, K_diag

    def ds_combine_prob(self, p1, p2, eps=1e-12):
        """
        简化 D-S 组合规则。
        p1、p2 是定义在同一目标类型空间上的质量分布。
        这里只考虑单元素焦元，因此组合结果为归一化逐元素乘积；
        冲突系数 K = 1 - sum_i p1_i p2_i。
        """
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)

        p1 = p1 / (np.sum(p1) + eps)
        p2 = p2 / (np.sum(p2) + eps)

        agree = float(np.sum(p1 * p2))
        K = 1.0 - agree

        if agree <= eps:
            fused = np.ones_like(p1) / len(p1)
        else:
            fused = (p1 * p2) / agree

        fused = fused / (np.sum(fused) + eps)
        return fused, float(K)

    def type_sensor_mass(self, sensor_type, reliability=0.70):
        """
        当前传感器识别 Type 对目标类型空间的支持。
        reliability 越大，越相信当前 Type 字段。
        """
        type_names = ['Missile', 'Fighter', 'Bomber', 'Heli', 'UAV', 'Recon', 'Fuel']
        n = len(type_names)

        p = np.ones(n, dtype=float) * ((1.0 - reliability) / (n - 1))
        if sensor_type in type_names:
            p[type_names.index(sensor_type)] = reliability
        else:
            p[:] = 1.0 / n

        return p / (np.sum(p) + 1e-12)

    def type_kinematic_mass(self, raw_enemy_state):
        """
        根据目标自身运动学特征反推目标类型支持度。

        新版说明：
        1) 在对抗进攻场景中，导弹可以瞄准空中编队并产生一定爬升，
           因此不能再把 Missile 的 Height 原型限制得过低；
        2) 类型识别不仅看 Speed/Height，还要看 Heading/Shortcut：
           如果目标速度方向强烈指向我方编队，且航路捷径小，则更符合“制导导弹/突防武器”的运动学特征；
        3) 仍然不使用真实类型标签，只使用当前状态中的运动学量。
        """
        type_names = ['Missile', 'Fighter', 'Bomber', 'Heli', 'UAV', 'Recon', 'Fuel']

        speed = float(raw_enemy_state.get('Speed', 0.0))
        height = float(raw_enemy_state.get('Height', 0.0))
        heading = float(raw_enemy_state.get('Heading', 90.0))
        shortcut = float(raw_enemy_state.get('Shortcut', 300.0))

        # 运动学原型。
        # Missile 的高度范围故意放宽：允许低空巡航、末端跃升或对空攻击过程中的中低空爬升。
        proto = {
            'Missile': {'Speed': (0.75, 0.30), 'Height': (1.20, 3.00)},
            'Fighter': {'Speed': (1.60, 0.45), 'Height': (8.00, 3.00)},
            'Bomber':  {'Speed': (0.70, 0.20), 'Height': (8.50, 2.50)},
            'Heli':    {'Speed': (0.32, 0.14), 'Height': (1.00, 1.20)},
            'UAV':     {'Speed': (0.25, 0.12), 'Height': (11.0, 3.00)},
            'Recon':   {'Speed': (0.75, 0.25), 'Height': (10.0, 3.00)},
            'Fuel':    {'Speed': (0.70, 0.25), 'Height': (8.00, 3.00)},
        }

        # 攻击指向性：Heading 越小、Shortcut 越小，说明目标越像在制导突防。
        heading_like = np.exp(-0.5 * (heading / 25.0) ** 2)
        shortcut_like = np.exp(-0.5 * (shortcut / 80.0) ** 2)
        attack_directness = heading_like * shortcut_like

        scores = []
        for tp in type_names:
            mu_v, sig_v = proto[tp]['Speed']
            mu_h, sig_h = proto[tp]['Height']

            lv = np.exp(-0.5 * ((speed - mu_v) / sig_v) ** 2)
            lh = np.exp(-0.5 * ((height - mu_h) / sig_h) ** 2)

            base = lv * lh

            # 不同类型对“攻击指向性”的敏感程度不同。
            # 导弹最敏感；战斗机次之；轰炸机主要靠高空中速特征，避免被误判成 Missile。
            if tp == 'Missile':
                factor = 0.35 + 1.80 * attack_directness
            elif tp == 'Fighter':
                factor = 0.80 + 0.45 * attack_directness
            elif tp == 'Bomber':
                factor = 0.90 + 0.20 * attack_directness
            elif tp == 'UAV':
                factor = 0.95 + 0.08 * attack_directness
            else:
                factor = 1.0

            scores.append(base * factor)

        scores = np.asarray(scores, dtype=float) + 1e-12
        return scores / np.sum(scores)

    def type_window_jump_score(self, type_window, jump_threshold=0.10):
        if not type_window or len(type_window) < 2:
            return {
                "jump_score": 0.0,
                "jump_count": 0,
                "window_size": len(type_window) if type_window else 0,
                "has_type_jump": False,
            }

        sensor_types = [entry.get("sensor_type") for entry in type_window]
        jump_count = sum(
            1 for prev, curr in zip(sensor_types[:-1], sensor_types[1:])
            if prev != curr
        )
        jump_score = jump_count / max(len(sensor_types) - 1, 1)
        return {
            "jump_score": float(jump_score),
            "jump_count": int(jump_count),
            "window_size": len(sensor_types),
            "has_type_jump": bool(jump_score > jump_threshold),
        }

    def ds_fuse_temporal_window(
        self,
        model,
        current_sensor_type,
        evidence_window,
        current_type_reliability=0.30
    ):
        type_score = {
            'Missile': 100,
            'Fighter': 88,
            'Bomber': 74,
            'Heli': 46,
            'UAV': 60,
            'Recon': 38,
            'Fuel': 22
        }

        masses = {}
        if current_sensor_type in type_score:
            current_id_ev = model.fuzzify_triangle(
                type_score[current_sensor_type], 30, 60, 90
            )[::-1]
            current_id_ev = self._normalize(current_id_ev)
            p_id = (
                current_type_reliability * current_id_ev
                + (1.0 - current_type_reliability) * np.ones(3) / 3.0
            )
        else:
            p_id = np.ones(3) / 3.0

        masses["current_ID"] = self._likelihood_to_mass(p_id, ig=0.05)

        for offset, evidence in enumerate(evidence_window or []):
            for key, value in evidence.items():
                if key == "ID":
                    continue
                masses[f"t{offset}_{key}"] = self._likelihood_to_mass(value, ig=0.10)

        if len(masses) == 1:
            fusion_mass = masses["current_ID"]
            k_total = 0.0
            local_conflicts = []
        else:
            fusion_mass, k_total, local_conflicts = self.ds_fuse_all(masses)

        corrected_id_ev = np.array([
            fusion_mass.get(1, 0.0) + 0.5 * fusion_mass.get(3, 0.0) + 0.5 * fusion_mass.get(5, 0.0) + fusion_mass.get(7, 0.0) / 3.0,
            fusion_mass.get(2, 0.0) + 0.5 * fusion_mass.get(3, 0.0) + 0.5 * fusion_mass.get(6, 0.0) + fusion_mass.get(7, 0.0) / 3.0,
            fusion_mass.get(4, 0.0) + 0.5 * fusion_mass.get(5, 0.0) + 0.5 * fusion_mass.get(6, 0.0) + fusion_mass.get(7, 0.0) / 3.0,
        ], dtype=float)
        corrected_id_ev = self._normalize(corrected_id_ev)

        return corrected_id_ev, {
            "window_conflict": float(k_total),
            "window_evidence_count": len(masses),
            "window_fusion_mass": fusion_mass,
            "window_local_conflicts": local_conflicts,
        }

    def ds_correct_id_evidence_by_type_fusion(
        self,
        model,
        raw_enemy_state,
        sensor_reliability=0.70,
        discounted_reliability=0.30,
        conflict_discount_th=0.45,
        type_window=None,
        evidence_window=None,
        jump_score_th=0.10
    ):
        """
        D-S 类型空间融合修正模块。

        输出 corrected_id_ev，而不是删除 ID。
        这样 D-S 真正作用于“类型证据修正”，再把修正结果送入 DBN。
        """
        type_names = ['Missile', 'Fighter', 'Bomber', 'Heli', 'UAV', 'Recon', 'Fuel']
        type_score = {
            'Missile': 100,
            'Fighter': 88,
            'Bomber': 74,
            'Heli': 46,
            'UAV': 60,
            'Recon': 38,
            'Fuel': 22
        }

        sensor_type = raw_enemy_state.get('Type', None)

        p_sensor = self.type_sensor_mass(sensor_type, reliability=sensor_reliability)
        p_kin = self.type_kinematic_mass(raw_enemy_state)

        p_fused, K_type = self.ds_combine_prob(p_sensor, p_kin)
        jump_info = self.type_window_jump_score(type_window, jump_threshold=jump_score_th)

        kin_type = type_names[int(np.argmax(p_kin))]
        sensor_score = type_score.get(sensor_type, 60)
        kin_score = type_score.get(kin_type, 60)

        # 只在 D-S 冲突大，且运动学支持的类型威胁等级高于传感器类型时，
        # 才折扣传感器 Type。这样可以处理 Missile->UAV、Fighter->Bomber，
        # 同时避免 Missile 等高威胁目标在正常时刻被运动学噪声错误降级。
        if jump_info["has_type_jump"]:
            corrected_id_ev, temporal_info = self.ds_fuse_temporal_window(
                model=model,
                current_sensor_type=sensor_type,
                evidence_window=evidence_window,
                current_type_reliability=discounted_reliability
            )
            p_for_id = corrected_id_ev
            K_type_after = temporal_info["window_conflict"]
            ds_action = 'temporal_window_conflict_by_DS'
        elif K_type > conflict_discount_th and kin_score > sensor_score:
            p_sensor_discounted = self.type_sensor_mass(
                sensor_type,
                reliability=discounted_reliability
            )
            p_for_id, K_type_after = self.ds_combine_prob(p_sensor_discounted, p_kin)
            ds_action = 'discount_sensor_type_by_DS'
            temporal_info = {
                "window_conflict": np.nan,
                "window_evidence_count": 0,
            }
        else:
            # 不满足折扣条件：保留原始传感器 Type 证据，避免正常时刻被 D-S 扰动。
            K_type_after = K_type
            p_for_id = np.zeros(len(type_names), dtype=float)
            if sensor_type in type_names:
                p_for_id[type_names.index(sensor_type)] = 1.0
            else:
                p_for_id[:] = 1.0 / len(type_names)
            ds_action = 'keep_sensor_type_by_DS'
            temporal_info = {
                "window_conflict": np.nan,
                "window_evidence_count": 0,
            }

        if ds_action != 'temporal_window_conflict_by_DS':
            corrected_id_ev = np.zeros(3, dtype=float)
            for idx, tp in enumerate(type_names):
                id_ev_tp = model.fuzzify_triangle(type_score[tp], 30, 60, 90)[::-1]
                corrected_id_ev += p_for_id[idx] * id_ev_tp

            corrected_id_ev = corrected_id_ev / (np.sum(corrected_id_ev) + 1e-12)

        return corrected_id_ev, {
            'sensor_type': sensor_type,
            'fused_type': type_names[int(np.argmax(p_for_id))] if len(p_for_id) == len(type_names) else kin_type,
            'K_type': K_type,
            'K_type_after': K_type_after,
            'ds_action': ds_action,
            'p_sensor': p_sensor,
            'p_kin': p_kin,
            'p_fused': p_for_id,
            'jump_score': jump_info["jump_score"],
            'jump_count': jump_info["jump_count"],
            'window_size': jump_info["window_size"],
            'has_type_jump': jump_info["has_type_jump"],
            'window_conflict': temporal_info["window_conflict"],
            'window_evidence_count': temporal_info["window_evidence_count"],
        }


