import numpy as np

def is_missing_value(value):
    """统一判断数值/类别字段是否缺失。"""
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False

# ================= 1. DBN-TOPSIS 融合评估模型 =================
class DBN_TOPSIS_Fusion_Assessment:
    def __init__(self):
        """初始化 DBN-TOPSIS 融合威胁评估模型"""
        self.states_L = 3 # 威胁 L: 0-高(H), 1-中(M), 2-低(L)
        self.states_E = 5
        self.intent_names = [
            "Attack",
            "Interference",
            "Reconnaissance",
            "Feint",
            "EscortEvasion"
        ]
        self.intent_names_cn = [
            "攻击突防",
            "电子干扰",
            "侦察监视",
            "佯攻欺骗",
            "护航规避"
        ]
        
        # 初始时刻(t=0)的先验概率
        self.prior_L = np.array([0.3, 0.4, 0.3])
        
        # 状态转移概率矩阵 P(L_t | L_{t-1})
        self.transition_matrix = np.array([
            [0.6, 0.3, 0.1],    # H 状态极易衰退为 M 状态
            [0.3, 0.35, 0.35],  
            [0.1, 0.35, 0.55]   
        ])
        self._init_cpds()

    def _init_cpds(self):
        """初始化条件概率表 (CPD)"""
        self.cpd_V_given_L = np.array([[0.8, 0.15, 0.05], [0.2, 0.6, 0.2], [0.05, 0.25, 0.7]])
        self.cpd_D_given_L = np.array([[0.8, 0.15, 0.05], [0.2, 0.6, 0.2], [0.05, 0.25, 0.7]])
        self.cpd_S_given_L = np.array([[0.8, 0.15, 0.05], [0.2, 0.6, 0.2], [0.05, 0.25, 0.7]])
        self.cpd_ID_given_L = np.array([
            [0.85, 0.10, 0.05],  # 增大此处权重，让网络容易被电子欺骗
            [0.10, 0.80, 0.10],
            [0.05, 0.10, 0.85]
        ])
        # E 状态顺序：
        # 0 Attack
        # 1 Interference
        # 2 Reconnaissance
        # 3 Feint
        # 4 EscortEvasion

        self.cpd_E_given_L = np.array([
            # Attack, Interference, Recon, Feint, EscortEvasion
            [0.58, 0.12, 0.03, 0.20, 0.07],  # L=High
            [0.22, 0.35, 0.08, 0.20, 0.15],  # L=Medium
            [0.05, 0.12, 0.43, 0.10, 0.30],  # L=Low
        ])

        # C: 接近角证据
        # C0: 小接近角，强指向我方
        # C1: 中等接近角
        # C2: 大接近角，不指向我方
        self.cpd_C_given_E = np.array([
            [0.75, 0.20, 0.05],  # Attack
            [0.25, 0.50, 0.25],  # Interference
            [0.10, 0.25, 0.65],  # Reconnaissance
            [0.45, 0.35, 0.20],  # Feint
            [0.05, 0.25, 0.70],  # EscortEvasion
        ])

        # G: 干扰强度证据
        # G0 Strong, G1 Mid, G2 Weak
        self.cpd_G_given_E = np.array([
            [0.30, 0.50, 0.20],  # Attack
            [0.85, 0.15, 0.00],  # Interference
            [0.10, 0.35, 0.55],  # Reconnaissance
            [0.45, 0.40, 0.15],  # Feint
            [0.65, 0.30, 0.05],  # EscortEvasion
        ])

        # H: 高度/高度差证据
        # H0 低, H1 中, H2 高
        self.cpd_H_given_E = np.array([
            [0.35, 0.45, 0.20],  # Attack
            [0.20, 0.60, 0.20],  # Interference
            [0.10, 0.30, 0.60],  # Reconnaissance
            [0.30, 0.45, 0.25],  # Feint
            [0.20, 0.45, 0.35],  # EscortEvasion
        ])

        # R: 接近速度/距离趋势证据
        # R0 接近, R1 保持, R2 远离
        self.cpd_R_given_E = np.array([
            [0.80, 0.15, 0.05],  # Attack
            [0.30, 0.55, 0.15],  # Interference
            [0.15, 0.65, 0.20],  # Reconnaissance
            [0.45, 0.20, 0.35],  # Feint
            [0.05, 0.35, 0.60],  # EscortEvasion
        ])

        # T: TTC 证据
        # T0 短, T1 中, T2 长
        self.cpd_T_given_E = np.array([
            [0.75, 0.20, 0.05],  # Attack
            [0.20, 0.55, 0.25],  # Interference
            [0.05, 0.25, 0.70],  # Reconnaissance
            [0.35, 0.35, 0.30],  # Feint
            [0.05, 0.30, 0.65],  # EscortEvasion
        ])

    def fuzzify_triangle(self, x, a1, a2, a3):
        """三角模糊隶属度函数"""
        if x < a1: mu_0 = 1.0
        elif a1 <= x <= a2: mu_0 = (a2 - x) / (a2 - a1)
        else: mu_0 = 0.0
        
        if a1 <= x <= a2: mu_1 = (x - a1) / (a2 - a1)
        elif a2 < x <= a3: mu_1 = (a3 - x) / (a3 - a2)
        else: mu_1 = 0.0
        
        if x <= a2: mu_2 = 0.0
        elif a2 < x <= a3: mu_2 = (x - a2) / (a3 - a2)
        else: mu_2 = 1.0
        return np.array([mu_0, mu_1, mu_2])

    def fuzzify_target_data(self, target, allow_missing=False, id_threat_score = None):
        """连续变量模糊化与离散变量编码，允许在缺维时跳过缺失证据。"""
        evidence = {}

        def add_continuous(feature_key, target_key, a1, a2, a3, reverse=False):
            value = target.get(target_key)
            if is_missing_value(value):
                if not allow_missing:
                    raise ValueError(f"{target.get('Name', 'Unknown')} 的 {target_key} 缺失，无法直接评估。")
                return
            fuzzy_value = self.fuzzify_triangle(value, a1, a2, a3)
            evidence[feature_key] = fuzzy_value[::-1] if reverse else fuzzy_value

        # 调整边界，精准触发论文中的“超越点”
        # 速度：战斗机>1.5马赫占据绝对高威胁，导弹<0.8处于严重劣势
        add_continuous('V', 'Speed', 0.8, 1.2, 1.5, reverse=True) 
        # 距离与捷径：突破 150km 就算进入极高威胁防空圈
        add_continuous('D', 'Distance', 100, 300, 450)
        add_continuous('S', 'Shortcut', 150, 300, 450)
        # 高度：放宽高度限制，使得7km左右高空突防的战斗机不被过度惩罚
        add_continuous('H', 'Height', 4, 8, 12)

        heading = target.get('Heading')
        if not is_missing_value(heading):
            if target['Heading'] < 30:
                evidence['C'] = np.array([0.4, 0.3, 0.3])  # 小于30度
            elif target['Heading'] < 60:
                evidence['C'] = np.array([0.3, 0.4, 0.3])  # 30到60度之间
            else:
                evidence['C'] = np.array([0.3, 0.3, 0.4])  # 大于60度
        elif not allow_missing:
            raise ValueError(f"{target.get('Name', 'Unknown')} 的 Heading 缺失，无法直接评估。")

        target_type = target.get('Type')
        if not is_missing_value(target_type):
            id_map = {'Missile': 100, 'Fighter': 88, 'Bomber': 74, 'Heli': 46, 'UAV': 60, 'Recon': 38, 'Fuel':22}
            id_threat_score = id_map.get(target['Type'], 60) if id_threat_score == None else id_threat_score
            evidence['ID'] = self.fuzzify_triangle(id_threat_score, 30, 60, 90)[::-1]
        elif not allow_missing:
            raise ValueError(f"{target.get('Name', 'Unknown')} 的 Type 缺失，无法直接评估。")
        closing_speed = target.get("ClosingSpeed", None)
        if not is_missing_value(closing_speed):
            # ClosingSpeed > 0 表示正在接近我方
            if closing_speed > 0.08:
                evidence["R"] = np.array([0.80, 0.15, 0.05])  # 接近
            elif closing_speed > -0.03:
                evidence["R"] = np.array([0.15, 0.75, 0.10])  # 保持
            else:
                evidence["R"] = np.array([0.05, 0.20, 0.75])  # 远离

        ttc = target.get("TTC", target.get("TTC_min", None))
        if not is_missing_value(ttc):
            if ttc < 80:
                evidence["T"] = np.array([0.80, 0.15, 0.05])  # 短 TTC
            elif ttc < 180:
                evidence["T"] = np.array([0.15, 0.75, 0.10])  # 中 TTC
            else:
                evidence["T"] = np.array([0.05, 0.20, 0.75])  # 长 TTC

        jamming = target.get('Jamming')
        if not is_missing_value(jamming):
            jam_map = {'Strong': 0, 'Mid': 1, 'Weak': 2}
            evidence['G'] = np.zeros(3)
            evidence['G'][jam_map.get(jamming, 2)] = 1.0
        elif not allow_missing:
            raise ValueError(f"{target.get('Name', 'Unknown')} 的 Jamming 缺失，无法直接评估。")

        return evidence

    def bayesian_inference(self, evidence, current_prior, return_intent=False):
        """
        双层 DBN 推理：
        L: 威胁等级节点
        E: 五状态意图节点

        返回：
        - 默认返回 posterior_L
        - return_intent=True 时返回 (posterior_L, posterior_E)
        """

        # 1. 计算 P(obs_E | E)
        likelihood_E = np.ones(self.states_E)

        for e in range(self.states_E):
            if 'C' in evidence:
                likelihood_E[e] *= np.dot(self.cpd_C_given_E[e], evidence['C'])
            if 'G' in evidence:
                likelihood_E[e] *= np.dot(self.cpd_G_given_E[e], evidence['G'])
            if 'H' in evidence:
                likelihood_E[e] *= np.dot(self.cpd_H_given_E[e], evidence['H'])
            if 'R' in evidence:
                likelihood_E[e] *= np.dot(self.cpd_R_given_E[e], evidence['R'])
            if 'T' in evidence:
                likelihood_E[e] *= np.dot(self.cpd_T_given_E[e], evidence['T'])

        # 2. 计算直接作用于 L 的证据似然 P(obs_L | L)
        likelihood_L_direct = np.ones(self.states_L)

        for l in range(self.states_L):
            if 'V' in evidence:
                likelihood_L_direct[l] *= np.dot(self.cpd_V_given_L[l], evidence['V'])
            if 'D' in evidence:
                likelihood_L_direct[l] *= np.dot(self.cpd_D_given_L[l], evidence['D'])
            if 'S' in evidence:
                likelihood_L_direct[l] *= np.dot(self.cpd_S_given_L[l], evidence['S'])
            if 'ID' in evidence:
                likelihood_L_direct[l] *= np.dot(self.cpd_ID_given_L[l], evidence['ID'])

        # 3. 构造联合后验 P(L,E | obs)
        # joint[l,e] ∝ P(L=l) P(E=e|L=l) P(obs_L|L=l) P(obs_E|E=e)
        joint_LE = (
            current_prior[:, None]
            * likelihood_L_direct[:, None]
            * self.cpd_E_given_L
            * likelihood_E[None, :]
        )

        total_prob = np.sum(joint_LE)
        if total_prob <= 1e-12:
            posterior_L = current_prior
            posterior_E = np.ones(self.states_E) / self.states_E
        else:
            joint_LE = joint_LE / total_prob
            posterior_L = np.sum(joint_LE, axis=1)
            posterior_E = np.sum(joint_LE, axis=0)

        if return_intent:
            return posterior_L, posterior_E

        return posterior_L


    def topsis_evaluation(self, prob_matrix):
        """DBN-TOPSIS：构造绝对理想解进行评估"""
        ideal_best = np.array([1.0, 0.0, 0.0])  
        ideal_worst = np.array([0.0, 0.0, 1.0]) 
        
        scores = []
        for i in range(prob_matrix.shape[0]):
            d_plus = np.sqrt(np.sum((prob_matrix[i] - ideal_best)**2))
            d_minus = np.sqrt(np.sum((prob_matrix[i] - ideal_worst)**2))
            c_i = d_minus / (d_plus + d_minus + 1e-10)
            scores.append(c_i)
            
        scores = np.array(scores)
        sum_scores = np.sum(scores)
        if sum_scores == 0:
            return np.ones(len(scores)) / len(scores)
        normalized_scores = scores / sum_scores 
        return normalized_scores

