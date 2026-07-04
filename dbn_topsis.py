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
        self.states_E = 3 # 意图 E: 0-攻击, 1-干扰, 2-侦察
        
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
        self.cpd_E_given_L = np.array([[0.8, 0.15, 0.05], [0.2, 0.6, 0.2], [0.05, 0.15, 0.8]])
        self.cpd_C_given_E = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.1, 0.2, 0.7]])
        self.cpd_G_given_E = np.array([[0.7, 0.2, 0.1], [0.8, 0.2, 0.0], [0.1, 0.4, 0.5]])
        self.cpd_H_given_E = np.array([
            [0.60, 0.20, 0.20],
            [0.25, 0.50, 0.25],
            [0.25, 0.20, 0.55] 
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

        jamming = target.get('Jamming')
        if not is_missing_value(jamming):
            jam_map = {'Strong': 0, 'Mid': 1, 'Weak': 2}
            evidence['G'] = np.zeros(3)
            evidence['G'][jam_map.get(jamming, 2)] = 1.0
        elif not allow_missing:
            raise ValueError(f"{target.get('Name', 'Unknown')} 的 Jamming 缺失，无法直接评估。")

        return evidence

    def bayesian_inference(self, evidence, current_prior):
        """双层贝叶斯网络后验概率计算，缺失证据时仅使用可观测维度。"""
        likelihood_E = np.ones(self.states_E)
        for e in range(self.states_E):
            if 'C' in evidence: likelihood_E[e] *= np.dot(self.cpd_C_given_E[e], evidence['C'])
            if 'G' in evidence: likelihood_E[e] *= np.dot(self.cpd_G_given_E[e], evidence['G'])
            if 'H' in evidence: likelihood_E[e] *= np.dot(self.cpd_H_given_E[e], evidence['H'])
            
        likelihood_L = np.ones(self.states_L)
        for l in range(self.states_L):
            p_E_obs = np.sum(self.cpd_E_given_L[l] * likelihood_E)
            if 'V' in evidence: likelihood_L[l] *= np.dot(self.cpd_V_given_L[l], evidence['V'])
            if 'D' in evidence: likelihood_L[l] *= np.dot(self.cpd_D_given_L[l], evidence['D'])
            if 'S' in evidence: likelihood_L[l] *= np.dot(self.cpd_S_given_L[l], evidence['S'])
            if 'ID' in evidence: likelihood_L[l] *= np.dot(self.cpd_ID_given_L[l], evidence['ID'])
            likelihood_L[l] *= p_E_obs
            
        posterior_L = current_prior * likelihood_L
        total_prob = np.sum(posterior_L)
        return posterior_L / total_prob if total_prob > 0 else current_prior

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

