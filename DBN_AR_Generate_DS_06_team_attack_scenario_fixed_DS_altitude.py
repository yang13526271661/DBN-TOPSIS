import numpy as np
import pandas as pd
import copy
from typing import Dict, List, Tuple, Any
import plot_utils
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

# ================= 2. 基于雷达球坐标系的异构目标动力学模型 =================
class SphericalBaseTarget:
    def __init__(self, tid, name, t_type, jamming, D, H, V_mach, C_deg, theta_deg=0.0):
        self.tid = tid
        self.name = name
        self.type = t_type
        self.jamming = jamming

        self.r = float(D)
        self.phi = np.arcsin(H / D) if D > 0 else 0.0
        self.theta = np.radians(theta_deg)
        self.V = V_mach * 0.340
        self.C = np.radians(C_deg)
        self.time = 0

        # 新增：三维速度向量，用于编队相对几何计算
        e_r, e_phi, e_theta = self._spherical_basis()

        v_r = -self.V * np.cos(self.C)
        v_phi = self.V * np.sin(self.C)

        self.v_cart = v_r * e_r + v_phi * e_phi

    def _cartesian_position(self):
        """球坐标转三维笛卡尔坐标，单位 km。"""
        x = self.r * np.cos(self.phi) * np.cos(self.theta)
        y = self.r * np.cos(self.phi) * np.sin(self.theta)
        z = self.r * np.sin(self.phi)
        return np.array([x, y, z], dtype=float)

    def _spherical_basis(self):
        """球坐标基向量：e_r, e_phi, e_theta。"""
        cp = np.cos(self.phi)
        sp = np.sin(self.phi)
        ct = np.cos(self.theta)
        st = np.sin(self.theta)

        e_r = np.array([cp * ct, cp * st, sp], dtype=float)
        e_phi = np.array([-sp * ct, -sp * st, cp], dtype=float)
        e_theta = np.array([-st, ct, 0.0], dtype=float)

        return e_r, e_phi, e_theta

    def get_state(self, current_time):
        H = self.r * np.sin(self.phi)
        S = self.r * np.sin(self.C)
        pos = self._cartesian_position()
        vel = self.v_cart

        return {
            'Time': current_time,
            'Target_ID': self.tid,
            'Name': self.name,
            'Type': self.type,
            'Jamming': self.jamming,

            # 保留你原 DBN-TOPSIS 使用的固定点特征
            'Height': round(max(0.001, H), 3),
            'Speed': round(self.V / 0.340, 3),
            'Distance': round(max(0.1, self.r), 2),
            'Heading': round(np.degrees(self.C), 2),
            'Shortcut': round(max(0.0, S), 2),

            # 新增：编队评估需要的三维位置和速度
            'X': float(pos[0]),
            'Y': float(pos[1]),
            'Z': float(pos[2]),
            'VX': float(vel[0]),
            'VY': float(vel[1]),
            'VZ': float(vel[2]),
        }

    def kinematic_update(self, dt, V_cmd, H_cmd, C_cmd, sign_theta=1.0):
        self.V += (V_cmd - self.V) * 0.1 * dt
        self.C += (C_cmd - self.C) * 0.1 * dt
        v_r = -self.V * np.cos(self.C)

        current_H = self.r * np.sin(self.phi)
        H_dot_cmd = (H_cmd - current_H) * 0.2
        cos_phi = np.cos(self.phi)

        if cos_phi > 0.01:
            v_phi = (H_dot_cmd - v_r * np.sin(self.phi)) / cos_phi
        else:
            v_phi = 0.0

        max_v_transverse = self.V * np.sin(self.C)
        v_phi = np.clip(v_phi, -max_v_transverse, max_v_transverse)

        v_theta_sq = max(0.0, max_v_transverse ** 2 - v_phi ** 2)
        v_theta = sign_theta * np.sqrt(v_theta_sq)

        # 新增：把球坐标速度分量转换为三维速度向量
        e_r, e_phi, e_theta = self._spherical_basis()
        self.v_cart = v_r * e_r + v_phi * e_phi + v_theta * e_theta

        self.r += v_r * dt
        if self.r < 0.1:
            self.r = 0.1

        self.phi += (v_phi / self.r) * dt
        self.theta += (v_theta / (self.r * cos_phi)) * dt if cos_phi > 0.01 else 0.0
        self.time += dt

class BGM109C_Spherical(SphericalBaseTarget):
    def update(self, dt):
        V_cmd = 0.65 * 0.340
        C_cmd = np.radians(10) 
        if 2.0 < self.r < 10.0: H_cmd = 0.3 
        elif self.r <= 2.0: H_cmd = 0.01
        else: H_cmd = 0.05
        self.kinematic_update(dt, V_cmd, H_cmd, C_cmd)

class AGM86B_Spherical(SphericalBaseTarget):
    def update(self, dt):
        # 引入正弦波，形成S型机动曲线
        C_cmd = np.radians(45) + np.radians(40) * np.sin(self.time * 0.03)
        self.kinematic_update(dt, 0.61 * 0.340, 0.03, C_cmd)

class AH64A_Spherical(SphericalBaseTarget):
    def update(self, dt):
        V_cmd = (0.27 + 0.05 * np.sin(self.time * 0.2)) * 0.340
        H_cmd = 0.1 if int(self.time / 30) % 2 == 0 else 0.02
        self.kinematic_update(dt, V_cmd, H_cmd, np.radians(15))

class F16C_Spherical(SphericalBaseTarget):
    def update(self, dt):
        C_cmd = np.radians(12) + np.radians(15) * abs(np.sin(self.time * 0.1))
        H_cmd = 1.0 if self.r < 150 else 8.0
        sign_t = 1.0 if np.sin(self.time * 0.1) > 0 else -1.0
        self.kinematic_update(dt, 1.5 * 0.340, H_cmd, C_cmd, sign_theta=sign_t)

class F22_Spherical(SphericalBaseTarget):
    def update(self, dt):
        # 隐身战机战术侧转机动
        target_C = np.radians(80) if 200 < self.time < 400 else np.radians(15)
        C_cmd = target_C + np.radians(10) * np.sin(self.time * 0.04)
        self.kinematic_update(dt, 1.9 * 0.340, 7.45, C_cmd)

class B52H_Spherical(SphericalBaseTarget):
    def update(self, dt):
        self.kinematic_update(dt, 0.65 * 0.340, 8.0, np.radians(12))

class MQ9_Spherical(SphericalBaseTarget):
    def update(self, dt):
        C_cmd = np.radians(30) * min(1.0, self.r / 100.0) 
        self.kinematic_update(dt, 0.2 * 0.340, 10.6, C_cmd)

# ================= 2.1 同构飞机编队态势建模与相对特征构造 =================
def create_homogeneous_fighter_formation(
    center=np.array([0.0, 0.0, 8.0]),
    velocity=np.array([0.85 * 0.340, 0.0, 0.0]),
    formation_type="diamond"
):
    """
    创建同类型飞机编队。
    单位：
    - 位置 km
    - 速度 km/s
    """

    if formation_type == "diamond":
        d = 25.0

        offsets = [
            np.array([0.0, 0.0, 0.0]),          # 领机 / 编队核心
            np.array([-d, -18.0, 0.0]),         # 左翼机
            np.array([ d, -18.0, 0.0]),         # 右翼机
            np.array([0.0, -45.0, 0.0]),        # 后卫机
        ]

        values = [1.4, 1.0, 1.0, 0.8]
        roles = ["Leader", "LeftWing", "RightWing", "RearGuard"]

    elif formation_type == "wedge":
        offsets = [
            np.array([0.0, 0.0, 0.0]),
            np.array([-2.0, -2.0, 0.0]),
            np.array([2.0, -2.0, 0.0]),
            np.array([-4.0, -4.0, 0.0]),
            np.array([4.0, -4.0, 0.0]),
        ]
    else:
        raise ValueError(f"未知编队类型: {formation_type}")

    friendlies = []
    for i, offset in enumerate(offsets):
        friendlies.append({
            "Aircraft_ID": i,
            "Name": f"Friendly_Fighter_{i+1}",
            "Role": roles[i],
            "Value": values[i],
            "Type": "FriendlyFighter",
            "X": float(center[0] + offset[0]),
            "Y": float(center[1] + offset[1]),
            "Z": float(center[2] + offset[2]),
            "VX": float(velocity[0]),
            "VY": float(velocity[1]),
            "VZ": float(velocity[2]),
        })

    return friendlies


def generate_friendly_series(num_steps, dt=1.0):
    """生成我方同类型飞机编队时间序列。"""
    friendly_series = []

    center0 = np.array([0.0, 0.0, 8.0], dtype=float)

    # 我方同构战斗机编队巡航速度：Mach 0.85
    friendly_mach = 0.85
    velocity = np.array([friendly_mach * 0.340, 0.0, 0.0], dtype=float)

    for t in range(num_steps):
        current_center = center0 + velocity * (t * dt)
        friendlies = create_homogeneous_fighter_formation(
            center=current_center,
            velocity=velocity,
            formation_type="diamond"
        )
        friendly_series.append(friendlies)

    return friendly_series


def vec_from_state(state):
    """从状态字典中读取三维位置和速度。"""
    pos = np.array([state["X"], state["Y"], state["Z"]], dtype=float)
    vel = np.array([state["VX"], state["VY"], state["VZ"]], dtype=float)
    return pos, vel


def safe_norm(x, eps=1e-8):
    return max(np.linalg.norm(x), eps)


def angle_deg(a, b):
    """计算两个向量夹角，单位 degree。"""
    na = safe_norm(a)
    nb = safe_norm(b)
    cos_val = np.dot(a, b) / (na * nb)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def closing_speed(enemy_pos, enemy_vel, ref_pos, ref_vel):
    """
    敌方目标相对参考对象的接近速度。
    >0 表示正在接近。
    """
    rel_pos = enemy_pos - ref_pos
    rel_vel = enemy_vel - ref_vel
    return float(-np.dot(rel_pos, rel_vel) / safe_norm(rel_pos))


def route_shortcut(enemy_pos, enemy_vel, ref_pos):
    """
    航路捷径：敌方速度方向所在直线到参考点的最小距离。
    """
    return float(np.linalg.norm(np.cross(ref_pos - enemy_pos, enemy_vel)) / safe_norm(enemy_vel))


def time_to_boundary(distance, vc, radius=5.0, eps=1e-6):
    """
    到达安全/威胁边界时间。
    radius 是安全半径或威胁边界半径，单位 km。
    """
    if vc <= eps:
        return 1e6
    return float(max(distance - radius, 0.0) / vc)

def route_shortcut_relative(enemy_pos, enemy_vel, ref_pos, ref_vel):
    """
    运动参考对象下的相对航路捷径。

    原固定点模型：
        只看敌方速度 enemy_vel 到固定点 ref_pos 的最小距离。

    运动单机/运动编队模型：
        应看敌方相对我方的相对速度 rel_vel = enemy_vel - ref_vel。
    """
    rel_pos = enemy_pos - ref_pos
    rel_vel = enemy_vel - ref_vel

    rel_speed = np.linalg.norm(rel_vel)
    if rel_speed <= 1e-8:
        return float(np.linalg.norm(rel_pos))

    return float(np.linalg.norm(np.cross(rel_pos, rel_vel)) / rel_speed)


def heading_angle_relative(enemy_pos, enemy_vel, ref_pos, ref_vel):
    """
    运动参考对象下的相对接近角。

    角度越小，说明敌方相对速度方向越指向我方平台，威胁越高。
    """
    rel_pos = enemy_pos - ref_pos
    rel_vel = enemy_vel - ref_vel

    if np.linalg.norm(rel_vel) <= 1e-8:
        return 90.0

    # ref_pos - enemy_pos = -rel_pos，表示从敌方指向我方的方向
    return angle_deg(rel_vel, -rel_pos)


def time_to_boundary_relative(enemy_pos, enemy_vel, ref_pos, ref_vel, radius=5.0):
    """
    运动目标相对运动参考对象进入威胁边界的时间。

    解方程：
        || rel_pos + rel_vel * t || = radius

    其中：
        rel_pos = enemy_pos - ref_pos
        rel_vel = enemy_vel - ref_vel
    """
    rel_pos = enemy_pos - ref_pos
    rel_vel = enemy_vel - ref_vel

    current_distance = np.linalg.norm(rel_pos)
    if current_distance <= radius:
        return 0.0

    a = float(np.dot(rel_vel, rel_vel))
    b = float(2.0 * np.dot(rel_pos, rel_vel))
    c = float(np.dot(rel_pos, rel_pos) - radius ** 2)

    if a <= 1e-10:
        return 1e6

    disc = b ** 2 - 4.0 * a * c
    if disc < 0:
        return 1e6

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    candidates = [t for t in (t1, t2) if t >= 0.0]
    if len(candidates) == 0:
        return 1e6

    return float(min(candidates))


def build_pairwise_target(enemy_state, friendly_state):
    """
    构造敌方目标 E_j 对我方单机 A_i 的动态相对威胁特征。

    注意：
    这里所有运动学指标都针对“运动中的我方单机”重新计算，
    不再是相对固定资产/固定防御点。
    """
    e_pos, e_vel = vec_from_state(enemy_state)
    f_pos, f_vel = vec_from_state(friendly_state)

    rel_pos = e_pos - f_pos
    rel_vel = e_vel - f_vel

    # 1. 动态相对距离
    d_ij = safe_norm(rel_pos)

    # 2. 动态接近速度，>0 表示敌方正在接近我方单机
    vc_ij = closing_speed(e_pos, e_vel, f_pos, f_vel)

    # 3. 相对接近角：用敌我相对速度，而不是敌方绝对速度
    heading_ij = heading_angle_relative(e_pos, e_vel, f_pos, f_vel)

    # 4. 相对航路捷径：用敌我相对运动轨迹，而不是敌方绝对航迹线
    shortcut_ij = route_shortcut_relative(e_pos, e_vel, f_pos, f_vel)

    # 5. 相对高度差
    height_diff = abs(e_pos[2] - f_pos[2])

    # 6. 把接近速度换算成 Mach，用于 DBN 的 Speed 输入
    speed_mach = max(vc_ij, 0.0) / 0.340

    # 7. 相对运动 TTC
    threat_radius = 100.0

    ttc_ij = time_to_boundary_relative(
        e_pos, e_vel,
        f_pos, f_vel,
        radius=threat_radius
    )

    return {
        "Time": enemy_state.get("Time", None),
        "Target_ID": enemy_state.get("Target_ID", None),
        "Name": enemy_state.get("Name", "Unknown"),
        "Type": enemy_state.get("Type", None),
        "Jamming": enemy_state.get("Jamming", None),

        "Distance": d_ij,
        "Speed": speed_mach,
        "Height": height_diff,
        "Heading": heading_ij,
        "Shortcut": shortcut_ij,

        "ClosingSpeed": vc_ij,
        "RelSpeed": float(np.linalg.norm(rel_vel)),
        "TTC": ttc_ij,
        "Friendly_ID": friendly_state.get("Aircraft_ID", None),
        "Friendly_Name": friendly_state.get("Name", "Friendly"),
    }

def compute_formation_state(friendlies):
    """计算同构飞机编队中心、平均速度和编队半径。"""
    positions = []
    velocities = []

    for f in friendlies:
        p, v = vec_from_state(f)
        positions.append(p)
        velocities.append(v)

    positions = np.stack(positions)
    velocities = np.stack(velocities)

    center = np.mean(positions, axis=0)
    velocity = np.mean(velocities, axis=0)
    radius = float(np.max(np.linalg.norm(positions - center, axis=1)))

    return {
        "center": center,
        "velocity": velocity,
        "radius": radius,
        "positions": positions,
        "velocities": velocities,
    }


def build_formation_target(enemy_state, friendlies):
    """
    构造敌方目标 E_j 对整个同构飞机编队的动态相对威胁特征。

    注意：
    这里的 Distance、Speed、Height、Heading、Shortcut
    都是针对“运动中的我方编队”重新定义的，不再是固定资产指标。
    """
    e_pos, e_vel = vec_from_state(enemy_state)
    form = compute_formation_state(friendlies)

    center = form["center"]
    f_vel = form["velocity"]
    R_f = form["radius"]

    # 1. 敌方目标到编队中心的距离
    d_center = safe_norm(e_pos - center)

    # 2. 敌方目标到编队外包络边界的距离
    d_boundary = max(d_center - R_f, 0.0)

    distances = []
    shortcuts = []
    ttcs = []
    closing_speeds = []

    # 3. 遍历编队内每架飞机，计算敌方目标相对每架运动飞机的指标
    for f in friendlies:
        f_pos, f_v = vec_from_state(f)

        d_i = safe_norm(e_pos - f_pos)
        vc_i = closing_speed(e_pos, e_vel, f_pos, f_v)

        # 核心修改 1：相对运动航路捷径
        s_i = route_shortcut_relative(e_pos, e_vel, f_pos, f_v)

        # 核心修改 2：相对运动 TTC
        threat_radius = 100.0

        ttc_i = time_to_boundary_relative(
            e_pos, e_vel,
            f_pos, f_v,
            radius=threat_radius
        )
        distances.append(d_i)
        shortcuts.append(s_i)
        ttcs.append(ttc_i)
        closing_speeds.append(vc_i)

    d_min = float(np.min(distances))
    s_min = float(np.min(shortcuts))
    ttc_min = float(np.min(ttcs))
    vc_max = float(np.max(closing_speeds))

    # 4. 敌方目标相对编队中心的接近速度
    vc_form = closing_speed(e_pos, e_vel, center, f_vel)

    # 5. 核心修改 3：相对运动接近角
    heading_form = heading_angle_relative(e_pos, e_vel, center, f_vel)

    # 6. 高度差：敌方目标与编队中心高度差
    height_diff = abs(e_pos[2] - center[2])

    # 7. 覆盖比例：敌方目标进入多少架飞机的有效威胁范围
    R_eff = 120.0

    distances_arr = np.array(distances, dtype=float)
    values = np.array([f.get("Value", 1.0) for f in friendlies], dtype=float)

    cover_ratio = float(
        np.sum(values * (distances_arr < R_eff)) / np.sum(values)
    )

    # 8. 编队级 Distance 输入
    #    仍然采用“到编队边界”和“到最近成员”中的较小值。
    formation_distance = min(d_boundary, d_min)

    # 9. 编队级 Speed 输入
    #    这里可以用相对编队中心接近速度，也可以用最大单机接近速度。
    #    为了更体现编队成员被快速接近的风险，建议用二者较大值。
    formation_closing_speed = max(vc_form, vc_max, 0.0)
    formation_speed = formation_closing_speed / 0.340

    return {
        "Time": enemy_state.get("Time", None),
        "Target_ID": enemy_state.get("Target_ID", None),
        "Name": enemy_state.get("Name", "Unknown"),
        "Type": enemy_state.get("Type", None),
        "Jamming": enemy_state.get("Jamming", None),

        # 输入 DBN-TOPSIS 的编队相对指标
        "Distance": formation_distance,
        "Speed": formation_speed,
        "Height": height_diff,
        "Heading": heading_form,
        "Shortcut": s_min,

        # 调试与论文解释用的编队指标
        "D_center": d_center,
        "D_boundary": d_boundary,
        "D_min": d_min,
        "S_min": s_min,
        "TTC_min": ttc_min,
        "VC_form": vc_form,
        "VC_max": vc_max,
        "CoverRatio": cover_ratio,
        "FormationRadius": R_f,
    }


def topsis_closeness(prob_matrix):
    """返回未归一化的 TOPSIS 贴近度。"""
    ideal_best = np.array([1.0, 0.0, 0.0])
    ideal_worst = np.array([0.0, 0.0, 1.0])

    scores = []
    for i in range(prob_matrix.shape[0]):
        d_plus = np.sqrt(np.sum((prob_matrix[i] - ideal_best) ** 2))
        d_minus = np.sqrt(np.sum((prob_matrix[i] - ideal_worst) ** 2))
        c_i = d_minus / (d_plus + d_minus + 1e-10)
        scores.append(c_i)

    return np.array(scores, dtype=float)


def normalize_scores(scores, eps=1e-10):
    scores = np.asarray(scores, dtype=float)
    s = np.sum(scores)
    if s <= eps:
        return np.ones_like(scores) / len(scores)
    return scores / s

# ================= 3. AR(p) 缺失数据预处理与辅助函数 =================
def ar_p_imputation(series, max_p=20, nonnegative=False):
    import numpy as np
    n = len(series)
    imputed_series = series.copy()
    nan_indices = np.where(np.isnan(series))[0]
    if len(nan_indices) == 0: return imputed_series
    
    first_nan = nan_indices[0]
    train_data = series[:first_nan]
    
    if len(train_data) < max_p + 1:
        for i in nan_indices: imputed_series[i] = imputed_series[i-1] if i > 0 else 0
        return imputed_series
        
    # ================= 终极折中方案：带阻尼衰减的趋势预测 =================
    # 1. 计算一阶差分（速度）
    diff_data = np.diff(train_data)
    
    # 2. 估算消失前的瞬时速度 (近期窗口平均) 和 长期平均速度
    window = min(10, len(diff_data))
    recent_velocity = np.mean(diff_data[-window:]) # 消失前那一刻的真实速度
    long_term_velocity = np.mean(diff_data)        # 整个历史的平均速度
    
    # 3. 提取用于 AR 建模的残差
    diff_centered = diff_data - long_term_velocity  
    
    p = min(max(8, max_p), len(diff_centered) - 1)
    N = len(diff_centered)
    
    A = np.zeros((N - p, p))
    phi = np.zeros(N - p)
    for t in range(p, N):
        A[t-p, :] = diff_centered[t-p:t][::-1] 
        phi[t-p] = diff_centered[t]
        
    try:
        a = np.linalg.inv(A.T @ A + np.eye(p) * 1e-4) @ A.T @ phi
    except:
        a = np.zeros(p)
        
    pred_diffs = list(diff_centered)
    
    # 阻尼系数：控制从瞬时速度向长期速度衰减的快慢 (0.9 表示衰减较慢，弧度更平滑)
    damping_factor = 0.95 
    current_velocity = recent_velocity
    
    for count, i in enumerate(nan_indices):
        hist = np.array(pred_diffs[-p:][::-1])
        pred_c = np.dot(a, hist)
        pred_c = np.clip(pred_c, -0.1, 0.1) 
        pred_diffs.append(pred_c)
        
        # 核心逻辑：当前步的速度 = 衰减后的瞬时速度 + 逐渐增加权重的长期速度
        current_velocity = current_velocity * damping_factor + long_term_velocity * (1 - damping_factor)
        
        # 预测距离 = 上一秒距离 + (动态速度) + AR微观波动
        pred_value = imputed_series[i-1] + pred_c + current_velocity
        imputed_series[i] = max(0.0, pred_value) if nonnegative else pred_value
        
    return imputed_series
def introduce_multiple_missing_blocks(series, missing_configs, start_time=0):
    """
    模拟多目标、多维度的复合电磁干扰场景
    :param missing_configs: 包含多个字典的列表，例如 [{'target_idx': 1, 'feature': 'Distance', 'start': 200, 'end': 250}, ...]
    """
    for cfg in missing_configs:
        start_idx = cfg['start'] - start_time
        end_idx = cfg['end'] - start_time
        t_idx = cfg['target_idx']
        feat = cfg['feature']
        
        # 确保索引在合法范围内
        for i in range(max(0, start_idx), min(len(series), end_idx + 1)):
            series[i][t_idx][feat] = np.nan
    return series

def apply_ar_imputation_to_multiple(series, missing_configs):
    """对复合缺失场景中的所有受影响特征应用 AR(p) 独立填补"""
    # 提取所有受影响的 (目标索引, 特征名) 唯一组合，避免重复填补
    unique_targets_feats = set((cfg['target_idx'], cfg['feature']) for cfg in missing_configs)
    
    for t_idx, feat in unique_targets_feats:
        vals = np.array([step_data[t_idx][feat] for step_data in series])
        nonnegative = feat in {'Distance', 'Shortcut', 'Height', 'Speed'}
        imputed_vals = ar_p_imputation(vals, nonnegative=nonnegative)
        for i, step_data in enumerate(series):
            step_data[t_idx][feat] = imputed_vals[i]
    return series

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

    def ds_correct_id_evidence_by_type_fusion(
        self,
        model,
        raw_enemy_state,
        sensor_reliability=0.70,
        discounted_reliability=0.30,
        conflict_discount_th=0.45
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

        kin_type = type_names[int(np.argmax(p_kin))]
        sensor_score = type_score.get(sensor_type, 60)
        kin_score = type_score.get(kin_type, 60)

        # 只在 D-S 冲突大，且运动学支持的类型威胁等级高于传感器类型时，
        # 才折扣传感器 Type。这样可以处理 Missile->UAV、Fighter->Bomber，
        # 同时避免 Missile 等高威胁目标在正常时刻被运动学噪声错误降级。
        if K_type > conflict_discount_th and kin_score > sensor_score:
            p_sensor_discounted = self.type_sensor_mass(
                sensor_type,
                reliability=discounted_reliability
            )
            p_for_id, K_type_after = self.ds_combine_prob(p_sensor_discounted, p_kin)
            ds_action = 'discount_sensor_type_by_DS'
        else:
            # 不满足折扣条件：保留原始传感器 Type 证据，避免正常时刻被 D-S 扰动。
            K_type_after = K_type
            p_for_id = np.zeros(len(type_names), dtype=float)
            if sensor_type in type_names:
                p_for_id[type_names.index(sensor_type)] = 1.0
            else:
                p_for_id[:] = 1.0 / len(type_names)
            ds_action = 'keep_sensor_type_by_DS'

        corrected_id_ev = np.zeros(3, dtype=float)
        for idx, tp in enumerate(type_names):
            id_ev_tp = model.fuzzify_triangle(type_score[tp], 30, 60, 90)[::-1]
            corrected_id_ev += p_for_id[idx] * id_ev_tp

        corrected_id_ev = corrected_id_ev / (np.sum(corrected_id_ev) + 1e-12)

        return corrected_id_ev, {
            'sensor_type': sensor_type,
            'fused_type': type_names[int(np.argmax(p_for_id))],
            'K_type': K_type,
            'K_type_after': K_type_after,
            'ds_action': ds_action,
            'p_sensor': p_sensor,
            'p_kin': p_kin,
            'p_fused': p_for_id,
        }


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
                vc_factor = max(vc, 0.0) / 0.340

                local_factor = (
                    1.0
                    + 0.35 * ttc_factor
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

            # 注意：这里不归一化，因为后面 c_max/c_avg/c_soft 会统一归一化。
            pair_scores[i, :] = raw_pair_scores * local_factors_i
            
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

        form_scores_raw = topsis_closeness(form_posteriors)
        form_scores = normalize_scores(form_scores_raw)

        # 编队结构修正：覆盖比例越大、TTC越小，整体威胁越高
        structure_factors = []
        for ft in form_targets_debug:
            cover = ft.get("CoverRatio", 0.0)
            ttc = ft.get("TTC_min", 1e6)

            ttc_factor = 1.0 / (1.0 + ttc / 100.0)
            cover_factor = cover

            factor = 1.0 + 0.3 * cover_factor + 0.3 * ttc_factor
            structure_factors.append(factor)

        structure_factors = np.array(structure_factors)
        form_scores = normalize_scores(form_scores * structure_factors)

        # 单机威胁矩阵聚合
        c_max = np.max(pair_scores, axis=0)
        c_avg = np.mean(pair_scores, axis=0)
        c_soft = tau * np.log(np.mean(np.exp(pair_scores / tau), axis=0) + 1e-10)

        c_max = normalize_scores(c_max)
        c_avg = normalize_scores(c_avg)
        c_soft = normalize_scores(c_soft)

        lambda_max, lambda_avg, lambda_soft = lambdas
        agg_scores = (
            lambda_max * c_max
            + lambda_avg * c_avg
            + lambda_soft * c_soft
        )
        agg_scores = normalize_scores(agg_scores)

        # 最终整体编队威胁度
        total_scores = beta * form_scores + (1.0 - beta) * agg_scores
        total_scores = normalize_scores(total_scores)

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

# ================= 5.5 面向编队的对抗进攻场景模型 =================
FRIENDLY_CENTER0_ATTACK = np.array([0.0, 0.0, 8.0], dtype=float)
FRIENDLY_VEL_ATTACK = np.array([0.85 * 0.340, 0.0, 0.0], dtype=float)

FORMATION_ATTACK_OFFSETS = {
    "Leader": np.array([0.0, 0.0, 0.0], dtype=float),
    "LeftWing": np.array([-25.0, -18.0, 0.0], dtype=float),
    "RightWing": np.array([25.0, -18.0, 0.0], dtype=float),
    "RearGuard": np.array([0.0, -45.0, 0.0], dtype=float),
    "Center": np.array([0.0, -20.25, 0.0], dtype=float),
}


def friendly_attack_point(time_s, role="Center"):
    """
    给定时间和攻击对象，返回敌方目标的瞄准点。

    这里直接使用我方编队的已知巡航轨迹：
    formation center = [0,0,8] + [0.85 Mach,0,0] * t
    再叠加编队成员相对位置。
    """
    return (
        FRIENDLY_CENTER0_ATTACK
        + FRIENDLY_VEL_ATTACK * float(time_s)
        + FORMATION_ATTACK_OFFSETS.get(role, FORMATION_ATTACK_OFFSETS["Center"])
    )


class DirectedAttackTarget:
    """
    直接面向我方编队/指定编队成员的敌方进攻目标模型。

    目的：
    - 让敌方目标从不同方向、不同初始位置向我方编队进攻；
    - 避免原球坐标机动模型中“看起来无目的乱动”的问题；
    - 保持 get_state()/update() 接口不变，不影响 DBN、AR、D-S 和可视化导出流程。
    """

    def __init__(
        self,
        tid,
        name,
        t_type,
        jamming,
        init_pos,
        speed_mach,
        attack_role="Center",
        lead_time=80.0,
        lateral_amp=0.0,
        lateral_freq=0.025,
        attack_altitude=None,
    ):
        self.tid = tid
        self.name = name
        self.type = t_type
        self.jamming = jamming

        self.pos = np.asarray(init_pos, dtype=float)
        self.speed_mach = float(speed_mach)
        self.speed = self.speed_mach * 0.340

        self.attack_role = attack_role
        self.lead_time = float(lead_time)
        self.lateral_amp = float(lateral_amp)
        self.lateral_freq = float(lateral_freq)
        self.attack_altitude = None if attack_altitude is None else float(attack_altitude)

        self.time = 0.0

        # 初始速度：指向未来一段时间后的攻击点
        self.v_cart = self._compute_velocity(self.time)

    def _compute_velocity(self, time_s):
        aim_point = friendly_attack_point(time_s + self.lead_time, self.attack_role)

        # 关键修正：
        # 敌方目标应当“在水平面上朝编队/某成员进攻”，
        # 但不同目标类型的飞行高度不应被强行拉到我方战斗机高度。
        # 否则低空巡航导弹会为了瞄准 8 km 高度的编队而持续爬升，
        # 导致 D-S 的 Speed/Height 类型证据不再支持 Missile，
        # 从而无法修正 Missile -> UAV。
        if self.attack_altitude is not None:
            aim_point = aim_point.copy()
            aim_point[2] = self.attack_altitude

        to_aim = aim_point - self.pos

        dist = np.linalg.norm(to_aim)
        if dist < 1e-8:
            main_dir = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            main_dir = to_aim / dist

        # 轻微横向机动，只用于让轨迹更自然，但仍保持“朝向编队进攻”的主趋势。
        if self.lateral_amp > 0.0:
            horizontal = np.array([main_dir[0], main_dir[1], 0.0], dtype=float)
            h_norm = np.linalg.norm(horizontal)
            if h_norm > 1e-8:
                horizontal = horizontal / h_norm
                perp = np.array([-horizontal[1], horizontal[0], 0.0], dtype=float)
                main_dir = main_dir + self.lateral_amp * np.sin(self.lateral_freq * time_s) * perp
                main_dir = main_dir / (np.linalg.norm(main_dir) + 1e-12)

        return self.speed * main_dir

    def get_state(self, current_time):
        center = friendly_attack_point(current_time, "Center")
        rel_pos = self.pos - center
        rel_vel = self.v_cart - FRIENDLY_VEL_ATTACK

        distance = np.linalg.norm(rel_pos)

        if np.linalg.norm(rel_vel) > 1e-8 and np.linalg.norm(rel_pos) > 1e-8:
            # heading 越小，说明越朝向我方编队。
            cos_val = np.dot(rel_vel, -rel_pos) / (
                np.linalg.norm(rel_vel) * np.linalg.norm(rel_pos)
            )
            heading = float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))
            shortcut = float(np.linalg.norm(np.cross(rel_pos, rel_vel)) / np.linalg.norm(rel_vel))
        else:
            heading = 90.0
            shortcut = float(distance)

        return {
            "Time": current_time,
            "Target_ID": self.tid,
            "Name": self.name,
            "Type": self.type,
            "Jamming": self.jamming,

            # 原 DBN-TOPSIS 所需字段
            "Height": round(max(0.001, float(self.pos[2])), 3),
            "Speed": round(self.speed_mach, 3),
            "Distance": round(max(0.1, float(distance)), 2),
            "Heading": round(heading, 2),
            "Shortcut": round(max(0.0, shortcut), 2),

            # 编队相对运动评估与三维可视化所需字段
            "X": float(self.pos[0]),
            "Y": float(self.pos[1]),
            "Z": float(self.pos[2]),
            "VX": float(self.v_cart[0]),
            "VY": float(self.v_cart[1]),
            "VZ": float(self.v_cart[2]),

            # 可视化/解释用字段
            "AttackRole": self.attack_role,
        }

    def update(self, dt=1.0):
        self.v_cart = self._compute_velocity(self.time)
        self.pos = self.pos + self.v_cart * dt
        self.time += dt


# ================= 6. 融合实验主控 =================
if __name__ == "__main__":
    # 使用异构球坐标系模型初始化目标
    spherical_targets = [
        # T1：巡航导弹，从左前低空方向突防，攻击编队领机
        DirectedAttackTarget(
            0, "T1(BGM-109C)", "Missile", "Mid",
            init_pos=np.array([285.0, -145.0, 0.20]),
            speed_mach=0.70,
            attack_role="Leader",
            lead_time=75.0,
            lateral_amp=0.04,
            attack_altitude=None,  # 导弹瞄准空中编队成员，允许末端爬升；D-S 不再只依赖低空高度识别导弹
        ),

        # T2：巡航导弹，从右前低空方向进入，攻击右翼机
        DirectedAttackTarget(
            1, "T2(AGM-86B)", "Missile", "Strong",
            init_pos=np.array([260.0, 165.0, 0.25]),
            speed_mach=0.68,
            attack_role="RightWing",
            lead_time=70.0,
            lateral_amp=0.05,
            attack_altitude=None,  # 导弹瞄准空中编队成员，允许末端爬升；D-S 不再只依赖低空高度识别导弹
        ),

        # T3：武装直升机，从左后低空接近后卫机，低速低威胁对照
        DirectedAttackTarget(
            2, "T3(AH-64A)", "Heli", "Mid",
            init_pos=np.array([145.0, -235.0, 1.60]),
            speed_mach=0.38,
            attack_role="RearGuard",
            lead_time=60.0,
            lateral_amp=0.03,
            attack_altitude=1.20,
        ),

        # T4：F-16，从右前高空高速斜插，主要威胁右翼机
        DirectedAttackTarget(
            3, "T4(F-16C)", "Fighter", "Mid",
            init_pos=np.array([390.0, 135.0, 9.20]),
            speed_mach=1.45,
            attack_role="RightWing",
            lead_time=95.0,
            lateral_amp=0.10,
            attack_altitude=8.80,
        ),

        # T5：F-22，从左前高空高速突防，主要威胁左翼机，后续误识别为 Bomber
        DirectedAttackTarget(
            4, "T5(F-22)", "Fighter", "Strong",
            init_pos=np.array([355.0, -185.0, 9.80]),
            speed_mach=1.85,
            attack_role="LeftWing",
            lead_time=95.0,
            lateral_amp=0.08,
            attack_altitude=9.20,
        ),

        # T6：B-52H，从正前方高空进入，持续压迫整个编队中心
        DirectedAttackTarget(
            5, "T6(B-52H)", "Bomber", "Strong",
            init_pos=np.array([430.0, 20.0, 8.50]),
            speed_mach=0.72,
            attack_role="Center",
            lead_time=120.0,
            lateral_amp=0.02,
            attack_altitude=8.50,
        ),

        # T7：MQ-9，从右侧高空侦察接近，低中威胁目标
        DirectedAttackTarget(
            6, "T7(MQ-9)", "UAV", "Strong",
            init_pos=np.array([185.0, 245.0, 12.50]),
            speed_mach=0.32,
            attack_role="Leader",
            lead_time=90.0,
            lateral_amp=0.04,
            attack_altitude=12.50,
        ),
    ]
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

    # [新增步骤]: 生成我方同类型飞机编队时间序列
    print("\n正在生成我方同类型飞机编队队形数据...")
    friendly_series = generate_friendly_series(len(full_time_series), dt=1.0)
    print(f"  -> 我方编队飞机数量: {len(friendly_series[0])}")

    # [步骤 2]: 模拟复杂的复合电磁干扰场景 (实施“开局 EMP 致盲攻击”)
    print("\n[突发战况] 防空雷达遭遇敌方高强度、全频段 EMP 覆盖干扰：")
    missing_configs = [
        # T2 三维位置和速度缺失
        {'target_idx': 1, 'feature': 'X',  'start': 80, 'end': 140},
        {'target_idx': 1, 'feature': 'Y',  'start': 80, 'end': 140},
        {'target_idx': 1, 'feature': 'Z',  'start': 80, 'end': 140},
        {'target_idx': 1, 'feature': 'VX', 'start': 80, 'end': 140},
        {'target_idx': 1, 'feature': 'VY', 'start': 80, 'end': 140},
        {'target_idx': 1, 'feature': 'VZ', 'start': 80, 'end': 140},

        # T5 三维位置和速度缺失
        {'target_idx': 4, 'feature': 'X',  'start': 80, 'end': 140},
        {'target_idx': 4, 'feature': 'Y',  'start': 80, 'end': 140},
        {'target_idx': 4, 'feature': 'Z',  'start': 80, 'end': 140},
        {'target_idx': 4, 'feature': 'VX', 'start': 80, 'end': 140},
        {'target_idx': 4, 'feature': 'VY', 'start': 80, 'end': 140},
        {'target_idx': 4, 'feature': 'VZ', 'start': 80, 'end': 140},
    ]

    inaccurate_time_series = copy.deepcopy(full_time_series)

    inaccurate_time_series = introduce_multiple_missing_blocks(inaccurate_time_series, missing_configs, start_time=0) 

    for cfg in missing_configs:
        t_name = spherical_targets[cfg['target_idx']].name
        print(f"  -> 目标 {t_name} 在 {cfg['start']}s - {cfg['end']}s 期间丢失 [{cfg['feature']}] 数据！")
    
    
    # [步骤 3]: 修改 inaccurate_time_series 中目标的类型
    print("\n[突发情况]己方传感器无法准确识别敌方目标类型：")
    # 修改 T1 和 T2 的类型为 "UAV"，修改 T6 的类型为 "Heli"
    misidentification_configs = [
        {'target_idx': 0, 'feature': "Type", 'misidentification': "UAV", 'start': 200, 'end': 300},
        {'target_idx': 1, 'feature': "Type", 'misidentification': "UAV", 'start': 200, 'end': 260},
        {'target_idx': 4, 'feature': "Type", 'misidentification': "Bomber", 'start': 240, 'end': 300},    
    ]

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
        beta=0.7
    )

    D_records = run_formation_dynamic_assessment(
        model=model,
        time_series=ar_time_series,
        friendly_series=friendly_series,
        start_time=0,
        use_DS=True,
        allow_missing=False,
        beta=0.7
    )

    # ================= 以上是你代码的 步骤 5 =================

    # ================= 为论文核心表格直接生成排版数据 =================
    print("\n" + "★"*80)
    print(">>> 【表 3：本文融合框架在 75s (受欺骗干扰期间) 的目标状态概率与威胁度】 <<<")
    print(f"{'Target ID':<10} | {'P(H)':<8} | {'P(M)':<8} | {'P(L)':<8} | {'Threat Degree'}")
    print("-" * 65)
    target_time = 75
    for i in range(7):
        p_h = D_records[target_time]['posteriors'][i][0]
        p_m = D_records[target_time]['posteriors'][i][1]
        p_l = D_records[target_time]['posteriors'][i][2]
        score = D_records[target_time]['scores'][i]
        print(f"T{i+1:<9} | {p_h:<8.4f} | {p_m:<8.4f} | {p_l:<8.4f} | {score:.4f}")

    debug_time = 90

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

    print("\n>>> 单机视角下，每架飞机认为最危险的敌方目标：")
    for i in range(pair_mat.shape[0]):
        local_rank = np.argsort(pair_mat[i])[::-1] + 1
        print(f"Friendly_Fighter_{i+1}: " + " > ".join([f"T{r}" for r in local_rank]))

    print("★"*80 + "\n")


    debug_time = 90

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

        # ================= 相对运动指标检查：验证编队指标是否真的按运动目标重新计算 =================
    debug_time = 90

    # 更稳妥：按 record["time"] 找记录，而不是直接用 D_records[90]
    debug_record = next(record for record in D_records if record["time"] == debug_time)

    print("\n" + "★"*80)
    print(f">>> 【{debug_time}s 时刻：相对运动指标检查】 <<<")
    print("Target | D_center | D_min | Heading | S_min | TTC_min | VC_form | VC_max")
    print("-" * 90)

    for j, ft in enumerate(debug_record["formation_targets"]):
        print(
            f"T{j+1:<5} | "
            f"{ft['D_center']:<8.2f} | "
            f"{ft['D_min']:<6.2f} | "
            f"{ft['Heading']:<7.2f} | "
            f"{ft['S_min']:<6.2f} | "
            f"{ft['TTC_min']:<8.2f} | "
            f"{ft.get('VC_form', np.nan):<7.3f} | "
            f"{ft.get('VC_max', np.nan):<7.3f}"
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
    debug_times = [210, 225, 250, 290, 300]

    print("\n" + "★"*100)
    print(">>> 【D-S 类型空间融合诊断检查】 <<<")
    print("Time | Target | SensorType | FusedType | K_type | K_after | Action")
    print("-" * 100)

    for ds_debug_time in debug_times:
        ds_debug_record = next(record for record in D_records if record["time"] == ds_debug_time)

        for j in range(7):
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
    print(">>> 【表 4：同构飞机编队场景下异常干扰区间内的敌方目标整体威胁排序对比】 <<<")
    print(f"{'Time(s)':<8} | {'无干扰对照组 (真实排序)':<40} | {'传统 DBN-TOPSIS (无纠偏)':<40} | {'同构编队 AR-DS-DBN-TOPSIS (稳健排序)'}")
    print("-" * 130)
    
    # 定义需要打印的所有关键时刻
    timeline = [
        75, 
        90, 110, 130,  # 缺维干扰区间
        150, 
        210, 225, 250, 290, # 误识干扰区间
        300, 375, 450, 525, 600
    ]
    
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
    import os

    # 1. 创建结果存放文件夹
    save_dir = "results_fig"
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

    VIS_STEP = 5
    D_by_time = {record["time"]: record for record in D_records}

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

        enemies_vis = []
        for j, enemy in enumerate(enemy_states):
            truth = enemy_truth[j]
            ds_info = rec.get("ds_k", {}).get(j, {})

            enemies_vis.append({
                "id": int(j + 1),
                "label": f"T{j + 1}",
                "name": enemy.get("Name", f"T{j + 1}"),
                "attack_role": enemy.get("AttackRole", None),
                "sensor_type": enemy.get("Type", None),
                "true_type": truth.get("Type", None),
                "fused_type": ds_info.get("fused_type", enemy.get("Type", None)),
                "ds_action": ds_info.get("ds_action", "None"),
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
            })

        friendlies_vis = []
        for i, f in enumerate(friendlies):
            friendlies_vis.append({
                "id": int(i + 1),
                "label": f"F{i + 1}",
                "name": f.get("Name", f"Friendly_Fighter_{i + 1}"),
                "role": f.get("Role", ""),
                "value": _safe_float(f.get("Value", 1.0)),
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
                "missing_configs": _active_configs(missing_configs, t),
                "misidentification_configs": _active_configs(misidentification_configs, t),
            }
        })

    visual_payload = {
        "meta": {
            "description": "3D formation threat-assessment visualization data",
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
    # ============================================================

    # 保存 Spearman deviation 逐时刻结果。
    spearman_deviation_df.to_csv(os.path.join(save_dir, 'Table_Spearman_Deviation.csv'), index=False, encoding='utf-8-sig')
    print(f"Spearman deviation 逐时刻结果已保存到 {save_dir}/Table_Spearman_Deviation.csv")

    # 提取有 DS 修复的实验 C 的 K 值记录
    # 把 C_records 换成经过 AR 修复的 D_records
    k_records = {r['time']: r['ds_k'] for r in D_records if 'ds_k' in r}

    # 2. 依次绘图并保存到指定文件夹
    print("正在绘制并保存图表...")
    
    # 【修改 2】强制对齐时间轴，解决 Figure 2 错位问题
    plot_utils.plot_emp_interference(full_time_series, target_indices=[1, 4], feature='X', missing_configs=missing_configs, start_time=0)
    plt.savefig(os.path.join(save_dir, 'Figure1_EMP_Interference.png'), dpi=300, bbox_inches='tight')
    
    # 【修改 1】传入受损轨迹 A宇宙，预测轨迹 B宇宙
    plot_utils.plot_ar_imputation(full_time_series, inaccurate_time_series, ar_time_series, target_idx=1, feature='X', missing_configs=missing_configs, start_time=0)
    plt.savefig(os.path.join(save_dir, 'Figure2_AR_Imputation.png'), dpi=300, bbox_inches='tight')
    
    plot_utils.plot_ds_conflict_diagnosis(k_records, target_idx=0, spoof_configs=misidentification_configs)
    plt.savefig(os.path.join(save_dir, 'Figure3_DS_Diagnosis.png'), dpi=300, bbox_inches='tight')
    
    plot_utils.plot_threat_scores_baseline(full_records)
    plt.savefig(os.path.join(save_dir, 'Figure4_Baseline_Threat.png'), dpi=300, bbox_inches='tight')
    
    # 把 A 和 C 换成真正使用了 AR 填补的 B 和 D
    plot_utils.plot_ds_restoration(full_records, B_records, D_records, target_idx=0, spoof_configs=misidentification_configs)
    plt.savefig(os.path.join(save_dir, 'Figure5_DS_Restoration.png'), dpi=300, bbox_inches='tight')
    
    plot_utils.plot_performance_metrics(summary_df1, summary_df2)
    plt.savefig(os.path.join(save_dir, 'Figure6_Performance_Metrics.png'), dpi=300, bbox_inches='tight')

    print(f"所有高清图表已自动保存到 {save_dir}/ 文件夹下！")
    
    # 最后一次性展示所有窗口 (如果不想在屏幕上弹出，只需注释掉下面这行即可)
    plt.show()