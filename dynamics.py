import numpy as np

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

