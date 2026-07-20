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
FRIENDLY_AIRCRAFT_CONFIG = [
    {
        "Role": "Leader",
        "AircraftType": "heavy_fighter",
        "Value": 1.6,
        "Vulnerability": 1.0,
        "Maneuverability": 1.0,
        "SensorRange": 260.0,
        "WeaponRange": 150.0,
        "ECMCapability": 0.7,
    },
    {
        "Role": "LeftWing",
        "AircraftType": "stealth_fighter",
        "Value": 1.0,
        "Vulnerability": 0.95,
        "Maneuverability": 1.12,
        "SensorRange": 220.0,
        "WeaponRange": 135.0,
        "ECMCapability": 0.5,
    },
    {
        "Role": "RightWing",
        "AircraftType": "stealth_fighter",
        "Value": 1.0,
        "Vulnerability": 0.95,
        "Maneuverability": 1.12,
        "SensorRange": 220.0,
        "WeaponRange": 135.0,
        "ECMCapability": 0.5,
    },
    {
        "Role": "RearGuard",
        "AircraftType": "escort",
        "Value": 1.2,
        "Vulnerability": 1.05,
        "Maneuverability": 0.98,
        "SensorRange": 210.0,
        "WeaponRange": 145.0,
        "ECMCapability": 0.6,
    },
]

HOMOGENEOUS_FRIENDLY_AIRCRAFT_CONFIG = [
    {
        "Role": "Member",
        "AircraftType": "fighter",
        "Value": 1.0,
        "Vulnerability": 1.0,
        "Maneuverability": 1.0,
        "SensorRange": 230.0,
        "WeaponRange": 140.0,
        "ECMCapability": 0.5,
    },
    {
        "Role": "Member",
        "AircraftType": "fighter",
        "Value": 1.0,
        "Vulnerability": 1.0,
        "Maneuverability": 1.0,
        "SensorRange": 230.0,
        "WeaponRange": 140.0,
        "ECMCapability": 0.5,
    },
    {
        "Role": "Member",
        "AircraftType": "fighter",
        "Value": 1.0,
        "Vulnerability": 1.0,
        "Maneuverability": 1.0,
        "SensorRange": 230.0,
        "WeaponRange": 140.0,
        "ECMCapability": 0.5,
    },
    {
        "Role": "Member",
        "AircraftType": "fighter",
        "Value": 1.0,
        "Vulnerability": 1.0,
        "Maneuverability": 1.0,
        "SensorRange": 230.0,
        "WeaponRange": 140.0,
        "ECMCapability": 0.5,
    },
]

FORMATION_MODE_DESCRIPTIONS = {
    "fixed_homogeneous": "Fixed homogeneous formation",
    "dynamic_homogeneous": "Dynamic homogeneous formation",
    "dynamic_heterogeneous": "Dynamic heterogeneous leader-wingman formation",
    "dynamic_heterogeneous_degraded": (
        "Dynamic heterogeneous formation with progressive F2 capability degradation"
    ),
}


FRIENDLY_DEGRADATION_EVENT = {
    "aircraft_index": 1,
    "label": "F2",
    "start_time": 280.0,
    "end_time": 340.0,
    "final_vulnerability": 1.35,
    "final_maneuverability": 0.72,
    "description": "F2 left-wing local damage",
}


def apply_friendly_degradation(friendlies, time_s, formation_mode):
    """Apply the scene-6 capability event without changing aircraft motion."""
    for friendly in friendlies:
        friendly["BaselineVulnerability"] = float(friendly["Vulnerability"])
        friendly["BaselineManeuverability"] = float(friendly["Maneuverability"])
        friendly["CapabilityState"] = "Healthy"
        friendly["DamageLevel"] = 0.0
        friendly["CapabilityEvent"] = ""

    if formation_mode != "dynamic_heterogeneous_degraded":
        return

    event = FRIENDLY_DEGRADATION_EVENT
    friendly = friendlies[event["aircraft_index"]]
    start_time = event["start_time"]
    end_time = event["end_time"]

    if time_s <= start_time:
        progress = 0.0
    elif time_s >= end_time:
        progress = 1.0
    else:
        progress = (time_s - start_time) / (end_time - start_time)
        progress = progress * progress * (3.0 - 2.0 * progress)

    vulnerability_0 = friendly["BaselineVulnerability"]
    maneuverability_0 = friendly["BaselineManeuverability"]
    friendly["Vulnerability"] = float(
        vulnerability_0
        + progress * (event["final_vulnerability"] - vulnerability_0)
    )
    friendly["Maneuverability"] = float(
        maneuverability_0
        + progress * (event["final_maneuverability"] - maneuverability_0)
    )
    friendly["DamageLevel"] = float(progress)
    friendly["CapabilityEvent"] = event["description"]

    if progress >= 1.0:
        friendly["CapabilityState"] = "Degraded"
    elif progress > 0.0:
        friendly["CapabilityState"] = "Degrading"


FORMATION_OFFSETS = {
    "CruiseWedge": [
        np.array([0.0, 0.0, 0.0]),
        np.array([-24.0, -30.0, 0.0]),
        np.array([-24.0, 30.0, 0.0]),
        np.array([-62.0, 0.0, 0.0]),
    ],
    "WideSearchLine": [
        np.array([0.0, 0.0, 0.0]),
        np.array([-8.0, -82.0, 0.0]),
        np.array([-8.0, 82.0, 0.0]),
        np.array([-58.0, 0.0, 0.0]),
    ],
    "ProtectiveBox": [
        np.array([0.0, 0.0, 0.0]),
        np.array([34.0, -48.0, 0.0]),
        np.array([34.0, 48.0, 0.0]),
        np.array([-70.0, 0.0, 0.0]),
    ],
}


def formation_stage_at_time(time_s):
    if time_s < 200.0:
        return "CruiseWedge"
    if time_s < 400.0:
        return "WideSearchLine"
    return "ProtectiveBox"


def formation_offsets_at_time(time_s, transition_seconds=120.0):
    stages = ["CruiseWedge", "WideSearchLine", "ProtectiveBox"]
    transition_points = [200.0, 400.0]

    for idx, switch_time in enumerate(transition_points):
        half = transition_seconds / 2.0
        if switch_time - half <= time_s <= switch_time + half:
            prev_stage = stages[idx]
            next_stage = stages[idx + 1]
            ratio = (time_s - (switch_time - half)) / transition_seconds
            ratio = float(np.clip(ratio, 0.0, 1.0))
            ratio = ratio * ratio * (3.0 - 2.0 * ratio)
            return [
                (1.0 - ratio) * old + ratio * new
                for old, new in zip(FORMATION_OFFSETS[prev_stage], FORMATION_OFFSETS[next_stage])
            ], f"{prev_stage}->{next_stage}"

    stage = formation_stage_at_time(time_s)
    return FORMATION_OFFSETS[stage], stage


def local_formation_offset_to_world(offset, velocity):
    """
    Convert local formation offset to world coordinates.

    Local offset convention:
    - offset[0]: forward/backward along the leader velocity direction.
    - offset[1]: left/right lateral direction in the horizontal plane.
    - offset[2]: vertical offset.
    """
    speed_xy = np.linalg.norm(velocity[:2])
    if speed_xy <= 1e-8:
        forward = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        forward = np.array([velocity[0], velocity[1], 0.0], dtype=float) / speed_xy

    lateral = np.array([-forward[1], forward[0], 0.0], dtype=float)
    vertical = np.array([0.0, 0.0, 1.0], dtype=float)

    return offset[0] * forward + offset[1] * lateral + offset[2] * vertical


def create_homogeneous_fighter_formation(
    center=np.array([0.0, 0.0, 8.0]),
    velocity=np.array([0.85 * 0.340, 0.0, 0.0]),
    formation_type="dynamic",
    time_s=0.0,
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


def create_homogeneous_fighter_formation(
    center=np.array([0.0, 0.0, 8.0]),
    velocity=np.array([0.85 * 0.340, 0.0, 0.0]),
    formation_type="dynamic",
    time_s=0.0,
    heterogeneous=True,
):
    """Create a four-aircraft formation with selectable role/capability settings."""
    if formation_type in ("dynamic", "Dynamic"):
        offsets, active_formation_type = formation_offsets_at_time(time_s)
    else:
        normalized_type = {
            "diamond": "CruiseWedge",
            "wedge": "CruiseWedge",
            "Wedge": "CruiseWedge",
            "line": "WideSearchLine",
            "line_abreast": "WideSearchLine",
            "LineAbreast": "WideSearchLine",
            "defensive_spread": "ProtectiveBox",
            "DefensiveSpread": "ProtectiveBox",
        }.get(str(formation_type), formation_type)
        if normalized_type not in FORMATION_OFFSETS:
            raise ValueError(f"Unknown formation type: {formation_type}")
        offsets = FORMATION_OFFSETS[normalized_type]
        active_formation_type = normalized_type

    friendlies = []
    aircraft_config = FRIENDLY_AIRCRAFT_CONFIG if heterogeneous else HOMOGENEOUS_FRIENDLY_AIRCRAFT_CONFIG
    for i, offset in enumerate(offsets):
        cfg = aircraft_config[i]
        world_offset = local_formation_offset_to_world(offset, velocity)
        friendlies.append({
            "Aircraft_ID": i,
            "Name": f"Friendly_Fighter_{i + 1}",
            "Role": cfg["Role"],
            "AircraftType": cfg["AircraftType"],
            "Value": cfg["Value"],
            "Vulnerability": cfg["Vulnerability"],
            "Maneuverability": cfg["Maneuverability"],
            "SensorRange": cfg["SensorRange"],
            "WeaponRange": cfg["WeaponRange"],
            "ECMCapability": cfg["ECMCapability"],
            "FormationType": active_formation_type,
            "OffsetForward": float(offset[0]),
            "OffsetLateral": float(offset[1]),
            "OffsetVertical": float(offset[2]),
            "OffsetX": float(world_offset[0]),
            "OffsetY": float(world_offset[1]),
            "OffsetZ": float(world_offset[2]),
            "Type": "FriendlyFighter",
            "X": float(center[0] + world_offset[0]),
            "Y": float(center[1] + world_offset[1]),
            "Z": float(center[2] + world_offset[2]),
            "VX": float(velocity[0]),
            "VY": float(velocity[1]),
            "VZ": float(velocity[2]),
        })

    return friendlies


def generate_friendly_series(num_steps, dt=1.0, formation_mode="dynamic_heterogeneous"):
    """Generate a friendly formation time series for a selected experiment mode."""
    if formation_mode not in FORMATION_MODE_DESCRIPTIONS:
        valid = ", ".join(sorted(FORMATION_MODE_DESCRIPTIONS))
        raise ValueError(f"Unknown formation_mode '{formation_mode}'. Valid modes: {valid}")

    friendly_series = []
    center0 = np.array([0.0, 0.0, 8.0], dtype=float)
    velocity = np.array([0.85 * 0.340, 0.0, 0.0], dtype=float)
    heterogeneous = formation_mode in (
        "dynamic_heterogeneous",
        "dynamic_heterogeneous_degraded",
    )

    for t in range(num_steps):
        time_s = t * dt
        current_center = center0 + velocity * time_s
        formation_type = "dynamic" if formation_mode != "fixed_homogeneous" else "CruiseWedge"
        friendlies = create_homogeneous_fighter_formation(
            center=current_center,
            velocity=velocity,
            formation_type=formation_type,
            time_s=time_s,
            heterogeneous=heterogeneous,
        )
        apply_friendly_degradation(friendlies, time_s, formation_mode)
        for f in friendlies:
            f["FormationMode"] = formation_mode
            f["FormationModeDescription"] = FORMATION_MODE_DESCRIPTIONS[formation_mode]
        friendly_series.append(friendlies)

    if num_steps >= 2 and dt > 0:
        for t in range(num_steps):
            for i in range(len(friendly_series[t])):
                if t == 0:
                    prev_state = friendly_series[t][i]
                    next_state = friendly_series[t + 1][i]
                    delta_t = dt
                elif t == num_steps - 1:
                    prev_state = friendly_series[t - 1][i]
                    next_state = friendly_series[t][i]
                    delta_t = dt
                else:
                    prev_state = friendly_series[t - 1][i]
                    next_state = friendly_series[t + 1][i]
                    delta_t = 2.0 * dt

                friendly_series[t][i]["VX"] = float((next_state["X"] - prev_state["X"]) / delta_t)
                friendly_series[t][i]["VY"] = float((next_state["Y"] - prev_state["Y"]) / delta_t)
                friendly_series[t][i]["VZ"] = float((next_state["Z"] - prev_state["Z"]) / delta_t)

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
        "IntentGT": enemy_state.get("IntentGT", None),
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


def compute_formation_risk_factor(enemy_state, friendlies):
    """
    Compute a formation-structure risk factor without duplicating distance/TTC.

    The factor measures whether the current formation geometry exposes the
    leader and whether wingmen shield the leader from the enemy-target line.
    It changes when the formation offsets change, but it does not use the
    formation name as a direct score.
    """
    e_pos, e_vel = vec_from_state(enemy_state)
    positions = []
    velocities = []
    leader_idx = 0

    for idx, f in enumerate(friendlies):
        f_pos, _ = vec_from_state(f)
        _, f_vel = vec_from_state(f)
        positions.append(f_pos)
        velocities.append(f_vel)
        if f.get("Role") == "Leader":
            leader_idx = idx

    positions = np.stack(positions)
    velocities = np.stack(velocities)
    leader_pos = positions[leader_idx]
    leader_vel = velocities[leader_idx]
    distances = np.linalg.norm(positions - e_pos, axis=1)
    d_min = float(np.min(distances))
    d_leader = float(distances[leader_idx])

    exposure_scale = 80.0
    relative_leader_exposure = 1.0 / (1.0 + max(d_leader - d_min, 0.0) / exposure_scale)

    enemy_to_leader = leader_pos - e_pos
    line_len_sq = float(np.dot(enemy_to_leader, enemy_to_leader))
    shielding = 0.0

    if line_len_sq > 1e-8:
        shield_width = 35.0
        for idx, f_pos in enumerate(positions):
            if idx == leader_idx:
                continue

            alpha = float(np.dot(f_pos - e_pos, enemy_to_leader) / line_len_sq)
            if alpha <= 0.05 or alpha >= 0.98:
                continue

            closest_on_line = e_pos + alpha * enemy_to_leader
            lateral_distance = float(np.linalg.norm(f_pos - closest_on_line))
            if lateral_distance > shield_width:
                continue

            role_weight = 1.0
            role = friendlies[idx].get("Role", "")
            if role == "RearGuard":
                role_weight = 0.85

            candidate = role_weight * (1.0 - lateral_distance / shield_width)
            shielding = max(shielding, candidate)

    shielding = float(np.clip(shielding, 0.0, 1.0))
    shielding_loss = 1.0 - shielding

    vc_leader = max(closing_speed(e_pos, e_vel, leader_pos, leader_vel), 0.0)
    heading_leader = heading_angle_relative(e_pos, e_vel, leader_pos, leader_vel)
    closing_factor = float(np.clip(vc_leader / 0.340, 0.0, 1.0))
    heading_factor = 1.0 / (1.0 + heading_leader / 45.0)
    approach_relevance = float(np.clip(0.55 * heading_factor + 0.45 * closing_factor, 0.0, 1.0))

    structure_risk = float(np.clip(
        0.55 * relative_leader_exposure + 0.45 * shielding_loss,
        0.0,
        1.0,
    ))
    formation_risk_factor = float(np.clip(
        structure_risk * approach_relevance,
        0.0,
        1.0,
    ))

    return {
        "D_leader": d_leader,
        "RelativeLeaderExposure": float(relative_leader_exposure),
        "Shielding": shielding,
        "ShieldingLoss": float(shielding_loss),
        "LeaderClosingSpeed": float(vc_leader),
        "LeaderHeading": float(heading_leader),
        "ApproachRelevance": approach_relevance,
        "StructureRisk": structure_risk,
        "FormationRiskFactor": formation_risk_factor,
        "FormationType": friendlies[0].get("FormationType", "Unknown") if friendlies else "Unknown",
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
    # Keep the signed trend for intent inference.  It becomes negative only
    # after the target is receding from both the formation center and every
    # member, which is the key observation for attack-to-feint recognition.
    signed_formation_closing_speed = max(vc_form, vc_max)

    # Threat-speed evidence remains nonnegative: receding speed must not be
    # interpreted as additional approach severity by the threat-level branch.
    formation_speed = max(signed_formation_closing_speed, 0.0) / 0.340
    formation_risk = compute_formation_risk_factor(enemy_state, friendlies)

    return {
    "Time": enemy_state.get("Time", None),
    "Target_ID": enemy_state.get("Target_ID", None),
    "Name": enemy_state.get("Name", "Unknown"),
    "Type": enemy_state.get("Type", None),
    "IntentGT": enemy_state.get("IntentGT", None),
    "Jamming": enemy_state.get("Jamming", None),

    # 输入 DBN-TOPSIS 的编队相对指标
    "Distance": formation_distance,
    "Speed": formation_speed,
    "Height": height_diff,
    "Heading": heading_form,
    "Shortcut": s_min,

    # 新增：给 DBN 意图节点使用的统一字段
    "ClosingSpeed": signed_formation_closing_speed,
    "TTC": ttc_min,

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
    "D_leader": formation_risk["D_leader"],
    "RelativeLeaderExposure": formation_risk["RelativeLeaderExposure"],
    "Shielding": formation_risk["Shielding"],
    "ShieldingLoss": formation_risk["ShieldingLoss"],
    "LeaderClosingSpeed": formation_risk["LeaderClosingSpeed"],
    "LeaderHeading": formation_risk["LeaderHeading"],
    "ApproachRelevance": formation_risk["ApproachRelevance"],
    "StructureRisk": formation_risk["StructureRisk"],
    "FormationRiskFactor": formation_risk["FormationRiskFactor"],
    "FormationType": formation_risk["FormationType"],
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

