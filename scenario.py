import numpy as np

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


# ================= ???? =================
def create_attack_targets():
    return [
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


def get_missing_configs():
    return [
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


def get_misidentification_configs():
    return [
        {'target_idx': 0, 'feature': "Type", 'misidentification': "UAV", 'start': 200, 'end': 300},
        {'target_idx': 1, 'feature': "Type", 'misidentification': "UAV", 'start': 200, 'end': 260},
        {'target_idx': 4, 'feature': "Type", 'misidentification': "Bomber", 'start': 240, 'end': 300},    
    ]
