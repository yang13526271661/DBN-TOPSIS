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

class IntentTarget:
    """
    五类开环意图目标模型：
    1. Attack：攻击突防
    2. Interference：电子干扰压制
    3. Reconnaissance：侦察监视
    4. Feint：佯攻欺骗
    5. EscortEvasion：护航规避/护航防御
    """

    def __init__(
        self,
        tid,
        name,
        t_type,
        intent,
        jamming,
        init_pos,
        speed_mach,
        attack_role="Center",
        lead_time=80.0,
        lateral_amp=0.0,
        lateral_freq=0.025,
        attack_altitude=None,
        feint_switch_distance=180.0,
        protected_role="Center",
        terminal_time=None,
    ):
        self.tid = tid
        self.name = name
        self.type = t_type
        self.intent = intent
        self.jamming = jamming

        self.pos = np.asarray(init_pos, dtype=float)
        self.speed_mach = float(speed_mach)
        self.speed = self.speed_mach * 0.340

        self.attack_role = attack_role
        self.lead_time = float(lead_time)
        self.lateral_amp = float(lateral_amp)
        self.lateral_freq = float(lateral_freq)
        self.attack_altitude = None if attack_altitude is None else float(attack_altitude)

        self.feint_switch_distance = float(feint_switch_distance)
        self.protected_role = protected_role
        self.terminal_time = terminal_time

        self.time = 0.0
        self.v_cart = self._compute_velocity(self.time)

    def _normalize(self, v):
        n = np.linalg.norm(v)
        if n < 1e-8:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        return v / n

    def _rotate_horizontal(self, direction, angle_deg):
        angle = np.radians(angle_deg)
        c, s = np.cos(angle), np.sin(angle)

        x, y, z = direction
        rotated = np.array([
            c * x - s * y,
            s * x + c * y,
            z
        ], dtype=float)

        return self._normalize(rotated)

    def _attack_velocity(self, time_s):
        """
        攻击意图：
        - 导弹类目标：始终指向我方编队未来拦截点，不做末端掉头/远离；
        - 为避免导弹在仿真末段贴近或穿越编队，通过初始位置、速度和 terminal_time 控制其到达时间；
        - 战斗机类目标仍可保留小幅侧向机动。
        """

        # ==========================================================
        # 1. 导弹类目标：使用固定未来拦截点
        #    这样导弹轨迹是平滑接近，不会每一秒重新追逐移动目标而产生奇怪末端转向
        # ==========================================================
        if self.type == "Missile":
            if self.terminal_time is None:
                aim_time = time_s + self.lead_time
            else:
                aim_time = self.terminal_time

            aim_point = friendly_attack_point(
                aim_time,
                self.attack_role
            )

            if self.attack_altitude is not None:
                aim_point = aim_point.copy()
                aim_point[2] = self.attack_altitude

            to_aim = aim_point - self.pos
            main_dir = self._normalize(to_aim)

            return self.speed * main_dir

        # ==========================================================
        # 2. 非导弹目标：仍然采用动态预测攻击点
        # ==========================================================
        aim_point = friendly_attack_point(
            time_s + self.lead_time,
            self.attack_role
        )

        if self.attack_altitude is not None:
            aim_point = aim_point.copy()
            aim_point[2] = self.attack_altitude

        to_aim = aim_point - self.pos
        main_dir = self._normalize(to_aim)

        # 战斗机可保留小幅横向机动
        if self.lateral_amp > 0.0:
            horizontal = np.array([main_dir[0], main_dir[1], 0.0], dtype=float)
            h_norm = np.linalg.norm(horizontal)

            if h_norm > 1e-8:
                horizontal = horizontal / h_norm
                perp = np.array([-horizontal[1], horizontal[0], 0.0], dtype=float)

                main_dir = main_dir + self.lateral_amp * np.sin(
                    self.lateral_freq * time_s
                ) * perp

                main_dir = self._normalize(main_dir)

        return self.speed * main_dir

    def _interference_velocity(self, time_s):
        """
        干扰意图：
        - 用于 B-52H 这类大型平台；
        - 不做 S 型机动；
        - 保持远距干扰站位；
        - 对我方编队实施电子压制，但不直接突防。
        """

        # 干扰平台保持在我方编队侧前方/侧方的远距站位
        # 这个站位随我方编队缓慢前移
        desired_offset = np.array([260.0, 180.0, 0.5], dtype=float)
        desired_pos = friendly_attack_point(time_s, "Center") + desired_offset

        to_station = desired_pos - self.pos

        # 轰炸机主要水平机动，高度保持
        to_station[2] = 0.0

        if np.linalg.norm(to_station) < 25.0:
            # 接近干扰站位后，与我方编队大致同向平稳飞行
            forward_dir = FRIENDLY_VEL_ATTACK / (
                np.linalg.norm(FRIENDLY_VEL_ATTACK) + 1e-12
            )
            main_dir = forward_dir
        else:
            # 平稳飞向干扰站位，不做蛇形/S型动作
            main_dir = self._normalize(to_station)

        if self.attack_altitude is not None:
            self.pos[2] += 0.03 * (self.attack_altitude - self.pos[2])

        return self.speed * main_dir

    def _recon_velocity(self, time_s):
        """
        侦察意图：
        - 用于 MQ-9 无人机；
        - 高空、低速、平稳；
        - 在战场侧翼进行大半径巡逻/侦察；
        - 不直接冲向我方编队。
        """

        # 固定一个高空侦察区域，不要跟着我方编队剧烈移动
        orbit_center = FRIENDLY_CENTER0_ATTACK + np.array(
            [160.0, 280.0, 4.5],
            dtype=float
        )

        radial = self.pos - orbit_center
        radial[2] = 0.0

        r = np.linalg.norm(radial)
        if r < 1e-8:
            radial = np.array([1.0, 0.0, 0.0], dtype=float)
            r = 1.0

        radial_dir = radial / r

        # 切向方向，用于平稳盘旋
        tangent_dir = np.array(
            [-radial_dir[1], radial_dir[0], 0.0],
            dtype=float
        )

        # 保持大半径巡逻
        desired_radius = 90.0
        radius_error = r - desired_radius
        radial_correction = -0.25 * radius_error / desired_radius * radial_dir

        main_dir = self._normalize(tangent_dir + radial_correction)

        # 高度缓慢锁定，不允许上下大幅振荡
        if self.attack_altitude is not None:
            self.pos[2] += 0.02 * (self.attack_altitude - self.pos[2])

        main_dir[2] = 0.0
        main_dir = self._normalize(main_dir)

        return self.speed * main_dir
    


    def _feint_velocity(self, time_s):
        """
        佯攻意图：
        - 只建议用于战斗机；
        - 前期模拟攻击；
        - 接近一定距离后平滑侧向脱离；
        - 不使用突然大角度折返。
        """

        aim_point = friendly_attack_point(
            time_s + self.lead_time,
            self.attack_role
        )

        if self.attack_altitude is not None:
            aim_point = aim_point.copy()
            aim_point[2] = self.attack_altitude

        to_aim = aim_point - self.pos
        dist_to_aim = np.linalg.norm(to_aim)
        attack_dir = self._normalize(to_aim)

        # 侧向脱离方向，不是掉头逃跑
        escape_dir = self._rotate_horizontal(attack_dir, 80.0)

        # 平滑过渡，不要突然折线
        transition_width = 70.0
        w = (
            self.feint_switch_distance + transition_width - dist_to_aim
        ) / transition_width
        w = np.clip(w, 0.0, 1.0)

        main_dir = self._normalize(
            (1.0 - w) * attack_dir + w * escape_dir
        )

        return self.speed * main_dir

    def _escort_evasion_velocity(self, time_s):
        """
        护航规避意图：
        - 用于 AH-64A 这类低空慢速目标；
        - 不直接攻击我方编队核心；
        - 向我方航线侧翼靠近；
        - 低空小幅规避，而不是远离战场。
        """

        # 侧翼掩护区域：位于我方编队未来航线的低空侧后方
        screen_point = friendly_attack_point(
            time_s + 80.0,
            "RearGuard"
        ) + np.array([80.0, -120.0, -6.8], dtype=float)

        if self.attack_altitude is not None:
            screen_point[2] = self.attack_altitude

        to_screen = screen_point - self.pos

        # 低空目标主要水平机动
        to_screen[2] = 0.0
        main_dir = self._normalize(to_screen)

        # 小幅蛇形规避，幅度要小
        lateral_dir = self._rotate_horizontal(main_dir, 90.0)
        weave = 0.12 * np.sin(0.12 * time_s) * lateral_dir

        main_dir = self._normalize(main_dir + weave)

        if self.attack_altitude is not None:
            self.pos[2] += 0.05 * (self.attack_altitude - self.pos[2])

        return self.speed * main_dir

    def _compute_velocity(self, time_s):
        if self.intent == "Attack":
            return self._attack_velocity(time_s)
        elif self.intent == "Interference":
            return self._interference_velocity(time_s)
        elif self.intent == "Reconnaissance":
            return self._recon_velocity(time_s)
        elif self.intent == "Feint":
            return self._feint_velocity(time_s)
        elif self.intent == "EscortEvasion":
            return self._escort_evasion_velocity(time_s)
        else:
            return self._attack_velocity(time_s)

    def get_state(self, current_time):
        center = friendly_attack_point(current_time, "Center")
        rel_pos = self.pos - center
        rel_vel = self.v_cart - FRIENDLY_VEL_ATTACK

        distance = np.linalg.norm(rel_pos)

        if np.linalg.norm(rel_vel) > 1e-8 and np.linalg.norm(rel_pos) > 1e-8:
            cos_val = np.dot(rel_vel, -rel_pos) / (
                np.linalg.norm(rel_vel) * np.linalg.norm(rel_pos)
            )
            heading = float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))
            shortcut = float(np.linalg.norm(np.cross(rel_pos, rel_vel)) / np.linalg.norm(rel_vel))
            closing_speed = float(-np.dot(rel_pos, rel_vel) / (distance + 1e-12))
        else:
            heading = 90.0
            shortcut = float(distance)
            closing_speed = 0.0

        return {
            "Time": current_time,
            "Target_ID": self.tid,
            "Name": self.name,
            "Type": self.type,
            "IntentGT": self.intent,
            "Jamming": self.jamming,

            "Height": round(max(0.001, float(self.pos[2])), 3),
            "Speed": round(self.speed_mach, 3),
            "Distance": round(max(0.1, float(distance)), 2),
            "Heading": round(heading, 2),
            "Shortcut": round(max(0.0, shortcut), 2),
            "ClosingSpeed": closing_speed,

            "X": float(self.pos[0]),
            "Y": float(self.pos[1]),
            "Z": float(self.pos[2]),
            "VX": float(self.v_cart[0]),
            "VY": float(self.v_cart[1]),
            "VZ": float(self.v_cart[2]),

            "AttackRole": self.attack_role,
        }

    def update(self, dt=1.0):
        self.v_cart = self._compute_velocity(self.time)
        self.pos = self.pos + self.v_cart * dt
        self.time += dt

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


def create_attack_targets():
    return [
        # T1：BGM-109C 巡航导弹，低空直接攻击
        IntentTarget(
            0,
            "T1(BGM-109C-like)",
            "Missile",
            "Attack",
            "Mid",
            init_pos=np.array([620.0, -190.0, 6.80]),
            speed_mach=0.95,
            attack_role="Leader",
            lead_time=40.0,
            lateral_amp=0.0,
            attack_altitude=None,
            terminal_time=760.0,
        ),

        # T2：AGM-86B 巡航导弹，仍然是攻击，不再佯攻
        IntentTarget(
            1,
            "T2(AGM-86B-like)",
            "Missile",
            "Attack",
            "Mid",
            init_pos=np.array([660.0, 210.0, 7.00]),
            speed_mach=0.90,
            attack_role="RightWing",
            lead_time=40.0,
            lateral_amp=0.0,
            attack_altitude=None,
            terminal_time=780.0,
        ),

        # T3：AH-64A，低空侧翼护航/规避
        IntentTarget(
            2,
            "T3(AH-64A)",
            "Heli",
            "EscortEvasion",
            "Mid",
            init_pos=np.array([160.0, -220.0, 1.20]),
            speed_mach=0.35,
            protected_role="RearGuard",
            attack_altitude=1.20,
        ),

        # T4：F-16C，执行佯攻
        IntentTarget(
            3,
            "T4(F-16C)",
            "Fighter",
            "Feint",
            "Mid",
            init_pos=np.array([390.0, 135.0, 8.80]),
            speed_mach=1.35,
            attack_role="RightWing",
            lead_time=95.0,
            lateral_amp=0.03,
            attack_altitude=8.80,
            feint_switch_distance=210.0,
        ),

        # T5：F-22，高速隐身突防攻击
        IntentTarget(
            4,
            "T5(F-22)",
            "Fighter",
            "Attack",
            "Strong",
            init_pos=np.array([355.0, -185.0, 9.20]),
            speed_mach=1.85,
            attack_role="LeftWing",
            lead_time=95.0,
            lateral_amp=0.03,
            attack_altitude=9.20,
        ),

        # T6：B-52H，远距电子干扰平台
        IntentTarget(
            5,
            "T6(B-52H)",
            "Bomber",
            "Interference",
            "Strong",
            init_pos=np.array([260.0, 220.0, 8.50]),
            speed_mach=0.78,
            attack_altitude=8.50,
        ),

        # T7：MQ-9，高空侦察监视
        IntentTarget(
            6,
            "T7(MQ-9)",
            "UAV",
            "Reconnaissance",
            "Weak",
            init_pos=np.array([120.0, 360.0, 12.50]),
            speed_mach=0.32,
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
