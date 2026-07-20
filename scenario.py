import numpy as np
from scenario_catalog import (
    DEFAULT_SCENARIO_ID,
    SMALL_SCENES,
    get_big_scenario,
    list_big_scenarios,
)
from dynamics import formation_offsets_at_time, local_formation_offset_to_world

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


FORMATION_ROLE_INDEX = {
    "Leader": 0,
    "LeftWing": 1,
    "RightWing": 2,
    "RearGuard": 3,
}


def friendly_attack_point(time_s, role="Center", dynamic_roles=False):
    """
    给定时间和攻击对象，返回敌方目标的瞄准点。

    这里直接使用我方编队的已知巡航轨迹：
    formation center = [0,0,8] + [0.85 Mach,0,0] * t
    再叠加编队成员相对位置。
    """
    center = FRIENDLY_CENTER0_ATTACK + FRIENDLY_VEL_ATTACK * float(time_s)
    if not dynamic_roles:
        return center + FORMATION_ATTACK_OFFSETS.get(
            role,
            FORMATION_ATTACK_OFFSETS["Center"],
        )

    offsets, _ = formation_offsets_at_time(float(time_s))
    if role == "Center":
        local_offset = np.mean(np.asarray(offsets, dtype=float), axis=0)
    else:
        role_index = FORMATION_ROLE_INDEX.get(role)
        if role_index is None:
            local_offset = np.mean(np.asarray(offsets, dtype=float), axis=0)
        else:
            local_offset = np.asarray(offsets[role_index], dtype=float)

    return center + local_formation_offset_to_world(
        local_offset,
        FRIENDLY_VEL_ATTACK,
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
        intent_switch=None,
        feint_transition_duration=None,
        feint_escape_angle=80.0,
        feint_turn_direction=1.0,
        dynamic_role_targeting=False,
    ):
        self.tid = tid
        self.name = name
        self.type = t_type
        self.intent = intent
        self.configured_intent = intent
        self.intent_switch = intent_switch or {}
        self.jamming = jamming

        self.pos = np.asarray(init_pos, dtype=float)
        self.speed_mach = float(speed_mach)
        self.speed = self.speed_mach * 0.340

        self.attack_role = attack_role
        self.dynamic_role_targeting = bool(dynamic_role_targeting)
        self.lead_time = float(lead_time)
        self.lateral_amp = float(lateral_amp)
        self.lateral_freq = float(lateral_freq)
        self.attack_altitude = None if attack_altitude is None else float(attack_altitude)

        self.feint_switch_distance = float(feint_switch_distance)
        self.feint_transition_duration = (
            None
            if feint_transition_duration is None
            else float(feint_transition_duration)
        )
        self.feint_escape_angle = float(feint_escape_angle)
        self.feint_turn_direction = float(feint_turn_direction)
        self.protected_role = protected_role
        self.terminal_time = terminal_time
        self.intent_switch_activated = False
        self.intent_switch_time = None
        self.feint_entry_direction = None

        self.time = 0.0
        self.v_cart = self._compute_velocity(self.time)

    def current_intent(self, time_s=None):
        """
        Return the current ground-truth intent.

        Intent switching is scenario-driven.  A typical feint target can be
        modeled as Attack during the approach phase and Feint once it reaches
        the planned deception/turn-away window.
        """
        switch = self.intent_switch
        if not switch:
            return self.configured_intent

        trigger = switch.get("trigger", "")
        from_intent = switch.get("from_intent", self.configured_intent)
        to_intent = switch.get("to_intent", self.configured_intent)
        current_time = self.time if time_s is None else float(time_s)

        if trigger == "time":
            switch_time = float(switch.get("time", 0.0))
            triggered = current_time >= switch_time
            if triggered and switch.get("lock_after_switch", False):
                self.intent_switch_activated = True
                if self.intent_switch_time is None:
                    self.intent_switch_time = current_time
                if self.feint_entry_direction is None:
                    aim_point = friendly_attack_point(
                        current_time + self.lead_time,
                        self.attack_role,
                        self.dynamic_role_targeting,
                    )
                    if self.attack_altitude is not None:
                        aim_point = aim_point.copy()
                        aim_point[2] = self.attack_altitude
                    self.feint_entry_direction = self._normalize(aim_point - self.pos)
            return to_intent if triggered else from_intent

        if trigger == "distance_to_attack_point":
            if switch.get("lock_after_switch", False) and self.intent_switch_activated:
                return to_intent

            aim_point = friendly_attack_point(
                current_time + self.lead_time,
                self.attack_role,
                self.dynamic_role_targeting,
            )
            if self.attack_altitude is not None:
                aim_point = aim_point.copy()
                aim_point[2] = self.attack_altitude

            dist_to_aim = float(np.linalg.norm(aim_point - self.pos))
            threshold = float(switch.get("threshold", self.feint_switch_distance))
            triggered = dist_to_aim <= threshold
            if triggered and switch.get("lock_after_switch", False):
                self.intent_switch_activated = True
                if self.intent_switch_time is None:
                    self.intent_switch_time = current_time
                if self.feint_entry_direction is None:
                    self.feint_entry_direction = self._normalize(aim_point - self.pos)
            return to_intent if triggered else from_intent

        return self.configured_intent

    def feint_turn_progress(self, time_s=None):
        """Return a smooth 0-1 turn progress for intent-switch visualization."""
        current_time = self.time if time_s is None else float(time_s)

        if self.feint_transition_duration is not None and self.intent_switch_time is not None:
            raw = (current_time - self.intent_switch_time) / max(
                self.feint_transition_duration,
                1e-6,
            )
            raw = float(np.clip(raw, 0.0, 1.0))
            return raw * raw * (3.0 - 2.0 * raw)

        aim_point = friendly_attack_point(
            current_time + self.lead_time,
            self.attack_role,
            self.dynamic_role_targeting,
        )
        if self.attack_altitude is not None:
            aim_point = aim_point.copy()
            aim_point[2] = self.attack_altitude

        dist_to_aim = float(np.linalg.norm(aim_point - self.pos))
        transition_width = 70.0
        return float(np.clip(
            (self.feint_switch_distance + transition_width - dist_to_aim)
            / transition_width,
            0.0,
            1.0,
        ))

    def intent_phase(self, time_s=None):
        """Describe the configured attack-to-feint phase at the requested time."""
        current_intent = self.current_intent(time_s)
        if not self.intent_switch:
            return current_intent, 0.0

        from_intent = self.intent_switch.get("from_intent", self.configured_intent)
        if current_intent == from_intent:
            return "AttackApproach", 0.0

        progress = self.feint_turn_progress(time_s)
        if progress < 1.0 - 1e-9:
            return "TurnTransition", progress
        return "FeintDeparture", 1.0

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
                self.attack_role,
                self.dynamic_role_targeting,
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
            self.attack_role,
            self.dynamic_role_targeting,
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
        desired_pos = friendly_attack_point(
            time_s,
            "Center",
            self.dynamic_role_targeting,
        ) + desired_offset

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
            self.attack_role,
            self.dynamic_role_targeting,
        )

        if self.attack_altitude is not None:
            aim_point = aim_point.copy()
            aim_point[2] = self.attack_altitude

        to_aim = aim_point - self.pos
        dist_to_aim = np.linalg.norm(to_aim)
        attack_dir = self._normalize(to_aim)

        # 侧向脱离方向，不是掉头逃跑
        if self.feint_transition_duration is not None:
            progress = self.feint_turn_progress(time_s)
            turn_angle = (
                self.feint_turn_direction
                * self.feint_escape_angle
                * progress
            )
            turn_base = (
                self.feint_entry_direction
                if self.feint_entry_direction is not None
                else attack_dir
            )
            main_dir = self._rotate_horizontal(turn_base, turn_angle)
            return self.speed * main_dir

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
            time_s + 120.0,
            self.protected_role,
            self.dynamic_role_targeting,
        ) + np.array([140.0, 180.0, 0.0], dtype=float)

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
        current_intent = self.current_intent(time_s)

        if current_intent == "Attack":
            return self._attack_velocity(time_s)
        elif current_intent == "Interference":
            return self._interference_velocity(time_s)
        elif current_intent == "Reconnaissance":
            return self._recon_velocity(time_s)
        elif current_intent == "Feint":
            return self._feint_velocity(time_s)
        elif current_intent == "EscortEvasion":
            return self._escort_evasion_velocity(time_s)
        else:
            return self._attack_velocity(time_s)

    def get_state(self, current_time):
        current_intent = self.current_intent(current_time)
        intent_phase, turn_progress = self.intent_phase(current_time)
        center = friendly_attack_point(
            current_time,
            "Center",
            self.dynamic_role_targeting,
        )
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
            "IntentGT": current_intent,
            "ConfiguredIntent": self.configured_intent,
            "IntentSwitch": bool(self.intent_switch),
            "IntentPhase": intent_phase,
            "TurnProgress": float(turn_progress),
            "IntentSwitchTime": self.intent_switch_time,
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
            "SmallSceneID": getattr(self, "small_scene_id", None),
            "SmallSceneLabel": getattr(self, "small_scene_label", None),
            "ThreatLevelGT": getattr(self, "threat_level", None),
            "RecommendedDecision": getattr(self, "recommended_decision", None),
            "DecisionReason": getattr(self, "decision_reason", None),
            "CoreFeatures": getattr(self, "core_features", None),
            "BigScenarioID": getattr(self, "big_scenario_id", None),
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


def get_available_scenarios():
    return list_big_scenarios()


def get_scenario_info(scenario_id=DEFAULT_SCENARIO_ID):
    return get_big_scenario(scenario_id)


def _make_target(tid, small_scene_id, spec, scenario_id):
    kwargs = {
        "attack_role": spec.get("attack_role", "Center"),
        "lead_time": spec.get("lead_time", 80.0),
        "lateral_amp": spec.get("lateral_amp", 0.0),
        "lateral_freq": spec.get("lateral_freq", 0.025),
        "attack_altitude": spec.get("attack_altitude", None),
        "feint_switch_distance": spec.get("feint_switch_distance", 180.0),
        "protected_role": spec.get("protected_role", "Center"),
        "terminal_time": spec.get("terminal_time", None),
        "intent_switch": spec.get("intent_switch", None),
        "feint_transition_duration": spec.get("feint_transition_duration", None),
        "feint_escape_angle": spec.get("feint_escape_angle", 80.0),
        "feint_turn_direction": spec.get("feint_turn_direction", 1.0),
        "dynamic_role_targeting": spec.get("dynamic_role_targeting", False),
    }

    target = IntentTarget(
        tid,
        f"T{tid + 1}({spec['name']})",
        spec["target_type"],
        spec["intent"],
        spec["jamming"],
        init_pos=np.array(spec["init_pos"], dtype=float),
        speed_mach=spec["speed_mach"],
        **kwargs,
    )

    target.small_scene_id = small_scene_id
    target.small_scene_label = spec.get("label")
    target.threat_level = spec.get("threat_level")
    target.recommended_decision = spec.get("recommended_decision")
    target.decision_reason = spec.get("decision_reason")
    target.core_features = spec.get("core_features")
    target.big_scenario_id = scenario_id
    return target


def create_attack_targets(scenario_id=DEFAULT_SCENARIO_ID):
    scenario = get_big_scenario(scenario_id)
    targets = []
    for tid, small_scene_id in enumerate(scenario["small_scenes"]):
        targets.append(_make_target(tid, small_scene_id, SMALL_SCENES[small_scene_id], scenario_id))
    return targets


def _target_index_map(scenario_id):
    scenario = get_big_scenario(scenario_id)
    return {small_scene_id: idx for idx, small_scene_id in enumerate(scenario["small_scenes"])}


def get_missing_configs(scenario_id=DEFAULT_SCENARIO_ID):
    scenario = get_big_scenario(scenario_id)
    index_by_scene = _target_index_map(scenario_id)
    configs = []

    for event in scenario.get("missing_events", []):
        target_key = event["target"]
        if target_key not in index_by_scene:
            raise ValueError(f"Missing event refers to target '{target_key}' not used by scenario '{scenario_id}'.")

        for feature in event.get("features", []):
            configs.append({
                "target_idx": index_by_scene[target_key],
                "target_scene": target_key,
                "feature": feature,
                "start": int(event["start"]),
                "end": int(event["end"]),
            })

    return configs


def get_misidentification_configs(scenario_id=DEFAULT_SCENARIO_ID):
    scenario = get_big_scenario(scenario_id)
    index_by_scene = _target_index_map(scenario_id)
    configs = []

    for event in scenario.get("misidentification_events", []):
        target_key = event["target"]
        if target_key not in index_by_scene:
            raise ValueError(f"Misidentification event refers to target '{target_key}' not used by scenario '{scenario_id}'.")

        configs.append({
            "target_idx": index_by_scene[target_key],
            "target_scene": target_key,
            "feature": "Type",
            "misidentification": event["misidentification"],
            "start": int(event["start"]),
            "end": int(event["end"]),
        })

    return configs


def get_scenario_timeline(scenario_id=DEFAULT_SCENARIO_ID):
    scenario = get_big_scenario(scenario_id)
    return list(scenario.get("timeline", [75, 90, 110, 130, 150, 210, 225, 250, 290, 300, 375, 450, 525, 600]))


def get_scenario_debug_time(scenario_id=DEFAULT_SCENARIO_ID):
    scenario = get_big_scenario(scenario_id)
    return int(scenario.get("debug_time", 90))
