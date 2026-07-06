from typing import List, Optional

from .bpa import support_to_mass
from .config import IFFConfig
from .conflict import discount_and_refine_masses
from .data import IFFResult, LABELS, Observation
from .ds_fusion import fuse_sequence
from .fuzzy import infer_identity_support
from .normalize import normalized_deltas, observation_deltas


# 对低空目标滑动窗口执行实时敌我识别。
class LowAltitudeIFFRecognizer:
    # 初始化识别器并保存算法配置。
    def __init__(self, config: Optional[IFFConfig] = None):
        self.config = config or IFFConfig()

    # 识别一个目标最近若干时刻观测的融合身份。
    def identify(self, track: List[Observation]) -> IFFResult:
        if not track:
            raise ValueError("track must contain at least one observation")

        window = track[-self.config.window_size :]
        per_timestep = []
        initial_masses = []

        for obs in window:
            deltas = observation_deltas(obs, self.config)
            norm = normalized_deltas(deltas, self.config)

            support, fuzzy_diagnostics = infer_identity_support(norm, self.config)
            mass = support_to_mass(support, self.config, reliability=self.config.source_reliability)

            per_timestep.append(
                {
                    "time": obs.time,
                    "deltas": deltas,
                    "normalized_deltas": norm,
                    "support": support,
                    "reliability": self.config.source_reliability,
                    "initial_mass": mass,
                    "mass": mass,
                    "fuzzy": fuzzy_diagnostics,
                }
            )
            initial_masses.append(mass)

        refined_masses, conflict_diagnostics = discount_and_refine_masses(initial_masses, self.config)
        for item, refined_mass in zip(per_timestep, refined_masses):
            item["mass"] = refined_mass

        fused, fusion_conflicts = fuse_sequence(refined_masses, self.config)
        label = max(LABELS, key=lambda key: fused.get(key, 0.0))
        last = window[-1]
        comprehensive = conflict_diagnostics.get("comprehensive_conflicts", [])
        pair_values = [
            comprehensive[i][j]
            for i in range(len(comprehensive))
            for j in range(i + 1, len(comprehensive))
        ]

        return IFFResult(
            time=last.time,
            target_id=last.target_id,
            name=last.name,
            label=label,
            mass=fused,
            deltas=per_timestep[-1]["deltas"],
            normalized_deltas=per_timestep[-1]["normalized_deltas"],
            window_size=len(window),
            per_timestep=per_timestep,
            diagnostics={
                "conflicts": fusion_conflicts,
                "fusion_conflicts": fusion_conflicts,
                "mean_conflict": sum(pair_values) / len(pair_values) if pair_values else 0.0,
                "used_h2": any(item["deltas"].get("H2") is not None for item in per_timestep),
                **conflict_diagnostics,
            },
        )
