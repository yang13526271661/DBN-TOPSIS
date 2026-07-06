from dataclasses import dataclass, field
from typing import Dict, List, Optional


LABELS = ("FR", "AC", "ST", "FO")
THETA = "Theta"


@dataclass(frozen=True)
# 表示某一时刻目标进入 IFF 算法的一条观测。
class Observation:
    time: int
    target_id: int
    name: str
    H1: Optional[float]
    V: Optional[float]
    C: Optional[float]
    H2: Optional[float] = None
    truth: Optional[str] = None


TargetTrack = List[Observation]


@dataclass
# 表示 IFF 算法对一个目标窗口的融合识别结果。
class IFFResult:
    time: int
    target_id: int
    name: str
    label: str
    mass: Dict[str, float]
    deltas: Dict[str, Optional[float]]
    normalized_deltas: Dict[str, Optional[float]]
    window_size: int
    per_timestep: List[Dict[str, object]] = field(default_factory=list)
    diagnostics: Dict[str, object] = field(default_factory=dict)
