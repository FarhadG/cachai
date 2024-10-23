from dataclasses import dataclass
from enum import Enum

import cachai.utils.constants as C


class ObservationType(Enum):
    HIT = C.HIT
    MISS = C.MISS
    STALE = C.STALE
    VALID_TTL = C.VALID_TTL


@dataclass
class CacheObservation:
    observation_time: int
    observation_type: ObservationType
    hits: int
    y_prev: float
    info: dict = {}
