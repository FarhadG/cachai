from enum import Enum

import cachai.utils.constants as C


class ObservationType(Enum):
    HIT = C.HIT
    MISS = C.MISS
    STALE = C.STALE
    VALID_TTL = C.VALID_TTL
