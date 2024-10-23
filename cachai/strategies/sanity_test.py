import cachai.utils.constants as C
from cachai.utils.models import BaseModel


class SanityTest(BaseModel):
    def predict(self, X, info):
        return info[C.Y_TRUE]

    def observe(self, observation_time, observation_type, hits, y_prev, info):
        pass
