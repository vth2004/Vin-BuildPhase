from typing import Literal
from pydantic import BaseModel


class TemporalReviewRequest(BaseModel):
    source: Literal["human", "temporal"]
