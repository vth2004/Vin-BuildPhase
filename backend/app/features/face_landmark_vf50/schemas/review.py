from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PointSourceUpdate(BaseModel):
    source: Literal["human", "model"]


class ResultStatusUpdate(BaseModel):
    status: Literal["pending", "reviewed", "fixed"]


class BatchFixRequest(BaseModel):
    confirm: bool = False
    only_violated: bool = True
    min_nme: float = Field(default=0.035, ge=0.0)
