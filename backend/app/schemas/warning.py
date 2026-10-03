from __future__ import annotations

from pydantic import BaseModel, Field


class ReviewWarning(BaseModel):
    review: str = Field(pattern="^(keep|use_suggestion|manual|skip)$")
