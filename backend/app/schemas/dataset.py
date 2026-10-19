from pydantic import BaseModel, Field


class SequenceConfirmation(BaseModel):
    fps: float | None = Field(default=None, gt=0, le=1000)
    sampling_confirmed: bool = False
    confirmation_source: str = Field(default="user_confirmation", min_length=1, max_length=64)
