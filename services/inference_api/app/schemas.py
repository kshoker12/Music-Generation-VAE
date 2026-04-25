from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ModelType = Literal["plain", "vae", "simple_vae"]


class GenerateBody(BaseModel):
    model_type: ModelType
    attributes: list[list[int]] = Field(..., description="Shape [8,4] bins in [0..7]")
    seed: int = 0
    temperature: float = 1.0
    top_p: float = 0.9
    block_size: int = 1024
    bars_per_sample: int = 8
    ckpt_path: str | None = None

    @field_validator("attributes")
    @classmethod
    def validate_attrs(cls, v: list[list[int]]) -> list[list[int]]:
        if len(v) != 8:
            raise ValueError("attributes must have length 8 (one row per bar)")
        for row in v:
            if len(row) != 4:
                raise ValueError("each attributes row must have length 4")
            for x in row:
                if not (0 <= int(x) <= 7):
                    raise ValueError("bins must be in [0..7]")
        return v

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, v: float) -> float:
        if not (0.0 < float(v) <= 1.0):
            raise ValueError("top_p must be in (0,1]")
        return float(v)

    @field_validator("temperature")
    @classmethod
    def validate_temp(cls, v: float) -> float:
        if float(v) <= 0:
            raise ValueError("temperature must be > 0")
        return float(v)
