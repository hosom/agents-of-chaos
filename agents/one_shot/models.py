"""Structured response contract for analytical math problems."""

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class Quantity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    value: FiniteFloat


class MathSolution(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    answer: str
    results: list[Quantity] = Field(min_length=1)
    method: list[str] = Field(min_length=1)
    verification: list[str] = Field(min_length=1)
