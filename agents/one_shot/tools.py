"""Numerical tools for polynomial and linear-algebra analysis."""

from typing import Annotated

import numpy as np
from copilot import define_tool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    field_validator,
    model_validator,
)

Numbers = Annotated[list[FiniteFloat], Field(min_length=1)]


class NumericalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class LinearSystemInput(NumericalModel):
    matrix: list[Numbers] = Field(
        min_length=1, description="Square coefficient matrix A, in row order."
    )
    vector: Numbers = Field(description="Right-hand-side vector b in A x = b.")

    @model_validator(mode="after")
    def validate_dimensions(self) -> "LinearSystemInput":
        size = len(self.matrix)
        if any(len(row) != size for row in self.matrix):
            raise ValueError("matrix must be square with equally sized rows")
        if len(self.vector) != size:
            raise ValueError("vector length must match the matrix size")
        return self


class LinearSystemResult(NumericalModel):
    solution: list[FiniteFloat]
    max_abs_residual: FiniteFloat
    condition_number: FiniteFloat


class PolynomialRootsInput(NumericalModel):
    coefficients: list[FiniteFloat] = Field(
        min_length=2,
        description="Descending powers: [1, -3, 2] represents x^2 - 3x + 2.",
    )

    @field_validator("coefficients")
    @classmethod
    def validate_leading_coefficient(cls, values: list[float]) -> list[float]:
        if values[0] == 0:
            raise ValueError(
                "the leading polynomial coefficient must be nonzero"
            )
        return values


class ComplexRoot(NumericalModel):
    real: FiniteFloat
    imaginary: FiniteFloat


class PolynomialRootsResult(NumericalModel):
    roots: list[ComplexRoot]
    max_abs_residual: FiniteFloat


class PolynomialEvaluationInput(NumericalModel):
    coefficients: Numbers = Field(
        description="Descending powers: [3, -22, 140] means 3x^2 - 22x + 140."
    )
    x_values: Numbers = Field(
        description="Points at which to evaluate the polynomial."
    )


class PolynomialValue(NumericalModel):
    x: FiniteFloat
    value: FiniteFloat


class PolynomialEvaluationResult(NumericalModel):
    evaluations: list[PolynomialValue]


@define_tool(
    description=(
        "Solve a square nonsingular linear system A x = b numerically. "
        "Returns the solution, maximum absolute residual, and matrix "
        "condition number. Large condition numbers indicate sensitivity "
        "to rounding or input errors."
    ),
    defer="never",
)
def solve_linear_system(params: LinearSystemInput) -> LinearSystemResult:
    matrix = np.asarray(params.matrix, dtype=np.float64)
    vector = np.asarray(params.vector, dtype=np.float64)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        solution = np.linalg.solve(matrix, vector)
        residual = matrix @ solution - vector
        return LinearSystemResult(
            solution=solution.tolist(),
            max_abs_residual=float(np.max(np.abs(residual))),
            condition_number=float(np.linalg.cond(matrix)),
        )


@define_tool(
    description=(
        "Find all numerical roots of a polynomial supplied as "
        "descending-power coefficients. Returns real and imaginary parts "
        "and a residual check. Roots are floating-point approximations, "
        "not symbolic proofs."
    ),
    defer="never",
)
def find_polynomial_roots(
    params: PolynomialRootsInput,
) -> PolynomialRootsResult:
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        roots = np.roots(params.coefficients)
        residuals = np.polyval(params.coefficients, roots)
        return PolynomialRootsResult(
            roots=[
                ComplexRoot(real=float(root.real), imaginary=float(root.imag))
                for root in roots
            ],
            max_abs_residual=float(np.max(np.abs(residuals))),
        )


@define_tool(
    description=(
        "Evaluate a descending-power polynomial at one or more x values. "
        "Use this to check candidates, residual equations, "
        "and boundary values."
    ),
    defer="never",
)
def evaluate_polynomial(
    params: PolynomialEvaluationInput,
) -> PolynomialEvaluationResult:
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        values = np.polyval(params.coefficients, params.x_values)
        return PolynomialEvaluationResult(
            evaluations=[
                PolynomialValue(x=x, value=float(value))
                for x, value in zip(params.x_values, values, strict=True)
            ]
        )


MATH_TOOLS = [solve_linear_system, find_polynomial_roots, evaluate_polynomial]
