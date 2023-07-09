"""Tests for bsa algorithm."""

from typing import Any

import numpy as np
import pytest
from scipy.optimize import Bounds, OptimizeResult

from optimize_it.bsa import BSA


@pytest.fixture
def bsa() -> BSA:
    return BSA(generations=500, dim_rate=2, seed=5)


@pytest.fixture
def bounds() -> Bounds:
    return Bounds(lb=[1, 2, 4, 5], ub=[9, 8, 7, 10])


def test_bsa_over_shekel(bsa: BSA, shekel: dict[str, Any]) -> None:
    """Example test with parametrization."""
    result = bsa.run(shekel["func"], bounds=Bounds(lb=np.array([0] * 4), ub=np.array([10] * 4)))
    assert shekel["fun"] == round(result.fun, 4)
    assert isinstance(result, OptimizeResult)


def test_generate_population(bsa: BSA, bounds: Bounds) -> None:
    pop = bsa.generate_population(bounds)
    lb_residual, ub_residual = bounds.residual(pop)
    assert pop.shape == (bsa.pop_size, bounds.ub.size)
    assert (lb_residual >= 0).all()
    assert (ub_residual >= 0).all()


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [(None, 1.3237), ("standard brownian-walk", 1.3237), ("brownian-walk", 3.0689)],
)
def test_get_scale_factor(bsa: BSA, strategy: str, expected: float) -> None:
    data = {} if not strategy else {"strategy": strategy}
    scale_number = bsa.get_scale_factor(**data)
    assert round(scale_number, 4) == expected


def test_boundary_control(bsa: BSA) -> None:
    pop = np.array(
        [
            [4.0, 6.0, 7.0, 3.0],
            [0.7, 3.6, 0.6, 11.9],
            [2.0, 4.3, 0.1, 2.0],
        ]
    )
    bounds = Bounds(lb=[1, 2, 4, 5], ub=[9, 8, 7, 10])
    controlled_pop = bsa.boundary_control(pop, bounds)
    lb_residual, ub_residual = bounds.residual(controlled_pop)
    assert (lb_residual >= 0).all()
    assert (ub_residual >= 0).all()


def test_change_one(bsa: BSA) -> None:
    row = np.zeros(10)
    new_row = bsa.change_one(row)
    assert new_row.sum() == 1


def test_change_multiple(bsa: BSA) -> None:
    row = np.zeros(10)
    new_row = bsa.change_multiple(row)
    assert new_row.sum() == 5
