"""Tests for bsa algorithm."""

from typing import Any

import numpy as np
import pytest
from scipy.optimize import Bounds, OptimizeResult

from optimize_it.bsa import BSA


@pytest.fixture
def bsa() -> BSA:
    return BSA(generations=500, dim_rate=2, seed=5)


def test_bsa_over_shekel(bsa: BSA, shekel: dict[str, Any]) -> None:
    """Example test with parametrization."""
    result = bsa.run(shekel["func"], bounds=Bounds(lb=np.array([0] * 4), ub=np.array([10] * 4)))
    assert shekel["fun"] == round(result.fun, 4)
    assert isinstance(result, OptimizeResult)


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [(None, 1.3237), ("standard brownian-walk", 1.3237), ("brownian-walk", 3.0689)],
)
def test_get_scale_factor(bsa: BSA, strategy: str, expected: float) -> None:
    data = {} if not strategy else {"strategy": strategy}
    scale_number = bsa.get_scale_factor(**data)
    assert round(scale_number, 4) == expected


def test_change_one(bsa: BSA) -> None:
    row = np.zeros(10)
    new_row = bsa.change_one(row)
    assert new_row.sum() == 1


def test_change_multiple(bsa: BSA) -> None:
    row = np.zeros(10)
    new_row = bsa.change_multiple(row)
    assert new_row.sum() == 5
