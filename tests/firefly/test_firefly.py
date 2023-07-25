"""Tests for bsa algorithm."""

from typing import Any

import numpy as np
import pytest
from scipy.optimize import Bounds, OptimizeResult

from optimize_it.firefly import Firefly
from optimize_it.wrapper import ObjectiveFunWrapper


@pytest.fixture
def firefly() -> Firefly:
    return Firefly(generations=250, pop_size=60, gamma=0.7, seed=10)


def test_firefly_over_shekel(firefly: Firefly, shekel: dict[str, Any]) -> None:
    result = firefly.run(shekel["func"], bounds=Bounds(lb=np.array([0] * 4), ub=np.array([10] * 4)))
    assert shekel["fun"] == round(result.fun, 4)
    assert isinstance(result, OptimizeResult)


def test_get_fireflies_distance(firefly: Firefly) -> None:
    first = np.array([1, -2, 3])
    second = np.array([7, 6, 3])
    assert firefly.get_fireflies_distance(first, second) == 10


@pytest.mark.parametrize(
    ("distance", "expected"),
    [(0.5, 0.8716), (1.0, 0.5973)],
)
def test_get_attractiveness_beta(distance: float, expected: float, firefly: Firefly) -> None:
    assert round(firefly.get_attractiveness_beta(distance), 4) == expected


def test_calculate_light_intensity(firefly: Firefly) -> None:
    func_wrapped = ObjectiveFunWrapper(lambda x: sum(x))
    pop = np.array([[1, 2, 3], [4, 5, 6]])
    light = firefly.calculate_light_intensity(func_wrapped, pop)
    assert light[0] == 6
    assert light[1] == 15


def test_get_alpha_generator(firefly: Firefly) -> None:
    firefly.alpha_decrease = False
    alpha_generator = firefly.get_alpha_generator(0.5)
    assert next(alpha_generator) == 0.5
    assert next(alpha_generator) == 0.5


def test_get_alpha_decreasing_generator(firefly: Firefly) -> None:
    alpha_generator = firefly.get_alpha_generator(0.5)
    assert next(alpha_generator) == 0.5
    assert next(alpha_generator) < 0.5


def test_move_fireflies(firefly: Firefly) -> None:
    firefly.pop_size = 2
    pop = np.array([[1, 2, 3], [4, 5, 6]])
    light = np.array([1, 2])
    bounds = Bounds(lb=np.array([0] * 3), ub=np.array([10] * 3))
    alpha = 0.5
    new_pop = firefly.move_fireflies(pop.copy(), light, bounds, alpha)
    assert new_pop.shape == pop.shape
    assert (new_pop[0] == pop[0]).all()
    assert not (new_pop[1] == pop[1]).all()
