"""Tests for generic functions."""

import numpy as np
import numpy.typing as npt
import pytest
from scipy.optimize import Bounds

from optimize_it.functions import (
    apply_random_boundary_control,
    apply_simple_boundary_control,
    generate_random_population,
    get_indexes_trespassing_bounds,
)


@pytest.fixture
def rand_state() -> np.random.RandomState:
    return np.random.RandomState(5)


@pytest.fixture
def bounds() -> Bounds:
    return Bounds(lb=[1, 2, 4, 5], ub=[9, 8, 7, 10])


@pytest.fixture
def sample_pop() -> npt.NDArray[np.float64]:
    return np.array(
        [
            [4.0, 6.0, 7.0, 3.0],
            [0.7, 3.6, 0.6, 11.9],
            [2.0, 4.3, 0.1, 2.0],
        ]
    )


def _check_in_bounds(pop: npt.NDArray[np.float64], bounds: Bounds) -> None:
    lb_residual, ub_residual = bounds.residual(pop)
    assert (lb_residual >= 0).all()
    assert (ub_residual >= 0).all()


def test_generate_random_population(bounds: Bounds, rand_state: np.random.RandomState) -> None:
    pop = generate_random_population(30, bounds, rand_state)
    assert pop.shape == (30, bounds.ub.size)
    _check_in_bounds(pop, bounds)


def test_random_boundary_control(
    bounds: Bounds, sample_pop: npt.NDArray[np.float64], rand_state: np.random.RandomState
) -> None:
    controlled_pop = apply_random_boundary_control(sample_pop.copy(), bounds, rand_state)
    _check_in_bounds(controlled_pop, bounds)
    lb_negative_indexes, ub_negative_indexes = get_indexes_trespassing_bounds(sample_pop, bounds)
    assert not (controlled_pop[lb_negative_indexes] == bounds.lb[lb_negative_indexes[1]]).all()
    assert not (controlled_pop[ub_negative_indexes] == bounds.ub[ub_negative_indexes[1]]).all()


def test_simple_boundary_control(bounds: Bounds, sample_pop: npt.NDArray[np.float64]) -> None:
    controlled_pop = apply_simple_boundary_control(sample_pop.copy(), bounds)
    _check_in_bounds(controlled_pop, bounds)
    lb_negative_indexes, ub_negative_indexes = get_indexes_trespassing_bounds(sample_pop, bounds)
    assert (controlled_pop[lb_negative_indexes] == bounds.lb[lb_negative_indexes[1]]).all()
    assert (controlled_pop[ub_negative_indexes] == bounds.ub[ub_negative_indexes[1]]).all()
