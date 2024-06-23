"""Generic functions."""

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds

TupleOfArrays = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]


def get_rand_boolean(rand_state: np.random.RandomState) -> bool:
    return bool(rand_state.randint(2))


def get_indexes_trespassing_bounds(
    pop: npt.NDArray[np.float64], bounds: Bounds
) -> tuple[TupleOfArrays, TupleOfArrays]:
    lb_residual, ub_residual = bounds.residual(pop)
    return np.nonzero(lb_residual < 0), np.nonzero(ub_residual < 0)  # type: ignore [return-value]


def generate_random_population(
    pop_size: int, bounds: Bounds, rand_state: np.random.RandomState
) -> npt.NDArray[np.float64]:
    return (bounds.ub - bounds.lb) * rand_state.random_sample(  # type: ignore [no-any-return]
        (pop_size, bounds.lb.size)
    ) + bounds.lb


def apply_simple_boundary_control(
    pop: npt.NDArray[np.float64], bounds: Bounds
) -> npt.NDArray[np.float64]:
    lb_negative_indexes, ub_negative_indexes = get_indexes_trespassing_bounds(pop, bounds)
    pop[lb_negative_indexes] = bounds.lb[lb_negative_indexes[1]]
    pop[ub_negative_indexes] = bounds.ub[ub_negative_indexes[1]]
    return pop


def apply_random_boundary_control(
    pop: npt.NDArray[np.float64],
    bounds: Bounds,
    rand_state: np.random.RandomState,
) -> npt.NDArray[np.float64]:
    lb_negative_indexes, ub_negative_indexes = get_indexes_trespassing_bounds(pop, bounds)
    pop[lb_negative_indexes] = np.array(
        [
            (
                bounds.lb[i]
                if get_rand_boolean(rand_state)
                else (rand_state.rand() * (bounds.ub[i] - bounds.lb[i]) + bounds.lb[i])
            )
            for i in lb_negative_indexes[1]
        ]
    )
    pop[ub_negative_indexes] = np.array(
        [
            (
                bounds.ub[i]
                if get_rand_boolean(rand_state)
                else (rand_state.rand() * (bounds.ub[i] - bounds.lb[i]) + bounds.lb[i])
            )
            for i in ub_negative_indexes[1]
        ]
    )
    return pop
