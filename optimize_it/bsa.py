"""Backtracking Search Optimization Algorithm (BSA).

Cite this algorithm as;
[1]  P. Civicioglu, "Backtracking Search Optimization Algorithm for 
numerical optimization problems", Applied Mathematics and Computation, 219, 81218144, 2013.
"""

from typing import Any, Callable, Union

import math

import numpy as np
import numpy.typing as npt
from scipy._lib._util import check_random_state
from scipy.optimize import Bounds, OptimizeResult


class ObjectiveFunWrapper:
    def __init__(self, func, *args):
        self.func = func
        self.args = args
        # Number of objective function evaluations
        self.nfev = 0

    def fun(self, x):
        self.nfev += 1
        return self.func(x, *self.args)


class BSA:
    """Backtracking Search Optimization Algorithm.

    A population-based iterative EA designed to be a global minimizer.

    Parameters
    ----------
    generations : int, optional
        The maximum number of generations (iterations in BSA algorithm).
    pop_size : int, population size
        Number of individuals in a population based optimization algorithm.
    dim_rate : int, optional
        Controls the amplitude of the search-direction matrix. Defaults for 3.
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
        Specify `seed` for repeatable minimization. The random numbers
        generated with this seed only affect the visiting distribution function
        and new coordinates generation.

    References
    ----------
    .. [1] [1]  P. Civicioglu, "Backtracking Search Optimization Algorithm for
        numerical optimization problems", Applied Mathematics and Computation,
        219, 8121-8144, 2013.
    """

    def __init__(
        self,
        generations: int,
        pop_size: int = 30,
        dim_rate: int = 3,
        seed: Union[np.random.RandomState, int, None] = None,
    ) -> None:
        self.generations = generations
        self.pop_size = pop_size
        self.dim_rate = dim_rate
        self.rand_state = check_random_state(seed)

    def generate_population(self, bounds: Bounds) -> npt.NDArray[np.float64]:
        return (bounds.ub - bounds.lb) * self.rand_state.random_sample(  # type: ignore [no-any-return]
            (self.pop_size, bounds.lb.size)
        ) + bounds.lb

    def boundary_control(
        self, pop: npt.NDArray[np.float64], bounds: Bounds
    ) -> npt.NDArray[np.float64]:
        lb_residual, ub_residual = bounds.residual(pop)
        lb_negative_indexes = np.nonzero(lb_residual < 0)
        pop[lb_negative_indexes] = np.array(
            [
                bounds.lb[i]
                if self.get_rand_boolean()
                else (self.rand_state.rand() * (bounds.ub[i] - bounds.lb[i]) + bounds.lb[i])
                for i in lb_negative_indexes[1]
            ]
        )
        ub_negative_indexes = np.nonzero(ub_residual < 0)
        pop[ub_negative_indexes] = np.array(
            [
                bounds.ub[i]
                if self.get_rand_boolean()
                else (self.rand_state.rand() * (bounds.ub[i] - bounds.lb[i]) + bounds.lb[i])
                for i in ub_negative_indexes[1]
            ]
        )
        return pop

    def get_scale_factor(self, strategy: str = "standard brownian-walk") -> float:
        """Get walking random scale factor based on walk strategy."""
        return {  # type: ignore [no-any-return]
            "standard brownian-walk": self.rand_state.normal(scale=3),
            "brownian-walk": self.rand_state.standard_gamma(4),
        }[strategy]

    def get_rand_boolean(self) -> bool:
        return bool(self.rand_state.randint(2))

    def change_multiple(self, row: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Change multiple individual characteristics randomly selected."""
        magic_number = math.ceil(self.dim_rate * self.rand_state.rand() * (row.size))
        u = self.rand_state.permutation(row.size)[:magic_number]
        row[u] = 1
        return row

    def change_one(self, row: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Change only one individual characteristic."""
        row[self.rand_state.randint(row.size)] = 1
        return row

    def apply_selection_I(
        self, pop: npt.NDArray[np.float64], historical_pop: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Determine the historical population to be used for calculating the search direction."""
        if self.get_rand_boolean():
            historical_pop = pop
        return self.rand_state.permutation(historical_pop)  # type: ignore [no-any-return]

    def apply_recombination(
        self, pop: npt.NDArray[np.float64], historical_pop: npt.NDArray[np.float64], bounds: Bounds
    ) -> npt.NDArray[np.float64]:
        """Mutation and crossover processes."""
        F = self.get_scale_factor()
        changes_map = np.zeros((self.pop_size, bounds.lb.size))
        changes_map = np.apply_along_axis(
            self.change_one if self.get_rand_boolean() else self.change_multiple, 1, changes_map
        )
        offsprings = pop + (changes_map * F) * (historical_pop - pop)
        return self.boundary_control(offsprings, bounds)

    def apply_selection_II(
        self,
        pop: npt.NDArray[np.float64],
        fitness_pop: npt.NDArray[np.float64],
        offsprings: npt.NDArray[np.float64],
        fitness_offsprings: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Select the individuals with better fitness."""
        index = fitness_offsprings < fitness_pop
        fitness_pop[index] = fitness_offsprings[index]
        pop[index] = offsprings[index]
        return pop, fitness_pop

    def calculate_fitness(
        self, func_wrapped: ObjectiveFunWrapper, pop: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate individual fitness over a population."""
        return np.apply_along_axis(func_wrapped.fun, 1, pop)

    def run(
        self, func: Callable[[npt.NDArray[np.float64]], np.float64], bounds: Bounds
    ) -> OptimizeResult:
        """Run minimization.

        Parameters
        ----------
        func : callable
            The objective function to be minimized. Must be in the form
            ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
            and ``args`` is a  tuple of any additional fixed parameters needed to
            completely specify the function.
        bounds : sequence or `Bounds`
            Bounds for variables. Specify the bounds using `Bounds` class.

        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a `OptimizeResult` object.
            Important attributes are: ``x`` the solution array, ``fun`` the value
            of the function at the solution, and ``message`` which describes the
            cause of the termination.
            See `OptimizeResult` for a description of other attributes.
        """
        pop = self.generate_population(bounds)
        historical_pop = self.generate_population(bounds)  # swarm-memory of BSA
        func_wrapped = ObjectiveFunWrapper(func)
        fitness_pop = self.calculate_fitness(func_wrapped, pop)
        for _ in range(self.generations):
            historical_pop = self.apply_selection_I(pop, historical_pop)
            offsprings = self.apply_recombination(pop, historical_pop, bounds)
            fitness_offsprings = self.calculate_fitness(func_wrapped, offsprings)
            pop, fitness_pop = self.apply_selection_II(
                pop, fitness_pop, offsprings, fitness_offsprings
            )
            min_index = np.argmin(fitness_pop)

        return OptimizeResult(
            success=True,
            status=0,
            x=pop[min_index],
            fun=fitness_pop[min_index],
            nfev=func_wrapped.nfev,
            nit=self.generations,
            message=["Maximum number of iteration reached"],
        )
