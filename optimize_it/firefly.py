"""Firefly Algorithm (FA).

Cite this algorithm as;
[1]  Yang X. S., "Firefly Algorithms for Multimodal Optimization", Proceedings
of the 5th international conference on Stochastic algorithms: foundations
and applications, 2010.
"""

from typing import Callable, Union

import itertools
import math
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, OptimizeResult

from optimize_it.base import OptimizationBase
from optimize_it.functions import apply_simple_boundary_control, generate_random_population
from optimize_it.wrapper import ObjectiveFunWrapper


class Firefly(OptimizationBase):
    """Firefly Algorithm.

    A population-based iterative EA designed to be a global minimizer.

    Parameters
    ----------
    generations : int, optional
        The maximum number of generations (iterations in BSA algorithm).
    pop_size : int, population size
        Number of individuals in a population based optimization algorithm.
    alpha : float, optional
        Controls the algorithm randomness from 0 to 1 (highly random). Defaults for 0.5.
    beta_min : float, optional
        Controls the minimum value of beta. Defaults for 0.2.
    gamma : float, optional
        Absorption coefficient. Defaults for 1.
    alpha_decrease : boolean
        Implements a decrease function in alpha parameter over generations. It can be
        used to control randomness. Defaults for True.
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
    .. [1]  Yang X. S., "Firefly Algorithms for Multimodal Optimization", Proceedings
        of the 5th international conference on Stochastic algorithms: foundations
        and applications, 2010.
    """

    def __init__(
        self,
        generations: int,
        pop_size: int = 60,
        alpha: float = 0.5,
        beta_min: float = 0.20,
        gamma: float = 1.0,
        alpha_decrease: bool = True,
        seed: Union[np.random.RandomState, int, None] = None,
    ) -> None:
        super().__init__(generations, pop_size, seed)
        self.alpha = alpha
        self.beta_min = beta_min
        self.gamma = gamma
        self.alpha_decrease = alpha_decrease

    def get_fireflies_distance(
        self, first: npt.NDArray[np.float64], second: npt.NDArray[np.float64]
    ) -> float:
        return math.sqrt(sum((first - second) ** 2))

    def get_attractiveness_beta(self, distance: float) -> float:
        return (1 - self.beta_min) * math.exp(-self.gamma * distance**2) + self.beta_min

    def calculate_light_intensity(
        self, func_wrapped: ObjectiveFunWrapper, pop: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate individual light_intensity over a population."""
        return np.apply_along_axis(func_wrapped.fun, 1, pop)

    def get_alpha_generator(self, alpha: float) -> Iterator[float]:
        delta = 1 - (10 ** (-4) / 0.9) ** (1 / self.generations)
        for _ in range(self.generations):
            yield alpha
            if self.alpha_decrease:
                alpha = (1 - delta) * alpha

    def move_fireflies(
        self,
        pop: npt.NDArray[np.float64],
        light: npt.NDArray[np.float64],
        bounds: Bounds,
        alpha: float,
    ) -> npt.NDArray[np.float64]:
        """Move all fireflies toward brighter ones."""
        initial_pop = pop.copy()
        for _i, _j in itertools.combinations(range(self.pop_size), 2):
            if light[_i] != light[_j]:
                i, j = (
                    (_i, _j) if light[_i] > light[_j] else (_j, _i)
                )  # Brighter and more attractive
                scale = bounds.ub - bounds.lb
                distance = self.get_fireflies_distance(pop[i, :], pop[j, :])
                beta = self.get_attractiveness_beta(distance)
                rand_vector = self.rand_state.rand(1, bounds.lb.size)
                random_movement = alpha * (rand_vector - 0.5) * scale
                pop[i, :] = pop[i, :] * (1 - beta) + initial_pop[j, :] * beta + random_movement
        return apply_simple_boundary_control(pop, bounds)

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
        pop = generate_random_population(self.pop_size, bounds, self.rand_state)
        func_wrapped = ObjectiveFunWrapper(func)
        light_intensity = self.calculate_light_intensity(func_wrapped, pop)
        alpha_generator = self.get_alpha_generator(self.alpha)
        for _ in range(self.generations):
            alpha = next(alpha_generator)
            pop = self.move_fireflies(pop, light_intensity, bounds, alpha)
            light_intensity = self.calculate_light_intensity(func_wrapped, pop)
        return self.get_optimize_result(pop, light_intensity, func_wrapped)
