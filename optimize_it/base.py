"""Base class for evolutionary optimization."""

from typing import Union

import numpy as np
import numpy.typing as npt
from scipy._lib._util import check_random_state
from scipy.optimize import OptimizeResult

from optimize_it.wrapper import ObjectiveFunWrapper


class OptimizationBase:
    """Base Evolutionary Optimization."""

    def __init__(
        self,
        generations: int,
        pop_size: int = 30,
        seed: Union[np.random.RandomState, int, None] = None,
    ) -> None:
        self.generations = generations
        self.pop_size = pop_size
        self.rand_state = check_random_state(seed)

    def get_optimize_result(
        self,
        final_pop: npt.NDArray[np.float64],
        final_evaluation: npt.NDArray[np.float64],
        func_wrapped: ObjectiveFunWrapper,
    ) -> OptimizeResult:
        min_index = np.argmin(final_evaluation)
        return OptimizeResult(
            success=True,
            status=0,
            x=final_pop[min_index],
            fun=final_evaluation[min_index],
            nfev=func_wrapped.number_func_evaluations,
            nit=self.generations,
            message=["Maximum number of iteration reached"],
        )
