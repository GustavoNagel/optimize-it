import numpy as np
import numpy.typing as npt


def shekel(x: npt.NDArray[np.float64], m: int = 10) -> np.float64:
    """Shekel.

    The number of variables n = 4
    The parameter m should be adjusted m = 5,7,10.
    The global minima: x* =  (4, 4, 4, 4)
    f(x*) = -10.1532 for m = 5.
    f(x*) = -10.4029 for m = 7.
    f(x*) = -10.5364 for m = 10. (default)
    """
    a = np.array(
        [
            [4.0] * 4,
            [1.0] * 4,
            [8.0] * 4,
            [6.0] * 4,
            [3.0, 7.0] * 2,
            [2.0, 9.0] * 2,
            [5.0] * 2 + [3.0] * 2,
            [8.0, 1.0] * 2,
            [6.0, 2.0] * 2,
            [7.0, 3.6] * 2,
        ]
    )
    c = np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5]) * 0.1
    return -sum(1 / (((x - a[j]) ** 2).sum() + c[j]) for j in range(m))  # type: ignore [no-any-return]
