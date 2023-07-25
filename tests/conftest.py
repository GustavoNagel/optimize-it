from typing import Any

from collections.abc import Iterator
from functools import partial

import pytest

from tests.benchmark.shekel import shekel as shekel_func


@pytest.fixture(
    params=[
        {"m": 5, "x": (4, 4, 4, 4), "fun": -10.1532},
        {"m": 7, "x": (4, 4, 4, 4), "fun": -10.4029},
        {"m": 10, "x": (4, 4, 4, 4), "fun": -10.5364},
    ]
)
def shekel(request: pytest.FixtureRequest) -> Iterator[dict[str, Any]]:
    request_param_data = request.param.copy()
    shekel_data = {"func": partial(shekel_func, m=request_param_data.pop("m"))}
    shekel_data.update(request_param_data)
    yield shekel_data
