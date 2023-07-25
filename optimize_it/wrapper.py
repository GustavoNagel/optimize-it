"""Basic Wrapper to objective function."""


class ObjectiveFunWrapper:
    """Basic Wrapper to objective function."""

    def __init__(self, func, *args):
        self.func = func
        self.args = args
        self.number_func_evaluations = 0

    def fun(self, x):
        self.number_func_evaluations += 1
        return self.func(x, *self.args)
