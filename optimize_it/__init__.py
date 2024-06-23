"""Awesome `optimize-it` is a Python cli/package with optimization tools"""

import sys
from importlib import metadata as importlib_metadata

from optimize_it.bsa import BSA
from optimize_it.firefly import Firefly


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
__all__ = ["version", "BSA", "Firefly"]
