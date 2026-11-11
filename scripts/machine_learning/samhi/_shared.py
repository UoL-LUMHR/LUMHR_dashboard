"""Small runtime helpers shared by the standalone SAMHI scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def project_root() -> Path:
    """Return the repository root for a script in this directory."""
    return Path(__file__).resolve().parents[3]


def import_module(path: Path, name: str) -> ModuleType:
    """Load a Python module from an explicit file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalise_lsoa_code(value: object) -> str:
    """Remove an optional label suffix from an LSOA code."""
    return str(value).split(":", 1)[0].strip()


def normal_crps(actual: object, mean: object, standard_deviation: object) -> float:
    """Return the mean CRPS for normally distributed predictions."""
    import numpy as np
    from scipy.stats import norm

    standard_deviation = np.maximum(
        np.asarray(standard_deviation, dtype=float),
        1e-6,
    )
    z_score = (
        np.asarray(actual, dtype=float) - np.asarray(mean, dtype=float)
    ) / standard_deviation
    score = standard_deviation * (
        z_score * (2 * norm.cdf(z_score) - 1)
        + 2 * norm.pdf(z_score)
        - 1 / np.sqrt(np.pi)
    )
    return float(np.mean(score))
