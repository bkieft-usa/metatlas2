from __future__ import annotations

import os
import sys
import getpass
import numpy as np
from datetime import datetime

def _is_jupyter() -> bool:
    """Return True when running inside a Jupyter kernel (notebook or lab)."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ in ("ZMQInteractiveShell", "TerminalInteractiveShell")
    except ImportError:
        return False


def should_disable_tqdm() -> bool:
    """Return True only when running in a non-interactive batch context.

    tqdm bars are kept enabled when:
      - Running inside a Jupyter notebook/lab (ZMQInteractiveShell), OR
      - stdout is a real TTY (interactive terminal).

    tqdm bars are disabled when:
      - Running as a SLURM batch job AND not in a Jupyter kernel.
    """
    in_jupyter = _is_jupyter()
    if in_jupyter:
        return False
    if "SLURM_JOB_ID" in os.environ:
        return True
    return not sys.stdout.isatty()

def safe_float(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)

def safe_isnan(value) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return True

def as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except TypeError:
        return []

def jsonable_list(value) -> list:
    out = []
    for elem in as_list(value):
        if isinstance(elem, np.generic):
            elem = elem.item()
        out.append(elem)
    return out

def get_provenance() -> dict[str, str]:
    return {
        "analyst": getpass.getuser(),
        "timestamp": datetime.now().isoformat(),
    }