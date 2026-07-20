from __future__ import annotations

import os
import sys
import getpass
import numpy as np
from datetime import datetime

def should_disable_tqdm() -> bool:
    return "SLURM_JOB_ID" in os.environ or not sys.stdout.isatty()

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
