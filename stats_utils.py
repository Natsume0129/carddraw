# stats_utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Sequence
import numpy as np

@dataclass(frozen=True)
class Summary:
    mean: float
    std: float
    q10: float
    q50: float
    q90: float

def summarize(x: Sequence[float]) -> Summary:
    arr = np.asarray(x, dtype=float)
    return Summary(
        mean=float(arr.mean()),
        std=float(arr.std(ddof=0)),
        q10=float(np.quantile(arr, 0.10)),
        q50=float(np.quantile(arr, 0.50)),
        q90=float(np.quantile(arr, 0.90)),
    )

def two_peak_diagnostic(x: Sequence[int], split: int = 80) -> Dict[str, float]:
    """
    Simple diagnostic: probability mass on <=80 pulls vs >80 pulls
    (often correlates with 1-gold UP vs 2-gold UP, not exact but indicative).
    """
    arr = np.asarray(x, dtype=int)
    return {
        "P(N<=80)": float(np.mean(arr <= split)),
        "P(N>80)": float(np.mean(arr > split)),
    }
