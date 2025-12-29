# pity_curve.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import math

from config import PityCurveConfig

@dataclass(frozen=True)
class PityCurve:
    cfg: PityCurveConfig

    def p_hazard(self, t: int) -> float:
        """
        Conditional probability of getting a 5★ at pull count t since last 5★.
        t is 1-indexed. Enforces p(hard_pity)=1.
        """
        if t < 1:
            raise ValueError("t must be >= 1")
        if t >= self.cfg.hard_pity:
            return 1.0

        if t <= self.cfg.soft_start:
            return float(self.cfg.p0)

        # ramp from (soft_start+1 .. hard_pity) using power curve
        # normalized x in (0,1): x = (t-soft_start)/(hard_pity-soft_start)
        denom = (self.cfg.hard_pity - self.cfg.soft_start)
        x = (t - self.cfg.soft_start) / denom
        x = max(0.0, min(1.0, x))
        p = self.cfg.p0 + (1.0 - self.cfg.p0) * (x ** self.cfg.power_b)
        return float(max(0.0, min(1.0, p)))

    def hazard_table(self) -> List[float]:
        """Return p(t) for t=1..hard_pity."""
        return [self.p_hazard(t) for t in range(1, self.cfg.hard_pity + 1)]

    def implied_expected_T_by_dp(self) -> float:
        """
        Compute E[T] exactly from the hazard curve (no simulation) via survival products:
        P(T=t) = S(t-1) * p(t), where S(t) = Π_{i=1..t}(1-p(i)).
        """
        S = 1.0
        e = 0.0
        for t in range(1, self.cfg.hard_pity + 1):
            p = self.p_hazard(t)
            pmf = S * p
            e += t * pmf
            S *= (1.0 - p)
        # By construction p(hard_pity)=1, so total mass should be 1.
        return e
