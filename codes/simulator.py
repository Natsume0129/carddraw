# simulator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from config import BannerConfig
from pity_curve import PityCurve

@dataclass
class BannerSimulator:
    curve: PityCurve
    banner: BannerConfig
    rng: np.random.Generator

    def sample_T(self) -> int:
        """
        Sample the number of pulls until next 5★ (inclusive), using hazard p(t).
        """
        hard = self.curve.cfg.hard_pity
        for t in range(1, hard + 1):
            p = self.curve.p_hazard(t)
            if self.rng.random() < p:
                return t
        # unreachable because p(hard)=1
        return hard

    def sample_one_up_cost(self, start_guaranteed_up: bool = False) -> Tuple[int, bool]:
        """
        Sample pulls N to obtain one UP 5★.
        Returns (N, end_guaranteed_up_state) where end state indicates whether the next 5★ is guaranteed UP.
        """
        guaranteed = start_guaranteed_up
        pulls = 0

        # First 5★
        pulls += self.sample_T()
        if guaranteed:
            # UP obtained, guarantee consumed
            return pulls, False

        # 50/50
        if self.rng.random() < self.banner.up_prob_on_five_star:
            # won 50/50 -> UP obtained
            return pulls, False
        else:
            # lost 50/50 -> next 5★ guaranteed UP (if rule enabled)
            if self.banner.guarantee_up_after_loss:
                pulls += self.sample_T()
                return pulls, False
            else:
                # If no guarantee rule, would loop, but not used here.
                return pulls, False

    def simulate_many_one_up(self, n_trials: int, start_guaranteed_up: bool = False) -> np.ndarray:
        out = np.empty(n_trials, dtype=np.int32)
        guaranteed = start_guaranteed_up
        # For independent trials, reset state each trial; guarantee is start state only.
        for i in range(n_trials):
            out[i], _ = self.sample_one_up_cost(start_guaranteed_up=guaranteed)
        return out
