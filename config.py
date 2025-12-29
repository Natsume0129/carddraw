# config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class PityCurveConfig:
    p0: float = 0.008          # base 5★ probability for t<=70
    soft_start: int = 70       # soft pity starts after this pull count
    hard_pity: int = 80        # hard pity (guarantee) at this pull count
    power_b: float = 1.0       # "power" exponent for the post-70 ramp

@dataclass(frozen=True)
class BannerConfig:
    up_prob_on_five_star: float = 0.5  # 50/50
    guarantee_up_after_loss: bool = True

@dataclass(frozen=True)
class SimConfig:
    n_trials: int = 200000
    seed: int = 42
