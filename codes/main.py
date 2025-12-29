# main.py
import numpy as np

from config import PityCurveConfig, BannerConfig, SimConfig
from pity_curve import PityCurve
from simulator import BannerSimulator
from stats_utils import summarize, two_peak_diagnostic

def print_curve_preview(curve: PityCurve) -> None:
    hard = curve.cfg.hard_pity
    soft = curve.cfg.soft_start
    print("=== Hazard p(t) preview ===")
    for t in list(range(1, 6)) + list(range(soft - 2, soft + 1)) + list(range(soft + 1, soft + 6)) + [hard]:
        if 1 <= t <= hard:
            print(f"t={t:2d}  p(t)={curve.p_hazard(t):.4f}")
    print()

def main():
    pity_cfg = PityCurveConfig(p0=0.008, soft_start=70, hard_pity=80, power_b=1.0)
    banner_cfg = BannerConfig(up_prob_on_five_star=0.5, guarantee_up_after_loss=True)
    sim_cfg = SimConfig(n_trials=200000, seed=42)

    curve = PityCurve(pity_cfg)
    print_curve_preview(curve)

    eT = curve.implied_expected_T_by_dp()
    print(f"E[T] (exact from curve) = {eT:.4f}  -> implied long-run 5★ rate ≈ {1.0/eT:.4%}")
    print()

    rng = np.random.default_rng(sim_cfg.seed)
    sim = BannerSimulator(curve=curve, banner=banner_cfg, rng=rng)

    N = sim.simulate_many_one_up(sim_cfg.n_trials, start_guaranteed_up=False)
    s = summarize(N)
    diag = two_peak_diagnostic(N, split=80)

    print("=== One UP cost N ===")
    print(f"trials = {sim_cfg.n_trials}")
    print(f"mean   = {s.mean:.3f}")
    print(f"std    = {s.std:.3f}")
    print(f"q10    = {s.q10:.1f}")
    print(f"median = {s.q50:.1f}")
    print(f"q90    = {s.q90:.1f}")
    print()
    print("=== Two-peak-ish diagnostic ===")
    for k, v in diag.items():
        print(f"{k:8s} = {v:.4f}")

if __name__ == "__main__":
    main()
