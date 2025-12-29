# simulate_losing_rate.py
# Simulate a gacha banner with:
#   - 5★ pity hazard curve (soft pity after soft_start, hard pity at hard_pity)
#   - 50/50 on "small pity" (not guaranteed)
#   - if lose 50/50, next 5★ is guaranteed UP
#
# This script includes:
# 1) Calibrate power_b so implied 5★ rate is in [1.799%, 1.800%]
# 2) Simulate n_people people, each with n_pulls pulls
# 3) Compute:
#    - % people with lose_rate_small > 0.55
#    - % people with lose_rate_small < 0.45
#      where lose_rate_small = (# lose 50/50 when NOT guaranteed) / (# small-pity trials)
# 4) Pattern events (all are "greater than", and must be CONSECUTIVE small-pity losses):
#    - 2 consecutive losses: sum of the 2 (loss+guarantee) cycle costs  > 240
#    - 3 consecutive losses: sum of the 3 (loss+guarantee) cycle costs  > 360
#    - 4 consecutive losses: sum of the 4 (loss+guarantee) cycle costs  > 480
#
# Cycle cost definition (for a small-pity loss):
#   cycle_cost = T1 (to losing 5★) + T2 (to next guaranteed 5★)
#
# Run:
#   python .\simulate_losing_rate.py

import numpy as np


# ----------------------------
# Module 1: pity curve + T distribution
# ----------------------------
def make_hazard_table(p0=0.008, soft_start=70, hard_pity=80, power_b=1.0) -> np.ndarray:
    if not (0.0 < p0 < 1.0):
        raise ValueError("p0 must be in (0,1)")
    if not (1 <= soft_start < hard_pity):
        raise ValueError("need 1 <= soft_start < hard_pity")
    if power_b <= 0:
        raise ValueError("power_b must be > 0")

    p = np.empty(hard_pity, dtype=np.float64)
    denom = (hard_pity - soft_start)

    for t in range(1, hard_pity + 1):
        if t >= hard_pity:
            p[t - 1] = 1.0
        elif t <= soft_start:
            p[t - 1] = p0
        else:
            x = (t - soft_start) / denom  # (0,1)
            val = p0 + (1.0 - p0) * (x ** power_b)
            p[t - 1] = min(1.0, max(0.0, val))

    return p


def hazard_to_cdf(hazard: np.ndarray) -> np.ndarray:
    """
    P(T=t) = S(t-1)*p(t) where S(t) = Π_{i<=t}(1-p(i))
    """
    S = 1.0
    pmf = np.empty_like(hazard, dtype=np.float64)
    for i, pt in enumerate(hazard):
        pmf[i] = S * pt
        S *= (1.0 - pt)
    cdf = np.cumsum(pmf)
    cdf[-1] = 1.0
    return cdf


def sample_T_from_cdf(cdf: np.ndarray, rng: np.random.Generator) -> int:
    u = rng.random()
    return int(np.searchsorted(cdf, u, side="left") + 1)


def expected_T_from_hazard(hazard: np.ndarray) -> float:
    S = 1.0
    eT = 0.0
    for t, pt in enumerate(hazard, start=1):
        pmf = S * pt
        eT += t * pmf
        S *= (1.0 - pt)
    return float(eT)


# ----------------------------
# Module 1.5: calibration (power_b -> target implied rate)
# ----------------------------
def implied_rate_from_params(p0: float, soft_start: int, hard_pity: int, power_b: float) -> float:
    hazard = make_hazard_table(p0=p0, soft_start=soft_start, hard_pity=hard_pity, power_b=power_b)
    eT = expected_T_from_hazard(hazard)
    return 1.0 / eT


def calibrate_power_b_for_rate(
    target_rate_low: float,
    target_rate_high: float,
    p0: float,
    soft_start: int,
    hard_pity: int,
    b_lo: float = 0.3,
    b_hi: float = 3.0,
    max_iter: int = 80,
) -> float:
    """
    Binary search power_b so implied 5★ rate in [target_rate_low, target_rate_high].
    For this curve family: smaller b => higher early post-soft probability => higher rate.
    """
    r_lo = implied_rate_from_params(p0, soft_start, hard_pity, b_lo)
    r_hi = implied_rate_from_params(p0, soft_start, hard_pity, b_hi)

    if not (r_lo >= r_hi):
        raise RuntimeError(
            f"Unexpected monotonicity: rate(b_lo)={r_lo:.6%}, rate(b_hi)={r_hi:.6%}. "
            "Try different b_lo/b_hi."
        )

    if target_rate_high > r_lo or target_rate_low < r_hi:
        raise RuntimeError(
            "Target rate not bracketed.\n"
            f"  target: [{target_rate_low:.6%}, {target_rate_high:.6%}]\n"
            f"  b_lo={b_lo} -> {r_lo:.6%}\n"
            f"  b_hi={b_hi} -> {r_hi:.6%}\n"
            "Expand b_lo/b_hi."
        )

    lo, hi = b_lo, b_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        r_mid = implied_rate_from_params(p0, soft_start, hard_pity, mid)

        if target_rate_low <= r_mid <= target_rate_high:
            return mid

        # r_mid too high -> increase b (reduce early probability) -> move lo up
        if r_mid > target_rate_high:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


# ----------------------------
# Module 2: simulation per person (with pattern detection)
# ----------------------------
def simulate_one_person(
    cdf_T: np.ndarray,
    rng: np.random.Generator,
    n_pulls: int,
    up_prob_on_five_star: float,
    thr2: int = 240,
    thr3: int = 360,
    thr4: int = 480,
) -> tuple[int, int, int, int, float, bool, bool, bool]:
    """
    Returns:
      pulls_used, five_star_count, lose_count, small_trials, lose_rate_small,
      hit_2_gt240, hit_3_gt360, hit_4_gt480

    lose_rate_small = lose_count / small_trials

    For each small-pity loss we complete a "loss + guaranteed" cycle:
      cycle_cost = T1 (to losing 5★) + T2 (to next guaranteed 5★)

    Pattern events (must be consecutive small-pity losses):
      - hit_2_gt240: exists 2 consecutive losses where sum(last 2 cycle_costs) > thr2
      - hit_3_gt360: exists 3 consecutive losses where sum(last 3 cycle_costs) > thr3
      - hit_4_gt480: exists 4 consecutive losses where sum(last 4 cycle_costs) > thr4
    """
    pulls_used = 0
    guaranteed_up = False

    five_star = 0
    lose = 0
    small_trials = 0

    lose_streak = 0
    last_cycle_costs: list[int] = []

    hit_2_gt240 = False
    hit_3_gt360 = False
    hit_4_gt480 = False

    def advance_one_five_star():
        nonlocal pulls_used, five_star
        t = sample_T_from_cdf(cdf_T, rng)
        if pulls_used + t > n_pulls:
            return None
        pulls_used += t
        five_star += 1
        return t

    while True:
        # step to next 5★
        t1 = advance_one_five_star()
        if t1 is None:
            break

        if guaranteed_up:
            # forced UP, does not create a small-pity roll
            guaranteed_up = False
            continue

        # small pity: roll 50/50
        small_trials += 1
        if rng.random() < up_prob_on_five_star:
            # win -> reset streak
            lose_streak = 0
            last_cycle_costs.clear()
        else:
            # lose -> next 5★ guaranteed UP
            lose += 1
            guaranteed_up = True

            # complete the cycle by drawing the guaranteed 5★
            t2 = advance_one_five_star()
            if t2 is None:
                break
            guaranteed_up = False

            cycle_cost = t1 + t2

            lose_streak += 1
            last_cycle_costs.append(cycle_cost)
            if len(last_cycle_costs) > 4:
                last_cycle_costs.pop(0)

            # 2 consecutive losses
            if lose_streak >= 2 and len(last_cycle_costs) >= 2:
                if (last_cycle_costs[-1] + last_cycle_costs[-2]) > thr2:
                    hit_2_gt240 = True

            # 3 consecutive losses
            if lose_streak >= 3 and len(last_cycle_costs) >= 3:
                if (last_cycle_costs[-1] + last_cycle_costs[-2] + last_cycle_costs[-3]) > thr3:
                    hit_3_gt360 = True

            # 4 consecutive losses
            if lose_streak >= 4 and len(last_cycle_costs) >= 4:
                if (last_cycle_costs[-1] + last_cycle_costs[-2] + last_cycle_costs[-3] + last_cycle_costs[-4]) > thr4:
                    hit_4_gt480 = True

    lose_rate_small = (lose / small_trials) if small_trials > 0 else 0.0
    return (
        pulls_used,
        five_star,
        lose,
        small_trials,
        lose_rate_small,
        hit_2_gt240,
        hit_3_gt360,
        hit_4_gt480,
    )


def simulate_population(
    n_people: int,
    n_pulls: int,
    seed: int,
    hazard: np.ndarray,
    up_prob_on_five_star: float = 0.5,
    thr2: int = 240,
    thr3: int = 360,
    thr4: int = 480,
) -> dict:
    rng = np.random.default_rng(seed)

    cdf_T = hazard_to_cdf(hazard)
    eT = expected_T_from_hazard(hazard)

    lose_rates = np.empty(n_people, dtype=np.float64)
    five_stars = np.empty(n_people, dtype=np.int32)
    loses = np.empty(n_people, dtype=np.int32)
    small_trials_arr = np.empty(n_people, dtype=np.int32)
    pulls_used_arr = np.empty(n_people, dtype=np.int32)

    hit_flags_2 = np.zeros(n_people, dtype=bool)
    hit_flags_3 = np.zeros(n_people, dtype=bool)
    hit_flags_4 = np.zeros(n_people, dtype=bool)

    total_pulls_used = 0
    total_five_stars = 0

    for i in range(n_people):
        pulls_used, fs, lo, st, lr, h2, h3, h4 = simulate_one_person(
            cdf_T=cdf_T,
            rng=rng,
            n_pulls=n_pulls,
            up_prob_on_five_star=up_prob_on_five_star,
            thr2=thr2,
            thr3=thr3,
            thr4=thr4,
        )
        pulls_used_arr[i] = pulls_used
        five_stars[i] = fs
        loses[i] = lo
        small_trials_arr[i] = st
        lose_rates[i] = lr

        hit_flags_2[i] = h2
        hit_flags_3[i] = h3
        hit_flags_4[i] = h4

        total_pulls_used += pulls_used
        total_five_stars += fs

    return {
        "E_T": eT,
        "lose_rates": lose_rates,
        "five_stars": five_stars,
        "loses": loses,
        "small_trials": small_trials_arr,
        "pulls_used": pulls_used_arr,
        "hit_flags_2": hit_flags_2,
        "hit_flags_3": hit_flags_3,
        "hit_flags_4": hit_flags_4,
        "total_pulls_used": total_pulls_used,
        "total_five_stars": total_five_stars,
    }


# ----------------------------
# Module 3: stats
# ----------------------------
def summarize_rates(x: np.ndarray) -> dict:
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "var": float(np.var(x, ddof=0)),
        "std": float(np.std(x, ddof=0)),
    }


if __name__ == "__main__":
    # ---- user controls ----
    n_people = 100000
    n_pulls = 2000
    seed = 42

    p0 = 0.008
    soft_start = 70
    hard_pity = 80

    up_prob_on_five_star = 0.5

    # Calibrate to 1.799% ~ 1.800%
    target_rate_low = 0.01799
    target_rate_high = 0.01800

    # Pattern thresholds (all are "greater than")
    thr2 = 240
    thr3 = 360
    thr4 = 480

    # ---- calibrate power_b ----
    power_b = calibrate_power_b_for_rate(
        target_rate_low=target_rate_low,
        target_rate_high=target_rate_high,
        p0=p0,
        soft_start=soft_start,
        hard_pity=hard_pity,
        b_lo=0.3,
        b_hi=3.0,
        max_iter=80,
    )

    hazard = make_hazard_table(p0=p0, soft_start=soft_start, hard_pity=hard_pity, power_b=power_b)
    eT = expected_T_from_hazard(hazard)
    implied_rate = 1.0 / eT

    print("=== Calibrated pity curve ===")
    print(f"power_b      = {power_b:.6f}")
    print(f"E[T]         = {eT:.6f}")
    print(f"implied rate = {implied_rate:.6%}")
    print()

    print("=== Hazard p(t) quick peek ===")
    for t in [1, 2, 3, 70, 71, 72, 73, 76, 78, 80]:
        print(f"t={t:2d}  p(t)={hazard[t-1]:.4f}")
    print()

    # ---- simulate ----
    result = simulate_population(
        n_people=n_people,
        n_pulls=n_pulls,
        seed=seed,
        hazard=hazard,
        up_prob_on_five_star=up_prob_on_five_star,
        thr2=thr2,
        thr3=thr3,
        thr4=thr4,
    )

    # Lose-rate stats
    lr = result["lose_rates"]
    stats = summarize_rates(lr)

    print("=== Lose-rate (small-pity) stats across people ===")
    print(f"people = {n_people}, pulls/person = {n_pulls}, seed = {seed}")
    print(f"min  = {stats['min']:.6f}")
    print(f"max  = {stats['max']:.6f}")
    print(f"mean = {stats['mean']:.6f}")
    print(f"var  = {stats['var']:.6f}")
    print(f"std  = {stats['std']:.6f}")

    # Threshold queries
    pct_gt_55 = float(np.mean(lr > 0.55)) * 100.0
    pct_lt_45 = float(np.mean(lr < 0.45)) * 100.0

    print("\n=== Lose-rate threshold stats ===")
    print(f"P(lose_rate_small > 0.55) = {pct_gt_55:.4f}%")
    print(f"P(lose_rate_small < 0.45) = {pct_lt_45:.4f}%")

    # Five-star sanity
    fs = result["five_stars"]
    print("\n=== Five-star count sanity ===")
    print(f"min_5★ = {int(fs.min())}, max_5★ = {int(fs.max())}, mean_5★ = {fs.mean():.2f}")

    # Average pulls per 5★ (global)
    total_pulls_used = result["total_pulls_used"]
    total_five_stars = result["total_five_stars"]
    avg_pulls_per_five_star = total_pulls_used / total_five_stars if total_five_stars > 0 else float("nan")

    print("\n=== Pull efficiency ===")
    print(f"total pulls used   = {total_pulls_used}")
    print(f"total 5★           = {total_five_stars}")
    print(f"avg pulls per 5★   = {avg_pulls_per_five_star:.6f}")
    print(f"target (1/0.018)   = {1/0.018:.6f}")

    # Pattern event stats (all "greater than")
    hits2 = result["hit_flags_2"]
    hits3 = result["hit_flags_3"]
    hits4 = result["hit_flags_4"]

    c2 = int(np.sum(hits2))
    c3 = int(np.sum(hits3))
    c4 = int(np.sum(hits4))

    p2 = float(np.mean(hits2)) * 100.0
    p3 = float(np.mean(hits3)) * 100.0
    p4 = float(np.mean(hits4)) * 100.0

    print("\n=== Consecutive-loss cycle-sum events ===")
    print(f"Event (2 losses): sum of 2 cycle_costs > {thr2}  -> {c2} ({p2:.6f}%)")
    print(f"Event (3 losses): sum of 3 cycle_costs > {thr3}  -> {c3} ({p3:.6f}%)")
    print(f"Event (4 losses): sum of 4 cycle_costs > {thr4}  -> {c4} ({p4:.6f}%)")
