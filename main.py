"""gta6-trophy-rarity-forecast-01 entry: threshold-cohort holders/players model.

Forecasts each of GTA VI's 55 PS5 trophies' Exophase earn rate at the case
target, as 5 quantiles (0.05/0.25/0.5/0.75/0.95).

Model: the displayed page only changes at Exophase refreshes, so the history
is collapsed to distinct states (t_k, P_k, H_{i,k}).  The player population
is reconstructed as cohorts from the P(t) history; each player accumulates
(diurnal-weighted) play hours, and earns trophy i when those hours cross a
per-trophy threshold distribution (median hours m_i, ceiling c_i, tight for
story chapters, wide for grinds).  (m_i, c_i) are fit per trophy to the
distinct-state jumps with class priors from the trophy's
name/description/tier; the player base P(t) is projected by a power-law fit
(with a launch-phase prior when the display is still frozen).  Quantiles
come from a Monte Carlo over parameter uncertainty with a fixed seed.

Stdlib only, offline, deterministic (two runs byte-identical).
Run by the grader as: python3 -P -s /bot/main.py, reading $MAKEFUN_INPUT_DIR
(or /input) and writing $MAKEFUN_OUTPUT_DIR (or /output).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import datetime
import json
import math
import random

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
SEED = 20261119
N_MC = 200
NB_RHO = 10
N_HBINS = 48  # log-spaced banked-hours bins


# ------------------------------------------------------------------ priors
def classify(trophy):
    """(class, median hours, ceiling, threshold sigma, floor frac)."""
    d = (trophy["name"] + " " + trophy["description"]).lower()
    tier = trophy["tier"]
    if tier == "platinum":
        return ("platinum", 200.0, 0.02, 0.50, 0.50)
    if "100% completion" in d:
        return ("completion", 120.0, 0.08, 0.50, 0.50)
    if "complete chapter 1" in d:
        return ("story_ch1", 2.0, 0.92, 0.25, 0.70)
    if "complete chapter 2" in d:
        return ("story_ch2", 5.0, 0.88, 0.25, 0.70)
    if "complete chapter 3" in d:
        return ("story_ch3", 10.0, 0.84, 0.25, 0.70)
    if "complete chapter 4" in d:
        return ("story_ch4", 18.0, 0.80, 0.25, 0.70)
    if "complete the final mission" in d:
        return ("story_final", 35.0, 0.72, 0.25, 0.70)
    if "complete gone fishing" in d or "just like paradise" in d:
        return ("intro", 0.4, 0.99, 0.30, 0.60)
    if d.startswith("complete ") and "chapter" not in d:
        return ("story_mission", 12.0, 0.78, 0.35, 0.60)
    grind_keys = ("complete 250", "complete 50", "complete 8 ", "complete 3 ",
                  "win 20", "win a top fun", "earn a gold medal", "earn $",
                  "spend $", "purchase all", "register 10", "explore all",
                  "meditate at every", "waink it", "discover 100", "rob 5 ",
                  "rob an illegal")
    if any(k in d for k in grind_keys):
        return ("grind", 35.0, 0.18, 0.70, 0.30)
    return ("activity", 5.0, 0.55, 0.50, 0.40)


# ------------------------------------------------------------------ helpers
def _parse(s):
    # Real timestamps carry fractional seconds, which strptime can't take;
    # fromisoformat handles them, but it wants +00:00 rather than a bare Z.
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))


def _round4(x):
    # round-half-up on 4 decimals (results are compared as exact decimals)
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    v = whole / 10000
    return int(v) if float(v).is_integer() else v


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(q):
    # Acklam's inverse normal CDF, |rel err| < 1.2e-9 over the open interval
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if q < plow:
        x = math.sqrt(-2 * math.log(q))
        return (((((c[0] * x + c[1]) * x + c[2]) * x + c[3]) * x + c[4]) * x + c[5]) / \
               ((((d[0] * x + d[1]) * x + d[2]) * x + d[3]) * x + 1)
    if q > phigh:
        x = math.sqrt(-2 * math.log(1 - q))
        return -(((((c[0] * x + c[1]) * x + c[2]) * x + c[3]) * x + c[4]) * x + c[5]) / \
                ((((d[0] * x + d[1]) * x + d[2]) * x + d[3]) * x + 1)
    x = q - 0.5
    r = x * x
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * x / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _thresh_cdf(x, median, sigma, floor_frac):
    """Lognormal CDF over hours-above-floor with a hard floor at f*median."""
    floor = floor_frac * median
    if x <= floor:
        return 0.0
    return _phi((math.log(x - floor) - math.log(median - floor)) / sigma)


def _diurnal_factor(utc_h):
    hh = utc_h % 24.0
    if 1.0 <= hh < 7.0:
        return 0.08
    if 7.0 <= hh < 12.0:
        return 0.45
    if 12.0 <= hh < 18.0:
        return 0.70
    if 18.0 <= hh < 24.0:
        return 1.0
    return 0.35


def _build_diurnal_cum():
    # cumulative integral of the diurnal factor over one day, 15-min steps,
    # phase-aligned to the first read (23:50 UTC)
    step = 0.25
    n = int(24 / step)
    cum = [0.0]
    for k in range(n):
        t0 = k * step
        utc_h = (23.0 + 50.0 / 60.0 + t0) % 24.0
        cum.append(cum[-1] + _diurnal_factor(utc_h) * step)
    return cum, step


_DIURNAL_CUM, _DIURNAL_STEP = _build_diurnal_cum()
_DIURNAL_C24 = _DIURNAL_CUM[-1]


def _cum_play(t):
    """Integral of the diurnal factor from 0 to t (hours since first read)."""
    if t <= 0:
        return 0.0
    days = int(t // 24.0)
    rem = t - days * 24.0
    k = int(rem / _DIURNAL_STEP)
    frac = (rem - k * _DIURNAL_STEP) / _DIURNAL_STEP
    return days * _DIURNAL_C24 + _DIURNAL_CUM[k] + frac * (
        _DIURNAL_CUM[k + 1] - _DIURNAL_CUM[k])


def _play_hours(t0, t1, rho_day):
    """Diurnal-weighted hours played by one player between t0 and t1."""
    if t1 <= t0:
        return 0.0
    return (rho_day / 24.0) * (_cum_play(t1) - _cum_play(t0))


# ------------------------------------------------------------------ case prep
def _distinct_states(case):
    """Collapse the history to the states the display actually showed.

    Returns (states, t0) with states = [(hours_since_first_read, players,
    holders_per_trophy)] and t0 the first read's datetime.
    """
    hist = case["history"]
    t0 = _parse(hist[0]["at"])
    states = []
    prev_key = None
    for r in hist:
        rates = r["rates"]
        key = (r["players_tracked"], tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        t = (_parse(r["at"]) - t0).total_seconds() / 3600.0
        p = float(r["players_tracked"])
        holders = [0.0 if v is None else v * p / 100.0 for v in rates]
        states.append((t, p, holders))
    return states, t0


def _fit_players(states, horizon_h, t_target):
    """Power-law P(t); returns (p_last, p_target, log_sd).
    t_target is the target time in hours since the first read."""
    t_last, p_last, _ = states[-1]
    if len(states) < 2:
        # Display frozen (no refresh seen yet): the true player base has
        # still been growing since launch.  Launch-phase prior, anchored at
        # the first read: P(t) = P_first * (1 + t/1h)^1.0.  The house's own
        # rules record the first refresh taking tracked players 638 -> 3,868
        # in ~4.7h, i.e. (1+t)^1.04 -- this prior just states that shape.
        # Wide sd: the exact jump is not predictable from a frozen display.
        t_first, p_first, _ = states[0]
        p_target = p_first * (1.0 + t_target / 1.0) ** 1.0
        return p_last, max(p_target, p_last), 0.6
    # grid over the time constant of log(1 + t/T0); keep the least-squares
    # winner and read the slope off as the power-law exponent
    best = None
    for T0 in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        xs = [math.log(1.0 + t / T0) for (t, p, _) in states]
        ys = [math.log(p) for (t, p, _) in states]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        alpha = sxy / sxx if sxx > 1e-9 else 0.0
        inter = my - alpha * mx
        sse = sum((y - (inter + alpha * x)) ** 2 for x, y in zip(xs, ys))
        if best is None or sse < best[0]:
            best = (sse, T0, alpha, inter)
    _, T0, alpha, inter = best
    alpha = max(alpha, 0.0)
    p_target = math.exp(inter + alpha * math.log(1.0 + t_target / T0))
    resid = math.sqrt(best[0] / max(len(states) - 2, 1))
    log_sd = max(resid * 2.0, 0.08) + 0.10 * math.sqrt(horizon_h / 24.0)
    return p_last, max(p_target, p_last), min(log_sd, 0.6)


# ------------------------------------------------------------------ cohorts
# Banked-hours histogram: geometric edges so the low end (a few hours) and
# the veteran end (hundreds of hours) both keep resolution.
_HEDGES = [0.0]
_edge = 0.05
while _edge < 400.0:
    _HEDGES.append(_edge)
    _edge *= 1.25
_HEDGES.append(1e9)
_HCENTERS = [math.sqrt(_HEDGES[b] * _HEDGES[b + 1]) if _HEDGES[b] > 0 else 0.025
             for b in range(len(_HEDGES) - 1)]


def _rho_bins(rho_med, rho_sig=0.8):
    """Play-rate quadrature: NB_RHO lognormal quantile midpoints, weights."""
    out = []
    for b in range(NB_RHO):
        z = _norm_ppf((b + 0.5) / NB_RHO)
        out.append((min(rho_med * math.exp(rho_sig * z), 14.0), 1.0 / NB_RHO))
    return out


class Population:
    """Player population rebuilt as (count, join time, rate, pre-hours) cells.

    Three sources of players:
      * initial cohort at the first read -- a small veteran share already
        32h in, the rest fresh with 0.75h banked;
      * newcomers between consecutive distinct display states, sized by the
        tracked-player increase, joining at the midpoint of the two reads;
      * horizon newcomers, spreading the projected P growth from the last
        state to the target in hourly steps.
    Each cell's play rate is lognormal around rho0*(1+join/24)^-beta; a
    newcomer idles 3h after joining before playing (install + tutorial).
    """

    def __init__(self, states, p_target, t_target, rho0, beta,
                 f_vet=0.0125, fresh_hours=0.75, vet_hours=32.0):
        self.rho0 = rho0
        self.beta = beta
        self.t_target = t_target
        self.cells = []
        p0 = states[0][1]
        vet_n = p0 * f_vet
        fresh_n = p0 - vet_n
        for n, pre in ((vet_n, vet_hours), (fresh_n, fresh_hours)):
            for rho, wt in _rho_bins(rho0):
                self.cells.append((n * wt, 0.0, rho, pre, True))
        # newcomers between distinct states
        for k in range(len(states) - 1):
            t_a, p_a, _ = states[k]
            t_b, p_b, _ = states[k + 1]
            dn = max(p_b - p_a, 0.0)
            if dn <= 0:
                continue
            join = 0.5 * (t_a + t_b)
            med = rho0 * (1.0 + max(join, 0.0) / 24.0) ** (-beta)
            for rho, wt in _rho_bins(med):
                self.cells.append((dn * wt, join, rho, 0.0, False))
        # horizon newcomers
        t_last, p_last, _ = states[-1]
        if p_target > p_last and t_target > t_last:
            steps = max(1, int(math.ceil(t_target - t_last)))
            log_step = (math.log(p_target) - math.log(p_last)) / steps
            p = p_last
            for s in range(steps):
                p_next = p * math.exp(log_step)
                dn = max(p_next - p, 0.0)
                join = t_last + (s + 0.5) * (t_target - t_last) / steps
                med = rho0 * (1.0 + max(join, 0.0) / 24.0) ** (-beta)
                for rho, wt in _rho_bins(med):
                    self.cells.append((dn * wt, join, rho, 0.0, False))
                p = p_next
        self.p_last = p_last

    def hours_hist(self, t):
        """Banked-hours histograms at time t, as (initial, newcomer) counts.

        Both lists are indexed like _HCENTERS; the caller folds in the
        per-trophy threshold curve and ceiling.
        """
        nb = len(_HEDGES) - 1
        init = [0.0] * nb
        new = [0.0] * nb
        for n, join, rho, pre, is_init in self.cells:
            if join > t:
                continue
            if is_init:
                h = pre + _play_hours(join, t, rho)
            else:
                h = _play_hours(join + 3.0, t, rho)
            # locate the bin: first edge strictly above h
            lo, hi = 0, nb
            while lo < hi:
                mid = (lo + hi) // 2
                if _HEDGES[mid + 1] <= h:
                    lo = mid + 1
                else:
                    hi = mid
            if is_init:
                init[lo] += n
            else:
                new[lo] += n
        return init, new


def _holders_from_hist(init_hist, new_hist, m, c, tsig, tfloor,
                       anchor_init0, anchor_obs0):
    """Holders implied by the histograms at one time.

    The initial cohort is anchored to the observed holder count at the first
    distinct state (only its modelled *delta* is trusted); newcomers are
    counted outright, since nothing was observed for them yet.
    """
    model_init = 0.0
    for cnt, hc in zip(init_hist, _HCENTERS):
        if cnt > 0:
            model_init += cnt * c * _thresh_cdf(hc, m, tsig, tfloor)
    h_new = 0.0
    for cnt, hc in zip(new_hist, _HCENTERS):
        if cnt > 0:
            h_new += cnt * c * _thresh_cdf(hc, m, tsig, tfloor)
    return anchor_obs0 + max(model_init - anchor_init0, 0.0) + h_new


# ------------------------------------------------------------------ forecast
def forecast(case, n_mc=N_MC, seed=SEED, verbose=False):
    horizon_h = case["horizon_minutes"] / 60.0
    states, t0 = _distinct_states(case)
    t_last, p_last, h_last = states[-1]
    # Target clock: checkpoint time + horizon.  Deliberately NOT last-read
    # time + horizon -- the display may not have refreshed since before the
    # checkpoint, which would shorten the projection.
    t_target = (_parse(case["at"]) - t0).total_seconds() / 3600.0 + horizon_h
    _, p_target, p_log_sd = _fit_players(states, horizon_h, t_target)

    # veteran share: the final-mission trophy's rate at the first read
    trophies = case["trophies"]
    idx_final = next((i for i, t in enumerate(trophies)
                      if "final mission" in t["description"].lower()), None)
    f_vet = 0.0125
    if idx_final is not None:
        r0 = states[0][2][idx_final] / states[0][1]
        f_vet = min(0.05, max(0.002, r0))

    # ---- (rho0, beta) fit on informative (early, high-rate) trophies
    info_idx = [i for i, t in enumerate(trophies)
                if classify(t)[0] in ("intro", "story_ch1", "story_ch2")]
    best_rho = None
    for rho0 in (2.5, 4.0, 6.0):
        for beta in (0.4, 0.8, 1.2):
            pop = Population(states, p_target, t_target, rho0, beta, f_vet)
            err = 0.0
            for i in info_idx:
                _, m0, c0, tsig, tfloor = classify(trophies[i])
                hists = [pop.hours_hist(t) for (t, _, _) in states]
                a_init0 = sum(cnt * c0 * _thresh_cdf(hc, m0, tsig, tfloor)
                              for cnt, hc in zip(hists[0][0], _HCENTERS))
                for k, (t, p, holders) in enumerate(states):
                    h = _holders_from_hist(
                        hists[k][0], hists[k][1], m0, c0, tsig, tfloor,
                        a_init0, states[0][2][i])
                    obs = holders[i]
                    err += (h - obs) ** 2 / (obs + 10.0)
            if best_rho is None or err < best_rho[0]:
                best_rho = (err, rho0, beta)
    _, rho0, beta = best_rho
    if verbose:
        print("fitted rho0=%.1f beta=%.1f (err %.1f)" % (rho0, beta, best_rho[0]))

    pop = Population(states, p_target, t_target, rho0, beta, f_vet)
    state_hists = [pop.hours_hist(t) for (t, _, _) in states]
    target_hist = pop.hours_hist(t_target)

    # ---- per-trophy (m, c) grid fit, then Monte Carlo quantiles
    rows = []
    rng = random.Random(seed)
    for i, trophy in enumerate(trophies):
        cls, m0, c0, tsig, tfloor = classify(trophy)
        obs = [s[2][i] for s in states]
        best = None
        m_grid = [m0 * f for f in
                  (0.25, 0.4, 0.6, 0.85, 1.0, 1.2, 1.5, 2.0, 2.8, 4.0)]
        c_grid = sorted(set(min(max(c0 * f, 0.005), 0.99) for f in
                            (0.25, 0.5, 1.0, 2.0, 4.0)))
        for m in m_grid:
            for c in c_grid:
                a_init0 = sum(cnt * c * _thresh_cdf(hc, m, tsig, tfloor)
                              for cnt, hc in zip(state_hists[0][0], _HCENTERS))
                err = 0.0
                for k in range(len(states)):
                    h = _holders_from_hist(
                        state_hists[k][0], state_hists[k][1], m, c,
                        tsig, tfloor, a_init0, obs[0])
                    err += (h - obs[k]) ** 2 / (obs[k] + 10.0)
                # class prior pulls (m, c) back toward the trophy's class
                pen = ((math.log(m / m0)) / 0.5) ** 2 + \
                      ((math.log(c / (1 - c)) - math.log(c0 / (1 - c0))) / 0.7) ** 2
                tot = err + 5.0 * pen
                if best is None or tot < best[0]:
                    best = (tot, m, c)
        _, m_fit, c_fit = best

        samples = []
        for _ in range(n_mc):
            m_s = m_fit * math.exp(rng.gauss(0, 0.35))
            lc = math.log(c_fit / (1 - c_fit))
            c_s = 1.0 / (1.0 + math.exp(-(lc + rng.gauss(0, 0.4))))
            rho_s = rho0 * math.exp(rng.gauss(0, 0.25))
            beta_s = max(0.1, beta + rng.gauss(0, 0.2))
            p_t = p_target * math.exp(rng.gauss(0, p_log_sd))
            a_init0 = sum(cnt * c_s * _thresh_cdf(hc, m_s, tsig, tfloor)
                          for cnt, hc in zip(state_hists[0][0], _HCENTERS))
            h_t = _holders_from_hist(
                target_hist[0], target_hist[1], m_s, c_s, tsig, tfloor,
                a_init0, obs[0])
            # rate jitter: a faster-playing base earns proportionally more
            h_t *= math.exp(0.3 * math.log(rho_s / rho0)) if rho_s > 0 else 1.0
            r = 100.0 * h_t / max(p_t, 1.0)
            samples.append(min(max(r, 0.01), 99.99))
        samples.sort()

        qs = []
        for tau in LEVELS:
            pos = tau * (len(samples) - 1)
            lo = int(math.floor(pos))
            hi = min(lo + 1, len(samples) - 1)
            v = samples[lo] + (samples[hi] - samples[lo]) * (pos - lo)
            qs.append(_round4(min(max(v, 0.01), 99.99)))
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
        if verbose:
            print("id %2d %-12s m=%.1f (prior %.1f) c=%.2f (prior %.2f) med=%.2f" %
                  (trophy["id"], cls, m_fit, m0, c_fit, c0, qs[2]))
    return {"trophies": rows}


INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        out = forecast(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
