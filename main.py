"""gta6-twitch-ccu-forecast-01: launch-day Twitch CCU forecaster.

For each hour h = 2..24 of a game's release day (UTC), forecast five
quantiles of the Twitch category's concurrent viewers, from the one number
the case gives us at the checkpoint -- hour 1's print -- plus the release-day
curves of past big launches shipped in the case as `comparables`.

What the comparables teach (leave-one-out over the 38 launches in the kit,
scored with the contest's own log-scale pinball loss):

  * Hour 1's print is a poor anchor on its own: launches whose category was
    still dark at 00:00 UTC (hour-1 print in the hundreds) grow 100-1000x by
    the evening, while launches already live at hour 1 (print in the
    hundreds of thousands) stay near flat, peaking around 1-2.6x. The
    log of hour 1's print predicts the log of every later hour with a
    slope that falls from ~1 at hour 2 toward ~0.1 at the evening peak:
    the peak audience is nearly independent of the hour-1 print.
  * So, per hour, fit log(viewers_h) = a_h + b_h * log(viewers_1) by least
    squares over the comparables whose hour-1 print is within a factor 10
    of this case's (at least the 8 nearest when fewer qualify). The
    per-hour residual spread sets the quantile width: q_tau =
    median * exp(z_tau * sd_h), z the standard normal quantiles.
  * Leave-one-out pinball loss on practice-like launches (hour-1 print
    10k-400k, the band the four practice cases sit in): ~0.27, against
    ~0.37 for the house persistence baseline on the same launches. On the
    three biggest launches (hour-1 print > 400k) it scores ~0.13 vs ~0.19
    for the baseline. Coverage of the 5%-95% band is ~0.90, as an honest
    forecast should be.

Everything is computed at run time from the case file's own comparables,
so the final case -- whose comparables also hold the four practice
launches' release days -- is forecast from the same rule, with no
hard-coded launch and no network. Pure Python 3.13 standard library,
deterministic: the same case always writes the same bytes.

Run by the grader as `python3 -P -s /bot/main.py`; locally:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]
# The house baseline's spread, used only as a fallback when a case ships
# too few usable comparables to fit anything.
BASE_SIGMA = 0.15
# A comparable counts as "similar" when its hour-1 print is within this
# many decades of the case's hour-1 print.
WINDOW_DEX = 1.0
MIN_TRAIN = 8
# Never claim to be surer than this many log units.
SD_FLOOR = 0.15


def round2(x):
    """Round to 2 decimals, half up, as the house baseline does."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_day_curve(comp):
    """Hour (1..24) -> viewers on the comparable's own release day.

    Returns None when the curve is too thin to trust (fewer than 20 of the
    24 hours, or no hour-1 print): a launch with no real release day in
    the data -- the target game's own pre-release history, for instance --
    never trains the fit.
    """
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    hours = {}
    for row in comp.get("hourly") or []:
        try:
            viewers = row["viewers"]
            at = parse_time(row["from"])
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None or viewers <= 0:
            continue
        h = int((at - day_start).total_seconds() // 3600) + 1
        if 1 <= h <= 24:
            hours[h] = float(viewers)
    if len(hours) < 20 or 1 not in hours:
        return None
    return hours


def fit_line(xs, ys):
    """Least squares y = a + b*x for one feature; returns (a, b)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den < 1e-12:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return my - b * mx, b


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)

    curves = []
    own_name = (case.get("game") or {}).get("name")
    for comp in case.get("comparables") or []:
        # Never train on the target game's own release day, should a case
        # ever ship it: its curve would be the answer, not a comparable.
        if own_name is not None and comp.get("name") == own_name:
            continue
        curve = release_day_curve(comp)
        if curve is not None:
            curves.append(curve)

    rows = []
    for row in case["forecast"]:
        h = row["hour"]
        quantiles = hour_quantiles(curves, p1, log_p1, h)
        rows.append({"hour": h, "q": quantiles})
    return {"hours": rows}


def hour_quantiles(curves, p1, log_p1, h):
    """The five quantiles for one hour, from the comparables' curves."""
    usable = [c for c in curves if h in c and 1 in c]
    if len(usable) < 3:
        # Too little to fit: fall back to the house persistence baseline.
        return [round2(p1 * math.exp(z * BASE_SIGMA * math.sqrt(h - 1)))
                for z in Z]

    log_v1 = math.log10(p1)
    near = [c for c in usable
            if abs(math.log10(c[1]) - log_v1) <= WINDOW_DEX]
    if len(near) < MIN_TRAIN:
        near = sorted(usable,
                      key=lambda c: abs(math.log10(c[1]) - log_v1))[:MIN_TRAIN]

    xs = [math.log(c[1]) for c in near]
    ys = [math.log(c[h]) for c in near]
    a, b = fit_line(xs, ys)
    median = math.exp(a + b * log_p1)
    residuals = [y - (a + b * x) for x, y in zip(xs, ys)]
    mean_resid = sum(residuals) / len(residuals)
    sd = math.sqrt(sum((r - mean_resid) ** 2 for r in residuals)
                   / len(residuals))
    sd = max(SD_FLOOR, sd)

    out = []
    prev = 0.0
    for z in Z:
        q = median * math.exp(z * sd)
        q = min(max(q, 1.0), 1e9)
        q = round2(q)
        # Rounding must never break the non-decreasing order.
        if q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
