#!/usr/bin/env python
"""Build week 8 — "Was our earthquake forecast wrong — or were we just unlucky?"

Emits both notebooks from one source so they cannot drift:

    docs/notebooks/08_wrong_or_unlucky_solution.ipynb   executed, every output saved
    docs/notebooks/08_wrong_or_unlucky.ipynb            the same file with the answers deleted

It also writes the cached fallbacks the week reads, into data/ — except the small-earthquake
catalogue, which is week 7's file read again rather than a second copy of the same query.

Every number that appears in prose or in a model answer is computed HERE, by running the same
code the notebook runs, with the same seed. Nothing is typed from memory or copied from the plan.

Both USGS queries are pinned by date range and magnitude floor, so they reproduce. NOAA's monthly
means are not: the query is pinned to 1900-2025, but the gauge record has gaps the operators
backfill later (nine months are missing as this is written, seven of them in 2024), and the
student's notebook reads NOAA LIVE. A cache downloaded once and never again is how that goes wrong
quietly — the build describes 1,503 months while the class sees more. Three things stop it:
`--refresh` downloads again, a cache older than STALE_DAYS says so rather than passing for
current, and the counts that can move are printed by the notebook's own cells at run time instead
of being written into markdown. What is still literal, and why, is in the build's closing report.

    python tools/build_week08.py              # build from the cached copies
    python tools/build_week08.py --refresh    # download all four again — do this before class
"""
import datetime
import json
import pathlib
import re
import subprocess
import sys
import textwrap
import time
import urllib.request

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/notebooks"
SLUG = "08_wrong_or_unlucky"

course = yaml.safe_load((ROOT / "course.yml").read_text())
WEEK = next(s for s in course["schedule"] if s["n"] == 8)
PLATFORM = course["platform"]
CACHE_BASE = PLATFORM["cache_base"]

# ---------------------------------------------------------------------------
# the three live reads, pinned here so cache, notebook and prose cannot drift
# ---------------------------------------------------------------------------
SEA_URL = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=monthly_mean"
           "&station=9414290&datum=STND&units=metric&time_zone=GMT&format=csv"
           "&begin_date=19000101&end_date=20251231")
SEA_CACHE = "week08_sf_sea_level_1900_2025.csv"

FDSN = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&orderby=time-asc"
CA_BOX = "&minlatitude=32&maxlatitude=42&minlongitude=-125&maxlongitude=-114"
# NO `maxmagnitude`. This query used to carry `&maxmagnitude=6.9`, which the dataset audit names
# and forbids: "a cumulative count is a property of the CATALOGUE, not of a subsample ... at
# magnitude 6 the truncated data reads 16 where California holds 21. Worse, dropping the M7s
# steepens the line and lowers the predicted M7 count, manufacturing part of the very shortfall
# the week exists to argue about" (notes/dataset-audit/usgs-fdsn.md §4.2). What is held back is
# the FITTING RANGE — the line is measured over 4.0-5.5 and then asked about 7.0, which is what
# makes magnitude 7 an extrapolation — never the earthquakes. Truncating cost 0.25 of an event:
# the shipped forecast was 1.64 where the same fit on the whole catalogue gives 1.87.
#
# It is BYTE-FOR-BYTE week 7's query, so it reads week 7's cache rather than shipping a second
# copy of a 1.2 MB file under a week08_ name. Same box, same dates, same floor, same catalogue
# the class fitted last week — which is the point the week's opening paragraph makes anyway.
EQ_URL = (FDSN + "&starttime=1990-01-01&endtime=2026-01-01"
                 "&minmagnitude=3.5" + CA_BOX)
EQ_CACHE = "week07_california_1990-01-01_2026-01-01_M3.5.csv"
BIG_URL = FDSN + "&starttime=1918-01-01&endtime=2026-01-01&minmagnitude=7.0" + CA_BOX
BIG_CACHE = "week08_ca_1918_2026_M7.csv"
# The same query with the threshold moved one notch down, which is the week's threshold-
# sensitivity check: 7.0 is a round number somebody chose, not a boundary in the rock.
BIG69_URL = FDSN + "&starttime=1918-01-01&endtime=2026-01-01&minmagnitude=6.9" + CA_BOX
BIG69_CACHE = "week08_ca_1918_2026_M6.9.csv"

B = 2000            # bootstrap resamples, everywhere in the week
SEED = 88
CHUNK_STARTS = [1900, 1925, 1950, 1975, 2000]
WINDOW_STARTS = [1918, 1954, 1990]
WINDOW_YEARS = 36
AHEAD = 30          # the forecast window week 5 used, and the one this week puts an interval on

REFRESH = "--refresh" in sys.argv
STALE_DAYS = 30     # after this NOAA may hold months the cached copy does not


def cache(url, name, volatile=False, owned_by=None):
    """Fetch a live source, store it byte-for-byte as the week's fallback, return the path.

    The cache used to be downloaded once and then never again, which is the quiet way a week
    goes wrong: the student's notebook reads the archive live, the build reads a months-old
    copy, and the two disagree in front of a class. `--refresh` downloads again. The USGS
    queries are pinned by date range and magnitude floor and reproduce; NOAA's monthly means are
    `volatile`, because a missing month can be filled in inside a window that is already closed,
    so an old cache of them warns.

    `owned_by` marks a file another week downloads and this one only reads. `--refresh` leaves it
    alone: the query is pinned by date range and magnitude floor and reproduces, and rewriting
    another week's cache from here would replace its normalised CSV with raw USGS bytes behind
    its back. Refresh it by rebuilding that week.
    """
    out = ROOT / "data" / name
    if owned_by and out.exists():
        return out
    if REFRESH or not out.exists():
        print(f"downloading {name}")
        out.write_bytes(urllib.request.urlopen(url, timeout=180).read())
    elif volatile:
        age = (time.time() - out.stat().st_mtime) / 86400
        if age > STALE_DAYS:
            print(f"WARNING: data/{name} was downloaded {age:.0f} days ago. NOAA backfills gaps "
                  f"in this record, and the student's notebook reads NOAA live. Rebuild with "
                  f"--refresh before the class.")
    return out


# ---------------------------------------------------------------------------
# 1. measure everything the notebook will say, with the notebook's own code
# ---------------------------------------------------------------------------
sea = pd.read_csv(cache(SEA_URL, SEA_CACHE, volatile=True))
sea.columns = sea.columns.str.strip()
sea["year"] = sea["Year"] + (sea["Month"] - 0.5) / 12
sea["sea_mm"] = sea["MSL"] * 1000

returned = pd.read_csv(cache(EQ_URL, EQ_CACHE, owned_by="week 7"))
big = pd.read_csv(cache(BIG_URL, BIG_CACHE))
big69 = pd.read_csv(cache(BIG69_URL, BIG69_CACHE))

# THE SLICE IS NOT ALL EARTHQUAKES. The query carries no `eventtype`, so what comes back is every
# seismic event the network located inside the box: 16 underground nuclear tests at the Nevada
# Test Site, 6 quarry blasts and 2 sonic booms among the earthquakes. Eight of the tests are
# magnitude 5.2 or larger, which is inside the 4.0-5.5 range this week fits, so they flatten the
# line and raise what it predicts at magnitude 7. They are DROPPED, and dropped VISIBLY in the
# notebook rather than silently by adding `&eventtype=earthquake` to the URL — week 7 made the
# same drop the same way, and week 10 is "Earthquake or explosion — how does the world verify a
# nuclear test ban?", so a student meets these rows twice as a nuisance before meeting them as
# the subject. Keeping the query unfiltered also keeps week 7's shipped cache valid for both.
TYPES = returned["type"].value_counts()
nukes = returned[returned["type"] == "nuclear explosion"]
quakes = returned[returned["type"] == "earthquake"]
big = big[big["type"] == "earthquake"]
big69 = big69[big69["type"] == "earthquake"]

M = {}
M["n_rows"] = len(returned)
M["n_dropped"] = M["n_rows"] - len(quakes)
M["n_nuclear"] = int(TYPES.get("nuclear explosion", 0))
M["nuke_mag_lo"] = float(nukes["mag"].min())
M["nuke_mag_hi"] = float(nukes["mag"].max())
M["nuke_first_year"] = min(t[:4] for t in nukes["time"])
M["nuke_last_year"] = max(t[:4] for t in nukes["time"])
M["n_big_in_slice"] = int((quakes["mag"] >= 7.0).sum())
M["n_months"] = len(sea)
M["first_year"], M["last_year"] = int(sea["Year"].min()), int(sea["Year"].max())
M["n_years_sea"] = M["last_year"] - M["first_year"] + 1
M["n_missing"] = M["n_years_sea"] * 12 - M["n_months"]

fit = LinearRegression().fit(sea[["year"]], sea["sea_mm"])
M["slope"] = float(fit.coef_[0])
M["r2"] = float(fit.score(sea[["year"]], sea["sea_mm"]))

# --- the five 25-year chunks (the attempt that fails) ----------------------
M["chunk_slopes"] = []
for start in CHUNK_STARTS:
    chunk = sea[(sea["year"] >= start) & (sea["year"] < start + 25)]
    M["chunk_slopes"].append(
        float(LinearRegression().fit(chunk[["year"]], chunk["sea_mm"]).coef_[0]))
M["chunk_low"] = min(M["chunk_slopes"])
M["chunk_high"] = max(M["chunk_slopes"])
M["chunk_span"] = M["chunk_high"] - M["chunk_low"]
M["n_chunk"] = int(len(sea[(sea["year"] >= 1900) & (sea["year"] < 1925)]))


def slope_bootstrap(table, seed=SEED, n=B):
    """B bootstrap slopes of sea_mm against year, resampling whole months."""
    np.random.seed(seed)
    out = []
    for _ in range(n):
        boot = table.sample(len(table), replace=True)
        out.append(LinearRegression().fit(boot[["year"]], boot["sea_mm"]).coef_[0])
    return np.array(out)


slopes = slope_bootstrap(sea)
M["ci_low"], M["ci_high"] = (float(x) for x in np.percentile(slopes, [2.5, 97.5]))
M["n_below_zero"] = int((slopes <= 0).sum())
M["boot_min"] = float(slopes.min())
M["boot_max"] = float(slopes.max())
M["iid_width"] = M["ci_high"] - M["ci_low"]

# --- the same interval on the two end chunks ------------------------------
# The class cannot say whether the 4.03 mm/yr spread between the five chunks is short-window
# wobble or a real change of rate without giving the chunks intervals of their own, and saying
# it without them is the error the week's third takeaway is about.
M["ends"] = []
for start in (CHUNK_STARTS[0], CHUNK_STARTS[-1]):
    chunk = sea[(sea["year"] >= start) & (sea["year"] < start + 25)]
    s = slope_bootstrap(chunk)
    M["ends"].append({
        "start": start, "n": len(chunk),
        "slope": float(LinearRegression().fit(chunk[["year"]], chunk["sea_mm"]).coef_[0]),
        "low": float(np.percentile(s, 2.5)), "high": float(np.percentile(s, 97.5))})
for end in M["ends"]:
    end["width"] = end["high"] - end["low"]
M["ends_gap"] = M["ends"][1]["low"] - M["ends"][0]["high"]
M["ends_ratio"] = M["ends"][0]["width"] / M["iid_width"]

M["cm_century"] = M["slope"] * 100 / 10
M["cm_century_low"] = M["ci_low"] * 100 / 10
M["cm_century_high"] = M["ci_high"] * 100 / 10
M["years_to_2100"] = 2100 - 2025
M["cm_2100"] = M["slope"] * M["years_to_2100"] / 10
M["cm_2100_low"] = M["ci_low"] * M["years_to_2100"] / 10
M["cm_2100_high"] = M["ci_high"] * M["years_to_2100"] / 10

# --- the block bootstrap (the last section) -------------------------------
np.random.seed(SEED)
by_year = {y: sub for y, sub in sea.groupby("Year")}
year_list = sorted(by_year)
block_slopes = []
for _ in range(B):
    picked = np.random.choice(year_list, size=len(year_list), replace=True)
    boot = pd.concat([by_year[y] for y in picked])
    block_slopes.append(LinearRegression().fit(boot[["year"]], boot["sea_mm"]).coef_[0])
block_slopes = np.array(block_slopes)
M["block_low"], M["block_high"] = (float(x) for x in np.percentile(block_slopes, [2.5, 97.5]))
M["block_width"] = M["block_high"] - M["block_low"]
M["width_ratio"] = M["block_width"] / M["iid_width"]
M["n_calendar_years"] = len(year_list)
# What the widening MEASURES: an interval that is k times wider is the interval you would have
# got from n / k**2 independent months. 126 is how many blocks the loop drew, which is the
# method, not the result.
M["n_effective"] = M["n_months"] / M["width_ratio"] ** 2

# --- the earthquake forecast ----------------------------------------------
EDGES = np.arange(4.0, 5.6, 0.1).round(1)
M["n_quakes"] = len(quakes)
M["n_bins"] = len(EDGES)
M["count_at_4"] = int((quakes["mag"] >= 4.0).sum())
M["count_at_55"] = int((quakes["mag"] >= 5.5).sum())


def predicted_m7(mags, edges=EDGES):
    """Fit Gutenberg-Richter on these magnitude bins and read off the expected M7+ count."""
    counts = []
    for edge in edges:
        counts.append((mags >= edge).sum())
    line = LinearRegression().fit(edges.reshape(-1, 1), np.log10(counts))
    return 10 ** line.predict([[7.0]])[0]


M["rate"] = float(predicted_m7(quakes["mag"]))
# How many of the non-earthquakes sit inside the range the line is fitted over. That is where
# they do their damage: the count at the top of the range rests on a few dozen events.
M["nuke_in_range"] = int(((nukes["mag"] >= EDGES[0]) & (nukes["mag"] <= EDGES[-1])).sum())
# What each of the two faults cost, measured rather than asserted, for the closing report: the
# shipped week fitted a catalogue truncated at 6.9 with the blasts still in it.
M["rate_as_shipped"] = float(predicted_m7(
    returned[returned["mag"] <= 6.9]["mag"]))
M["rate_blasts_in"] = float(predicted_m7(returned["mag"]))


def rate_bootstrap(mags, edges=EDGES, seed=SEED, n=B):
    """B bootstrap values of the predicted M7+ count, resampling the catalogue."""
    np.random.seed(seed)
    out = []
    for _ in range(n):
        out.append(predicted_m7(mags.sample(len(mags), replace=True), edges))
    return np.array(out)


boot_rates = rate_bootstrap(quakes["mag"])
M["rate_low"], M["rate_high"] = (float(x) for x in np.percentile(boot_rates, [2.5, 97.5]))

np.random.seed(SEED)
boot_counts = np.random.poisson(boot_rates)
M["count_low"], M["count_high"] = (int(x) for x in np.percentile(boot_counts, [2.5, 97.5]))
M["frac_5"] = float((boot_counts >= 5).mean())
M["max_count"] = int(boot_counts.max())

big["when"] = pd.to_datetime(big["time"])
M["window_counts"] = []
for start in WINDOW_STARTS:
    inside = ((big["when"] >= f"{start}-01-01")
              & (big["when"] < f"{start + WINDOW_YEARS}-01-01"))
    M["window_counts"].append(int(inside.sum()))
M["observed"] = M["window_counts"][-1]
M["total_big"] = sum(M["window_counts"])
M["n_windows"] = len(WINDOW_STARTS)
M["span_years"] = WINDOW_YEARS * M["n_windows"]
M["excess_7"] = M["observed"] / np.mean(M["window_counts"][:-1])

# The Poisson draw above assumes the large earthquakes arrive independently. The catalogue's own
# clock says they do not, so the notebook prints the gaps and the caveat names what they mean.
ordered = big.sort_values("time").reset_index(drop=True)
day_gaps = pd.to_datetime(ordered["time"]).diff().dt.days
M["gaps"] = sorted(int(g) for g in day_gaps.dropna())
M["shortest_gap"] = M["gaps"][0]


def short_name(place):
    """'The 1992 Petrolia, California Earthquake' -> '1992 Petrolia'."""
    return f"{place.split(' ')[1]} {place.split(',')[0].split(' ', 2)[2]}"


closest = int(day_gaps.iloc[1:].idxmin())
M["pair"] = [short_name(ordered["place"][closest - 1]), short_name(ordered["place"][closest])]
# The caveat about independence used to add "and the three window counts scatter more than a
# Poisson process allows, their variance 1.63 times their mean". Three counts cannot show that.
# Under a Poisson process with this mean, three counts are at least this dispersed 21% of the
# time — measured here so the claim is not made again. What carries the caveat instead is the
# catalogue's own clock: two of these events are four minutes apart.
np.random.seed(SEED)
_trio = np.random.poisson(np.mean(M["window_counts"]), (200_000, M["n_windows"]))
M["dispersion"] = float(np.var(M["window_counts"], ddof=1) / np.mean(M["window_counts"]))
with np.errstate(invalid="ignore", divide="ignore"):
    M["p_dispersion"] = float(
        (_trio.var(1, ddof=1) / _trio.mean(1) >= M["dispersion"]).mean())

# Week 5 counted events, turned the count into a rate, and turned the rate into a chance. This
# week's exercise is that chance with an interval on it.
M["p_class"] = float(1 - np.exp(-M["rate"] / WINDOW_YEARS * AHEAD))
boot_probs = 1 - np.exp(-boot_rates / WINDOW_YEARS * AHEAD)
M["p_low"], M["p_high"] = (float(x) for x in np.percentile(boot_probs, [2.5, 97.5]))

big69["when"] = pd.to_datetime(big69["time"])
M["window_counts_69"] = []
for start in WINDOW_STARTS:
    inside = ((big69["when"] >= f"{start}-01-01")
              & (big69["when"] < f"{start + WINDOW_YEARS}-01-01"))
    M["window_counts_69"].append(int(inside.sum()))
M["total_big_69"] = sum(M["window_counts_69"])
M["excess_69"] = M["window_counts_69"][-1] / np.mean(M["window_counts_69"][:-1])
ordered69 = big69.sort_values("time").reset_index(drop=True)
when69 = pd.to_datetime(ordered69["time"])
M["gaps_69"] = sorted(int(g) for g in when69.diff().dropna().dt.days)
# The closest pair in the record, which is where "independent events" stops being tenable.
tight = int(when69.diff().iloc[1:].idxmin())
M["pair_69"] = [short_name(ordered69["place"][tight - 1]), short_name(ordered69["place"][tight])]
M["pair_69_minutes"] = int((when69[tight] - when69[tight - 1]).total_seconds() // 60)
# The events the threshold move adds, named so the prose cannot invent a different four.
added = big69[~big69["id"].isin(big["id"])]
M["added_69"] = [short_name(p) for p in added["place"]]
M["n_added_69"] = len(added)

np.random.seed(SEED)
long_counts = np.random.poisson(boot_rates * M["n_windows"])
M["frac_total"] = float((long_counts >= M["total_big"]).mean())
M["long_low"], M["long_high"] = (int(x) for x in np.percentile(long_counts, [2.5, 97.5]))

# --- homework -------------------------------------------------------------
HALVES = [(1900, 1963), (1963, 2026)]
M["halves"] = []
for lo, hi in HALVES:
    part = sea[(sea["year"] >= lo) & (sea["year"] < hi)]
    s = slope_bootstrap(part)
    M["halves"].append({
        "lo": lo, "hi": hi, "n": len(part),
        "slope": float(LinearRegression().fit(part[["year"]], part["sea_mm"]).coef_[0]),
        "low": float(np.percentile(s, 2.5)), "high": float(np.percentile(s, 97.5))})

# WHY the two halves come out alike, measured rather than guessed at. The model answer used to
# say "each half averages slow decades in with fast ones", which is true of the first half and
# false of the second: cut the halves on the same 25-year grid the class used and the second
# half's pieces run fast, yet the half fits BELOW two of the three. A 63-year least-squares line
# is not the average of its pieces — the record steps back down between them, so the pieces rise
# further between them than the half does in total.
SUB_EDGES = [[1900, 1925, 1950, 1963], [1963, 1975, 2000, 2026]]


def piece(lo, hi):
    part = sea[(sea["year"] >= lo) & (sea["year"] < hi)]
    line = LinearRegression().fit(part[["year"]], part["sea_mm"])
    return (float(line.coef_[0]), float(line.predict([[lo]])[0]),
            float(line.predict([[hi]])[0]))


for half, cuts in zip(M["halves"], SUB_EDGES):
    pieces = [piece(lo, hi) for lo, hi in zip(cuts, cuts[1:])]
    half["cuts"] = cuts
    half["sub_slopes"] = [p[0] for p in pieces]
    half["above"] = sum(1 for s in half["sub_slopes"] if s > half["slope"])
    half["piece_rise"] = sum(p[2] - p[1] for p in pieces)
    whole = piece(half["lo"], half["hi"])
    half["own_rise"] = whole[2] - whole[1]
    # the largest downward step between one piece's fitted end and the next piece's fitted start
    half["biggest_step"] = min(pieces[k + 1][1] - pieces[k][2] for k in range(len(pieces) - 1))
M["decade_means"] = {d: float(sea[(sea["year"] >= d) & (sea["year"] < d + 10)]["sea_mm"].mean())
                     for d in (1990, 2000, 2010)}

FORKS = {"low": np.arange(3.5, 5.1, 0.1).round(1), "high": np.arange(4.5, 6.1, 0.1).round(1)}
M["forks"] = {}
for key, edges in FORKS.items():
    r = rate_bootstrap(quakes["mag"], edges)
    np.random.seed(SEED)
    c = np.random.poisson(r)
    M["forks"][key] = {
        "lo": float(edges[0]), "hi": float(edges[-1]),
        "rate": float(predicted_m7(quakes["mag"], edges)),
        "rate_low": float(np.percentile(r, 2.5)), "rate_high": float(np.percentile(r, 97.5)),
        "count_high": int(np.percentile(c, 97.5)), "frac_5": float((c >= 5).mean())}

# --- has the ground moved under the prose? --------------------------------
have_month = set(zip(sea["Year"], sea["Month"]))
M["missing_months"] = [f"{y}-{m:02d}" for y in range(M["first_year"], M["last_year"] + 1)
                       for m in range(1, 13) if (y, m) not in have_month]
M["pinned_months"] = int(re.search(r"([\d,]+) monthly means",
                                   WEEK["pinned"]["slice"]).group(1).replace(",", ""))


def pinned_drift():
    """Re-measure everything course.yml pinned for this week and report anything that moved.

    Every number in the markdown is interpolated from M, so a rebuild refreshes the prose. What
    a rebuild cannot do is notice that the record itself is no longer the record the plan was
    verified against — and course.yml, not the notebook, is where that has to be corrected.
    """
    expect = WEEK["pinned"]["expect"]
    measured = {"slope_mm_per_yr": [M["slope"]],
                "ci95_iid": [M["ci_low"], M["ci_high"]],
                "ci95_block_by_year": [M["block_low"], M["block_high"]],
                "cm_per_century": [M["cm_century"]],
                "cm_now_to_2100": [M["cm_2100"]]}
    moved = []
    if M["n_months"] != M["pinned_months"]:
        moved.append(f"the record holds {M['n_months']:,} monthly means; course.yml's pinned "
                     f"slice says {M['pinned_months']:,}")
    for key, want in expect.items():
        want = want if isinstance(want, list) else [want]
        if any(abs(w - g) > 0.01 * abs(w) for w, g in zip(want, measured[key])):
            moved.append(f"{key}: course.yml pins {want}, this build measures "
                         f"{[round(g, 3) for g in measured[key]]}")
    return moved


if "--measure" in sys.argv:
    print(json.dumps(M, indent=1, default=float))
    sys.exit()

LO, HI = M["forks"]["low"], M["forks"]["high"]

# Two later sections read `slopes`, `ci_low` and `ci_high`, so both open by rebuilding them,
# silently — a checkpoint that rebuilt only the fit sent a restarted student to NameError on the
# very next question. This is the one thing TEMPLATE 8 lets a notebook say twice.
REBUILD_SLOPES = f"""np.random.seed(88)
slopes = []
for i in range({B}):
    boot = sea.sample(len(sea), replace=True)
    slopes.append(LinearRegression().fit(boot[["year"]], boot["sea_mm"]).coef_[0])

slopes = np.array(slopes)
ci_low, ci_high = np.percentile(slopes, [2.5, 97.5])"""

# The homework re-runs the earthquake argument on its own fitting range, and `predicted_m7` is
# the one thing it needs that the setup cell does not build.
REBUILD_FORECAST = '''edges = np.arange(4.0, 5.6, 0.1).round(1)


def predicted_m7(mags):
    """Fit Gutenberg-Richter to these magnitudes and read off the expected number of M7+."""
    counts = []
    for edge in edges:
        counts.append((mags >= edge).sum())
    line = LinearRegression().fit(edges.reshape(-1, 1), np.log10(counts))
    return 10 ** line.predict([[7.0]])[0]'''


# ---------------------------------------------------------------------------
# 2. the cells
# ---------------------------------------------------------------------------
CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n"), None))


def code(text):
    CELLS.append(("code", text.strip("\n"), None))


def ask(text):
    """A question: the markdown that asks. The answer cell follows."""
    md(text)


def answer(model, check=""):
    """A code answer cell: the solution carries the model answer, the student a stub."""
    solution = model.strip("\n") + (("\n\n" + check.strip("\n")) if check else "")
    student = "# ← your answer here\n\n" + (("\n" + check.strip("\n")) if check else "")
    CELLS.append(("code", solution, student))


def answer_prose(model):
    CELLS.append(("markdown", model.strip("\n"),
                  "*(Double-click this cell and replace this line with your answer.)*"))


def note(sentence):
    """The one-sentence answer a part asks for, wrapped as a comment for the solution."""
    return textwrap.fill(" ".join(sentence.split()), width=96,
                         initial_indent="# ", subsequent_indent="# ")


datahub = (f"{PLATFORM['datahub']}/hub/user-redirect/git-pull"
           f"?repo={PLATFORM['repo'].replace(':', '%3A').replace('/', '%2F')}"
           f"&branch={PLATFORM['branch']}"
           f"&urlpath=lab%2Ftree%2FEPS88_PyEarth%2F{PLATFORM['notebook_dir']}%2F{SLUG}.ipynb")

HOOK = f"""
You have already fitted Gutenberg-Richter to California's small earthquakes and used the line to
predict how many magnitude-7 events the state should expect in thirty-six years. Depending which
of three defensible ranges you fitted it over, the line said somewhere between {LO['rate']:.2f}
and {HI['rate']:.2f}. Five happened. You were asked to write down whether you thought your model
had failed, and the honest answer was that you could not tell — because {M['rate']:.2f} and 5 are
two bare numbers, and two bare numbers cannot be compared.

What is missing is a sense of how far that {M['rate']:.2f} would have moved if the earthquakes had
fallen slightly differently. Today you build it, out of nothing but the data you already have. You will
practise on something slower — a hundred and twenty-six years of sea level in San Francisco Bay,
where the question is whether the water is rising at all — and then take the same six lines back
to the earthquakes and settle the argument.
"""

md(weekkit.OPENING.format(question=WEEK["question"], datahub=datahub, hook=HOOK.strip()))

md("""
## What you'll be able to do

**The science.** Say how fast San Francisco Bay is rising, with a range rather than a number, and
say whether the rise could be zero. Then say whether California's five large earthquakes are
evidence that a Gutenberg-Richter forecast is broken, or the kind of run of luck a working
forecast produces anyway — and know which of those two questions your interval answered.

**The skills.** Resampling: `table.sample(n, replace=True)` draws a new dataset out of the one you
have, `np.percentile` turns a thousand answers into an interval, and a `for` loop puts the two
together. You will put an error bar on a fitted slope, which is something a single call to
`LinearRegression` will never give you.

**The four questions, in order:**

1. Could the Bay's rise be zero?
2. How much water is that by 2100?
3. Was the forecast wrong, or were we unlucky?
4. What did we assume without saying so?

**Nine turns where you write something: six in class, three at home.** Each one is headed *Your
turn*, with an empty cell under it — turn 5 has two, one for code and one for your verdict in
words.
""")

code(weekkit.setup_cell(
    imports="import numpy as np\nfrom sklearn.linear_model import LinearRegression\n",
    figsize="(7, 4)",
    cache_base=CACHE_BASE,
    signature="url, cache_name",
    docstring="Read one live source; fall back to the copy stored with the course.",
    url_expr="url",
    cache_expr="cache_name",
    unpack=f'''
sea = load("{SEA_URL}",
           "{SEA_CACHE}")
sea.columns = sea.columns.str.strip()      # the NOAA file pads its column names with spaces
sea["year"] = sea["Year"] + (sea["Month"] - 0.5) / 12   # 1903.5 means the middle of 1903
sea["sea_mm"] = sea["MSL"] * 1000                       # the file is in metres, we want mm

FDSN = "{FDSN}"
CA_BOX = "{CA_BOX}"

quakes = load(FDSN + "&starttime=1990-01-01&endtime=2026-01-01"
                     "&minmagnitude=3.5" + CA_BOX,        # the same query as last week's
              "{EQ_CACHE}")
big = load(FDSN + "&starttime=1918-01-01&endtime=2026-01-01&minmagnitude=7.0" + CA_BOX,
           "{BIG_CACHE}")

print("sea level:", sea.shape, " small earthquakes:", quakes.shape, " large ones:", big.shape)
'''.strip("\n")))

# --- section 1 -------------------------------------------------------------
md(f"""
## Could the Bay's rise be zero?

A tide gauge is a float in a stilling well, bolted to a pier, writing down where the water sits.
The one in San Francisco — NOAA station 9414290 — has been doing that since before anyone thought
sea level was a question, and NOAA publishes a monthly mean from it. That is what `sea` holds: one
monthly mean per month from {M['first_year']} to {M['last_year']}, in the `MSL` column, which the
setup cell turned into millimetres in `sea_mm`. The setup cell printed how many rows that is, and
it is short of the {M['n_years_sea'] * 12:,} months those {M['n_years_sea']} years would hold: the
gauge has gaps, and the most recent of them are new enough that NOAA may yet fill them in.

Two warnings about what that column is. The zero is arbitrary: the heights are quoted against
*station datum*, a mark on the pier chosen for convenience, so the height itself means nothing and
only the *change* does. And a tide gauge measures the sea against the land it is bolted to, so
what it records is **relative** sea level: if the pier were sinking, the water would appear to
rise by exactly the same amount.
""")

code("""
plt.scatter(sea["year"], sea["sea_mm"], s=2)
plt.xlabel("year")
plt.ylabel("monthly mean sea level (mm above station datum)")
plt.title(f"San Francisco, {len(sea):,} monthly means")
plt.show()
""")

md("""
There is a rise in there, and a great deal of noise on top of it: winter storms, El Niño years,
the seasons. A straight line through it is the regression you already know, with `year` as the one
input column and `sea_mm` as the thing to predict. Its slope is millimetres per year.
""")

code("""
fit = LinearRegression().fit(sea[["year"]], sea["sea_mm"])
slope = fit.coef_[0]

print("slope:", round(slope, 3), "mm per year")
print("R squared:", round(fit.score(sea[["year"]], sea["sea_mm"]), 3))
""")

md(f"""
{M['slope']:.3f} millimetres a year, and an R squared of {M['r2']:.3f} — the line explains about
half of what the gauge did, and the rest is the month-to-month weather and ocean variability the
line cannot see. Now ask the question that matters: **could
the answer be zero?** The number {M['slope']:.3f} does not answer it. It has nothing after it.

Here is the obvious first move. If the record were telling us something stable, then any big piece
of it should give roughly the same slope. So cut it up and look.
""")

ask(f"""
### ✏️ Your turn 1

Cut the record into five twenty-five-year pieces and fit the same line to each one on its own.

Loop over `{CHUNK_STARTS}`. For each `start`, keep the rows with
`(sea["year"] >= start) & (sea["year"] < start + 25)`, fit `LinearRegression` to that piece exactly
as the cell above did, and append the slope to a list. Print each piece's start year, how many
months it holds, and its slope.

**Use these names**, because the self-check looks for them: `chunk_slopes`.
""")

answer(f"""
chunk_slopes = []

for start in {CHUNK_STARTS}:
    chunk = sea[(sea["year"] >= start) & (sea["year"] < start + 25)]
    chunk_fit = LinearRegression().fit(chunk[["year"]], chunk["sea_mm"])
    chunk_slopes.append(chunk_fit.coef_[0])
    print(start, "-", start + 25, ":", len(chunk), "months, slope",
          round(chunk_fit.coef_[0], 3), "mm/yr")
""", f"""
assert chunk_slopes[-1] > chunk_slopes[0] + 1, \\
    "the last piece should be more than 1 mm/yr steeper than the first"
assert 3 < max(chunk_slopes) - min(chunk_slopes) < 5, \\
    "the five slopes should span about {M['chunk_span']:.0f} mm/yr \u2014 check the filter"
print("\u2713 the record cut five ways \u2014 slopes from", round(min(chunk_slopes), 2),
      "to", round(max(chunk_slopes), 2), "mm/yr, against", round(slope, 2), "for the whole record")
""")

# --- section 2 -------------------------------------------------------------
md(f"""
Five pieces of the same record, and the slopes run from {M['chunk_low']:.2f} to
{M['chunk_high']:.2f} mm/yr. The first quarter of the twentieth century says the Bay was barely
moving; the last twenty-five years say four millimetres a year.

That is honest but useless, for two separate reasons. First, two different things are mixed
together in that spread and staring at five numbers cannot separate them: how much a
twenty-five-year slope wobbles by chance, and whether the rise has genuinely sped up. Second — and
this is the fatal one — we did not ask how much a *twenty-five-year* slope wobbles. We asked how
much the {M['n_years_sea']}-year slope wobbles, and each of those fits threw away four fifths of
the data. Five numbers also cannot tell you what "95% of the time" means.

What we want is the whole record's worth of answer, many times over. We only have one record. So
we make more of it out of the one we have.

### How do you get an error bar out of one record?

The trick has a name and one line of code. **Bootstrap:** *Ask the data the same question a
thousand times, using a different random slice of itself each time.*

The slice is drawn **with replacement**: pick a row at random, write it down, put it back, and
pick again, until you have as many rows as you started with. Some rows get picked twice, some not
at all — and that is the whole point, because that is what a second run of history would have
looked like. Watch it happen on eight numbers first.
""")

code("""
np.random.seed(88)
eight = pd.DataFrame({"height": [1, 2, 3, 4, 5, 6, 7, 8]})

print("the original:", list(eight["height"]))
print("one resample:", list(eight.sample(8, replace=True)["height"]))
print("another:     ", list(eight.sample(8, replace=True)["height"]))
""")

md(f"""
Repeats in one, holes in the other, same length as the original. `np.random.seed` is there for the
same reason as in the probability week: it makes the randomness repeatable, so your numbers and
your neighbour's match.

Now do it to the tide gauge. One resample of `sea` is as many months as the record holds, drawn
with replacement from those same months; fit the line to that and you get a slope that is nearly,
but not quite, {M['slope']:.3f}. Do it {B:,} times and you have {B:,} slopes.

Notice which way round this is. When you simulated earthquake times you started from a
distribution you had *assumed* — `np.random.poisson(lam)` — and asked what worlds it would make.
Here you assume nothing at all and draw from the data itself.
""")

ask(f"""
### ✏️ Your turn 2

Bootstrap the slope, and turn the answer into a range.

Seed with `np.random.seed(88)`. Then, {B:,} times: draw `sea.sample(len(sea), replace=True)`, fit
`LinearRegression` to that resample as you have twice already, and append `.coef_[0]` to a list.
Make the finished list an array with `np.array(...)` so you can do arithmetic on it.

Then read the middle 95% off it: `np.percentile(slopes, [2.5, 97.5])` returns the two values that
cut off the bottom 2.5% and the top 2.5%. Print the slope, the two ends, and how many of your
{B:,} slopes came out at zero or below — `(slopes <= 0).sum()`.

Then, as a comment at the end of your cell: **how many of your {B:,} resamples make the Bay flat,
and what does that let you say that the single number {M['slope']:.2f} did not?**

**Use these names**, because the self-check looks for them: `slopes`, `ci_low`, `ci_high`.
""")

answer(f"""
np.random.seed(88)
slopes = []

for i in range({B}):
    boot = sea.sample(len(sea), replace=True)
    slopes.append(LinearRegression().fit(boot[["year"]], boot["sea_mm"]).coef_[0])

slopes = np.array(slopes)
ci_low, ci_high = np.percentile(slopes, [2.5, 97.5])

print("slope:", round(slope, 3), "mm/yr")
print("95% interval:", round(ci_low, 3), "to", round(ci_high, 3), "mm/yr")
print("resamples at zero or below:", (slopes <= 0).sum(), "out of", len(slopes))

{note(f"{M['n_below_zero']} of my {B:,} resamples make the Bay flat, so a rise of zero is not "
       f"something this record can be made to say by redrawing it; {M['slope']:.2f} on its own "
       f"could not rule that out, because a single number carries no information about how far "
       f"it would have moved had the months fallen differently.")}
""", f"""
assert slopes.min() > 0, \\
    "no resample of this record makes the Bay flat — check you fitted sea_mm against year"
assert ci_low < slope < ci_high, "the interval should straddle the slope you started from"
print("\u2713 the slope, with an interval \u2014", round(slope, 2), "mm/yr, 95% interval",
      round(ci_low, 2), "to", round(ci_high, 2))
""")

md(f"""
That range is a **confidence interval**: *Not one number but the range your number would have
wandered over, had the world rolled differently.* Written the way a paper would write it:
{M['slope']:.2f} mm/yr, 95% CI [{M['ci_low']:.2f}, {M['ci_high']:.2f}].

Every one of those {B:,} slopes is worth drawing, because the interval is just two cuts through a
shape.
""")

code(f"""
plt.hist(slopes, bins=40)
plt.axvline(ci_low, color="firebrick")
plt.axvline(ci_high, color="firebrick")
plt.xlabel("bootstrap slope (mm per year)")
plt.ylabel("number of resamples")
plt.title(f"{{len(slopes):,}} bootstrap slopes, 95% interval marked")
plt.show()
""")

md(f"""
A hump, and the red lines cut {int(round(0.025 * B))} resamples off each side. Two things to read
off it. Nothing in it comes anywhere near zero — the count you printed of resamples at zero or
below was {M['n_below_zero']} out of {B:,} — so the answer to *could the Bay be flat?* is no, and
now we can say so rather than assert it.

Second, look at how narrow it is: {M['iid_width']:.2f} mm/yr from end to end, against the
{M['chunk_span']:.2f} mm/yr that separated the highest of your five chunks from the lowest. It is
tempting to read that as the chunks having been noise all along — and it is the exact mistake this
week exists to prevent. The two numbers are not the same kind of thing. {M['iid_width']:.2f} is a
95% interval; {M['chunk_span']:.2f} is the largest of five point estimates minus the smallest.
Setting one beside the other settles nothing, in either direction.

What would settle it is giving the chunks intervals of their own, and that is the loop you have
just written, run on a slice of `sea` instead of all of it. Do it for the two ends.
""")

code(f"""
for start in [{CHUNK_STARTS[0]}, {CHUNK_STARTS[-1]}]:
    chunk = sea[(sea["year"] >= start) & (sea["year"] < start + 25)]

    np.random.seed(88)
    chunk_boot = []
    for i in range({B}):
        boot = chunk.sample(len(chunk), replace=True)
        chunk_boot.append(LinearRegression().fit(boot[["year"]], boot["sea_mm"]).coef_[0])

    end_low, end_high = np.percentile(chunk_boot, [2.5, 97.5])
    print(start, "-", start + 25, ": 95% interval",
          round(end_low, 2), "to", round(end_high, 2), "mm/yr")
""")

# --- section 3 -------------------------------------------------------------
md(f"""
[{M['ends'][0]['low']:.2f}, {M['ends'][0]['high']:.2f}] for the first quarter-century and
[{M['ends'][1]['low']:.2f}, {M['ends'][1]['high']:.2f}] for the last. They do not overlap: there
are {M['ends_gap']:.2f} mm/yr of daylight between the top of one and the bottom of the other.

So both halves of the story are true and only one of them was worth saying. Short windows really
are wobbly — the first chunk's interval is {M['ends'][0]['width']:.2f} mm/yr wide, nearly
{M['ends_ratio']:.0f} times the whole record's {M['iid_width']:.2f}, because a quarter of the data
has to fit through a quarter-century of weather. But that wobble is nowhere near large enough to
open a gap of {M['ends_gap']:.2f}. Something in this record really did change, and
{M['slope']:.3f} mm/yr is an average across a rate that was not constant. The interval around it
says how well we know that average — not that the Bay rose steadily.
""")

md("""
## How much water is that by 2100?

Millimetres per year is not a quantity anyone plans a seawall around. Two conversions get asked
for, they are not the same number, and they are confused constantly:

- **per century** — how much the water rises in a hundred years at this rate.
- **from the end of this record to 2100** — which is a shorter stretch, so it is a smaller number.

Whichever you quote, the interval comes with it: convert `ci_low` and `ci_high` exactly as you
convert the slope, and the answer stays a range.
""")

code(weekkit.CHECKPOINT.format(body='''fit = LinearRegression().fit(sea[["year"]], sea["sea_mm"])
slope = fit.coef_[0]

''' + REBUILD_SLOPES))

ask(f"""
### ✏️ Your turn 3

Report the rise both ways, each as a point estimate and a 95% interval, in **centimetres**.

There are 10 millimetres in a centimetre. The record ends at the close of {M['last_year']}, so
2100 is {M['years_to_2100']} years away — not 100.

Print two lines: the rise per century, and the rise from the end of the record to 2100, each with
its interval. Do the conversion on `ci_low` and `ci_high` as well as on `slope`.

**Use these names**, because the self-check looks for them: `cm_century`, `cm_2100`.
""")

answer(f"""
cm_century = slope * 100 / 10
cm_2100 = slope * {M['years_to_2100']} / 10

print("per century:      ", round(cm_century, 1), "cm  95% interval",
      round(ci_low * 100 / 10, 1), "to", round(ci_high * 100 / 10, 1))
print("end of record to 2100:", round(cm_2100, 1), "cm  95% interval",
      round(ci_low * {M['years_to_2100']} / 10, 1), "to",
      round(ci_high * {M['years_to_2100']} / 10, 1))
""", """
assert cm_2100 < cm_century, "2100 is closer than a century away, so it must be the smaller number"
print("\u2713 the same slope, two questions \u2014", round(cm_century, 1),
      "cm per century, but", round(cm_2100, 1), "cm between the end of the record and 2100")
""")

# --- section 4 -------------------------------------------------------------
md(f"""
{M['cm_century']:.1f} cm per century and {M['cm_2100']:.1f} cm by 2100 are the same slope answering
two different questions, and quoting one for the other is a {M['cm_century'] - M['cm_2100']:.0f}-centimetre
mistake.

Be careful about what the second number is. It is what the water does **if the rise continues at
the average rate of the last {M['n_years_sea']} years**, and the chunk slopes you computed give a
plain reason to doubt that: the most recent twenty-five years came out at {M['chunk_slopes'][-1]:.1f}
mm/yr, more than twice the long-run figure. The interval [{M['cm_2100_low']:.1f},
{M['cm_2100_high']:.1f}] cm is the uncertainty in *this straight line*, not the uncertainty in what
the ocean will do. It is a floor, not a forecast.
""")

md(f"""
## Was the forecast wrong, or were we unlucky?

Back to the earthquakes, with the same six lines.

`quakes` is last week's file, read again: every **event** of magnitude 3.5 and above the USGS
recorded in the box between 1990 and 2026. Two things about it are worth saying out loud before
anything is fitted, and you met both last week.

The first is that a seismic catalogue lists things that shook the ground, not earthquakes:
{M['n_dropped']} of these {M['n_rows']:,} rows are something else, and {M['n_nuclear']} of those
are underground **nuclear tests** at the Nevada Test Site, magnitude {M['nuke_mag_lo']} to
{M['nuke_mag_hi']}, fired between {M['nuke_first_year']} and {M['nuke_last_year']}. That matters
here because {M['nuke_in_range']} of them land inside the magnitudes this section is about to fit
a line through. Drop them, visibly, and look at what is left — and remember where they went,
because week 10 is *"Earthquake or explosion — how does the world verify a nuclear test ban?"* and
these are the events it is about.

The second is that the {M['n_big_in_slice']} magnitude 7s stay **in**. It is tempting to take them
out, since they are the thing being predicted — and it is wrong: "how many events reached
magnitude 5" is a fact about the catalogue, and a count that quietly skips the largest events is
not that count. What today holds back is the *range the line is fitted over*, never the
earthquakes.

The Gutenberg-Richter recipe is then the one you already know: count how many events reached each
magnitude, plot those counts on a log axis against magnitude, fit a straight line, and read the
line off at a magnitude you have not observed. Written as a function it is short enough to call
{B:,} times.
""")

code("""
print(quakes["type"].value_counts())

n_rows = len(quakes)
quakes = quakes[quakes["type"] == "earthquake"]   # the line that does it
big = big[big["type"] == "earthquake"]            # same rule, same catalogue
print("\\nkept", len(quakes), "earthquakes out of", n_rows, "rows")
print("magnitude 7 and above still in:", (quakes["mag"] >= 7.0).sum())
""")

# No checkpoint here. This section reads nothing that an earlier SECTION built: `quakes` comes
# from the setup cell, and TEMPLATE 1.4 asks for a checkpoint only where a section needs state
# from an earlier one. The cell that stood here re-issued the setup cell's own query verbatim —
# and could not have run at all unless setup had already run, since it needs `load`, `FDSN` and
# `CA_BOX` from it. A checkpoint that rebuilds nothing teaches that a checkpoint is a ritual.
code(f"""
edges = np.arange(4.0, 5.6, 0.1).round(1)      # fit between magnitude 4.0 and 5.5


def predicted_m7(mags):
    \"\"\"Fit Gutenberg-Richter to these magnitudes and read off the expected number of M7+.\"\"\"
    counts = []
    for edge in edges:
        counts.append((mags >= edge).sum())
    line = LinearRegression().fit(edges.reshape(-1, 1), np.log10(counts))
    return 10 ** line.predict([[7.0]])[0]


print("events at magnitude 4.0 and above:", (quakes["mag"] >= 4.0).sum())
print("events at magnitude 5.5 and above:", (quakes["mag"] >= 5.5).sum())
print("expected number of M7+ in the same span:", round(predicted_m7(quakes["mag"]), 2))
""")

md(f"""
{M['rate']:.2f} expected, against five that actually happened. That is where the argument stopped
when you first fitted the line — we fit {EDGES[0]} to {EDGES[-1]} this week rather than
{LO['lo']} to {LO['hi']}, one of the three defensible ranges you tried last week, which is why
this is {M['rate']:.2f} and not the {LO['rate']:.2f} you may have written down then.

### Predict before you run

Bootstrap that {M['rate']:.2f} the way you just bootstrapped the slope and you get a 95% interval
around it. **How high do you think the top of that interval reaches?** Commit to a number — change
`my_guess` and run it — and then find out.
""")

CELLS.extend(("code", s, a) for s, a in
             weekkit.predict_cell("3.0", "M7+ earthquakes at the top of that interval, "
                                         "against the five that actually happened"))

ask(f"""
### ✏️ Your turn 4

Bootstrap the forecast. This is the loop you wrote for the slope, with one line changed.

Seed with `np.random.seed(88)`. Then, {B:,} times: draw
`quakes["mag"].sample(len(quakes), replace=True)`, pass it to `predicted_m7`, and append the
answer. Make the list an array, then take `np.percentile(boot_rates, [2.5, 97.5])`.

Print the forecast and its interval.

**Use these names**, because the self-check looks for them: `boot_rates`, `rate_low`, `rate_high`.
""")

answer(f"""
np.random.seed(88)
boot_rates = []

for i in range({B}):
    boot_mags = quakes["mag"].sample(len(quakes), replace=True)
    boot_rates.append(predicted_m7(boot_mags))

boot_rates = np.array(boot_rates)
rate_low, rate_high = np.percentile(boot_rates, [2.5, 97.5])

print("forecast:", round(predicted_m7(quakes["mag"]), 2), "M7+ earthquakes")
print("95% interval:", round(rate_low, 2), "to", round(rate_high, 2))
""", f"""
assert rate_low < predicted_m7(quakes["mag"]) < rate_high, \\
    "the interval should straddle the forecast \u2014 check you resampled inside the loop"
assert rate_high < 5, "if your interval reaches five, predicted_m7 was fed something else"
print("\u2713 the forecast, with an interval \u2014", round(rate_low, 2), "to",
      round(rate_high, 2), "M7+ earthquakes, against 5 observed")
""")

code(f"""
plt.hist(boot_rates, bins=40)
plt.axvline(rate_high, color="firebrick")
plt.axvline(5, color="black")
plt.xlabel("bootstrap forecast (M7+ earthquakes in {M['span_years'] // M['n_windows']} years)")
plt.ylabel("number of resamples")
plt.title(f"{{len(boot_rates):,}} bootstrap forecasts; interval top red, 5 observed black")
plt.show()
""")

md(f"""
The interval runs [{M['rate_low']:.2f}, {M['rate_high']:.2f}], and the black line at 5 is off past
the end of everything. Not one of the {B:,} resampled catalogues produced a forecast anywhere near
five. Case closed?

No — and this is the most important paragraph in the week. That interval answers the question
*how well do we know the average rate?* Five is not an average rate. Five is a **count**, one
draw of a thing that scatters even when the rate is exactly right, and you have already simulated
exactly that scatter: `np.random.poisson(lam)` turns a rate into the number of events an
individual thirty-six years actually delivers. Comparing a count to an interval on a rate is
comparing two different quantities.

So put the two sources of wobble together. Take each of your {B:,} bootstrapped rates, and let the
world roll once at that rate.
""")

ask(f"""
### ✏️ Your turn 5

Two things here: the count, and then the week's question. **(a)** is code, **(b)** is words.

**(a) Turn the interval on the rate into an interval on the count.**
`np.random.poisson(boot_rates)` draws one Poisson count for every rate in the array at once, so
this needs no loop: seed with `np.random.seed(88)`, make `boot_counts`, and take
`np.percentile(boot_counts, [2.5, 97.5])`.

Then print the fraction of those simulated worlds that delivered five or more —
`(boot_counts >= 5).mean()`. That fraction is the answer to *were we unlucky?*

**Use these names**, because the self-check looks for them: `boot_counts`.

**(b) Then answer the question, in the markdown cell under the histogram.** *Was the forecast
wrong, or were we unlucky?* Two or three sentences, quoting your own numbers — the interval on the
rate from your turn 4, the interval on the count you have just printed, and the fraction of
simulated worlds that reached five. Say which of the two words you would use, or say that your
numbers do not let you choose between them and what it is about them that stops you.

The cell after yours reads the same histogram and reaches its own verdict. Yours is worth more if
you write it first, so commit before you scroll past it.
""")

answer("""
np.random.seed(88)
boot_counts = np.random.poisson(boot_rates)

count_low, count_high = np.percentile(boot_counts, [2.5, 97.5])

print("95% interval on the COUNT:", count_low, "to", count_high, "M7+ earthquakes")
print("fraction of simulated worlds with 5 or more:", round((boot_counts >= 5).mean(), 4))
print("the busiest simulated world had:", boot_counts.max())
""", f"""
assert boot_counts.max() >= 5, \\
    "a Poisson draw reaches five now and then; rounding the rates off never would"
assert (boot_counts == 0).sum() > 0, \\
    "and it delivers empty thirty-six-year stretches too — check you drew from Poisson"
print("\u2713 an interval on the count \u2014 5 or more happened in",
      round(100 * (boot_counts >= 5).mean(), 1), "% of simulated worlds")
""")

code("""
plt.hist(boot_counts, bins=np.arange(-0.5, boot_counts.max() + 1.5, 1))
plt.axvline(5, color="black")
plt.xlabel("M7+ earthquakes in a simulated 36 years")
plt.ylabel("number of simulated worlds")
plt.title(f"{len(boot_counts):,} simulated worlds, 5 observed marked in black")
plt.show()
""")

# Your turn 5(b): the decision cell. It carries no ✏️ heading of its own — the pencil above owns
# both halves of turn 5 — so the notebook still has nine turns and the homework keeps its own
# prose part, which is the one that is hand-marked. What matters is the ORDER: the student's
# verdict is written between the histogram and the paragraph that reads it, so the notebook
# cannot hand them the word before they have chosen it.
answer_prose(f"""
I would say unlucky rather than wrong, but only just. The interval on the *rate* is
[{M['rate_low']:.2f}, {M['rate_high']:.2f}] and five is nowhere near it, which on its own looks
damning; once the Poisson scatter is folded in, though, the interval on the *count* runs
[{M['count_low']}, {M['count_high']}] and five sits on its top edge, with
{100 * M['frac_5']:.1f}% of my simulated worlds delivering five or more. That is about one
thirty-six-year stretch in {round(1 / M['frac_5']):.0f}, which is unlikely enough to be worth
explaining and far too common to call a broken model — so what my numbers really say is that a
single window of this length cannot separate the two, not that I have chosen between them.
""")

md(f"""
The picture is completely different. Once the counting noise is in, worlds with five **or more**
large earthquakes do turn up: {100 * M['frac_5']:.1f}% of them, and the busiest reached
{M['max_count']}. The 95% interval on the count is [{M['count_low']}, {M['count_high']}], and five
is sitting on its upper edge rather than outside it.

Read that percentage carefully, because the "or more" is doing work. It is the fraction of worlds
that reached *at least* five, which is what you want when you are asking whether an observation is
extreme — the worlds that landed on exactly five are a smaller share again. A tail is always
counted outward from the observation, never at it.

So the honest verdict on this window is *unlikely, not impossible* — five or more is a
one-in-{round(1 / M['frac_5']):.0f} outcome. That is the kind of result that should send you to look at more data rather than to a
conclusion. And there is more data: thirty-six years is not the only thirty-six years California
has had.
""")

code(f"""
big["when"] = pd.to_datetime(big["time"])
print(big[["mag", "place"]])
print("days from each one to the next:", sorted(big["when"].diff().dropna().dt.days))

window_counts = []
for start in {WINDOW_STARTS}:
    inside = (big["when"] >= f"{{start}}-01-01") & (big["when"] < f"{{start + {WINDOW_YEARS}}}-01-01")
    window_counts.append(inside.sum())
    print(start, "-", start + {WINDOW_YEARS}, ":", inside.sum(), "M7+ earthquakes")

# and then the same question asked of all {M['span_years']} years at once: if the rate really is
# what Gutenberg-Richter says, how often does a stretch this long deliver as many as we saw?
np.random.seed(88)
long_counts = np.random.poisson(boot_rates * {M['n_windows']})

print()
print("expected over", {M['span_years']}, "years:", round({M['n_windows']} * predicted_m7(quakes["mag"]), 2))
print("observed:", sum(window_counts))
print("fraction of simulated worlds reaching that:",
      round((long_counts >= sum(window_counts)).mean(), 3))
""")

md(f"""
{M['window_counts'][0]}, {M['window_counts'][1]}, {M['observed']}. The forecast of {M['rate']:.2f}
per {WINDOW_YEARS} years is an excellent description of the two earlier windows and a poor one of
the most recent — but stop treating the most recent one as the whole story. Over all
{M['span_years']} years, {M['total_big']} large earthquakes happened against an expected
{M['n_windows'] * M['rate']:.1f}, and {100 * M['frac_total']:.0f}% of simulated
{M['span_years']}-year worlds do at least that well. On the long record the forecast is not in
trouble at all.
""")

ask(f"""
### ✏️ Your turn 6

One last number out of that rate — and this time the interval comes with it from the start.

In the probability week you counted earthquakes near campus, divided by the years to get a rate,
and turned the rate into a chance of at least one with `1 - np.exp(-rate * years)`. You reported
that chance as a single number, because a single rate was all you had. You now have {B:,} rates.

`boot_rates` counts M7+ earthquakes per {WINDOW_YEARS} years, so `boot_rates / {WINDOW_YEARS}` is
a rate per year, and `1 - np.exp(-boot_rates / {WINDOW_YEARS} * {AHEAD})` is the chance of at
least one M7+ inside the box in the next {AHEAD} years — all {B:,} of them at once, no loop
needed.

Print the chance the class forecast gives, which is the same formula applied to
`predicted_m7(quakes["mag"])`, and then the 95% interval from `np.percentile`.

**Use these names**, because the self-check looks for them: `boot_probs`, `p_low`, `p_high`.
""")

answer(f"""
boot_probs = 1 - np.exp(-boot_rates / {WINDOW_YEARS} * {AHEAD})
p_low, p_high = np.percentile(boot_probs, [2.5, 97.5])

print("chance of at least one M7+ in the next {AHEAD} years:",
      round(1 - np.exp(-predicted_m7(quakes["mag"]) / {WINDOW_YEARS} * {AHEAD}), 3))
print("95% interval:", round(p_low, 3), "to", round(p_high, 3))
""", f"""
assert p_low < 1 - np.exp(-predicted_m7(quakes["mag"]) / {WINDOW_YEARS} * {AHEAD}) < p_high, \\
    "the interval should straddle the chance the class forecast gives"
assert boot_probs.max() < 0.99, \\
    "divide the rate by {WINDOW_YEARS} before multiplying it by {AHEAD}"
print("✓ a probability with an interval — between", round(100 * p_low), "and",
      round(100 * p_high), "% chance of at least one M7+ in the next {AHEAD} years")
""")

# --- section 5 -------------------------------------------------------------
md(f"""
## What did we assume without saying so?

Four caveats you should carry, because none of them is settled by anything above. The box is a
rectangle of latitude and longitude, not the state, so it collects earthquakes in Nevada and Baja
California as well: of the {M['total_big']} large earthquakes listed above, Fairview Peak is in
Nevada and Sierra El Mayor is in Baja California. And the earlier windows
depend on a catalogue that was thinner: *A catalogue lists what somebody's instruments recorded,
not what happened.* If those windows are undercounted, the true long-run rate is higher than
{M['total_big']} in {M['span_years']} years, and that pushes the same way — it makes five look less
exceptional, not more.

The third is an assumption every `np.random.poisson` above made without saying so: that
large earthquakes arrive **independently**, one roll of the dice each. The gaps you printed say
they do not. The shortest is {M['shortest_gap']} days — {M['pair'][0]} in April and
{M['pair'][1]} in June, two months apart in a record whose typical gap is years — and in a moment
you will meet a pair four minutes apart. Clustered events make a
count scatter *more* than Poisson, so the true share of thirty-six-year worlds reaching five or
more is **larger** than the {100 * M['frac_5']:.1f}% you computed, not smaller. That is the same
direction as the caveat before it, and it leaves the verdict standing — but it is an assumption,
it is visible in our own table, and it should have been said out loud.

The fourth is the one that moves the numbers, and it is hiding inside the words *magnitude 7*. A
magnitude is a measurement with an uncertainty of a tenth or two, and 7.0 is a round number
somebody chose, not a boundary in the rock. So try the number on the other side of it. Re-run the
same query with the threshold at 6.9 — still an earthquake any seismologist would describe as
roughly magnitude 7 — and count the same three windows again.
""")

code(f"""
big69 = load(FDSN + "&starttime=1918-01-01&endtime=2026-01-01&minmagnitude=6.9" + CA_BOX,
             "{BIG69_CACHE}")
big69 = big69[big69["type"] == "earthquake"]     # same rule as before, though none are dropped
big69["when"] = pd.to_datetime(big69["time"])
print("days from each one to the next:", sorted(big69["when"].diff().dropna().dt.days))

for start in {WINDOW_STARTS}:
    inside = (big69["when"] >= f"{{start}}-01-01") & (big69["when"] < f"{{start + {WINDOW_YEARS}}}-01-01")
    print(start, "-", start + {WINDOW_YEARS}, ":", inside.sum(), "M6.9+ earthquakes")
""")

md(f"""
{M['window_counts_69'][0]}, {M['window_counts_69'][1]}, {M['window_counts_69'][2]} — against
{M['window_counts'][0]}, {M['window_counts'][1]}, {M['observed']} a moment ago. One tenth of a
magnitude adds {M['n_added_69']} earthquakes ({', '.join(M['added_69'])}), every one of them in an
earlier window. The counts are still rising and the most recent window is still the busiest, so
the excess has not vanished — it has **shrunk**, from {M['excess_7']:.1f} times the average of the
two earlier windows to {M['excess_69']:.1f} times. The difference matters: {M['excess_7']:.1f}
times is the sort of gap that starts an argument, and {M['excess_69']:.1f} times is what Poisson
scatter around a mean of {np.mean(M['window_counts_69']):.0f} hands you routinely.

One of the {M['n_added_69']} is worth a second look on its own. {M['pair_69'][1]} is the zero in
that list of gaps: it follows {M['pair_69'][0]} by {M['pair_69_minutes']} minutes, and the two are
counted here as two independent events. That is the third caveat made concrete.

So *{M['window_counts'][0]}, {M['window_counts'][1]}, {M['observed']}* was never a fact about
California on its own. It was a fact about California **and a threshold**, and the threshold moved
the story further than the earthquakes did. That is the same lesson as the interval, applied to a
choice instead of a sample: when a conclusion rests on one round number, try the number either
side of it and report what you find.

Now back to the Bay for the assumption the bootstrap slipped past you. Resampling *months* treats each
month as an independent draw — as if the sea level in March told you nothing about April. It
plainly does. A wet winter, an El Niño, a warm year: those last longer than a month, so
neighbouring months carry much of the same information, and the record's months are worth rather
less than that many independent measurements.

The fix is to resample bigger pieces. Draw whole **calendar years** with replacement — all twelve
months of 1931 or none of them — so that whatever is shared inside a year travels with it. There
are {M['n_calendar_years']} years to draw from. `pd.concat` is the function that stacks the chosen
years back into one table.
""")

code(weekkit.CHECKPOINT.format(body=REBUILD_SLOPES))

code(f"""
np.random.seed(88)
by_year = {{}}
for y, one_year in sea.groupby("Year"):
    by_year[y] = one_year

block_slopes = []
for i in range({B}):
    picked = np.random.choice(sorted(by_year), size=len(by_year), replace=True)
    boot = pd.concat([by_year[y] for y in picked])
    block_slopes.append(LinearRegression().fit(boot[["year"]], boot["sea_mm"]).coef_[0])

block_slopes = np.array(block_slopes)
block_low, block_high = np.percentile(block_slopes, [2.5, 97.5])

print("resampling months:", round(ci_low, 2), "to", round(ci_high, 2), "mm/yr")
print("resampling years: ", round(block_low, 2), "to", round(block_high, 2), "mm/yr")

bins = np.arange(min(slopes.min(), block_slopes.min()),               # ONE set of bins for both,
                 max(slopes.max(), block_slopes.max()) + 0.01, 0.01)  # or the bar widths compare

plt.hist(slopes, bins=bins, label="months resampled")
plt.hist(block_slopes, bins=bins, alpha=0.6, label="whole years resampled")
plt.xlabel("bootstrap slope (mm per year)")
plt.ylabel("number of resamples")
plt.title(f"{{len(slopes):,}} resamples each, the same record")
plt.legend()
plt.show()
""")

md(f"""
The interval got **wider** — [{M['block_low']:.2f}, {M['block_high']:.2f}] against
[{M['ci_low']:.2f}, {M['ci_high']:.2f}], about {M['width_ratio']:.1f} times the width — and that is
the direction nobody guesses. Respecting the structure of your data does not sharpen the answer;
it stops you from claiming a sharpness you never had.

The month-by-month interval was too narrow because it counted every month as a fresh piece of
information. How many were there really? An interval {M['width_ratio']:.1f} times wider is
precisely the interval you would get from {M['n_effective']:.0f} independent months, so that is
what this record is worth. And note that {M['n_effective']:.0f} is *not*
{M['n_calendar_years']}: {M['n_calendar_years']} is how many blocks the loop drew, which is the
method, not the result.

Zero is still nowhere near either interval, so the conclusion about the Bay survives. That will
not always be true, and when it is not, the wider interval is the one to believe.
""")

# --- the closing -----------------------------------------------------------
md(f"""
{weekkit.CLOSING_HEADING}

**We cannot show it was wrong — and the reason we can now say that, rather than shrug, is that
the forecast finally has an interval attached.** Bootstrapping the catalogue puts the
Gutenberg-Richter rate at {M['rate']:.2f} M7+ earthquakes per {WINDOW_YEARS} years, 95% CI
[{M['rate_low']:.2f}, {M['rate_high']:.2f}], and five is far outside that. But a rate is not a
count: fold in the Poisson scatter that any real thirty-six years is subject to and five or more
happens in {100 * M['frac_5']:.1f}% of simulated worlds, sitting on the top edge of the interval
on the count rather than beyond it. Widen the window and the case for a broken model gets weaker still — {M['total_big']} large
earthquakes in {M['span_years']} years against {M['n_windows'] * M['rate']:.1f} expected, which
{100 * M['frac_total']:.0f}% of simulated worlds match or beat. The forecast was not convicted; it
was not acquitted either, and the honest report is the interval rather than the verdict. Written
that way it is still usable: the same rate says the chance of at least one M7+ in the box in the
next {AHEAD} years is {100 * M['p_class']:.0f}%, 95% CI [{100 * M['p_low']:.0f}%,
{100 * M['p_high']:.0f}%].
""")

md(weekkit.week_cheatsheet(8))

# --- homework --------------------------------------------------------------
md("""
## Homework

Three parts on the two datasets you already have loaded. Part 1 goes back to the Bay and asks
something class did not; parts 2 and 3 make you re-run the earthquake argument on a fitting range
you choose yourself, and say what that does to the verdict you committed to in class. If you have
restarted since class, run the setup cell at the top first, then the checkpoint just below.
""")

code(weekkit.CHECKPOINT.format(body=REBUILD_FORECAST))

ask(f"""
### ✏️ Your turn 7

In class the first and the last twenty-five years came out with intervals that do not overlap —
[{M['ends'][0]['low']:.2f}, {M['ends'][0]['high']:.2f}] against
[{M['ends'][1]['low']:.2f}, {M['ends'][1]['high']:.2f}] — which is how we concluded that the rise
had really changed. Those were the two extremes of five. Ask the same question of a cut that uses
every month: does the conclusion survive?

Split the record in half — {HALVES[0][0]} up to {HALVES[0][1]}, and {HALVES[1][0]} up to
{HALVES[1][1]} — and bootstrap each half separately, exactly as you bootstrapped the whole record
in your turn 2. Print each half's slope and 95% interval.

Then print whether the two intervals overlap. Two intervals overlap when the lower one's top end is
above the higher one's bottom end, so `first_high > second_low` is the test. Report what you get,
whichever way it comes out.

Resample **months**, as your turn 2 did, not whole years. Class showed that months give an
interval that is too narrow — and too narrow is the direction that makes two intervals *miss* each
other, so if these two overlap anyway, the wider honest version would only overlap more.

Then, as a comment at the end of your cell, answer in one sentence from your own two intervals:
does the conclusion that the rise really changed survive a cut that uses every month, or was it an
artefact of comparing the two extremes of five?

**Use these names**, because the self-check looks for them: `first`, `second`, `first_slope`,
`first_low`, `first_high`, `second_slope`, `second_low`, `second_high`.
""")

answer(f"""
first = sea[(sea["year"] >= {HALVES[0][0]}) & (sea["year"] < {HALVES[0][1]})]
second = sea[(sea["year"] >= {HALVES[1][0]}) & (sea["year"] < {HALVES[1][1]})]

np.random.seed(88)
first_slopes = []
for i in range({B}):
    boot = first.sample(len(first), replace=True)
    first_slopes.append(LinearRegression().fit(boot[["year"]], boot["sea_mm"]).coef_[0])
first_low, first_high = np.percentile(first_slopes, [2.5, 97.5])

np.random.seed(88)
second_slopes = []
for i in range({B}):
    boot = second.sample(len(second), replace=True)
    second_slopes.append(LinearRegression().fit(boot[["year"]], boot["sea_mm"]).coef_[0])
second_low, second_high = np.percentile(second_slopes, [2.5, 97.5])

first_slope = LinearRegression().fit(first[["year"]], first["sea_mm"]).coef_[0]
second_slope = LinearRegression().fit(second[["year"]], second["sea_mm"]).coef_[0]

print("{HALVES[0][0]}-{HALVES[0][1]}:", len(first), "months, slope", round(first_slope, 2),
      "95% interval", round(first_low, 2), "to", round(first_high, 2))
print("{HALVES[1][0]}-{HALVES[1][1]}:", len(second), "months, slope", round(second_slope, 2),
      "95% interval", round(second_low, 2), "to", round(second_high, 2))
print("do the intervals overlap?", first_high > second_low)

{note(f"My two halves overlap — {M['halves'][0]['low']:.2f} to "
       f"{M['halves'][0]['high']:.2f} against {M['halves'][1]['low']:.2f} to "
       f"{M['halves'][1]['high']:.2f} mm/yr — so a cut that uses every month does not on its "
       f"own show the rise changing, but that is not because either half is slow: cut my "
       f"second half on the same quarter-century grid class used and its pieces run "
       f"{', '.join(f'{s:.2f}' for s in M['halves'][1]['sub_slopes'])} mm/yr, "
       f"{M['halves'][1]['above']} of the {len(M['halves'][1]['sub_slopes'])} of them above the "
       f"{M['halves'][1]['slope']:.2f} the half itself fits at. A "
       f"{M['halves'][1]['hi'] - M['halves'][1]['lo']}-year least-squares line is not the "
       f"average of its pieces: those pieces rise "
       f"{M['halves'][1]['piece_rise']:.0f} mm between them while the half rises only "
       f"{M['halves'][1]['own_rise']:.0f} mm, because the record steps back down where one "
       f"piece ends and the next begins — the decade mean fell from "
       f"{M['decade_means'][1990]:.0f} mm in the 1990s to {M['decade_means'][2000]:.0f} mm in "
       f"the 2000s before reaching {M['decade_means'][2010]:.0f} mm in the 2010s. Nor does the "
       f"change class saw sit inside one half: it ran from the "
       f"{CHUNK_STARTS[0]}-{CHUNK_STARTS[0] + 25} chunk to the "
       f"{CHUNK_STARTS[-1]}-{CHUNK_STARTS[-1] + 25} chunk, which straddles the "
       f"{HALVES[0][1]} cut, so both halves get a share of it and neither can show it alone.")}
""", """
assert len(first) + len(second) == len(sea), "every month belongs to exactly one half"
assert first_low < first_slope < first_high, \\
    "the first half's interval should straddle the first half's own slope"
assert second_low < second_slope < second_high, \\
    "and the second half's should straddle the second half's"
print("\u2713 the two halves \u2014 first", round(first_low, 2), "to", round(first_high, 2),
      ", second", round(second_low, 2), "to", round(second_high, 2),
      "; overlapping:", first_high > second_low)
""")

ask(f"""
### ✏️ Your turn 8

Class fit Gutenberg-Richter between magnitude 4.0 and 5.5. That was a choice, and it was not the
only defensible one — too low and the catalogue is missing small events, too high and there are
barely any events left to fit.

**Pick one:** fit between {LO['lo']} and {LO['hi']}, or between {HI['lo']} and {HI['hi']}. Change
one line — `edges = np.arange(...).round(1)` — and re-run the whole argument on your choice:
`predicted_m7`, the {B:,} bootstrap rates, the Poisson draw, and the fraction of simulated worlds
reaching five. Redefining `edges` is enough: `predicted_m7` reads it, so nothing else needs editing.

Print your forecast, your 95% interval on the count, and your fraction. Then, as a comment at the
end of your cell, say in one sentence why moving the fitting range moves the forecast at all —
what changes about the line when you fit lower or higher.

**Use these names**, because the self-check looks for them: `edges`, `my_counts`, `my_fraction`.
""")

answer(f"""
edges = np.arange({LO['lo']}, {LO['hi'] + 0.1:.1f}, 0.1).round(1)   # the low choice

np.random.seed(88)
my_rates = []
for i in range({B}):
    boot_mags = quakes["mag"].sample(len(quakes), replace=True)
    my_rates.append(predicted_m7(boot_mags))
my_rates = np.array(my_rates)

np.random.seed(88)
my_counts = np.random.poisson(my_rates)
my_fraction = (my_counts >= 5).mean()

print("fitting magnitude", edges[0], "to", edges[-1])
print("forecast:", round(predicted_m7(quakes["mag"]), 2), "M7+ earthquakes")
print("95% interval on the count:", np.percentile(my_counts, [2.5, 97.5]))
print("fraction of simulated worlds with 5 or more:", round(my_fraction, 4))

{note("Moving the range moves the slope of the fitted line, because a different stretch of "
       "magnitudes gets to set it — and magnitude 7 lies well outside every one of those "
       "stretches, so a small change in slope is stretched into a sizeable change in the count "
       "the line reports at 7.")}
""", f"""
assert edges[0] != 4.0 or edges[-1] != 5.5, "change the range — this is class's fit, not a choice"
assert 0 < my_fraction < 0.2, \\
    "a small but nonzero share should reach five — zero means the Poisson draw is missing"
print("\u2713 your own fitting range \u2014 magnitude", edges[0], "to", edges[-1],
      ", 5 or more reached in", round(100 * my_fraction, 1), "% of simulated worlds")
""")

ask("""
### \u270f\ufe0f Your turn 9

Two or three sentences, quoting your own numbers from your turn 8.

In class you committed to a verdict \u2014 *wrong* or *unlucky* \u2014 using the fitting range class chose.
Say what your own range does to it: where five falls against your interval on the count, and
whether a reader of your notebook would come away thinking the forecast is broken or thinking the
last thirty-six years were busy. Then say what would have to be true for a single thirty-six-year
count to settle the question either way.
""")

answer_prose(f"""
Fitting between {LO['lo']} and {LO['hi']} gives a forecast of {LO['rate']:.2f} M7+ earthquakes, a
95% interval on the count topping out at {LO['count_high']}, and five or more in only
{100 * LO['frac_5']:.1f}% of simulated worlds \u2014 so on my choice five is *outside* the interval and
the forecast does look broken, where in class I called it unlucky. Class's range, 4.0 to 5.5, gave
{100 * M['frac_5']:.1f}%, with five sitting on the edge rather than beyond it, and the third
choice, {HI['lo']} to {HI['hi']}, has only {M['count_at_55']} events above magnitude 5.5 left to
fit, so its interval is wider still. The verdict therefore turns on a choice nobody can defend as
the only right one, and the answer moves more when I change my fitting range than the observation
moves it. For a single thirty-six-year count to settle the question, it would have to fall outside
the interval under *every* defensible fitting range \u2014 five does not \u2014 or the window would have to
be long enough that the Poisson scatter, which is roughly the square root of the expected count,
became small beside the gap being argued about.
""")



# ---------------------------------------------------------------------------
# 3. emit, execute, gate
# ---------------------------------------------------------------------------
def notebook(cells):
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4, "nbformat_minor": 5}


def cell(kind, source):
    c = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    sol = notebook([cell(k, s) for k, s, _ in CELLS])
    stu = notebook([cell(k, alt if alt is not None else s) for k, s, alt in CELLS])

    sol_path = OUT / f"{SLUG}_solution.ipynb"
    sol_path.write_text(json.dumps(sol, indent=1) + "\n")

    print(f"executing {sol_path.name} ...")
    r = weekkit.execute(sol_path, timeout=900)
    if r.returncode:
        print(r.stderr[-4000:])
        sys.exit("the solution did not execute")

    (OUT / f"{SLUG}.ipynb").write_text(json.dumps(stu, indent=1) + "\n")
    print(f"wrote {SLUG}.ipynb ({len(CELLS)} cells) and the executed solution")
    for name in (SEA_CACHE, EQ_CACHE, BIG_CACHE, BIG69_CACHE):
        read = datetime.date.fromtimestamp((ROOT / "data" / name).stat().st_mtime)
        owner = " (week 7's file — this week only reads it)" if name == EQ_CACHE else ""
        print(f"cache: data/{name}, downloaded {read}{owner}")
    print(f"catalogue: the query returns {M['n_rows']:,} rows and {M['n_quakes']:,} of them are "
          f"earthquakes. Dropped: {dict(TYPES.drop('earthquake'))}. The {M['n_nuclear']} nuclear "
          f"tests are at the Nevada Test Site, M{M['nuke_mag_lo']}-{M['nuke_mag_hi']}, "
          f"{M['nuke_first_year']}-{M['nuke_last_year']}, {M['nuke_in_range']} of them inside the "
          f"M{EDGES[0]}-{EDGES[-1]} fitting range. The drop is made in the NOTEBOOK, visibly, not "
          f"in the URL — same choice week 7 made, and week 10 gets these events as its subject. "
          f"The {M['n_big_in_slice']} M7+ events stay IN the counted catalogue: the query no "
          f"longer carries &maxmagnitude=6.9, which notes/dataset-audit/usgs-fdsn.md §4.2 forbids "
          f"by name. Forecast at M{EDGES[0]}-{EDGES[-1]}: truncated with blasts in (as shipped) "
          f"{M['rate_as_shipped']:.2f}, untruncated with blasts in {M['rate_blasts_in']:.2f}, "
          f"clean {M['rate']:.2f} — the two faults pushed in opposite directions, which is why "
          f"the shipped number looked plausible.")
    print(f"independence caveat: the claim that the three window counts {M['window_counts']} "
          f"'scatter more than a Poisson process allows' is GONE. Their dispersion is "
          f"{M['dispersion']:.3f}, and three Poisson counts with this mean are at least that "
          f"dispersed {100 * M['p_dispersion']:.0f}% of the time. The caveat now rests on the "
          f"{M['pair_69'][0]} / {M['pair_69'][1]} pair, {M['pair_69_minutes']} minutes apart, and "
          f"on the {M['shortest_gap']}-day {M['pair'][0]}-to-{M['pair'][1]} gap.")
    print(f"NOAA file: {M['n_months']:,} monthly means, {M['first_year']}-{M['last_year']}, "
          f"{M['n_missing']} months missing ({', '.join(M['missing_months'])})")
    print("prose: every count that NOAA can move is printed by a notebook cell at run time, not "
          "written into markdown — the setup cell's shape, the figure titles, and each "
          "self-check's closing print. The literals left are the ones the pinned 1900-2025 range "
          f"fixes ({M['n_years_sea']} years, {M['n_years_sea'] * 12:,} months) and the fitted "
          "results themselves, which are interpolated from this build and so are refreshed by "
          "rebuilding. Rebuild with --refresh before release.")
    drift = pinned_drift()
    if drift:
        print("\nWARNING: the data has moved away from course.yml's pinned expectations. The "
              "notebook prose has been rebuilt around the new numbers; course.yml has not.")
        for line in drift:
            print(f"  - {line}")
    else:
        print(f"pinned: course.yml's five expected values all reproduce within 1%, on "
              f"{M['pinned_months']:,} monthly means.")


if __name__ == "__main__":
    main()
    weekkit.gate(8)
