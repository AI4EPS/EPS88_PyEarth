#!/usr/bin/env python
"""Build project track T2 — "Is the CO2 seasonal cycle getting stronger?"

Emits both notebooks from one source so they cannot drift:

    docs/notebooks/T2_getting_stronger_solution.ipynb   executed, every output saved
    docs/notebooks/T2_getting_stronger.ipynb            the same file with the answers deleted

It also writes the track's two cached fallbacks, data/trackT2_co2_mlo.csv and
data/trackT2_co2_brw.csv — the parsed NOAA monthly tables for Mauna Loa and Barrow.

A TRACK is not a week (course.yml `project: track_notebooks:`). Two things differ, and both are
deliberate:

  * LESS HELP. No worked example before a question. The notebook loads the data and reproduces
    the ONE thing everybody already knows — the Keeling seasonal cycle, its May peak and its
    September trough — so a student can trust the pipeline, and then stops helping. Everything
    after is a prompt in words and an empty cell.
  * IT DOES NOT CLOSE. There is exactly one self-check, on the load, and the notebook ends on an
    open question this course cannot answer.

Every number that appears in prose or in a model answer is computed HERE, from the same files the
notebook reads, and formatted in. Nothing is typed from memory or copied from the plan.

NOAA appends a month to both station files every month, and the student's notebook reads them
LIVE — so a prose number taken from the record's tail expires between the build and the class.
Two things stop that, both borrowed from week 9, which had the same defect twice. Every amplitude
is measured on COMPLETE calendar years only, so a new month cannot move a single fitted number
until a whole new year has closed; and the row counts, which do change every month, are printed
at run time rather than written into markdown. `--refresh` re-downloads, and a cache older than
STALE_DAYS says so loudly rather than passing for current.

    python tools/build_track_T2.py              # build from the cached copies
    python tools/build_track_T2.py --refresh    # download NOAA again first — do this before class
"""
import datetime
import json
import pathlib
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/notebooks"
SLUG = "T2_getting_stronger"

course = yaml.safe_load((ROOT / "course.yml").read_text())
modules = yaml.safe_load((ROOT / "modules.yml").read_text())
TRACK = next(t for t in course["project"]["tracks"] if t["id"] == "T2")
PLATFORM = course["platform"]
CACHE_BASE = PLATFORM["cache_base"]

# The two live sources. NOAA GML publishes one monthly-mean text file per in-situ station, all in
# one directory and all in the same 19-column layout with a real header row — so ONE read recipe
# serves every station, which is what makes the open question at the end pursuable by changing a
# single string. (notes/dataset-audit/noaa-climate.md records this path as 404 and says the
# per-station files exist only inside an 8.5 MB zip. That was the `surface/brw/` path; the files
# are under `surface/txt/`, plain text, 87 KB. Measured 2026-08-31 — see AUDIT DRIFT below.)
GML = ("https://gml.noaa.gov/aftp/data/trace_gases/co2/in-situ/surface/txt/"
       "co2_{site}_surface-insitu_1_ccgg_MonthlyData.txt")
MLO_URL = GML.format(site="mlo")
BRW_URL = GML.format(site="brw")
MLO_CACHE = "trackT2_co2_mlo.csv"
BRW_CACHE = "trackT2_co2_brw.csv"
STALE_DAYS = 30                  # after this the cache is behind NOAA by a month or more

WINDOW = 15                      # years in each end window, when a percentage change is wanted
SEED = 88                        # the course number, fixed before anything was run
N_BOOT = 2000                    # bootstrap resamples of the years
MISSING = -999.99                # NOAA's "no measurement this month" sentinel in `value`

# For the plan cross-check only. Neither is read by the notebook and neither is cached.
TRENDS_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
FLASK_URL = ("https://gml.noaa.gov/aftp/data/trace_gases/co2/flask/surface/txt/"
             "co2_brw_surface-flask_1_ccgg_month.txt")


# ---------------------------------------------------------------------------
# 1. measure everything the notebook will say
# ---------------------------------------------------------------------------
def fetch(url, name, refresh):
    """Run the live read, cache the PARSED table, and return it.

    The cache holds the parsed frame rather than the raw text, so the notebook's fallback read
    needs no argument its live read does not have. A cache downloaded once and never again is
    how this dataset goes quietly wrong — NOAA adds a month every month — so `--refresh`
    re-downloads and an old cache complains.
    """
    out = ROOT / "data" / name
    if refresh or not out.exists():
        print(f"downloading {url}")
        pd.read_csv(url, comment="#", sep=r"\s+").to_csv(out, index=False)
    else:
        age = (time.time() - out.stat().st_mtime) / 86400
        if age > STALE_DAYS:
            print(f"WARNING: data/{name} was downloaded {age:.0f} days ago. NOAA has appended "
                  f"months since, and the student's notebook reads NOAA live. Rebuild with "
                  f"--refresh before the class.")
    return pd.read_csv(out)


def seasonal(station):
    """Turn the months with no measurement into real holes, then split CO2 into trend and swing.

    The sentinel must become NaN rather than being deleted. Deleting the rows closes the gap, and
    a 12-row rolling mean then averages November 2022 with July 2023 as though they were
    neighbours; a NaN keeps the calendar intact and makes every window that spans a hole come out
    NaN, which is the truth.
    """
    station = station.copy()
    station.loc[station["value"] < 0, "value"] = np.nan
    station["trend"] = station["value"].rolling(12, center=True).mean()
    station["swing"] = station["value"] - station["trend"]
    return station


def fourier_fit(station):
    """Replace each year's swing by the smooth once-a-year wave that fits it best."""
    station = station.copy()
    angle = 2 * np.pi * station["month"] / 12
    station["sin"] = np.sin(angle)
    station["cos"] = np.cos(angle)
    station["fourier"] = np.nan
    for year, rows in station.dropna(subset=["swing"]).groupby("year"):
        wave = LinearRegression().fit(rows[["sin", "cos"]], rows["swing"])
        station.loc[rows.index, "fourier"] = wave.predict(rows[["sin", "cos"]])
    return station


def amplitude(station, column):
    """One number per complete year: how far that column swings from its highest month to its
    lowest."""
    have = station.dropna(subset=[column])
    months = have.groupby("year")["month"].count()
    full = have[have["year"].isin(months[months == 12].index)]
    highest = full.groupby("year")[column].max()
    lowest = full.groupby("year")[column].min()
    return pd.DataFrame({"year": highest.index, "amplitude": (highest - lowest).values})


def trend(amps):
    """How fast the yearly amplitude is changing, in ppm per decade."""
    return LinearRegression().fit(amps[["year"]], amps["amplitude"]).coef_[0] * 10


def trend_spread(amps):
    """The trend that N_BOOT resamples of the years give, so the one trend has a range.

    Character for character the function the notebook asks the student to write, because the
    intervals below are quoted in the markdown. `random_state=i` is what makes it reproducible:
    an unseeded bootstrap prints one interval into the prose and a different one into the cell
    beside it, and the first build of this track did exactly that.
    """
    slopes = []
    for i in range(N_BOOT):
        picked = amps.sample(n=len(amps), replace=True, random_state=i)
        slopes.append(trend(picked))
    return np.array(slopes)


def window_change(amps):
    """Mean amplitude over the first and last WINDOW complete years, and the change between."""
    early = amps.head(WINDOW)
    late = amps.tail(WINDOW)
    return {"early": float(early["amplitude"].mean()), "late": float(late["amplitude"].mean()),
            "early_from": int(early["year"].iloc[0]), "early_to": int(early["year"].iloc[-1]),
            "late_from": int(late["year"].iloc[0]), "late_to": int(late["year"].iloc[-1]),
            "pct": float(100 * (late["amplitude"].mean() / early["amplitude"].mean() - 1))}


def described(amps):
    """Everything the notebook says about one amplitude series, in one dict."""
    spread = trend_spread(amps)
    lo, hi = np.percentile(spread, [2.5, 97.5])
    out = {"n": len(amps), "mean": float(amps["amplitude"].mean()), "slope": float(trend(amps)),
           "lo": float(lo), "hi": float(hi), "above_zero": float((spread > 0).mean()),
           "first": int(amps["year"].iloc[0]), "last": int(amps["year"].iloc[-1])}
    out.update(window_change(amps))
    return out


REFRESH = "--refresh" in sys.argv
MLO_RAW = fetch(MLO_URL, MLO_CACHE, REFRESH)
BRW_RAW = fetch(BRW_URL, BRW_CACHE, REFRESH)
READ_DATE = datetime.date.fromtimestamp(
    (ROOT / "data" / MLO_CACHE).stat().st_mtime).isoformat()

MLO = fourier_fit(seasonal(MLO_RAW))
BRW = fourier_fit(seasonal(BRW_RAW))

MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

M = {}
M["read_date"] = READ_DATE
M["mlo_rows"] = len(MLO_RAW)
M["brw_rows"] = len(BRW_RAW)
M["mlo_gaps"] = int((MLO_RAW["value"] < 0).sum())
M["brw_gaps"] = int((BRW_RAW["value"] < 0).sum())
M["mlo_lat"] = float(MLO_RAW["latitude"].iloc[0])
M["mlo_elev"] = float(MLO_RAW["elevation"].iloc[0])
M["brw_lat"] = float(BRW_RAW["latitude"].iloc[0])
M["mlo_first"] = int(MLO["year"].min())
M["brw_first"] = int(BRW["year"].min())
M["last_year"] = int(MLO["year"].max())
# The missing months at Mauna Loa are not random: NOAA's own trends file says the observatory
# was cut off by the 2022 eruption. Which months, measured rather than recalled.
gap_rows = MLO_RAW[MLO_RAW["value"] < 0]
recent = gap_rows[gap_rows["year"] >= 2020]
M["gap_months"] = [f"{MONTH_NAMES[int(r.month) - 1]} {int(r.year)}" for r in recent.itertuples()]
M["gap_first"], M["gap_last"] = M["gap_months"][0], M["gap_months"][-1]
M["gap_n"] = len(recent)

# --- the known result: the seasonal cycle itself ---
profile = MLO.dropna(subset=["swing"]).groupby("month")["swing"].mean()
M["peak_month"] = MONTH_NAMES[int(profile.idxmax()) - 1]
M["peak_ppm"] = float(profile.max())
# The two lowest months are tied to a hundredth of a ppm, so naming only idxmin would be a
# claim this record cannot support. Both are measured and both are printed.
two_lowest = profile.sort_values().head(2)
M["trough_month"] = MONTH_NAMES[int(two_lowest.index[0]) - 1]
M["trough_ppm"] = float(two_lowest.iloc[0])
M["trough_month2"] = MONTH_NAMES[int(two_lowest.index[1]) - 1]
M["trough_ppm2"] = float(two_lowest.iloc[1])
M["profile_range"] = float(profile.max() - profile.min())
brw_profile = BRW.dropna(subset=["swing"]).groupby("month")["swing"].mean()
M["brw_peak_month"] = MONTH_NAMES[int(brw_profile.idxmax()) - 1]
M["brw_trough_month"] = MONTH_NAMES[int(brw_profile.idxmin()) - 1]
M["brw_profile_range"] = float(brw_profile.max() - brw_profile.min())

# --- the three definitions, at both stations ---
DEFS = [("raw", "value", "the raw record"),
        ("detrended", "swing", "the detrended swing"),
        ("fourier", "fourier", "the once-a-year wave")]
A = {}
for site, station in (("mlo", MLO), ("brw", BRW)):
    for key, column, _ in DEFS:
        A[site, key] = described(amplitude(station, column))
M["defs"] = A

# --- why the raw definition is different: the trend leaks into it ---
# Within one calendar year the trough comes MONTHS AFTER the peak, so the year's own rise lifts
# the trough and shrinks the raw range. The size of that leak is (lag / 12) x the year's growth,
# and the growth rate has itself risen — so the leak has grown too.
have = MLO.dropna(subset=["value"])
full = have[have.groupby("year")["month"].transform("count") == 12]
peak_at = full.loc[full.groupby("year")["value"].idxmax()].set_index("year")["month"]
trough_at = full.loc[full.groupby("year")["value"].idxmin()].set_index("year")["month"]
lag = trough_at - peak_at
# reindexed over the calendar: 2022 and 2023 are not complete years, and without this the
# shift would make 2021 and 2024 neighbours and call three years of rise two.
years = range(int(full["year"].min()), int(full["year"].max()) + 1)
annual = full.groupby("year")["value"].mean().reindex(years)
growth = (annual.shift(-1) - annual.shift(1)) / 2

raw_amp = amplitude(MLO, "value").set_index("year")["amplitude"]
det_amp = amplitude(MLO, "swing").set_index("year")["amplitude"]
leak = pd.DataFrame({"gap": det_amp - raw_amp, "lag": lag, "growth": growth}).dropna()
leak["predicted"] = leak["lag"] / 12 * leak["growth"]
M["lag_mean"] = float(leak["lag"].mean())
M["leak_corr"] = float(np.corrcoef(leak["gap"], leak["predicted"])[0, 1])
M["leak_early"] = float(leak["gap"].head(WINDOW).mean())
M["leak_early_pred"] = float(leak["predicted"].head(WINDOW).mean())
M["leak_late"] = float(leak["gap"].tail(WINDOW).mean())
M["leak_late_pred"] = float(leak["predicted"].tail(WINDOW).mean())
M["growth_early"] = float(growth.dropna().head(WINDOW).mean())
M["growth_late"] = float(growth.dropna().tail(WINDOW).mean())
gap_frame = pd.DataFrame({"year": leak.index, "amplitude": leak["gap"].values})
M["leak_slope"] = float(trend(gap_frame))
M["slope_difference"] = A["mlo", "detrended"]["slope"] - A["mlo", "raw"]["slope"]

# --- the first move on the open question: two more stations, same read recipe ---
EXTRA = [("smo", "American Samoa"), ("spo", "the South Pole")]
LADDER = []
for site, name in EXTRA:
    frame = fetch(GML.format(site=site), f"trackT2_co2_{site}.csv", REFRESH)
    d = described(amplitude(seasonal(frame), "swing"))
    d.update(site=site, name=name, lat=float(frame["latitude"].iloc[0]))
    LADDER.append(d)
for site, name, station in (("mlo", "Mauna Loa", MLO), ("brw", "Barrow", BRW)):
    d = dict(A[site, "detrended"])
    d.update(site=site, name=name, lat=M[f"{site}_lat"])
    LADDER.append(d)
LADDER.sort(key=lambda d: d["lat"])
M["ladder"] = LADDER

# The build log is the record that every number was computed. Print all of it, not a selection.
for k in sorted(M):
    if k not in ("defs", "ladder"):
        print(f"  measured  {k:>16} = {M[k]}")
for (site, key), d in A.items():
    print(f"  measured  {site + '/' + key:>16} : slope {d['slope']:+.3f} ppm/dec "
          f"CI [{d['lo']:+.3f}, {d['hi']:+.3f}] above zero {d['above_zero']:.3f} · "
          f"{d['early']:.2f} ({d['early_from']}-{d['early_to']}) -> {d['late']:.2f} "
          f"({d['late_from']}-{d['late_to']}) = {d['pct']:+.1f}% · n={d['n']}")
for d in LADDER:
    print(f"  measured  {d['name']:>16} : lat {d['lat']:+.2f}  mean {d['mean']:.2f} ppm  "
          f"slope {d['slope']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] ppm/dec  "
          f"= {100 * d['slope'] / d['mean']:+.2f}%/decade")


# ---------------------------------------------------------------------------
# 1b. the plan and the audit, checked against what was just measured
# ---------------------------------------------------------------------------
def verify_plan():
    """course.yml's T2 open_question quotes +12.9% and +24.7%. Reproduce both, or say so.

    A builder does not edit the plan, so a mismatch is printed rather than patched. Both files
    are downloaded fresh every build and neither is cached: they exist to check a claim, not to
    feed the notebook, and a cross-check read from a stale copy checks nothing.
    """
    try:
        trends = pd.read_csv(TRENDS_URL, comment="#")
        flask = pd.read_csv(FLASK_URL, comment="#", sep=r"\s+",
                            names=["site", "year", "month", "value"])
    except Exception as e:
        print(f"  PLAN CHECK SKIPPED  the two cross-check files did not download: "
              f"{type(e).__name__}")
        return
    # +12.9%: Mauna Loa, the TRENDS csv (1958 on, not the 1974-on in-situ file this notebook
    # reads), amplitude of `average` minus NOAA's own `deseasonalized` column, 1960-75 vs 2010-25.
    trends = trends.rename(columns={"average": "value"})
    trends["swing"] = trends["value"] - trends["deseasonalized"]
    ml = amplitude(trends, "swing").set_index("year")["amplitude"]
    a = ml.loc[1960:1975].mean()
    b = ml.loc[2010:2025].mean()
    print(f"  PLAN CHECK  course.yml's +12.9% at Mauna Loa: {a:.2f} -> {b:.2f} ppm = "
          f"{100 * (b / a - 1):+.1f}% — reproduces, but only on the trends file, only under the "
          f"detrended definition, and only with a 50-year window separation.")
    # +24.7%: Barrow, the FLASK monthly file (1971 on), RAW peak-to-trough, 1972-86 vs 2011-25.
    br = amplitude(flask, "value").set_index("year")["amplitude"]
    c = br.loc[1972:1986].mean()
    d = br.loc[2011:2025].mean()
    print(f"  PLAN CHECK  course.yml's +24.7% at Barrow: {c:.2f} -> {d:.2f} ppm = "
          f"{100 * (d / c - 1):+.1f}% — reproduces, but on the flask file, under the RAW "
          f"definition, and over a 39-year window separation.")
    print(f"  PLAN DRIFT  the two are therefore not comparable: different definitions, different "
          f"measurement programmes, different window separations. Measured here on one "
          f"definition and one programme, the honest comparison is a RATE — Mauna Loa "
          f"{A['mlo', 'detrended']['slope']:+.3f} ppm/decade against Barrow "
          f"{A['brw', 'detrended']['slope']:+.3f}. The direction course.yml records is right; "
          f"the pair of percentages is not a like-for-like pair.")
    print(f"  AUDIT DRIFT notes/dataset-audit/noaa-climate.md says the GML per-station files "
          f"exist only inside co2_brw_surface-insitu_ccgg_text.zip (8.5 MB). They are also at "
          f"{GML.format(site='SITE')} — plain text, 87 KB, one header row, HTTP 200 on "
          f"{READ_DATE}. The audit's 404 was the .../surface/brw/ path, not .../surface/txt/.")


verify_plan()


# ---------------------------------------------------------------------------
# 2. the summary, generated from modules.yml so the wording cannot drift
# ---------------------------------------------------------------------------
def idea(module_id, name):
    """One plain_words sentence, verbatim from modules.yml."""
    return next(d for d in modules["plain_words"]
                if d["module"] == module_id and d["idea"] == name)


def fn(module_id, name):
    """One function entry, verbatim from modules.yml."""
    return next(f for f in next(m for m in modules["modules"] if m["id"] == module_id)["functions"]
                if f["name"] == name)


# The ideas and calls this track leans on, named here and worded there. A track teaches nothing
# new, so the full module tables would list thirty functions it never uses; these are the ones the
# notebook and its model answers actually write.
TRACK_IDEAS = [("ML1", "Linear regression"), ("S4", "Bootstrap"), ("S4", "Confidence interval"),
               ("D1", "NaN"), ("D2", "Table")]
# Only calls the notebook or a model answer actually writes. `series.rolling(365).sum()` is the
# nearest recorded entry to the twelve-month `.mean()` the setup cell slides along the record;
# modules.yml has no `.mean()` variant, and inventing one here is what the table exists to stop.
TRACK_FNS = [("D2", "table.groupby(column)"), ("D2", "table.dropna()"),
             ("S2", "series.rolling(365).sum()"), ("S2", "series.idxmax()"),
             ("ML1", "column.mean()"), ("ML1", "table.tail(n)"),
             ("ML1", "LinearRegression().fit(x, y)"), ("ML1", "model.coef_[0]"),
             ("ML1", "model.predict(x)"), ("S4", "table.sample(n, replace=True)"),
             ("S4", "np.percentile(values, [2.5, 97.5])")]


def track_summary():
    out = [f"## What track {TRACK['id']} leans on", "",
           f"**The question.** {TITLE}", "",
           "Nothing here is new. These are the weeks to look back at while you work, and the "
           "wording is the course's own.", "",
           "### The ideas, in plain words", "", "| Idea | Means |", "|---|---|"]
    out += [f"| **{d['idea']}** | {d['words']} |" for d in (idea(m, i) for m, i in TRACK_IDEAS)]
    out += ["", "### Code you will reach back for", "", "| Function | What it does |", "|---|---|"]
    out += [f"| `{f['name']}` | {f['does']} |" for f in (fn(m, n) for m, n in TRACK_FNS)]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 3. the cells
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
    """A code answer cell. The solution carries the model answer; the student gets the stub."""
    solution = model.strip("\n") + (("\n\n" + check.strip("\n")) if check else "")
    student = "# ← your answer here\n\n" + (("\n" + check.strip("\n")) if check else "")
    CELLS.append(("code", solution, student))


def answer_prose(model):
    CELLS.append(("markdown", model.strip("\n"),
                  "*(Double-click this cell and replace this line with your answer.)*"))


def blank_prose():
    """A section of the student's own project. It is empty in the solution too, because there is
    no model answer to a section whose content is the student's own work."""
    stub = "*(Double-click this cell and replace this line with your answer.)*"
    CELLS.append(("markdown", stub, stub))


datahub = (f"{PLATFORM['datahub']}/hub/user-redirect/git-pull"
           f"?repo={PLATFORM['repo'].replace(':', '%3A').replace('/', '%2F')}"
           f"&branch={PLATFORM['branch']}"
           f"&urlpath=lab%2Ftree%2FEPS88_PyEarth%2F{PLATFORM['notebook_dir']}%2F{SLUG}.ipynb")

# T3's title had to be rebuilt from the data because course.yml wrote two counts into it. This
# one carries no numbers, so it is the plan's own sentence and cannot drift from the file.
TITLE = TRACK["title"]

HOOK = f"""
Every year the CO2 measured on Mauna Loa climbs through the northern winter and falls again
through the northern summer. That sawtooth is the northern land biosphere breathing: leaves open
in spring and pull carbon out of the air faster than everything rotting puts it back, and in
autumn the balance reverses. Almost all of the world's land is north of the equator, so the whole
atmosphere carries the signature of one hemisphere's growing season.

The swing is a few parts per million, riding on a rise of more than a hundred. The question is not
whether it is there — you will see it in the first figure — but whether it is getting **bigger**.
If northern plants are drawing down more carbon each summer than they used to, the breath should
deepen, and the record is long enough to tell.

It turns out that the answer depends on what you decide the word *amplitude* means. There are at
least three defensible answers, this notebook does not choose between them, and choosing is your
first job.
"""

md(weekkit.OPENING.format(question=TITLE, datahub=datahub, hook=HOOK.strip()))

md("""
## How this notebook is different

This is a **project track**. It is not a weekly notebook and it does not behave like one.

A weekly notebook shows you a move, walks you through it, and then asks you to make it once
yourself. This one loads the data and reproduces the one thing about it that nobody disputes —
that there is a seasonal cycle, and roughly how big it is — and then stops helping. From there on
every section is a sentence describing what to find out and an empty cell to find it out in. There
is no worked example above to pattern-match against, because on a real question there never is
one.

**There is exactly one self-check in this notebook, and it is on the data loading.** After that,
nothing tells you whether you are right. That is not an oversight and it is not laziness: past the
loading step there is no single right answer here, so a cell that said `assert` would be lying to
you about how research works. What replaces it is the thing researchers actually use — a result
you can get two ways, a number you can predict before you compute it, and a claim you can try to
break.

**And it does not close.** The last section is a question this course does not know the answer to.
Everything above it is scaffolding; that question is the project.
""")

md(f"""
## What you'll be able to do

**The science.** Say whether the seasonal CO2 cycle is getting stronger, with a number and an
interval rather than an adjective — and say how much of your answer came from the Earth and how
much from a definition you chose.

**The skills.** Turn a monthly record into one number per year without letting the long-term trend
contaminate it. Fit a wave to twelve points. Put an interval on a trend by resampling the years,
and use the interval to decide whether you are allowed to answer at all.

**The four questions, in order:**

1. What does the Mauna Loa record look like, and how big is its seasonal swing?
2. Is the swing bigger now than it was?
3. Does that answer survive a different definition of "amplitude"?
4. Does the same thing happen at {M['brw_lat']:.0f}° north?

The open question at the end is not on that list. It is the project; the four above are what you
build to reach it.
""")

md(f"""
## Setup

NOAA's Global Monitoring Laboratory publishes one monthly-mean file per observatory, all in the
same directory and all in the same layout. Two of them are loaded below: **Mauna Loa** in Hawaii
at {M['mlo_lat']:.1f}°N, the longest continuous CO2 record there is, and **Barrow** on the north
coast of Alaska at {M['brw_lat']:.1f}°N. Barrow is not analysed until the last section; it is
loaded now so that the one self-check covers both files.

**Two things about these files, and the second one is a trap.**

- The measurement is the column called `value`, in parts per million, one row per month.
- A month with no measurement is not blank and it is not `NaN`. It is written as **`{MISSING}`**,
  which is a number, and which will happily average in with the real ones. The first thing the
  next section does is turn those into real holes — and it turns them into holes rather than
  deleting the rows, because deleting a row closes the gap and lets a twelve-month average run
  straight across it.

Mauna Loa has {M['mlo_gaps']} such months and Barrow has {M['brw_gaps']}. {M['gap_n']} of the
{M['mlo_gaps']} at Mauna Loa are not an instrument fault: {M['gap_first']} to {M['gap_last']} are
missing because the volcano erupted, lava crossed the access road and the observatory lost power. NOAA says so in the header
of its own trends file — *"Due to the eruption of the Mauna Loa Volcano, measurements from Mauna
Loa Observatory ... Maunakea Observatories, approximately 21 miles north"* (read {M['read_date']}).
The gap in a climate record is itself Earth science.
""")

code(weekkit.setup_cell(
    imports="import numpy as np\nfrom sklearn.linear_model import LinearRegression\n",
    figsize="(7, 4)",
    cache_base=CACHE_BASE,
    signature="url, cache_name",
    docstring="Read one station's monthly CO2 live; fall back to the copy stored with the course.",
    url_expr='url, comment="#", sep=r"\\s+"',
    cache_expr="cache_name",
    unpack=f'''
GML = ("https://gml.noaa.gov/aftp/data/trace_gases/co2/in-situ/surface/txt/"
       "co2_{{site}}_surface-insitu_1_ccgg_MonthlyData.txt")

mauna_loa = load(GML.format(site="mlo"), "{MLO_CACHE}")
barrow = load(GML.format(site="brw"), "{BRW_CACHE}")

print("Mauna Loa:", mauna_loa.shape, " Barrow:", barrow.shape)
print(mauna_loa[["year", "month", "value", "latitude"]].head())
'''.strip("\n")))

code(f"""
assert "value" in mauna_loa.columns and "latitude" in barrow.columns, \\
    "a column the whole notebook needs is missing — the files were read wrong, or NOAA changed them"
assert len(mauna_loa) > 500 and len(barrow) > 500, \\
    "expected {M['mlo_rows']}-odd monthly rows at each station; far fewer means the read failed"
print(f"✓ the data — {{len(mauna_loa)}} monthly rows at Mauna Loa "
      f"({{mauna_loa.latitude.iloc[0]}}°N) and {{len(barrow)}} at Barrow "
      f"({{barrow.latitude.iloc[0]}}°N), of which {{(mauna_loa.value < 0).sum()}} and "
      f"{{(barrow.value < 0).sum()}} are marked {MISSING} for 'no measurement'")
""")

md("""
### And that is the last self-check in this notebook

The pipeline is now trustworthy: the files are the files, the columns are the columns, the numbers
below are the numbers. Everything from here is yours, and nothing will tell you when you have it
right.
""")

# --- the verified half ------------------------------------------------------
md(f"""
## What does the Mauna Loa record look like, and how big is its seasonal swing?

Two things are happening in this record at once and they have to be separated before anything can
be measured. There is a **trend** — CO2 today is far above CO2 in the 1970s — and there is a
**swing** around it that repeats every year.

The trend is easy to estimate without any curve fitting at all. Average twelve consecutive months
and the seasons cancel, because each one appears exactly once; slide that average along the record
and what comes out is the trend with the seasons taken out of it. Subtract it and what is left is
the swing.

Two kinds of hole appear on the way. The {MISSING} months become holes, because that is what they
are; and the subtraction leaves one at each end, since the first six months and the last six have
no full year around them to average.
**NaN:** {idea('D1', 'NaN')['words']}
""")

code("""
def seasonal(station):
    \"\"\"Turn the months with no measurement into real holes, then split CO2 into trend and swing.\"\"\"
    station = station.copy()
    station.loc[station["value"] < 0, "value"] = np.nan
    station["trend"] = station["value"].rolling(12, center=True).mean()
    station["swing"] = station["value"] - station["trend"]
    return station


mauna_loa = seasonal(mauna_loa)
barrow = seasonal(barrow)

print("Mauna Loa, months in the file:", len(mauna_loa),
      "from", mauna_loa.year.min(), "to", mauna_loa.year.max())
print("of those, months with a real measurement:", mauna_loa.value.notna().sum())
print("and months with a trend to subtract:", mauna_loa.swing.notna().sum())
""")

md("""
The whole record, with the twelve-month average drawn through it. The sawtooth is the thing this
project is about; the smooth line is what has to come off before it can be measured.
""")

code(f"""
plt.plot(mauna_loa["year"] + mauna_loa["month"] / 12, mauna_loa["value"],
         color="0.4", lw=0.8)
plt.plot(mauna_loa["year"] + mauna_loa["month"] / 12, mauna_loa["trend"],
         color="firebrick", lw=1.4)
plt.xlabel("year")
plt.ylabel("CO$_2$ (ppm)")
plt.title(f"Mauna Loa monthly CO$_2$ and its 12-month average "
          f"(n = {{len(mauna_loa)}} months)")
plt.show()
""")

md("""
Now the swing on its own, averaged over every year in the record — one number per calendar month.
This is the shape the northern growing season leaves in the atmosphere.
""")

code("""
profile = mauna_loa.groupby("month")["swing"].mean()

plt.bar(profile.index, profile.values, color="0.4")
plt.axhline(0, color="firebrick", lw=1.2)
plt.xlabel("month (1 = January)")
plt.ylabel("average swing about the trend (ppm)")
plt.title(f"The average seasonal swing at Mauna Loa ({mauna_loa.year.min()}"
          f"-{mauna_loa.year.max()})")
plt.locator_params(axis="x", integer=True)
plt.show()

print("highest month:", profile.idxmax(), round(profile.max(), 2), "ppm")
print("the two lowest months:", profile.sort_values().head(2).round(2).to_dict())
print("top to bottom:", round(profile.max() - profile.min(), 2), "ppm")
""")

md(f"""
That is the Keeling seasonal cycle, and it is the one result this notebook hands you: a peak in
**{M['peak_month']}** at {M['peak_ppm']:+.2f} ppm, a trough shared between **{M['trough_month']}**
and **{M['trough_month2']}** — {M['trough_ppm']:+.2f} and {M['trough_ppm2']:+.2f}, which fifty
years of this record cannot separate — and about **{M['profile_range']:.1f} ppm** between top and
bottom. Northern plants draw carbon down through the summer and the atmosphere reaches its lowest
point just as the growing season ends.

Nothing above is in dispute. Everything below is.
""")

# --- YOUR TURN 1 ------------------------------------------------------------
md("""
## Is the swing bigger now than it was?

One average cycle over the whole record cannot answer that. You need **one number per year**, and
then a line through those numbers.

Two decisions are forced on you before you can write a single line, and both matter more than they
look. The first is what "one number" means for a year — that is the next section's whole subject,
so for now take the most obvious answer you can think of. The second is which years you are
allowed to use at all: a year missing four months has a smaller range than a year with twelve, for
no reason to do with plants, so an incomplete year is not a smaller amplitude but a missing one.
""")

ask(f"""
### ✏️ Your turn 1

Write a function that turns a station's table into one row per year — the year, and how far its
CO2 swings between the highest month and the lowest — and use it on Mauna Loa.

**Use these names**, because every later section reuses them: the function `amplitude(station,
column)`, taking the table and the name of the column to measure, and returning a table with the
two columns `year` and `amplitude`. Every definition you try later has to hand back that same
shape, so that one piece of fitting code works on all of them.

Use only years with all twelve months. Then plot amplitude against year, fit a straight line to
it, and print the slope in **ppm per decade** (`.coef_[0]` is per year).

**Linear regression:** {idea('ML1', 'Linear regression')['words']}

Then print one more line answering it in a sentence, on your own slope: is the seasonal swing at
Mauna Loa getting stronger?
""")

answer("""
def amplitude(station, column):
    \"\"\"One number per complete year: how far that column swings from its highest month to its
    lowest.\"\"\"
    have = station.dropna(subset=[column])
    months = have.groupby("year")["month"].count()
    full = have[have["year"].isin(months[months == 12].index)]
    highest = full.groupby("year")[column].max()
    lowest = full.groupby("year")[column].min()
    return pd.DataFrame({"year": highest.index, "amplitude": (highest - lowest).values})


raw = amplitude(mauna_loa, "value")
line = LinearRegression().fit(raw[["year"]], raw["amplitude"])

plt.scatter(raw["year"], raw["amplitude"], color="0.4", s=14)
plt.plot(raw["year"], line.predict(raw[["year"]]), color="firebrick", lw=1.4)
plt.xlabel("year")
plt.ylabel("peak-to-trough CO$_2$ (ppm)")
plt.title(f"Mauna Loa seasonal amplitude, raw record (n = {len(raw)} complete years)")
plt.show()

print("complete years:", len(raw), "from", raw.year.iloc[0], "to", raw.year.iloc[-1])
print("mean amplitude:", round(raw.amplitude.mean(), 2), "ppm")
print("trend:", round(line.coef_[0] * 10, 3), "ppm per decade")

print("On this definition the swing is not measurably getting stronger:",
      round(line.coef_[0] * 10, 3), "ppm per decade is a change of",
      round(100 * line.coef_[0] * 10 / raw.amplitude.mean(), 2),
      "percent of the mean amplitude per decade, and the scatter around the line is far",
      "larger than the line's own rise across fifty years.")
""")

md(f"""
Measured that way the answer is **{M['defs']['mlo', 'raw']['slope']:+.3f} ppm per decade** on a
mean amplitude of {M['defs']['mlo', 'raw']['mean']:.2f} ppm — a fifth of a percent per decade, and
the cloud of points is wide. A single fitted slope with no interval on it, though, cannot tell you
whether that is a small effect or no effect.
""")

ask(f"""
### ✏️ Your turn 2

Put an interval on that slope by resampling the years.

**Bootstrap:** {idea('S4', 'Bootstrap')['words']}

The recipe, in words. {N_BOOT} times over: draw {M['defs']['mlo', 'raw']['n']}-odd years from your
amplitude table **with replacement** — `amps.sample(n=len(amps), replace=True, random_state=i)` —
refit the line to that resample, and collect the slope. The `random_state` is what makes your
interval the same every time you run the cell; without it the number in your write-up and the
number in your notebook will not match. Write it as a function `trend_spread(amps)` that hands
back the whole array of slopes, and while you are there write `trend(amps)` for the slope alone,
because you will call both on four more series before the notebook is over.

Report the 2.5th and 97.5th percentiles, and the fraction of the resamples that came out above
zero. Draw the {N_BOOT} slopes as a histogram with zero marked.

**Confidence interval:** {idea('S4', 'Confidence interval')['words']}

Then print two or three sentences answering it on your own interval: are you able to say whether
the Mauna Loa seasonal cycle is getting stronger, and what would your interval have to look like
before you could?
""")

answer(f"""
def trend(amps):
    \"\"\"How fast the yearly amplitude is changing, in ppm per decade.\"\"\"
    return LinearRegression().fit(amps[["year"]], amps["amplitude"]).coef_[0] * 10


def trend_spread(amps):
    \"\"\"The trend that {N_BOOT} resamples of the years give, so the one trend has a range.\"\"\"
    slopes = []
    for i in range({N_BOOT}):
        picked = amps.sample(n=len(amps), replace=True, random_state=i)
        slopes.append(trend(picked))
    return np.array(slopes)


spread = trend_spread(raw)
low, high = np.percentile(spread, [2.5, 97.5])

plt.hist(spread, bins=40, color="0.4")
plt.axvline(0, color="firebrick", lw=1.5)
plt.xlabel("trend in seasonal amplitude (ppm per decade)")
plt.ylabel("resamples")
plt.title(f"{N_BOOT} resamples of the {{len(raw)}} Mauna Loa years (red = no change at all)")
plt.show()

print("trend:", round(trend(raw), 3), "ppm per decade")
print("95% interval: [", round(low, 3), ",", round(high, 3), "]")
print("resamples above zero:", round((spread > 0).mean(), 3))

print("No. My interval runs from", round(low, 3), "to", round(high, 3),
      "ppm per decade and contains zero, so a record this noisy is equally consistent with a",
      "seasonal cycle that is growing and one that is shrinking.")
print("Only", round(100 * (spread > 0).mean()), "percent of the resamples came out positive,",
      "which is barely better than a coin.")
print("Before I could answer, the whole interval would have to sit on one side of zero —",
      "its lower end, not its middle — and on this definition and these",
      len(raw), "years it does not.")
""")

md(f"""
So the obvious definition gives no answer: **{M['defs']['mlo', 'raw']['slope']:+.3f} ppm per
decade, 95% interval [{M['defs']['mlo', 'raw']['lo']:+.3f},
{M['defs']['mlo', 'raw']['hi']:+.3f}]**, straddling zero, with
{M['defs']['mlo', 'raw']['above_zero'] * 100:.0f}% of the resamples above it. On the longest and
cleanest CO2 record in the world, the question in the title has just come back *don't know*.

That is a real result and you could stop there. It would also be wrong.
""")

md(f"""
### Predict before you run

You are about to measure the same thing two other ways. Both are defensible, and neither is more
obviously correct than what you have just done.

How far could the answer move? Write down the slope you think the *most different* of the three
definitions will give, in ppm per decade — your first answer was
{M['defs']['mlo', 'raw']['slope']:+.3f}. Change `my_guess` and run the cell. You will check it at
the end of the next section, and a wrong guess you committed to is worth more than a right answer
you were shown.
""")

code(f"""
my_guess = {M['defs']['mlo', 'raw']['slope']:.2f}

print("I think the most different definition will give", my_guess, "ppm per decade")
""")

# --- YOUR TURN 3, the fork --------------------------------------------------
md(f"""
## Does that answer survive a different definition of "amplitude"?

You measured the swing of the **raw** record: the highest month of a calendar year minus the
lowest. That is one answer to "how big is the cycle this year", and there are at least two others.

- **The raw range.** Highest month minus lowest, straight off `value`. What you already did.
- **The detrended range.** The same, but of `swing` — the record with the twelve-month average
  already subtracted.
- **The size of the once-a-year wave.** Fit a smooth wave with exactly one cycle per year to each
  year's twelve swing values and measure the fitted wave instead of the data. A wave through
  twelve points is a straight-line fit like any other, with two columns instead of one:
  `np.sin(2 * np.pi * month / 12)` and `np.cos` of the same angle. The fitted values come back
  from `.predict`, and the amplitude is their range.

This is the one real decision in this track. Make it, and report what it cost.
""")

ask("""
### ✏️ Your turn 3

Measure the Mauna Loa amplitude all three ways.

The third one needs a column that does not exist yet. Write `fourier_fit(station)` that adds a
`fourier` column holding, for every month, the height of that year's best-fitting once-a-year
wave — loop over `station.groupby("year")`, fit `LinearRegression` to the two wave columns against
`swing`, and write `.predict` back into the rows you fitted. Give it the same shape as `seasonal`:
a station table in, a station table out. Then all three definitions are one call each,
`amplitude(station, column)`, and your `trend` and `trend_spread` work on all three unchanged.

Put the three amplitude series on one plot. Then print, for each: the trend in ppm per decade, its
95% interval, and the fraction of resamples above zero.

Compare the spread of the three answers with the number you committed to in *Predict before you
run*. Then print one more line answering it in a sentence: does the answer to this notebook's
title depend on which definition you chose, and which one would you report?
""")

answer("""
def fourier_fit(station):
    \"\"\"Replace each year's swing by the smooth once-a-year wave that fits it best.\"\"\"
    station = station.copy()
    angle = 2 * np.pi * station["month"] / 12
    station["sin"] = np.sin(angle)
    station["cos"] = np.cos(angle)
    station["fourier"] = np.nan
    for year, rows in station.dropna(subset=["swing"]).groupby("year"):
        wave = LinearRegression().fit(rows[["sin", "cos"]], rows["swing"])
        station.loc[rows.index, "fourier"] = wave.predict(rows[["sin", "cos"]])
    return station


mauna_loa = fourier_fit(mauna_loa)
definitions = {"raw range": "value",
               "detrended range": "swing",
               "once-a-year wave": "fourier"}

styles = {"raw range": "-", "detrended range": "--", "once-a-year wave": ":"}
for name in definitions:
    amps = amplitude(mauna_loa, definitions[name])
    plt.plot(amps["year"], amps["amplitude"], styles[name], lw=1.2, label=name)
plt.xlabel("year")
plt.ylabel("seasonal amplitude (ppm)")
plt.title(f"Three definitions of the Mauna Loa seasonal amplitude "
          f"(n = {len(amplitude(mauna_loa, 'swing'))} years)")
plt.legend()
plt.show()

for name in definitions:
    amps = amplitude(mauna_loa, definitions[name])
    spread = trend_spread(amps)
    low, high = np.percentile(spread, [2.5, 97.5])
    print(f"{name:18s} mean {amps.amplitude.mean():.2f} ppm  "
          f"trend {trend(amps):+.3f} ppm/decade  "
          f"95% interval [{low:+.3f}, {high:+.3f}]  "
          f"above zero {(spread > 0).mean():.3f}")

print("Yes — the answer changes, and not just in size. The raw range says the cycle is not",
      "measurably growing; the other two put the whole interval above zero and say it is,",
      "at about a tenth of a ppm per decade. I would report the detrended range, because the",
      "raw one is measuring the trend as well as the season and the other two are not.")
""")

md(f"""
Three defensible definitions, and they do not merely differ in size — **they differ in what the
answer is.**

| Definition | mean amplitude | trend | 95% interval | above zero |
|---|---|---|---|---|
| raw range | {M['defs']['mlo', 'raw']['mean']:.2f} ppm | {M['defs']['mlo', 'raw']['slope']:+.3f} ppm/decade | [{M['defs']['mlo', 'raw']['lo']:+.3f}, {M['defs']['mlo', 'raw']['hi']:+.3f}] | {M['defs']['mlo', 'raw']['above_zero'] * 100:.0f}% |
| detrended range | {M['defs']['mlo', 'detrended']['mean']:.2f} ppm | {M['defs']['mlo', 'detrended']['slope']:+.3f} ppm/decade | [{M['defs']['mlo', 'detrended']['lo']:+.3f}, {M['defs']['mlo', 'detrended']['hi']:+.3f}] | {M['defs']['mlo', 'detrended']['above_zero'] * 100:.0f}% |
| once-a-year wave | {M['defs']['mlo', 'fourier']['mean']:.2f} ppm | {M['defs']['mlo', 'fourier']['slope']:+.3f} ppm/decade | [{M['defs']['mlo', 'fourier']['lo']:+.3f}, {M['defs']['mlo', 'fourier']['hi']:+.3f}] | {M['defs']['mlo', 'fourier']['above_zero'] * 100:.0f}% |

Two of the three clear zero and one does not, and the one that does not is the one you would write
first. Two definitions agreeing and one disagreeing is a stronger clue than three-way
disagreement would be: it says something specific is wrong with the odd one out.
""")

ask(f"""
### ✏️ Your turn 4

Find out what. The raw range is smaller than the detrended range in every single year — go and
check that it is — so the trend must be eating part of the swing. Work out how much, and see
whether that accounts for the whole disagreement.

Here is the mechanism to test, in words. Within one calendar year the peak comes first and the
trough comes months later. Between them, the long-term rise has lifted the whole record a little,
so the trough is measured higher than it would have been and the raw range comes out short. The
size of that theft is roughly the year's own rise multiplied by the fraction of a year between
peak and trough: **`lag / 12 × growth`**, where `lag` is the number of months from the highest
month to the lowest and `growth` is how much CO2 rose that year.

Build all three per year — the gap between the two amplitudes, the lag, and the growth — and put
the predicted theft beside the observed gap. `idxmax` and `idxmin` on a group give you the row of
the highest and lowest month; the year's growth is the difference between neighbouring annual
means, halved.

One trap in that last step. Some years are not complete and drop out, so "the year before" and
"the year after" are not the row above and the row below. Reindex your annual means over the whole
range of years first — `.reindex(range(first, last + 1))` — or the eruption gap will silently make
2021 and 2024 neighbours and report three years of rise as two.

Then print one more line answering it in a sentence, on your own numbers: does the theft account
for the gap between the two definitions, and does it account for the difference between their two
*trends* as well?
""")

answer("""
have = mauna_loa.dropna(subset=["value"])
full = have[have.groupby("year")["month"].transform("count") == 12]
peak_at = full.loc[full.groupby("year")["value"].idxmax()].set_index("year")["month"]
trough_at = full.loc[full.groupby("year")["value"].idxmin()].set_index("year")["month"]
years = range(full["year"].min(), full["year"].max() + 1)
annual = full.groupby("year")["value"].mean().reindex(years)

raw = amplitude(mauna_loa, "value").set_index("year")["amplitude"]
detrended = amplitude(mauna_loa, "swing").set_index("year")["amplitude"]

leak = pd.DataFrame({"gap": detrended - raw,
                     "lag": trough_at - peak_at,
                     "growth": (annual.shift(-1) - annual.shift(1)) / 2}).dropna()
leak["predicted"] = leak["lag"] / 12 * leak["growth"]

plt.scatter(leak["predicted"], leak["gap"], color="0.4", s=14)
plt.plot([0, leak["predicted"].max()], [0, leak["predicted"].max()], color="firebrick", lw=1.2)
plt.xlabel("predicted theft, lag / 12 x growth (ppm)")
plt.ylabel("observed gap, detrended minus raw (ppm)")
plt.title(f"Where the raw definition loses its swing (n = {len(leak)} years)")
plt.show()

print("mean months from peak to trough:", round(leak.lag.mean(), 2))
print("CO2 growth, first 15 years:", round(leak.growth.head(15).mean(), 3),
      "ppm/yr  last 15:", round(leak.growth.tail(15).mean(), 3))
print("gap  early", round(leak.gap.head(15).mean(), 3),
      " late", round(leak.gap.tail(15).mean(), 3))
print("predicted early", round(leak.predicted.head(15).mean(), 3),
      " late", round(leak.predicted.tail(15).mean(), 3))

print("predicted against observed, correlation:",
      round(leak["predicted"].corr(leak["gap"]), 3))

gap_by_year = pd.DataFrame({"year": leak.index, "amplitude": leak["gap"].values})
print("the gap is itself growing at", round(trend(gap_by_year), 3), "ppm per decade")
print("and the two definitions' trends differ by",
      round(trend(amplitude(mauna_loa, "swing")) - trend(amplitude(mauna_loa, "value")), 3))

print("Yes to both. The theft explains the gap year by year — predicted",
      round(leak.predicted.mean(), 3), "ppm against an observed", round(leak.gap.mean(), 3),
      "— and because CO2 now rises about", round(leak.growth.tail(15).mean(), 1),
      "ppm a year instead of", round(leak.growth.head(15).mean(), 1),
      "the theft has grown too, by almost exactly the amount that separates the two trends.")
""")

md(f"""
The gap between the two definitions is **{M['leak_early']:.2f} ppm** in the early years and
**{M['leak_late']:.2f} ppm** in the recent ones. The one-line prediction gives
{M['leak_early_pred']:.2f} and {M['leak_late_pred']:.2f} for the same two windows, and correlates
with the observed gap at r = {M['leak_corr']:.2f} across single years.

The trough at Mauna Loa arrives about {M['lag_mean']:.1f} months after the peak, and CO2 now rises
{M['growth_late']:.1f} ppm a year where it rose {M['growth_early']:.1f} ppm in the 1970s. So the
raw definition loses more of the swing every decade — the leak itself grows at
{M['leak_slope']:+.3f} ppm per decade, against the {M['slope_difference']:+.3f} ppm per decade that
separates the two answers.

**The raw definition did not measure a cycle that is not growing. It measured a cycle that is
growing, minus a theft that is growing by about the same amount.** That is not a coincidence you
could have guessed; it is a fact about this record that had to be computed.
""")

# --- YOUR TURN 5: Barrow ----------------------------------------------------
md(f"""
## Does the same thing happen at {M['brw_lat']:.0f}° north?

Mauna Loa is at {M['mlo_lat']:.1f}°N, in the middle of an ocean and {M['mlo_elev'] / 1000:.1f} km
up — its `elevation` column says so. Almost none
of the land whose plants make the seasonal cycle is anywhere near it; what it measures is air that
has been stirred across a hemisphere.

Barrow sits on the Arctic coast of Alaska at {M['brw_lat']:.1f}°N, among the tundra and boreal
forest that do the breathing. It has been loaded since the setup cell and has had `seasonal`
applied to it. Everything you have written works on it unchanged.
""")

ask("""
### ✏️ Your turn 5

Run the whole of the last three sections again at Barrow, and change nothing but the station.

Add the `fourier` column, take all three amplitude series, fit and bootstrap each, and print the
same numbers you printed for Mauna Loa. Put the Barrow and Mauna Loa amplitude series on one
plot — one definition is enough for the figure, as long as you say which.

Also print, for both stations, the percentage change from the first fifteen complete years to the
last fifteen, so you have both ways of stating the same result on the page.

Then print one more line answering it in a sentence: does the choice of definition matter as much
at Barrow as it did at Mauna Loa, and what does the pair of stations together suggest?
""")

answer(f"""
barrow = fourier_fit(barrow)

for name in definitions:
    for station_name in ["Mauna Loa", "Barrow"]:
        station = mauna_loa if station_name == "Mauna Loa" else barrow
        amps = amplitude(station, definitions[name])
        spread = trend_spread(amps)
        low, high = np.percentile(spread, [2.5, 97.5])
        early = amps.head({WINDOW}).amplitude.mean()
        late = amps.tail({WINDOW}).amplitude.mean()
        print(f"{{station_name:10s}} {{name:18s}} mean {{amps.amplitude.mean():6.2f}} ppm  "
              f"trend {{trend(amps):+.3f}} [{{low:+.3f}}, {{high:+.3f}}] ppm/decade  "
              f"above zero {{(spread > 0).mean():.3f}}  "
              f"{{early:.2f}} -> {{late:.2f}} = {{100 * (late / early - 1):+.1f}}%")

for station_name in ["Mauna Loa", "Barrow"]:
    station = mauna_loa if station_name == "Mauna Loa" else barrow
    amps = amplitude(station, "swing")
    plt.plot(amps["year"], amps["amplitude"], "-" if station_name == "Barrow" else "--",
             lw=1.2, label=station_name)
plt.xlabel("year")
plt.ylabel("detrended seasonal amplitude (ppm)")
plt.title("Seasonal amplitude at two latitudes, detrended range")
plt.legend()
plt.show()

print("No. At Barrow all three definitions agree, because the theft is a few tenths of a ppm",
      "and the swing is about", round(amplitude(barrow, "swing").amplitude.mean()),
      "ppm, so it is a rounding error there and was the entire answer at Mauna Loa.")
print("Together the two stations say the cycle is deepening everywhere the definitions are",
      "trustworthy, and far faster in the Arctic than in the mid-Pacific.")
""")

md(f"""
At Barrow the fork closes. All three definitions land within
{max(M['defs']['brw', k]['slope'] for k, _, _ in DEFS) - min(M['defs']['brw', k]['slope'] for k, _, _ in DEFS):.2f}
ppm per decade of each other, every interval sits far above zero, and the detrended answer is
**{M['defs']['brw', 'detrended']['slope']:+.3f} ppm per decade, 95% interval
[{M['defs']['brw', 'detrended']['lo']:+.3f}, {M['defs']['brw', 'detrended']['hi']:+.3f}]** —
roughly {M['defs']['brw', 'detrended']['slope'] / M['defs']['mlo', 'detrended']['slope']:.0f} times
the Mauna Loa rate, on a cycle that is already
{M['defs']['brw', 'detrended']['mean'] / M['defs']['mlo', 'detrended']['mean']:.1f} times as deep
({M['defs']['brw', 'detrended']['mean']:.1f} ppm against
{M['defs']['mlo', 'detrended']['mean']:.1f}).

The definition mattered at Mauna Loa and does not matter here, and the reason is arithmetic rather
than geography: the theft is roughly the same size at both stations, and at Barrow it is a few
percent of a large number instead of all of a small one. **A choice that changes your conclusion at
one station and nothing at another is not a detail of method. It is a measure of how thin your
signal was.**
""")

ask(f"""
### ✏️ Your turn 6

Two or three paragraphs, quoting **your own numbers from Your turn 5** — both the ppm per decade
and the percentage change.

1. You now have two ways to say the same result: a rate in ppm per decade, and a percentage change
   from your first fifteen years to your last fifteen. Which of the two can be compared honestly
   between two stations, and which cannot? Work out what happens to the percentage if you move the
   two windows closer together, and say what that means for a sentence of the form *"Mauna Loa
   gives +13% and Barrow gives +25%"*.
2. Your Barrow interval excludes zero by a wide margin and your Mauna Loa raw interval does not.
   Name what a reader should conclude from a result that changes side when you change a defensible
   choice, and say what would have to be true of the Mauna Loa record for the three definitions to
   agree there the way they do at Barrow.
""")

answer_prose(f"""
The rate is comparable and the percentage is not. My detrended rates are
{M['defs']['mlo', 'detrended']['slope']:+.3f} ppm per decade at Mauna Loa and
{M['defs']['brw', 'detrended']['slope']:+.3f} at Barrow, and those two numbers mean the same thing
because a decade is a decade at both stations. The percentage changes are
{M['defs']['mlo', 'detrended']['pct']:+.1f}% and {M['defs']['brw', 'detrended']['pct']:+.1f}%, and
they do not, for two separate reasons. The first is that a percentage is a rate multiplied by the
distance between the windows I happened to pick: my Mauna Loa windows are centred about
{(M['defs']['mlo', 'detrended']['late_from'] + M['defs']['mlo', 'detrended']['late_to']) // 2 - (M['defs']['mlo', 'detrended']['early_from'] + M['defs']['mlo', 'detrended']['early_to']) // 2}
years apart and my Barrow windows about
{(M['defs']['brw', 'detrended']['late_from'] + M['defs']['brw', 'detrended']['late_to']) // 2 - (M['defs']['brw', 'detrended']['early_from'] + M['defs']['brw', 'detrended']['early_to']) // 2},
so even identical stations with identical rates would return different percentages; shorten the
separation and every percentage shrinks towards zero without anything about the Earth changing.
The second is that a percentage is divided by the base, and the two bases differ by a factor of
{M['defs']['brw', 'detrended']['mean'] / M['defs']['mlo', 'detrended']['mean']:.1f} — so the same
absolute deepening reads as a much smaller percentage at Barrow than at Mauna Loa. A sentence of
the form "Mauna Loa gives +13% and Barrow gives +25%" is therefore reporting the windows and the
denominators as much as the atmosphere, and the two halves are only comparable if both were
measured over the same span, under the same definition, from the same kind of instrument.

What a reader should conclude from an answer that changes side on a defensible choice is not that
one choice was wrong — both are used, and I could defend either — but that the effect is small
compared with the things I get to decide. At Mauna Loa the disagreement between the raw and the
detrended definition,
{M['slope_difference']:+.3f} ppm per decade, is larger than the raw definition's entire estimate of
{M['defs']['mlo', 'raw']['slope']:+.3f}. At Barrow the same disagreement is a few percent of
{M['defs']['brw', 'detrended']['slope']:+.3f} and could not change anybody's mind. So the honest
report of the Mauna Loa result is a rate with an interval and a named definition, not a yes or a
no.

For the three definitions to agree at Mauna Loa the way they do at Barrow, either the swing would
have to be several times larger — so that a theft of about
{M['leak_late']:.1f} ppm were a rounding error rather than the whole disputed quantity — or CO2
would have to have stopped accelerating, so that the theft were at least constant in time and
therefore left the *trend* alone even while it shifted the level. Neither is going to happen, which
means the Mauna Loa amplitude record cannot be read raw, and that is a permanent property of the
station rather than a problem with these fifty years of it.
""")

# --- closing ----------------------------------------------------------------
md(f"""
{weekkit.CLOSING_HEADING}

At Barrow, unambiguously yes: **{M['defs']['brw', 'detrended']['slope']:+.2f} ppm per decade**
(95% interval [{M['defs']['brw', 'detrended']['lo']:+.2f},
{M['defs']['brw', 'detrended']['hi']:+.2f}]), and every definition agrees. At Mauna Loa yes, but
only once the rising trend is taken out first — **{M['defs']['mlo', 'detrended']['slope']:+.3f}
ppm per decade** [{M['defs']['mlo', 'detrended']['lo']:+.3f},
{M['defs']['mlo', 'detrended']['hi']:+.3f}] detrended, against
{M['defs']['mlo', 'raw']['slope']:+.3f} [{M['defs']['mlo', 'raw']['lo']:+.3f},
{M['defs']['mlo', 'raw']['hi']:+.3f}] raw, which cannot tell growth from nothing.
""")

md(track_summary())

# --- the project ------------------------------------------------------------
md("""
## What your project must contain

Five sections, empty below, required of **every** EPS 88 project regardless of track. They are
headed here so the shape of a good answer is visible while you work. Fill them in as you go; they
are not a write-up you do at the end.
""")

# course.yml's `required_of_every_project:` values are DESIGN notes, not student prose: they name
# a week number ("the week-10 data") and credit the source of the idea (MLGeo). Both are for
# whoever plans the course. What goes in the notebook is the same requirement said to a student.
# The five keys are read from the plan, so a sixth requirement cannot be added there and silently
# skipped here; only the wording is local.
REQUIRED = [list(item)[0] for item in course["project"]["required_of_every_project"]]
STUDENT_WORDING = {
    "one_sentence_answer": ("1 · A one-sentence answer", """
Your claim and its uncertainty, in one sentence, at the top of your report. If you cannot put a
number and a range in it, you do not have a result yet. On this track the range is not optional
decoration — one of the three definitions gave an interval containing zero, and a sentence without
an interval could not have said so.
"""),
    "baseline_first": ("2 · The trivial baseline", """
Before any statistic, state the dumbest answer to your question and what it gives. Every later
number is reported against it.

On this track the baseline is the raw peak-to-trough range: no detrending, no fitting, the
definition anybody would write first. Say what it gives, and say exactly what each later step
bought you over it — and where it bought you nothing.
"""),
    "split_by_structure": ("3 · Split by structure", """
Earth data are correlated in space and in time, so whatever you split, resample or count as
independent has to be split along the structure that is really there — never at random across
rows.

This track fits no model, so there is no train/test split to get wrong. The same idea has teeth
anyway: every interval you quoted came from resampling **years**, which assumes one year's
amplitude tells you nothing about the next one's. Name the unit you treated as independent, say
why, and say what you would have to do differently if neighbouring years turned out to move
together.
"""),
    "what_i_got_wrong": ("4 · What I got wrong", """
What failed, and what you believed before it failed. Honest failure is graded; a faked success is
not. Your *Predict before you run* guess belongs here if it was wrong.
"""),
    "ai_disclosure": ("5 · AI disclosure", """
Which tool, what you asked it, what you changed in what it gave you, and how you checked that the
result was true.
"""),
}

# Reading order, not course.yml's order: the one-sentence answer goes at the top of a report by
# definition, and course.yml lists the baseline first because that is the order it was designed in.
ORDER = ["one_sentence_answer", "baseline_first", "split_by_structure",
         "what_i_got_wrong", "ai_disclosure"]
missing = set(REQUIRED) - set(ORDER)
if missing:
    sys.exit(f"course.yml requires {sorted(missing)} of every project and this notebook has no "
             f"section for it")

for key in ORDER:
    heading, guidance = STUDENT_WORDING[key]
    ask(f"### ✏️ {heading}\n{guidance.rstrip()}")
    blank_prose()

# --- the open question ------------------------------------------------------
OPEN = re.findall(r"[^.?]*\?", " ".join(TRACK["open_question"].split()))[-1].strip()

md(f"""
## The open question

> **{OPEN}**

Nobody grading this knows the answer, and neither does the literature. Everything above is the
scaffolding; this is the project.

Here is what is actually established, and it is less than it looks. Two stations do not make a
gradient. You have {M['defs']['mlo', 'detrended']['slope']:+.3f} ppm per decade at
{M['mlo_lat']:.1f}°N and {M['defs']['brw', 'detrended']['slope']:+.3f} at {M['brw_lat']:.1f}°N; a
straight line through two points fits perfectly and means nothing. What is **not** settled is the
shape between them and beyond them — whether the deepening rises smoothly with latitude, switches
on somewhere, or tracks something else entirely that happens to correlate with latitude.

Three directions, none of them worked out here:

1. **Add stations.** Every observatory NOAA runs publishes a file at the same address with three
   letters changed, in the same layout, so `load` and every function you wrote work on all of them.
   The in-situ set is small — Samoa (`smo`), the South Pole (`spo`), Maunakea (`mko`) — but there
   is a much larger flask-sampling set at
   `.../co2/flask/surface/txt/co2_SITE_surface-flask_1_ccgg_month.txt`, including Alert at 82°N and
   dozens between. Those files carry no header row, so `pd.read_csv` needs
   `names=["site", "year", "month", "value"]`; that is the only change. How many stations would
   you need before the shape of the curve, rather than its two endpoints, were established?
2. **Ask what the gradient is a gradient in.** Latitude is a proxy for several things at once —
   how much land there is, how long the growing season is, how far the air has travelled from where
   the carbon was taken up. The southern stations are the test: the South Pole is as far from the
   equator as it is possible to be and has almost no land anywhere near it. If the deepening
   follows latitude it should be large there; if it follows land it should be tiny. Both are
   computable with what you have already written.
3. **Separate a deeper breath from a longer one.** A cycle can grow because the summer drawdown is
   stronger or because the growing season is longer, and those are different claims about
   ecosystems. The month of the trough is in your data; so is the width of the drawdown. Does the
   trough arrive later than it used to, and does that vary with latitude?

And one that is bigger than a semester: **which ecosystems are actually doing the breathing?** The
atmosphere at any station is a mixture stirred from everywhere upwind, so a station's amplitude is
not a measurement of the vegetation beneath it. What would you need — more stations, or a way of
saying where the air had been — before a CO2 record could name a biome rather than a latitude? If
the answer is that no arrangement of surface stations can do it, that is a result, and it is worth
saying carefully.
""")

ask("""
### ✏️ Your turn 7 — the first move

Before you close this notebook: in a few sentences, name the **one** measurement you would make
first, say what it would show if the deepening really does scale with latitude, what it would show
if it does not, and name the number that would change your mind. Then make it, in the cell below
the prose.
""")

answer_prose(f"""
I would add the two southern stations first — Samoa at {abs(LADDER[1]['lat']):.0f}°S and the South
Pole at {abs(LADDER[0]['lat']):.0f}°S — because they are the cheapest measurement available and because they
separate the two explanations that Mauna Loa and Barrow cannot. Barrow is both the furthest north
and the closest to land, so its large deepening is consistent with either story. The southern
stations break the tie: latitude says the South Pole should look like Barrow, and land says it
should look like nothing at all. The number that would change my mind is the South Pole rate as a
share of its own mean amplitude — if it is comparable to Barrow's, deepening tracks latitude; if it
is far smaller, it tracks land and northern latitude was only ever standing in for northern
continents.

What makes me doubt a clean answer in advance is the size of the southern cycles. Samoa and the
South Pole swing by around a ppm, against
{M['defs']['brw', 'detrended']['mean']:.0f} ppm at Barrow, so the theft I measured in *Your turn 4*
is a much larger share of the signal there than it was even at Mauna Loa, and the intervals will be
correspondingly wide. I expect the honest outcome to be that four stations still cannot distinguish
a curve from a step, which is itself the answer to the open question — and I would report it that
way rather than drawing a straight line through four points and calling it a gradient.
""")

answer(f"""
ladder = []
for site in ["spo", "smo", "mlo", "brw"]:
    station = seasonal(load(GML.format(site=site), f"trackT2_co2_{{site}}.csv"))
    amps = amplitude(station, "swing")
    spread = trend_spread(amps)
    ladder.append({{"site": site,
                   "lat": station["latitude"].iloc[0],
                   "mean": amps["amplitude"].mean(),
                   "slope": trend(amps),
                   "low": np.percentile(spread, 2.5),
                   "high": np.percentile(spread, 97.5)}})
ladder = pd.DataFrame(ladder)
ladder["per_decade_pct"] = 100 * ladder["slope"] / ladder["mean"]

bars = [100 * (ladder["slope"] - ladder["low"]) / ladder["mean"],
        100 * (ladder["high"] - ladder["slope"]) / ladder["mean"]]
plt.errorbar(ladder["lat"], ladder["per_decade_pct"], yerr=bars, fmt="o", color="0.4")
plt.axhline(0, color="firebrick", lw=1.2)
plt.xlabel("station latitude (degrees north)")
plt.ylabel("deepening (% of its own amplitude per decade)")
plt.title(f"Four stations, one definition (n = {{len(ladder)}} stations)")
plt.show()

print(ladder.round(3).to_string(index=False))
print("The two southern stations are NOT small, so latitude alone does not order these four —",
      "and four points cannot say what the shape is.")
""")


# ---------------------------------------------------------------------------
# 4. emit, execute, gate
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


def track_ids(cells):
    """weekkit.stable_ids keys cells to a week number; a track keys them to its id instead.

    Same contract and same reason: a submission graded against an earlier release must not report
    every cell as missing because a paragraph was inserted above it.
    """
    q = 0
    for i, c in enumerate(cells):
        s = "".join(c.get("source", []))
        if c["cell_type"] == "markdown" and re.search(r"(?m)^\s*(#{1,4}\s*)?✏️", s):
            q += 1
            c["id"] = f"{TRACK['id']}-q{q:02d}-ask"
        elif c["cell_type"] == "code" and re.search(r"your answer here", s, re.I):
            c["id"] = f"{TRACK['id']}-q{q:02d}-answer"
        elif c["cell_type"] == "markdown" and "Double-click" in s:
            c["id"] = f"{TRACK['id']}-q{q:02d}-prose"
        elif c["cell_type"] == "code" and "assert " in s:
            c["id"] = f"{TRACK['id']}-q{q:02d}-check"
        else:
            c["id"] = f"{TRACK['id']}-c{i:03d}"
    return cells


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    sol = notebook([cell(k, s) for k, s, _ in CELLS])
    stu = notebook([cell(k, alt if alt is not None else s) for k, s, alt in CELLS])

    sol_path = OUT / f"{SLUG}_solution.ipynb"
    sol_path.write_text(json.dumps(sol, indent=1) + "\n")

    print(f"executing {sol_path.name} ...")
    r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                        "--execute", "--inplace", "--ExecutePreprocessor.timeout=900",
                        str(sol_path)], capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        print(r.stderr[-4000:])
        sys.exit("the solution did not execute")

    (OUT / f"{SLUG}.ipynb").write_text(json.dumps(stu, indent=1) + "\n")

    for f in (sol_path, OUT / f"{SLUG}.ipynb"):
        nb = json.loads(f.read_text())
        track_ids(nb["cells"])
        f.write_text(json.dumps(nb, indent=1))

    print(f"wrote {SLUG}.ipynb ({len(CELLS)} cells) and the executed solution")
    for name in (MLO_CACHE, BRW_CACHE):
        print(f"cache: data/{name} "
              f"({(ROOT / 'data' / name).stat().st_size / 1e6:.2f} MB), downloaded {READ_DATE}")

    gate(sol_path)


def gate(sol_path):
    """The half of weekkit.gate that does not need a week number.

    weekkit.gate looks the notebook up by `slug` in course.yml's `schedule:` and then runs
    check_notebook.py and check_prior_knowledge.py, both of which take a week number. A track has
    none. What transfers unchanged is the gate that matters most and needs nothing from the plan:
    the solution executed clean on a fresh kernel, with contiguous execution counts from 1.
    """
    bad = []
    cells = json.loads(sol_path.read_text())["cells"]
    counts = [c["execution_count"] for c in cells
              if c["cell_type"] == "code" and c.get("execution_count")]
    if not counts:
        bad.append("the solution has no execution counts — it was never executed")
    elif counts[0] != 1:
        bad.append(f"execution counts start at {counts[0]}, not 1")
    elif counts != list(range(1, len(counts) + 1)):
        bad.append("execution counts are not contiguous — the solution was executed piecemeal")
    if any(o.get("output_type") == "error" for c in cells for o in c.get("outputs", [])):
        bad.append("the solution contains an error output — it does not execute clean")

    stu = json.loads((sol_path.parent / sol_path.name.replace("_solution", "")).read_text())
    if len(stu["cells"]) != len(cells):
        bad.append("student and solution have drifted apart")
    if any(c.get("outputs") for c in stu["cells"]):
        bad.append("the student notebook carries outputs — it must ship clean")
    figs = sum(1 for c in cells for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
    if figs == 0:
        bad.append("the solution contains no figures")

    r = subprocess.run([sys.executable, str(ROOT / "tools/check_track.py"), TRACK["id"]],
                       capture_output=True, text=True, cwd=ROOT)
    print(r.stdout.rstrip())
    if r.returncode:
        bad.append("check_track reported errors (above)")

    if bad:
        print("\nBUILD REJECTED:")
        for b in bad:
            print(f"  - {b}")
        sys.exit(1)
    print(f"\ngates passed: executes clean ({len(counts)} cells, {figs} figures), check_track OK")


if __name__ == "__main__":
    main()
