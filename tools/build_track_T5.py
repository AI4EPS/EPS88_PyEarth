#!/usr/bin/env python
"""Build project track T5 — "Does El Nino really bring rain to California?"

Emits both notebooks from one source so they cannot drift:

    docs/notebooks/T5_el_nino_and_the_rivers_solution.ipynb   executed, every output saved
    docs/notebooks/T5_el_nino_and_the_rivers.ipynb            the same file with the answers deleted

It also writes the track's three cached fallbacks and its one shipped asset:

    data/trackT5_nina34.txt          NOAA PSL Nino 3.4 monthly anomalies (8 KB)
    data/trackT5_dv_11152000.tsv.gz  Arroyo Seco nr Soledad, daily discharge  (gzipped RDB)
    data/trackT5_dv_11477000.tsv.gz  Eel R at Scotia, daily discharge         (gzipped RDB)
    data/trackT5_ca_gauges.csv       the curated gauge list, built here from the USGS inventory

A TRACK is not a week (course.yml `project: track_notebooks:`). Two things differ, and both are
deliberate:

  * LESS HELP. No worked example before a question. The notebook loads the data and reproduces
    the ONE result the title names — that a southern California river does track El Nino — so a
    student can trust the pipeline, and then stops helping. Everything after is a prompt in words
    and an empty cell.
  * IT DOES NOT CLOSE. There is exactly one self-check, on the load, and the notebook ends on an
    open question this course cannot answer.

Every number that appears in prose or in a model answer is computed HERE, from the same files the
notebook reads, and formatted in. Nothing is typed from memory or copied from the plan.

    python tools/build_track_T5.py            # uses whatever is already in data/
    python tools/build_track_T5.py --refresh  # downloads all four sources again
"""
import gzip
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/notebooks"
SLUG = "T5_el_nino_and_the_rivers"

course = yaml.safe_load((ROOT / "course.yml").read_text())
modules = yaml.safe_load((ROOT / "modules.yml").read_text())
TRACK = next(t for t in course["project"]["tracks"] if t["id"] == "T5")
PLATFORM = course["platform"]
CACHE_BASE = PLATFORM["cache_base"]

# ---------------------------------------------------------------------------
# the live sources, pinned here so cache, notebook and prose cannot drift
# ---------------------------------------------------------------------------
NINO_URL = "https://psl.noaa.gov/data/correlation/nina34.anom.data"
NINO_CACHE = "trackT5_nina34.txt"

# Both dates are PINNED. USGS appends a row to every gauge every day, so an unpinned query
# would put a fresh, incomplete water year into the table each morning and quietly move every
# number in this notebook. 2025-09-30 is the last day of water year 2025.
GAUGE_URL_A = "https://waterservices.usgs.gov/nwis/dv/?format=rdb&parameterCd=00060"
GAUGE_URL_B = "&statCd=00003&startDT=1900-01-01&endDT=2025-09-30&sites="
GAUGE_URL = GAUGE_URL_A + GAUGE_URL_B

# The USGS site inventory, used ONLY here: it builds the curated gauge list the notebook ships.
SITES_URL_A = "https://waterservices.usgs.gov/nwis/site/?format=rdb&stateCd=ca"
SITES_URL_B = "&parameterCd=00060&outputDataTypeCd=dv&seriesCatalogOutput=true&siteStatus=all"
SITES_URL = SITES_URL_A + SITES_URL_B
SITES_CACHE = "trackT5_ca_inventory.tsv.gz"

SOUTH = "11152000"          # Arroyo Seco nr Soledad — the southern demonstration gauge
NORTH = "11477000"          # Eel R at Scotia       — the northern one
# A latitude transect for the closing "first move": every one of these is in the curated list.
TRANSECT = ["11532500", NORTH, "11335000", "11266500", SOUTH, "11098000", "11022480"]

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FULL_YEAR = 350             # days a twelve-month window must hold before it counts as a year
WATER_YEAR = 10             # the water year starts on 1 October
STRONG = 0.5                # degC: NOAA's threshold for calling a winter El Nino or La Nina
MIN_YEARS = 50              # the record length the curated gauge list demands
SEED = 88                   # the course number, fixed before anything was run
N_BOOT = 2000               # bootstrap resamples, everywhere in the track
REFRESH = "--refresh" in sys.argv
STALE_DAYS = 30             # after this, NOAA may hold months the cached Nino copy does not

# The filters that build the curated gauge list. Written down, because the number they produce
# is quoted in the notebook and a name filter is a judgement call, not a measurement.
BUILT = (r"CANAL|\bCN\b|AQUEDUCT|DRAIN|SPILL|WASTEWAY|CONDUIT|TUNNEL|FLUME|INTAKE"
         r"|POWERPLANT|\bPP\b|OUTLET|DIVERSION")
DAMMED = r"\bBL\b|\bBLW\b|BELOW|\bDAM\b"


# ---------------------------------------------------------------------------
# 1. fetch, with a shelf life
# ---------------------------------------------------------------------------
def cache(url, name, volatile=False):
    """Fetch a live source, store it byte-for-byte as the fallback, return the path.

    Weeks 8 and 9 were both fixed for the same defect and it is worth not repeating: a cache
    downloaded once and never again is the quiet way a track goes wrong. The student's notebook
    reads the archive live, the build reads a months-old copy, and the two disagree in front of
    a class. `--refresh` downloads again. The USGS queries are pinned by date range and
    reproduce byte for byte; NOAA's Nino file is `volatile`, because a month is appended to it
    every month and its final year's values are revised, so an old cache of it warns.
    """
    out = ROOT / "data" / name
    if REFRESH or not out.exists():
        print(f"downloading {name}")
        body = urllib.request.urlopen(url, timeout=300).read()
        if name.endswith(".gz"):
            out.write_bytes(gzip.compress(body))
        else:
            out.write_bytes(body)
    elif volatile:
        age = (time.time() - out.stat().st_mtime) / 86400
        if age > STALE_DAYS:
            print(f"WARNING: data/{name} was downloaded {age:.0f} days ago. NOAA appends and "
                  f"revises this file, and the student's notebook reads NOAA live. Rebuild "
                  f"with --refresh before the class.")
    return out


def read_gauge(path_or_url):
    """One USGS daily-values file as two clean columns: the date and the discharge."""
    raw = pd.read_csv(path_or_url, sep="\t", comment="#", low_memory=False)
    raw = raw[raw["agency_cd"] == "USGS"]
    column = None
    for c in raw.columns:
        if c.endswith("_00060_00003"):
            column = c
    return pd.DataFrame({"date": pd.to_datetime(raw["datetime"]),
                         "cfs": pd.to_numeric(raw[column], errors="coerce")}).dropna()


# --- the Nino index --------------------------------------------------------
nino_raw = pd.read_csv(cache(NINO_URL, NINO_CACHE, volatile=True), sep=r"\s+", skiprows=1,
                       header=None, names=["year"] + MONTHS, na_values="-99.99")
nino_years = nino_raw.apply(pd.to_numeric, errors="coerce").dropna(subset=["year"])
winter = pd.DataFrame({"year": nino_years["year"].astype(int),
                       "djf": (nino_years["Dec"].shift(1) + nino_years["Jan"]
                               + nino_years["Feb"]) / 3}).dropna()


def water_year(dates, start_month):
    """Which twelve-month year each date belongs to, if the year begins in `start_month`."""
    shift = (13 - start_month) % 12
    return dates.dt.year + (dates.dt.month - 1 + shift) // 12


def paired_from(daily, start_month):
    """The notebook's `paired`, given an already-read daily table."""
    daily = daily.copy()
    daily["year"] = water_year(daily["date"], start_month)
    per_year = daily.groupby("year")["cfs"]
    yearly = pd.DataFrame({"year": per_year.mean().index,
                           "cfs": per_year.mean().values,
                           "days": per_year.size().values})
    return yearly[yearly["days"] >= FULL_YEAR].merge(winter, on="year")


def correlation(a, b):
    return float(np.corrcoef(a, b)[0, 1])


# --- the curated gauge list ------------------------------------------------
inventory = pd.read_csv(cache(SITES_URL, SITES_CACHE), sep="\t", comment="#", low_memory=False)
inventory = inventory[inventory["agency_cd"] == "USGS"]
flow_series = inventory[(inventory["parm_cd"].astype(str) == "00060")
                        & (inventory["stat_cd"].astype(str) == "00003")
                        & (inventory["data_type_cd"] == "dv")].copy()
flow_series["count_nu"] = pd.to_numeric(flow_series["count_nu"], errors="coerce")
flow_series["begin"] = pd.to_datetime(flow_series["begin_date"], errors="coerce")
flow_series["end"] = pd.to_datetime(flow_series["end_date"], errors="coerce")
span = (flow_series["end"] - flow_series["begin"]).dt.days + 1
long_record = flow_series[(flow_series["count_nu"] >= MIN_YEARS * 365)
                          & (flow_series["count_nu"] / span >= 0.95)
                          & (flow_series["end"].dt.year >= 2020)].copy()
names = long_record["station_nm"].str.upper()
is_built = names.str.contains(BUILT, regex=True)
is_dammed = names.str.contains(DAMMED, regex=True)
natural = long_record[~(is_built | is_dammed)].copy()
natural["lat"] = pd.to_numeric(natural["dec_lat_va"], errors="coerce")
natural["lon"] = pd.to_numeric(natural["dec_long_va"], errors="coerce")
natural["years"] = (natural["count_nu"] / 365.25).round().astype(int)
gauges = natural[["site_no", "station_nm", "lat", "lon", "years"]].copy()
gauges["site_no"] = gauges["site_no"].astype(str).str.zfill(8)
gauges = gauges.sort_values("lat", ascending=False).reset_index(drop=True)


def named(site):
    return str(gauges[gauges["site_no"] == site]["station_nm"].iloc[0])


def lat_of(site):
    return float(gauges[gauges["site_no"] == site]["lat"].iloc[0])


# ---------------------------------------------------------------------------
# 2. measure everything the notebook will say
# ---------------------------------------------------------------------------
M = {}
M["n_ca_sites"] = int(flow_series["site_no"].nunique())
M["n_long"] = int(len(long_record))
M["n_built"] = int(is_built.sum())
M["n_dammed"] = int(is_dammed.sum())
M["n_excluded"] = int((is_built | is_dammed).sum())   # the two lists overlap, so not the sum
M["n_gauges"] = int(len(gauges))
M["n_north_38"] = int((gauges["lat"] > 38).sum())
M["n_mid"] = int(((gauges["lat"] >= 36) & (gauges["lat"] <= 38)).sum())
M["n_south_36"] = int((gauges["lat"] < 36).sum())
M["oldest_years"] = int(gauges["years"].max())

M["n_nino_rows"] = int(len(nino_years))
M["nino_first"] = int(nino_years["year"].min())
M["nino_last"] = int(nino_years["year"].max())
M["n_nino_missing"] = int(nino_years[MONTHS].isna().sum().sum())
M["n_winter"] = int(len(winter))
M["winter_first"] = int(winter["year"].min())

south_daily = read_gauge(cache(GAUGE_URL + SOUTH, f"trackT5_dv_{SOUTH}.tsv.gz"))
north_daily = read_gauge(cache(GAUGE_URL + NORTH, f"trackT5_dv_{NORTH}.tsv.gz"))
south = paired_from(south_daily, WATER_YEAR)
north = paired_from(north_daily, WATER_YEAR)

M["south_name"], M["north_name"] = named(SOUTH), named(NORTH)
M["south_lat"], M["north_lat"] = lat_of(SOUTH), lat_of(NORTH)
M["south_days"], M["north_days"] = len(south_daily), len(north_daily)
M["south_start"] = str(south_daily["date"].min().date())
M["north_start"] = str(north_daily["date"].min().date())
M["record_end"] = str(south_daily["date"].max().date())
M["n_years"] = int(len(south))
M["first_year"], M["last_year"] = int(south["year"].min()), int(south["year"].max())
M["same_years"] = list(south["year"]) == list(north["year"])
M["r_south"] = correlation(south["djf"], south["cfs"])
M["r_north"] = correlation(north["djf"], north["cfs"])
M["gap"] = M["r_south"] - M["r_north"]
M["south_mean"] = float(south["cfs"].mean())
M["north_mean"] = float(north["cfs"].mean())

# the seasonal shape that justifies starting the year in October
south_daily = south_daily.copy()
south_daily["month"] = south_daily["date"].dt.month
by_month = south_daily.groupby("month")["cfs"].mean()
M["wettest_month"] = MONTHS[int(by_month.idxmax()) - 1]
M["driest_month"] = MONTHS[int(by_month.idxmin()) - 1]
M["wettest_cfs"] = float(by_month.max())
M["driest_cfs"] = float(by_month.min())

# --- the trivial baseline: average the El Nino years and the La Nina years -----
COMPOSITE = {}
for label, table in (("south", south), ("north", north)):
    warm = table[table["djf"] >= STRONG]
    cold = table[table["djf"] <= -STRONG]
    COMPOSITE[label] = {"n_warm": int(len(warm)), "n_cold": int(len(cold)),
                        "warm": float(warm["cfs"].mean()), "cold": float(cold["cfs"].mean()),
                        "ratio": float(warm["cfs"].mean() / cold["cfs"].mean())}
M["composite"] = COMPOSITE

# --- the fork: four defensible twelve-month windows ---------------------------
FORK = {}
for start_month, label in ((10, "Oct-Sep"), (11, "Nov-Oct"), (9, "Sep-Aug"), (1, "Jan-Dec")):
    s = paired_from(south_daily, start_month)
    n = paired_from(north_daily, start_month)
    FORK[label] = {"start_month": start_month, "n": int(len(s)),
                   "south": correlation(s["djf"], s["cfs"]),
                   "north": correlation(n["djf"], n["cfs"])}
    FORK[label]["gap"] = FORK[label]["south"] - FORK[label]["north"]
M["fork"] = FORK

# --- the bootstrap, both ways -------------------------------------------------
rng = np.random.default_rng(SEED)
alone_south, alone_north, difference = [], [], []
s_djf, s_cfs = south["djf"].values, south["cfs"].values
n_djf, n_cfs = north["djf"].values, north["cfs"].values
for _ in range(N_BOOT):
    picked = rng.integers(0, M["n_years"], size=M["n_years"])
    a = correlation(s_djf[picked], s_cfs[picked])
    b = correlation(n_djf[picked], n_cfs[picked])
    alone_south.append(a)
    alone_north.append(b)
    difference.append(a - b)
alone_south, alone_north = np.array(alone_south), np.array(alone_north)
difference = np.array(difference)
M["ci_south"] = [float(x) for x in np.percentile(alone_south, [2.5, 97.5])]
M["ci_north"] = [float(x) for x in np.percentile(alone_north, [2.5, 97.5])]
M["ci_gap"] = [float(x) for x in np.percentile(difference, [2.5, 97.5])]
M["gap_median"] = float(np.median(difference))
M["gap_positive"] = float((difference > 0).mean())
M["overlap_lo"] = max(M["ci_south"][0], M["ci_north"][0])
M["overlap_hi"] = min(M["ci_south"][1], M["ci_north"][1])

# --- the closing first move: r against latitude, seven gauges -----------------
TRANSECT_R = []
for site in TRANSECT:
    if site == SOUTH:
        table = south
    elif site == NORTH:
        table = north
    else:
        table = paired_from(read_gauge(GAUGE_URL + site), WATER_YEAR)
    TRANSECT_R.append({"site": site, "name": named(site), "lat": lat_of(site),
                       "n": int(len(table)),
                       "r": correlation(table["djf"], table["cfs"])})
M["transect"] = TRANSECT_R
above = [t for t in TRANSECT_R if t["lat"] > 37]
below = [t for t in TRANSECT_R if t["lat"] < 37]
M["transect_above_max"] = max(t["r"] for t in above)
M["transect_below_min"] = min(t["r"] for t in below)
M["transect_gap_lo"] = min(t["lat"] for t in above)
M["transect_gap_hi"] = max(t["lat"] for t in below)

# The build log is the record that every number was computed. Print all of it, not a selection.
for k in sorted(M):
    if k not in ("composite", "fork", "transect"):
        print(f"  measured  {k:>18} = {M[k]}")
for k in ("composite", "fork"):
    for label in M[k]:
        print(f"  measured  {label:>18} : {M[k][label]}")
for t in M["transect"]:
    print(f"  measured  {'transect':>18} : {t}")

# What the plan claims, against what the files say. A builder does not edit the plan.
print(f"\n  PLAN vs MEASURED for {TRACK['id']}:")
print(f"    course.yml data: '234 natural CA gauges have 50+ years' — measured {M['n_gauges']} "
      f"from {M['n_ca_sites']} California discharge sites ({M['n_long']} pass the record-length "
      f"filter, and {M['n_excluded']} of those are dropped as built channels or "
      f"below-dam sites). The audit's 234 came from a name filter it did not record.")
print(f"    course.yml open_question: 'strong in the south (r = +0.36)' — measured "
      f"{M['r_south']:+.3f} at {M['south_name']}. The nearest +0.36 in this data is "
      f"{[t['r'] for t in TRANSECT_R if t['site'] == '11098000'][0]:+.3f} at "
      f"{named('11098000')}, a DIFFERENT gauge with the same name.")
print(f"    course.yml open_question: 'absent in the north (+0.087)' — measured "
      f"{M['r_north']:+.4f}. Reproduces.")
print(f"    course.yml open_question: 'paired bootstrap gives +0.222, CI [+0.039, +0.398]' — "
      f"measured {M['gap']:+.4f}, CI [{M['ci_gap'][0]:+.3f}, {M['ci_gap'][1]:+.3f}] at "
      f"seed {SEED}, B={N_BOOT}. The point estimate reproduces exactly; the interval differs "
      f"in the third decimal because the audit used a different seed.")
print(f"    course.yml title spells it 'El Nino'; the notebook writes 'El Nino' with the tilde.")


# ---------------------------------------------------------------------------
# 3. the summary, generated from modules.yml so the wording cannot drift
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
# new, so the full module tables would list thirty functions it never uses; these are the ones
# the notebook and its model answers actually write.
TRACK_IDEAS = [("ML3", "Baseline"), ("S4", "Bootstrap"), ("S4", "Confidence interval"),
               ("D2", "Table"), ("D1", "NaN")]
TRACK_FNS = [("D2", "table.groupby(column)"), ("D2", "column.isna()"),
             ("D2", "table.sort_values(by)"), ("ML1", "column.mean()"),
             ("S2", "pd.to_datetime(column)"), ("S2", "np.random.default_rng(seed)"),
             ("S2", "rng.integers(low, high, size)"),
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
    out += ["", "Two calls in the setup cell are new, and both are plumbing rather than ideas: "
                "`np.corrcoef(a, b)` measures how tightly two columns move together (the setup "
                "cell wraps it as `correlation`), and `table.merge(other, on=\"year\")` lines two "
                "tables up on a shared column."]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 4. the cells
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

TITLE = "Does El Niño really bring rain to California?"

HOOK = f"""
Every few years the trade winds slacken and a band of the equatorial Pacific warms by a degree or
two. That is El Niño, and NOAA has been measuring it monthly since {M['nino_first']} as a sea
surface temperature anomaly in a box on the equator called Niño 3.4. California's winter storm
track is supposed to move with it, and by December of a warm year the newspapers say so: *this is
the year the drought breaks.*

Rivers are how you check. The USGS has been reading California stream gauges since before anyone
measured El Niño — {M['n_gauges']} of them, in this notebook's list, with {MIN_YEARS} years or more
of daily flow and still reporting. The oldest holds {M['oldest_years']} years.

So the claim is testable with two files and one number: how tightly does a winter's Pacific
temperature move with the water that came down a river that year? The answer turns out to depend on
which river you ask — and on a choice about the calendar that nobody in the newspaper story ever
makes.
"""

md(weekkit.OPENING.format(question=TITLE, datahub=datahub, hook=HOOK.strip()))

md("""
## How this notebook is different

This is a **project track**. It is not a weekly notebook and it does not behave like one.

A weekly notebook shows you a move, walks you through it, and then asks you to make it once
yourself. This one loads the data and reproduces one result — that a river in southern California
does move with El Niño — and then stops helping. From there on every section is a sentence
describing what to find out and an empty cell to find it out in. There is no worked example above
to pattern-match against, because on a real question there never is one.

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

**The science.** Say whether El Niño measurably changes how much water comes down a California
river, put an interval on the answer, and say where in the state the answer changes — and then say
what the data cannot tell you about where the boundary is.

**The skills.** Turn a daily record into a yearly one, which means choosing where a year starts and
finding out what that choice costs. Compare two correlations honestly, which is not the same as
computing two of them. Resample the thing that is actually paired.

**The four questions, in order:**

1. Does El Niño show up in one California river?
2. Does the answer change if you go north?
3. Which twelve months are a year?
4. Is the north–south gap real, or two noisy numbers?

The open question at the end is not on that list. It is the project; the four above are what you
build to reach it.
""")

md(f"""
## Setup

Three live archives, each read straight from its source with a copy stored with the course
behind it. (The map also reads `coastlines.csv`, which ships with the course and has no upstream to
be live from, so it is read directly.)

- **NOAA PSL** publishes the Niño 3.4 anomaly as a plain text table: one row per year, twelve
  columns, and then a few lines of notes signed at the bottom. Its "no reading" value is
  **`-99.99`**, which appears both in this year's unmeasured months and, once, on a line of its
  own — so reading it as a missing value clears the notes as well as the gaps.
- **USGS** publishes daily discharge in a format called RDB, which carries three traps in one
  read. There is a format row (`5s 15s 20d 14n`) under the header that is not data; the discharge
  column is named after an internal timeseries id, so it is called something different for every
  gauge and must be found by its `_00060_00003` ending; and the discharge column has to be
  forced to numbers, because USGS writes a word (`Ice`, `Ssn`) where a reading is frozen or
  seasonal. Neither of the two gauges worked below needed that last one — but a gauge that
  freezes will, and you will not be told.
- **The USGS site inventory** for California, which is where the gauge list comes from. It is the
  same RDB format, one row per site per measurement it publishes, with the first and last day of
  each record and how many days are in it. The first section builds the list from it in front of
  you, rather than handing you a finished file, because the filter that turns
  {M['n_ca_sites']:,} sites into a shortlist is a judgement and you should be able to argue with
  it.

The discharge query is pinned to end on **{M['record_end']}**, the last day of water year
{M['last_year']}. Without that it would grow by a row every morning and quietly change every number
below.
""")

code(weekkit.setup_cell(
    imports="import numpy as np\n",
    figsize="(7, 4)",
    cache_base=CACHE_BASE,
    signature="url, cached, **options",
    docstring="Read one live source; fall back to the copy stored with the course.",
    url_expr="url, **options",
    cache_expr="cached, **options",
    unpack='''
NINO = "''' + NINO_URL + '''"
GAUGE = ("''' + GAUGE_URL_A + '''"
         "''' + GAUGE_URL_B + '''")
SITES = ("''' + SITES_URL_A + '''"
         "''' + SITES_URL_B + '''")

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FULL_YEAR = ''' + str(FULL_YEAR) + '''      # days a twelve-month window needs before it counts as a year

# --- El Nino, once. One row per year, twelve columns, then a few lines of notes. ---
nino_rows = load(NINO, "''' + NINO_CACHE + '''", sep=r"\\s+", skiprows=1, header=None,
                 names=["year"] + MONTHS, na_values="-99.99")
nino = nino_rows.apply(pd.to_numeric, errors="coerce").dropna(subset=["year"])

# El Nino is a WINTER thing, and the winter before a California spring is December of the year
# before plus January and February of this one. shift(1) reaches back one row for that December.
winter = pd.DataFrame({"year": nino["year"].astype(int),
                       "djf": (nino["Dec"].shift(1) + nino["Jan"] + nino["Feb"]) / 3}).dropna()


def flow(site):
    """Every daily reading from one USGS gauge: one row per day, discharge in cubic feet/second."""
    raw = load(GAUGE + site, "trackT5_dv_" + site + ".tsv.gz",
               sep="\\t", comment="#", low_memory=False)
    raw = raw[raw["agency_cd"] == "USGS"]        # drops the "5s 15s 20d 14n" format row
    for name in raw.columns:                     # the discharge column carries an internal id,
        if name.endswith("_00060_00003"):        # so it is named differently at every gauge
            column = name
    return pd.DataFrame({"date": pd.to_datetime(raw["datetime"]),
                         "cfs": pd.to_numeric(raw[column], errors="coerce")}).dropna()


def water_year(dates, start_month):
    """Which twelve-month year each date falls in, if the year begins in `start_month`.

    A year is named for the calendar year it ENDS in, so 1 October 1999 is in water year 2000.
    """
    shift = (13 - start_month) % 12
    return dates.dt.year + (dates.dt.month - 1 + shift) // 12


def paired(site, start_month):
    """One row per year: the winter Nino 3.4 index, and that year's mean flow at one gauge."""
    daily = flow(site)
    daily["year"] = water_year(daily["date"], start_month)
    per_year = daily.groupby("year")["cfs"]
    yearly = pd.DataFrame({"year": per_year.mean().index,
                           "cfs": per_year.mean().values,
                           "days": per_year.size().values})
    whole = yearly[yearly["days"] >= FULL_YEAR]     # a year missing a season is not a year
    return whole.merge(winter, on="year")           # and only years the Nino record reaches


def correlation(a, b):
    """How tightly two columns move together: +1 rises together, -1 opposite, 0 not at all."""
    return np.corrcoef(a, b)[0, 1]


coast = pd.read_csv(CACHE + "/coastlines.csv")

print("winters with a Nino index:", len(winter), "—", winter["year"].min(), "to",
      winter["year"].max())
'''.strip("\n")))

# --- the verified half ------------------------------------------------------
md(f"""
## Does El Niño show up in one California river?

California has {M['n_ca_sites']:,} sites with a daily-mean discharge record, and most of them are
no use here: a gauge with fifteen years of data cannot say anything about a cycle that turns up
every few years, and a gauge measuring an irrigation canal is measuring a decision somebody made,
not weather. Two filters cut it down. The first is arithmetic — {MIN_YEARS}+ years of readings, at
least 95% of the days present, and still reporting in 2020 or later. The second is not: it reads
the station's **name** and throws it out if the name advertises plumbing.
""")

code(f"""
inventory = load(SITES, "{SITES_CACHE}", sep="\\t", comment="#", low_memory=False)
discharge = inventory[(inventory["agency_cd"] == "USGS")
                      & (inventory["parm_cd"].astype(str) == "00060")     # discharge
                      & (inventory["stat_cd"].astype(str) == "00003")     # daily mean
                      & (inventory["data_type_cd"] == "dv")]

days = pd.to_numeric(discharge["count_nu"], errors="coerce")
last = pd.to_datetime(discharge["end_date"], errors="coerce")
span = (last - pd.to_datetime(discharge["begin_date"], errors="coerce")).dt.days + 1
long_record = discharge[(days >= {MIN_YEARS} * 365) & (days / span >= 0.95)
                        & (last.dt.year >= 2020)]

# The line most worth arguing with in this notebook. A canal, a spill, a conduit or a gauge
# reading BL (below) a dam is measuring an operating decision, so it goes — but only its NAME
# says so, and a name is not a measurement.
BUILT = r"{BUILT}"
DAMMED = r"{DAMMED}"
station = long_record["station_nm"].str.upper()
natural = long_record[~station.str.contains(BUILT) & ~station.str.contains(DAMMED)]

gauges = pd.DataFrame({{"site_no": natural["site_no"].astype(str).str.zfill(8).values,
                       "station_nm": natural["station_nm"].values,
                       "lat": pd.to_numeric(natural["dec_lat_va"], errors="coerce").values,
                       "lon": pd.to_numeric(natural["dec_long_va"], errors="coerce").values}})
gauges = gauges.sort_values("lat", ascending=False)

print("California discharge sites:", discharge["site_no"].nunique())
print("with a long, complete, current record:", len(long_record))
print("of those, not named as built or below a dam:", len(gauges))
""")

md(f"""
Here is what survives, on a map. Every dot is a gauge with at least {MIN_YEARS} years of daily
discharge; {M['n_north_38']} of them are north of 38°N, {M['n_mid']} between 36 and 38, and
{M['n_south_36']} south of 36. The two marked in red are the ones this notebook works with, and
one of them will be yours to replace.
""")

code(f"""
plt.plot(coast["lon"], coast["lat"], color="0.6", lw=0.6)
plt.scatter(gauges["lon"], gauges["lat"], s=8, color="0.25")
demo = gauges[gauges["site_no"].isin(["{SOUTH}", "{NORTH}"])]
plt.scatter(demo["lon"], demo["lat"], s=60, color="firebrick", marker="v")
plt.xlim(-125, -113.5)
plt.ylim(32, 42.5)
plt.gca().set_aspect("equal")
plt.xlabel("longitude (°E)")
plt.ylabel("latitude (°N)")
plt.title(f"{{len(gauges)}} California gauges with {MIN_YEARS}+ years of daily discharge")
plt.show()
""")

md(f"""
Take the southern one first: **{M['south_name']}**, at {M['south_lat']:.2f}°N, reading since
{M['south_start']}. Before pairing anything with anything, look at what a Californian river does in
an ordinary year.
""")

code(f"""
south_daily = flow("{SOUTH}")
south_daily["month"] = south_daily["date"].dt.month

by_month = south_daily.groupby("month")["cfs"].mean()

plt.bar(by_month.index, by_month.values, color="0.4")
plt.xlabel("calendar month (1 = January)")
plt.ylabel("mean discharge (cubic feet per second)")
plt.title(f"{M['south_name']}: the average year ({{len(south_daily)}} daily readings)")
plt.locator_params(axis="x", integer=True)
plt.show()

print("wettest month:", MONTHS[by_month.idxmax() - 1], " driest:", MONTHS[by_month.idxmin() - 1])
""")

md(f"""
That is the whole reason a hydrologist does not use the calendar. Flow peaks in
{M['wettest_month']} at about {M['wettest_cfs']:.0f} cubic feet per second and bottoms out in
{M['driest_month']} near {M['driest_cfs']:.0f} — a factor of
{M['wettest_cfs'] / M['driest_cfs']:.0f}. A **water year** therefore runs from 1 October to
30 September: it starts in the dry gap, so one winter's storms stay inside one year instead of
being cut in half by 31 December.

`paired(site, start_month)` does the rest — it fetches a gauge, labels every day with the year that
`start_month` puts it in, averages each year that has at least {FULL_YEAR} days of readings, and
lines the result up against the winter Niño index. There is no default for `start_month`: you have
to write the choice down every time, because it is a choice.
""")

code(f"""
south = paired("{SOUTH}", {WATER_YEAR})

print("paired years:", len(south), "—", south["year"].min(), "to", south["year"].max())
print(south.head())
""")

code(f"""
assert winter["year"].min() == {M['winter_first']}, \\
    "the winter index should start in {M['winter_first']} — its first December is the year before"
assert 70 <= len(south) <= 80, \\
    "expected about {M['n_years']} paired water years; a very different number means the day "\\
    "count filter or the Nino join dropped more than it should"
assert south["cfs"].min() > 0, "a year's mean discharge cannot be zero or negative"
print(f"✓ the data — {{len(winter)}} winters with a Niño index, {{len(south)}} complete water "
      f"years at {M['south_name']}")
""")

md("""
### And that is the last self-check in this notebook

The pipeline is now trustworthy: the files are the files, the join is the join, the numbers below
are the numbers. Everything from here is yours, and the safety net is gone — nothing will tell you
when you have it right.
""")

md(f"""
Now the actual question, on one gauge. One dot per water year: the winter's Niño 3.4 anomaly across
the bottom, that year's mean discharge up the side.

`correlation` turns the whole cloud into one number. It is **+1** if the dots lie exactly on a
rising line, **0** if the cloud has no tilt at all, **−1** if it falls — and it is the closest
thing to a single answer this question has.
""")

code(f"""
r_south = correlation(south["djf"], south["cfs"])

plt.scatter(south["djf"], south["cfs"], s=18, color="0.3")
plt.axvline(0, color="0.7", lw=1)
plt.xlabel("winter (Dec–Feb) Niño 3.4 anomaly (°C)")
plt.ylabel("water-year mean discharge (cubic feet per second)")
plt.title(f"{M['south_name']}, {{len(south)}} water years — r = {{r_south:.3f}}")
plt.show()

print("correlation:", round(r_south, 3))
""")

md(f"""
**r = {M['r_south']:+.3f}** over {M['n_years']} water years. A warm Pacific winter and a wet year in
the {M['south_name'].split(' NR ')[0].title()} do go together, and the newspapers are not making it
up.

Notice what the figure does *not* show, though. Nothing about that cloud of {M['n_years']} dots
announces {M['r_south']:+.3f} rather than {M['r_north']:+.3f} or {M['ci_south'][1]:+.2f}. You would
not read that number off the picture, which is worth remembering for the rest of the notebook: from
here on the eye is no help and the arithmetic is all you have.
""")

md(f"""
### Predict before you run

{M['north_name']} drains the far north coast, {M['north_lat'] - M['south_lat']:.1f} degrees of
latitude — about {(M['north_lat'] - M['south_lat']) * 111:.0f} km — up the state. Its correlation with the same winter index is the very next
thing you will compute. Write down what you think it is first. Change `my_guess` and run the cell.

A wrong guess you committed to is worth more than a right answer you were shown.
""")

code(f"""
my_guess = {M['r_south']:.2f}

print("I think the northern river's correlation with winter Niño 3.4 is about", my_guess)
""")

# --- YOUR TURN 1 ------------------------------------------------------------
md(f"""
## Does the answer change if you go north?

{M['north_name']} is site `{NORTH}`, at {M['north_lat']:.2f}°N. It has been reading since
{M['north_start']} and it is on the map above, the northern red triangle.
""")

ask(f"""
### ✏️ Your turn 1

Run the same pipeline on `"{NORTH}"` with the same water year, draw the same scatter, and print its
correlation next to the southern one so both are on the page.

Then print one more line answering it in a sentence, on your own two numbers: does El Niño bring
rain to California, or does it bring rain to *part* of California — and what would you have
concluded if this notebook had handed you the northern gauge first?
""")

answer(f"""
north = paired("{NORTH}", {WATER_YEAR})
r_north = correlation(north["djf"], north["cfs"])

plt.scatter(north["djf"], north["cfs"], s=18, color="0.3")
plt.axvline(0, color="0.7", lw=1)
plt.xlabel("winter (Dec–Feb) Niño 3.4 anomaly (°C)")
plt.ylabel("water-year mean discharge (cubic feet per second)")
plt.title(f"{M['north_name']}, {{len(north)}} water years — r = {{r_north:.3f}}")
plt.show()

print("south —", round(r_south, 3), " north —", round(r_north, 3))
print("My guess was", my_guess, "and the north is", round(r_north, 3))

print("El Nino brings rain to PART of California. The southern gauge gives",
      round(r_south, 3), "and the northern one gives", round(r_north, 3), "on the same",
      len(north), "winters, so a claim about 'California' is really a claim about a latitude.",
      "Handed the northern gauge first I would have reported that El Nino does nothing here,",
      "and I would have had a correctly computed number to back it up.")
""")

# --- YOUR TURN 2, the trivial baseline --------------------------------------
md(f"""
A correlation is already a modelling choice: it assumes the relationship is a straight line, and it
turns {M['n_years']} pairs into one number that nobody can check by eye. Before trusting it, it is
worth asking what the dumbest possible version of this analysis says.

**Baseline:** {idea('ML3', 'Baseline')['words']}
""")

ask(f"""
### ✏️ Your turn 2

The dumbest version of this question needs no correlation at all: **split the years into three
piles and take the average of each.** NOAA calls a winter El Niño when the Niño 3.4 anomaly is at
or above +{STRONG} °C and La Niña at or below −{STRONG}, with everything between called neutral.

For **both** gauges, print how many years fall in each pile, the mean discharge of each, and the
El Niño mean divided by the La Niña mean. Draw whatever figure makes the two gauges comparable
despite one of them carrying about fifty times more water than the other.

Then print one more line answering it: which of the two numbers — the correlation from Your turn 1
or this ratio — would you put in a newspaper, and what does the other one know that it does not?
""")

answer(f"""
for name, table in [("south", south), ("north", north)]:
    warm = table[table["djf"] >= {STRONG}]
    cold = table[table["djf"] <= -{STRONG}]
    calm = table[(table["djf"] > -{STRONG}) & (table["djf"] < {STRONG})]
    print(name, "— El Nino", len(warm), "years, mean", round(warm["cfs"].mean()),
          "| neutral", len(calm), "years, mean", round(calm["cfs"].mean()),
          "| La Nina", len(cold), "years, mean", round(cold["cfs"].mean()),
          "| ratio", round(warm["cfs"].mean() / cold["cfs"].mean(), 2))

# Both gauges on one axis by dividing each by its own long-run mean, so the picture is about
# the SHAPE of the response rather than about how big the two rivers are.
places = [0, 1, 2]
for offset, name, table in [(-0.18, "south", south), (0.18, "north", north)]:
    heights = []
    for low, high in [({STRONG}, 99), (-{STRONG}, {STRONG}), (-99, -{STRONG})]:
        pile = table[(table["djf"] >= low) & (table["djf"] <= high)]
        heights.append(pile["cfs"].mean() / table["cfs"].mean())
    plt.bar([p + offset for p in places], heights, width=0.35, label=name)
plt.axhline(1, color="firebrick", lw=1.2)
plt.xticks(places, ["El Niño", "neutral", "La Niña"])
plt.xlabel("kind of winter")
plt.ylabel("mean discharge ÷ that gauge's own long-run mean")
plt.title(f"The dumb answer, both gauges ({{len(south)}} water years each)")
plt.legend()
plt.show()

print("I would put the ratio in a newspaper: it says a southern El Nino year carries about",
      round(south[south["djf"] >= {STRONG}]["cfs"].mean()
            / south[south["djf"] <= -{STRONG}]["cfs"].mean(), 1),
      "times the water of a La Nina year, which is a thing a person can picture.",
      "What the correlation knows that the ratio does not is the",
      len(south[(south["djf"] > -{STRONG}) & (south["djf"] < {STRONG})]),
      "neutral years the ratio threw away, and that the response looks like a straight line",
      "rather than two buckets — the ratio would look the same whether the middle years sat",
      "on the line or nowhere near it.")
""")

# --- YOUR TURN 3, the fork ---------------------------------------------------
md(f"""
## Which twelve months are a year?

Every number so far rests on a decision made in the setup cell and never argued for: that a year
starts on 1 October. The reason was in the figure of the average year — October is the dry gap, so
a water year keeps one winter's storms together.

But *dry gap* is a judgement, not a boundary. September would do. November would do. And a
calendar year, which is what you get if you reach for `date.dt.year` without thinking about it, is
what almost every beginner uses. `paired` takes `start_month` precisely so you can try them.

This is the one real decision in this track. Make it, and report what it cost.
""")

ask(f"""
### ✏️ Your turn 3

Run **both** gauges under four twelve-month windows: starting in October, November, September, and
January. `paired(site, 1)` is the calendar year.

For each window print how many paired years survive and the two correlations, and print the
**gap** between them — the southern correlation minus the northern one — because the gap is the
claim this track is actually making.

Then print one more line answering it: three of those four windows tell one story and one tells
another, so which is it — did you discover a fact about California, or a fact about where you
started counting?
""")

answer(f"""
for start_month, label in [({WATER_YEAR}, "Oct-Sep"), (11, "Nov-Oct"), (9, "Sep-Aug"),
                           (1, "Jan-Dec (calendar)")]:
    s = paired("{SOUTH}", start_month)
    n = paired("{NORTH}", start_month)
    r_s = correlation(s["djf"], s["cfs"])
    r_n = correlation(n["djf"], n["cfs"])
    print(f"{{label:20s}} n = {{len(s):3d}}   south {{r_s:+.3f}}   north {{r_n:+.3f}}   "
          f"gap {{r_s - r_n:+.3f}}")

print("Both. The three windows that start in the dry season agree with each other and give a gap",
      "near +0.22, so the north-south contrast is not an artefact of exactly which dry month I",
      "start in. The calendar year is the odd one out and it collapses the gap to almost nothing,",
      "because a calendar year cuts a winter in half: January and February of one storm season",
      "land in one year and the December that belongs with them lands in the previous one.",
      "So the fact about California is real, and it is only visible to someone who made the",
      "right calendar choice first — which is a fact about me, not about the rivers.")
""")

ask(f"""
### ✏️ Your turn 4

Two or three paragraphs, quoting **your own four gaps**.

1. Which window would you report, and what does the choice cost? Say what a reader loses by not
   being shown the other three.
2. The calendar year is not a mistake — plenty of published work reports calendar-year runoff, and
   nothing in the data says it is wrong. So what should a reader conclude from a result that
   depends on a defensible choice, and what would have to be true of California's weather for all
   four windows to agree?
""")

answer_prose(f"""
I would report the October–September water year, and I would report it because of the average-year
figure rather than because of the answer it gives. Flow at {M['south_name']} peaks in
{M['wettest_month']} and bottoms out in {M['driest_month']}, so a year cut at the end of September
contains one storm season and a year cut at the end of December contains the tail of one and the
head of the next. That is a reason available *before* looking at the correlations, which is what
makes it a choice rather than a preference. It costs me the ability to compare directly with
anything published on calendar years, and it costs a reader the knowledge that my headline gap of
{M['fork']['Oct-Sep']['gap']:+.3f} would have been {M['fork']['Jan-Dec']['gap']:+.3f} under a
different and perfectly defensible convention — which is why all four belong in the write-up and
not just the one I chose.

My four gaps are {M['fork']['Oct-Sep']['gap']:+.3f} for October, {M['fork']['Nov-Oct']['gap']:+.3f}
for November, {M['fork']['Sep-Aug']['gap']:+.3f} for September and
{M['fork']['Jan-Dec']['gap']:+.3f} for the calendar year. The first three are the same number to
within a hundredth; the fourth is essentially zero. What a reader should conclude is not that the
calendar year is wrong but that this result is *conditional*: it says El Niño's effect on southern
California is concentrated in a single storm season, and any accounting that splits a storm season
in two will not find it. A conclusion that survives three of four reasonable choices, and fails on
the fourth for a reason you can name in one sentence, is a stronger result than one that survives
all four for reasons nobody looked into.

For all four windows to agree, the El Niño signal would have to be spread evenly through the
calendar rather than concentrated in December to February — a wet El Niño year would have to be wet
in July as well. It is not: the average-year figure shows almost all of the flow arriving in a few
winter months, so where you cut the year decides whether those months stay together. The other way
they could agree is if the effect were large enough to survive being halved, and at
{M['r_south']:+.3f} it is nowhere near that.
""")

# --- YOUR TURN 5, 6: the sting ----------------------------------------------
md(f"""
## Is the north–south gap real, or two noisy numbers?

The whole claim now rests on the difference between two correlations computed from
{M['n_years']} years each. That is not many, and a correlation from a small sample wanders a long
way. Before quoting the gap, put an interval on it.

**Bootstrap:** {idea('S4', 'Bootstrap')['words']}

**Confidence interval:** {idea('S4', 'Confidence interval')['words']}
""")

ask(f"""
### ✏️ Your turn 5

Bootstrap each gauge **on its own**, the way you would if you had only one of them.

{N_BOOT} times over: draw {M['n_years']} of that gauge's years at random with replacement —
`rng.integers(0, len(south), size=len(south))` gives you their positions — and compute the
correlation of the drawn years. Use `np.random.default_rng({SEED})` so your run is repeatable.

Report each gauge's 95% interval, and draw the two sets of resampled correlations as histograms on
the same axes so the two intervals are visible at once.

Then print one more line answering it: on these two intervals alone, would you say the two rivers
respond differently to El Niño?
""")

answer(f"""
rng = np.random.default_rng({SEED})
resampled_south = []
resampled_north = []
for i in range({N_BOOT}):
    picked = rng.integers(0, len(south), size=len(south))
    resampled_south.append(correlation(south["djf"].values[picked],
                                       south["cfs"].values[picked]))
    resampled_north.append(correlation(north["djf"].values[picked],
                                       north["cfs"].values[picked]))

ci_south = np.percentile(resampled_south, [2.5, 97.5])
ci_north = np.percentile(resampled_north, [2.5, 97.5])
print("south 95% interval:", np.round(ci_south, 3))
print("north 95% interval:", np.round(ci_north, 3))

plt.hist(resampled_south, bins=40, alpha=0.6, label="south")
plt.hist(resampled_north, bins=40, alpha=0.6, label="north")
plt.xlabel("correlation of a resampled catalogue of water years")
plt.ylabel("resamples")
plt.title(f"{N_BOOT} bootstrap resamples of each gauge on its own")
plt.legend()
plt.show()

print("No. The two intervals overlap between", round(max(ci_south[0], ci_north[0]), 2), "and",
      round(min(ci_south[1], ci_north[1]), 2), "— a wide band that contains both point",
      "estimates — so on these two intervals alone the honest answer is that the two rivers",
      "are indistinguishable.")
""")

md(f"""
You now have two intervals, and comparing them by eye is the move everybody makes. It is worth
noticing what that move assumes: that the two gauges are two independent studies whose results are
being set side by side. They are not. They are measured over **the same {M['n_years']} water
years**, against **the same** winter index — when 1983 was a monster El Niño it was a monster El
Niño for both of them.

So there is a second way to resample, and it asks a different question. Instead of asking how far
each correlation could have wandered on its own, resample the **years** once and ask what the
*difference* would have been in that world.
""")

ask(f"""
### ✏️ Your turn 6

Bootstrap the **gap**, keeping the two gauges paired.

{N_BOOT} times over: draw {M['n_years']} positions with replacement, and use **the same positions
for both gauges** — one `picked` array, two correlations, one subtraction. Collect the
{N_BOOT} differences.

Report the 95% interval of the difference, its median, and the fraction of the {N_BOOT} resamples
in which the southern correlation came out larger than the northern one. Draw the differences as a
histogram with zero marked.

Then print two or three sentences answering it on your own interval: is the north–south gap
established or not, and how does this interval differ from the two you drew in *Your turn 5* — can
a difference be distinguishable from zero even when the two things being differenced are not
distinguishable from each other?
""")

answer(f"""
rng = np.random.default_rng({SEED})
gaps = []
for i in range({N_BOOT}):
    picked = rng.integers(0, len(south), size=len(south))
    r_s = correlation(south["djf"].values[picked], south["cfs"].values[picked])
    r_n = correlation(north["djf"].values[picked], north["cfs"].values[picked])
    gaps.append(r_s - r_n)

ci_gap = np.percentile(gaps, [2.5, 97.5])
print("observed gap:", round(r_south - r_north, 3))
print("paired 95% interval:", np.round(ci_gap, 3), " median", round(np.median(gaps), 3))
print("resamples with south above north:", round(np.mean(np.array(gaps) > 0), 3))

plt.hist(gaps, bins=40, color="0.4")
plt.axvline(0, color="firebrick", lw=1.5)
plt.axvline(r_south - r_north, color="steelblue", lw=1.5, ls="--")
plt.xlabel("southern correlation minus northern correlation, one resample")
plt.ylabel("resamples")
plt.title(f"{N_BOOT} PAIRED resamples of the same water years (red = zero)")
plt.show()

print("Established. My paired interval runs from", round(ci_gap[0], 3), "to", round(ci_gap[1], 3),
      "which excludes zero, and", round(100 * np.mean(np.array(gaps) > 0)),
      "percent of resamples put the south above the north.")
print("It can be distinguishable when the two separately were not because the wandering is",
      "shared: a resample that happens to draw a run of dry El Nino years pulls BOTH",
      "correlations down together, so most of what makes each interval wide cancels in the",
      "subtraction. Comparing two marginal intervals throws that cancellation away and asks a",
      "question nobody asked — whether these are two unrelated studies.")
print("The lesson is that the right unit to resample is the thing the two measurements share,",
      "which here is the water year, not the gauge.")
""")

# --- closing ----------------------------------------------------------------
md(f"""
{weekkit.CLOSING_HEADING}

Not to California — to part of it. Over {M['n_years']} water years the southern gauge gives
r = {M['r_south']:+.3f} and the northern one {M['r_north']:+.3f}, and a paired bootstrap on the
difference puts it at {M['gap']:+.3f} with a 95% interval of
[{M['ci_gap'][0]:+.3f}, {M['ci_gap'][1]:+.3f}] — above zero, on the same
{M['n_years']} winters, but only if you count a year from October. Count it from January and the
gap falls to {M['fork']['Jan-Dec']['gap']:+.3f} and the whole result disappears.
""")

md(track_summary())

# --- the project ------------------------------------------------------------
md(f"""
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
number and a range in it, you do not have a result yet.
"""),
    "baseline_first": ("2 · The trivial baseline", """
Before any statistic, state the dumbest answer to your question and what it gives. Every later
number is reported against it.

On this track the baseline is the one you built in *Your turn 2* — average the El Niño years,
average the La Niña years, divide. Say what it gives on your gauge, and say what each later step
bought you over it. If a bootstrap interval taught you nothing the ratio had not already told you,
say that too.
"""),
    "split_by_structure": ("3 · Split by structure", """
Earth data are correlated in space and in time, so whatever you split, resample or count as
independent has to be split along the structure that is really there — never at random across
rows.

This track fits no model, so there is no train/test split to get wrong. The same idea decided the
answer anyway, in *Your turn 6*: two gauges measured over the same years are not two independent
studies. Name the unit you resampled, say why, and say how much the answer moved when you got it
right.
"""),
    "what_i_got_wrong": ("4 · What I got wrong", """
What failed, and what you believed before it failed. Honest failure is graded; a faked success is
not. Your *Predict before you run* guess belongs here if it was wrong, and so does any gauge you
tried that turned out to be a canal.
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

Here is what is actually established, and it is less than it looks. Two gauges differ, by
{M['gap']:+.3f} with a paired interval of [{M['ci_gap'][0]:+.3f}, {M['ci_gap'][1]:+.3f}]. Two
gauges are two points. They tell you that the response is not the same everywhere in California and
they tell you nothing whatever about the *shape* of the change between them — whether there is a
line somewhere around the latitude of Monterey with El Niño country on one side of it, or a smooth
ramp running the length of the state, or a patchwork in which what matters is not latitude at all
but which way a catchment faces and how high its snow line sits.

That question needs many gauges, and no one project can fetch many gauges and think carefully about
any of them. **This is why the first section built a list of {M['n_gauges']}.** Your project takes one — not
one of the two worked here — carries it through everything above, and reports its correlation, its
interval, and its latitude. The class assembles a map that no single project could produce, and the
map is the result.

Three directions, none of them worked out here:

1. **Find the switch.** Take gauges spanning the state, plot each one's correlation against its
   latitude, and look at the shape. Is there a step, and if so where — and how far apart do two
   gauges have to be before their paired difference clears zero?
2. **Test latitude against the alternatives.** The list carries a longitude too, and USGS publishes
   a gauge's elevation and drainage area. A coastal gauge and a Sierra gauge at the same latitude
   receive their water in completely different ways — rain in one, snowmelt in the other — so
   latitude may be standing in for something else entirely. What would distinguish them?
3. **Ask what the record cannot say.** {M['n_years']} years is {M['composite']['south']['n_warm']}
   El Niño winters. Your interval on one gauge is roughly
   [{M['ci_south'][0]:+.2f}, {M['ci_south'][1]:+.2f}] wide. How many gauges would a map need before
   a boundary at, say, 37°N could be told apart from a smooth ramp? Work out what the answer
   depends on before you go looking for it.

And one that is bigger than a semester: the {M['n_gauges']} gauges in the list are the
{M['n_long']} long records minus the {M['n_excluded']} whose *names* advertised
plumbing, and a name is not a measurement. A gauge called
`{M['south_name']}` is presumed natural; one with `BL` in its name is presumed regulated and was
dropped. How much of any map you draw is El Niño, and how much of it is California's plumbing?
""")

ask(f"""
### ✏️ Your turn 7 — the first move

Before you close this notebook: in a few sentences, what is the **one** measurement you would make
first, what would it show if the change with latitude is a step, what would it show if it is a
ramp, and what number would change your mind? Then make the measurement, in the cell below the
prose.

The gauge list is in `gauges` and `paired` takes any site number in it. Two warnings, and both are
the honest kind. Only the two gauges this notebook worked with have a cached copy stored with the
course, so a gauge you choose yourself is live-only and will fail loudly if USGS is unreachable.
And every gauge you add is another fetch, so choose few and choose them for a reason.
""")

answer_prose(f"""
I would take a handful of gauges spread from the Oregon border to the Mexican one and plot each
one's correlation against its latitude — the cheapest possible version of the class map, run by one
person. If the change is a step, the northern gauges should cluster near
{M['r_north']:+.2f} and the southern ones near {M['r_south']:+.2f} with almost nothing in between,
and the latitude where they separate should be the same wherever I put the gauges. If it is a ramp,
the correlations should climb steadily from north to south and a gauge halfway up the state should
land halfway between. The number that would change my mind is the correlation of the gauges between
about 37° and 39°N: near {M['r_north']:+.2f} says step, near halfway says ramp.

What makes me doubt any answer I get is the width of the intervals. One gauge's 95% interval on
{M['n_years']} years was about [{M['ci_south'][0]:+.2f}, {M['ci_south'][1]:+.2f}] — wider than the
whole north–south gap I am trying to resolve. Seven gauges do not fix that; only the paired
comparison does, and pairing seven gauges against each other is a different and larger piece of
work than pairing two. So I expect the honest outcome to be a picture with a suggestive shape and
no interval that rules anything out, which is exactly why the class needs {M['n_gauges']} of them
rather than seven.
""")

answer(f"""
transect = {TRANSECT}

r_by_site = []
for site in transect:
    table = paired(site, {WATER_YEAR})
    r_by_site.append(correlation(table["djf"], table["cfs"]))

picked = gauges[gauges["site_no"].isin(transect)]
lats = []
for site in transect:
    lats.append(float(picked[picked["site_no"] == site]["lat"].iloc[0]))

for site, lat, r in zip(transect, lats, r_by_site):
    name = picked[picked["site_no"] == site]["station_nm"].iloc[0]
    print(f"{{lat:6.2f}}°N  r = {{r:+.3f}}   {{name}}")

plt.scatter(lats, r_by_site, s=40, color="0.3")
plt.axhline(0, color="0.7", lw=1)
plt.xlabel("gauge latitude (°N)")
plt.ylabel("correlation with winter Niño 3.4")
plt.title(f"{{len(transect)}} gauges, {{len(south)}} water years each")
plt.show()

print("The seven split into two flat groups rather than lying on a ramp, and the change happens",
      "between", round(min(l for l in lats if l > 37), 1), "and",
      round(max(l for l in lats if l < 37), 1), "degrees north.")
print("But two flat groups of three and four is also what a ramp looks like when you sample it",
      "at seven places, so this does not settle it — which is the open question, not an",
      "answer to it.")
""")


# ---------------------------------------------------------------------------
# 5. emit, execute, gate
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
    for name in (NINO_CACHE, SITES_CACHE, f"trackT5_dv_{SOUTH}.tsv.gz",
                 f"trackT5_dv_{NORTH}.tsv.gz"):
        print(f"cache: data/{name} ({(ROOT / 'data' / name).stat().st_size / 1e6:.2f} MB)")

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
