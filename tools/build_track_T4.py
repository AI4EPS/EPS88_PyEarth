#!/usr/bin/env python
"""Build project track T4 — "Are we finding more Earth-like planets, or just better telescopes?"

Emits both notebooks from one source so they cannot drift:

    docs/notebooks/T4_more_planets_or_better_telescopes_solution.ipynb   executed, outputs saved
    docs/notebooks/T4_more_planets_or_better_telescopes.ipynb            the answers deleted

It also writes the track's two cached fallbacks, data/trackT4_ps.csv and data/trackT4_pscomppars.csv.

A TRACK is not a week (course.yml `project: track_notebooks:`). Two things differ, and both are
deliberate:

  * LESS HELP. No WATCH cell and no worked example before a question. The notebook loads the data
    and reproduces the ONE result the title names — the collapse of the typical planet radius —
    so a student can trust the pipeline, and then stops helping. Everything after is a prompt in
    words and an empty cell.
  * IT DOES NOT CLOSE. There is exactly one self-check, on the load, and the notebook ends on an
    open question this course cannot answer.

THE TABLE IS THE TRAP. The NASA Exoplanet Archive serves two one-row-per-planet tables and both
return 6,354 rows today, which is how they get confused. `ps where default_flag=1` carries what
somebody measured and leaves a hole where nobody did; `pscomppars` fills those holes with values
derived from a model. Every number in this track comes from `ps`. `pscomppars` is loaded too, and
only so that Your turn 2 can show what the difference costs.

Every number that appears in prose or in a model answer is computed HERE, from the same files the
notebook reads, and formatted in. Nothing is typed from memory or copied from the plan.

    python tools/build_track_T4.py
"""
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
SLUG = "T4_more_planets_or_better_telescopes"

course = yaml.safe_load((ROOT / "course.yml").read_text())
modules = yaml.safe_load((ROOT / "modules.yml").read_text())
TRACK = next(t for t in course["project"]["tracks"] if t["id"] == "T4")
PLATFORM = course["platform"]
CACHE_BASE = PLATFORM["cache_base"]

# The live source, pinned here so the cached CSVs, the notebook and the prose cannot drift.
# The column list is shared by both tables so the two reads are one call shape, not two.
# Written exactly as the notebook's setup cell writes it, and sliced from there, so the cache this
# script fills and the URL a student's kernel calls cannot become two different queries.
ARCHIVE = ('ARCHIVE = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+"\n'
           '           "pl_name,pl_rade,pl_radeerr1,pl_bmasse,pl_bmassprov,pl_orbsmax,st_rad,'
           'sy_dist,"\n'
           '           "disc_year,discoverymethod,disc_facility+from+")')
PREFIX = "".join(re.findall(r'"([^"]*)"', ARCHIVE))
PS_URL = PREFIX + "ps+where+default_flag=1&format=csv"
PC_URL = PREFIX + "pscomppars&format=csv"
PS_CACHE = "trackT4_ps.csv"
PC_CACHE = "trackT4_pscomppars.csv"

EARLY = (2016, 2020)         # the two eras Your turn 1 compares
LATE = (2021, 2026)
SAWTOOTH_FROM = 2014         # the first year the yearly median is built on hundreds of planets
ROCKY = 1.6                  # radii below this are rocky; the course's week-2 threshold
SEED = 88                    # the course number, fixed before anything was run
N_BOOT = 2000                # bootstrap resamples

# IAU DEFINED values, not measurements read off a web page — which is why they carry a resolution
# number rather than a read-date. The nominal solar radius (6.957e8 m) and the nominal Earth
# equatorial radius (6.3781e6 m) are IAU 2015 Resolution B3; the astronomical unit is exactly
# 1.495978707e11 m, IAU 2012 Resolution B2. Two ratios follow: one converts a stellar radius in
# solar radii into AU (transit probability), the other into Earth radii (transit depth).
R_SUN_M = 6.957e8
R_EARTH_M = 6.3781e6
AU_M = 1.495978707e11
R_SUN_IN_AU = R_SUN_M / AU_M
R_SUN_IN_REARTH = R_SUN_M / R_EARTH_M

MEASURED_SURVEYS = ["Kepler", "K2", "Transiting Exoplanet Survey Satellite (TESS)"]


# ---------------------------------------------------------------------------
# 1. measure everything the notebook will say
# ---------------------------------------------------------------------------
def fetch(url, name):
    """Run the live query once, cache it beside the course, and return the cached copy.

    The archive fails or stalls on roughly one request in four and `pd.read_csv` has no timeout,
    so the build retries rather than hanging a student's kernel. The notebook itself reads the
    cache through the standard try/except.
    """
    out = ROOT / "data" / name
    if not out.exists():
        for attempt in range(6):
            try:
                pd.read_csv(url).to_csv(out, index=False)
                break
            except Exception as e:                       # the archive 503s and stalls; retry
                print(f"  fetch {name} attempt {attempt}: {type(e).__name__}")
                time.sleep(5)
        else:
            sys.exit(f"the NASA Exoplanet Archive did not answer for {name}")
    return pd.read_csv(out)


def transit_probability(frame):
    """The chance a randomly oriented orbit shows a transit: the star's radius over the orbit."""
    return frame["st_rad"] * R_SUN_IN_AU / frame["pl_orbsmax"]


planets = fetch(PS_URL, PS_CACHE)
composite = fetch(PC_URL, PC_CACHE)

M = {}
M["n_ps"] = len(planets)
M["n_pc"] = len(composite)
M["n_cols"] = planets.shape[1]
M["has_radius"] = int(planets["pl_rade"].count())
M["no_radius"] = int(planets["pl_rade"].isna().sum())
M["radius_pct"] = M["has_radius"] / M["n_ps"] * 100
M["pc_radius_pct"] = composite["pl_rade"].count() / M["n_pc"] * 100
M["dup_names"] = int(planets["pl_name"].duplicated().sum())

with_radius = planets[planets["pl_rade"].notna()]

# --- the reproduced result: the collapse -----------------------------------
by_year = with_radius.groupby("disc_year")["pl_rade"].median()
by_year_n = with_radius.groupby("disc_year")["pl_rade"].count()
M["year_first"] = int(by_year.index.min())
M["year_last"] = int(by_year.index.max())
old = with_radius[with_radius["disc_year"] <= 2010]
new = with_radius[with_radius["disc_year"] >= EARLY[0]]
M["old_n"], M["old_med"] = len(old), float(old["pl_rade"].median())
M["new_n"], M["new_med"] = len(new), float(new["pl_rade"].median())
M["collapse"] = M["old_med"] / M["new_med"]

saw = by_year.loc[SAWTOOTH_FROM:]
M["saw_sigma"] = float(saw.std())
M["saw_swing"] = float(saw.max() - saw.min())
M["saw_years"] = {int(y): round(float(saw[y]), 2) for y in (2015, 2016, 2017, 2025) if y in saw}

# --- Your turn 1: the eras and the bootstrap -------------------------------
early = with_radius[with_radius["disc_year"].between(*EARLY)]["pl_rade"].values
late = with_radius[with_radius["disc_year"].between(*LATE)]["pl_rade"].values
M["early_n"], M["early_med"] = len(early), float(np.median(early))
M["late_n"], M["late_med"] = len(late), float(np.median(late))
M["rebound"] = M["late_med"] - M["early_med"]

rng = np.random.default_rng(SEED)
diffs = []
for i in range(N_BOOT):
    diffs.append(np.median(rng.choice(late, size=len(late), replace=True))
                 - np.median(rng.choice(early, size=len(early), replace=True)))
diffs = np.array(diffs)
M["boot_lo"], M["boot_hi"] = [float(x) for x in np.percentile(diffs, [2.5, 97.5])]
M["boot_below_zero"] = float((diffs <= 0).mean())
M["rebound_vs_sawtooth"] = M["rebound"] / M["saw_sigma"]

# --- Your turn 2: who has a radius at all ----------------------------------
rv = planets[planets["discoverymethod"] == "Radial Velocity"]
tr = planets[planets["discoverymethod"] == "Transit"]
M["rv_n"] = len(rv)
M["rv_radius"] = int(rv["pl_rade"].count())
M["rv_radius_med"] = float(rv["pl_rade"].median())
M["rv_no_errorbar"] = int((rv["pl_rade"].notna() & rv["pl_radeerr1"].isna()).sum())
M["tr_n"] = len(tr)
M["tr_radius"] = int(tr["pl_rade"].count())
M["tr_radius_med"] = float(tr["pl_rade"].median())

pc_rv = composite[composite["discoverymethod"] == "Radial Velocity"]
M["pc_rv_radius"] = int(pc_rv["pl_rade"].count())
M["pc_rv_med"] = float(pc_rv["pl_rade"].median())
M["pc_rv_no_errorbar"] = int((pc_rv["pl_rade"].notna() & pc_rv["pl_radeerr1"].isna()).sum())
M["pc_rv_distinct"] = int(pc_rv["pl_rade"].dropna().nunique())

prov = planets["pl_bmassprov"].value_counts()
M["prov_mass"] = int(prov.get("Mass", 0))
M["prov_msini"] = int(prov.get("Msini", 0)) + int(prov.get("Msin(i)/sin(i)", 0))
M["prov_none"] = int(planets["pl_bmassprov"].isna().sum())
rv_prov = rv["pl_bmassprov"].value_counts()
M["rv_msini"] = int(rv_prov.get("Msini", 0)) + int(rv_prov.get("Msin(i)/sin(i)", 0))
M["rv_true_mass"] = int(rv_prov.get("Mass", 0))

# --- Your turn 3: the mass-radius line -------------------------------------
train = planets[(planets["pl_bmassprov"] == "Mass") & planets["pl_rade"].notna()
                & planets["pl_bmasse"].notna()]
M["fit_n"] = len(train)
x_train = np.log10(train["pl_bmasse"].values).reshape(-1, 1)
y_train = np.log10(train["pl_rade"].values)
line = LinearRegression().fit(x_train, y_train)
M["fit_slope"] = float(line.coef_[0])
M["fit_intercept"] = float(line.intercept_)
M["fit_r2"] = float(line.score(x_train, y_train))

guessing = rv[rv["pl_rade"].isna() & rv["pl_bmasse"].notna()]
M["guess_n"] = len(guessing)
guessed = 10 ** line.predict(np.log10(guessing["pl_bmasse"].values).reshape(-1, 1))
M["guess_med"] = float(np.median(guessed))
M["guess_ratio"] = M["guess_med"] / M["tr_radius_med"]
M["real_ratio"] = M["rv_radius_med"] / M["tr_radius_med"]
M["rv_mass_med"] = float(rv["pl_bmasse"].median())
M["tr_mass_med"] = float(tr["pl_bmasse"].median())
M["mass_ratio"] = M["rv_mass_med"] / M["tr_mass_med"]
M["train_lo"], M["train_hi"] = float(train["pl_bmasse"].min()), float(train["pl_bmasse"].max())
M["guess_outside"] = int(((guessing["pl_bmasse"] < M["train_lo"])
                          | (guessing["pl_bmasse"] > M["train_hi"])).sum())

# --- Your turn 4: the surveys ----------------------------------------------
SURVEYS = {}
for short, full in zip(["Kepler", "K2", "TESS"], MEASURED_SURVEYS):
    rows = tr[tr["disc_facility"] == full]
    SURVEYS[short] = {"n": len(rows),
                      "radius": float(rows["pl_rade"].median()),
                      "orbit": float(rows["pl_orbsmax"].median()),
                      "distance": float(rows["sy_dist"].median()),
                      "rocky": float((rows["pl_rade"] < ROCKY).mean())}
M["surveys"] = SURVEYS

# --- Your turn 5 and 6: transit probability --------------------------------
geo = tr[tr["st_rad"].notna() & tr["pl_orbsmax"].notna() & tr["pl_rade"].notna()].copy()
geo["p"] = transit_probability(geo)
M["geo_n"] = len(geo)
M["geo_of"] = M["tr_n"]
M["p_med"] = float(geo["p"].median())
M["p_min"], M["p_max"] = float(geo["p"].min()), float(geo["p"].max())
M["stands_for"] = 1 / M["p_med"]
M["weight_sum"] = float((1 / geo["p"]).sum())
# The two illustrations in the section-4 prose, from the formula and the constants just given.
M["earth_odds"] = 1 / R_SUN_IN_AU                                 # Earth at 1 AU round the Sun
M["hot_odds"] = 1 / (R_SUN_IN_AU / (3.0 / 365.25) ** (2 / 3))     # a 3-day orbit, Sun-like star

WEIGHTED = {}
for label, frame in (("all transits", geo),
                     ("Kepler", geo[geo["disc_facility"] == MEASURED_SURVEYS[0]]),
                     ("TESS", geo[geo["disc_facility"] == MEASURED_SURVEYS[2]])):
    w = 1 / frame["p"]
    small = frame["pl_rade"] < ROCKY
    WEIGHTED[label] = {"n": len(frame),
                       "raw": float(small.mean()),
                       "weighted": float(w[small].sum() / w.sum()),
                       "med_p": float(frame["p"].median())}
M["weighted"] = WEIGHTED

# where the weight lands, which is the reason the correction goes the way it does
quart = geo.copy()
quart["band"] = pd.qcut(quart["pl_orbsmax"], 4, labels=["closest", "near", "far", "farthest"])
BANDS = {}
for band, rows in quart.groupby("band", observed=True):
    BANDS[str(band)] = {"n": len(rows), "orbit": float(rows["pl_orbsmax"].median()),
                        "rocky": float((rows["pl_rade"] < ROCKY).mean()),
                        "share": float((1 / rows["p"]).sum() / (1 / quart["p"]).sum())}
M["bands"] = BANDS

# The build log is the record that every number was computed. Print all of it, not a selection.
for k in sorted(M):
    if k not in ("surveys", "weighted", "bands", "saw_years"):
        print(f"  measured  {k:>18} = {M[k]}")
for k in ("saw_years", "surveys", "weighted", "bands"):
    print(f"  measured  {k:>18} : {M[k]}")

# The plan records two numbers about this dataset. A builder does not edit the plan, so a
# mismatch is printed rather than patched.
if str(M["n_ps"]) not in TRACK["data"].replace(",", ""):
    print(f"  PLAN DRIFT  course.yml T4 data: says 6,354 rows; ps now returns {M['n_ps']:,} "
          f"(pscomppars {M['n_pc']:,})")
if str(M["rv_radius"]) not in TRACK["fork"]:
    print(f"  PLAN DRIFT  course.yml T4 fork: says 31 RV planets carry a measured radius; "
          f"measured {M['rv_radius']}")


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
# new, so the full module tables would list forty functions it never uses; these are the ones the
# notebook and its model answers actually write.
TRACK_IDEAS = [("P1", "Catalogue completeness"), ("D1", "NaN"), ("D2", "Table"),
               ("S3", "Log axes"), ("ML1", "Linear regression"),
               ("S4", "Bootstrap"), ("S4", "Confidence interval")]
TRACK_FNS = [("D2", "column.isna()"), ("D2", "column.value_counts()"),
             ("D2", "table.groupby(column)"), ("D2", "column.count() / column.median()"),
             ("S3", "np.log10(values)"), ("S3", "array.reshape(-1, 1)"),
             ("ML1", "LinearRegression().fit(x, y)"), ("ML1", "model.coef_[0]"),
             ("ML1", "model.intercept_"), ("ML1", "model.predict(x)"),
             ("ML1", "model.score(x, y)"),
             ("S2", "np.random.default_rng(seed)"),
             ("S4", "np.random.choice(items, size=n, replace=True)"),
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

TITLE = TRACK["title"]

HOOK = f"""
The archive of confirmed planets around other stars holds {M['n_ps']:,} of them. Sort those by the
year they were announced and take the middle radius of each year, and the series falls off a cliff:
the typical planet found up to 2010 was {M['old_med']:.1f} times the radius of the Earth, and the
typical planet found since {EARLY[0]} is {M['new_med']:.2f}.

Read one way, that is the best news in astronomy. We went looking for other Earths and we started
finding them, and the number of small planets in the file has been climbing ever since.

Read the other way, nothing about the planets changed at all. Kepler launched in 2009 and stared at
one patch of sky for four years; TESS launched in 2018 and has swept the whole sky for bright
nearby stars. Each of them could see a particular kind of planet and was nearly blind to the rest.
What fell in 2013 might be the size of the planets we were finding, or it might only be the size of
the planets we had become able to find. This project is about telling those two apart — and then
about what is left over once you have.
"""

md(weekkit.OPENING.format(question=TITLE, datahub=datahub, hook=HOOK.strip()))

md("""
## How this notebook is different

This is a **project track**. It is not a weekly notebook and it does not behave like one.

A weekly notebook shows you a move, walks you through it, and then asks you to make it once
yourself. This one loads the data and reproduces the single result its title rests on — the fall in
the typical planet radius — and then stops helping. From there on every section is a sentence
describing what to find out and an empty cell to find it out in. There is no worked example above
to pattern-match against, because on a real question there never is one.

**There is exactly one self-check in this notebook, and it is on the data loading.** After that,
nothing tells you whether you are right. That is not an oversight and it is not laziness: past the
loading step there is no single right answer here, so a cell that said `assert` would be lying to
you about how research works. What replaces it is the thing researchers actually use — a result you
can get two ways, a number you can predict before you compute it, and a claim you can try to break.

**And it does not close.** The last section is a question this course does not know the answer to.
Everything above it is scaffolding; that question is the project.
""")

md(f"""
## What you'll be able to do

**The science.** Say whether the shrinking of the typical known exoplanet is a fact about planets
or a fact about telescopes, and defend the answer with a number rather than an adjective. Say which
planets two instruments can honestly be compared on, and which comparisons are between a
measurement and the output of somebody's model. Then correct a survey for one of its biases, and
find out what correcting one bias out of several actually buys you.

**The skills.** Tell a measured column from a modelled one in a real archive, using nothing but the
error bars and a provenance column. Fit a straight line in log space and then use it on data it was
never fitted to, deliberately, to see what that does. Turn a piece of orbital geometry into a
weight, and reweight a sample by it.

**The four questions, in order:**

1. Has the typical newly found planet been getting smaller?
2. Which planets can you honestly compare?
3. Which telescope found them?
4. What would we have seen if geometry had not chosen for us?

The open question at the end is not on that list. It is the project; the four above are what you
build to reach it.
""")

md(f"""
## Setup

The NASA Exoplanet Archive publishes the confirmed-planet catalogue over a plain URL, with no key
and no login. The cell below reads it live and falls back to the copy stored with the course.

**Read this before you go on — it is the whole project in miniature.** The archive serves *two*
tables that both hold one row per planet, and today they both hold **{M['n_ps']:,}** rows. They are
not the same data:

- **`ps` with `default_flag=1`** is the literature. One row per planet, taken from whichever
  published paper the archive considers the default reference, and a **hole wherever nobody has
  measured that quantity.** Only {M['radius_pct']:.1f}% of these planets have a radius.
- **`pscomppars`** is the *composite* table. Same planets, but the holes are filled in — with
  values derived from other columns through a model. {M['pc_radius_pct']:.1f}% of these planets
  have a radius.

Everything in this notebook is measured on `ps`. `pscomppars` is loaded as well, and it is used
exactly once, in *Your turn 2*, to show you what the difference between the two is made of.

**NaN:** {idea('D1', 'NaN')['words']} That is the whole distinction above, and a comparison like
`radius < 1.6` is `False` for a hole, silently, with no error — so a planet nobody has measured
gets quietly counted as "not rocky".
""")

code(weekkit.setup_cell(
    imports="import numpy as np\nfrom sklearn.linear_model import LinearRegression\n",
    figsize="(7, 4)",
    cache_base=CACHE_BASE,
    signature="url, cache_name",
    docstring="Read the live archive; fall back to the copy stored with the course.",
    url_expr="url",
    cache_expr="cache_name",
    unpack=f'''
{ARCHIVE}

planets = load(ARCHIVE + "ps+where+default_flag=1&format=csv", "{PS_CACHE}")
composite = load(ARCHIVE + "pscomppars&format=csv", "{PC_CACHE}")

print("ps, the literature: ", planets.shape)
print("pscomppars, filled: ", composite.shape)
print(planets[["pl_name", "pl_rade", "pl_bmasse", "pl_bmassprov", "discoverymethod"]].head())
'''.strip("\n")))

code(f"""
assert "pl_bmassprov" in planets.columns, \\
    "the mass-provenance column is missing — the query was read wrong, or the schema changed"
assert 6000 < len(planets) < 7500, \\
    "expected about {M['n_ps']} planets; 40,000 means default_flag=1 is missing from the query"
assert planets["pl_name"].duplicated().sum() == 0, \\
    "duplicate planet names — without default_flag=1 this is one row per published paper"
print(f"✓ the data — {{len(planets)}} planets from ps and {{len(composite)}} from pscomppars, "
      f"{{planets['pl_rade'].count()}} of the ps rows carrying a measured radius")
""")

md("""
### And that is the last self-check in this notebook

The pipeline is now trustworthy: the table is the table, the query is the query, the counts below
are the counts. Everything from here is yours, and the safety net is gone — nothing will tell you
when you have it right.
""")

# --- the verified half ------------------------------------------------------
md(f"""
## Has the typical newly found planet been getting smaller?

One point per year: the middle radius of every planet announced that year, in radii of the Earth.
Only the {M['has_radius']:,} planets that have a measured radius can be on this plot at all; the
other {M['no_radius']:,} are holes, and `median()` steps over them without saying so.

**Catalogue completeness:** {idea('P1', 'Catalogue completeness')['words']} That sentence was
written about earthquakes. Read it again with "telescopes" in it and it is this project's whole
argument.

The archive is revised continuously, so your counts may differ from the ones printed in this
notebook by a few. Say so if they do — a record that changes under you is the subject here, not a
nuisance.
""")

code(f"""
with_radius = planets[planets["pl_rade"].notna()]
per_year = with_radius.groupby("disc_year")["pl_rade"].median()
count_per_year = with_radius.groupby("disc_year")["pl_rade"].count()

plt.plot(per_year.index, per_year.values, marker="o", color="0.3")
plt.xlabel("year the planet was announced")
plt.ylabel("median radius of that year's planets (Earth radii)")
plt.title(f"Median measured planet radius by year (n = {{len(with_radius)}})")
plt.show()

print(per_year.round(2).to_string())
print("planets per year:", count_per_year.to_dict())

old_planets = with_radius[with_radius["disc_year"] <= 2010]["pl_rade"].values
new_planets = with_radius[with_radius["disc_year"] >= {EARLY[0]}]["pl_rade"].values
print("up to 2010: ", len(old_planets), "planets, median radius",
      round(np.median(old_planets), 2))
print("since {EARLY[0]}: ", len(new_planets), "planets, median radius",
      round(np.median(new_planets), 2))
print("spread of the yearly medians since {SAWTOOTH_FROM}:",
      round(per_year.loc[{SAWTOOTH_FROM}:].std(), 2), "Earth radii, largest minus smallest",
      round(per_year.loc[{SAWTOOTH_FROM}:].max() - per_year.loc[{SAWTOOTH_FROM}:].min(), 2))
""")

md(f"""
That is the observation the project exists to explain, and it needs no statistics to see: the
middle planet fell from about {M['old_med']:.0f} Earth radii to about {M['new_med']:.1f}, a factor
of {M['collapse']:.1f}, and the whole of it happened between 2010 and 2014.

Look at the line after 2014 as well, because a second thing is going on. The yearly median does not
settle — it saws up and down by whole Earth radii from one year to the next
({', '.join(f"{y} → {v}" for y, v in M['saw_years'].items())}), a swing of {M['saw_swing']:.1f}
across {SAWTOOTH_FROM}–{M['year_last']} with a standard deviation of {M['saw_sigma']:.2f}. Nothing
about the sky changes that fast. What changes is which team published a batch that year.

So a *year* is not a useful unit for this data. The first question is what happens when you stop
using it.

One detail in the cell above, because you will copy the shape: the two eras were pulled out with
`.values`, which hands back a plain array instead of a column of the table. Everything below that
resamples or takes a logarithm wants an array, so that is the form this notebook uses whenever a
set of numbers is about to be worked on rather than looked up.
""")

ask(f"""
### ✏️ Your turn 1

Compare two eras instead of {M['year_last'] - SAWTOOTH_FROM + 1} years: **{EARLY[0]}–{EARLY[1]}**
against **{LATE[0]}–{LATE[1]}**. Print how many planets with a measured radius each era holds, the
median radius of each, and the difference between the two medians.

Then put an interval on that difference, because a difference with no interval is not a result.

**Bootstrap:** {idea('S4', 'Bootstrap')['words']}
**Confidence interval:** {idea('S4', 'Confidence interval')['words']}

Resample each era **with replacement** to its own size, take the difference of the two medians, and
do that {N_BOOT} times. `rng = np.random.default_rng({SEED})` once, then
`rng.choice(values, size=len(values), replace=True)` on each era inside the loop. Report the 2.5th
and 97.5th percentiles, and draw the {N_BOOT} differences as a histogram with the observed
difference marked.

Now answer it, in a line your code prints, on your own three numbers — the difference, its
interval, and the year-to-year standard deviation of {M['saw_sigma']:.2f} printed above: is the
later era's larger typical planet something you could have seen in the yearly series, and what
would that series have had to look like for the answer to be yes?
""")

answer(f"""
early = with_radius[with_radius["disc_year"].between({EARLY[0]}, {EARLY[1]})]["pl_rade"].values
late = with_radius[with_radius["disc_year"].between({LATE[0]}, {LATE[1]})]["pl_rade"].values

print("{EARLY[0]}-{EARLY[1]}: n =", len(early), " median", round(np.median(early), 3))
print("{LATE[0]}-{LATE[1]}: n =", len(late), " median", round(np.median(late), 3))
observed = np.median(late) - np.median(early)
print("the later era's median is bigger by", round(observed, 3), "Earth radii")

rng = np.random.default_rng({SEED})
differences = []
for i in range({N_BOOT}):
    a = rng.choice(early, size=len(early), replace=True)
    b = rng.choice(late, size=len(late), replace=True)
    differences.append(np.median(b) - np.median(a))
differences = np.array(differences)

low, high = np.percentile(differences, [2.5, 97.5])
print("95% interval:", round(low, 3), "to", round(high, 3))

plt.hist(differences, bins=40, color="0.4")
plt.axvline(observed, color="firebrick", lw=1.5)
plt.xlabel("difference of era medians, resampled (Earth radii)")
plt.ylabel("resamples")
plt.title(f"{N_BOOT} bootstrap resamples of the era difference (red = observed)")
plt.show()

spread = per_year.loc[{SAWTOOTH_FROM}:].std()
print("No — not in the yearly series. The difference is", round(observed, 2),
      "Earth radii with an interval of", round(low, 2), "to", round(high, 2),
      "which never touches zero, so the shift itself is real; but the year-to-year standard",
      "deviation is", round(spread, 2), "Earth radii —", round(spread / observed, 1),
      "times larger — so a real shift of this size is invisible inside it.")
print("For the yearly series to show it, the scatter between neighbouring years would have",
      "to be smaller than the shift — which means each year would have to be a fair sample",
      "of the sky rather than whichever survey's batch happened to be published that year.")
""")

# --- section 2: the fork ----------------------------------------------------
md(f"""
## Which planets can you honestly compare?

Every number above is a **radius**, and there is a reason the archive has so many holes in that
column. A transit measures a radius directly: the planet crosses the star, the star dims, and the
fraction it dims by is the ratio of the two areas. Radial velocity measures nothing of the sort. It
watches the star wobble along the line of sight, and the wobble depends on the planet's mass *and*
on how the orbit happens to be tilted — which is unknown. So what comes out is not a mass but a
**minimum** mass, `M·sin(i)`, and no radius at all.

The archive records which one you are looking at, in `pl_bmassprov`: `Mass` means somebody pinned
the true mass, `Msini` means the tilt is unknown and the number is a floor.

**Table:** {idea('D2', 'Table')['words']}
""")

md(f"""
### Predict before you run

{M['rv_n']:,} of the planets in `ps` were discovered by radial velocity. Change `my_guess` to how
many of those you think carry a **measured** radius, and run the cell. You will check it in the
next question, and a wrong guess you committed to is worth more than a right answer you were shown.
""")

code(f"""
my_guess = {M['rv_n'] // 2}

print("I think", my_guess, "of the {M['rv_n']} radial-velocity planets have a measured radius")
""")

ask(f"""
### ✏️ Your turn 2

Find out, in `ps` and then in `pscomppars`, and print all of it:

1. For each of the two discovery methods that dominate the archive — `"Transit"` and
   `"Radial Velocity"` — how many planets `ps` holds, and how many of those have a radius.
   `column.count()` counts what is there and skips the holes.
2. The same two counts in `composite`. One of the four numbers will move enormously.
3. `pl_radeerr1` is the upper error bar on the radius. For the radial-velocity planets in each
   table, count how many have a radius **and no error bar at all**. A measurement without an
   uncertainty is not a measurement.
4. `value_counts()` on `pl_bmassprov` for the radial-velocity planets in `ps`.

Then answer it, in a line your code prints, on your own numbers: which radial-velocity planets can
you put on the same axis as the transit planets, and if you took every radius `composite` offers
for them instead, what exactly would you be comparing the transit radii against?
""")

answer(f"""
for method in ["Transit", "Radial Velocity"]:
    a = planets[planets["discoverymethod"] == method]
    b = composite[composite["discoverymethod"] == method]
    print(f"{{method:16}} ps: {{len(a):5}} planets, {{a['pl_rade'].count():5}} with a radius"
          f"   |  pscomppars: {{b['pl_rade'].count():5}} with a radius")

for name in ["ps", "pscomppars"]:
    table = planets if name == "ps" else composite
    rv = table[table["discoverymethod"] == "Radial Velocity"]
    no_bar = rv[rv["pl_rade"].notna() & rv["pl_radeerr1"].isna()]
    print(f"{{name:11}} radial-velocity radii: {{rv['pl_rade'].count():5}},"
          f" of which {{len(no_bar):5}} carry no error bar"
          f" — median radius {{round(rv['pl_rade'].median(), 2)}}")

rv_ps = planets[planets["discoverymethod"] == "Radial Velocity"]
rv_pc = composite[composite["discoverymethod"] == "Radial Velocity"]
print("what ps says the mass numbers are:", rv_ps["pl_bmassprov"].value_counts().to_dict())
print("distinct radius values across all", rv_pc["pl_rade"].count(), "pscomppars radii:",
      rv_pc["pl_rade"].nunique())

print("Only the", rv_ps["pl_rade"].count(), "radial-velocity planets with a measured radius —",
      "the ones found by wobble and later caught in transit as well — can go on the same axis",
      "as the transit radii. If I used all", rv_pc["pl_rade"].count(), "from pscomppars I would",
      "be comparing measured transit radii against numbers no telescope produced: a mass-radius",
      "model's output, run on a minimum mass. That is why", len(no_bar), "of them carry no",
      "error bar, and why only", rv_pc["pl_rade"].nunique(), "distinct values cover them all.")
""")

md(f"""
Mass and radius really are related — a big planet is usually a heavy one — so estimating the second
from the first is not a silly thing to do, and somebody has to do something about a planet whose
radius nobody has measured. Whether such an estimate can then stand beside a measurement is a
separate question, and the way to settle it is to build the estimator yourself and look at what
comes out.

**Log axes:** {idea('S3', 'Log axes')['words']}
**Linear regression:** {idea('ML1', 'Linear regression')['words']}
""")

ask(f"""
### ✏️ Your turn 3

**Fit it.** Take every planet in `ps` whose `pl_bmassprov` is `"Mass"` — a real mass, not a floor —
and which also has a measured radius. Fit a straight line to `log10(radius)` against
`log10(mass)`. `np.log10(...)`, then `.reshape(-1, 1)` on the x values, then
`LinearRegression().fit(x, y)`. Print how many planets you fitted, the slope, the intercept and
`model.score(x, y)`, and draw the points and the line.

**Then use it.** Take the radial-velocity planets that have a mass but **no** measured radius, feed
their `pl_bmasse` — which for most of them is an `Msini`, a floor — through your line, and undo the
log with `10 ** ...`. Print the median radius your line invents for them.

**Then put three numbers side by side**: that invented median, the median measured radius of the
transit planets, and the median measured radius of the radial-velocity planets that *do* have one.

Now answer it, in a line your code prints, on those three numbers: "radial-velocity surveys find
planets several times larger than transit surveys do" is a claim you will read in print. On your
own output, is it a fact about planets or an output of the line you just fitted — and what would
you have to compare instead to settle it with measurements only?
""")

answer(f"""
train = planets[(planets["pl_bmassprov"] == "Mass") & planets["pl_rade"].notna()
                & planets["pl_bmasse"].notna()]
x = np.log10(train["pl_bmasse"].values).reshape(-1, 1)
y = np.log10(train["pl_rade"].values)
model = LinearRegression().fit(x, y)

print("fitted on", len(train), "planets with a true mass and a measured radius")
print("log10 radius =", round(model.intercept_, 3), "+", round(model.coef_[0], 3), "* log10 mass")
print("R2 =", round(model.score(x, y), 3))

plt.scatter(x, y, s=4, color="0.5")
plt.plot(x, model.predict(x), color="firebrick", lw=1.5)
plt.xlabel("log10 planet mass (Earth masses)")
plt.ylabel("log10 planet radius (Earth radii)")
plt.title(f"The mass-radius line, fitted on {{len(train)}} measured planets")
plt.show()

rv = planets[planets["discoverymethod"] == "Radial Velocity"]
guessing = rv[rv["pl_rade"].isna() & rv["pl_bmasse"].notna()]
invented = 10 ** model.predict(np.log10(guessing["pl_bmasse"].values).reshape(-1, 1))

transits = planets[planets["discoverymethod"] == "Transit"]
measured_rv = rv["pl_rade"].median()
print("radii my line invents for", len(guessing), "radial-velocity planets: median",
      round(np.median(invented), 2))
print("transit planets, measured:", transits["pl_rade"].count(), "median",
      round(transits["pl_rade"].median(), 2))
print("radial-velocity planets, measured:", rv["pl_rade"].count(), "median", round(measured_rv, 2))

print("An output of the line. It says radial-velocity planets are",
      round(np.median(invented) / transits["pl_rade"].median(), 1),
      "times bigger, but every one of those radii came out of a model I fitted, run on a",
      "minimum mass; on the", rv["pl_rade"].count(), "radial-velocity planets anyone has",
      "actually measured, the ratio is", round(measured_rv / transits["pl_rade"].median(), 2),
      "- barely a difference at all.")
print("To settle it with measurements only I would compare the quantity both methods really",
      "measure, which is mass:", round(rv["pl_bmasse"].median(), 1), "Earth masses for",
      "radial velocity against", round(transits["pl_bmasse"].median(), 1), "for transits, a",
      "factor of", round(rv["pl_bmasse"].median() / transits["pl_bmasse"].median(), 1),
      "- and I would say out loud that the", rv["pl_rade"].count(), "with radii are themselves",
      "the nearby small ones that happened to transit too, so they are not a fair sample either.")
print("The scatter also shows my line is wrong where I used it. Above about 100 Earth masses",
      "the real radii flatten off near 12 Earth radii - add mass to a gas giant and it gets",
      "denser, not bigger - while a straight line in log-log has to keep climbing. The median",
      "mass I fed it was", round(guessing["pl_bmasse"].median(), 0),
      "Earth masses, which is inside exactly that flat part.")
""")

# --- section 3: the surveys -------------------------------------------------
md(f"""
## Which telescope found them?

Three spacecraft found most of the transiting planets in this file, and they were built to do
different things.

**Kepler** (2009–2013) pointed at one 115-square-degree patch of Cygnus and did not move, staring
at 150,000 stars for four years. A long stare finds planets on long orbits, and a fixed field
reaches whatever stars happen to be in it, however faint and however far.

**K2** (2014–2018) was Kepler after two of its reaction wheels failed: the same telescope, but only
able to hold a field for about 80 days at a time, along the ecliptic.

**TESS** (2018–) does the opposite of Kepler. It sweeps almost the whole sky in 27-day sectors,
which is too short to catch a long orbit, and it was designed to look at stars that are **bright**
— which mostly means near — so that the planets it finds can be followed up from the ground.

Three instruments, three different slices of the same galaxy. The columns to ask with are
`disc_facility`, `pl_rade`, `pl_orbsmax` (the orbit's width in AU) and `sy_dist` (how far away the
system is, in parsecs).
""")

ask(f"""
### ✏️ Your turn 4

For the transit-discovered planets in `ps`, compare the three surveys. Their `disc_facility` strings
are exactly `"Kepler"`, `"K2"` and `"Transiting Exoplanet Survey Satellite (TESS)"` — print the
short name, not the long one, or your output will run off the page.

For each: how many planets, the median radius, the median orbit width in AU, the median system
distance in parsecs, and the fraction with a radius below {ROCKY} Earth radii. Then draw one figure
that puts Kepler's and TESS's radius distributions side by side — two `plt.hist` calls of
`np.log10` of the radius, with `plt.legend()`, is enough.

Now answer it, in a line your code prints, on your own numbers: the two big surveys disagree about
what a typical planet is. Is one of them wrong, and if not, what does it mean to ask which of the
two is a picture of the sky?
""")

answer(f"""
transits = planets[planets["discoverymethod"] == "Transit"]
surveys = {{"Kepler": "Kepler", "K2": "K2",
           "TESS": "Transiting Exoplanet Survey Satellite (TESS)"}}

for short in surveys:
    rows = transits[transits["disc_facility"] == surveys[short]]
    print(f"{{short:7}} n={{len(rows):5}}"
          f"  radius {{round(rows['pl_rade'].median(), 2):6}} Re"
          f"  orbit {{round(rows['pl_orbsmax'].median(), 3):6}} AU"
          f"  distance {{round(rows['sy_dist'].median()):6.0f}} pc"
          f"  below {ROCKY} Re {{round((rows['pl_rade'] < {ROCKY}).mean(), 3)}}")

kepler = transits[transits["disc_facility"] == "Kepler"]["pl_rade"].dropna()
tess = transits[transits["disc_facility"] == surveys["TESS"]]["pl_rade"].dropna()

plt.hist(np.log10(kepler), bins=40, color="0.3", label="Kepler")
plt.hist(np.log10(tess), bins=40, color="firebrick", alpha=0.6, label="TESS")
plt.xlabel("log10 planet radius (Earth radii)")
plt.ylabel("planets")
plt.title(f"What each survey calls a typical planet "
          f"(n = {{len(kepler)}} Kepler, {{len(tess)}} TESS)")
plt.legend()
plt.show()

print("Neither is wrong. Kepler's median planet is", round(kepler.median(), 2), "Earth radii at",
      round(transits[transits['disc_facility'] == 'Kepler']['sy_dist'].median()), "parsecs and",
      "TESS's is", round(tess.median(), 2), "at",
      round(transits[transits['disc_facility'] == surveys['TESS']]['sy_dist'].median()),
      "parsecs; each is an honest census of the planets its own design could detect.")
print("Asking which is a picture of the sky is asking a question neither of them answers,",
      "because the sky was never sampled - the instrument chose the sample. The only way to",
      "get from either census to a population is to write down what each survey could not",
      "have seen and put it back, and nothing in this file does that for you.")
""")

# --- section 4: geometry ----------------------------------------------------
md(f"""
## What would we have seen if geometry had not chosen for us?

Here is one thing a transit survey misses that you can compute exactly.

A planet transits only if its orbit is edge-on enough that it passes in front of the star from
where we sit. For orbits pointing in random directions, the chance of that is the star's radius
divided by the width of the orbit:

$$p_{{\\rm transit}} \\approx \\frac{{R_\\star}}{{a}}$$

Small numbers, and they follow from that one line. A planet like the Earth, one AU from a star like
the Sun, transits for about one observer in {M['earth_odds']:.0f}; a planet on a three-day orbit
round the same star transits for about one in {M['hot_odds']:.0f}. **Every close-in planet in the
archive stands for a handful of identical planets nobody could see; every distant one stands for
hundreds.** The bias is not subtle and it is not a matter of opinion — it is geometry.

The units do not match, so one conversion is needed. `st_rad` is in radii of the Sun and
`pl_orbsmax` is in AU. The IAU *defines* the nominal solar radius as {R_SUN_M:.4g} m (2015
Resolution B3) and the astronomical unit as exactly {AU_M:.10g} m (2012 Resolution B2), so one
solar radius is {R_SUN_IN_AU:.8f} AU. Those are definitions, not measurements off a web page, which
is why they carry a resolution number instead of a read-date.

This estimate ignores the planet's own radius and any eccentricity. Both matter at the ten-percent
level; neither changes anything below.
""")

ask(f"""
### ✏️ Your turn 5

Work with the transit-discovered planets in `ps` that have `st_rad`, `pl_orbsmax` **and** `pl_rade`.
Print how many that is out of the {M['tr_n']:,} transit planets — the ones you have to drop are
themselves a selection, and it is worth knowing how big it is.

Compute the transit probability for each, using `st_rad * {R_SUN_IN_AU:.8f} / pl_orbsmax`. Print the
smallest, the median and the largest. Then plot `np.log10` of the probability against `np.log10` of
the orbit width.

**Use these names**: call the table `geometry` and the new column `p_transit`, because *Your turn 6*
carries straight on from them.

Now answer it, in a line your code prints, on your own numbers: for the median planet in this
sample, how many planets on similar orbits does each one you can see stand for — and does that
multiplier stay the same across the plot, or does it depend on where the planet is?
""")

answer(f"""
geometry = transits[transits["st_rad"].notna() & transits["pl_orbsmax"].notna()
                    & transits["pl_rade"].notna()].copy()
geometry["p_transit"] = geometry["st_rad"] * {R_SUN_IN_AU:.8f} / geometry["pl_orbsmax"]

print("transit planets with all three columns:", len(geometry), "of", len(transits))
print("transit probability — smallest", round(geometry["p_transit"].min(), 5),
      " median", round(geometry["p_transit"].median(), 4),
      " largest", round(geometry["p_transit"].max(), 4))

plt.scatter(np.log10(geometry["pl_orbsmax"]), np.log10(geometry["p_transit"]),
            s=4, color="0.4")
plt.xlabel("log10 orbit width (AU)")
plt.ylabel("log10 transit probability")
plt.title(f"Geometry, for {{len(geometry)}} transiting planets")
plt.show()

print("The median planet here transits for", round(geometry["p_transit"].median(), 4),
      "of observers, so each one I can see stands for about",
      round(1 / geometry["p_transit"].median()), "planets on similar orbits.")
print("The multiplier is not constant. The cloud slopes down across the plot, because p is one",
      "over the orbit width, so it runs from about", round(1 / geometry["p_transit"].max()),
      "at the closest orbits to", round(1 / geometry["p_transit"].min()), "at the widest — a",
      "factor of", round(geometry["p_transit"].max() / geometry["p_transit"].min()),
      "across the sample. The width of the cloud at any orbit is the range of stellar radii.")
""")

md(f"""
The step every survey paper takes next is to treat `1 / p_transit` as a **weight**, so that a
planet standing in for many is counted many times. A share computed with those weights is an
estimate of the share in the *population*, rather than of the share in the file.

That is one correction, and it is the only bias in this dataset you can compute exactly.
""")

ask(f"""
### ✏️ Your turn 6

Give every planet in your `geometry` table the weight `1 / p_transit`, and compute the fraction
with a radius below {ROCKY} Earth radii **twice**: the plain fraction, and the weighted one, which
is `weights[small].sum() / weights.sum()` where `small` is the mask.

Do it for the whole transit sample **and** for Kepler alone, so you have four numbers. Draw them as
a bar chart, four bars, raw and weighted side by side for each sample.

Then, so you can see where the weight went, split `geometry` into four bands by orbit width with
`pd.qcut(geometry["pl_orbsmax"], 4)` and print, per band: how many planets, the median orbit width,
the fraction below {ROCKY} Earth radii, and the band's share of the total weight.

Now answer it, in a line your code prints, on your own numbers: correcting for geometry alone moves
the small-planet fraction by some amount and in some direction — say which, then use the four bands
to say *why* it moved that way, and what that tells you about the bias you have **not** corrected.
""")

answer(f"""
weights = 1 / geometry["p_transit"]
small = geometry["pl_rade"] < {ROCKY}

kepler_rows = geometry[geometry["disc_facility"] == "Kepler"]
kepler_weights = 1 / kepler_rows["p_transit"]
kepler_small = kepler_rows["pl_rade"] < {ROCKY}

shares = [small.mean(), weights[small].sum() / weights.sum(),
          kepler_small.mean(), kepler_weights[kepler_small].sum() / kepler_weights.sum()]
labels = ["all raw", "all weighted", "Kepler raw", "Kepler weighted"]
for label, share in zip(labels, shares):
    print(f"{{label:16}} {{round(share, 3)}}")

plt.bar(labels, shares, color=["0.4", "firebrick", "0.4", "firebrick"])
plt.xlabel("sample, and whether the transit-probability weight was applied")
plt.ylabel(f"fraction below {ROCKY} Earth radii")
plt.title(f"Geometric correction, {{len(geometry)}} transiting planets")
plt.show()

geometry["band"] = pd.qcut(geometry["pl_orbsmax"], 4)
widest_share = 0
for band, rows in geometry.groupby("band", observed=True):
    widest_share = (1 / rows["p_transit"]).sum() / weights.sum()      # the last band is the widest
    print("orbit", str(band), " n =", len(rows),
          " median", round(rows["pl_orbsmax"].median(), 3), "AU",
          " below {ROCKY} Re", round((rows["pl_rade"] < {ROCKY}).mean(), 3),
          " share of the weight", round(widest_share, 3))

print("It moved DOWN, not up:", round(shares[0], 3), "to", round(shares[1], 3), "overall and",
      round(shares[2], 3), "to", round(shares[3], 3), "for Kepler. Correcting for geometry made",
      "the population look LESS Earth-like, not more.")
print("The bands say why. The widest-orbit quarter carries", round(widest_share, 2),
      "of all the weight, because 1/p is largest out there — and it is also the band with the",
      "fewest small planets, because a small planet on a wide orbit makes a shallow, rare",
      "transit that a survey cannot detect.")
print("So the bias I have not corrected is detectability, and it runs the opposite way to",
      "geometry. Fixing only the one I can compute exactly makes the answer worse, which means a",
      "partial debiasing is not a small version of the right answer — it can point the wrong way.")
""")

# --- closing ----------------------------------------------------------------
md(f"""
{weekkit.CLOSING_HEADING}

Mostly better telescopes. The typical known planet fell from {M['old_med']:.1f} Earth radii to
{M['new_med']:.2f} because Kepler arrived, stared at one faint patch of sky for four years and
returned {M['surveys']['Kepler']['n']:,} planets with a median radius of
{M['surveys']['Kepler']['radius']:.2f}; it has been climbing again because TESS arrived, swept the
bright nearby sky and returned {M['surveys']['TESS']['n']:,} with a median of
{M['surveys']['TESS']['radius']:.2f}. Both censuses are honest and neither is the sky. What that
leaves unanswered is what the sky actually holds, and the next section is about how far this file
can get you towards it.
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
    "baseline_first": ("2 · The trivial baseline", f"""
Before any statistic, state the dumbest answer to your question and what it gives. Every later
number is reported against it.

On this track the baseline is one median over the whole file, with no year, no method and no
instrument in it: the archive's {M['has_radius']:,} measured radii have a middle value, and that
number is what somebody quoting "the typical exoplanet" is quoting. Say what it is, then say what
each later split bought you over it.
"""),
    "split_by_structure": ("3 · Split by structure", """
Earth data are correlated in space and in time, so whatever you split, resample or count as
independent has to be split along the structure that is really there — never at random across
rows.

This track fits one line and does not score it, so there is no train/test split to get wrong. The
same idea has teeth anyway, and the structure here is the **survey**: two planets found by the same
instrument in the same field share everything about how they were found. Name the unit you treated
as independent, say why, and say what changed when you grouped by it instead of by row.
"""),
    "what_i_got_wrong": ("4 · What I got wrong", """
What failed, and what you believed before it failed. Honest failure is graded; a faked success is
not. Both of your *Predict before you run* guesses belong here if they were wrong.
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

Here is what is actually established, and it is less than it looks. The fall from
{M['old_med']:.1f} Earth radii to {M['new_med']:.2f} is Kepler arriving, and that is settled. That
each instrument returns a different galaxy is settled: Kepler's median planet sits
{M['surveys']['Kepler']['radius']:.2f} Earth radii at
{M['surveys']['Kepler']['distance']:.0f} parsecs, TESS's {M['surveys']['TESS']['radius']:.2f} at
{M['surveys']['TESS']['distance']:.0f}. What is **not** settled is the thing all of that is
evidence about: what the population is. Your turn 6 is where you found out what happens when you
correct one bias out of several, and it is the reason this question is open rather than merely
unfinished.

Four directions, none of them worked out here:

1. **Put the detections you never made back in.** Transit probability is one factor. The other is
   whether a transit that did occur was deep enough to find: the fraction of the star's light a
   planet blocks is `(pl_rade / st_rad)²` once both are in the same units, and one solar radius is
   {R_SUN_IN_REARTH:.3f} Earth radii — the ratio of the two IAU 2015 Resolution B3 nominal radii
   quoted above. Compute that depth for every planet, find the value below which each survey stops
   reporting anything, and use that floor to say what each survey could not have seen. It is a
   completeness limit built from the archive's own columns, and it is the piece Your turn 6 was
   missing.
2. **Correct one survey properly rather than all of them badly.** Kepler is the only one here that
   stared at a fixed, documented set of stars, so it is the only one where "the sample" means
   something. Debias Kepler alone, quote the small-planet fraction with an interval, and say what
   population that number is a statement about — which stars, at what distances, on what orbits.
3. **Ask whether the population has structure the census is hiding.** The radius distribution of
   small planets is not smooth: there is a reported deficit near 1.8 Earth radii, and whether it is
   real or an artefact of the radii being uncertain has been argued about for years. Weighting
   changes a histogram's shape, not just its median — does the gap survive your weights, and does
   it survive in each survey separately?
4. **Decide what this file cannot answer.** The archive holds {M['no_radius']:,} planets with no
   radius at all and {M['rv_msini']:,} radial-velocity planets whose mass is only a floor. Write
   down what fraction of the population you are estimating actually rests on a measurement, and
   what fraction rests on a model. If the honest answer is that most of it is model, that is a
   result.

And one that is bigger than a semester: what would a survey have to be like for its census to be a
population? Not "how many more planets" — what property would the sample need. Write down the
condition, then check it against the three surveys in this file, and see whether any of them could
ever have met it.
""")

ask(f"""
### ✏️ Your turn 7 — the first move

Before you close this notebook, answer this in a few sentences and then make the measurement in the
cell below your prose. Of everything left undone above, which **one** measurement would you make
first — what would it show if the population really is mostly small worlds, what would it show if
it is not, and what number would change your mind?
""")

answer_prose(f"""
I would compute each survey's transit **depth** floor first, because it is the missing half of the
correction I already made and it is the only one of the four directions I can do with columns that
are already in the file. The depth of a transit is `(pl_rade / st_rad)²` in the same units, and if a
survey has a detection limit then its shallowest reported planets should pile up against a floor
rather than trailing off smoothly. If the floor is sharp, I can say what each survey could not have
seen, and put those planets back the same way I put back the ones whose orbits were tilted wrong.

If the population really is mostly small worlds, the correction should push the small-planet
fraction back **up** once the depth floor is in — because the planets a depth limit removes are
exactly the small ones, and they are removed hardest at the wide orbits that the geometric weight
counts most heavily. If it is not, the two corrections will roughly cancel and the weighted fraction
will land near the raw one, which would say that the file's own census is not far from the truth for
the stars these surveys actually looked at.

The number that would change my mind is the weighted small-planet fraction for Kepler alone with
both corrections applied. My geometric-only correction took it from
{M['weighted']['Kepler']['raw']:.3f} to {M['weighted']['Kepler']['weighted']:.3f}. If adding the
depth floor sends it back above the raw {M['weighted']['Kepler']['raw']:.3f}, the small planets were
being hidden and the population is more Earth-like than the file; if it stays below, then the fall I
have been calling an instrument effect is smaller than I thought, and I would have to say so.

What makes me doubt the whole exercise in advance is that a depth floor fitted from the reported
planets is fitted from the survivors — the planets below the floor are the ones that are not in the
file to be fitted. That is the same circularity as inventing radii from masses, one level up, and I
should expect the answer to be sensitive to how I draw the floor.
""")

answer(f"""
geometry["depth"] = (geometry["pl_rade"] / (geometry["st_rad"] * {R_SUN_IN_REARTH:.3f})) ** 2

for short in surveys:
    rows = geometry[geometry["disc_facility"] == surveys[short]]
    print(f"{{short:7}} n={{len(rows):5}}"
          f"  shallowest {{rows['depth'].min():.2e}}"
          f"  5th percentile {{np.percentile(rows['depth'], 5):.2e}}"
          f"  median {{rows['depth'].median():.2e}}")

plt.hist(np.log10(geometry["depth"]), bins=50, color="0.4")
plt.xlabel("log10 transit depth (fraction of the star's light blocked)")
plt.ylabel("planets")
plt.title(f"How deep a transit had to be to be found (n = {{len(geometry)}})")
plt.show()

print("The floor is not one number — each survey stops in a different place, so 'what could not",
      "have been seen' has to be asked survey by survey, which is direction 2.")

impossible = geometry[geometry["depth"] > 1]
print("And the formula breaks somewhere:", len(impossible), "row(s) come out deeper than 1,",
      "which would be blocking more than all of the star's light —",
      list(impossible["pl_name"]))
print("That is not bad data. It is a planet larger than its star: a white dwarf, whose radius",
      "is about the Earth's. (Rp / Rstar) ** 2 is only a depth while the planet is the smaller",
      "of the two, so my floor has to be computed on the systems where that holds.")
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
    for name in (PS_CACHE, PC_CACHE):
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
