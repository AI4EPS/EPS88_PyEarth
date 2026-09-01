#!/usr/bin/env python
"""Build week 10 — "Earthquake or explosion — how does the world verify a nuclear test ban?"

Emits both notebooks from one source so they cannot drift:

    docs/notebooks/10_earthquake_or_explosion_solution.ipynb   executed, every output saved
    docs/notebooks/10_earthquake_or_explosion.ipynb            the same file with the answers deleted

It also writes the week's two data files into data/. BOTH halves ship with the course and the
notebook reads both directly, with no live branch. The earthquake half has no choice: its query
matches far more than the archive's 20,000-event ceiling. The blast half could be fetched live,
and used to be — but the two halves are ONE catalogue, and a live half joined to a frozen half is
no longer the slice this week is pinned to. USGS keeps editing events inside the pinned window
(18 of them were revised in the month before this build), and one late-inserted blast or one
revised depth moves the blast count, both split sizes and all four F1 scores, under prose that
hardcodes the old ones and asserts that still pass. Weeks 7 and 9 keep their live read and take
`--refresh` because their prose is derived from a cache the student's own live read reproduces;
here it cannot be, because the other half of the join can never be refreshed at all. So the cache
is the data — TEMPLATE 1.3's shipped-dataset exception, via `weekkit.asset_setup_cell` — and
`--refresh` belongs to the instructor, who re-pulls both halves together and rebuilds the prose
around them.

Every number that appears in prose or in a model answer is computed HERE, from the same files
the notebook reads. Nothing is typed from memory or copied from the plan.

    python tools/build_week10.py
    python tools/build_week10.py --refresh    # re-pull BOTH halves, then rebuild every number
"""
import json
import math
import pathlib
import subprocess
import sys

import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/notebooks"
SLUG = "10_earthquake_or_explosion"

course = yaml.safe_load((ROOT / "course.yml").read_text())
WEEK = next(s for s in course["schedule"] if s["n"] == 10)
PLATFORM = course["platform"]
CACHE_BASE = PLATFORM["cache_base"]

REFRESH = "--refresh" in sys.argv

# The slice, pinned so the shipped files, the notebook and the prose below cannot drift apart.
# course.yml pins "M1.5+ 2015-01-01 onward"; `onward` grows every day, so the end is pinned here
# at the date the files were built. The notebook no longer asks the archive for anything — see
# the module docstring — so these constants describe the files rather than a live query.
START, END, MINMAG = "2015-01-01", "2026-08-31", 1.5
BOX = "&minlatitude=32&maxlatitude=42&minlongitude=-125&maxlongitude=-114"
QUERY = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&orderby=time-asc"
         f"&starttime={START}&endtime={END}&minmagnitude={MINMAG}{BOX}")

BLAST_CACHE = "week10_ca_quarry_blasts_2015_2026.csv"
QUAKE_CACHE = "week10_ca_earthquakes_2015_2026.csv.gz"
COLUMNS = ["time", "latitude", "longitude", "depth", "mag", "type"]
FEATURES = ["latitude", "longitude", "depth", "mag", "hour", "weekday"]
NO_DEPTH = [f for f in FEATURES if f != "depth"]
TWO = ["hour", "depth"]
FIRST_YEAR, LAST_YEAR = 2015, 2026

# 2019 holds the Ridgecrest sequence and on its own exceeds the archive's 20,000-event ceiling,
# so that one year is fetched in five pieces. Every other year fits in a single request.
SPANS = []
for _y in range(FIRST_YEAR, LAST_YEAR + 1):
    _end = f"{_y + 1}-01-01" if _y < LAST_YEAR else END
    if _y == 2019:
        SPANS += [("2019-01-01", "2019-04-01"), ("2019-04-01", "2019-07-01"),
                  ("2019-07-01", "2019-08-01"), ("2019-08-01", "2019-10-01"),
                  ("2019-10-01", _end)]
    else:
        SPANS.append((f"{_y}-01-01", _end))


# ---------------------------------------------------------------------------
# 1. build the two cached files, then measure everything the notebook will say
# ---------------------------------------------------------------------------
def build_caches():
    """Write data/week10_*. Both files are the exact bytes the notebook reads.

    `--refresh` re-pulls both halves TOGETHER. Refreshing one alone would break the join: the
    two files are one catalogue cut by `type`, and half a catalogue from August against half
    from November is not the pinned slice.
    """
    blast_path = ROOT / "data" / BLAST_CACHE
    if REFRESH or not blast_path.exists():
        pd.read_csv(QUERY + "&eventtype=quarry%20blast").to_csv(blast_path, index=False)
    quake_path = ROOT / "data" / QUAKE_CACHE
    if REFRESH or not quake_path.exists():
        parts = []
        for start, end in SPANS:
            url = (QUERY.replace(f"starttime={START}", f"starttime={start}")
                        .replace(f"endtime={END}", f"endtime={end}") + "&eventtype=earthquake")
            parts.append(pd.read_csv(url))
        whole = pd.concat(parts, ignore_index=True).drop_duplicates(subset="id")
        whole[COLUMNS].to_csv(quake_path, index=False, compression="gzip")
    return blast_path, quake_path


BLAST_PATH, QUAKE_PATH = build_caches()
blasts = pd.read_csv(BLAST_PATH)[COLUMNS]
quakes = pd.read_csv(QUAKE_PATH)
events = pd.concat([quakes, blasts], ignore_index=True)
events["is_blast"] = events["type"] == "quarry blast"
local = pd.to_datetime(events["time"]).dt.tz_convert("US/Pacific")
events["hour"] = local.dt.hour
events["weekday"] = local.dt.dayofweek
events["year"] = events["time"].str[:4]

blast_rows = events[events["is_blast"]]
quake_rows = events[events["type"] == "earthquake"]

M = {}
M["n_blasts"] = int(len(blast_rows))
M["n_quakes"] = int(len(quake_rows))
M["n_events"] = int(len(events))
M["per_blast"] = round(M["n_quakes"] / M["n_blasts"], 1)
# The spine's first question counts BLASTS AGAINST THE WHOLE CATALOGUE, which is a different
# number from per_blast (blasts against EARTHQUAKES) and one larger, so it is measured here
# rather than reused: per_blast rounds to one blast per 46 earthquakes, and this to one in 47.
M["one_in"] = int(round(M["n_events"] / M["n_blasts"]))
M["frac_eq"] = float((~events["is_blast"]).mean())

M["blast_depth_median"] = float(blast_rows["depth"].median())
M["quake_depth_median"] = float(quake_rows["depth"].median())
M["blast_above_sea"] = float((blast_rows["depth"] <= 0).mean())
M["quake_above_sea"] = float((quake_rows["depth"] <= 0).mean())
M["quake_deeper_25"] = float((quake_rows["depth"] > 25).mean())
M["blast_distinct_depths"] = int(blast_rows["depth"].nunique())
M["quake_distinct_depths"] = int(quake_rows["depth"].nunique())
depth_counts = blast_rows["depth"].value_counts()
M["top_depth"] = float(depth_counts.index[0])
M["top_depth_n"] = int(depth_counts.iloc[0])
same_depth = blast_rows[blast_rows["depth"] == M["top_depth"]]
M["top_depth_lat_span"] = float(same_depth["latitude"].max() - same_depth["latitude"].min())
M["top_depth_lon_span"] = float(same_depth["longitude"].max() - same_depth["longitude"].min())
# A degree of latitude is a degree of latitude; a degree of longitude shrinks with the cosine.
# The whole point below is that the two spans are NOT the same distance, so convert both.
M["top_depth_lat_km"] = M["top_depth_lat_span"] * 111.32
M["top_depth_lon_km"] = (M["top_depth_lon_span"] * 111.32
                         * math.cos(math.radians(float(same_depth["latitude"].mean()))))
M["second_depth"] = float(depth_counts.index[1])
M["second_depth_n"] = int(depth_counts.iloc[1])
second_depth = blast_rows[blast_rows["depth"] == M["second_depth"]]
M["second_depth_lat_span"] = float(second_depth["latitude"].max()
                                   - second_depth["latitude"].min())
M["second_depth_lon_span"] = float(second_depth["longitude"].max()
                                   - second_depth["longitude"].min())
M["second_depth_lat_km"] = M["second_depth_lat_span"] * 111.32
# Blast depths are quoted on a 10 m grid, not to the metre: -0.82 km is two decimal places of a
# kilometre. Measure the share so the prose can say "10 m grid" rather than assert it. The
# comparison needs a tolerance, not `==`: -0.82 * 100 is -81.99999999999999 in binary floating
# point, so exact equality reports 88% where the true figure is 95%.
M["depth_grid_share"] = float((((blast_rows["depth"] * 100).round()
                                - blast_rows["depth"] * 100).abs() < 1e-6).mean())

# `place` and `depthError` are two of the columns COLUMNS drops. They carry the mechanism: place
# names the site the analyst matched the event to, and depthError says whether the location
# routine ever solved for depth at all.
blast_meta = pd.read_csv(BLAST_PATH)
# `place` reads "5 km NNW of Boron, CA" — and, in the same column, "5km NNW of Boron, CA" and
# "5km N of Boron, California". The distance, the bearing, the space and the state's name all
# vary, so counting the raw text scatters ONE quarry across dozens of rows: the largest raw
# value_counts row at the top depth is 82 of 360, which reads as a fifth of a site and is really
# one spelling of all of it. Cut out what sits between " of " and the comma before counting, and
# the notebook prints exactly this, so the share the prose quotes is on the screen.
sites = blast_meta["place"].str.split(" of ").str[-1].str.split(",").str[0]
towns = sites[blast_meta["depth"] == M["top_depth"]].value_counts()
M["top_site_a"], M["top_site_a_n"] = str(towns.index[0]), int(towns.iloc[0])
M["top_site_b"], M["top_site_b_n"] = str(towns.index[1]), int(towns.iloc[1])
M["top_sites"] = int(len(towns))
# The second depth was asserted to be "most of them back at Boron" and never measured; the cell
# beside the claim printed raw place strings, whose top row is 69 of 247 — 28%, not "most".
second_towns = sites[blast_meta["depth"] == M["second_depth"]].value_counts()
M["second_site_a"], M["second_site_a_n"] = str(second_towns.index[0]), int(second_towns.iloc[0])
M["second_site_b"] = str(second_towns.index[1])
M["second_site_share"] = M["second_site_a_n"] / M["second_depth_n"]
M["second_sites"] = int(len(second_towns))
M["depth_err_top"] = float(blast_meta["depthError"].value_counts().index[0])
M["depth_err_share"] = float((blast_meta["depthError"] == M["depth_err_top"]).mean())
M["depth_err_n"] = int((blast_meta["depthError"] == M["depth_err_top"]).sum())

M["blast_workhours"] = float(((blast_rows["hour"] >= 10) & (blast_rows["hour"] < 17)).mean())
M["quake_workhours"] = float(((quake_rows["hour"] >= 10) & (quake_rows["hour"] < 17)).mean())
M["blast_weekdays"] = float((blast_rows["weekday"] <= 4).mean())
M["quake_weekdays"] = float((quake_rows["weekday"] <= 4).mean())

X = events[FEATURES]
y = events["is_blast"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,
                                                    stratify=y)
M["n_train"] = int(len(X_train))
M["n_test"] = int(len(X_test))
M["blasts_train"] = int(y_train.sum())
M["blasts_test"] = int(y_test.sum())


def score(tag, guess):
    """Accuracy, precision, recall, F1 and the confusion matrix of one set of test predictions."""
    M[tag + "_acc"] = float(accuracy_score(y_test, guess))
    M[tag + "_prec"] = float(precision_score(y_test, guess, zero_division=0))
    M[tag + "_rec"] = float(recall_score(y_test, guess, zero_division=0))
    M[tag + "_f1"] = float(f1_score(y_test, guess, zero_division=0))
    cm = confusion_matrix(y_test, guess)
    M[tag + "_tn"], M[tag + "_fp"] = int(cm[0, 0]), int(cm[0, 1])
    M[tag + "_fn"], M[tag + "_tp"] = int(cm[1, 0]), int(cm[1, 1])


def hand_rule(rows, low=10, high=17):
    """The two-condition baseline: at or above sea level, and inside working hours."""
    return (rows["depth"] <= 0) & (rows["hour"] >= low) & (rows["hour"] < high)


score("always", [False] * len(y_test))
score("hand", hand_rule(X_test))

model_two = LogisticRegression(max_iter=1000).fit(X_train[TWO], y_train)
guess_two = model_two.predict(X_test[TWO])
score("lr2", guess_two)
M["lr2_hour_coef"] = float(model_two.coef_[0][0])
M["lr2_depth_coef"] = float(model_two.coef_[0][1])
M["lr2_flagged"] = int(guess_two.sum())
M["lr2_line_at_noon"] = float(-(M["lr2_hour_coef"] * 12 + model_two.intercept_[0])
                              / M["lr2_depth_coef"])

score("lr", LogisticRegression(max_iter=1000).fit(X_train, y_train).predict(X_test))
score("nb", GaussianNB().fit(X_train, y_train).predict(X_test))
score("lrnd", LogisticRegression(max_iter=1000).fit(X_train[NO_DEPTH], y_train)
      .predict(X_test[NO_DEPTH]))
score("nbnd", GaussianNB().fit(X_train[NO_DEPTH], y_train).predict(X_test[NO_DEPTH]))
score("wide", hand_rule(X_test, 7, 19))
score("narrow", hand_rule(X_test, 11, 15))

per_year, year_prec, year_rec, biggest, year_blasts = {}, {}, {}, {}, {}
for yr in range(FIRST_YEAR, LAST_YEAR + 1):
    rows = events[events["year"] == str(yr)]
    per_year[yr] = float(f1_score(rows["is_blast"], hand_rule(rows), zero_division=0))
    year_prec[yr] = float(precision_score(rows["is_blast"], hand_rule(rows), zero_division=0))
    year_rec[yr] = float(recall_score(rows["is_blast"], hand_rule(rows), zero_division=0))
    biggest[yr] = float(rows["mag"].max())
    year_blasts[yr] = int(rows["is_blast"].sum())
# The last row of that table is NOT a whole year — the slice is pinned at END — so its blast
# count sits below every full year's and means nothing until the reader is told why.
M["last_year"] = LAST_YEAR
M["last_year_blasts"] = year_blasts[LAST_YEAR]
M["last_year_months"] = int(events[events["year"] == str(LAST_YEAR)]["time"].str[5:7].max())
M["full_year_blasts_lo"] = min(year_blasts[y] for y in year_blasts if y != LAST_YEAR)
M["full_year_blasts_hi"] = max(year_blasts[y] for y in year_blasts if y != LAST_YEAR)
worst_two = sorted(per_year, key=per_year.get)[:2]
M["year_worst"], M["year_second"] = sorted(worst_two)
M["year_best"] = max(per_year, key=per_year.get)
M["f1_worst"] = min(per_year.values())
M["f1_best"] = max(per_year.values())
M["n_worst_year"] = int((events["year"] == str(M["year_worst"])).sum())
M["n_second_year"] = int((events["year"] == str(M["year_second"])).sum())
M["n_year_before"] = int((events["year"] == str(M["year_worst"] - 1)).sum())
M["mag_worst_year"] = biggest[M["year_worst"]]
M["mag_second_year"] = biggest[M["year_second"]]
M["prec_worst_year"] = year_prec[M["year_worst"]]
M["prec_year_before"] = year_prec[M["year_worst"] - 1]
M["rec_worst_year"] = year_rec[M["year_worst"]]
M["rec_year_before"] = year_rec[M["year_worst"] - 1]

# The pinned split is ONE draw. The week's headline is an ordering of three F1 scores that sit
# within 0.012 of each other, so before any of it is written down, re-draw the split and find out
# how far these numbers move on their own. `random_state=42` stays pinned for everything the
# notebook reports; this is the scatter the reported numbers have to be read against.
#
# READ THE PAIRING, NOT THE RANGE. An earlier version of this build reported only the two ranges
# and the min/max of the gap, saw one negative excursion, and concluded the two TIE. They do not.
# Every seed scores both classifiers on the SAME held-out rows, so the gap is a paired difference,
# and a paired difference that keeps its sign in 8 draws out of 10 is the opposite of a straddle:
# the mean and the sign count are what settle it, and the range is what hides it. Week 8 taught
# exactly this, so a week that reads a range here teaches the wrong test three weeks later.
# The notebook prints 10 splits, which is what the prose quotes; 20 are run here because a
# standard deviation off 10 numbers is itself noisy, and the pin in course.yml records the 20.
NOTEBOOK_SEEDS, PIN_SEEDS = 10, 20
seed_hand, seed_lr, seed_nb = [], [], []
for seed in range(PIN_SEEDS):
    X_fit, X_held, y_fit, y_held = train_test_split(X, y, test_size=0.3, random_state=seed,
                                                    stratify=y)
    seed_hand.append(float(f1_score(y_held, hand_rule(X_held))))
    seed_lr.append(float(f1_score(y_held, LogisticRegression(max_iter=1000)
                                  .fit(X_fit, y_fit).predict(X_held))))
    seed_nb.append(float(f1_score(y_held, GaussianNB().fit(X_fit, y_fit).predict(X_held))))
seed_gap = [h - l for h, l in zip(seed_hand, seed_lr)]

shown_hand, shown_lr, shown_nb = (seed_hand[:NOTEBOOK_SEEDS], seed_lr[:NOTEBOOK_SEEDS],
                                  seed_nb[:NOTEBOOK_SEEDS])
shown_gap = seed_gap[:NOTEBOOK_SEEDS]
M["seed_n"] = NOTEBOOK_SEEDS
M["seed_hand_lo"], M["seed_hand_hi"] = min(shown_hand), max(shown_hand)
M["seed_lr_lo"], M["seed_lr_hi"] = min(shown_lr), max(shown_lr)
M["seed_nb_lo"], M["seed_nb_hi"] = min(shown_nb), max(shown_nb)
M["seed_gap_lo"], M["seed_gap_hi"] = min(shown_gap), max(shown_gap)
M["seed_gap_mean"] = sum(shown_gap) / len(shown_gap)
M["seed_hand_ahead"] = sum(1 for g in shown_gap if g > 0)
M["seed_lr_ahead"] = sum(1 for g in shown_gap if g < 0)
M["seed_level"] = sum(1 for g in shown_gap if g == 0)
M["pinned_gap"] = M["hand_f1"] - M["lr_f1"]
M["seed_worst_spread"] = max(max(shown_hand) - min(shown_hand), max(shown_lr) - min(shown_lr))

# The 20-seed paired summary. Not printed in the notebook — it is what course.yml pins, so that
# the next reader of the pin finds the statistic that settles the comparison rather than a range.
M["pin_seed_n"] = PIN_SEEDS
M["pin_gap_mean"] = sum(seed_gap) / PIN_SEEDS
M["pin_gap_sd"] = math.sqrt(sum((g - M["pin_gap_mean"]) ** 2 for g in seed_gap) / (PIN_SEEDS - 1))
M["pin_gap_t"] = M["pin_gap_mean"] / (M["pin_gap_sd"] / math.sqrt(PIN_SEEDS))
M["pin_hand_ahead"] = sum(1 for g in seed_gap if g > 0)
M["pin_lr_ahead"] = sum(1 for g in seed_gap if g < 0)

for k in sorted(M):
    print(f"{k:22s} {M[k]}")


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


def answer(model_answer, check=""):
    """A code answer cell. The solution carries the model answer; the student gets the stub."""
    solution = model_answer.strip("\n") + (("\n\n" + check.strip("\n")) if check else "")
    student = "# ← your answer here\n\n" + (("\n" + check.strip("\n")) if check else "")
    CELLS.append(("code", solution, student))


def answer_prose(model_answer):
    CELLS.append(("markdown", model_answer.strip("\n"),
                  "*(Double-click this cell and replace this line with your answer.)*"))


datahub = (f"{PLATFORM['datahub']}/hub/user-redirect/git-pull"
           f"?repo={PLATFORM['repo'].replace(':', '%3A').replace('/', '%2F')}"
           f"&branch={PLATFORM['branch']}"
           f"&urlpath=lab%2Ftree%2FEPS88_PyEarth%2F{PLATFORM['notebook_dir']}%2F{SLUG}.ipynb")

HOOK = """
The Comprehensive Nuclear-Test-Ban Treaty, opened for signature in 1996, bans every nuclear
explosion anywhere on Earth, and the way a ban like that is checked is that the planet listens.
Seismometers do not care what shook the ground. They record an earthquake, a landslide, a mine
collapse and a bomb in the same way, and somebody has to look at each recording and decide which
of those it was.

California will not hand you a nuclear test. It hands you the same decision, thousands of times a
year, at a smaller scale: quarries blast rock, the seismic network records the blasts alongside
real earthquakes, and a USGS analyst labels every event either `earthquake` or `quarry blast`.
That is tens of thousands of decisions somebody has already made — which is exactly what you need
if you want to find out whether a machine could have made them instead.

Today you build that classifier. Then you find out what it actually learned.
"""

md(weekkit.OPENING.format(question=WEEK["question"], datahub=datahub, hook=HOOK.strip()))

md(f"""
## What you'll be able to do

**The science.** Say what an event catalogue does and does not tell you about the source of a
seismic signal, name the things that make a quarry blast recognisable in one, and say which of
them would still be there if the event you were hunting were a secret nuclear test.

**The skills.** Split labelled data into a training half and a held-out half with `stratify`, fit
`LogisticRegression` and `GaussianNB` to it, and judge a classifier with a `confusion_matrix` and
with `precision_score`, `recall_score` and `f1_score` rather than with accuracy.

**Eight places where you write something: five in class, three at home.** Each one is headed
*Your turn*, with an empty cell under it.

**The four questions, in order:**

1. When one event in {M['one_in']} is a blast, what counts as getting it right?
2. What does a catalogue row actually know about a quarry blast?
3. Can a fitted model beat two conditions you wrote by hand?
4. Would the same numbers survive a different cut of the data?
""")

setup = weekkit.asset_setup_cell(
    imports=("import numpy as np\n"
             "from sklearn.model_selection import train_test_split\n"
             "from sklearn.linear_model import LogisticRegression\n"
             "from sklearn.naive_bayes import GaussianNB\n"
             "from sklearn.metrics import accuracy_score, precision_score, recall_score\n"
             "from sklearn.metrics import f1_score, confusion_matrix\n"),
    figsize="(7, 4)",
    cache_base=CACHE_BASE,
    unpack=f'''
COLUMNS = {COLUMNS}

# Both files came out of ONE query to the USGS catalogue, asked twice — once for each label:
#
#   https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&orderby=time-asc
#     &starttime={START}&endtime={END}&minmagnitude={MINMAG}
#     {BOX}
#     &eventtype=earthquake   ... and again with &eventtype=quarry%20blast
#
# Unlike most weeks, this one does not run that query for you. It cannot: the archive refuses any
# request matching more than 20,000 events and the earthquake half matches far more, so that half
# has to travel with the course as a file. Fetching only the blast half live would leave you
# joining this week's blasts to last summer's earthquakes — not the same slice of the catalogue,
# and every count in the text below would drift away from what your screen says. So both halves
# ship together, and the numbers you print are the numbers you read.
#
# The archive sends more columns than COLUMNS keeps. Hold on to the whole blast table too: two
# of the columns we are about to drop turn out to say where a blast was and how its depth was
# arrived at, and we come back for them.
blasts_all = pd.read_csv(CACHE + "/{BLAST_CACHE}")
blasts = blasts_all[COLUMNS]
quakes = pd.read_csv(CACHE + "/{QUAKE_CACHE}")
coast = pd.read_csv(CACHE + "/coastlines.csv")

print("columns:", list(quakes.columns))
print(blasts.head(2))
'''.strip("\n"))
code(setup)

# --- section 1 -------------------------------------------------------------
md(f"""
## When one event in {M['one_in']} is a blast, what counts as getting it right?

Both files came out of the same query: the same box of California and western Nevada, the same
years, the same magnitude floor. The only difference is what the analyst wrote in the `type`
column.

Stack them into one table. `pd.concat` takes a list of tables that share their columns and
returns one longer table, and `ignore_index=True` renumbers the rows from 0 instead of restarting
the count at the join. While you are there, turn the label into the column a classifier can
actually use: `True` when the event is a blast, `False` when it is an earthquake.
""")

code("""
events = pd.concat([quakes, blasts], ignore_index=True)
events["is_blast"] = events["type"] == "quarry blast"

print(events.shape)
print(events.head(3))
""")

md("""
Remember what a file like this is: *A catalogue lists what somebody's instruments recorded, not
what happened. Where there are no seismometers there are no earthquakes in the file.* Every label
in the `type` column was put there by a person, and that is going to matter more than it sounds
like it should.

First, how many of each.
""")

ask("""
### ✏️ Your turn 1

`events["type"].value_counts()` counts how many rows carry each label. Print it, then print how
many earthquakes this catalogue holds **for every one quarry blast** — one count divided by the
other, rounded to one decimal place.

**Use these names**, because the self-check looks for them: `n_blasts` and `n_quakes`.
""")

answer("""
counts = events["type"].value_counts()
n_blasts = counts["quarry blast"]
n_quakes = counts["earthquake"]

print(counts)
print("earthquakes per blast:", round(n_quakes / n_blasts, 1))
""", """
assert n_blasts < n_quakes, "blasts are the rare label here — if your n_blasts is the bigger of \
the two numbers, the two names are the wrong way round"
print("✓ the two classes —", n_blasts, "quarry blasts and", n_quakes,
      "earthquakes, a ratio of", round(n_quakes / n_blasts, 1), "to 1")
""")

# --- section 2: the second half of spine question 1 -------------------------
md(f"""
That ratio is the whole problem in one number. Quarry blasts are about {M['per_blast']:.0f} times
rarer than earthquakes here, and a class that rare changes what you are allowed to call success.

The obvious score is **accuracy**: out of every event, what fraction did you label correctly?
Before you compute it, commit to a number.

### Predict before you run

Here is a classifier that took no effort at all. It ignores its input and answers "earthquake"
every single time, so it never once flags a blast. What accuracy does it get on this catalogue?
Change `my_guess` to a fraction between 0 and 1, then run the cell.
""")

CELLS.extend(("code", s, a) for s, a in
             weekkit.predict_cell("0.50", "is the accuracy of a rule that always says earthquake"))

code("""
always_earthquake = [False] * len(events)
print("you guessed:  ", my_guess)
print("this rule got:", round(accuracy_score(events["is_blast"], always_earthquake), 4))
""")

md(f"""
A rule that is wrong about every single thing you care about is right
{M['frac_eq'] * 100:.2f}% of the time, because {M['frac_eq'] * 100:.2f}% of the catalogue is the
answer it always gives. That is what a rare class does to accuracy, and it is why nobody who
works on rare events reports accuracy on its own.

The two numbers reported instead split the question in half. *Of the ones you flagged, how many
were right? Of the real ones, how many did you catch?* The first is **precision**, the second is
**recall**, and they pull against each other: flag everything and your recall is perfect while
your precision collapses; flag almost nothing and the reverse. **F1** is the single number that
refuses to let you cheat either way. It is the harmonic mean of precision and recall, which is a
way of saying that it sits near the smaller of the two, so it is only good when both are.
""")

ask("""
### ✏️ Your turn 2

Score **both** ways of cheating, because the paragraph above claims they fail in opposite
directions and a claim like that is worth checking.

The first is the rule you just ran: `always_earthquake`, which never says blast. The second is its
mirror image — a rule that calls every single event a blast, which is `[True] * len(events)`.

With `events["is_blast"]` as the truth, print the precision, the recall and the F1 of each, using
`precision_score`, `recall_score` and `f1_score`. Six numbers, three per rule.

Each of those three takes `zero_division=0` as a third argument. It tells scikit-learn what to do
when a rule flags nothing at all: there is no *"of the ones you flagged"* left to divide by, so
count it as zero rather than stopping with an error.

**Use these names**, because the self-check looks for them: `always_precision`, `always_recall`
and `always_f1` for the first rule, then `always_blast` for the second rule's predictions and
`blast_precision`, `blast_recall` and `blast_f1` for its three scores.
""")

answer("""
always_precision = precision_score(events["is_blast"], always_earthquake, zero_division=0)
always_recall = recall_score(events["is_blast"], always_earthquake, zero_division=0)
always_f1 = f1_score(events["is_blast"], always_earthquake, zero_division=0)

always_blast = [True] * len(events)
blast_precision = precision_score(events["is_blast"], always_blast, zero_division=0)
blast_recall = recall_score(events["is_blast"], always_blast, zero_division=0)
blast_f1 = f1_score(events["is_blast"], always_blast, zero_division=0)

print("never says blast  — precision:", always_precision, " recall:", always_recall,
      " F1:", always_f1)
print("always says blast — precision:", round(blast_precision, 4), " recall:", blast_recall,
      " F1:", round(blast_f1, 4))
""", """
assert always_recall == 0 and blast_recall == 1, "recall answers 'of the real blasts, how many \
did you catch?' — a rule that never says blast catches none of them and a rule that always says \
blast catches every one, so those two recalls have to come out 0 and 1. If yours did not, check \
which score you put in which name"
print("✓ the two ways to cheat — never saying blast scores accuracy",
      round(accuracy_score(events["is_blast"], always_earthquake), 4), "and F1", always_f1,
      "; always saying blast scores recall", blast_recall, "and F1", round(blast_f1, 4))
""")

# --- section 3 -------------------------------------------------------------
md("""
## What does a catalogue row actually know about a quarry blast?

Six columns arrived with each event: when it happened, where, how deep, how big, and the label.
Anything a classifier learns has to come out of the first four, so look at them — starting with
*when*, because the times need a moment's work first.

`pd.to_datetime` turns the text into real timestamps. The catalogue records them in UTC, which is
no use for a question about a working day, so `.dt.tz_convert("US/Pacific")` moves them onto the
clock the quarry crew actually works to, and `.dt.hour` and `.dt.dayofweek` read the hour and the
day off each one (Monday is 0, Sunday is 6). Then *where*.
""")

code("""
local = pd.to_datetime(events["time"]).dt.tz_convert("US/Pacific")
events["hour"] = local.dt.hour
events["weekday"] = local.dt.dayofweek

blast_rows = events[events["is_blast"]]
quake_rows = events[events["type"] == "earthquake"]
few_quakes = quake_rows.iloc[::40]      # one earthquake in forty, or every plot is solid ink

print(events[["time", "hour", "weekday", "is_blast"]].head(3))
""")

code("""
plt.figure(figsize=(5, 5))            # California is nearly as tall as it is wide
plt.scatter(few_quakes["longitude"], few_quakes["latitude"], s=2, color="0.7",
            label="earthquakes")
plt.scatter(blast_rows["longitude"], blast_rows["latitude"], s=2, color="firebrick",
            label="blasts")
plt.plot(coast["lon"], coast["lat"], color="0.3", lw=0.6)
plt.xlim(-125, -114)
plt.ylim(32, 42)
plt.gca().set_aspect("equal")
plt.xlabel("degrees east")
plt.ylabel("degrees north")
plt.title(f"{len(blast_rows)} blasts, {len(few_quakes)} of {len(quake_rows)} earthquakes")
plt.legend()
plt.show()
""")

md("""
The earthquakes are spread over the whole box, in broad belts hundreds of kilometres long: those
are the region's active belts. The long one down the coast is the boundary where the Pacific and
North American plates grind past each other — and the first thing this map tells you is that the
boundary is not a line. The scatter filling the right-hand side belongs to it too: the belts
running up the eastern side of the map, inland of the Sierra Nevada, are the same two plates
sliding past each other in the same direction, a few hundred kilometres further east, and a
large share of the motion is taken up out there rather than on the coastal faults. The densest
knot on the whole map sits near 35.8 N, -117.6 — one earthquake sequence, from 2019, that you
meet again later in this notebook — and the tight cluster in the bottom right corner, below the
Salton Sea, is where the boundary itself comes ashore. Only the sparse north-eastern corner of
the box, beyond those belts, is crust that is genuinely pulling apart rather than sliding past
itself, and you can see how few events it holds. The dense knot at about 37.6 N, -118.9 is the
volcanic swarm under Long Valley, and the tight cluster on the coast near 40.3 N, -124.5 is the
Mendocino triple junction, where three plates meet at once. The blasts are not spread at all.
They sit in small tight clumps, because a quarry is a fixed hole in the ground that gets blasted
again and again for decades. That is already a usable clue, and also a warning — a model that
learns *where* the quarries are has learned a list of addresses, not a piece of physics.

Now *how deep*. The two labels are wildly different in number, so `density=True` scales each
histogram to the same total area; without it the blasts would be invisible.
""")

code("""
depth_bins = np.arange(-4, 25.5, 0.5)
plt.hist(quake_rows["depth"], bins=depth_bins, density=True, label="earthquakes")
plt.hist(blast_rows["depth"], bins=depth_bins, density=True, alpha=0.6, label="blasts")
plt.axvline(0, color="black", lw=1)
plt.xlabel("depth (km; negative means above sea level)")
plt.ylabel("share of events per km of depth")
plt.title(f"depth of {len(quake_rows)} earthquakes and {len(blast_rows)} blasts")
plt.legend()
plt.show()

print("median depth — blasts:", blast_rows["depth"].median(),
      " earthquakes:", quake_rows["depth"].median())
print("share deeper than the 25 km the axis reaches — blasts:",
      round((blast_rows["depth"] > 25).mean(), 4),
      " earthquakes:", round((quake_rows["depth"] > 25).mean(), 4))
""")

md(f"""
The blasts pile up to the **left** of the black line. Not at zero — below it, at negative depths,
which sounds like nonsense until you remember that depth in this catalogue is measured down from
sea level and a quarry is a hole in a hillside several hundred metres up. The median blast sits at
{M['blast_depth_median']} km, that is {abs(M['blast_depth_median']) * 1000:.0f} metres *above*
sea level, against a median earthquake {M['quake_depth_median']} km below it. (The axis stops at
25 km, which leaves {M['quake_deeper_25'] * 100:.1f}% of the earthquakes off the right-hand side
and no blasts at all.)

That looks like a gift. Look at the actual values before you accept it.
""")

code("""
print(blast_rows["depth"].value_counts().head(5))
print("distinct depth values, blasts:     ", blast_rows["depth"].nunique())
print("distinct depth values, earthquakes:", quake_rows["depth"].nunique())

# How far apart are the blasts that share one repeated depth? Ask in BOTH directions, and in
# kilometres rather than degrees: a degree of latitude is 111.32 km anywhere on Earth, but a
# degree of longitude is that times the cosine of the latitude you are standing at — about 0.8
# of it here — so the two spans cannot be compared until they are converted.
for repeated in blast_rows["depth"].value_counts().index[:2]:
    same_depth = blast_rows[blast_rows["depth"] == repeated]
    lat_span = same_depth["latitude"].max() - same_depth["latitude"].min()
    lon_span = same_depth["longitude"].max() - same_depth["longitude"].min()
    lon_km = lon_span * 111.32 * np.cos(np.deg2rad(same_depth["latitude"].mean()))
    print(repeated, "km:", len(same_depth), "blasts spanning",
          round(lat_span, 2), "deg latitude =", int(round(lat_span * 111.32)), "km, and",
          round(lon_span, 2), "deg longitude =", int(round(lon_km)), "km")
""")

code("""
top_depth = blast_rows["depth"].value_counts().index[0]
second_depth = blast_rows["depth"].value_counts().index[1]

# `place` and `depthError` arrived with the blasts and were dropped when we cut down to COLUMNS.
# `place` reads like "5km NNW of Boron, CA" — but the distance, the bearing, the space after the
# number and even the state's name change from row to row, so counting that text as it stands
# scatters one quarry over dozens of rows. Keep only what sits between " of " and the comma and
# each site is counted once.
blasts_all["site"] = blasts_all["place"].str.split(" of ").str[-1].str.split(",").str[0]

print("the", (blasts_all["depth"] == top_depth).sum(), "blasts at", top_depth, "km:")
print(blasts_all[blasts_all["depth"] == top_depth]["site"].value_counts())
print("the", (blasts_all["depth"] == second_depth).sum(), "blasts at", second_depth, "km:")
print(blasts_all[blasts_all["depth"] == second_depth]["site"].value_counts().head(3))
print(blasts_all["depthError"].value_counts().head(3))
""")

md(f"""
{M['top_depth_n']} separate blasts share the depth {M['top_depth']} km — the same value to the
nearest 10 metres, which is as fine as this catalogue quotes a blast depth at all
({M['depth_grid_share'] * 100:.0f}% of them sit on that 10 m grid). The obvious reading is that
they are one quarry, and asking in both directions already strains it. Those {M['top_depth_n']}
events sit within {M['top_depth_lat_km']:.0f} km of each other north to south — the district runs
east–west, so the latitude span was always going to look tight — but
{M['top_depth_lon_km']:.0f} km apart east to west, and `place`, one of the columns the archive
sent and we dropped, names two towns: {M['top_site_a_n']} of the blasts at {M['top_site_a']} and
{M['top_site_b_n']} at {M['top_site_b']}. Two quarries, one number. You could still argue that
away, though. Two pits {M['top_depth_lon_km']:.0f} km apart on the same desert plateau might
genuinely sit at the same height to the nearest 10 metres; that would be a coincidence, not an
impossibility.

The next row down cannot be argued away. {M['second_depth_n']} blasts share
{M['second_depth']} km, and {M['second_site_a_n']} of them —
{M['second_site_share'] * 100:.0f}% — are at {M['second_site_a']}: the same quarry that has
already been handed {M['top_depth']} km. One site, two different depths, so neither of them can
be its elevation. The rest of that row is scattered over {M['second_depth_lat_km']:.0f} km of
latitude, from {M['second_site_a']} in the southern desert to {M['second_site_b']} in the far
north of the state. And `depthError` settles it. On {M['depth_err_n']:,} of the
{M['n_blasts']:,} blasts — {M['depth_err_share'] * 100:.0f}% — it is the single constant
{M['depth_err_top']} km. An uncertainty of {M['depth_err_top']} km attached to an event placed
{abs(M['top_depth']) * 1000:.0f} metres above sea level is not a statement about that event; it
is what a fixed depth looks like when the routine that would have solved for one never ran.

**The depth was not measured. It was set** — once an analyst recognises an event as a quarry
blast, they hold its depth at a value they choose. The column is not a property of the ground
shaking at all. It is a note about a decision the analyst had already made.

That is what **leakage** looks like, and it comes with a plain-language test.

> You got 99 percent. Be suspicious. Did one of your columns already know the answer?

Keep depth for now. You will take it away in the homework and see what is left. Last, the clock.
""")

code("""
hour_bins = np.arange(0, 25, 1)
plt.hist(quake_rows["hour"], bins=hour_bins, density=True, label="earthquakes")
plt.hist(blast_rows["hour"], bins=hour_bins, density=True, alpha=0.6, label="blasts")
plt.xlabel("hour of the day, local California time")
plt.ylabel("share of events per hour")
plt.title(f"time of day, {len(quake_rows)} earthquakes and {len(blast_rows)} blasts")
plt.legend()
plt.show()

work = (blast_rows["hour"] >= 10) & (blast_rows["hour"] < 17)
print("between 10 and 17 — blasts:", round(work.mean(), 3), " earthquakes:",
      round(((quake_rows["hour"] >= 10) & (quake_rows["hour"] < 17)).mean(), 3))
print("Monday to Friday  — blasts:", round((blast_rows["weekday"] <= 4).mean(), 3),
      " earthquakes:", round((quake_rows["weekday"] <= 4).mean(), 3))
""")

md(f"""
The earthquakes are flat across the day, which is what a process that has never heard of a clock
should look like. The blasts are a working day: {M['blast_workhours'] * 100:.1f}% of them fall
between 10 in the morning and 5 in the afternoon against {M['quake_workhours'] * 100:.1f}% of the
earthquakes, and {M['blast_weekdays'] * 100:.1f}% land Monday to Friday against
{M['quake_weekdays'] * 100:.1f}%.
""")

# --- section 4 -------------------------------------------------------------
md(f"""
## Can a fitted model beat two conditions you wrote by hand?

So the catalogue offers three kinds of clue: an address that repeats, a depth somebody typed in,
and a human timetable. Before any model, the rule you already have in your head:

> Write the dumbest rule you can, first. Any model that cannot beat it is decoration.

That is the **baseline**. Its job is not to be good; its job is to be the number everything
clever has to clear afterwards, because a model that scores below it has taught you nothing
except that you can call `.fit`.

Comparing anything fairly needs data that neither the rule nor the models were allowed to see, so
split the catalogue in two. One argument here is new: `stratify=y` forces the rare class into both
halves in the same proportion. Without it a random 30% could easily take a lopsided share of the
blasts, and the test score would be measuring the split rather than the classifier.
""")

code(weekkit.CHECKPOINT.format(body='''events = pd.concat([quakes, blasts], ignore_index=True)
events["is_blast"] = events["type"] == "quarry blast"
local = pd.to_datetime(events["time"]).dt.tz_convert("US/Pacific")
events["hour"] = local.dt.hour
events["weekday"] = local.dt.dayofweek
blast_rows = events[events["is_blast"]]
quake_rows = events[events["type"] == "earthquake"]
few_quakes = quake_rows.iloc[::40]'''))

code(f"""
features = {FEATURES}
X = events[features]
y = events["is_blast"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,
                                                    stratify=y)

print("train:", len(X_train), "events,", y_train.sum(), "of them blasts")
print("test: ", len(X_test), "events,", y_test.sum(), "of them blasts")
""")

md("""
The two clues that need no computer are the two you just plotted: a blast is at or above sea
level, and a blast happens in the middle of a working day. Write that down as one line of Python.

`&` is the array version of `and`. It asks the question of every row at once, the way `earth < 0`
asked one question of every cell of an elevation grid, and each condition needs its own brackets
because `&` binds more tightly than `<=` does.
""")

ask("""
### ✏️ Your turn 3

Build the baseline and score it on the held-out half.

`hand_rule` should be `True` where **both** of these hold for a row of `X_test`: `depth` is at or
below 0, and `hour` is at least 10 and less than 17. Then print its precision, its recall and its
F1 against `y_test` — the same three calls as your turn 2.

**Use these names**, because the self-check looks for them: `hand_rule`.
""")

answer("""
hand_rule = (X_test["depth"] <= 0) & (X_test["hour"] >= 10) & (X_test["hour"] < 17)

print("precision:", round(precision_score(y_test, hand_rule), 4))
print("recall:   ", round(recall_score(y_test, hand_rule), 4))
print("F1:       ", round(f1_score(y_test, hand_rule), 4))
""", """
assert len(hand_rule) == len(y_test), "score the rule on X_test, not on the whole catalogue"
print("✓ the baseline — two conditions, F1",
      round(f1_score(y_test, hand_rule), 4), "on", len(y_test), "held-out events")
""")

# --- section 5: the second part of spine question 3 -------------------------
md(f"""
Two conditions, nothing fitted to anything, F1 {M['hand_f1']:.4f}. Write that number down;
everything from here is measured against it. Beating it will have to mean clearing it by more
than the number wanders on its own when the split is re-cut — hold that thought, because you
measure how far it wanders before the end of the hour.

**Logistic regression** is the straight-line fit from earlier in the course, bent to answer a
yes-or-no question.

> The same straight line — but now it outputs a probability between 0 and 1.

Where that line lands is the **decision boundary**: the model answers "blast" on one side of it
and "earthquake" on the other, and a straight boundary is the only thing this model can ever draw.
`.fit(X_train, y_train)` finds it and `.predict` applies it — the same two calls you used to fit a
regression line.

### Predict before you run

Your two hand-written conditions used `depth` and `hour` and scored F1 {M['hand_f1']:.4f}. Give
logistic regression exactly the same two columns, and {M['n_train']:,} labelled events to fit on.
What F1 does it get? Change `my_guess` and run.
""")

CELLS.extend(("code", s, a) for s, a in
             weekkit.predict_cell("0.85", "is the F1 logistic regression gets on hour and depth"))

code("""
model_two_columns = LogisticRegression(max_iter=1000).fit(X_train[["hour", "depth"]], y_train)
guess_two_columns = model_two_columns.predict(X_test[["hour", "depth"]])

print("you guessed:", my_guess)
print("it scored:  ", round(f1_score(y_test, guess_two_columns), 4))
print("it called", guess_two_columns.sum(), "events blasts;", y_test.sum(), "really are")
""")

md(f"""
Not a little worse than two `if` conditions — several times worse. It flagged
{M['lr2_flagged']} events as blasts, out of {M['blasts_test']} real ones in the held-out half.
Draw the line it found and the reason is visible.

`model_two_columns.coef_[0]` holds one number per column and `model_two_columns.intercept_[0]` the
offset, and the boundary is where they add to zero:
`hour_coef * hour + depth_coef * depth + intercept == 0`. Rearranged for depth, that is a line you
can plot.
""")

code("""
plt.scatter(few_quakes["hour"], few_quakes["depth"], s=3, color="0.7", label="earthquakes")
plt.scatter(blast_rows["hour"], blast_rows["depth"], s=3, color="firebrick", label="blasts")

hours = np.arange(0, 24)
hour_coef, depth_coef = model_two_columns.coef_[0]
boundary = -(hour_coef * hours + model_two_columns.intercept_[0]) / depth_coef
plt.plot(hours, boundary, color="black", lw=2, label="decision boundary")

plt.ylim(-4, 25)
plt.xlabel("hour of the day, local California time")
plt.ylabel("depth (km; negative means above sea level)")
plt.title(f"{len(blast_rows)} blasts and {len(few_quakes)} earthquakes, with the boundary")
plt.legend()
plt.show()

print("at midday the boundary sits at", round(boundary[12], 2), "km")
""")

md(f"""
The model answers *blast* below that line and *earthquake* above it, and the line is almost flat.
It sits at about {M['lr2_line_at_noon']:.2f} km and barely tilts, which means the model threw the
clock away and kept only *very shallow* — and it put its depth cut well below zero rather than at
zero, because with one blast for every {M['per_blast']:.0f} earthquakes it has to be extremely
sure before it dares say blast at all. Almost every red point sits above the line, which is why so
few of them were flagged.

It threw the clock away because it could not use it. Blasts happen in the *middle* of the day, and
a middle is not a side of a line. A straight boundary can say *later than 10* or *earlier than 5*;
saying both at once needs a corner, and a straight line has none. Your two `if` conditions had a
corner. That is the whole difference, and it is what the week on trees and forests comes back for.

Meanwhile the model has been working with one hand tied: it saw two of the six columns you
prepared. Give it all of them.
""")

code("""
model_logistic = LogisticRegression(max_iter=1000).fit(X_train, y_train)
guess_logistic = model_logistic.predict(X_test)

print("accuracy: ", round(accuracy_score(y_test, guess_logistic), 4))
print("precision:", round(precision_score(y_test, guess_logistic), 4))
print("recall:   ", round(recall_score(y_test, guess_logistic), 4))
print("F1:       ", round(f1_score(y_test, guess_logistic), 4))
""")

md("""
Hold your reaction to that F1 until you have seen where the mistakes are, because one number hides
which of the two kinds a classifier is making. The **confusion matrix** is the table that
separates them: one row per true label, one column per predicted label, so the off-diagonal cells
are the misses and the false alarms, counted apart.
""")

ask("""
### ✏️ Your turn 4

Print the confusion matrix of the six-column model with
`confusion_matrix(y_test, guess_logistic)`. It comes back as a 2 by 2 grid of counts: the top row
is the events that really were earthquakes and the bottom row the ones that really were blasts,
and inside each row the first column is "the model said earthquake" and the second "the model said
blast".

Then print the two mistakes on their own lines — the blasts it missed, and the earthquakes it
falsely flagged. Then one more printed line, in words: which of the two mistakes is it making more
of — and which of the two would matter more to somebody verifying a test ban?

**Use these names**, because the self-check looks for them: `matrix`.
""")

answer("""
matrix = confusion_matrix(y_test, guess_logistic)
print(matrix)

print("blasts missed:              ", matrix[1][0])
print("earthquakes falsely flagged:", matrix[0][1])

print("It misses more blasts than it falsely flags earthquakes. To somebody verifying a test ban "
      "the misses are the worse of the two: a missed event is an explosion nobody looked at, "
      "while a false alarm only costs an analyst the time it takes to read the waveform and say "
      "no.")
""", """
assert matrix.sum() == len(y_test), "the matrix should count every held-out event exactly once"
print("✓ the confusion matrix —", matrix[1][1], "blasts caught,", matrix[1][0],
      "missed and", matrix[0][1], "earthquakes falsely flagged")
""")

# --- section 6: the third part of spine question 3 --------------------------
md("""
There is a second way to use the same six columns, and it draws nothing at all.

> What does a quarry blast usually look like? Shallow, weekday, mid-afternoon. Score each clue
> and multiply. Pretending the clues are independent is obviously wrong, and it works anyway.

That is **Naive Bayes**. It learns, one column at a time, what values blasts tend to have and what
values earthquakes tend to have; then for a new event it multiplies the clues together and takes
whichever label comes out ahead. The naive part is the multiplying, which assumes the clues are
independent of one another — and here they plainly are not, since a shallow event in this
catalogue is *more* likely to be at 2pm, not equally likely. `GaussianNB` is the version that
treats each column as a bell curve, and it is used through the identical `.fit` and `.predict`.
""")

ask("""
### ✏️ Your turn 5

Fit `GaussianNB()` on `X_train` and `y_train`, predict on `X_test`, and print its precision,
recall and F1.

Then print, one per line, the four F1 scores this notebook has produced, so they can be read
together: the always-earthquake rule, your `hand_rule` from your turn 3, the six-column logistic
regression (its predictions are in `guess_logistic`), and this one. Then one line: does either
fitted model clear the hand rule, and by how much?

**Use these names**, because the self-check looks for them: `model_bayes` and `guess_bayes`.
""")

answer("""
model_bayes = GaussianNB().fit(X_train, y_train)
guess_bayes = model_bayes.predict(X_test)

print("precision:", round(precision_score(y_test, guess_bayes), 4))
print("recall:   ", round(recall_score(y_test, guess_bayes), 4))
print("F1:       ", round(f1_score(y_test, guess_bayes), 4))

print("always earthquake:  ", round(f1_score(y_test, [False] * len(y_test), zero_division=0), 4))
print("hand rule:          ", round(f1_score(y_test, hand_rule), 4))
print("logistic regression:", round(f1_score(y_test, guess_logistic), 4))
print("naive Bayes:        ", round(f1_score(y_test, guess_bayes), 4))

print("Neither fitted model clears the hand rule: logistic regression finishes",
      round(f1_score(y_test, hand_rule) - f1_score(y_test, guess_logistic), 4),
      "of F1 short of it, and naive Bayes",
      round(f1_score(y_test, hand_rule) - f1_score(y_test, guess_bayes), 4), "short.")
""", """
assert len(guess_bayes) == len(y_test), "predict on X_test, the half the model was not fitted to"
print("✓ naive Bayes — F1", round(f1_score(y_test, guess_bayes), 4), "on the same",
      len(y_test), "held-out events")
""")

# --- section 7 -------------------------------------------------------------
md("""
## Would the same numbers survive a different cut of the data?

One split of one catalogue gives one number, and a number that only holds for the window you
happened to pick is not a result. There are two ways to find out whether these numbers are
results, and the cheaper one first: the hand rule has nothing fitted to anything, so it can be
scored on every event of every year with no risk of cheating. Do that.

You wrote those two conditions once, in your turn 3. From here you need them again on every year,
again on every re-cut split, and twice more in the homework — so give them a name first. `def`
does that, and putting the two hour bounds in as arguments means every line below says out loud
which window it is asking about.
""")

code(f"""
def two_condition_rule(rows, low, high):
    \"\"\"True where a row is at or above sea level AND its hour falls inside the window.\"\"\"
    # the window is an argument rather than fixed at 10 and 17 because you change it later
    return (rows["depth"] <= 0) & (rows["hour"] >= low) & (rows["hour"] < high)


events["year"] = events["time"].str[:4]

for year in range({FIRST_YEAR}, {LAST_YEAR + 1}):
    rows = events[events["year"] == str(year)]
    guess_year = two_condition_rule(rows, 10, 17)
    print(year, len(rows), "events ", rows["is_blast"].sum(), "blasts  Mmax", rows["mag"].max(),
          " precision", round(precision_score(rows["is_blast"], guess_year), 3),
          " recall", round(recall_score(rows["is_blast"], guess_year), 3),
          " F1", round(f1_score(rows["is_blast"], guess_year), 3))
""")

md(f"""
The baseline holds: every year between {M['f1_worst']:.3f} and {M['f1_best']:.3f}, no drift, no
year where it falls apart. Read the last row as what it is, though: the catalogue is cut off at
the end of August, so the {M['last_year']} row covers only the first {M['last_year_months']}
months of that year, and its {M['last_year_blasts']} blasts are not a fall in blasting — a full
year here runs {M['full_year_blasts_lo']} to {M['full_year_blasts_hi']}. Its F1 is comparable
with the rest, because a score is a rate and a rate does not care how long you watched. Its
counts are not.

The two lowest F1 scores are {M['year_worst']} and {M['year_second']}, and the
same printout says why. Those are the two years the catalogue swells — {M['n_worst_year']:,} and
{M['n_second_year']:,} events against {M['n_year_before']:,} the year before — and the two years
holding the biggest earthquakes in the file, M{M['mag_worst_year']} and M{M['mag_second_year']}. A
large earthquake is followed by aftershocks for months afterwards, and the rule has to say
something about every one of them, while the number of blasts to be caught stays where it was.
Read the two columns and you can see which half of the score gave way: recall barely moves
({M['rec_year_before']:.3f} to {M['rec_worst_year']:.3f} — the same blasts, still caught), while
precision falls from {M['prec_year_before']:.3f} to {M['prec_worst_year']:.3f}, because there are
three times as many earthquakes for the rule to be wrong about.

The years were the cheap way to ask. The second way goes at the comparison itself. Your three F1
scores — hand rule, logistic regression, naive Bayes — all came out of one random cut of the
catalogue into two halves, and `random_state=42` is nothing more than the number that decided
which rows went which way. Change it, refit, rescore, and see how much of the difference between
the three was ever there.

Keep the difference itself as you go, in a column of its own. Every split scores the rule and the
model on **the same** held-out events, so subtracting one score from the other on each split is a
fair, like-for-like comparison — which the two columns read separately are not. You have met the
question *is this difference real?* before, and it is a question about this column: how big it
is on average and whether it keeps its sign, not how wide either of the other two columns is.
""")

code(f"""
def score_one_split(seed):
    \"\"\"Re-cut the catalogue with a different shuffle and score all three predictors on it.\"\"\"
    # a new seed sends different rows into each half; stratify keeps the blasts as rare in
    # each half as they are in the whole catalogue, exactly as it did for the pinned split
    X_fit, X_held, y_fit, y_held = train_test_split(X, y, test_size=0.3,
                                                    random_state=seed, stratify=y)
    # the hand rule was fitted to nothing, so it can be scored straight off the held-out half
    rule_f1 = f1_score(y_held, two_condition_rule(X_held, 10, 17))
    # each model is fitted on X_fit alone — then all three are judged on the SAME X_held
    logistic_f1 = f1_score(y_held,
                           LogisticRegression(max_iter=1000).fit(X_fit, y_fit).predict(X_held))
    bayes_f1 = f1_score(y_held, GaussianNB().fit(X_fit, y_fit).predict(X_held))
    return rule_f1, logistic_f1, bayes_f1


gaps = []
for seed in range({M['seed_n']}):
    rule_f1, logistic_f1, bayes_f1 = score_one_split(seed)

    gaps.append(rule_f1 - logistic_f1)
    print("split", seed, " hand rule", round(rule_f1, 4), " logistic", round(logistic_f1, 4),
          " naive Bayes", round(bayes_f1, 4),
          " gap", round(rule_f1 - logistic_f1, 4))

print("the gap column — average", round(sum(gaps) / len(gaps), 4),
      " in the rule's favour on", sum(1 for gap in gaps if gap > 0), "of", len(gaps),
      "splits, against it on", sum(1 for gap in gaps if gap < 0))
""")

# --- the question, answered ------------------------------------------------
md(f"""
## The question, answered

**Not from a catalogue.** On the {M['n_test']:,} held-out events of the pinned split, two
hand-written conditions scored F1 {M['hand_f1']:.4f}, logistic regression on six columns scored
{M['lr_f1']:.4f} and naive Bayes {M['nb_f1']:.4f}. Read alone, those three look like an ordering,
and they are not one: the {M['pinned_gap']:.4f} separating rule from model on this split is far
less than either number moves when nothing changes but the shuffle. Across {M['seed_n']} cuts the
hand rule ran {M['seed_hand_lo']:.4f} to {M['seed_hand_hi']:.4f} and logistic regression
{M['seed_lr_lo']:.4f} to {M['seed_lr_hi']:.4f}, which between them cover that gap several times
over. Nothing in this notebook says a fitted model beat the baseline.

It does not say the two are level either, and the gap column is why. The rule finished ahead on
{M['seed_hand_ahead']} of the {M['seed_n']} splits, level on {M['seed_level']} and behind on
{M['seed_lr_ahead']}, averaging {M['seed_gap_mean']:+.4f}. Read the two ranges and you see them
overlap; read the paired difference and you see it keep its sign in {M['seed_hand_ahead']} draws
out of {M['seed_n']}, which is not what a coin flip looks like. So the honest verdict is neither
a victory nor a tie: **two lines of Python are consistently ahead of both fitted models, by an
amount too small to be worth having.** Six columns, {M['n_train']:,} labelled examples and two
fitted classifiers bought you nothing over two conditions you wrote before any of it — and that,
not a ranking, is the result. None of the three is good enough to hand a decision that matters.

The reason is what the clues are made of. Real discrimination of an explosion from an earthquake
is done on the waveform, not on a catalogue row. An explosion is a sudden push outward from a
point at or near the surface, so it radiates compressional P energy in every direction and makes
comparatively feeble shear and surface waves, while an earthquake is rock sliding past rock on a
fault, which is efficient at making exactly those waves. That contrast is what the ratio of P to S
amplitude and the comparison of body-wave with surface-wave magnitude are built to measure, and
none of it is in the six columns you had. Depth would be a genuine discriminant — nobody buries a
device kilometres down — except that in this file the depth of a blast is not measured but
assigned.

What you classified on instead was a quarry's routine: a fixed address, a working day, a weekday,
and an analyst's convention about depth. Whether any of those four would still be there for the
event you actually care about is the last thing the homework asks you. Either way, this is why
treaty verification reads waveforms rather than catalogues — and why a small F1 on the wrong
features is a more useful thing to report than a large one.
""")

# --- summary and homework --------------------------------------------------
md(weekkit.week_cheatsheet(10))

md("""
## Homework

Three parts, all on the table you already have. If you have restarted since class, run the setup
cell at the top, then the checkpoint below, which rebuilds everything class left in memory.
""")

code(weekkit.CHECKPOINT.format(body=f'''events = pd.concat([quakes, blasts], ignore_index=True)
events["is_blast"] = events["type"] == "quarry blast"
local = pd.to_datetime(events["time"]).dt.tz_convert("US/Pacific")
events["hour"] = local.dt.hour
events["weekday"] = local.dt.dayofweek
features = {FEATURES}
X = events[features]
y = events["is_blast"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,
                                                    stratify=y)


def two_condition_rule(rows, low, high):
    """True where a row is at or above sea level AND its hour falls inside the window."""
    return (rows["depth"] <= 0) & (rows["hour"] >= low) & (rows["hour"] < high)'''))

ask(f"""
### ✏️ Your turn 6

Class left one column under suspicion. Take it away and find out what the models were standing on.

Build `features_no_depth` — the same list as `features`, without `"depth"` — then fit a fresh
`LogisticRegression(max_iter=1000)` and a fresh `GaussianNB()` on `X_train[features_no_depth]` and
`y_train`, predict on `X_test[features_no_depth]`, and print the F1 of each. Print beside them the
F1 of the always-earthquake rule, so you have something to read them against; that one needs
`zero_division=0` again. With depth, the two models scored {M['lr_f1']:.4f} and {M['nb_f1']:.4f}.

Finish by printing one more line that answers the question, in words and on your three numbers:
which of the two models lost more when depth went, and is what either of them has left a
classifier at all?

**Use these names**, because the self-check looks for them: `features_no_depth`,
`f1_logistic_no_depth` and `f1_bayes_no_depth`.
""")

answer(f"""
features_no_depth = {NO_DEPTH}

model_logistic_no_depth = LogisticRegression(max_iter=1000).fit(X_train[features_no_depth],
                                                                y_train)
model_bayes_no_depth = GaussianNB().fit(X_train[features_no_depth], y_train)

f1_logistic_no_depth = f1_score(y_test, model_logistic_no_depth.predict(X_test[features_no_depth]),
                                zero_division=0)
f1_bayes_no_depth = f1_score(y_test, model_bayes_no_depth.predict(X_test[features_no_depth]),
                             zero_division=0)

print("logistic regression, no depth:", round(f1_logistic_no_depth, 4))
print("naive Bayes, no depth:        ", round(f1_bayes_no_depth, 4))
print("always earthquake:            ",
      round(f1_score(y_test, [False] * len(y_test), zero_division=0), 4))

print("Logistic regression lost more — it fell all the way to the always-earthquake rule's own "
      "F1, so it stopped flagging blasts entirely; naive Bayes is barely above that. Neither is "
      "a classifier of quarry blasts any more.")
""", """
assert "depth" not in features_no_depth, "the point of this part is to leave depth out"
print("✓ without depth — logistic regression F1", round(f1_logistic_no_depth, 4),
      "and naive Bayes F1", round(f1_bayes_no_depth, 4))
""")

ask(f"""
### ✏️ Your turn 7

Class chose 10:00 to 17:00 for "working hours" and never argued about it. It is a choice, and this
one is yours. Take **one** of these, not both:

- **wide**, 7 to 19 — catch the early and the late blasts as well
- **narrow**, 11 to 15 — only the solid middle of the day

Set `my_low` and `my_high` to the window you chose, then build `my_rule` by calling the class
function on the held-out half with your two numbers in place of 10 and 17:
`two_condition_rule(X_test, my_low, my_high)`. Print its precision, its recall and its F1. Then
build the 10-to-17 window class used the same way, score it too, and print it underneath so the
two are side by side.

Then say it, in one more printed line: which of precision and recall did your window buy, which
did it sell, and would you defend that trade to somebody who has to act on the flags?

**Use these names**, because the self-check looks for them: `my_low`, `my_high` and `my_rule`.
""")

answer("""
# You were asked for ONE window. This answer works both of them, so whichever you picked you can
# read your own numbers off it — and see what the choice you did not make would have bought.
my_low = 7
my_high = 19

my_rule = two_condition_rule(X_test, my_low, my_high)
hand_rule = two_condition_rule(X_test, 10, 17)
other_rule = two_condition_rule(X_test, 11, 15)

print("mine, 7 to 19:  precision", round(precision_score(y_test, my_rule), 4),
      " recall", round(recall_score(y_test, my_rule), 4),
      " F1", round(f1_score(y_test, my_rule), 4))
print("class, 10 to 17: precision", round(precision_score(y_test, hand_rule), 4),
      " recall", round(recall_score(y_test, hand_rule), 4),
      " F1", round(f1_score(y_test, hand_rule), 4))
print("the other choice, 11 to 15: precision", round(precision_score(y_test, other_rule), 4),
      " recall", round(recall_score(y_test, other_rule), 4),
      " F1", round(f1_score(y_test, other_rule), 4))

print("My wider window bought recall and sold precision:",
      round(recall_score(y_test, my_rule) - recall_score(y_test, hand_rule), 4),
      "more recall for",
      round(precision_score(y_test, hand_rule) - precision_score(y_test, my_rule), 4),
      "less precision. I would not defend that trade to somebody who has to act on the flags,",
      "because it makes", round(100 * (1 - precision_score(y_test, my_rule))),
      "percent of everything it flags a false alarm.")
""", """
assert (my_low, my_high) == (7, 19) or (my_low, my_high) == (11, 15), "pick one of the two \
windows offered — wide is 7 to 19, narrow is 11 to 15 — not the 10 to 17 class used"
print("✓ your window —", my_low, "to", my_high, "gives precision",
      round(precision_score(y_test, my_rule), 4), "and recall",
      round(recall_score(y_test, my_rule), 4))
""")

ask("""
### ✏️ Your turn 8

Two or three sentences, quoting your own printed numbers.

Your turn 6 gave you an F1 for logistic regression with depth and an F1 for the same model without
it. Quote both, say what the gap between them tells you about what the model had actually learned,
and say whether you would report the first of the two as a result. Then, in one more sentence:
name one clue this week used that would still be there if the event you were trying to catch were
a secret nuclear test rather than a quarry — or say that none would, and why.
""")

answer_prose(f"""
With depth, my logistic regression scored F1 {M['lr_f1']:.4f}; with the depth column taken out and
nothing else changed, it scored {M['lrnd_f1']:.4f}, which is the same F1 as the rule that answers
"earthquake" every time, and naive Bayes fell from {M['nb_f1']:.4f} to {M['nbnd_f1']:.4f}, which is
barely different from it. So essentially all of the apparent skill was coming from one column, and
that column is not a measurement: {M['top_depth_n']} blasts share the single depth
{M['top_depth']} km to the nearest 10 metres across two different quarries, {M['top_site_a']} and
{M['top_site_b']}, and `depthError` carries the same constant {M['depth_err_top']} km on
{M['depth_err_share'] * 100:.0f}% of the blasts, so somebody assigned that depth after they had
already decided the event was a blast. I would not report the first score as a result, because
the model was reading the answer off the label-maker's own notes rather than off the ground.

None of the four clues would survive. The repeated address works because a quarry is blasted from
the same pit for decades, the working-hours and weekday clues work because blasting is a legal job
with a shift pattern, and the depth clue works because an analyst wrote it down once the label was
already chosen. A test nobody has announced is one event, at a site with no history in the
catalogue, at an hour picked to attract no attention, and with nobody to type a depth in for it —
so all four go at once, and the honest answer is that this week's classifier would have nothing
left to read.
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
        print(r.stderr[-6000:])
        sys.exit("the solution did not execute")

    (OUT / f"{SLUG}.ipynb").write_text(json.dumps(stu, indent=1) + "\n")
    print(f"wrote {SLUG}.ipynb ({len(CELLS)} cells) and the executed solution")
    print(f"cache: data/{BLAST_CACHE}, data/{QUAKE_CACHE}")


if __name__ == "__main__":
    main()
    weekkit.gate(10)
