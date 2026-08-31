#!/usr/bin/env python
"""Build week 10 — "Earthquake or explosion — how does the world verify a nuclear test ban?"

Emits both notebooks from one source so they cannot drift:

    docs/notebooks/10_earthquake_or_explosion_solution.ipynb   executed, every output saved
    docs/notebooks/10_earthquake_or_explosion.ipynb            the same file with the answers deleted

It also writes the week's two cached data files into data/. The quarry-blast query is small
enough to run live, so its cache is a fallback; the earthquake query matches 128,736 events and
the archive refuses anything over 20,000, so the earthquakes can only travel with the course as
a file, and the notebook reads that file directly.

Every number that appears in prose or in a model answer is computed HERE, from the same files
the notebook reads. Nothing is typed from memory or copied from the plan.

    python tools/build_week10.py
"""
import json
import pathlib
import subprocess
import sys

import numpy as np
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

# The slice, pinned so the cached files, the notebook and the prose below cannot drift apart.
# course.yml's pinned: block says "M1.5+ 2015-01-01 onward"; `onward` grows every day, so the
# end is pinned here at the date the files were built and the notebook asks for the same window.
START, END, MINMAG = "2015-01-01", "2026-08-31", 1.5
BOX = "&minlatitude=32&maxlatitude=42&minlongitude=-125&maxlongitude=-114"
QUERY = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&orderby=time-asc"
         f"&starttime={START}&endtime={END}&minmagnitude={MINMAG}{BOX}")

BLAST_CACHE = "week10_ca_quarry_blasts_2015_2026.csv"
QUAKE_CACHE = "week10_ca_earthquakes_2015_2026.csv.gz"
COLUMNS = ["time", "latitude", "longitude", "depth", "mag", "type"]
FEATURES = ["latitude", "longitude", "depth", "mag", "hour", "weekday"]
TWO = ["hour", "depth"]

# 2019 holds the Ridgecrest sequence and on its own exceeds the archive's 20,000-event cap, so
# that one year is fetched in five pieces. Everything else fits in a single year-long request.
SPANS = []
for _y in range(2015, 2027):
    _end = f"{_y + 1}-01-01" if _y < 2026 else END
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
    """Write data/week10_*.csv once. Both files are the exact bytes the notebook reads."""
    blast_path = ROOT / "data" / BLAST_CACHE
    if not blast_path.exists():
        pd.read_csv(QUERY + "&eventtype=quarry%20blast").to_csv(blast_path, index=False)
    quake_path = ROOT / "data" / QUAKE_CACHE
    if not quake_path.exists():
        parts = []
        for start, end in SPANS:
            url = QUERY.replace(f"starttime={START}", f"starttime={start}")
            url = url.replace(f"endtime={END}", f"endtime={end}") + "&eventtype=earthquake"
            parts.append(pd.read_csv(url))
        whole = pd.concat(parts, ignore_index=True).drop_duplicates(subset="id")
        whole[COLUMNS].to_csv(quake_path, index=False, compression="gzip")
    return blast_path, quake_path


blast_path, quake_path = build_caches()
blasts = pd.read_csv(blast_path)[COLUMNS]
quakes = pd.read_csv(quake_path)
events = pd.concat([quakes, blasts], ignore_index=True)
local = pd.to_datetime(events["time"]).dt.tz_convert("US/Pacific")
events["hour"] = local.dt.hour
events["weekday"] = local.dt.dayofweek
events["is_blast"] = events["type"] == "quarry blast"
events["year"] = events["time"].str[:4]

only_blasts = events[events["is_blast"]]
only_quakes = events[~events["is_blast"]]

M = {}
M["n_blasts"] = int(len(only_blasts))
M["n_quakes"] = int(len(only_quakes))
M["n_events"] = int(len(events))
M["per_blast"] = round(M["n_quakes"] / M["n_blasts"], 1)
M["frac_eq"] = float((~events["is_blast"]).mean())

M["blast_depth_median"] = float(only_blasts["depth"].median())
M["quake_depth_median"] = float(only_quakes["depth"].median())
M["blast_depth_min"] = float(only_blasts["depth"].min())
M["blast_depth_max"] = float(only_blasts["depth"].max())
M["blast_shallow"] = float((only_blasts["depth"] <= 0).mean())
M["quake_shallow"] = float((only_quakes["depth"] <= 0).mean())
M["quake_deeper_25"] = float((only_quakes["depth"] > 25).mean())
M["blast_distinct_depths"] = int(only_blasts["depth"].nunique())
M["quake_distinct_depths"] = int(only_quakes["depth"].nunique())
blast_depth_counts = only_blasts["depth"].value_counts()
M["blast_top20"] = float(blast_depth_counts.head(20).sum() / M["n_blasts"])
M["quake_top20"] = float(only_quakes["depth"].value_counts().head(20).sum() / M["n_quakes"])
M["top_depth"] = float(blast_depth_counts.index[0])
M["top_depth_n"] = int(blast_depth_counts.iloc[0])
same_depth = only_blasts[only_blasts["depth"] == M["top_depth"]]
M["top_depth_lat_lo"] = float(same_depth["latitude"].min())
M["top_depth_lat_hi"] = float(same_depth["latitude"].max())
M["top_depth_lon_lo"] = float(same_depth["longitude"].min())
M["top_depth_lon_hi"] = float(same_depth["longitude"].max())

M["blast_workhours"] = float(((only_blasts["hour"] >= 10) & (only_blasts["hour"] < 17)).mean())
M["quake_workhours"] = float(((only_quakes["hour"] >= 10) & (only_quakes["hour"] < 17)).mean())
M["blast_weekdays"] = float((only_blasts["weekday"] <= 4).mean())
M["quake_weekdays"] = float((only_quakes["weekday"] <= 4).mean())
M["blast_mag_median"] = float(only_blasts["mag"].median())
M["quake_mag_median"] = float(only_quakes["mag"].median())
M["blast_mag_max"] = float(only_blasts["mag"].max())
M["quake_mag_max"] = float(only_quakes["mag"].max())

X = events[FEATURES]
y = events["is_blast"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,
                                                    stratify=y)
M["n_train"] = int(len(X_train))
M["n_test"] = int(len(X_test))
M["blasts_train"] = int(y_train.sum())
M["blasts_test"] = int(y_test.sum())
M["train_blast_share"] = float(y_train.mean())
M["test_blast_share"] = float(y_test.mean())


def score(tag, guess):
    """Accuracy, precision, recall and F1 of one set of predictions on the test set."""
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
score("lr2", model_two.predict(X_test[TWO]))
M["lr2_hour_coef"] = float(model_two.coef_[0][0])
M["lr2_depth_coef"] = float(model_two.coef_[0][1])
M["lr2_intercept"] = float(model_two.intercept_[0])
M["lr2_flagged"] = int(model_two.predict(X_test[TWO]).sum())
M["lr2_line_at_noon"] = float(-(M["lr2_hour_coef"] * 12 + M["lr2_intercept"]) / M["lr2_depth_coef"])

model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
score("lr", model.predict(X_test))
model_nb = GaussianNB().fit(X_train, y_train)
score("nb", model_nb.predict(X_test))

NO_DEPTH = [f for f in FEATURES if f != "depth"]
score("lrnd", LogisticRegression(max_iter=1000).fit(X_train[NO_DEPTH], y_train)
      .predict(X_test[NO_DEPTH]))
score("nbnd", GaussianNB().fit(X_train[NO_DEPTH], y_train).predict(X_test[NO_DEPTH]))
score("wide", hand_rule(X_test, 7, 19))
score("narrow", hand_rule(X_test, 11, 15))

per_year = {}
for yr in sorted(events["year"].unique()):
    rows = events[events["year"] == yr]
    per_year[yr] = float(f1_score(rows["is_blast"], hand_rule(rows), zero_division=0))
M["year_worst"] = min(per_year, key=per_year.get)
M["year_best"] = max(per_year, key=per_year.get)
M["year_worst_f1"] = per_year[M["year_worst"]]
M["year_best_f1"] = per_year[M["year_best"]]
M["n_2019"] = int((events["year"] == "2019").sum())
M["n_2018"] = int((events["year"] == "2018").sum())

for k in sorted(M):
    print(f"{k:24s} {M[k]}")


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
Seismometers do not care what shook the ground. They record an earthquake, a landslide, a
mine collapse and a bomb the same way, and somebody has to look at each recording and decide
which it was.

California will not hand you a nuclear test. It will hand you the same decision, thousands of
times a year, at a smaller scale: quarries blast rock, the seismic network records the blasts
alongside real earthquakes, and a USGS analyst labels every event either `earthquake` or
`quarry blast`. That gives you tens of thousands of decisions somebody has already made — which
is exactly what you need to find out whether a machine could have made them instead.

Today you build that classifier, and then you find out what it actually learned.
"""

md(weekkit.OPENING.format(question=WEEK["question"], datahub=datahub, hook=HOOK.strip()))

md("""
## What you'll be able to do

**The science.** Say what an event catalogue does and does not tell you about the source of a
seismic signal, name the things that make a quarry blast recognisable, and say which of them
would survive if the event you were hunting were a secret nuclear test instead.

**The skills.** Split labelled data into a training half and a held-out half with `stratify`,
fit `LogisticRegression` and `GaussianNB` to it, and score a classifier with a
`confusion_matrix` and with `precision_score`, `recall_score` and `f1_score` rather than
accuracy.

**Eight places where you write something: five in class, three at home.** Each one is headed
*Your turn*, with an empty cell under it.
""")

setup = weekkit.setup_cell(
    imports=("import numpy as np\n"
             "from sklearn.model_selection import train_test_split\n"
             "from sklearn.linear_model import LogisticRegression\n"
             "from sklearn.naive_bayes import GaussianNB\n"
             "from sklearn.metrics import accuracy_score, precision_score, recall_score\n"
             "from sklearn.metrics import f1_score, confusion_matrix\n"),
    figsize="(7, 4)",
    cache_base=CACHE_BASE,
    docstring="Ask the USGS catalogue for California's quarry blasts; fall back to the cached copy.",
    url_expr=('"https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&orderby=time-asc"\n'
              f'                           "&starttime={START}&endtime={END}'
              f'&minmagnitude={MINMAG}"\n'
              f'                           "{BOX}"\n'
              '                           "&eventtype=quarry%20blast"'),
    cache_expr=f'"{BLAST_CACHE}"',
    unpack=f'''
COLUMNS = ["time", "latitude", "longitude", "depth", "mag", "type"]

# The blast query runs live. The matching earthquake query cannot: the archive refuses any
# request that matches more than 20,000 events, and this one matches far more, so the
# earthquakes travel with the course as a file instead of coming down the wire.
blasts = load()[COLUMNS]
quakes = pd.read_csv(CACHE + "/{QUAKE_CACHE}")
coast = pd.read_csv(CACHE + "/coastlines.csv")

print("quarry blasts:", blasts.shape, " earthquakes:", quakes.shape)
'''.strip("\n"))
code(setup)

# --- section 1 -------------------------------------------------------------
md("""
## Two kinds of shaking in one catalogue

Both files came from the same query, over the same box of California and Nevada, over the same
years, above the same magnitude. The only difference is the `type` column: one file holds the
events an analyst labelled `earthquake`, the other the ones labelled `quarry blast`.

Stack them into one table. `pd.concat` takes a list of tables with the same columns and returns
one longer table; `ignore_index=True` renumbers the rows from 0 rather than restarting the count
at the join.
""")

code("""
events = pd.concat([quakes, blasts], ignore_index=True)

print(events.shape)
print(events.head())
""")

md("""
Remember what a file like this is: *A catalogue lists what somebody's instruments recorded, not
what happened. Where there are no seismometers there are no earthquakes in the file.* Every label
in the `type` column was put there by a person, and that will matter more than it sounds like it
should.

First, how many of each.
""")

ask("""
### ✏️ Your turn 1

`events["type"].value_counts()` counts how many rows carry each label. Print it, then print how
many earthquakes the catalogue holds **for every one quarry blast** — one count divided by the
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
assert n_blasts + n_quakes == len(events), "the two labels should account for every row"
print("✓ the two classes —", n_blasts, "quarry blasts and", n_quakes,
      "earthquakes, a ratio of", round(n_quakes / n_blasts, 1), "to 1")
""")

md(f"""
That ratio is the whole problem in one number. Quarry blasts are about {M['per_blast']:.0f} times
rarer than earthquakes in this catalogue, and a class that rare does something to the way you are
allowed to measure success.
""")

# --- section 2 -------------------------------------------------------------
md("""
## Scoring a classifier when one class is rare

The obvious score for a classifier is **accuracy**: out of every event, what fraction did you
label correctly? Before you compute it, commit to a number.

### Predict before you run

Here is a classifier that took no effort at all: it ignores its input and answers "earthquake"
every single time. It never once flags a blast. What accuracy does it get on this catalogue?
Change `my_guess` to a fraction between 0 and 1, then run the cell.
""")

code("""
my_guess = 0.50

always_earthquake = [False] * len(events)
print("you guessed:  ", my_guess)
print("this rule got:", round(accuracy_score(events["is_blast"], always_earthquake), 4))
""")

md(f"""
{M['frac_eq']:.4f}. A rule that is wrong about every single thing you care about is right
{M['frac_eq'] * 100:.2f}% of the time, because {M['frac_eq'] * 100:.2f}% of the catalogue is the
answer it always gives. That is what a rare class does to accuracy, and it is why no one who
works on rare events reports accuracy on its own.

The two numbers they report instead split the question in half. *Of the ones you flagged, how
many were right? Of the real ones, how many did you catch?* The first is **precision**, the
second is **recall**, and the always-earthquake rule fails both in a way accuracy never showed.

They pull against each other: flag everything and your recall is perfect while your precision
collapses; flag almost nothing and the reverse. **F1** is the single number that refuses to let
you cheat either way — it is the harmonic mean of precision and recall, which is a way of saying
that it stays near the smaller of the two, so a score is only good when both are.
""")

ask("""
### ✏️ Your turn 2

Score the always-earthquake rule properly. Using `events["is_blast"]` as the truth and
`always_earthquake` as the prediction, print its precision, its recall and its F1 with
`precision_score`, `recall_score` and `f1_score`.

Each of those three takes `zero_division=0` as a third argument, which tells scikit-learn what to
do when a rule flags nothing at all: there is no *"of the ones you flagged"* to divide by, so
count it as zero rather than crashing.

**Use these names**, because the self-check looks for them: `always_precision`, `always_recall`
and `always_f1`.
""")

answer("""
always_precision = precision_score(events["is_blast"], always_earthquake, zero_division=0)
always_recall = recall_score(events["is_blast"], always_earthquake, zero_division=0)
always_f1 = f1_score(events["is_blast"], always_earthquake, zero_division=0)

print("precision:", always_precision)
print("recall:   ", always_recall)
print("F1:       ", always_f1)
""", """
assert always_recall == 0, "a rule that never says blast cannot catch any blast"
print("✓ accuracy against F1 — the same rule scores",
      round(accuracy_score(events["is_blast"], always_earthquake), 4),
      "on accuracy and", always_f1, "on F1")
""")

# --- section 3 -------------------------------------------------------------
md("""
## What the catalogue knows about a quarry blast

Six columns arrived with each event: when it happened, where, how deep, how big, and the label.
Anything a classifier learns has to come out of the first four. So look at them.

Start with where. One earthquake in forty is drawn, or the map is a solid block of ink.
""")

code(f"""
plt.figure(figsize=(5, 5))       # California is nearly as tall as it is wide
few = quakes.iloc[::40]
plt.scatter(few["longitude"], few["latitude"], s=2, color="0.7", label="earthquakes")
plt.scatter(blasts["longitude"], blasts["latitude"], s=2, color="firebrick", label="blasts")
plt.plot(coast["lon"], coast["lat"], color="0.3", lw=0.6)
plt.xlim(-125, -114)
plt.ylim(32, 42)
plt.gca().set_aspect("equal")
plt.xlabel("degrees east")
plt.ylabel("degrees north")
plt.title("{{}} blasts, {{}} of {M['n_quakes']:,} earthquakes".format(len(blasts), len(few)))
plt.legend()
plt.show()
""")

md("""
The earthquakes trace faults — long, thin, continuous lines of them. The blasts do not: they sit
in tight knots, because a quarry is a fixed hole in the ground that gets blasted again and again
for decades. Already that is a usable clue, and also a warning: a model that learns *where* the
quarries are has learned a list of addresses, not a piece of physics.

Now depth. Every event in the file has one, and the two labels do not use the same range, so
`density=True` scales each histogram to the same total area — otherwise 2,803 blasts vanish
beside 128,736 earthquakes.
""")

code(f"""
depth_bins = np.arange(-4, 25.5, 0.5)
plt.hist(quakes["depth"], bins=depth_bins, density=True, label="earthquakes")
plt.hist(blasts["depth"], bins=depth_bins, density=True, alpha=0.6, label="blasts")
plt.axvline(0, color="black", lw=1)
plt.xlabel("depth (km; negative means above sea level)")
plt.ylabel("share of events per km of depth")
plt.title("depth of {M['n_quakes']:,} earthquakes and {M['n_blasts']:,} blasts")
plt.legend()
plt.show()
""")

md(f"""
The blasts pile up to the **left** of zero. Not at zero — below it, at negative depths, which
sounds like nonsense until you remember that depth in this catalogue is measured from sea level
and a quarry is a hole in a hillside several hundred metres up. The median blast in this file
sits at {M['blast_depth_median']} km, that is {abs(M['blast_depth_median']) * 1000:.0f} metres
*above* sea level, against a median earthquake at {M['quake_depth_median']} km below it.
(The axis stops at 25 km; {M['quake_deeper_25'] * 100:.1f}% of the earthquakes are deeper and
none of the blasts are.)

That looks like a gift. Look harder at the actual values.
""")

code("""
print(blasts["depth"].value_counts().head(5))
print("distinct depth values, blasts:     ", blasts["depth"].nunique())
print("distinct depth values, earthquakes:", quakes["depth"].nunique())
""")

md(f"""
{M['top_depth_n']} separate blasts share the depth {M['top_depth']} km — to the metre. A depth
that repeats to the metre was not measured; it was **set**. When an analyst recognises an event
as a quarry blast, they fix its depth at the quarry's own surface rather than letting the
inversion float it, so the depth column is not a property of the ground shaking. It is a note
about what the analyst had already decided. The
{M['top_depth_n']} events at {M['top_depth']} km all sit inside
{M['top_depth_lat_hi'] - M['top_depth_lat_lo']:.2f} degrees of latitude of each other: one
quarry, one number.

That is what **leakage** looks like, and it has a plain-language test.

> You got 99 percent. Be suspicious. Did one of your columns already know the answer?

Keep depth for now — you will take it away in the homework and see what is left. Last, when.
`pd.to_datetime` turns the time column into real timestamps; the catalogue records them in UTC,
so `.dt.tz_convert("US/Pacific")` moves them to the clock the quarry crew actually works to, and
`.dt.hour` and `.dt.dayofweek` read the hour and the day off each one (Monday is 0, Sunday is 6).
""")

code(f"""
local = pd.to_datetime(events["time"]).dt.tz_convert("US/Pacific")
events["hour"] = local.dt.hour
events["weekday"] = local.dt.dayofweek
events["is_blast"] = events["type"] == "quarry blast"

hour_bins = np.arange(0, 25, 1)
plt.hist(events[~events["is_blast"]]["hour"], bins=hour_bins, density=True, label="earthquakes")
plt.hist(events[events["is_blast"]]["hour"], bins=hour_bins, density=True, alpha=0.6,
         label="blasts")
plt.xlabel("hour of the day, local California time")
plt.ylabel("share of events per hour")
plt.title("time of day, {M['n_quakes']:,} earthquakes and {M['n_blasts']:,} blasts")
plt.legend()
plt.show()
""")

md(f"""
Earthquakes are flat across the day, which is what you would expect of a process that has never
heard of a clock. The blasts are a working day: {M['blast_workhours'] * 100:.1f}% of them fall
between 10 in the morning and 5 in the afternoon, against {M['quake_workhours'] * 100:.1f}% of
the earthquakes, and {M['blast_weekdays'] * 100:.1f}% land Monday to Friday against
{M['quake_weekdays'] * 100:.1f}%.

So the catalogue gives you three kinds of clue: a place that repeats, a depth somebody typed in,
and a human timetable. Time to use them.
""")

# --- section 4 -------------------------------------------------------------
md("""
## A rule you can write by hand

Before any model, the rule you already have in your head:

> Write the dumbest rule you can, first. Any model that cannot beat it is decoration.

That is the **baseline**, and its job is not to be good. Its job is to be the number every clever
thing you build afterwards has to clear, because a model that scores below it has taught you
nothing except that you can call `.fit`.

To compare anything fairly you need data the rule and the models were both kept away from, so
split the catalogue into a training set and a held-out test set. One argument here is new:
`stratify=y` forces the rare class to appear in the same proportion in both halves. Without it a
random 30% could easily take a lopsided share of the 2,803 blasts, and your test score would be
measuring the split instead of the classifier.
""")

code(weekkit.CHECKPOINT.format(body='''events = pd.concat([quakes, blasts], ignore_index=True)
local = pd.to_datetime(events["time"]).dt.tz_convert("US/Pacific")
events["hour"] = local.dt.hour
events["weekday"] = local.dt.dayofweek
events["is_blast"] = events["type"] == "quarry blast"'''))

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
The two clues that do not need a computer are the two you just plotted: a blast is at or above
sea level, and a blast happens in the middle of a working day. Write that down as one line of
Python.

The `&` is the array version of `and` — it asks the question of every row at once, the way
`earth < 0` asked one question of every cell of the elevation grid. Each condition needs its own
brackets, because `&` binds tighter than `<=` does.
""")

ask("""
### ✏️ Your turn 3

Build the baseline and score it on the held-out set.

`hand_rule` should be `True` where **both** of these hold for a row of `X_test`: the `depth` is
at or below 0, and the `hour` is at least 10 and less than 17. Then print its precision, recall
and F1 against `y_test`, the same three calls you used in your turn 2.

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

md(f"""
Two conditions, no fitting, F1 {M['hand_f1']:.4f}. Write that number down. Everything from here
has to beat it.
""")

# --- section 5 -------------------------------------------------------------
md(f"""
## Letting the computer draw the line

**Logistic regression** is the straight-line fit from earlier in the course, bent to answer a
yes-or-no question.

> The same straight line — but now it outputs a probability between 0 and 1.

Where the line lands is the **decision boundary**: the model says "blast" on one side of it and
"earthquake" on the other, and the boundary is the only thing a straight-line classifier can ever
draw. `.fit(X_train, y_train)` finds it and `.predict` applies it, exactly the two calls you used
for a regression line.

### Predict before you run

Your two hand-written conditions used `depth` and `hour` and scored F1 {M['hand_f1']:.4f}. Give
logistic regression the very same two columns and {M['n_train']:,} labelled events to fit on.
What F1 does it get? Change `my_guess` and run.
""")

code("""
my_guess = 0.85

model_two = LogisticRegression(max_iter=1000).fit(X_train[["hour", "depth"]], y_train)
guess_two = model_two.predict(X_test[["hour", "depth"]])

print("you guessed:", my_guess)
print("it scored:  ", round(f1_score(y_test, guess_two), 4))
""")

md(f"""
{M['lr2_f1']:.4f}. Not a little worse than your two `if` conditions — five times worse. It flagged
{M['lr2_flagged']} events as blasts out of {M['blasts_test']} real ones. Draw the line it found
and the reason is visible.

`model_two.coef_[0]` holds one number per column and `model_two.intercept_[0]` the offset, and
the boundary is where they add to zero: `hour_coef * hour + depth_coef * depth + intercept == 0`.
Rearranged for depth, that is a line you can plot.
""")

code("""
plt.scatter(few["hour"], few["depth"], s=3, color="0.7", label="earthquakes")
plt.scatter(blasts["hour"], blasts["depth"], s=3, color="firebrick", label="blasts")

hours = np.arange(0, 24)
hour_coef, depth_coef = model_two.coef_[0]
boundary = -(hour_coef * hours + model_two.intercept_[0]) / depth_coef
plt.plot(hours, boundary, color="black", lw=2, label="decision boundary")

plt.ylim(-4, 25)
plt.xlabel("hour of the day, local California time")
plt.ylabel("depth (km; negative means above sea level)")
plt.title("{} blasts and {} earthquakes, with the fitted boundary".format(len(blasts), len(few)))
plt.legend()
plt.show()
""")

md(f"""
The line is almost flat. It sits at about {M['lr2_line_at_noon']:.2f} km and barely tilts, which
means the model threw the clock away and kept only "very shallow" — and then set its depth cut
well *below* zero, because with one blast for every {M['per_blast']:.0f} earthquakes it has to be
extremely sure before it dares say blast.

It threw the clock away because it could not use it. Blasts happen in the *middle* of the day, and
"the middle" is not a side of a line. A straight boundary can say *later than 10* or *earlier
than 5*; saying both at once needs a corner, and a straight line has none. Your two `if`
statements had a corner. That is the whole difference, and it is the thing week after next comes
back for.

Meanwhile the model has been fighting with one hand tied: it only ever saw two of the six columns
you prepared. Give it all of them.
""")

code("""
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
guess = model.predict(X_test)

print("accuracy: ", round(accuracy_score(y_test, guess), 4))
print("precision:", round(precision_score(y_test, guess), 4))
print("recall:   ", round(recall_score(y_test, guess), 4))
print("F1:       ", round(f1_score(y_test, guess), 4))
""")

md(f"""
Six columns, {M['n_train']:,} training events, F1 {M['lr_f1']:.4f}. Compare it with the
{M['hand_f1']:.4f} your two conditions got and hold your reaction until you have seen where its
mistakes are, because a single number hides which of the two mistakes a classifier is making. The
**confusion matrix** is the table that separates them: one row per true label, one column per
predicted label, so the off-diagonal cells are the misses and the false alarms counted apart.
""")

ask("""
### ✏️ Your turn 4

Print the confusion matrix of the six-column model with
`confusion_matrix(y_test, guess)`. It comes back as a 2 by 2 grid of counts: the top row is the
events that really were earthquakes, the bottom row the ones that really were blasts, and within
each row the first column is "the model said earthquake" and the second "the model said blast".

Then print which of the two mistakes is the larger — the blasts it missed, or the earthquakes it
falsely flagged.

**Use these names**, because the self-check looks for them: `matrix`.
""")

answer("""
matrix = confusion_matrix(y_test, guess)
print(matrix)

print("blasts missed:            ", matrix[1][0])
print("earthquakes falsely flagged:", matrix[0][1])
""", """
assert matrix.sum() == len(y_test), "the matrix should count every held-out event exactly once"
print("✓ the confusion matrix —", matrix[1][1], "blasts caught,", matrix[1][0],
      "missed,", matrix[0][1], "earthquakes falsely flagged")
""")

# --- section 6 -------------------------------------------------------------
md("""
## Scoring the clues instead of drawing a line

There is a second way to use the same six columns, and it does not draw anything.

> What does a quarry blast usually look like? Shallow, weekday, mid-afternoon. Score each clue
> and multiply. Pretending the clues are independent is obviously wrong, and it works anyway.

That is **Naive Bayes**. It learns, one column at a time, what values blasts tend to have and
what values earthquakes tend to have, then for a new event it multiplies the clues together and
takes whichever label comes out ahead. The "naive" part is the multiplying: it assumes the clues
are independent of each other, and here they plainly are not — a shallow event in this catalogue
is *more* likely to be at 2pm, not equally likely. `GaussianNB` is the version that treats each
column as a bell curve, and it is used through the identical `.fit` and `.predict` pair.
""")

ask("""
### ✏️ Your turn 5

Fit `GaussianNB()` on `X_train` and `y_train`, predict on `X_test`, and print its precision,
recall and F1. Then print, on one line each, the four F1 scores this notebook has produced so
far, so they can be read together: the always-earthquake rule, your hand rule, the six-column
logistic regression (`guess`), and this one.

**Use these names**, because the self-check looks for them: `model_nb` and `guess_nb`.
""")

answer("""
model_nb = GaussianNB().fit(X_train, y_train)
guess_nb = model_nb.predict(X_test)

print("precision:", round(precision_score(y_test, guess_nb), 4))
print("recall:   ", round(recall_score(y_test, guess_nb), 4))
print("F1:       ", round(f1_score(y_test, guess_nb), 4))

print("always earthquake:  ", round(f1_score(y_test, [False] * len(y_test),
                                             zero_division=0), 4))
print("hand rule:          ", round(f1_score(y_test, hand_rule), 4))
print("logistic regression:", round(f1_score(y_test, guess), 4))
print("naive Bayes:        ", round(f1_score(y_test, guess_nb), 4))
""", """
assert len(guess_nb) == len(y_test), "predict on X_test, the half the model was not fitted to"
print("✓ four classifiers — the best F1 is",
      max(round(f1_score(y_test, hand_rule), 4), round(f1_score(y_test, guess), 4),
          round(f1_score(y_test, guess_nb), 4)))
""")

# --- section 7 -------------------------------------------------------------
md(f"""
## The same rule, year by year

One split of one catalogue is one number, and a number that only holds for the window you happened
to choose is not a result. The hand rule has nothing fitted to anything, so it can be scored on
every event of every year without any of it counting as cheating. Do that.
""")

code("""
events["year"] = events["time"].str[:4]

for year in sorted(events["year"].unique()):
    rows = events[events["year"] == year]
    guess_year = (rows["depth"] <= 0) & (rows["hour"] >= 10) & (rows["hour"] < 17)
    print(year, len(rows), "events,", rows["is_blast"].sum(), "blasts,  F1",
          round(f1_score(rows["is_blast"], guess_year), 3))
""")

md(f"""
The baseline holds: every year between {M['year_worst_f1']:.3f} and {M['year_best_f1']:.3f}, with
no drift and no year where it falls apart. The worst two are {M['year_worst']} and its neighbour,
and the printout says why — those are the years the catalogue swells to {M['n_2019']:,} events
against {M['n_2018']:,} the year before. The Ridgecrest earthquake sequence of 2019 dumped tens of
thousands of small, shallow aftershocks into the earthquake class, and shallow daytime
aftershocks are precisely what the hand rule mistakes for blasts. More earthquakes to be wrong
about, the same blasts to catch: precision falls, and F1 with it.
""")

# --- the question, answered ------------------------------------------------
md(f"""
## The question, answered

**Not from a catalogue.** On these {M['n_test']:,} held-out events, two hand-written conditions
scored F1 {M['hand_f1']:.4f}, logistic regression on six columns scored {M['lr_f1']:.4f} and
naive Bayes {M['nb_f1']:.4f} — the models did not beat the rule, and neither did well enough to
be trusted with a decision that matters.

The reason is what the clues are made of. Real discrimination of an explosion from an earthquake
is done on the waveform, not on a catalogue row: an explosion is a sudden push outward from a
point at or near the surface, so it radiates compressional P energy in every direction and makes
comparatively feeble shear and surface waves, while an earthquake is rock sliding past rock on a
fault, which is efficient at making exactly those waves. That contrast is what the ratio of P to
S amplitude and the comparison of body-wave with surface-wave magnitude are built to measure, and
none of it is in the six columns you had. Depth would be a genuine discriminant — nobody buries a
device kilometres down — except that in this file the depth of a blast is not measured but
assigned.

What you classified on instead was a quarry's routine: a fixed address, a working day, a weekday,
an analyst's convention. Every one of those is a property of *legal, advertised, repeated*
blasting. A clandestine nuclear test is a single event, at a site nobody has recorded before, at
an hour chosen by people who would rather not be identified — so all four clues go dark at once,
which is why treaty monitoring reads waveforms rather than catalogues. And it is why the honest
report of this week's work is that a small F1 on the wrong features is a more useful result than
a large one.
""")

# --- summary and homework --------------------------------------------------
md(weekkit.week_cheatsheet(10))

md("""
## Homework

Three parts on the table you already have. If you have restarted since class, run the setup cell
at the top and then the checkpoint below, which rebuilds everything class left in memory.
""")

code(weekkit.CHECKPOINT.format(body=f'''events = pd.concat([quakes, blasts], ignore_index=True)
local = pd.to_datetime(events["time"]).dt.tz_convert("US/Pacific")
events["hour"] = local.dt.hour
events["weekday"] = local.dt.dayofweek
events["is_blast"] = events["type"] == "quarry blast"
features = {FEATURES}
X = events[features]
y = events["is_blast"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,
                                                    stratify=y)'''))

ask(f"""
### ✏️ Your turn 6

Class left one column under suspicion. Take it away and see what the models were really standing
on.

Build `features_no_depth` — the same list of six as `features`, minus `"depth"` — then fit a fresh
`LogisticRegression(max_iter=1000)` and a fresh `GaussianNB()` on `X_train[features_no_depth]` and
`y_train`, predict on `X_test[features_no_depth]`, and print the F1 of each. Print, beside them,
the F1 of the always-earthquake rule from your turn 2 so you have something to read them against.
For reference, with depth the two models scored {M['lr_f1']:.4f} and {M['nb_f1']:.4f}.

**Use these names**, because the self-check looks for them: `features_no_depth`, `f1_lr_no_depth`
and `f1_nb_no_depth`.
""")

answer(f"""
features_no_depth = {NO_DEPTH}

model_lr_nd = LogisticRegression(max_iter=1000).fit(X_train[features_no_depth], y_train)
model_nb_nd = GaussianNB().fit(X_train[features_no_depth], y_train)

f1_lr_no_depth = f1_score(y_test, model_lr_nd.predict(X_test[features_no_depth]),
                          zero_division=0)
f1_nb_no_depth = f1_score(y_test, model_nb_nd.predict(X_test[features_no_depth]),
                          zero_division=0)

print("logistic regression, no depth:", round(f1_lr_no_depth, 4))
print("naive Bayes, no depth:        ", round(f1_nb_no_depth, 4))
print("always earthquake:            ",
      round(f1_score(y_test, [False] * len(y_test), zero_division=0), 4))
""", """
assert "depth" not in features_no_depth, "the point of this part is to leave depth out"
print("✓ without depth — logistic regression F1", round(f1_lr_no_depth, 4),
      "and naive Bayes F1", round(f1_nb_no_depth, 4))
""")

ask(f"""
### ✏️ Your turn 7

Class chose 10:00 to 17:00 for "working hours" without arguing about it. It is a choice, and it is
yours to make. Pick **one** of these two, and only one:

- **wide**, 7 to 19 — catch the early and late blasts too
- **narrow**, 11 to 15 — only the solid middle of the day

Set `my_low` and `my_high` to the window you picked, rebuild the hand rule on `X_test` with your
window in place of 10 and 17, and print its precision, its recall and its F1. Then print the same
three for the 10-to-17 window so you can see which of them your choice moved, and in which
direction. Class got precision {M['hand_prec']:.4f}, recall {M['hand_rec']:.4f} and F1
{M['hand_f1']:.4f}.

**Use these names**, because the self-check looks for them: `my_low`, `my_high` and `my_rule`.
""")

answer(f"""
my_low = 7
my_high = 19

my_rule = (X_test["depth"] <= 0) & (X_test["hour"] >= my_low) & (X_test["hour"] < my_high)
class_rule = (X_test["depth"] <= 0) & (X_test["hour"] >= 10) & (X_test["hour"] < 17)

print("mine   ", my_low, "to", my_high,
      " precision", round(precision_score(y_test, my_rule), 4),
      " recall", round(recall_score(y_test, my_rule), 4),
      " F1", round(f1_score(y_test, my_rule), 4))
print("class's 10 to 17",
      " precision", round(precision_score(y_test, class_rule), 4),
      " recall", round(recall_score(y_test, class_rule), 4),
      " F1", round(f1_score(y_test, class_rule), 4))
""", """
assert (my_low, my_high) != (10, 17), "pick one of the two windows offered, not the one class used"
print("✓ your window —", my_low, "to", my_high, "gives precision",
      round(precision_score(y_test, my_rule), 4), "and recall",
      round(recall_score(y_test, my_rule), 4))
""")

ask("""
### ✏️ Your turn 8

Two or three sentences, quoting your own printed numbers.

Your turn 6 gave you an F1 for logistic regression with depth and an F1 without it. Quote both,
say what the gap between them tells you about what the model had actually learned, and say
whether you would report the first of the two as a result. Then, in one more sentence: name one
of the clues this week used that would still be there if the event you were trying to catch were
a secret nuclear test rather than a quarry, or say that none would and why.
""")

answer_prose(f"""
With depth, my logistic regression scored F1 {M['lr_f1']:.4f}; with the depth column removed and
nothing else changed, it scored {M['lrnd_f1']:.4f}, which is the same F1 as the rule that answers
"earthquake" every time — and naive Bayes fell from {M['nb_f1']:.4f} to {M['nbnd_f1']:.4f}, which
is barely different. So essentially all of the models' apparent skill was coming from one column,
and that column is not a measurement: {M['top_depth_n']} blasts share the single depth
{M['top_depth']} km to the metre, which means an analyst assigned it after they had already
decided the event was a blast. The first score is not a result I would report, because the model
was reading the answer off the label-maker's own notes rather than off the ground.

None of the clues would survive. The repeated location works because a quarry is blasted for
decades from the same pit, the working-hours and weekday clues work because blasting is a legal
job with a shift pattern, and the depth clue works because somebody wrote it down; a clandestine
test is a single event at an unrecorded site at a time chosen to attract no attention, so all
four go at once.
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
    r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                        "--execute", "--inplace", "--ExecutePreprocessor.timeout=900",
                        str(sol_path)], capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        print(r.stderr[-6000:])
        sys.exit("the solution did not execute")

    (OUT / f"{SLUG}.ipynb").write_text(json.dumps(stu, indent=1) + "\n")
    print(f"wrote {SLUG}.ipynb ({len(CELLS)} cells) and the executed solution")
    print(f"cache: data/{BLAST_CACHE}, data/{QUAKE_CACHE}")


if __name__ == "__main__":
    main()
    weekkit.gate(10)
