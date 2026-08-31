#!/usr/bin/env python
"""Build week 9 — "Is CO2 rising faster than it used to?"

Emits both notebooks from one source so they cannot drift:

    docs/notebooks/09_rising_faster_solution.ipynb   executed, every output saved
    docs/notebooks/09_rising_faster.ipynb            the same file with the answers deleted

It also writes the week's cached fallback for its one live read, NOAA's Mauna Loa monthly CO2
file. The cached copy is the PARSED table (the 42 comment lines already stripped), so the
fallback read needs no arguments the live read does not have.

Every number that appears in prose or in a model answer is computed HERE, from the same file the
notebook reads, and formatted in. Nothing is typed from memory or copied from the plan.

NOAA appends a month to that file every month, and the student's notebook reads it LIVE — so a
prose number taken from the record's tail expires between the build and the class. Two things
stop that. The notebook averages only the years with all twelve months, so a new month cannot
move a single fitted number; and the counts and months that do change are printed at run time
rather than written into markdown. What remains literal is listed in the build's closing report.

    python tools/build_week09.py              # build from the cached copy
    python tools/build_week09.py --refresh    # download NOAA again first — do this before class
"""
import datetime
import json
import pathlib
import subprocess
import sys
import textwrap
import time

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/notebooks"
SLUG = "09_rising_faster"

course = yaml.safe_load((ROOT / "course.yml").read_text())
WEEK = next(s for s in course["schedule"] if s["n"] == 9)
PLATFORM = course["platform"]
CACHE_BASE = PLATFORM["cache_base"]

# The one live read this week runs. Pinned here so the cached CSV, the notebook and the prose
# below cannot drift apart.
CO2_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
CACHE_NAME = "week09_co2_mm_mlo.csv"
STALE_DAYS = 30                   # after this the cache is behind NOAA by a month or more

SPLIT_YEAR = 2000                 # class hides everything after this from itself
OTHER_SPLITS = [1990, 1995, 2005]  # the neighbouring windows, so one cut is not the whole story
DEGREES = list(range(1, 10))
TARGET_PPM = 500
SHUFFLE_SEED = 88


# ---------------------------------------------------------------------------
# 1. measure everything the notebook will say
# ---------------------------------------------------------------------------
def fetch_monthly(refresh=False):
    """Run the live read, cache the PARSED table, and return it.

    The cache used to be downloaded once and then never again, which is the quiet way this week
    goes wrong: NOAA adds a month to the file every month, the student's notebook reads it live,
    and a build months later would keep describing a record that no longer exists. So `--refresh`
    downloads it again, and a cache older than STALE_DAYS says so loudly rather than passing for
    current.
    """
    out = ROOT / "data" / CACHE_NAME
    if refresh or not out.exists():
        print(f"downloading {CO2_URL}")
        pd.read_csv(CO2_URL, comment="#").to_csv(out, index=False)
    else:
        age = (time.time() - out.stat().st_mtime) / 86400
        if age > STALE_DAYS:
            print(f"WARNING: data/{CACHE_NAME} was downloaded {age:.0f} days ago. NOAA has "
                  f"appended months since, and the student's notebook reads NOAA live. "
                  f"Rebuild with --refresh before the class.")
    return pd.read_csv(out)


def annual_means(monthly):
    """One number per year — but only the years that have all twelve months.

    The record begins in March and its final year is still being measured, so both end years
    would be averages of whichever seasons happened to be present. That is a real bias (the
    notebook measures it), and leaving those two years in is also what made every fitted number
    in the prose expire the moment NOAA published another month.
    """
    full = monthly[(monthly["year"] >= monthly["year"].min() + 1)
                   & (monthly["year"] <= monthly["year"].max() - 1)]
    return (full.groupby("year", as_index=False)["average"].mean()
            .rename(columns={"average": "co2"}))


def fit_curve(table, degree):
    """Fit a polynomial of this degree to a table of years and CO2; hand back its coefficients."""
    return np.polyfit(table["year"] - START_YEAR, table["co2"], degree)


def curve_value(coeffs, year):
    """What a fitted curve says CO2 is in that year."""
    return np.polyval(coeffs, year - START_YEAR)


def typical_miss(actual, predicted):
    """The usual size of the gap between what happened and what the curve said, in ppm."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def crossing_year(coeffs):
    """The first year from 2027 on at which a fitted curve reaches the target, or None."""
    for year in range(2027, 2301):
        if curve_value(coeffs, year) >= TARGET_PPM:
            return year
    return None


REFRESH = "--refresh" in sys.argv
monthly_full = fetch_monthly(REFRESH)
READ_DATE = datetime.date.fromtimestamp(
    (ROOT / "data" / CACHE_NAME).stat().st_mtime).isoformat()
monthly = monthly_full[["year", "month", "average"]]
annual = annual_means(monthly)
START_YEAR = int(annual["year"].min())   # the first complete year; every fit measures x from here
train = annual[annual["year"] <= SPLIT_YEAR]
later = annual[annual["year"] > SPLIT_YEAR]

M = {}
M["n_monthly"] = len(monthly)
M["n_annual"] = len(annual)
M["first_year"] = START_YEAR
M["last_year"] = int(annual["year"].max())
M["observed"] = round(float(annual["co2"].iloc[-1]), 2)
# The two part-years the record starts and ends on, and what averaging only the months a
# part-year happens to have does to a year's mean. These are the only numbers here that a new
# month of NOAA data moves, so none of them may be written into the prose. lc_bias is the sharp
# case: it is 1.15 ppm on seven months of a year and 0.36 on nine, and NOAA GML resets about half
# of 46 simultaneous requests, so on any given Monday part of the room reads the live file and
# part falls back to this cache — the two halves would disagree with a single sentence written
# here. The notebook recomputes and prints it at RUN time instead; the value below is measured
# only so the build report can state what the builder saw.
M["record_first_year"] = int(monthly["year"].min())
M["record_first_month"] = int(monthly["month"].iloc[0])
M["part_year"] = int(monthly["year"].max())
M["part_months"] = int(monthly["month"].iloc[-1])
recent = monthly[monthly["year"] == M["last_year"]]
M["season_low"] = round(float(recent["average"].min()), 2)
M["season_high"] = round(float(recent["average"].max()), 2)
M["season_range"] = round(M["season_high"] - M["season_low"], 2)
M["lc_bias"] = round(float(recent[recent["month"] <= M["part_months"]]["average"].mean())
                     - float(recent["average"].mean()), 2)

line = fit_curve(annual, 1)
M["line_slope"] = round(float(line[0]), 3)
M["line_at_start"] = round(float(curve_value(line, START_YEAR)), 1)
M["line_miss"] = round(typical_miss(annual["co2"], curve_value(line, annual["year"])), 2)

resid = annual["co2"] - curve_value(line, annual["year"])
RESID_DECADES = [START_YEAR, 1980, M["last_year"] - 9]   # first ten years, the 1980s, last ten
M["resid_decade"] = []
for start in RESID_DECADES:
    dec = resid[(annual["year"] >= start) & (annual["year"] < start + 10)]
    M["resid_decade"].append(round(float(dec.mean()), 1))
M["resid_min"] = round(float(resid.min()), 1)
M["resid_max"] = round(float(resid.max()), 1)

M["full_miss"] = {d: round(typical_miss(annual["co2"],
                                        curve_value(fit_curve(annual, d), annual["year"])), 3)
                  for d in DEGREES}

DECADES = [1960, 1970, 1980, 1990, 2000, 2010]
M["decade_rate"] = {}
for start in DECADES:
    dec = annual[(annual["year"] >= start) & (annual["year"] < start + 10)]
    M["decade_rate"][start] = round(float(fit_curve(dec, 1)[0]), 2)

M["n_train"] = len(train)
M["n_later"] = len(later)
# The notebook's own cell prints `round(predicted - observed, 1)` from the UNROUNDED prediction
# and the UNROUNDED observation. Taking the error from M["pred"] and M["observed"] instead rounds
# twice — 400.8 - 427.35 = -26.55 -> -26.6 — and the prose then contradicts the output printed
# directly above it. Round once, at the end, exactly as the notebook does.
PRED_RAW = {d: float(curve_value(fit_curve(train, d), M["last_year"])) for d in [1, 2, 3]}
OBSERVED_RAW = float(annual["co2"].iloc[-1])
M["pred"] = {d: round(PRED_RAW[d], 1) for d in [1, 2, 3]}
M["err"] = {d: round(PRED_RAW[d] - OBSERVED_RAW, 1) for d in [1, 2, 3]}
M["pred_spread"] = round(max(M["pred"].values()) - min(M["pred"].values()), 1)

M["train_miss"] = {}
M["test_miss"] = {}
for d in DEGREES:
    c = fit_curve(train, d)
    M["train_miss"][d] = round(typical_miss(train["co2"], curve_value(c, train["year"])), 3)
    M["test_miss"][d] = round(typical_miss(later["co2"], curve_value(c, later["year"])), 3)
M["best_train_degree"] = min(DEGREES, key=lambda d: M["train_miss"][d])
M["best_test_degree"] = min(DEGREES, key=lambda d: M["test_miss"][d])
# Degree 9 on 42 points with x in [0, 41] is a badly conditioned fit. Jittering the annual means
# by +/-0.02 ppm — below NOAA's own retrospective revisions — moves this miss across roughly
# 13,200-14,200, so the five figures the raw value prints are not all real. The conclusion
# ("orders of magnitude worse") does not depend on them, so the prose quotes the magnitude.
M["test_miss_9_about"] = f"{round(M['test_miss'][9], -3):,.0f}"

M["other_best"] = {}
for cut in OTHER_SPLITS:
    tr, te = annual[annual["year"] <= cut], annual[annual["year"] > cut]
    scores = {d: typical_miss(te["co2"], curve_value(fit_curve(tr, d), te["year"]))
              for d in DEGREES}
    M["other_best"][cut] = min(scores, key=scores.get)

shuffled = annual.sample(frac=1, random_state=SHUFFLE_SEED)
rand_test = shuffled.iloc[:M["n_later"]]
rand_train = shuffled.iloc[M["n_later"]:]
M["rand_miss"] = {}
for d in [1, 2, 3, 9]:
    c = fit_curve(rand_train, d)
    M["rand_miss"][d] = round(typical_miss(rand_test["co2"], curve_value(c, rand_test["year"])), 3)
M["leak_factor"] = round(M["test_miss"][9] / M["rand_miss"][9])

M["cross_full"] = {d: crossing_year(fit_curve(annual, d)) for d in [1, 2, 3]}
M["cross_train"] = {d: crossing_year(fit_curve(train, d)) for d in [1, 2]}
M["cross_spread"] = max(M["cross_full"][1], M["cross_train"][1]) - min(
    M["cross_full"][2], M["cross_train"][2])
# The four years the homework computes, so the opening hook can name the stake in the week's own
# numbers instead of a decade nothing here produces.
_crossings = [M["cross_full"][1], M["cross_full"][2], M["cross_train"][1], M["cross_train"][2]]
M["cross_early_decade"] = min(_crossings) // 10 * 10
M["cross_late_decade"] = max(_crossings) // 10 * 10


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
    """A code answer cell. The solution carries the model answer; the student gets the stub."""
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


# What class builds and every later section reads: the three functions and the split. A
# checkpoint has to rebuild the scalars and the functions too, not only the dataframes — the
# section it opens is what a student who restarted the kernel runs next, and a missing helper is
# a NameError from the one cell whose job is to get them going again. Written once here and
# used by both checkpoints so the two cannot drift.
CLASS_TOOLKIT = f"""
def fit_curve(table, degree):
    \"\"\"Fit a polynomial of this degree to a table of years and CO2; hand back its coefficients.\"\"\"
    return np.polyfit(table["year"] - START_YEAR, table["co2"], degree)


def curve_value(coeffs, year):
    \"\"\"What a fitted curve says CO2 is in that year — one year, or a whole column of them.\"\"\"
    return np.polyval(coeffs, year - START_YEAR)


def typical_miss(actual, predicted):
    \"\"\"The usual size of the gap between what happened and what the curve said, in ppm.\"\"\"
    return np.sqrt(np.mean((actual - predicted) ** 2))


train = annual[annual["year"] <= {SPLIT_YEAR}]
later = annual[annual["year"] > {SPLIT_YEAR}]
""".strip("\n")

datahub = (f"{PLATFORM['datahub']}/hub/user-redirect/git-pull"
           f"?repo={PLATFORM['repo'].replace(':', '%3A').replace('/', '%2F')}"
           f"&branch={PLATFORM['branch']}"
           f"&urlpath=lab%2Ftree%2FEPS88_PyEarth%2F{PLATFORM['notebook_dir']}%2F{SLUG}.ipynb")

HOOK = f"""
In {M['record_first_year']} Charles David Keeling put a carbon-dioxide analyser high on a
Hawaiian volcano, far from anybody's chimney, and started measuring. It has been running ever
since, and the monthly numbers it has produced — one more of them every month, including the
month you are reading this in — are among the most consequential in science.

It goes up. Everybody knows it goes up. The question worth asking is whether it goes up in a
*straight line* — because a straight line and a gently bending curve look nearly identical over
the years you have measured, and say completely different things about the years you have not.
The difference between them is the difference between reaching {TARGET_PPM} parts per million in
the {M['cross_early_decade']}s and reaching it in the {M['cross_late_decade']}s.

Today you decide how much bend this record can actually support. Not by arguing about it: by
hiding {M['n_later']} years of it from yourself, fitting curves to what is left, and checking
what they say about the years you hid.
"""

md(weekkit.OPENING.format(question=WEEK["question"], datahub=datahub, hook=HOOK.strip()))

md(f"""
## What you'll be able to do

**The science.** Say how fast CO2 is rising, whether that rate is itself rising, and what the
record does and does not let you say about when it reaches {TARGET_PPM} ppm.

**The skills.** Fit a curve instead of a line with `np.polyfit` and read it back with
`np.polyval`; measure how badly a model misses in the units of the data; split a record into
years you fit on and years you keep back; and read the two error curves that tell you when a
model has stopped learning the pattern and started memorising the data.

**The four questions this week works through:**

1. Is a straight line enough?
2. How much bend does the record want?
3. How do you choose a degree when a flexible curve always fits better?
4. Does it matter which years you hide?

**Eight places where you write something: five in class, three at home.** Each one is headed
*Your turn*, with an empty cell under it.
""")

setup = weekkit.setup_cell(
    imports="import numpy as np\n",
    figsize="(7, 4)",
    cache_base=CACHE_BASE,
    docstring=(f"Read NOAA's Mauna Loa monthly CO2 file live; fall back to the copy stored with "
               f"the course, downloaded {READ_DATE}."),
    url_expr=f'"{CO2_URL}", comment="#"',
    cache_expr=f'"{CACHE_NAME}"',
    unpack='''
# NOAA Global Monitoring Laboratory, Mauna Loa monthly mean CO2 — the Keeling curve.
# The comment="#" above skips the file's own header notes. NOAA adds a month to this file
# every month, so the record below is longer every time you run it.
monthly = load()[["year", "month", "average"]]

# The record starts partway through its first year and its last year is still being measured,
# so averaging those two would average whichever seasons happen to be there. Keep the rest.
first_full = monthly["year"].min() + 1
last_full = monthly["year"].max() - 1
annual = (monthly[(monthly["year"] >= first_full) & (monthly["year"] <= last_full)]
          .groupby("year", as_index=False)["average"].mean()
          .rename(columns={"average": "co2"}))

START_YEAR = first_full   # every fit below counts years from here, not from year zero

print("monthly readings:", len(monthly))
print("complete years:", len(annual), "—", first_full, "to", last_full)
print(annual.head())
'''.strip("\n"))
code(setup)

# --- section 1 -------------------------------------------------------------
md("""
## Is a straight line enough?

Three lines of pandas turned every monthly reading into one number per complete year. Draw both
and you can see why we bothered — and why the setup cell threw two years away.
""")

code("""
plt.plot(monthly["year"] + monthly["month"] / 12, monthly["average"],
         color="0.7", lw=0.8, label="every month")
plt.plot(annual["year"], annual["co2"], marker="o", ms=3, label="annual average")
plt.xlabel("year")
plt.ylabel("CO2 (parts per million)")
plt.title(f"Mauna Loa CO2, {len(monthly)} monthly readings, {len(annual)} complete years")
plt.legend()
plt.show()
""")

md("""
The grey line has teeth. That is the northern hemisphere breathing: most of the world's land
plants live north of the equator, they pull CO2 out of the air through the growing season and
give it back as leaves and soil decay through the winter, and Mauna Loa sits far enough north to
feel it. Measure one whole year of it, then average only as many months as the record's
unfinished last year has, and compare the two.
""")

code("""
recent = monthly[monthly["year"] == last_full]      # the last whole year
unfinished = monthly[monthly["year"] > last_full]   # the year still being measured
months_so_far = unfinished["month"].max()

whole_year = recent["average"].mean()
part_year = recent[recent["month"] <= months_so_far]["average"].mean()

print(last_full, "ran from", round(recent["average"].min(), 2), "to",
      round(recent["average"].max(), 2), "ppm — a swing of",
      round(recent["average"].max() - recent["average"].min(), 2), "ppm inside one year")
print("its average over all twelve months:", round(whole_year, 2), "ppm")
print("its average over the first", months_so_far, "months only:", round(part_year, 2), "ppm")
print("so keeping only", months_so_far, "months shifts that year's mean by",
      round(part_year - whole_year, 2), "ppm")
""")

md(f"""
{M['season_range']} ppm of swing inside a single year — and the last line printed above is what
averaging only the months a part-year happens to have does to that year's mean, because the
months you keep are not a fair sample of the seasons. That shift is not written here because it
is not a fixed number: it depends on how many months NOAA has published on the day you run this,
and it shrinks as the year fills up. That is why the setup cell kept only whole years: the record
begins partway through {M['record_first_year']} and its last year is still being measured, so
both would have carried that bias into every curve we fit. It also means nothing below changes
when NOAA publishes next month's number.

The teeth otherwise are not what this week is about. Averaging each year removes them and leaves
the rise, which is what we want to model. So from here on we work with `annual`: one number per
year, from {M['first_year']} to {M['last_year']}.
""")

md("""
You fitted a straight line in the regression week; here it is again, with a new pair of functions.
`np.polyfit(x, y, degree)` finds the best polynomial of that degree — at degree 1 it *is* the
least-squares line — and hands back its coefficients. `np.polyval(coeffs, x)` reads the fitted
curve back at any x you like, including x values that were never in the data. Wrapping both in
functions of our own means the rest of the notebook is one line per fit.
""")

code("""
def fit_curve(table, degree):
    \"\"\"Fit a polynomial of this degree to a table of years and CO2; hand back its coefficients.\"\"\"
    return np.polyfit(table["year"] - START_YEAR, table["co2"], degree)


def curve_value(coeffs, year):
    \"\"\"What a fitted curve says CO2 is in that year — one year, or a whole column of them.\"\"\"
    return np.polyval(coeffs, year - START_YEAR)
""")

ask("""
### ✏️ Your turn 1

Fit a straight line to the whole of `annual` and read two numbers off it: how fast it says CO2
rises, in ppm per year, and what it says CO2 was at the start of the record.

`fit_curve(annual, 1)` gives you the coefficients. For a straight line the first coefficient is
the slope, so `line[0]` is the ppm per year. For the second number, ask `curve_value` what the
line says at `START_YEAR`.

**Use these names**, because the self-check looks for them: `line` and `slope`.
""")

answer(f"""
line = fit_curve(annual, 1)
slope = line[0]

print("the line rises", round(slope, 3), "ppm per year")
print("and puts", round(curve_value(line, START_YEAR), 1), "ppm at the start of the record")
""", """
assert 0.5 < slope < 5, \\
    "CO2 rises by a couple of ppm a year, so a slope in the hundreds is the wrong coefficient: \\
line[0] is the ppm per year and line[1] is where the line starts"
print("✓ the straight line —", round(slope, 3), "ppm per year, starting from",
      round(curve_value(line, START_YEAR), 1), "ppm")
""")

md(f"""
So a single straight line says CO2 has climbed {M['line_slope']} ppm every year since
{M['first_year']}, starting from {M['line_at_start']} ppm. Both numbers look reasonable. The
question is whether the line is *right*, and for that we need a number that says how badly a
model misses.

The one we will use all week is the **typical miss**: square every gap between what happened and
what the curve said, average the squares, take the square root. Squaring stops a miss of +5 and a
miss of −5 cancelling, and the square root at the end puts the answer back in ppm, so it means
something you can read.
""")

code("""
def typical_miss(actual, predicted):
    \"\"\"The usual size of the gap between what happened and what the curve said, in ppm.\"\"\"
    return np.sqrt(np.mean((actual - predicted) ** 2))


print("the straight line misses by", round(typical_miss(annual["co2"],
                                                        curve_value(line, annual["year"])), 2),
      "ppm on average")
""")

md(f"""
{M['line_miss']} ppm, on a record that spans more than a hundred. By the standards of the
regression week that is a good fit. But an average miss hides *where* the misses are, and that is the whole
question here. Subtract the line from the data and plot what is left over.
""")

code("""
residual = annual["co2"] - curve_value(line, annual["year"])

for start in [first_full, 1980, last_full - 9]:
    decade = residual[(annual["year"] >= start) & (annual["year"] < start + 10)]
    print(start, "to", start + 9, ": the line misses by", round(decade.mean(), 1), "ppm on average")

print("furthest above the line:", round(residual.max(), 1), "ppm")
print("furthest below the line:", round(residual.min(), 1), "ppm")
""")

code("""
plt.plot(annual["year"], residual, marker="o", ms=3)
plt.axhline(0, color="0.5", lw=0.8)      # the line itself, for the eye to measure against
plt.xlabel("year")
plt.ylabel("data minus straight line (ppm)")
plt.title(f"What the straight line leaves behind, {len(annual)} years")
plt.show()
""")

md(f"""
Those are not random misses. They are a smile: the line runs below the data at both ends and
above it in the middle: {M['resid_decade'][0]:+.1f} ppm on average over the record's first
ten years, {M['resid_decade'][1]:+.1f} ppm over the 1980s, {M['resid_decade'][2]:+.1f} ppm
over the last ten. The two extremes are {M['resid_max']:+.1f} and {M['resid_min']:+.1f} ppm.
A least-squares line always leaves misses that average to zero overall, so seeing them arranged
in an arc rather than scattered means the line has the wrong *shape*, not just some noise around
it.

A straight line has one rate of rise and cannot change it. This record changes it.
""")

# --- section 2 -------------------------------------------------------------
md("""
## How much bend does the record want?

The **degree** of a polynomial is how many bends it is allowed. Degree 1 is a straight line, no
bends. Degree 2 is a parabola: one bend, one steady change of slope. Degree 3 can bend twice,
degree 9 can bend eight times. `fit_curve` already takes the degree as its second argument, so
trying more flexible curves costs nothing but the number.
""")

code("""
for degree in [1, 2, 3]:
    coeffs = fit_curve(annual, degree)
    plt.plot(annual["year"], curve_value(coeffs, annual["year"]), label="degree " + str(degree))

plt.plot(annual["year"], annual["co2"], "k.", ms=4, label="annual average")
plt.xlabel("year")
plt.ylabel("CO2 (parts per million)")
plt.title(f"Three curves fitted to the same {len(annual)} years")
plt.legend()
plt.show()
""")

md("""
The parabola and the cubic go through the data; the straight line does not. Look for the orange
line and you will not find it — over these years the cubic sits on top of the parabola almost
exactly, which is worth remembering, because in a moment we will have to tell those two apart.
Put numbers on the three of them and the improvement is not subtle.
""")

code("""
for degree in [1, 2, 3, 5, 9]:
    coeffs = fit_curve(annual, degree)
    print("degree", degree, "misses by",
          round(typical_miss(annual["co2"], curve_value(coeffs, annual["year"])), 3), "ppm")
""")

md(f"""
Degree 2 cuts the miss from {M['full_miss'][1]} ppm to {M['full_miss'][2]}, and it keeps falling
after that: {M['full_miss'][3]} at degree 3, {M['full_miss'][9]} at degree 9. That is not a
coincidence and it is not evidence. **A more flexible curve can always sit closer to the data it
was fitted on** — a degree-9 polynomial has ten coefficients to adjust where a line has two, so
of course it gets nearer. Fit quality on the data you fitted cannot tell you which degree to use,
because it will
always vote for the most flexible one available.

What the bend itself means, though, is real, and the data says so without any model at all. Fit a
separate straight line to each decade and read off its slope.
""")

code("""
for start in [1960, 1970, 1980, 1990, 2000, 2010]:
    decade = annual[(annual["year"] >= start) & (annual["year"] < start + 10)]
    decade_line = fit_curve(decade, 1)
    print(str(start) + "s: CO2 rose", round(decade_line[0], 2), "ppm per year")
""")

md(f"""
{M['decade_rate'][1960]} ppm a year in the 1960s, {M['decade_rate'][2010]} in the 2010s — the rise
itself has roughly tripled, with the 1990s the one decade that barely moved on the one before it.
That is the acceleration itself, straight from the data with no model in it, and it has a
straightforward cause. Part of each year's fossil-fuel and land-use emissions is taken up by the
ocean and by the land biosphere; the rest stays in the air. So the concentration follows the
emissions that have accumulated, and emissions have grown decade on decade. A curve with a bend
in it is not a statistical convenience here; it is what an accelerating source looks like.

Which leaves the harder question. The rise is bending, so degree 1 is too simple. How much bend
should we allow — and how would we know?
""")

# --- section 3 -------------------------------------------------------------
md(f"""
## How do you choose a degree when a flexible curve always fits better?

The trouble with judging a curve by how close it sits to the data is that the data has already
been used. The fix is the oldest trick in the subject and it is one sentence: **Hide some data
from yourself, then check.**

Fit only the years up to {SPLIT_YEAR}, then ask each fitted curve what CO2 was in `last_full` —
a year it has never seen — and compare with what actually happened. {M['n_later']} years of the
future, already in hand, waiting to mark the answer.
""")

code(f"""
train = annual[annual["year"] <= {SPLIT_YEAR}]
later = annual[annual["year"] > {SPLIT_YEAR}]
observed = annual["co2"].iloc[-1]

print("fitting on", len(train), "years, holding back", len(later))
print("what actually happened in", last_full, "was", round(observed, 2), "ppm")
""")

md(f"""
### Predict before you run

Three curves are about to be fitted to {M['first_year']}–{SPLIT_YEAR} and asked about the last
complete year: a straight line, a parabola and a cubic. One of them will come closest. Which?
Change `my_guess` to 1, 2 or 3 and run the cell — committing to a wrong answer is worth more
than being told the right one.
""")

code("""
my_guess = 3

print("I think degree", my_guess, "will come closest")
""")

ask(f"""
### ✏️ Your turn 2

Fit degrees 1, 2 and 3 to `train` only — nothing after {SPLIT_YEAR} — and ask each one what CO2
was in `last_full`, the last complete year of the record. Print, for each degree, what it
predicted and how far that is from `observed`.

Loop over `[1, 2, 3]`, and inside the loop use `fit_curve(train, degree)` and then
`curve_value(coeffs, last_full)`. Collect the three predictions in a list as you go.

**Use these names**, because the self-check looks for them: `predictions`.
""")

answer("""
predictions = []

for degree in [1, 2, 3]:
    coeffs = fit_curve(train, degree)
    predicted = curve_value(coeffs, last_full)
    predictions.append(predicted)
    print("degree", degree, "predicts", round(predicted, 1), "ppm, missing by",
          round(predicted - observed, 1), "ppm")
""", """
assert max(predictions) - min(predictions) > 20, \\
    "three curves that never saw a year after the split disagree about it by tens of ppm; if \\
yours nearly agree they were fitted on the whole record and already know the answer \u2014 check you \\
passed train, not annual, to fit_curve"
print("✓ three forecasts — the closest misses by",
      round(min(abs(p - observed) for p in predictions), 1), "ppm, the worst by",
      round(max(abs(p - observed) for p in predictions), 1), "ppm")
""")

md(f"""
Read those three numbers again, because they are the point of the week.

The straight line predicted {M['pred'][1]} ppm and was {abs(M['err'][1])} ppm low: too simple, it
never saw the acceleration coming. The parabola predicted {M['pred'][2]} and was
{abs(M['err'][2])} ppm low — off by less than the seasonal swing in a single year, from a fit
that stopped {M['last_year'] - SPLIT_YEAR} years before the answer.

And the cubic — the most flexible of the three, the one that sat closest to the training years —
predicted {M['pred'][3]}, missing by {abs(M['err'][3])} ppm. **Worse than the straight line.**
*A curve that memorises the data you gave it fails on the data you did not.* The picture says it
better than the numbers do.
""")

code(f"""
span = np.arange(first_full, last_full + 1)
for degree in [1, 2, 3]:
    coeffs = fit_curve(train, degree)
    plt.plot(span, curve_value(coeffs, span), label="degree " + str(degree))

plt.plot(train["year"], train["co2"], "k.", ms=4, label="fitted on these")
plt.plot(later["year"], later["co2"], "r.", ms=5, label="held back")
plt.axvline({SPLIT_YEAR}, color="0.5", lw=0.8)     # where the curves stopped seeing data
plt.xlabel("year")
plt.ylabel("CO2 (parts per million)")
plt.title(f"Fitted on {{len(train)}} years, checked against {{len(later)}}")
plt.legend()
plt.show()
""")

md("""
Three curves that were nearly on top of each other over the years they were fitted on fan apart
the moment they leave them. *Too simple and you miss the pattern; too flexible and you memorise
the noise.* The held-out years are the only thing on that plot that can tell the three apart.
""")

# --- section 4 -------------------------------------------------------------
md(f"""
## Which degree do the held-out years choose?

One year of checking is thin. Score each curve on all {M['n_later']} held-out years instead, and
score it on its training years too, so the two can be compared.
""")

code(weekkit.CHECKPOINT.format(body=CLASS_TOOLKIT))

ask(f"""
### ✏️ Your turn 3

Loop over `degrees` and, for each one, fit on `train` and record two typical misses: one against
`train` itself, one against `later`. Print both as you go, then plot the two lists against
`degrees`.

The held-out misses run from about one ppm to tens of thousands, so put the y-axis on a log
scale with `plt.yscale("log")` — the log axes from the plotting week, for exactly the reason
you met them there.

**Use these names**, because the self-check looks for them: `degrees`, `train_miss`, `test_miss`.

```
degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9]
train_miss = []
test_miss = []
```
""")

answer("""
degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9]
train_miss = []
test_miss = []

for degree in degrees:
    coeffs = fit_curve(train, degree)
    train_miss.append(typical_miss(train["co2"], curve_value(coeffs, train["year"])))
    test_miss.append(typical_miss(later["co2"], curve_value(coeffs, later["year"])))
    print("degree", degree, "  training", round(train_miss[-1], 3),
          "  held out", round(test_miss[-1], 3))

plt.plot(degrees, train_miss, marker="o", label="years it was fitted on")
plt.plot(degrees, test_miss, marker="o", label="years it never saw")
plt.yscale("log")
plt.xlabel("polynomial degree")
plt.ylabel("typical miss (ppm)")
plt.title(f"Fitted on {len(train)} years, checked on {len(later)}")
plt.legend()
plt.show()
""", """
assert max(test_miss) > 100 * max(train_miss), \\
    "the most flexible degrees should miss the held-out years by hundreds or thousands of ppm, \\
not by about as much as they miss the training years; if the two lists are the same size, the \\
held-out years were inside the fit, or both lists were scored against the same rows \u2014 \\
check you passed train, not annual, to fit_curve, and later, not train, to the second \\
typical_miss"
print("✓ two error curves — training is best at degree",
      degrees[train_miss.index(min(train_miss))], "and the held-out years at degree",
      degrees[test_miss.index(min(test_miss))])
""")

md(f"""
*Watch two lines. When training keeps falling and test turns up, stop.* The training line slides
downhill the whole way, from {M['train_miss'][1]} ppm at degree 1 to {M['train_miss'][9]} at
degree {M['best_train_degree']}, exactly as it must. The held-out line dives to
{M['test_miss'][2]} ppm at degree {M['best_test_degree']} and then climbs: {M['test_miss'][3]} at
degree 3, {M['test_miss'][5]:,.0f} at degree 5, about {M['test_miss_9_about']} at degree 9. It
does not climb smoothly — degree 4 dips back below degree 3 — but by degree 5 the direction is
not in doubt. The widening gap between the two lines is the model learning the training years by heart.

One cut is one cut, though. Before believing degree {M['best_test_degree']}, move the split and
see whether the answer moves with it.
""")

code(f"""
for cut in {OTHER_SPLITS}:
    before_cut = annual[annual["year"] <= cut]
    after_cut = annual[annual["year"] > cut]

    misses = []
    for degree in degrees:
        coeffs = fit_curve(before_cut, degree)
        misses.append(typical_miss(after_cut["co2"], curve_value(coeffs, after_cut["year"])))

    print("splitting at", cut, "the best degree is", degrees[misses.index(min(misses))])
""")

ask(f"""
### ✏️ Your turn 4

Which degree would you use, and why? Two or three sentences in the cell below, quoting your own
numbers from your turn 3 and from the four splits above.

A good answer says what the two error curves do differently, gives the held-out miss of the
degree you picked and of at least one you rejected, and says something about whether moving the
split changed your mind.
""")

answer_prose(f"""
Degree {M['best_test_degree']}. On the {SPLIT_YEAR} split the held-out miss falls from
{M['test_miss'][1]} ppm at degree 1 to
{M['test_miss'][2]} ppm at degree {M['best_test_degree']}, and then rises again —
{M['test_miss'][3]} ppm at degree 3 and about {M['test_miss_9_about']} ppm at degree 9 — so
degree {M['best_test_degree']} is the bottom of that curve. The training miss cannot be used to choose,
because it falls the whole way, from {M['train_miss'][1]} ppm to {M['train_miss'][9]} ppm; that
is what more flexibility always buys on the years you fitted. Moving the split did not change my
mind much: {OTHER_SPLITS[1]} and {OTHER_SPLITS[2]} also chose degree
{M['other_best'][OTHER_SPLITS[1]]} and {OTHER_SPLITS[0]} chose degree
{M['other_best'][OTHER_SPLITS[0]]}, so I would say two bends, possibly three, and certainly not
nine.
""")

md(f"""
Between them the four cuts say something a single cut could not. Splitting at {OTHER_SPLITS[1]}
and {OTHER_SPLITS[2]} gives degree {M['other_best'][OTHER_SPLITS[1]]}, as {SPLIT_YEAR} did;
splitting at {OTHER_SPLITS[0]} gives degree {M['other_best'][OTHER_SPLITS[0]]}. So what this
record supports is "two bends, possibly three", not "exactly two" — and every one of the four
cuts rules out the flexible end completely.
""")

# --- section 5 -------------------------------------------------------------
md(f"""
## Does it matter which years you hide?

We held back the *last* {M['n_later']} years. The obvious alternative is to hold back
{M['n_later']} years chosen at random from anywhere in the record — that is what most textbook
train/test splits do, and it is what you would reach for if the rows were unrelated to each
other. `.sample(frac=1, random_state=...)` shuffles a table into a random order, and
`random_state` fixes the shuffle so everyone in the room gets the same one.
""")

code(weekkit.CHECKPOINT.format(body=CLASS_TOOLKIT))

code(f"""
shuffled = annual.sample(frac=1, random_state={SHUFFLE_SEED})
rand_test = shuffled.iloc[:len(later)]      # the same number of years as the honest split
rand_train = shuffled.iloc[len(later):]

print("random split:", len(rand_train), "years to fit on,", len(rand_test), "held out")
print("held-out years, in order:", sorted(rand_test["year"])[:8], "...")
""")

ask(f"""
### ✏️ Your turn 5

Score degrees 1, 2, 3 and 9 two ways, in one loop. The random split: fit on `rand_train`,
measure the typical miss against `rand_test`. The honest split class used: fit on `train`,
measure against `later`. Print the pair for each degree.

You worked the honest split out in your turn 3, but build it again here rather than reading
`test_miss` back — that list is above the checkpoint, and this part should still run if you
restarted the kernel since.

**Use these names**, because the self-check looks for them: `rand_train`, `rand_test`,
`rand_miss`, `honest_miss`.
""")

answer("""
rand_miss = []
honest_miss = []

for degree in [1, 2, 3, 9]:
    rand_coeffs = fit_curve(rand_train, degree)
    rand_miss.append(typical_miss(rand_test["co2"], curve_value(rand_coeffs, rand_test["year"])))

    honest_coeffs = fit_curve(train, degree)
    honest_miss.append(typical_miss(later["co2"], curve_value(honest_coeffs, later["year"])))

    print("degree", degree, "  random split", round(rand_miss[-1], 3),
          "  future held back", round(honest_miss[-1], 3))
""", """
assert len(rand_miss) == 4 and len(honest_miss) == 4, \\
    "each list should hold one miss per degree — 1, 2, 3 and 9"
assert rand_miss[3] > typical_miss(rand_train["co2"],
                                   curve_value(fit_curve(rand_train, 9), rand_train["year"])), \\
    "a curve always sits closer to the years it was fitted on than to years it has never seen, \\
so a degree-9 miss below the fitted-on miss means the score did not come from held-out years \\
\u2014 check you fitted on rand_train and scored against rand_test"
assert rand_miss[3] < honest_miss[3] / 100, \\
    "scattering the held-out years through the record should flatter degree 9 by orders of \\
magnitude; if it does not, check rand_miss and honest_miss did not both get the same split"
print("✓ two ways to split — at degree 9 the random split says",
      round(rand_miss[3], 3), "ppm and the held-back future says",
      round(honest_miss[3], 3), "ppm")
""")

md(f"""
The random split says degree 9 is the *best* of the four, missing by {M['rand_miss'][9]} ppm. The
honest split says degree 9 misses by about {M['test_miss_9_about']} ppm. Same data, same curve,
an answer roughly {int(round(M['leak_factor'], -3)):,} times apart, and only one of them is
true.

The reason is that a year is not independent of its neighbours. Shuffle the record and 1987 can
end up in the held-out set while 1986 and 1988 stay in the training set — and a curve that has
been fitted through both neighbours arrives at 1987 nearly right without knowing anything about
CO2, because all it has to do is bridge a gap two years wide. The years you held back gave their
answers away by sitting next to the years you fitted on. That is called **leakage**: any route by
which information about the data you are scoring on reaches the model before you score it. It is
one of the commonest ways a model that does not work reports that it does.

The cure is not statistical, it is a question about the job. We want to know what CO2 will do
*next*, so the test has to be exactly that: fit on the past, predict the future, never the other
way round.
""")

# --- closing ---------------------------------------------------------------
md(f"""
{weekkit.CLOSING_HEADING}

**Yes — the rise itself is rising, by roughly a factor of three.** CO2 at Mauna Loa went up
{M['decade_rate'][1960]} ppm a year in the 1960s and {M['decade_rate'][2010]} ppm a year in the
2010s, and a straight line through the whole record leaves an arc of misses that no amount of
noise explains. One bend is enough to capture it: a parabola fitted with nothing after
{SPLIT_YEAR} predicted {M['last_year']} to within {abs(M['err'][2])} ppm, while a straight
line was {abs(M['err'][1])} ppm low and a cubic — more flexible, closer to the training
years, worse at the job — was {abs(M['err'][3])} ppm out.

What that does *not* license is trusting a polynomial far beyond the record. The parabola has no
physics in it; it fits because emissions have grown smoothly so far, and it will keep bending
upward whatever happens next, because that is what parabolas do. The homework takes it out to
{TARGET_PPM} ppm anyway, and the spread you get back is the honest measure of how much of a
forecast is data and how much is the modelling choice.
""")

md(weekkit.week_cheatsheet(9))

# --- homework --------------------------------------------------------------
md("""
## Homework

Three parts, on the same record you have had open all along. If you have restarted since class,
run the setup cell at the top and then the cell just below, which rebuilds the functions class
wrote and the split it used.

Class never once asked a curve about a year beyond the record. That is the whole homework.
""")

code(weekkit.CHECKPOINT.format(body=CLASS_TOOLKIT))

ask(f"""
### ✏️ Your turn 6

{TARGET_PPM} ppm is a round number people quote as a milestone; nothing physical happens at
exactly {TARGET_PPM}. When does this record say we get there?

Write `crossing_year(coeffs)`. It should walk the years from 2027 to 2300 in a loop, ask
`curve_value(coeffs, year)` what CO2 is in each, and `return` the first year that reaches
{TARGET_PPM} or more. If none of them does, return `None`. Give it a docstring.

Then fit degrees 1, 2 and 3 to the **whole** of `annual`, collect the three years in a list
called `whole_record_crossings`, and print what each degree says.

Then, as a comment at the end of your cell, say in one sentence how far apart your three answers
are, and whether "when do we reach {TARGET_PPM} ppm" has one answer in this record.

**Use these names**, because the self-check looks for them: `crossing_year`,
`whole_record_crossings`.
""")

answer(f"""
def crossing_year(coeffs):
    \"\"\"The first year from 2027 on at which a fitted curve reaches {TARGET_PPM} ppm, or None.\"\"\"
    for year in range(2027, 2301):
        if curve_value(coeffs, year) >= {TARGET_PPM}:
            return year
    return None


whole_record_crossings = []
for degree in [1, 2, 3]:
    coeffs = fit_curve(annual, degree)
    whole_record_crossings.append(crossing_year(coeffs))
    print("degree", degree, "reaches {TARGET_PPM} ppm in", whole_record_crossings[-1])

{note(f"My three answers run from {M['cross_full'][3]} to {M['cross_full'][1]}, "
       f"{M['cross_full'][1] - M['cross_full'][3]} years apart, so this record does not give "
       f"one date for {TARGET_PPM} ppm — the degree I allow the curve moves it by a quarter of "
       f"a century.")}
""", f"""
assert whole_record_crossings[0] is not None, \\
    "crossing_year gave back nothing for the straight line, which does reach {TARGET_PPM} ppm \\
inside the loop: a function that prints the year hands back None instead, so end it with \\
return, not print"
assert 2027 <= whole_record_crossings[0] <= 2300, \\
    "crossing_year should hand back a YEAR, not a position in the loop"
print("✓ the {TARGET_PPM} ppm crossing — a straight line through the whole record puts it in",
      whole_record_crossings[0])
""")

ask(f"""
### ✏️ Your turn 7

Part 1 fitted the whole record. Class validated its degrees on a fit that used nothing after
{SPLIT_YEAR}. Both are defensible — the whole record uses every year you have, while the
{M['first_year']}–{SPLIT_YEAR} fit is the only one whose forecasting anybody has actually tested
— and they do not agree.

Build `crossings`, a list of four years: degree 1 and degree 2, each fitted to `annual` and to
`train`. Print all four with a label saying which is which, print the spread between the earliest and the
latest, and finally print the one year you would quote.

Then, as a comment at the end of your cell, say in one sentence why you chose that one — the two
facts in the paragraph above pull in opposite directions, so say which of them decided it.

This part re-uses `crossing_year` from part 1; its self-check tells you whether that is working.

**Use these names**, because the self-check looks for them: `crossings`.
""")

answer(f"""
crossings = []

for table, label in [(annual, "the whole record"), (train, "the training years only")]:
    for degree in [1, 2]:
        coeffs = fit_curve(table, degree)
        year = crossing_year(coeffs)
        crossings.append(year)
        print("degree", degree, "fitted on", label, "reaches it in", year)

print("earliest", min(crossings), " latest", max(crossings),
      " spread", max(crossings) - min(crossings), "years")
print("I would quote:", crossings[1])

{note(f"I quote the parabola fitted to the whole record: the parabola is the one degree "
       f"whose forecasting has been tested — on the {SPLIT_YEAR} split it missed the held-out "
       f"years by {M['test_miss'][2]} ppm — and fitting it to every year I have lets it use "
       f"the decades the {M['first_year']}–{SPLIT_YEAR} fit never saw.")}
""", """
assert max(crossings) - min(crossings) > 40, \\
    "four years within forty of each other means the degree or the training table did not \\
change: holding either one fixed leaves you two answers twice over, and they span about half \\
what four different fits do"
print("✓ four forecasts — they span", max(crossings) - min(crossings),
      "years, from", min(crossings), "to", max(crossings))
""")

ask(f"""
### ✏️ Your turn 8

Two or three sentences in the cell below, and every claim in them has to be one of *your* numbers.

Quote at least three: the year you chose to quote in part 2, the spread between your four
crossing years, and the held-out miss from your turn 3 of the degree you chose. Then answer this:
a newspaper wants one year. What do you give them, and what does the spread tell their reader
about how much of that year came from the record and how much from your choice of curve?
""")

answer_prose(f"""
I would give them {M['cross_full'][2]}, from the parabola fitted to the whole record. My four
crossing years were {M['cross_full'][1]} and {M['cross_full'][2]} on the whole record and
{M['cross_train'][1]} and {M['cross_train'][2]} on {M['first_year']}–{SPLIT_YEAR}, a spread of
{M['cross_spread']} years. The degree matters more than the training window does: changing the
degree moved my answer by {M['cross_full'][1] - M['cross_full'][2]} years on the whole record and
{M['cross_train'][1] - M['cross_train'][2]} years on the {M['first_year']}–{SPLIT_YEAR} fit,
while changing the
window moved it {M['cross_train'][1] - M['cross_full'][1]} years for degree 1 and only
{M['cross_train'][2] - M['cross_full'][2]} for degree 2. Degree 1 is the one I
distrust, because on the held-out years it missed by {M['test_miss'][1]} ppm against
{M['test_miss'][2]} ppm for degree 2, and its {TARGET_PPM} ppm date assumes the rise stops
accelerating on the day the record ends. So the honest thing to tell the reader is that the data
says "some time this century, most likely mid-century", and that the difference between
{min(M['cross_full'][2], M['cross_train'][2])} and {M['cross_train'][1]} is not a measurement at
all — it is which curve I chose to draw through the same points.
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
                        "--execute", "--inplace", "--ExecutePreprocessor.timeout=600",
                        str(sol_path)], capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        print(r.stderr[-4000:])
        sys.exit("the solution did not execute")

    (OUT / f"{SLUG}.ipynb").write_text(json.dumps(stu, indent=1) + "\n")
    print(f"wrote {SLUG}.ipynb ({len(CELLS)} cells) and the executed solution")
    print(f"cache: data/{CACHE_NAME}, downloaded {READ_DATE}")
    print(f"part-year bias at build time: averaging {M['last_year']}'s first {M['part_months']} "
          f"months shifts its mean by {M['lc_bias']:+} ppm. NOT written into the prose — the "
          f"notebook recomputes and prints it, because it falls as the year fills up and the "
          f"room splits between the live file and this cache.")
    print(f"NOAA file: {M['n_monthly']} monthly readings, "
          f"{M['record_first_year']}-{M['record_first_month']:02d} to "
          f"{M['part_year']}-{M['part_months']:02d}")
    print(f"prose is written from the {M['n_annual']} COMPLETE years, "
          f"{M['first_year']}-{M['last_year']}. Every literal number in the markdown is a "
          f"function of those years alone, so months NOAA adds after this build cannot "
          f"contradict it; the counts and months that do move are printed at run time. The "
          f"literals go stale only when {M['last_year'] + 1} becomes complete (NOAA publishes "
          f"its December around February) or if NOAA revises the historical series.")


if __name__ == "__main__":
    main()
    weekkit.gate(9)
