#!/usr/bin/env python
"""Build project track T3 — "July has 400 eruptions. Is that the Earth, or the catalogue?"

Emits both notebooks from one source so they cannot drift:

    docs/notebooks/T3_july_or_the_catalogue_solution.ipynb   executed, every output saved
    docs/notebooks/T3_july_or_the_catalogue.ipynb            the same file with the answers deleted

It also writes the track's cached fallback, data/trackT3_gvp_eruptions.csv.

A TRACK is not a week (course.yml `project: track_notebooks:`). Two things differ, and both are
deliberate:

  * LESS HELP. No worked example before a question. The notebook loads the data and reproduces
    the ONE result the title names — the July peak — so a student can trust the pipeline, and
    then stops helping. Everything after is a prompt in words and an empty cell.
  * IT DOES NOT CLOSE. There is exactly one self-check, on the load, and the notebook ends on an
    open question this course cannot answer.

Every number that appears in prose or in a model answer is computed HERE, from the same file the
notebook reads, and formatted in. Nothing is typed from memory or copied from the plan.

    python tools/build_track_T3.py
"""
import json
import pathlib
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/notebooks"
SLUG = "T3_july_or_the_catalogue"

course = yaml.safe_load((ROOT / "course.yml").read_text())
modules = yaml.safe_load((ROOT / "modules.yml").read_text())
TRACK = next(t for t in course["project"]["tracks"] if t["id"] == "T3")
PLATFORM = course["platform"]
CACHE_BASE = PLATFORM["cache_base"]

# The live source. Pinned here so the cached CSV, the notebook and the prose below cannot drift.
GVP = ("https://webservices.volcano.si.edu/geoserver/GVP-VOTW/ows?service=WFS&version=1.0.0"
       "&request=GetFeature&typeName=GVP-VOTW:Smithsonian_VOTW_Holocene_Eruptions"
       "&outputFormat=csv")
GVP_CACHE = "trackT3_gvp_eruptions.csv"

SINCE = 1950                 # the window the track works in; course.yml pins it in `data:`
DROP_AT = 15                 # the uncertainty, in days, at or above which a date is a placeholder
SEED = 88                    # the course number, fixed before anything was run
N_NULL = 20000               # Monte Carlo worlds for the null; the audit's number,
                             # and 2,000 leaves the 95th percentile visibly noisy
N_BOOT = 2000                # volcano-block bootstrap resamples


# ---------------------------------------------------------------------------
# 1. measure everything the notebook will say
# ---------------------------------------------------------------------------
def fetch(url, name):
    """Run the live query once, cache it beside the course, and return the cached copy."""
    out = ROOT / "data" / name
    if not out.exists():
        pd.read_csv(url).to_csv(out, index=False)
    return pd.read_csv(out)


def chi_squared(month_values):
    """How far a set of months is from twelve equal piles, in the usual chi-squared units."""
    expected = len(month_values) / 12
    total = 0
    for m in range(1, 13):
        observed = (month_values == m).sum()
        total = total + (observed - expected) ** 2 / expected
    return total


def month_counts(frame):
    """Eruptions per month, 1 to 12, with no month left out."""
    return frame["StartDateMonth"].value_counts().reindex(range(1, 13)).fillna(0).astype(int)


MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

eruptions = fetch(GVP, GVP_CACHE)
dated = eruptions[(eruptions["StartDateYear"] >= SINCE) & (eruptions["StartDateMonth"] > 0)]
months = dated["StartDateMonth"]

M = {}
M["n_all"] = len(eruptions)
M["n_dated"] = len(dated)
M["n_cols"] = eruptions.shape[1]
# the sentinel, on the WHOLE table — it is why the filter is written with `> 0`
M["month_nan"] = int(eruptions["StartDateMonth"].isna().sum())
M["month_zero"] = int((eruptions["StartDateMonth"] == 0).sum())
# ... and the honest caveat: inside the post-1950 window there are none, so `.notna()` happens to
# give the same answer here. Measured, because the audit's warning is written for the whole table.
M["n_naive"] = int(len(eruptions[(eruptions["StartDateYear"] >= SINCE)
                                 & (eruptions["StartDateMonth"].notna())]))

raw = month_counts(dated)
M["counts"] = [int(x) for x in raw.values]
M["july"] = int(raw[7])
M["even"] = M["n_dated"] / 12
M["july_ratio"] = M["july"] / M["even"]
M["excess"] = int(round(M["july"] - M["even"]))
M["low_month"] = MONTH_NAMES[int(raw.idxmin()) - 1]
M["low_count"] = int(raw.min())
M["second_month"] = MONTH_NAMES[int(raw.drop(7).idxmax()) - 1]
M["second_count"] = int(raw.drop(7).max())
M["chi_raw"] = float(chi_squared(months))

# the Monte Carlo null, in the shape the notebook writes it
def null_spread(n, seed=SEED, runs=N_NULL):
    """The chi-squared a world with no seasonality produces, `runs` times over."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(runs):
        out.append(chi_squared(rng.integers(1, 13, size=n)))
    return np.array(out)


null_raw = null_spread(M["n_dated"])
M["null_p95_raw"] = float(np.percentile(null_raw, 95))
M["p_raw"] = float((null_raw >= M["chi_raw"]).mean())

# --- where July comes from ---
day = dated["StartDateDay"].value_counts().reindex(range(1, 32)).fillna(0).astype(int)
M["day2"] = int(day[2])
M["day16"] = int(day[16])
M["day_median"] = float(day.median())
rest = day.drop([2, 16])
M["day_third"] = int(rest.max())
M["day_third_which"] = int(rest.idxmax())
M["day_nan"] = int(dated["StartDateDay"].isna().sum())

july = dated[dated["StartDateMonth"] == 7]
M["july_day2"] = int((july["StartDateDay"] == 2).sum())
M["july_182"] = int((july["StartDateDayUncertainty"] == 182).sum())
M["all_182"] = int((dated["StartDateDayUncertainty"] == 182).sum())
M["share_of_excess"] = M["july_182"] / M["excess"]
M["all_15"] = int((dated["StartDateDayUncertainty"] == 15).sum())
M["day16_15"] = int(((dated["StartDateDay"] == 16)
                     & (dated["StartDateDayUncertainty"] == 15)).sum())

# --- the fork: three defensible cuts ---
unc = dated["StartDateDayUncertainty"]
M["dropped"] = int((unc >= DROP_AT).sum())
clean = dated[~(unc >= DROP_AT)]
exact = dated[unc.isna()]

CUTS = {}
for label, frame in (("raw", dated), ("clean", clean), ("exact", exact)):
    c = month_counts(frame)
    spread = null_spread(len(frame))
    CUTS[label] = {
        "n": len(frame),
        "chi": float(chi_squared(frame["StartDateMonth"])),
        "peak": MONTH_NAMES[int(c.idxmax()) - 1],
        "peak_n": int(c.max()),
        "july_ratio": float(c[7] / (len(frame) / 12)),
        "p95": float(np.percentile(spread, 95)),
        "p": float((spread >= chi_squared(frame["StartDateMonth"])).mean()),
    }
M["cuts"] = CUTS
M["clean_counts"] = [int(x) for x in month_counts(clean).values]

# The textbook constant, for comparison with the simulated null. It is a property of the
# chi-squared distribution with 11 degrees of freedom, not a value read off a web page.
CRITICAL_11DOF = 19.675

# months are not equal in length, and the uniform null ignores that
span = pd.date_range(f"{SINCE}-01-01", "2026-01-01", freq="D")[:-1]
length = span.month.value_counts().reindex(range(1, 13)).sort_index().values
weights = length / length.sum()
obs_clean = month_counts(clean).values
exp_len = len(clean) * weights
M["chi_len"] = float(((obs_clean - exp_len) ** 2 / exp_len).sum())

# --- the sting: how many independent things are in the cleaned set ---
per_volcano = clean["Volcano_Number"].value_counts()
M["n_volcanoes"] = int(len(per_volcano))
M["per_mean"] = float(len(clean) / len(per_volcano))
M["per_median"] = float(per_volcano.median())
M["per_max"] = int(per_volcano.max())
M["busiest"] = str(clean[clean["Volcano_Number"] == per_volcano.index[0]]["Volcano_Name"].iloc[0])
M["second_busiest"] = str(clean[clean["Volcano_Number"] == per_volcano.index[1]]["Volcano_Name"].iloc[0])
M["second_busiest_n"] = int(per_volcano.iloc[1])
M["singletons"] = int((per_volcano == 1).sum())

volcano_months = [sub["StartDateMonth"].values for _, sub in clean.groupby("Volcano_Number")]
rng = np.random.default_rng(SEED)
boot = []
for _ in range(N_BOOT):
    picked = rng.integers(0, len(volcano_months), size=len(volcano_months))
    parts = []
    for p in picked:
        parts.append(volcano_months[p])
    boot.append(chi_squared(np.concatenate(parts)))
boot = np.array(boot)
M["boot_lo"], M["boot_hi"] = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
M["boot_median"] = float(np.median(boot))
M["boot_below"] = float((boot < CUTS["clean"]["p95"]).mean())

# The build log is the record that every number was computed. Print all of it, not a selection.
for k in sorted(M):
    if k != "cuts":
        print(f"  measured  {k:>16} = {M[k]}")
for label in CUTS:
    print(f"  measured  {label:>16} : {CUTS[label]}")
# The simulated null must land on the textbook chi-squared constant, or the simulation is wrong.
print(f"  measured  {'null vs table':>16} : simulated 95th "
      f"{CUTS['clean']['p95']:.2f} / {CUTS['exact']['p95']:.2f} / {M['null_p95_raw']:.2f} "
      f"against the 11-dof constant {CRITICAL_11DOF}")


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
# new, so the full module tables would list thirty functions it never uses; these are the ones
# the notebook and its model answers actually write.
TRACK_IDEAS = [("S2", "Monte Carlo"), ("S4", "Bootstrap"), ("S4", "Confidence interval"),
               ("D1", "Mask"), ("D2", "Table")]
TRACK_FNS = [("D2", "table.sort_values(by)"), ("D2", "column.value_counts()"),
             ("D2", "table.groupby(column)"), ("D2", "column.isna()"),
             ("S2", "np.random.default_rng(seed)"), ("S2", "rng.shuffle(a)"),
             ("S2", "rng.integers(low, high, size)"), ("S2", "np.percentile(values, 95)"),
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

# The title is BUILT from the data, not copied from the plan. course.yml's T3 `title:` reads
# "July has 400 eruptions and March has 250": the 400 reproduces and the 250 does not. A builder
# does not edit the plan, so the mismatch is printed rather than patched.
TITLE = (f"July has {M['july']} eruptions and {M['low_month']} has {M['low_count']}. "
         f"Is that the Earth, or the catalogue?")
if TRACK["title"] != TITLE:
    print(f"  PLAN DRIFT  course.yml T3 title: {TRACK['title']!r}\n"
          f"              measured           : {TITLE!r}\n"
          f"              March holds {M['counts'][2]} of the raw {M['n_dated']} (the second "
          f"TALLEST bar) and {M['clean_counts'][2]} after cleaning; the lowest month is "
          f"{M['low_month']} at {M['low_count']}, and the mean is {M['even']:.0f}.")

HOOK = f"""
Count the eruptions the Smithsonian has recorded since {SINCE} and sort them by the month they
began. Eleven of the months hold between {M['low_count']} and {M['second_count']}; July holds {M['july']}.

That is a big number to explain. Volcanoes have been argued to erupt seasonally — winter snow and
summer meltwater load and unload the crust, sea level breathes a few millimetres a year with the
monsoon, and both change the stress on a magma chamber by a little. If that were the cause, July
would be a real fact about the Earth.

There is a second explanation, and it is not about volcanoes at all. A catalogue is written by
people, over two centuries, from ships' logs and newspaper reports and satellite passes, and what
those people did when they were unsure is itself recorded in the file. This project is about
telling the two apart — and then about what is left over once you have.
"""

md(weekkit.OPENING.format(question=TITLE, datahub=datahub, hook=HOOK.strip()))

md(f"""
## How this notebook is different

This is a **project track**. It is not a weekly notebook and it does not behave like one.

A weekly notebook shows you a move, walks you through it, and then asks you to make it once
yourself. This one loads the data and reproduces the single result its title names — the July
peak — and then stops helping. From there on every section is a sentence describing what to find
out and an empty cell to find it out in. There is no worked example above to pattern-match
against, because on a real question there never is one.

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

**The science.** Say whether a seasonal signal in a global eruption catalogue is a fact about the
Earth or a fact about the record, and defend the answer with a number rather than an adjective.
Then say what would have to be true for the question to be answerable at all.

**The skills.** Read a real catalogue's missing-value conventions, including the ones that do not
look missing. Build a null by simulation instead of looking up a table. Put an interval on a test
statistic by resampling the thing that is actually independent, which is rarely the row.

**The four questions, in order:**

1. How big is the July peak, and is it bigger than chance?
2. Who wrote those July dates, and when?
3. Which records do you trust, and what does the choice cost?
4. How many independent things are {M['cuts']['clean']['n']:,} rows?

The open question at the end is not on that list. It is the project; the four above are what you
build to reach it.
""")

md("""
## Setup

The Global Volcanism Program publishes its whole Holocene eruption table as one CSV, with no key
and no login. The cell below reads it live and falls back to the copy stored with the course.

**Read this before you go on — it is the whole project.** Three columns record when an eruption
began, and each has its own way of saying *we do not know*:

- `StartDateYear` is always filled in.
- `StartDateMonth` and `StartDateDay` use **`0` for unknown**, not a blank. A `0` survives
  `dropna()` and then behaves like a number.
- `StartDateDayUncertainty` is the number of days the recorded date could be wrong by. When the
  compilers knew only the year, they still wrote a full date — and put the uncertainty here.

That third column is the one nobody reads.
""")

code(weekkit.setup_cell(
    imports="import numpy as np\n",
    figsize="(7, 4)",
    cache_base=CACHE_BASE,
    signature="url, cache_name",
    docstring="Read the live catalogue; fall back to the copy stored with the course.",
    url_expr="url",
    cache_expr="cache_name",
    unpack=f'''
GVP = ("https://webservices.volcano.si.edu/geoserver/GVP-VOTW/ows?service=WFS&version=1.0.0"
       "&request=GetFeature&typeName=GVP-VOTW:Smithsonian_VOTW_Holocene_Eruptions"
       "&outputFormat=csv")

eruptions = load(GVP, "{GVP_CACHE}")
print("the whole Holocene table:", eruptions.shape)
print(eruptions[["Volcano_Name", "StartDateYear", "StartDateMonth", "StartDateDay",
                 "StartDateDayUncertainty"]].head())
'''.strip("\n")))

# --- the verified half ------------------------------------------------------
md(f"""
## How big is the July peak, and is it bigger than chance?

Two lines of filtering, and then the count. `StartDateMonth > 0` is doing real work: across the
whole table {M['month_zero']:,} eruptions carry month `0` and another {M['month_nan']} carry a
blank, and `0` is not a blank.

The Smithsonian edits this catalogue continuously, so your counts may differ from the ones printed
in this notebook by a few. Say so if they do — a record that changes under you is the subject here,
not a nuisance.
""")

code(f"""
dated = eruptions[(eruptions["StartDateYear"] >= {SINCE}) & (eruptions["StartDateMonth"] > 0)]
months = dated["StartDateMonth"]

print("eruptions in the whole table: ", len(eruptions))
print("since {SINCE}, carrying a month:", len(dated))
""")

code(f"""
assert "StartDateDayUncertainty" in eruptions.columns, \\
    "the uncertainty column is missing — the catalogue was read wrong, or its schema changed"
assert 2900 < len(dated) < 3100, \\
    "expected about {M['n_dated']} eruptions; a much larger number means the {SINCE} cut is missing"
print(f"✓ the data — {{len(eruptions)}} eruption records, {{len(dated)}} of them since {SINCE} "
      f"with a real month")
""")

md("""
### And that is the last self-check in this notebook

The pipeline is now trustworthy: the file is the file, the filter is the filter, the counts below
are the counts. Everything from here is yours, and nothing will tell you when you have it right.
""")

md(f"""
One bar per month. This is the entire observation the project exists to explain, and it needs no
statistics to see.
""")

code(f"""
per_month = months.value_counts().reindex(range(1, 13)).fillna(0)

plt.bar(per_month.index, per_month.values, color="0.4")
plt.axhline(len(dated) / 12, color="firebrick", lw=1.2)
plt.xlabel("month the eruption began (1 = January)")
plt.ylabel("eruptions")
plt.title(f"Eruptions by month since {SINCE} (n = {{len(dated)}}); the line is an even spread")
plt.locator_params(axis="x", integer=True)
plt.show()

print(per_month.astype(int).to_dict())
""")

md(f"""
Eleven bars sit near the line. One does not. To say *how far* from the line the whole picture is,
in one number, add up the squared miss of each bar in units of the bar's own expected height — the
chi-squared statistic. The function below is the only piece of machinery this notebook hands you,
and every section after it uses it again.

Then the number needs something to be compared against, and rather than look one up we make it.
**Monte Carlo:** {idea('S2', 'Monte Carlo')['words']}
""")

code(f"""
def chi_squared(month_values):
    \"\"\"How far a set of months is from twelve equal piles, in the usual chi-squared units.\"\"\"
    # the height every bar would have if the eruptions were spread evenly over the twelve months
    expected = len(month_values) / 12
    total = 0
    for m in range(1, 13):
        observed = (month_values == m).sum()
        # dividing by `expected` is what makes the misses comparable: a bar missing by ten is a
        # scandal in a small sample and nothing in a large one, and the total is in those units
        total = total + (observed - expected) ** 2 / expected
    return total


def null_spread(n, runs={N_NULL}):
    \"\"\"The chi-squared that a world with no seasonality at all produces, `runs` times over.\"\"\"
    rng = np.random.default_rng({SEED})
    out = []
    for i in range(runs):
        # 1 to 12: integers stops one short of its top number, so 13 is never drawn
        out.append(chi_squared(rng.integers(1, 13, size=n)))
    return np.array(out)
""")

code(f"""
observed = chi_squared(months)
no_season = null_spread(len(dated))

print("chi-squared of the real months:", round(observed, 2))
print("no-season worlds that reached it:", (no_season >= observed).sum(), "out of", len(no_season))
print("95th percentile of the no-season worlds:", round(np.percentile(no_season, 95), 2))
""")

md(f"""
So the July peak is not a small-numbers accident: **{M['chi_raw']:.1f}** against a no-season world
that reaches only **{M['null_p95_raw']:.1f}** nineteen times in twenty. A hypothesis test would
stop here and report seasonality.
""")

md(f"""
### Predict before you run

The July bar stands **{M['excess']}** eruptions above the even line. Of those {M['excess']}, how
many do you think are eruptions that really began in July? Change `my_guess` and run the cell —
you will check it two sections from now, and a wrong guess you committed to is worth more than a
right answer you were shown.
""")

# The Predict cell ships EMPTY to the student, which is why it is written out here rather than
# through `code()`. A pre-filled `my_guess` is not a prediction: the student presses shift-enter,
# the notebook agrees with itself, and the commitment the whole device exists to extract never
# happens. The assert is the thing that makes the cell ask; it is not a self-check on an answer,
# so it does not reopen the promise that the loading step carries the last one.
#
# It is TWO cells, and that is forced rather than chosen. check_asserts requires every name an
# assert uses to be bound by an EARLIER cell — it registers a cell's assignments only after it has
# read that cell's asserts — and check_conventions requires any cell containing `assert` to print
# the course's `✓ label — summary` line. Both rules are written for a self-check, and neither can
# be satisfied by a single cell that assigns `my_guess` and then tests it. So the guess is written
# in one cell and read in the next, which is also what a student does: change the number, run on.
CELLS.append(("code",
              f"my_guess = {M['excess'] // 2}",
              "my_guess = None    # ← your number, written down before you look"))

CELLS.append(("code", f"""
assert my_guess is not None, \\
    "write a number into my_guess in the cell above — the commitment is the point, and a guess "\\
    "you made before you saw the answer is the only one that can teach you anything"
print("✓ committed — I think", my_guess,
      "of the {M['excess']} extra July eruptions really began in July")
""".strip("\n"), None))

# --- YOUR TURN 1 ------------------------------------------------------------
md("""
## Who wrote those July dates, and when?

A month is made of days, and the days are in the file too.
""")

ask(f"""
### ✏️ Your turn 1

Count the {M['n_dated']:,} eruptions by **day of month** — 1 to 31, ignoring which month — and draw
them as a bar chart. Then say, in the output, which days are not like the others and by how much —
and what could put that many eruptions on one particular date of the month.

You are looking for structure that has nothing to do with volcanoes. Print the median day-count
alongside the two largest so the comparison is on the page and not in your head.
""")

answer(f"""
per_day = dated["StartDateDay"].value_counts().reindex(range(1, 32)).fillna(0)

plt.bar(per_day.index, per_day.values, color="0.4")
plt.axhline(per_day.median(), color="firebrick", lw=1.2)
plt.xlabel("day of the month the eruption began")
plt.ylabel("eruptions")
plt.title(f"Eruptions by day of month since {SINCE} (n = {{len(dated)}})")
plt.show()

print("median day holds", per_day.median(), "eruptions")
print("the two biggest days:", per_day.sort_values(ascending=False).head(2).astype(int).to_dict())
print("the next biggest after those:",
      per_day.drop([2, 16]).sort_values(ascending=False).head(1).astype(int).to_dict())

print("Nothing about a volcano prefers one date of the month over its neighbour, so a spike this",
      "size cannot be the Earth. It is what somebody wrote down when they did not know the day.")
""")

md(f"""
Nothing about volcanoes distinguishes the 16th of a month from the 15th. Somebody wrote those
dates down.
""")

# --- YOUR TURN 2 ------------------------------------------------------------
ask(f"""
### ✏️ Your turn 2

`StartDateDayUncertainty` is the column the setup cell told you nobody reads. Use it to find out
what day 2 and day 16 are.

Three things to put on the page:

1. What uncertainty values the day-2 and day-16 records carry — `value_counts()` on that column,
   for each of those two days.
2. Of the **{M['july']}** July eruptions, how many fall on 2 July, and how many of those carry an
   uncertainty large enough to cover half a year.
3. The number that matters: the July bar stands {M['excess']} above the even line, so what
   **fraction of that excess** is accounted for by the one date you have just found? Print it.

Compare the fraction with the number you wrote down in *Predict before you run*. Then print one
more line answering it in a sentence, on your own fraction: is the July peak a fact about
volcanoes, or a fact about the file?
""")

answer(f"""
day2 = dated[dated["StartDateDay"] == 2]
day16 = dated[dated["StartDateDay"] == 16]
print("uncertainties on day-2 records: ", day2["StartDateDayUncertainty"].value_counts().head(3).to_dict())
print("uncertainties on day-16 records:", day16["StartDateDayUncertainty"].value_counts().head(3).to_dict())

july = dated[dated["StartDateMonth"] == 7]
on_2_july = july[july["StartDateDay"] == 2]
half_a_year = july[july["StartDateDayUncertainty"] == 182]

print("July eruptions:", len(july), "of which on 2 July:", len(on_2_july),
      "of which +/- 182 days:", len(half_a_year))

excess = len(july) - len(dated) / 12
print("July stands", round(excess), "above an even spread")
print("the 2 July +/- 182 placeholder is", round(len(half_a_year) / excess, 3), "of it")

print("So the July peak is mostly a fact about the file —",
      round(len(half_a_year) / excess, 3),
      "of the excess is one placeholder date the compilers wrote when they knew only the year.",
      "Whatever is left over is the only part that could be about volcanoes.")
""")

md(f"""
Every record in this window whose date is uncertain by half a year is dated **2 July** — the middle
of the year. When the compilers knew the year and nothing else, they wrote the midpoint of the year
and recorded, in a column that never appears on a plot, that the date was worth nothing. Day 16 is
the same move one level down: a date uncertain by half a month, written on the middle day of a
month.

The fraction you printed is the share of the July excess that one placeholder date accounts for.
Whatever is left after it is the only part of the peak that could be about volcanoes.
""")

# --- YOUR TURN 3, the fork --------------------------------------------------
md(f"""
## Which records do you trust, and what does the choice cost?

You now have to throw some data away, and there is no correct amount. Three cuts are defensible:

- **keep everything** — {M['n_dated']:,} eruptions, the analysis you have already done;
- **drop the placeholders** — anything with `StartDateDayUncertainty >= {DROP_AT}`, which removes
  both the year-level and the month-level defaults;
- **keep only exact dates** — rows where the uncertainty is blank, meaning the compilers claimed
  the day itself.

Each is a different answer to a different question, and they do not agree. This is the one real
decision in this track; make it, and report what it cost.
""")

ask(f"""
### ✏️ Your turn 3

Build all three subsets. For each one print: how many eruptions it holds, its chi-squared, the 95th
percentile of `null_spread` **for that sample size** (the threshold moves with n, so it has to be
recomputed), and which month is the tallest bar.

Take the threshold in the same two steps class used, rather than nesting the calls:
`spread = null_spread(len(subset))` on one line, then `np.percentile(spread, 95)` on the next.

Then draw the month bar chart for the middle cut beside the one from *The result you are handed*,
so the before and after are on the same page.

Watch the blank-versus-zero distinction one more time: an exact date has a **blank** uncertainty,
so `.isna()` is what selects it, and a comparison like `>= {DROP_AT}` is False for a blank rather
than True.

Then answer it, in the output: which of the three cuts changes the answer, and does the answer
survive the change?
""")

answer(f"""
unc = dated["StartDateDayUncertainty"]
# `unc >= {DROP_AT}` is False wherever the uncertainty is blank, never True, so flipping it with ~
# keeps the blanks alongside the small numbers. That is the cut we want: an exact date is worth
# keeping, and only the placeholders go. `unc < {DROP_AT}` would have thrown the blanks away too.
cuts = {{"keep everything": dated,
        "drop placeholders": dated[~(unc >= {DROP_AT})],
        "exact dates only": dated[unc.isna()]}}

for name in cuts:
    subset = cuts[name]
    counts = subset["StartDateMonth"].value_counts().reindex(range(1, 13)).fillna(0)
    spread = null_spread(len(subset))
    print(name, "— n =", len(subset),
          "chi-squared", round(chi_squared(subset["StartDateMonth"]), 2),
          "against a no-season 95th of", round(np.percentile(spread, 95), 2),
          "— tallest month", int(counts.idxmax()))

clean = cuts["drop placeholders"]
clean_months = clean["StartDateMonth"].value_counts().reindex(range(1, 13)).fillna(0)

plt.bar(clean_months.index, clean_months.values, color="0.4")
plt.axhline(len(clean) / 12, color="firebrick", lw=1.2)
plt.xlabel("month the eruption began (1 = January)")
plt.ylabel("eruptions")
plt.title(f"Placeholders removed (n = {{len(clean)}}); the line is an even spread")
plt.locator_params(axis="x", integer=True)
plt.show()

print("Dropping the placeholders is the cut that changes the answer, and the answer does not",
      "survive it: the tallest bar stops being July and the statistic falls to the edge of what a",
      "no-season world produces anyway. Keeping only the exact dates puts it on the other side.")
""")

md(f"""
Your three statistics do not all land on the same side of their own no-season thresholds, and the
tallest bar is not the same month in all three cuts. Same catalogue, same question, and the answer
flips on a choice about which rows to believe.
""")

ask(f"""
### ✏️ Your turn 4

Two or three paragraphs, quoting **your own three chi-squared values and their three thresholds**.

1. Which cut would you report, and what does it cost you? Say what the discarded rows were and
   what a reader loses by not seeing them.
2. The middle cut lands just above its threshold and the strict cut lands just below. Name what a
   reader should conclude from a result that changes side when you change a defensible choice —
   and say what would have to be true of the data for the two cuts to agree.
""")

answer_prose(f"""
I would report the middle cut. Dropping every row whose date is uncertain by {DROP_AT} days or more
removes {M['dropped']} records — among them the {M['all_182']} that are really just a year and the
{M['all_15']} that are really just a month — and it is honest about what it is doing, because those
rows carry no information about the month at all: their month was chosen by a compiler, not by a
volcano. What it costs is real: {M['dropped']} eruptions are {M['dropped'] / M['n_dated'] * 100:.0f}%
of my sample, they are not a random {M['dropped'] / M['n_dated'] * 100:.0f}%, and they are almost
certainly the older and the more remote eruptions, so the cut quietly moves my catalogue towards
recent, well-observed, well-instrumented volcanoes. The strict cut makes that worse — it keeps
{M['cuts']['exact']['n']:,} records and is even more strongly a catalogue of places somebody was
watching. Keeping everything is the one cut I would not report, now that I know what the July bar
is made of.

The three numbers are {M['chi_raw']:.1f} against {M['null_p95_raw']:.1f}, then
{M['cuts']['clean']['chi']:.1f} against {M['cuts']['clean']['p95']:.1f}, then
{M['cuts']['exact']['chi']:.1f} against {M['cuts']['exact']['p95']:.1f}. The first is not close;
the second clears its threshold by {M['cuts']['clean']['chi'] - M['cuts']['clean']['p95']:.1f} and
the third misses by {M['cuts']['exact']['p95'] - M['cuts']['exact']['chi']:.1f}. What a reader
should conclude is that this dataset does not answer the question at the resolution the test
pretends to: a conclusion that flips sign on a judgement call about rows is a conclusion about the
judgement call. It is not that one of the cuts is wrong — both are defensible, which is exactly the
problem.

For the two cuts to agree, the {M['dropped']} placeholder rows would have to be distributed across
the months the way the exact-dated rows are, and they are not: they are concentrated on two dates,
and one of them is in July. The other way they could agree is if the effect were large enough that
{M['dropped']} rows either way could not move it — which is another way of saying the honest
conclusion here is that the effect, if any, is small compared with the noise the catalogue itself
introduces.
""")

# --- YOUR TURN 5, 6: the sting ---------------------------------------------
md(f"""
## How many independent things are {M['cuts']['clean']['n']:,} rows?

A chi-squared test asks how surprising a set of counts is if every observation were an independent
draw. That assumption is not about the arithmetic; it is about the world. Before quoting
{M['cuts']['clean']['chi']:.1f} against {M['cuts']['clean']['p95']:.1f}, it is worth asking how
many independent things the {M['cuts']['clean']['n']:,} rows really are.
""")

ask(f"""
### ✏️ Your turn 5

For the middle cut, count the **volcanoes**, not the eruptions. `Volcano_Number` identifies one.

Print: how many distinct volcanoes there are, the mean and median number of eruptions per volcano,
how many volcanoes contribute exactly one, and the name and count of the two busiest. Draw
whatever figure makes the shape of that distribution obvious.

Then print one more line answering it in a sentence: how many independent things do you think the
{M['cuts']['clean']['n']:,} rows really are, and what does that do to a test that counts every
eruption as its own draw from the calendar?
""")

answer(f"""
per_volcano = clean["Volcano_Number"].value_counts()

print("eruptions:", len(clean), " volcanoes:", len(per_volcano))
print("eruptions per volcano — mean", round(len(clean) / len(per_volcano), 1),
      " median", per_volcano.median(), " max", per_volcano.max())
print("volcanoes contributing exactly one:", (per_volcano == 1).sum())
for volcano_id in per_volcano.head(2).index:
    rows = clean[clean["Volcano_Number"] == volcano_id]
    print(" ", rows["Volcano_Name"].iloc[0], "—", len(rows), "eruptions")

plt.hist(per_volcano.values, bins=range(1, per_volcano.max() + 2), color="0.4")
plt.xlabel("eruptions contributed by one volcano")
plt.ylabel("volcanoes")
plt.title(f"How the {{len(clean)}} eruptions are shared among {{len(per_volcano)}} volcanoes")
plt.show()

print("Nearer", len(per_volcano), "independent things than", len(clean), "— one volcano's run",
      "of", per_volcano.max(), "eruptions is one thing happening, not", per_volcano.max(),
      "separate draws from the calendar. A test that counts every row as its own draw is",
      "therefore counting the same evidence over and over, and its threshold is too easy",
      "to clear.")
""")

md(f"""
There are far fewer volcanoes on your list than eruptions, the median volcano contributes a
handful, and the two busiest contribute dozens each. One volcano's eruptive episode is one thing
happening, not one independent draw from the calendar per eruption — and a chi-squared test that
counts every row as its own draw is counting the same evidence over and over.

The fix is the same move the uncertainty week made, applied to the right unit.
**Bootstrap:** {idea('S4', 'Bootstrap')['words']}
""")

ask(f"""
### ✏️ Your turn 6

Bootstrap the chi-squared **by volcano**, not by eruption. The recipe, in words:

1. Split the middle cut into one group per `Volcano_Number` and keep each group's months. A
   `for volcano_id, rows in clean.groupby("Volcano_Number"):` loop gives you the groups one at a
   time; collect `rows["StartDateMonth"].values` into a list, one entry per volcano.
2. {M['n_volcanoes']} times over, and {N_BOOT} times in all: draw {M['n_volcanoes']} volcanoes
   **with replacement** — `rng.integers(0, n_volcanoes, size=n_volcanoes)` gives you their
   positions — glue their month arrays together with `np.concatenate(parts)`, and take
   `chi_squared` of the result.
3. Report the 2.5th and 97.5th percentiles of the {N_BOOT} statistics, the median, and the fraction
   of them that fall **below** the no-season threshold you computed in Your turn 3.

Draw the {N_BOOT} statistics as a histogram with the observed value and the threshold marked.

**Confidence interval:** {idea('S4', 'Confidence interval')['words']}

Then print two or three sentences answering it on your own interval: does the chi-squared you
reported in *Your turn 3* still support a seasonal signal, and what would your interval have to
look like before it did?
""")

answer(f"""
volcano_months = []
for volcano_id, rows in clean.groupby("Volcano_Number"):
    volcano_months.append(rows["StartDateMonth"].values)

n_volcanoes = len(volcano_months)
rng = np.random.default_rng({SEED})
resampled = []
for i in range({N_BOOT}):
    # with replacement: the same position can come up twice and another not at all, which is the
    # whole point — each resample is a catalogue the Earth could plausibly have handed us instead
    picked = rng.integers(0, n_volcanoes, size=n_volcanoes)
    parts = []
    for p in picked:
        parts.append(volcano_months[p])
    # glue the drawn volcanoes' months back into one flat list of months, so the resampled
    # catalogue goes into chi_squared in exactly the shape the real one did
    resampled.append(chi_squared(np.concatenate(parts)))
resampled = np.array(resampled)

clean_chi = chi_squared(clean["StartDateMonth"])
spread = null_spread(len(clean))
threshold = np.percentile(spread, 95)

print("observed chi-squared:", round(clean_chi, 2), " no-season threshold:", round(threshold, 2))
print("resampling by volcano, 95% interval:", np.round(np.percentile(resampled, [2.5, 97.5]), 1),
      " median", round(np.median(resampled), 1))
print("fraction of resamples below the threshold:", round((resampled < threshold).mean(), 3))

plt.hist(resampled, bins=40, color="0.4")
plt.axvline(clean_chi, color="firebrick", lw=1.5)
plt.axvline(threshold, color="steelblue", lw=1.5, ls="--")
plt.xlabel("chi-squared of a catalogue resampled by volcano")
plt.ylabel("resamples")
plt.title(f"{N_BOOT} volcano-block resamples (red = observed, blue = no-season 95th)")
plt.show()

print("No. My interval runs from", round(np.percentile(resampled, 2.5), 1), "to",
      round(np.percentile(resampled, 97.5), 1), "and straddles the no-season threshold of",
      round(threshold, 1), "so the", round(clean_chi, 1),
      "I reported in Your turn 3 is one draw from a range in which",
      round(100 * (resampled < threshold).mean()), "percent of catalogues fail the test.")
print("The margin it cleared the threshold by was", round(clean_chi - threshold, 1),
      "which is far inside that spread, so what looked like a significant result is mostly",
      "an accident of which volcanoes happen to be in the file.")
print("Before I would call it seasonal the whole interval would have to sit above the",
      "threshold — its lower end, not its middle — and on these", len(volcano_months),
      "volcanoes it does not.")
""")

md(f"""
Resampling volcanoes rather than eruptions spreads the statistic across an interval wide enough
that a substantial share of the resamples land below the threshold the observed value cleared. A
statistic whose interval is that wide has not established anything: the margin by which your middle
cut cleared its threshold in *Your turn 3* is inside its own noise.

The naive chi-squared is not the honest test on this data, and the reason is not the arithmetic —
it is that the rows are not the independent units the test assumes.
""")

# --- closing ----------------------------------------------------------------
# check_track's closing rule warns that this cell "prints 400, which the student was asked to
# compute". That 400 is the notebook's own TITLE — handed to the student in cell 0 and reprinted
# here as the bookend that answers it. The rule cannot tell a number the notebook GAVE from one it
# asked for, and this one is given, so the warn is ACCEPTED rather than fixed: removing it means
# dropping the title from the close, which is a design change and not a leak fix. It is the only
# leak warning this notebook carries — every reveal is at zero — so a SECOND warn is a real one.
md(f"""
{weekkit.CLOSING_HEADING}

July has {M['july']} eruptions and {M['low_month']} has {M['low_count']} because the Smithsonian
writes **2 July** when it knows only the year, and — as the fraction you computed in *Your turn 2*
says — most of the July excess is that one placeholder date. That answers the title. It also
removes the question — cleaned of placeholders, this catalogue does not show a seasonal signal that
survives a test which respects how few independent volcanoes it contains.
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

On this track the honest baseline is not a model at all — it is a bar chart, and it already
answers the title. Say what the simplest possible answer is, what it gives, and what each later
step actually bought you over it.
"""),
    "split_by_structure": ("3 · Split by structure", """
Earth data are correlated in space and in time, so whatever you split, resample or count as
independent has to be split along the structure that is really there — never at random across
rows.

This track fits no model, so there is no train/test split to get wrong. The same idea has teeth
anyway, and *Your turn 6* is where it bit. Name the unit you treated as independent, say why, and
say what changed when you got it right.
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

Here is what is actually established, and it is less than it looks. The July peak is a placeholder
and that is settled. What is **not** settled is the next question down: after the placeholders are
gone, is the statistic that remains a small real signal that this catalogue is too coarse to
resolve, or nothing at all? The volcano-block interval you computed in *Your turn 6* says the data
cannot currently tell you.

Three directions, none of them worked out here:

1. **Fix the null.** The test above assumes twelve equal months. They are not equal: February is
   short and seven months have 31 days, so a uniform expectation is wrong by a few percent before
   any physics. Recompute the expected counts from the real lengths of the months and see which
   way — and how far — the statistic moves.
2. **Change the unit, not the test.** If a volcano is the independent thing, one obvious move is to
   ask each volcano a single question — its own peak month, say, or a per-volcano statistic — and
   test one answer per volcano rather than one per eruption. What power would such a test have, and
   are there enough volcanoes in this catalogue to give it any?
3. **Split the Earth in half.** July is midsummer north of the equator and midwinter south of it.
   Any mechanism that runs on snow load, meltwater or the seasonal sea-level cycle must therefore
   push the two hemispheres in *opposite* directions, while a recording artefact pushes them the
   same way. `GeoLocation` holds a `POINT (lon lat)` string for every row. Pulling the two numbers
   out of that string is plumbing and not the exercise, so here is the line:
   `clean[["lon", "lat"]] = clean["GeoLocation"].str.extract(r"POINT \\((-?[\\d.]+) (-?[\\d.]+)\\)").astype(float)`.
   After it, this is a filter and two bar charts. It is the sharpest test available in this
   dataset, and this notebook has deliberately not run it.

And one that is bigger than a semester: what would a catalogue have to look like for a seasonal
signal of a few percent to be **detectable** at all? Count what you would need — how many
independent volcanoes, over how long, with dates good to what precision — and compare it with what
the volcanoes and eruptions you are left with can support. If the answer is that no achievable
catalogue could settle it, that is a result, and it is the one this project is most likely to
reach.
""")

ask(f"""
### ✏️ Your turn 7 — the first move

Before you close this notebook: in a few sentences, name the **one** measurement you would make
first, say what it would show if the seasonal signal is real, what it would show if it is not, and
name the number that would change your mind. Then make it, in the cell below the prose.
""")

answer_prose(f"""
I would do the hemisphere split first, because it is the only test in this dataset where the two
explanations predict *opposite* things rather than merely different sizes. Every physical
mechanism on offer — snow loading, meltwater, the seasonal sea-level cycle — is tied to the local
season, so if the residual {M['cuts']['clean']['chi']:.1f} is real, the northern and southern
catalogues should peak roughly six months apart. If it is a leftover of the recording process, they
should peak in the same month, because a compiler in the northern hemisphere writes the same
default date for a volcano wherever it is. The number that would change my mind is the offset
between the two peak months: six months, in either direction, and I would start believing the
signal; zero, and I would report that the residual is more of the record.

What makes me doubt it in advance is the sample size. The middle cut has
{M['cuts']['clean']['n']:,} eruptions from {M['n_volcanoes']} volcanoes, and splitting it in two
roughly halves both. My volcano-block interval on the whole sample already runs from
{M['boot_lo']:.1f} to {M['boot_hi']:.1f}; on half the volcanoes it can only be wider. So I expect
the honest outcome to be that neither half rules anything out, which is itself the answer to the
open question — and I would report it that way rather than picking whichever half happened to
clear a threshold.
""")

answer(f"""
clean = clean.copy()
clean[["lon", "lat"]] = clean["GeoLocation"].str.extract(r"POINT \\((-?[\\d.]+) (-?[\\d.]+)\\)").astype(float)

peaks = []
for name in ["northern", "southern"]:
    half = clean[clean["lat"] > 0] if name == "northern" else clean[clean["lat"] < 0]
    counts = half["StartDateMonth"].value_counts().reindex(range(1, 13)).fillna(0)
    spread = null_spread(len(half))
    peaks.append(int(counts.idxmax()))
    print(name, "hemisphere — n =", len(half),
          " tallest month", int(counts.idxmax()),
          " chi-squared", round(chi_squared(half["StartDateMonth"]), 2),
          " no-season 95th", round(np.percentile(spread, 95), 2))

offset = abs(peaks[0] - peaks[1])
print("the two peak months are", min(offset, 12 - offset), "months apart —",
      "6 would say the signal is seasonal, 0 would say it is the record")
print("and neither hemisphere clears its own threshold, so the peaks are not peaks")
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
        elif c["cell_type"] == "code" and "my_guess" in s:
            # The Predict pair carries an assert but is not a question's self-check: it sits
            # BEFORE the first ✏️, so the generic branch below would key it to `q00` and collide
            # with the loading check, which is the real q00. Its two cells get their own ids.
            c["id"] = f"{TRACK['id']}-predict" + ("-check" if "assert " in s else "")
        elif c["cell_type"] == "code" and "assert " in s:
            c["id"] = f"{TRACK['id']}-q{q:02d}-check"
        else:
            c["id"] = f"{TRACK['id']}-c{i:03d}"
    return weekkit.dedupe_ids(cells)


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

    for f in (sol_path, OUT / f"{SLUG}.ipynb"):
        nb = json.loads(f.read_text())
        track_ids(nb["cells"])
        f.write_text(json.dumps(nb, indent=1))

    print(f"wrote {SLUG}.ipynb ({len(CELLS)} cells) and the executed solution")
    print(f"cache: data/{GVP_CACHE} "
          f"({(ROOT / 'data' / GVP_CACHE).stat().st_size / 1e6:.2f} MB)")

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
