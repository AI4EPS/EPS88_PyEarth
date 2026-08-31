#!/usr/bin/env python
"""Build project track T7 — "Can you tell where a basalt erupted from its chemistry alone?"

Emits both notebooks from one source so they cannot drift:

    docs/notebooks/T7_from_its_chemistry_alone_solution.ipynb   executed, every output saved
    docs/notebooks/T7_from_its_chemistry_alone.ipynb            the same file with the answers gone

It also writes the track's data asset, data/trackT7_vermeesch_basalts.csv, byte-identical to the
Vermeesch (2006) compilation that has shipped with every EPS 88 offering since 2019. That
compilation has no upstream — it ships with the course — so the notebook reads it straight out of
data/ with `weekkit.asset_setup_cell` and no live/cached fallback. THE CSV MUST BE PUSHED BEFORE
THE NOTEBOOK IS RELEASED, because there is nothing else for it to read.

A TRACK is not a week (course.yml `project: track_notebooks:`). Three things differ:

  * LESS HELP. No worked example before a question. The notebook loads the data and reproduces
    the ONE result the title needs — how far three elements get you — so a student can trust the
    pipeline, and then stops helping. Everything after is a prompt in words and an empty cell.
  * ASSERTS ONLY ON THE LOAD. Downstream there is no single right answer.
  * IT DOES NOT CLOSE. The last section is the open question course.yml records for T7.

Every number in prose or in a model answer is computed HERE, by the same two helpers the notebook
runs, and formatted in. Nothing is typed from memory or copied from the plan. The plan's own
figures (81.2 and 89.7) come from a DIFFERENT protocol — 50-fold repeated cross-validation — which
this course excludes; they are re-measured under that protocol in `verify_plan_numbers()` and the
result is printed, so the plan and the notebook can be compared rather than conflated.

    python tools/build_track_T7.py
"""
import contextlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/notebooks"
SLUG = "T7_from_its_chemistry_alone"
CACHE_NAME = "trackT7_vermeesch_basalts.csv"
SOURCE = ROOT.parent / "offerings/2024-fall_zhu_solutions/docs/exercises/data/Vermeesch2006.csv"

course = yaml.safe_load((ROOT / "course.yml").read_text())
modules = yaml.safe_load((ROOT / "modules.yml").read_text())
TRACK = next(t for t in course["project"]["tracks"] if t["id"] == "T7")
PLATFORM = course["platform"]
CACHE_BASE = PLATFORM["cache_base"]

# Pearce & Cann (1973) Ti-Zr-Y, in this file's column names. Sr is deliberately NOT here: it is
# the element the fork adds.
CLASSIC = ["TiO2_wt_percent", "Zr_ppm", "Y_ppm"]
MAJOR_OXIDES = ["SiO2_wt_percent", "TiO2_wt_percent", "Al2O3_wt_percent", "Fe2O3_wt_percent",
                "FeO_wt_percent", "CaO_wt_percent", "MgO_wt_percent", "MnO_wt_percent",
                "K2O_wt_percent", "Na2O_wt_percent"]
MOBILE = ["Sr_ppm", "K2O_wt_percent"]
SEEDS = list(range(10))          # split seeds; a single split is what week 11 got burnt on
FOREST_SEEDS = list(range(10))   # forest seeds, for the importance ranking
STRENGTHS = [0.0, 0.4, 0.8, 1.2, 1.6]   # simulated alteration, in log units


# ---------------------------------------------------------------------------
# 0. the data asset
# ---------------------------------------------------------------------------
def write_cache():
    """Copy the shipped compilation into data/ under this track's name, byte for byte."""
    out = ROOT / "data" / CACHE_NAME
    if not out.exists():
        shutil.copyfile(SOURCE, out)
    return pd.read_csv(out)


basalts = write_cache()
feature_columns = [c for c in basalts.columns if c != "affinity"]
missing = basalts[feature_columns].isna().mean()


# ---------------------------------------------------------------------------
# 1. the notebook's two helpers, verbatim, so prose and outputs cannot disagree
# ---------------------------------------------------------------------------
def score(rocks, columns, seed=0):
    """Fit a forest on 70% of `rocks` and report how often it is right on the other 30%."""
    labels = rocks["affinity"]
    X_train, X_test, y_train, y_test = train_test_split(
        rocks[columns], labels, test_size=0.3, random_state=seed, stratify=labels)
    filler = SimpleImputer(strategy="median")
    X_train = filler.fit_transform(X_train)
    X_test = filler.transform(X_test)
    forest = RandomForestClassifier(n_estimators=200, random_state=0)
    forest.fit(X_train, y_train)
    return forest.score(X_test, y_test)


def importances(rocks, columns, seed=0):
    """How much of one forest's decisions each column carried. `seed` picks which forest."""
    filler = SimpleImputer(strategy="median")
    filled = filler.fit_transform(rocks[columns])
    forest = RandomForestClassifier(n_estimators=200, random_state=seed)
    forest.fit(filled, rocks["affinity"])
    return pd.Series(forest.feature_importances_,
                     index=columns).sort_values(ascending=False)


def weathered(rocks, strength, seed=0):
    """A copy of the table with the mobile elements multiplied by a random factor."""
    rng = np.random.default_rng(seed)
    out = rocks.copy()
    for name in MOBILE:
        out[name] = out[name] * np.exp(rng.normal(0.0, strength, len(out)))
    return out


def swept(rocks, columns):
    """The accuracy of one column list over every split seed, as an array."""
    return np.array([score(rocks, columns, s) for s in SEEDS])


# ---------------------------------------------------------------------------
# 2. measure everything the notebook will say
# ---------------------------------------------------------------------------
M = {}
counts = basalts["affinity"].value_counts()
M["n_rows"] = len(basalts)
M["n_columns"] = len(basalts.columns)
M["n_features"] = len(feature_columns)
M["n_oib"], M["n_iab"], M["n_morb"] = int(counts["OIB"]), int(counts["IAB"]), int(counts["MORB"])
M["baseline"] = float(counts.max() / len(basalts))

# missingness: the fact that decides whether the fork is even runnable
M["n_complete"] = int((missing == 0).sum())
M["best_column"], M["best_missing"] = str(missing.idxmin()), float(missing.min())
M["worst_column"], M["worst_missing"] = str(missing.idxmax()), float(missing.max())
M["rows_all"] = int(len(basalts.dropna(subset=feature_columns)))
M["rows_major"] = int(len(basalts.dropna(subset=MAJOR_OXIDES)))
M["rows_classic"] = int(len(basalts.dropna(subset=CLASSIC)))
M["n_test"] = int(len(train_test_split(basalts, test_size=0.3, random_state=0,
                                       stratify=basalts["affinity"])[1]))
drawn = basalts[CLASSIC].dropna()
M["zr_span"] = float(drawn["Zr_ppm"].max() / drawn["Zr_ppm"].min())

CUTS = {"classic": CLASSIC, "major": MAJOR_OXIDES, "all": feature_columns,
        "classic_sr": CLASSIC + ["Sr_ppm"]}
S = {k: swept(basalts, v) for k, v in CUTS.items()}
for k, v in S.items():
    M[f"{k}_mean"], M[f"{k}_min"], M[f"{k}_max"] = float(v.mean()), float(v.min()), float(v.max())
M["all_gain"] = M["all_mean"] - M["classic_mean"]
M["major_gain"] = M["major_mean"] - M["classic_mean"]
M["sr_gain"] = M["classic_sr_mean"] - M["classic_mean"]
M["sr_share_of_gain"] = M["sr_gain"] / M["all_gain"]

# per-seed, which is the whole point of sweeping
gap_all = S["all"] - S["classic"]
gap_sr = S["classic_sr"] - S["classic"]
M["gap_all_min"], M["gap_all_max"] = float(gap_all.min()), float(gap_all.max())
M["gap_sr_min"], M["gap_sr_max"] = float(gap_sr.min()), float(gap_sr.max())
M["gap_sr_negative"] = int((gap_sr <= 0).sum())
M["gap_major_all"] = float((S["all"] - S["major"]).mean())
M["gap_major_all_min"] = float((S["all"] - S["major"]).min())
M["single_split_spread"] = float(S["classic"].max() - S["classic"].min())
M["worst_seed"] = int(SEEDS[int(np.argmin(S["all"] - S["major"]))])

# the single-element search: 48 candidates, each swept
single = {}
for name in feature_columns:
    if name in CLASSIC:
        continue
    single[name] = float(swept(basalts, CLASSIC + [name]).mean())
single = pd.Series(single).sort_values(ascending=False)
M["best_single"] = str(single.index[0])
M["best_single_score"] = float(single.iloc[0])
M["runners_up"] = [(str(n), float(v)) for n, v in single.iloc[1:4].items()]
M["n_candidates"] = int(len(single))

# what the forest leans on
imp_sum = None
for s in FOREST_SEEDS:
    one = importances(basalts, feature_columns, s)
    imp_sum = one if imp_sum is None else imp_sum.add(one, fill_value=0)
imp = (imp_sum / len(FOREST_SEEDS)).sort_values(ascending=False)
M["top"] = [(str(n), float(v), float(missing[n])) for n, v in imp.head(10).items()]
M["top_share"] = float(imp.head(10).sum())
M["rank"] = {c: int(list(imp.index).index(c) + 1)
             for c in ["TiO2_wt_percent", "Sr_ppm", "Zr_ppm", "K2O_wt_percent",
                       "Nb_ppm", "Y_ppm"]}
M["top_by_seed"] = sorted({str(importances(basalts, feature_columns, s).index[0])
                           for s in FOREST_SEEDS})

# the alteration stress test — the LAST answer only, not the body
W = {}
for strength in STRENGTHS:
    rocks = basalts if strength == 0 else weathered(basalts, strength, 500)
    W[strength] = {k: float(np.array([score(rocks, CUTS[k], s) for s in SEEDS]).mean())
                   for k in ("classic", "classic_sr", "all")}
M["weather"] = W
M["worst_strength"] = STRENGTHS[-1]
M["sr_left"] = W[STRENGTHS[-1]]["classic_sr"] - W[STRENGTHS[-1]]["classic"]
M["all_left"] = W[STRENGTHS[-1]]["all"] - W[0.0]["all"]


def verify_plan_numbers():
    """course.yml quotes 81.2 and 89.7 for T7. Re-measure them under the protocol they came from.

    The audit used a random forest of 500 trees under RepeatedStratifiedKFold(5, 10, seed 0) — 50
    fits — with a median imputer in the pipeline. This course excludes cross-validation
    deliberately (`course.yml`, and the EPS 88 CLAUDE.md's "deliberately excluded" list), so the
    notebook cannot use it and its own numbers are single-held-out-split means over ten seeds.
    Both are computed; neither is quoted as the other.
    """
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
    out = {}
    for label, cols in (("classic", CLASSIC), ("classic+Sr", CLASSIC + ["Sr_ppm"]),
                        ("major 10", MAJOR_OXIDES), ("all 51", feature_columns)):
        pipe = make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=500, random_state=0))
        s = cross_val_score(pipe, basalts[cols], basalts["affinity"], cv=cv, n_jobs=-1)
        out[label] = (float(s.mean() * 100), float(s.std() * 100))
    return out


def pct(x):
    return f"{x * 100:.1f}"


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


TRACK_IDEAS = [("ML3", "Baseline"), ("ML2", "Train/test split"), ("ML4", "Random forest"),
               ("ML4", "Imputation"), ("ML4", "Feature importance"),
               ("ML4", "Missingness as data")]
TRACK_FNS = [("ML3", "train_test_split(X, y, test_size=0.3, random_state=n, stratify=y)"),
             ("ML4", "SimpleImputer(strategy=\"median\")"),
             ("ML4", "filler.fit_transform(X_train) / filler.transform(X_test)"),
             ("ML4", "RandomForestClassifier(n_estimators=200, random_state=0)"),
             ("ML4", "forest.feature_importances_"),
             ("ML4", "pd.Series(values, index=names)"),
             ("ML4", "plt.bar(x, heights) / plt.barh(labels, values)"),
             ("D2", "column.value_counts()"), ("D2", "column.isna()"),
             ("D2", "table.sort_values(by)"),
             ("S2", "np.random.default_rng(seed)")]


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
    """A section of the student's own project — empty in the solution too, because there is no
    model answer to a section whose content is the student's own work."""
    stub = "*(Double-click this cell and replace this line with your answer.)*"
    CELLS.append(("markdown", stub, stub))


datahub = (f"{PLATFORM['datahub']}/hub/user-redirect/git-pull"
           f"?repo={PLATFORM['repo'].replace(':', '%3A').replace('/', '%2F')}"
           f"&branch={PLATFORM['branch']}"
           f"&urlpath=lab%2Ftree%2FEPS88_PyEarth%2F{PLATFORM['notebook_dir']}%2F{SLUG}.ipynb")

TITLE = TRACK["title"]
# The plan's `data:` line is the one thing in the T7 entry that is a measurement. Check it rather
# than trust it; a builder does not edit the plan, so a mismatch is printed.
PLAN_DATA = f"{M['n_rows']} samples, OIB {M['n_oib']} / IAB {M['n_iab']} / MORB {M['n_morb']}"
if PLAN_DATA not in " ".join(TRACK["data"].split()):
    print(f"  PLAN DRIFT  course.yml T7 data: {TRACK['data']!r}\n"
          f"              measured          : {PLAN_DATA!r}")

HOOK = f"""
Basalt is the commonest lava on Earth, and nearly all of it reaches the surface in one of three
places. Under a mid-ocean ridge the mantle rises, the pressure on it falls, and it melts on its
own. Under an island arc the mantle is too cold to melt at all until water squeezed out of a
sinking plate lowers its melting point. Under an ocean island — Hawaii, Iceland — neither applies,
and something hotter appears to be arriving from deeper down. Three plumbing systems, three
magmas; and once the lava has cooled into black rock on a beach, the three look much alike.

Geochemists have claimed since the 1970s that the setting is written in the chemistry, and they
made the claim with three elements on a sheet of graph paper. You have {M['n_rows']} basalts whose
setting somebody established in the field, up to {M['n_features']} measurements on each, and a
machine that can read all of them at once. The question is not only whether the machine can do it.
It is which chemistry you let it see — and what you give up by letting it see more.
"""

md(weekkit.OPENING.format(question=TITLE, datahub=datahub, hook=HOOK.strip()))

md("""
## How this notebook is different

This is a **project track**. It is not a weekly notebook and it does not behave like one.

A weekly notebook shows you a move, walks you through it, and then asks you to make it once
yourself. This one loads the data and reproduces the single result the argument starts from — how
far three elements get you — and then stops helping. From there on every section is a sentence
describing what to find out and an empty cell to find it out in. There is no worked example above
to pattern-match against, because on a real question there never is one.

**There is exactly one self-check in this notebook, and it is on the data loading.** After that,
nothing tells you whether you are right. That is not an oversight and it is not laziness: past the
loading step there is no single right answer here, so a cell that said `assert` would be lying to
you about how research works. What replaces it is the thing researchers actually use — a number
you can predict before you compute it, a result you get twice from different directions, and a
claim you try to break.

**And it does not close.** The last section is a question this course does not know the answer to.
Everything above it is scaffolding; that question is the project.
""")

md(f"""
## What you'll be able to do

**The science.** Say how much of a basalt's tectonic setting is recoverable from its chemistry
alone, and defend a choice of which chemistry to use — against a petrologist who will ask whether
the elements your model leaned on are ones a fifty-million-year-old seafloor rock still remembers.

**The skills.** Compare feature sets honestly: the same model, the same split procedure, and a
sweep over splits rather than one lucky one. Read a model's own account of what it used, and check
that account against how often each column was even measured. Build a stress test for a result
instead of asserting that it is robust.

**The four questions, in order:**

1. How well do three elements tell three settings apart?
2. Which chemistry do you feed the forest?
3. Is the difference you found real, or is it your split?
4. What is the forest actually leaning on?

The open question at the end is not on that list. It is the project; the four above are what you
build to reach it.
""")

md(f"""
## Setup

The table is the compilation from Vermeesch, P. (2006), *Tectonic discrimination of basalts with
classification trees*, Geochimica et Cosmochimica Acta 70, 1839–1848
(doi:10.1016/j.gca.2005.12.016). It ships with the course, so there is nothing to fetch and
nothing to clean: one header row, {M['n_features']} numeric chemistry columns, and an `affinity`
column holding the setting somebody established in the field.

**One property of this file decides everything you do next, so read it before you go on.** *No
column in it is complete.* The best-measured column is `{M['best_column']}` and it is still
{pct(M['best_missing'])}% blank; the worst, `{M['worst_column']}`, is {pct(M['worst_missing'])}%
blank. Nobody measures every element on every rock — you measure what your question needed and
what your laboratory could do that year.

So `dropna()` is not an option here, it is a trap: asking for rows complete across all
{M['n_features']} columns leaves **{M['rows_all']}**. Every accuracy below therefore comes from
filling the holes rather than deleting the rows.

**Imputation:** {idea('ML4', 'Imputation')['words']}
""")

code(weekkit.asset_setup_cell(
    imports=("import numpy as np\n"
             "from sklearn.model_selection import train_test_split\n"
             "from sklearn.impute import SimpleImputer\n"
             "from sklearn.ensemble import RandomForestClassifier\n"),
    figsize="(7, 4)",
    cache_base=CACHE_BASE,
    unpack=f'''
# the Vermeesch (2006) compilation, which ships with the course, so there is nothing to fetch
basalts = pd.read_csv(CACHE + "/{CACHE_NAME}")

feature_columns = []
for name in basalts.columns:
    if name != "affinity":
        feature_columns.append(name)

print(basalts.shape, "-", len(feature_columns), "chemistry columns and one label")
print(basalts["affinity"].value_counts().to_dict())
'''.strip("\n")))

code(f"""
assert basalts.shape == ({M['n_rows']}, {M['n_columns']}), \\
    "expected {M['n_rows']} basalts and {M['n_columns']} columns — the file was read wrong"
assert set(basalts["affinity"]) == {{"IAB", "MORB", "OIB"}}, \\
    "the label column should hold exactly three settings"
print(f"✓ the data — {{len(basalts)}} basalts, {{len(feature_columns)}} chemistry columns, "
      f"and no column complete (the fullest is {{basalts[feature_columns].isna().mean().min():.3f}} blank)")
""")

md("""
### And that is the last self-check in this notebook

The pipeline is now trustworthy: the file is the file, the labels are the labels. Everything from
here is yours, and nothing will tell you when you have it right.
""")

# --- spine question 1, the verified half ------------------------------------
md(f"""
## How well do three elements tell three settings apart?

The three settings differ in how the melt was made, and that shows up in which elements the melt
carries. Ridge basalt comes from mantle that has already had melt taken out of it once, so it is
poor in the elements that leave easily. Arc basalt is melted by water coming off the sinking
plate, and water carries potassium, rubidium, barium, strontium and lead with it while leaving
niobium, tantalum and titanium behind — so an arc basalt is enriched in the first group and
conspicuously short of the second. Ocean island basalt owes nothing to either mechanism and is the
titanium-rich, niobium-rich one.

Pearce, J.A. and Cann, J.R. (1973), *Tectonic setting of basic volcanic rocks determined using
trace element analyses*, Earth and Planetary Science Letters 19, 290–300, turned that into
diagrams you could draw by hand. They published **two** ternary diagrams: Ti–Zr–Y, and
Ti/100–Zr–Sr/2. Strontium is in the second one, so it was never a forbidden element — it was used.
What its authors said, in the same paper, is that alteration moves strontium around, and that is
why Ti–Zr–Y rather than the strontium diagram is the one people reach for on a basalt that has sat
under the ocean. Shervais, J.W. (1982), *Ti–V plots and the petrogenesis of modern and ophiolitic
lavas*, Earth and Planetary Science Letters 59, 101–118, added a third pair, titanium against
vanadium. Niobium is in none of them; niobium-based discrimination is later work.

Start where they did. Two of Ti–Zr–Y, on a log axis because zirconium spans a factor of
{M['zr_span']:.0f} across these rocks and a linear axis would pile most of them against the left edge.
""")

code(f"""
drawn = basalts[{CLASSIC + ["affinity"]}].dropna()

for setting in ["MORB", "OIB", "IAB"]:
    rows = drawn[drawn["affinity"] == setting]
    plt.scatter(rows["Zr_ppm"], rows["TiO2_wt_percent"], s=12, label=setting)

plt.xscale("log")
plt.xlabel("Zr (ppm)")
plt.ylabel("TiO2 (weight percent)")
plt.title(f"{{len(drawn)}} basalts with Ti, Zr and Y all measured")
plt.legend()
plt.show()
""")

md(f"""
Three fields, and they are real: the arc basalts sit low and to the left, the ocean island basalts
high and to the right. They are also not clean. Ridge and ocean island overlap through the whole
middle of the picture, and any line you draw by hand through that overlap will cut some of both.
Whatever this diagram is worth, it is not worth 100%.

To put a number on it, three moves you have already met. **Baseline:**
{idea('ML3', 'Baseline')['words']} **Train/test split:** {idea('ML2', 'Train/test split')['words']}
**Random forest:** {idea('ML4', 'Random forest')['words']}

The cell below is the whole machine, and it is the only machinery this notebook hands you. Both
helpers take the table first, then the columns, then a `seed` — and `seed` means the same thing in
both places: *the arbitrary choice you did not think about*. In `score` it decides which rocks were
held out; in `importances` it decides which forest grew. Vary it.
""")

code(f"""
def score(rocks, columns, seed=0):
    \"\"\"Fit a forest on 70% of `rocks` and report how often it is right on the other 30%.\"\"\"
    labels = rocks["affinity"]
    X_train, X_test, y_train, y_test = train_test_split(
        rocks[columns], labels, test_size=0.3, random_state=seed, stratify=labels)
    filler = SimpleImputer(strategy="median")
    X_train = filler.fit_transform(X_train)
    X_test = filler.transform(X_test)
    forest = RandomForestClassifier(n_estimators=200, random_state=0)
    forest.fit(X_train, y_train)
    return forest.score(X_test, y_test)


def importances(rocks, columns, seed=0):
    \"\"\"How much of one forest's decisions each column carried. `seed` picks which forest.\"\"\"
    filler = SimpleImputer(strategy="median")
    filled = filler.fit_transform(rocks[columns])
    forest = RandomForestClassifier(n_estimators=200, random_state=seed)
    forest.fit(filled, rocks["affinity"])
    return pd.Series(forest.feature_importances_, index=columns).sort_values(ascending=False)


CLASSIC = {CLASSIC}
MAJOR_OXIDES = {MAJOR_OXIDES}
SEEDS = {SEEDS}
""")

code(f"""
baseline = basalts["affinity"].value_counts().max() / len(basalts)

classic = []
for seed in SEEDS:
    classic.append(score(basalts, CLASSIC, seed))
classic = np.array(classic)

print("always guess the commonest setting:", round(baseline, 3))
print("Ti, Zr and Y over", len(SEEDS), "splits — mean", round(classic.mean(), 3),
      " lowest", round(classic.min(), 3), " highest", round(classic.max(), 3))
""")

md(f"""
So Pearce and Cann's three elements carry most of the way: **{pct(M['classic_mean'])}%** against a
{pct(M['baseline'])}% baseline, averaged over {len(SEEDS)} different splits of the same
{M['n_rows']} rocks. Notice what that average is hiding — the ten splits range from
{pct(M['classic_min'])}% to {pct(M['classic_max'])}%, a spread of
{M['single_split_spread'] * 100:.1f} points. Any single one of them would have been quotable, and
would have been {M['single_split_spread'] * 100:.1f} points wrong about its neighbours.

That is the result you are handed, and the last thing this notebook will do for you.
""")

md(f"""
### Predict before you run

You have used 3 of the {M['n_features']} chemistry columns in the file. The other
{M['n_features'] - 3} are sitting there unread — every major oxide, every rare earth, the isotope
ratios.

Change `my_guess` to the number of **accuracy points** you think all {M['n_features'] - 3} of them
would add to {pct(M['classic_mean'])}%, and run the cell. You check it in the next section, and a
wrong guess you committed to is worth more than a right answer you were shown.
""")

code(f"""
my_guess = 5

print("I think the other", len(feature_columns) - 3, "columns are worth", my_guess,
      "accuracy points on top of", round(classic.mean() * 100, 1))
""")

# --- spine question 2, the fork ---------------------------------------------
md(f"""
## Which chemistry do you feed the forest?

This is the one real decision in this track, and there is no correct answer to it. Three cuts are
defensible:

- **the classic diagram** — `CLASSIC`, the three elements above, the ones a geologist would have
  plotted by hand and can still interpret;
- **the ten major oxides** — `MAJOR_OXIDES`, the analysis every laboratory runs on every rock as a
  matter of course, so it is the set most likely to exist for a rock you have not seen yet;
- **everything** — `feature_columns`, all {M['n_features']} of them, including columns that are
  blank for most of the file.

They will not agree. Make the choice, and report what it cost.
""")

ask(f"""
### ✏️ Your turn 1

Score all three cuts the way the cell above scored the first one: over every seed in `SEEDS`, not
on one split. Print each cut's mean, lowest and highest accuracy, and draw the three means as a bar
chart with the {pct(M['baseline'])}% baseline marked on it.

Then answer, in a printed sentence: which of the three would you report to a geochemist, and what
did your guess in *Predict before you run* miss?
""")

answer(f"""
cuts = {{"classic diagram": CLASSIC,
        "ten major oxides": MAJOR_OXIDES,
        "everything": feature_columns}}

means = []
for name in cuts:
    got = []
    for seed in SEEDS:
        got.append(score(basalts, cuts[name], seed))
    got = np.array(got)
    means.append(got.mean())
    print(name, "-", len(cuts[name]), "columns - mean", round(got.mean(), 3),
          " lowest", round(got.min(), 3), " highest", round(got.max(), 3))

plt.bar(list(cuts), means, color="0.4")
plt.axhline(baseline, color="firebrick", lw=1.2)
plt.xlabel("which chemistry the forest was given")
plt.ylabel("accuracy over " + str(len(SEEDS)) + " splits")
plt.title(f"{{len(basalts)}} basalts; the line is always guessing the commonest setting")
plt.show()

gained = (means[2] - means[0]) * 100
print("I would report everything:", round(means[2] * 100, 1),
      "percent, against", round(means[0] * 100, 1), "for the classic diagram —",
      round(gained, 1), "points, where I guessed", my_guess,
      "— I was low by", round(gained - my_guess, 1), "points.")
print("What I missed is that the columns nobody bothered to measure on every rock still carry",
      "information, and a median imputer lets the forest use the rows where they exist.")
""")

md(f"""
Three numbers that disagree, and a fork that has now been taken. Before you defend the choice,
though, there is a prior question, and this course got caught by it on this very file. Two ways of
handling the blanks — deleting the incomplete rows, or filling them — were compared on one split
and differed by 0.105. Swept over ten splits, 0.028 of that was the blanks and 0.077 was the split.

You swept ten seeds above, so you have what you need to check your own.
""")

# --- spine question 3, the seeds --------------------------------------------
md(f"""
## Is the difference you found real, or is it your split?

An accuracy is a property of a model *and* of the {M['n_test']} rocks that happened to land in the
held-out half. Two feature sets compared on one split are two numbers with an unknown amount of coin-flip
in them. Compared on the same ten splits, the coin-flip is largely shared, and the difference is
the thing you can actually talk about.
""")

ask(f"""
### ✏️ Your turn 2

Take the two cuts furthest apart — the classic diagram and everything — and this time keep the
**per-seed difference**, not the two means: one number for each seed, `everything minus classic`.

Print the mean difference, its smallest and largest value across the seeds, and how many of the
{len(SEEDS)} seeds gave a difference of zero or less. Draw the ten differences however makes the
spread visible.

Then answer in a printed sentence: is the gap you reported in *Your turn 1* bigger than the wobble
between splits, and how much of the number you would quote is the split rather than the chemistry?
""")

answer(f"""
differences = []
for seed in SEEDS:
    differences.append(score(basalts, feature_columns, seed) - score(basalts, CLASSIC, seed))
differences = np.array(differences)

print("everything minus the classic diagram, per split:", np.round(differences, 3))
print("mean", round(differences.mean(), 3),
      " lowest", round(differences.min(), 3), " highest", round(differences.max(), 3))
print("splits where the classic diagram won or tied:", (differences <= 0).sum(), "of", len(SEEDS))

plt.bar(SEEDS, differences, color="0.4")
plt.axhline(differences.mean(), color="firebrick", lw=1.2)
plt.xlabel("split seed")
plt.ylabel("accuracy gained by using every column")
plt.title(f"The same comparison on {{len(SEEDS)}} different splits of {{len(basalts)}} basalts")
plt.show()

print("Yes. The gap runs from", round(differences.min(), 3), "to", round(differences.max(), 3),
      "and never reaches zero, so the direction is not a split artefact.")
print("The size still is, partly: the spread of", round(differences.max() - differences.min(), 3),
      "means a single split could have had me quoting anything in that range, and the honest",
      "number to report is the mean of", round(differences.mean(), 3), "with the range beside it.")
""")

md(f"""
Whatever your sweep said about that comparison, it was a comparison between three columns and
{M['n_features'] - 3} more. The place a sweep decides an answer rather than refining it is where
the two things being compared are close together — and there is a comparison like that waiting,
because those {M['n_features'] - 3} columns did not contribute equally.
""")

ask(f"""
### ✏️ Your turn 3

Go through the {M['n_candidates']} chemistry columns that are **not** in `CLASSIC`, one at a time.
For each, score the classic three *plus that one column*, swept over `SEEDS`, and keep the mean.

Print the five that help most, with their accuracies, and print how much of the whole
{M['n_features'] - 3}-column gain from *Your turn 2* the single best one recovers on its own.

That is {M['n_candidates']} columns times ten splits, so the cell has real work to do; a loop
inside a loop is the plainest way to write it, and it does not have to be quick.

Then answer in a printed sentence: is the gain you found spread across the file, or does it live in
a small number of columns — and does the best single column have a clear lead over the runners-up,
or is it a photo finish?
""")

answer(f"""
one_more = {{}}
for name in feature_columns:
    if name in CLASSIC:
        continue
    got = []
    for seed in SEEDS:
        got.append(score(basalts, CLASSIC + [name], seed))
    one_more[name] = np.mean(got)

ranked = pd.Series(one_more).sort_values(ascending=False)
print(ranked.head(5).round(3).to_dict())

best = ranked.iloc[0] - classic.mean()
print("the best single column recovers", round(best, 3), "of the",
      round(differences.mean(), 3), "that all", len(feature_columns) - 3, "columns bought —",
      round(best / differences.mean() * 100), "percent of it")

print(f"One column carries over half of what {{len(feature_columns) - 3}} columns bought, and "
      f"the best is {{ranked.index[0]}} — the gain is concentrated, not spread across the file.")
print(f"Which column it is, though, is nearly a tie: {{ranked.index[1]}} and {{ranked.index[2]}} "
      f"are within {{round((ranked.iloc[0] - ranked.iloc[2]) * 100, 1)}} points of it. So 'the one "
      f"element that matters' is a story the data only half supports.")
""")

# --- spine question 4, the importances --------------------------------------
md(f"""
## What is the forest actually leaning on?

You have just asked which column *helps most when added to three others*. A forest trained on all
{M['n_features']} at once will answer a related but different question — which columns it split on,
and how often — and there is no guarantee the two agree.

**Feature importance:** {idea('ML4', 'Feature importance')['words']}

Two warnings before you read one. A forest's importance ranking wobbles between forests the way an
accuracy wobbles between splits, so one forest's top ten is not a result. And a column that is
blank on most of the file cannot be important however informative it is, so the ranking has to be
read against how often each column was measured at all.
""")

ask(f"""
### ✏️ Your turn 4

Run `importances(basalts, feature_columns, seed)` for several forest seeds and average the
{M['n_features']} numbers across them. Draw the top ten as a horizontal bar chart, and print each
one beside the fraction of the file where that column is blank — `basalts[name].isna().mean()`.

Then answer in a printed sentence, using a criterion instead of an impression. The elements
seawater and low-grade metamorphism move around long after the lava has cooled are the ones with
large ions and a single charge — **strontium, potassium, rubidium, barium, caesium**. Titanium,
zirconium, yttrium and niobium stay put. Where do the mobile elements land in *your* ranking, and
what would that mean for a model shown a rock that had spent fifty million years on the seafloor?
""")

answer(f"""
total = importances(basalts, feature_columns, {FOREST_SEEDS[0]})
for seed in {FOREST_SEEDS[1:]}:
    total = total.add(importances(basalts, feature_columns, seed), fill_value=0)
averaged = (total / {len(FOREST_SEEDS)}).sort_values(ascending=False)

top = averaged.head(10)
for name in top.index:
    print(f"{{name:20s}} {{top[name]:.3f}}   blank on {{basalts[name].isna().mean():.3f}} of the file")

plt.barh(list(top.index)[::-1], list(top.values)[::-1], color="0.4")
plt.xlabel("share of the forest's decisions")
plt.ylabel("chemistry column")
plt.title(f"What {len(FOREST_SEEDS)} forests used, averaged ({{len(basalts)}} basalts)")
plt.show()

mobile = ["Sr_ppm", "K2O_wt_percent", "Rb_ppm", "Ba_ppm", "Cs_ppm"]
places = []
for name in mobile:
    places.append(list(averaged.index).index(name) + 1)
print("the mobile elements rank", places, "out of", len(averaged))

print("Two of the five mobile elements are inside the top five, and Sr is second overall —",
      "so a large part of what this model knows is carried by measurements that seawater can",
      "change. On a fresh compilation that is free accuracy. On a rock that has sat on the",
      "seafloor for fifty million years it is a model reading the rock's later life and",
      "reporting it as the rock's birthplace, with no way to tell the difference and no",
      "warning that it cannot.")
""")

md(f"""
If a mobile element came out near the top of your ranking, you have met the tension this track
exists for, and it is older than machine learning. Pearce and Cann had strontium in one of their two diagrams and warned about it in the same paper; the community
kept the Ti–Zr–Y diagram and largely dropped the strontium one, not because strontium says less but
because what it says stops being trustworthy once seawater has been through the rock.

A forest cannot make that distinction. It cannot tell an altered sample from a fresh one, so it
takes every measurement at face value and is rewarded for doing so on a compilation of mostly fresh
rocks. Choosing the classic diagram over the full chemistry is choosing to give up accuracy you can
measure in exchange for robustness you cannot — at least, not yet, and not from anything you have
computed so far.
""")

ask(f"""
### ✏️ Your turn 5

Two or three paragraphs, quoting **your own numbers** — the three accuracies from *Your turn 1*,
the per-seed range from *Your turn 2*, and where the mobile elements sat in *Your turn 4*.

1. Which cut would you report, and to whom? Say what a reader loses by taking it, and name the kind
   of rock on which you would expect your reported accuracy to be wrong.
2. Pearce and Cann's community preferred the diagram that scores worse. On the evidence you have
   produced, is that preference defensible — and what have you actually measured about it, as
   against what you have assumed?
""")

answer_prose(f"""
I would report the full {M['n_features']}-column model, at {pct(M['all_mean'])}% against
{pct(M['classic_mean'])}% for the classic diagram and {pct(M['major_mean'])}% for the ten major
oxides, and I would report it to somebody classifying rocks from a modern, well-characterised
compilation. The gain is not a split artefact: on the same ten splits the full model beat the
classic diagram every time, by between {pct(M['gap_all_min'])} and {pct(M['gap_all_max'])} points.
What a reader loses is interpretability — nobody can draw {M['n_features']} columns on graph paper,
and a geologist handed a prediction cannot see which measurement produced it. The rock on which I
would expect my number to be wrong is an altered one: a seafloor basalt whose strontium and
potassium have been through hydrothermal circulation. My model has never seen one, because this
compilation is mostly fresh material, and nothing in my accuracy would warn me.

The preference for Ti–Zr–Y is defensible, and I have measured almost nothing about it. What I have
measured is the cost: {pct(M['sr_gain'])} points to drop strontium from the trio, and
{pct(M['all_gain'])} points to drop everything back to three elements. What I have only assumed is
the benefit — that the classic diagram's numbers survive alteration and the others' do not. I know
from the petrology that titanium, zirconium and yttrium are immobile and that strontium and
potassium are not, and I can see in my own ranking that {M['top'][1][0]} is rank
{M['rank']['Sr_ppm']} and {M['top'][3][0]} is rank {M['rank']['K2O_wt_percent']} of
{M['n_features']}. That is enough to know where the risk is. It is not enough to say how large it
is, and every sentence I have written about robustness so far has been an argument from mechanism
rather than a measurement.

So the honest report is two-sided. On this compilation, more chemistry is better, decisively and
repeatably. Off this compilation, the model I would report is the one whose most-used column is the
one a petrologist trusts least — and I have no number for what that costs.
""")

# --- closing ----------------------------------------------------------------
md(f"""
{weekkit.CLOSING_HEADING}

Yes, largely. {pct(M['baseline'])}% is what guessing gets you; three elements chosen in 1973 get
{pct(M['classic_mean'])}%; the ten oxides every laboratory measures get {pct(M['major_mean'])}%;
and all {M['n_features']} columns, holes filled rather than rows deleted, get
{pct(M['all_mean'])}%. The setting really is written in the chemistry.

What the numbers do not settle is which of those you should report, because the best-scoring model
leans hardest on the measurements a fifty-million-year-old rock is least likely to have kept.
""")

md(track_summary())

# --- the project ------------------------------------------------------------
md("""
## What your project must contain

Five sections, empty below, required of **every** EPS 88 project regardless of track. They are
headed here so the shape of a good answer is visible while you work. Fill them in as you go; they
are not a write-up you do at the end.
""")

# course.yml's `required_of_every_project:` values are DESIGN notes, not student prose: they name a
# week number and credit the source of the idea (MLGeo). What goes in the notebook is the same
# requirement said to a student. The five keys are READ from the plan, so a sixth requirement
# cannot be added there and silently skipped here; only the wording is local.
REQUIRED = [list(item)[0] for item in course["project"]["required_of_every_project"]]
STUDENT_WORDING = {
    "one_sentence_answer": ("1 · A one-sentence answer", """
Your claim and its uncertainty, in one sentence, at the top of your report. If you cannot put a
number and a range in it, you do not have a result yet. On this track the range is not optional:
every accuracy you have is a mean over splits with a spread beside it.
"""),
    "baseline_first": ("2 · The trivial baseline", """
Before any statistic, state the dumbest answer to your question and what it gives. Every later
number is reported against it.

On this track you computed it in the first minute — guess the commonest of the three settings — and
the three classes are near enough equal that it is a real floor rather than a formality. Quote
every accuracy in your report against it, and say what each step of extra chemistry bought over it.
"""),
    "split_by_structure": ("3 · Split by structure", """
Earth data are correlated in space and in time, so whatever you split, resample or count as
independent has to be split along the structure that is really there — never at random across rows.

This track splits at random, and you should say why that is a problem here rather than repeating
that it is one. These {n} rows are analyses from many separate published studies; several rows can
be the same lava flow, the same island, the same cruise. A random split puts two analyses of one
rock on opposite sides of it. The file carries no study column, so you cannot fix this by splitting
on one. Say what you would need in order to, and what your accuracy means until you have it.
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

ORDER = ["one_sentence_answer", "baseline_first", "split_by_structure",
         "what_i_got_wrong", "ai_disclosure"]
missing_required = set(REQUIRED) - set(ORDER)
if missing_required:
    sys.exit(f"course.yml requires {sorted(missing_required)} of every project and this notebook "
             f"has no section for it")

for key in ORDER:
    heading, guidance = STUDENT_WORDING[key]
    ask(f"### ✏️ {heading}\n{guidance.rstrip().format(n=M['n_rows'])}")
    blank_prose()

# --- the open question ------------------------------------------------------
# course.yml's open_question wraps the question in a paragraph of evidence and a correction note.
# The QUESTION is what the notebook must end on; the evidence is the plan's summary of an audit,
# and this notebook prints its own measurements instead.
OPEN = re.findall(r"[^.?]*\?", " ".join(TRACK["open_question"].split()))[-1].strip()

md(f"""
## The open question

> **{OPEN}**

Nobody grading this knows the answer, and neither does the literature settle it. Everything above
is the scaffolding; this is the project.

Here is exactly what is established and what is not. Established: on this compilation, more
chemistry classifies better, by {pct(M['all_gain'])} points over the classic diagram, on every one
of ten splits. Established: the model's second-most-used column is one that seawater moves. **Not
established: anything at all about the robustness the classic diagram is supposed to buy.** That
half of the trade has been argued from mechanism in every sentence above, including the ones in
this notebook, and never measured.

Four directions, none of them worked out here:

1. **Score the diagram, not the elements.** Pearce and Cann plot a *ternary*: Ti/100, Zr and 3×Y,
   each divided by their sum. Only the ratios survive that; the overall abundance is thrown away.
   Feeding a forest raw concentrations gives it something the published diagram never had. Convert
   the three columns to ternary fractions, re-score, and find out how much of the
   {pct(M['classic_mean'])}% was the diagram and how much was the extra information.
2. **Measure the robustness instead of assuming it.** You cannot get an altered basalt out of this
   file, but you can make one: multiply the mobile columns by a random factor and see which cut
   degrades. That is the direction *Your turn 6* takes, and it settles less than it looks like it
   does, because the size of the factor is a free parameter and this file cannot tell you the right
   one. The harder version, which nobody here has run, trains on fresh rocks and tests on altered
   ones — the real deployment case, and a strictly nastier test than altering both halves.
3. **Find out how independent {M['n_rows']} rows are.** Which elements a row has is a fingerprint
   of which study measured it, and studies tend to be about one setting at a time. Cluster the rows
   by their pattern of blanks, check how the settings fall across the clusters, and split by
   cluster instead of at random. If the accuracy drops, part of what you measured was bookkeeping.
4. **Ask what would settle it.** The measurement that would decide the whole question is one this
   file does not carry: an alteration index — loss on ignition, or a measured degree of
   sea-floor weathering — on every sample. What would you do with such a column if you had it, and
   how many altered samples would you need before you could say which cut to trust?
""")

ask(f"""
### ✏️ Your turn 6 — the first move

Before you close this notebook: in a few sentences, name the **one** measurement you would make
first. What would it show if the classic diagram's robustness is worth its
{pct(M['all_gain'])}-point cost, what would it show if it is not, and what number would change your
mind?

Then make the measurement, in the cell below the prose.
""")

answer_prose(f"""
I would build an altered version of this compilation and re-score each cut on it, because that is
the only claim in my whole report that I have argued rather than measured. The elements seawater
moves are strontium and potassium, and the elements it leaves alone are titanium, zirconium and
yttrium, so I can imitate alteration by multiplying the mobile columns by a random factor and
leaving the rest untouched. If the classic diagram's robustness is worth its
{pct(M['all_gain'])}-point cost, then somewhere in the range of factors I try the classic diagram
should catch and overtake the cuts that use strontium: the mobile-element models should fall until
they are worse than {pct(M['classic_mean'])}%. If it is not worth the cost, the mobile-element
models should still be ahead at any alteration strength a real rock would plausibly have. The
number that would change my mind is the crossing point — the alteration strength at which the
classic diagram wins — measured against what a real altered seafloor basalt looks like.

I expect the result to be genuinely two-sided, and I expect my own experiment to be the weakest
part of it. The strength of the shake is mine to choose, and nothing in this file calibrates it, so
whatever crossing point I find is a statement about my simulation as much as about basalts. Worse,
altering both halves of the split is the *gentle* version: a forest trained on scrambled strontium
simply learns to distrust strontium, whereas a forest trained on fresh rocks and shown an altered
one is confidently wrong. So I read whatever I get below as a lower bound on the damage, not a
measurement of it, and I would say so in a report.
""")

answer(f"""
def weathered(rocks, strength, seed=0):
    \"\"\"A copy of the table with strontium and potassium multiplied by a random factor.\"\"\"
    rng = np.random.default_rng(seed)
    out = rocks.copy()
    for name in ["Sr_ppm", "K2O_wt_percent"]:
        out[name] = out[name] * np.exp(rng.normal(0.0, strength, len(out)))
    return out


strengths = {STRENGTHS}
tracks = {{"classic diagram": CLASSIC,
          "classic + Sr": CLASSIC + ["Sr_ppm"],
          "everything": feature_columns}}

curves = {{}}
for name in tracks:
    curves[name] = []
    for strength in strengths:
        rocks = weathered(basalts, strength, 500)
        got = []
        for seed in SEEDS:
            got.append(score(rocks, tracks[name], seed))
        curves[name].append(np.mean(got))
    print(name, np.round(curves[name], 3))

for name in curves:
    plt.plot(strengths, curves[name], marker="o", label=name)
plt.xlabel("simulated alteration (spread of the log multiplier on Sr and K2O)")
plt.ylabel("accuracy over " + str(len(SEEDS)) + " splits")
plt.title(f"What alteration costs each cut ({{len(basalts)}} basalts)")
plt.legend()
plt.show()

print("The classic diagram is flat by construction — none of its elements moved.")
edge = np.array(curves["classic + Sr"]) - np.array(curves["classic diagram"])
print("Adding Sr is worth", round(edge[0], 3), "on fresh rocks and", round(edge[-1], 3),
      "at the strongest alteration I tried, so the whole advantage is gone by then and the",
      "two curves cross at a shake of about", strengths[int(np.argmin(np.abs(edge)))], "—",
      "that crossing point is the number I said would change my mind, and it is the answer",
      "my simulation gives rather than one basalts give.")
print("Everything still scores", round(curves["everything"][-1], 3), "after the same alteration,",
      "down only", round(curves["everything"][0] - curves["everything"][-1], 3),
      "— it has enough immobile columns to fall back on, which is a robustness argument the",
      "three-element diagram does not have and I had not thought of before running this.")
""")


# ---------------------------------------------------------------------------
# 5. emit, execute, gate
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def execution_env():
    """The environment nbconvert runs the solution in.

    The notebook reads its data from `platform: cache_base:`, which is this repository on `main`.
    A NEW asset is not there until it is pushed, and a build never pushes — so between writing
    data/{CACHE_NAME} and releasing it, that URL 404s and the solution cannot execute at all.

    Rather than weaken the notebook to suit the build, the kernel is started with a startup file
    that maps THAT ONE url to the byte-identical local copy this build just wrote. It is installed
    only while the url is genuinely missing, it lives in a temporary IPython profile that is
    deleted afterwards, and nothing about it reaches either notebook. Once the CSV is pushed the
    check below passes and the kernel runs with no shim at all.
    """
    url = f"{CACHE_BASE}/{CACHE_NAME}"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            live = r.status == 200
    except Exception:
        live = False
    if live:
        print(f"  cache url resolves: {url}")
        yield None
        return

    local = ROOT / "data" / CACHE_NAME
    print(f"  cache url 404s ({url})\n"
          f"  -> executing against {local}, which is what will be pushed. PUSH IT BEFORE RELEASE.")
    home = tempfile.mkdtemp(prefix="eps88-t7-")
    startup = pathlib.Path(home) / "profile_default" / "startup"
    startup.mkdir(parents=True)
    (startup / "00_unpushed_cache.py").write_text(
        "import pandas as _pd\n"
        f"_URL = {url!r}\n"
        f"_LOCAL = {str(local)!r}\n"
        "_real = _pd.read_csv\n"
        "def _read_csv(path, *a, **k):\n"
        "    return _real(_LOCAL if path == _URL else path, *a, **k)\n"
        "_pd.read_csv = _read_csv\n")
    env = dict(os.environ, IPYTHONDIR=home)
    try:
        yield env
    finally:
        shutil.rmtree(home, ignore_errors=True)


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


def report():
    """The build log is the record that every number was computed. Print all of it."""
    for k in sorted(M):
        if k not in ("cuts", "weather", "top", "rank", "runners_up", "top_by_seed"):
            print(f"  measured  {k:>20} = {M[k]}")
    for k in ("top", "rank", "runners_up", "top_by_seed"):
        print(f"  measured  {k:>20} : {M[k]}")
    for strength in STRENGTHS:
        print(f"  measured  {('alteration ' + str(strength)):>20} : {M['weather'][strength]}")
    plan = verify_plan_numbers()
    print("\n  PLAN CHECK  course.yml quotes 81.2 -> 89.7 for T7. Under the audit's protocol")
    print("              (RF 500, RepeatedStratifiedKFold(5, 10, seed 0), median imputer):")
    for label, (mean, sd) in plan.items():
        print(f"                {label:>12} = {mean:.2f} +/- {sd:.2f}")
    print("              Under THIS notebook's protocol (one 70/30 split, RF 200, ten split")
    print("              seeds, no cross-validation — the course excludes it):")
    print(f"                {'classic':>12} = {M['classic_mean'] * 100:.2f}")
    print(f"                {'classic+Sr':>12} = {M['classic_sr_mean'] * 100:.2f}")
    print(f"                {'major 10':>12} = {M['major_mean'] * 100:.2f}")
    print(f"                {'all 51':>12} = {M['all_mean'] * 100:.2f}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report()

    sol = notebook([cell(k, s) for k, s, _ in CELLS])
    stu = notebook([cell(k, alt if alt is not None else s) for k, s, alt in CELLS])

    sol_path = OUT / f"{SLUG}_solution.ipynb"
    sol_path.write_text(json.dumps(sol, indent=1) + "\n")

    print(f"\nexecuting {sol_path.name} ...")
    with execution_env() as env:
        r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                            "--execute", "--inplace", "--ExecutePreprocessor.timeout=1800",
                            str(sol_path)], capture_output=True, text=True, cwd=ROOT, env=env)
    if r.returncode:
        print(r.stderr[-4000:])
        sys.exit("the solution did not execute")

    (OUT / f"{SLUG}.ipynb").write_text(json.dumps(stu, indent=1) + "\n")

    for f in (sol_path, OUT / f"{SLUG}.ipynb"):
        nb = json.loads(f.read_text())
        track_ids(nb["cells"])
        f.write_text(json.dumps(nb, indent=1))

    print(f"wrote {SLUG}.ipynb ({len(CELLS)} cells) and the executed solution")
    print(f"data asset: data/{CACHE_NAME} "
          f"({(ROOT / 'data' / CACHE_NAME).stat().st_size / 1e3:.0f} kB) — PUSH IT BEFORE RELEASE")

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
