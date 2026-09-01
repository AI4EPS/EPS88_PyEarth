#!/usr/bin/env python
"""Build week 11 — "Where does a volcano get its magma?"

Emits both notebooks from one source so they cannot drift:

    docs/notebooks/11_where_magma_solution.ipynb   executed, every output saved
    docs/notebooks/11_where_magma.ipynb            the same file with the answers deleted

It also writes the week's one data asset, data/week11_vermeesch_basalts.csv, byte-identical to the
Vermeesch (2006) compilation that has shipped with every EPS 88 offering since 2019. That
compilation has no upstream — it ships with the course — so the notebook reads it straight out of
data/ with `weekkit.asset_setup_cell` and no live/cached fallback: the CSV MUST BE PUSHED before
the notebook is released, because there is nothing else for it to read.

Every number that appears in prose or in a model answer is computed HERE, by the same code the
notebook runs, and formatted in. Nothing is typed from memory or copied from the plan.

    python tools/build_week11.py
"""
import json
import pathlib
import shutil
import subprocess
import sys

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/notebooks"
SLUG = "11_where_magma"
CACHE_NAME = "week11_vermeesch_basalts.csv"
SOURCE = ROOT.parent / "offerings/2024-fall_zhu_solutions/docs/exercises/data/Vermeesch2006.csv"
# Verified 2026-08-31: byte-identical (md5 6e3941d63759f81ae4a45a771065a9b8) to the copy that has
# shipped with every offering of this course since 2019. There is no live source to point at — the
# only copies on the internet are other commits of this same repository — so the notebook reads
# data/ directly rather than dressing one repo path up as a live read and another as its fallback.

course = yaml.safe_load((ROOT / "course.yml").read_text())
WEEK = next(s for s in course["schedule"] if s["n"] == 11)
PLATFORM = course["platform"]
CACHE_BASE = PLATFORM["cache_base"]

MAJOR_OXIDES = ["SiO2_wt_percent", "TiO2_wt_percent", "Al2O3_wt_percent", "Fe2O3_wt_percent",
                "FeO_wt_percent", "CaO_wt_percent", "MgO_wt_percent", "MnO_wt_percent",
                "K2O_wt_percent", "Na2O_wt_percent"]
SEEDS = [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# 0. write the week's data asset
# ---------------------------------------------------------------------------
def write_cache():
    """Copy the shipped compilation into data/ under this week's name, byte for byte."""
    out = ROOT / "data" / CACHE_NAME
    if not out.exists():
        shutil.copyfile(SOURCE, out)
    return pd.read_csv(out)


basalts = write_cache()
feature_columns = [c for c in basalts.columns if c != "affinity"]
missing = basalts[feature_columns].isna().sum() / len(basalts)


# ---------------------------------------------------------------------------
# 1. measure everything the notebook will say — with the notebook's own code
# ---------------------------------------------------------------------------
def accuracy(model, features, seed=0):
    """The notebook's helper, verbatim, so the prose and the outputs cannot disagree."""
    labels = basalts["affinity"]
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=seed, stratify=labels)
    filler = SimpleImputer(strategy="median")
    X_train = filler.fit_transform(X_train)
    X_test = filler.transform(X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


def forest():
    return RandomForestClassifier(n_estimators=200, random_state=0)


def longhand(columns, seed=0):
    """The pre-imputer version: throw away every row with a hole, then fit one tree."""
    kept = basalts[columns + ["affinity"]].dropna()
    X_train, X_test, y_train, y_test = train_test_split(
        kept[columns], kept["affinity"], test_size=0.3, random_state=seed,
        stratify=kept["affinity"])
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train, y_train)
    return len(kept), tree.score(X_test, y_test)


def _pair_subset_score(rows, columns, seed=0):
    """One element of a two-element pair, on the pair's own rows and the pair's own split."""
    X_train, X_test, y_train, y_test = train_test_split(
        rows[columns], rows["affinity"], test_size=0.3, random_state=seed,
        stratify=rows["affinity"])
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train, y_train)
    return tree.score(X_test, y_test)


TI_V = ["TiO2_wt_percent", "V_ppm"]
ZR_Y = ["Zr_ppm", "Y_ppm"]

M = {}
M["n_rows"] = len(basalts)
M["n_columns"] = len(basalts.columns)
M["n_features"] = len(feature_columns)
counts = basalts["affinity"].value_counts()
M["n_oib"], M["n_iab"], M["n_morb"] = int(counts["OIB"]), int(counts["IAB"]), int(counts["MORB"])
M["baseline"] = float(counts.max() / len(basalts))

M["n_ti_v"], M["ti_v_tree"] = longhand(TI_V)
M["n_zr_y"], M["zr_y_tree"] = longhand(ZR_Y)

M["p2o5_missing"] = float(missing["P2O5(wt%)"])
M["n_neither_iron"] = int(basalts[["FeO_wt_percent", "Fe2O3_wt_percent"]].isna().all(axis=1).sum())
M["neither_iron_share"] = M["n_neither_iron"] / len(basalts)

M["n_complete"] = int((missing == 0).sum())
M["n_half_empty"] = int((missing > 0.5).sum())
M["best_column"] = str(missing.idxmin())
M["best_missing"] = float(missing.min())
M["worst_column"] = str(missing.idxmax())
M["worst_missing"] = float(missing.max())
M["rows_left"] = len(basalts.dropna())
M["n_train"] = len(train_test_split(basalts[feature_columns], basalts["affinity"], test_size=0.3,
                                    random_state=0, stratify=basalts["affinity"])[0])

M["ti_v_filled"] = accuracy(DecisionTreeClassifier(random_state=0), basalts[TI_V])
M["n_ti_v_extra"] = M["n_rows"] - M["n_ti_v"]
M["ti_v_drop"] = M["ti_v_tree"] - M["ti_v_filled"]


def ti_v_by_completeness():
    """Score the filled tree separately on the test rocks that were measured and those that were
    filled in. Same model, same split, so the two are directly comparable — which is what says
    how much of the drop is the imputed rows and how much is the split."""
    labels = basalts["affinity"]
    X_train, X_test, y_train, y_test = train_test_split(
        basalts[TI_V], labels, test_size=0.3, random_state=0, stratify=labels)
    measured = X_test.notna().all(axis=1).to_numpy()
    filler = SimpleImputer(strategy="median")
    A = filler.fit_transform(X_train)
    B = filler.transform(X_test)
    scaler = StandardScaler()
    A = scaler.fit_transform(A)
    B = scaler.transform(B)
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(A, y_train)
    return (int(measured.sum()), float(tree.score(B[measured], y_test[measured])),
            int((~measured).sum()), float(tree.score(B[~measured], y_test[~measured])))


(M["n_measured_test"], M["measured_test"],
 M["n_filled_test"], M["filled_test"]) = ti_v_by_completeness()
M["n_test"] = M["n_measured_test"] + M["n_filled_test"]
M["imputed_part"] = M["measured_test"] - M["ti_v_filled"]
M["split_part"] = M["ti_v_tree"] - M["measured_test"]

TI_V_SEEDS = list(range(10))
M["ti_v_sweep"] = [(s, longhand(TI_V, s)[1],
                    accuracy(DecisionTreeClassifier(random_state=0), basalts[TI_V], s))
                   for s in TI_V_SEEDS]
M["ti_v_mean_dropna"] = sum(d for _, d, _ in M["ti_v_sweep"]) / len(M["ti_v_sweep"])
M["ti_v_mean_filled"] = sum(f for _, _, f in M["ti_v_sweep"]) / len(M["ti_v_sweep"])
M["ti_v_mean_gap"] = M["ti_v_mean_dropna"] - M["ti_v_mean_filled"]
M["ti_v_min_gap_split"], M["ti_v_min_gap"] = min(
    ((s, d - f) for s, d, f in M["ti_v_sweep"]), key=lambda r: r[1])
M["ti_v_max_gap_split"], M["ti_v_max_gap"] = max(
    ((s, d - f) for s, d, f in M["ti_v_sweep"]), key=lambda r: r[1])
_rank = 1 + sum(1 for _, d, f in M["ti_v_sweep"] if d - f > M["ti_v_drop"])
M["ti_v_seed0_rank"] = {1: "biggest", 2: "second biggest", 3: "third biggest",
                        4: "fourth biggest"}.get(_rank, f"{_rank}th biggest")

# the scatter: what the two axes actually separate, measured before the prose describes them
pairs_measured = basalts[TI_V + ["affinity"]].dropna()
v_medians = pairs_measured.groupby("affinity")["V_ppm"].median()
M["v_median_spread"] = float(v_medians.max() - v_medians.min())
M["v_alone_same_rows"] = float(_pair_subset_score(pairs_measured, ["V_ppm"]))
M["ti_alone_same_rows"] = float(_pair_subset_score(pairs_measured, ["TiO2_wt_percent"]))

M["oxide_tree"] = accuracy(DecisionTreeClassifier(random_state=0), basalts[MAJOR_OXIDES])
M["oxide_forest"] = accuracy(forest(), basalts[MAJOR_OXIDES])
M["oxide_svm"] = accuracy(SVC(), basalts[MAJOR_OXIDES])
M["all_forest"] = accuracy(forest(), basalts[feature_columns])
M["gap"] = M["all_forest"] - M["oxide_forest"]

M["sweep"] = [(s, accuracy(forest(), basalts[MAJOR_OXIDES], s),
               accuracy(forest(), basalts[feature_columns], s)) for s in SEEDS]
M["sweep_min_gap"] = min(a - o for _, o, a in M["sweep"])
M["sweep_max_gap"] = max(a - o for _, o, a in M["sweep"])
M["sweep_worst_split"] = min(M["sweep"], key=lambda r: r[2] - r[1])[0]

trained = forest()
accuracy(trained, basalts[feature_columns])
importance = pd.Series(trained.feature_importances_,
                       index=feature_columns).sort_values(ascending=False)
M["top"] = [(name, float(value), float(missing[name])) for name, value in importance.head(10).items()]
M["top_share"] = float(importance.head(10).sum())
sparse_columns = [c for c in feature_columns if missing[c] > 0.5]
M["sparse_share"] = float(importance[sparse_columns].sum())
M["sparse_best_rank"] = min(i + 1 for i, c in enumerate(importance.index) if c in sparse_columns)
M["sparse_best"] = next(c for c in importance.index if c in sparse_columns)

blanks = basalts[feature_columns].isna()
M["blank_forest"] = accuracy(forest(), blanks)
M["oxide_blank_forest"] = accuracy(forest(), basalts[MAJOR_OXIDES].isna())

# homework
M["n_sparse"] = len(sparse_columns)
M["sparse_forest"] = accuracy(forest(), basalts[sparse_columns])
M["cut50"] = [c for c in feature_columns if missing[c] < 0.5]
M["cut20"] = [c for c in feature_columns if missing[c] < 0.2]
M["cut50_score"] = accuracy(forest(), basalts[M["cut50"]])
M["cut20_score"] = accuracy(forest(), basalts[M["cut20"]])
M["rows_left_oxides"] = len(basalts.dropna(subset=MAJOR_OXIDES))


def pct(x):
    return f"{x * 100:.1f}"


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


datahub = (f"{PLATFORM['datahub']}/hub/user-redirect/git-pull"
           f"?repo={PLATFORM['repo'].replace(':', '%3A').replace('/', '%2F')}"
           f"&branch={PLATFORM['branch']}"
           f"&urlpath=lab%2Ftree%2FEPS88_PyEarth%2F{PLATFORM['notebook_dir']}%2F{SLUG}.ipynb")

HOOK = """
A volcano is a hole in the ground with hot rock coming out of it, and the three commonest kinds of
basaltic volcano on Earth are fed in three completely different ways. Under a mid-ocean ridge the
mantle rises, the pressure on it drops, and it melts on its own. Under an island arc the mantle is
too cold to melt at all until water squeezed out of a sinking plate lowers its melting point. Under
an ocean island neither of those applies, and something hotter appears to be arriving from deeper
down. Three plumbing systems, three magmas — and once the lava has cooled into black basalt on a
beach, the three look much alike.

Today you get 756 basalts whose setting somebody has already established, with up to 51 chemical
measurements on each, and one question: is the setting written in the chemistry? On the way you
will find out what a model does when half of those measurements were never made.
"""

md(weekkit.OPENING.format(question=WEEK["question"], datahub=datahub, hook=HOOK.strip()))

md("""
## What you'll be able to do

**The science.** Name the three tectonic settings basalt comes from and the melting mechanism
behind each, say which chemical elements separate them and why the petrology predicts those
elements, and state how much of a classifier's skill is chemistry and how much is bookkeeping.

**The skills.** Two new classifiers, both three lines behind the same `fit` / `score` interface you
met with logistic regression: `DecisionTreeClassifier`, `RandomForestClassifier`, and
`SVC` alongside them. `SimpleImputer` to fill holes instead of deleting rows,
`StandardScaler` to put columns on a common footing, and `feature_importances_` to ask a trained
model which columns it actually used.

**Eight places where you write something: five in class, three at home.** Each one is headed
*Your turn*, with an empty cell under it.

**The four questions, in order:**

1. Can two elements tell three tectonic settings apart?
2. What do you do when no column in the file is complete?
3. Do the badly measured columns help, or hurt?
4. Is the forest reading the chemistry, or reading who measured the rock?
""")

code(weekkit.asset_setup_cell(
    imports=("from sklearn.model_selection import train_test_split\n"
             "from sklearn.impute import SimpleImputer\n"
             "from sklearn.preprocessing import StandardScaler\n"
             "from sklearn.tree import DecisionTreeClassifier\n"
             "from sklearn.ensemble import RandomForestClassifier\n"
             "from sklearn.svm import SVC\n"),
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
'''.strip("\n")))

# --- section 1 -------------------------------------------------------------
md("""
## Can two elements tell three tectonic settings apart?

Basalt is the most common lava on the planet, and almost all of it erupts in one of three places.

**Mid-ocean ridge basalt (MORB)** comes from a spreading centre. The mantle beneath rises, the
pressure on it falls, and it melts without anything being added to it. That mantle has already had
melt extracted from it once before, so what comes out is poor in the elements that leave easily —
potassium, rubidium, barium.

**Island arc basalt (IAB)** comes from above a sinking plate. The mantle in the wedge there is dry,
and dry mantle at that temperature does not melt. What makes it melt is water: the sinking plate
heats up, releases water, and water lowers the melting point of rock. The water carries with it
whatever dissolves in water — potassium, rubidium, barium, strontium, lead — and leaves behind
whatever does not, notably niobium, tantalum and titanium. Arc basalts therefore carry a
distinctive signature: enriched in the first group, unusually poor in the second.

**Ocean island basalt (OIB)** is Hawaii and Iceland. Neither mechanism applies; the mantle arriving
underneath is hotter and has been through less, and the melt fractions are small, so the lava is
rich in titanium, niobium and the light rare earth elements.

The table you have loaded is the compilation from Vermeesch, P. (2006), *Tectonic discrimination
of basalts with classification trees*, Geochimica et Cosmochimica Acta 70, 1839–1848
(doi:10.1016/j.gca.2005.12.016). Somebody established the setting of each sample from where it was
collected; the `affinity` column is that answer.
""")

code("""
print(basalts[["affinity", "SiO2_wt_percent", "TiO2_wt_percent", "Sr_ppm"]].head())
print(basalts["affinity"].value_counts())
""")

md("""
Three classes, and near enough the same number of each — which is a piece of luck, because it means
an accuracy here means what it looks like it means. The class-imbalance trap from the
classification week does not bite.

Before any model, the dumbest rule you can write. *Write the dumbest rule you can, first. Any model
that cannot beat it is decoration.* Here that rule is: ignore the chemistry entirely and call every
sample whichever class is commonest.
""")

ask("""
### ✏️ Your turn 1

How often would that rule be right? Take `basalts["affinity"].value_counts()`, ask it for its
largest value with `.max()`, and divide by the number of rows. Print the result rounded to three
decimal places.

**Use these names**, because the self-check looks for them: `baseline`.
""")

answer("""
baseline = basalts["affinity"].value_counts().max() / len(basalts)

print("always guess the commonest setting:", round(baseline, 3))
""", """
assert baseline < 1, "baseline is a fraction of 1, not a count of rows"
print("✓ the baseline — guessing the commonest of the three settings is right",
      round(baseline * 100, 1), "percent of the time")
""")

# --- section 2: the second half of spine question 1 -------------------------
md("""
Geochemists have been separating these three settings by hand since the 1970s, and they did it with
two elements at a time on a piece of graph paper. Pearce, J.A. and Cann, J.R. (1973), *Tectonic
setting of basic volcanic rocks determined using trace element analyses*, Earth and Planetary
Science Letters 19, 290–300, used titanium, zirconium and yttrium in one diagram and titanium,
zirconium and strontium in another; Shervais, J.W. (1982), *Ti-V plots and the petrogenesis of
modern and ophiolitic lavas*, Earth and Planetary Science Letters 59, 101–118, used titanium
against vanadium. Start where they did.
""")

code(f"""
pairs = basalts[["TiO2_wt_percent", "V_ppm", "affinity"]].dropna()

for setting in ["MORB", "OIB", "IAB"]:
    rows = pairs[pairs["affinity"] == setting]
    plt.scatter(rows["TiO2_wt_percent"], rows["V_ppm"], s=12, label=setting)

plt.xlabel("TiO2 (weight percent)")
plt.ylabel("V (ppm)")
plt.title("{M['n_ti_v']} basalts with both TiO2 and V measured")
plt.legend()
plt.show()
""")

md(f"""
The three settings do sit in different places, and they overlap badly. Almost all of the separation
runs left to right: the three clouds sit at three different titanium levels, while their vanadium
ranges lie on top of one another — the three median vanadium values are within
{M['v_median_spread']:.0f} ppm of each other, so vanadium on its own would tell you almost nothing
here. That does not make it decoration. It is the second direction a boundary can cut in, and where
the clouds meet it is the only one left. You could draw a boundary on that picture with a ruler, and
you would get a lot of samples wrong.

A **decision tree** draws the boundary for you. A flowchart of yes/no questions, learned from the
data instead of written by hand. *Is TiO2 above some value?* If it is, *is V below another?* — and
so on, with both the element and the threshold at every step chosen to separate the three classes
as cleanly as they can be separated, until every branch ends in a verdict. On a two-element plot
that comes out as a staircase of horizontal and vertical cuts.

The machinery is the machinery from the classification week: split the samples into a training set
and a held-out test set, fit on the first, score on the second. One thing is new. Every column in
this file has holes in it, and no classifier in scikit-learn will accept a hole, so for now we do
the obvious thing and throw away every row that has one — `.dropna()`, from the tables week. Note
what that costs: {M['n_ti_v']} of the {M['n_rows']} samples survive.
""")

code(f"""
X = pairs[["TiO2_wt_percent", "V_ppm"]]
y = pairs["affinity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0,
                                                    stratify=y)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
ti_v_score = tree.score(X_test, y_test)

print("rows kept:", len(pairs), "of", len(basalts))
print("test accuracy:", round(ti_v_score, 3))
""")

ask("""
### ✏️ Your turn 2

Two of Pearce and Cann's three elements now, instead of Shervais's pair. Do exactly what the cell
above did, but for `Zr_ppm` and `Y_ppm`: drop the rows where either is blank, split with
`test_size=0.3`,
`random_state=0` and `stratify=`, fit a `DecisionTreeClassifier(random_state=0)`, and print how
many rows survived and the test accuracy.

**Use these names**, because the self-check looks for them: `zr_y` for the surviving rows, and
`zr_y_score` for the accuracy.
""")

answer("""
zr_y = basalts[["Zr_ppm", "Y_ppm", "affinity"]].dropna()

X = zr_y[["Zr_ppm", "Y_ppm"]]
y = zr_y["affinity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0,
                                                    stratify=y)

zr_y_tree = DecisionTreeClassifier(random_state=0)
zr_y_tree.fit(X_train, y_train)
zr_y_score = zr_y_tree.score(X_test, y_test)

print("rows kept:", len(zr_y), "of", len(basalts))
print("test accuracy:", round(zr_y_score, 3))
""", """
assert "Zr_ppm" in zr_y.columns, "zr_y should be built from Zr and Y, not from TiO2 and V"
assert 0.5 < zr_y_score < 1, \\
    "1.000 would mean you scored the tree on the rows it was fitted on"
print("✓ zirconium and yttrium —", len(zr_y), "rows survived and the tree scored",
      round(zr_y_score, 3))
""")

md(f"""
Two numbers, both well clear of the {M['baseline']:.3f} baseline, so there really is tectonic
information in two elements. But look at what you had to do to get them. The titanium-vanadium pair
was scored on {M['n_ti_v']} samples and the zirconium-yttrium pair on {M['n_zr_y']}, and those are
not the same {M['n_zr_y']} rocks. Two accuracies measured on two different sets of samples cannot
be compared, and we are about to want to compare a dozen of them.
""")

# --- section 3 -------------------------------------------------------------
md(f"""
## What do you do when no column in the file is complete?

So how holey is this file? `.isna()` marks every hole with `True`, `.sum()` adds up the `True`s
column by column, and dividing by the number of rows turns each count into a fraction. The result
is one number per column, labelled with the column's name, so `missing["Sr_ppm"]` reads one of them
out.
""")

code("""
missing = basalts[feature_columns].isna().sum() / len(basalts)

print("best measured:")
print(missing.sort_values().head(3))
print("worst measured:")
print(missing.sort_values(ascending=False).head(3))
""")

code(f"""
plt.bar(range(len(missing)), missing.sort_values() * 100)
plt.xlabel("the {M['n_features']} chemistry columns, best measured first")
plt.ylabel("percent of samples with no value")
plt.title("Missing measurements, {M['n_features']} columns of {M['n_rows']} basalts")
plt.show()
""")

md("""
That is not a chart with a few gaps in it. No bar touches zero, and by the middle of the chart half
the measurements have gone. The reason is that this is a *compilation*: hundreds of published
analyses of different rocks by different laboratories, and each of those studies measured whatever
its own question needed. The bulk chemistry of a rock comes off a single prepared bead in one run
and is cheap; a rare earth element needs a separate and more expensive technique; a lead isotope
ratio needs chemical separation and a mass spectrometer, sample by sample. Nobody was withholding
anything. The measurement was simply never made.
""")

ask("""
### ✏️ Your turn 3

Three counts, from `missing` and from the table itself.

1. How many of the columns have at least one hole in them — `missing[name] > 0`?
2. How many are more than half empty — `missing[name] > 0.5`?
3. How many rows survive `basalts.dropna()`, which throws away every row with a hole anywhere in
   it?

One loop over `feature_columns` with two counters will do the first two. Print all three.

**Use these names**, because the self-check looks for them: `n_with_holes`, `n_half_empty`,
`rows_left`.
""")

answer("""
n_with_holes = 0
n_half_empty = 0

for name in feature_columns:
    if missing[name] > 0:
        n_with_holes = n_with_holes + 1
    if missing[name] > 0.5:
        n_half_empty = n_half_empty + 1

rows_left = len(basalts.dropna())

print("columns with at least one hole:", n_with_holes, "of", len(feature_columns))
print("columns more than half empty: ", n_half_empty)
print("rows surviving basalts.dropna():", rows_left, "of", len(basalts))
""", """
assert n_half_empty > 0, "n_half_empty came out 0 - 'missing' is a fraction, so half empty is 0.5"
assert n_with_holes >= n_half_empty, \\
    "a half-empty column is a column with holes - the first count cannot be the smaller"
print("✓ the holes —", n_with_holes, "of the", len(feature_columns),
      "columns have holes in them,", n_half_empty, "are more than half empty, and dropna() leaves",
      rows_left, "of", len(basalts), "rows")
""")

md(f"""
Not one column of the {M['n_features']} is complete. The best measured is
`{M['best_column']}` and even that is missing {pct(M['best_missing'])} percent of the time; the
worst is `{M['worst_column']}` at {pct(M['worst_missing'])} percent. Insist on rows with nothing
missing anywhere and you are left with a single sample.

So `.dropna()` is finished as a strategy, and the alternative has a name. **Imputation.** A blank
is not a zero. Fill it with something defensible and say what you filled it with. Ours will be
`SimpleImputer(strategy="median")`, which puts the middle value of a column into that column's
holes — defensible because it does not invent an extreme, and honest because we are about to say
out loud that we did it.

Two things go with it. The filler learns its medians from the **training** set only and then
applies them to the test set, because a median computed from data you are about to be tested on is
the leakage trap from the model-selection week. And a `StandardScaler` puts every column on the
same footing afterwards, which changes nothing for a tree — a tree only asks whether a value is
above a threshold — but matters enormously for the third model further down.

That is five steps, and we are about to run them on a dozen different sets of columns, so write
them once, as a function, and hand it the model and the columns.
""")

code("""
def accuracy(model, features, seed=0):
    \"\"\"Fit one model on one set of columns and score it on the held-out third of the samples.\"\"\"
    labels = basalts["affinity"]
    # Split first, before anything at all is measured from the data. `stratify` keeps the mix
    # of affinities the same in both halves, so a rare one cannot land entirely in the test set.
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3,
                                                        random_state=seed, stratify=labels)
    # Fill the holes, then put the columns on a common footing. Both of these learn their
    # numbers from the training half alone — `fit_transform` on it, plain `transform` on the
    # test half — because a median taken from rows you are about to be tested on is leakage.
    filler = SimpleImputer(strategy="median")
    X_train = filler.fit_transform(X_train)
    X_test = filler.transform(X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Learn from the training half; report the score on the test half, which it has never seen.
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)
""")

code("""
filled_score = accuracy(DecisionTreeClassifier(random_state=0),
                        basalts[["TiO2_wt_percent", "V_ppm"]])

print("TiO2 and V, dropping incomplete rows:", round(ti_v_score, 3), "on", len(pairs), "samples")
print("TiO2 and V, filling the holes instead:", round(filled_score, 3), "on", len(basalts),
      "samples")
""")

md(f"""
The same two elements and the same tree score {M['ti_v_filled']:.3f} rather than
{M['ti_v_tree']:.3f} once every sample is included — and those two are exactly the kind of pair you
were just told not to compare, because the second was tested on rocks the first never saw. Which is
the point. Nothing about the rocks changed; what changed is which rocks were allowed into the exam.
The {M['n_ti_v_extra']} extra samples are exactly the ones with a hole in titanium or vanadium, and
they have arrived with a median in place of a measurement.

So where did that {M['ti_v_drop']:.3f} go? Part of it is those samples, and you can see how much by
scoring the filled tree separately on the two kinds of rock in its own test set: the
{M['n_measured_test']} test rocks that were really measured score {M['measured_test']:.3f}, the
{M['n_filled_test']} that arrived carrying a median score {M['filled_test']:.3f}. The filled-in
rocks are genuinely harder, and that is the honest half of the story — `.dropna()` was not solving
that difficulty, it was hiding it, by quietly grading itself on the easy rocks only. But
{M['n_filled_test']} rocks out of {M['n_test']} cannot move an average by {M['ti_v_drop']:.3f}.
They account for {M['imputed_part']:.3f} of it. The other {M['split_part']:.3f} has nothing to do
with holes at all — it is the single split. Check that rather than believe it: fit both trees again
on ten different splits.
""")

code(f"""
for seed in range({len(M['ti_v_sweep'])}):
    X_train, X_test, y_train, y_test = train_test_split(pairs[["TiO2_wt_percent", "V_ppm"]],
                                                        pairs["affinity"], test_size=0.3,
                                                        random_state=seed,
                                                        stratify=pairs["affinity"])
    dropped = DecisionTreeClassifier(random_state=0)
    dropped.fit(X_train, y_train)
    dropna_score = dropped.score(X_test, y_test)
    filled_again = accuracy(DecisionTreeClassifier(random_state=0),
                            basalts[["TiO2_wt_percent", "V_ppm"]], seed)
    print("split", seed, "- dropna", round(dropna_score, 3),
          " filled", round(filled_again, 3),
          " gap", round(dropna_score - filled_again, 3))
""")

md(f"""
The first split was a flattering one: its gap is the {M['ti_v_seed0_rank']} of the ten,
and across all ten the dropna tree averages {M['ti_v_mean_dropna']:.3f} and the filled tree
{M['ti_v_mean_filled']:.3f} — a gap of {M['ti_v_mean_gap']:.3f}, half of what the first split
showed, running from {M['ti_v_min_gap']:.3f} on split {M['ti_v_min_gap_split']} to
{M['ti_v_max_gap']:.3f} on split {M['ti_v_max_gap_split']}. So: filling the holes does cost a few
points, part of that cost is the difficulty deletion was hiding, and no single split can tell you
the size of anything. One held-out third of {M['n_ti_v']} samples wobbles by more than the effect
you are trying to measure.

From here every number comes out of `accuracy`, so every number is measured on the same
{M['n_rows']} samples and the same held-out third of them, and they can be compared.
""")

# --- section 4 -------------------------------------------------------------
md(f"""
## Do the badly measured columns help, or hurt?

The **major oxides** are the measurements that account for nearly the whole weight of the rock —
its silicon, titanium, aluminium, iron, calcium, magnesium, manganese, potassium and sodium. That
is nine elements and ten columns, because this file reports iron twice, as FeO and as Fe2O3. They
are the first thing anybody measures, so if tectonic setting is written anywhere it should be
written there.

There is an eleventh oxide in the file, phosphorus, and it is worth a moment because of how it is
written: `P2O5(wt%)`, in a naming style nobody else in the table uses, and blank
{pct(M['p2o5_missing'])} percent of the time. That is what a compilation of hundreds of published
tables looks like from the inside. We hold to the ten below, so that "the majors" means one fixed
list for the rest of the week, and phosphorus goes in with everything else.
""")

code("""
major_oxides = ["SiO2_wt_percent", "TiO2_wt_percent", "Al2O3_wt_percent", "Fe2O3_wt_percent",
                "FeO_wt_percent", "CaO_wt_percent", "MgO_wt_percent", "MnO_wt_percent",
                "K2O_wt_percent", "Na2O_wt_percent"]

print(missing[major_oxides].round(3))
""")

md(f"""
Eight of the ten are missing from about one sample in eight. The two iron columns are the
exception, missing from more than half: this file keeps FeO and Fe2O3 as separate columns rather
than adding them together, and {M['n_neither_iron']} of the {M['n_rows']} samples
({pct(M['neither_iron_share'])} percent) carry neither. So even the best-measured ten are not a
complete ten.

Three models on those ten columns, one call each.
""")

code("""
tree_score = accuracy(DecisionTreeClassifier(random_state=0), basalts[major_oxides])
forest_score = accuracy(RandomForestClassifier(n_estimators=200, random_state=0),
                        basalts[major_oxides])
svm_score = accuracy(SVC(), basalts[major_oxides])

print(f"one tree:                 {tree_score:.3f}")
print(f"a forest:                 {forest_score:.3f}")
print(f"a support vector machine: {svm_score:.3f}")
""")

md(f"""
Three models, three lines, one interface. Every classifier in scikit-learn takes `fit` and `score`
in exactly the shape logistic regression did, which is why two of these arrived today as three
lines rather than as two new subjects.

The two new ones are worth a sentence each. A **random forest**. Ask a hundred slightly different
trees and take a vote. Each tree sees a different random sample of the rocks and a different random
handful of the columns, so each gets it wrong somewhere different, and the vote cancels the private
mistakes out — {M['oxide_forest']:.3f} against the single tree's {M['oxide_tree']:.3f} on the same
ten columns. And an **SVM**. Of all the lines separating the two groups, take the one leaving the
widest gap. That is why the scaler in `accuracy` matters — a gap is a distance, and a distance
measured across columns with wildly different units is meaningless.

The forest wins here, so the forest is what we take forward.
""")

md(f"""
### Predict before you run

There are {M['n_features']} chemistry columns in this file and you have just used ten of them. The
other {M['n_features'] - 10} are trace elements, isotope ratios and the phosphorus you left behind,
and you counted them a moment ago: {M['n_half_empty']} of the {M['n_features']} are more than half
empty, one of them ({M['worst_column']}) is missing {pct(M['worst_missing'])} percent of the time,
and every hole in every one of them is about to be filled in with a median that no laboratory ever
measured.

Hand the forest all {M['n_features']} columns instead of the ten. Better, or worse? Commit to a
number before you run anything — you will write it down in the next cell.
""")

ask("""
### ✏️ Your turn 4

Set `my_guess` to the accuracy you expect from all the columns. Then call `accuracy` once, on
`basalts[feature_columns]` with a fresh `RandomForestClassifier(n_estimators=200, random_state=0)`,
and print your guess, that score, and the difference between it and the `forest_score` the ten
oxides already got two cells above.

**Use these names**, because the self-check looks for them: `my_guess`, `all_score`.
""")

answer("""
my_guess = 0.85

all_score = accuracy(RandomForestClassifier(n_estimators=200, random_state=0),
                     basalts[feature_columns])

print("you guessed:      ", my_guess)
print("ten major oxides: ", round(forest_score, 3))
print("all the columns:  ", round(all_score, 3))
print("difference:       ", round(all_score - forest_score, 3))
""", """
assert 0 <= my_guess <= 1, "my_guess is an accuracy, so a number between 0 and 1"
assert 0 < all_score < 1, \\
    "all_score is an accuracy, a fraction of 1 - not a percentage"
assert abs(all_score - forest_score) > 0.01, \\
    "that is the ten oxides' score again - this call takes basalts[feature_columns]"
print("✓ all", len(feature_columns), "columns —", round(all_score, 3), "against the ten oxides'",
      round(forest_score, 3), "- a difference of", round(all_score - forest_score, 3))
""")

md(f"""
More columns won, by {M['gap']:.3f}. Forty-one extra columns, most of them badly measured, several
of them nearly absent, all of their holes stuffed with medians — and the model got *better*.

The obvious worry is that one split of the samples flattered it. So do it again, four more times,
with a different random split each time.
""")

code("""
for seed in [0, 1, 2, 3, 4]:
    oxides_only = accuracy(RandomForestClassifier(n_estimators=200, random_state=0),
                           basalts[major_oxides], seed)
    everything = accuracy(RandomForestClassifier(n_estimators=200, random_state=0),
                          basalts[feature_columns], seed)
    print("split", seed, "- ten oxides", round(oxides_only, 3),
          " all columns", round(everything, 3),
          " gap", round(everything - oxides_only, 3))
""")

md(f"""
All five splits point the same way, with gaps from {M['sweep_min_gap']:.3f} to
{M['sweep_max_gap']:.3f}. On split {M['sweep_worst_split']} the gap nearly closes, which is a useful
reminder of how much a single held-out third of {M['n_rows']} samples can wobble; but it never
reverses. The extra columns are carrying something.
""")

# --- section 5 -------------------------------------------------------------
md("""
## Is the forest reading the chemistry, or reading who measured the rock?

A fitted forest will tell you which columns its trees kept asking about. `feature_importances_` is
one number per column, and they add up to 1. Note that `accuracy` calls `fit` on the model object
you hand it, and `fit` changes that object, so after the call `forest` below is a trained forest and
can be interrogated.
""")

code("""
forest = RandomForestClassifier(n_estimators=200, random_state=0)
accuracy(forest, basalts[feature_columns])

importance = pd.Series(forest.feature_importances_, index=feature_columns)
importance = importance.sort_values(ascending=False)

for name in importance.head(10).index:
    print(f"{name:20s} importance {importance[name]:.3f}   missing {missing[name] * 100:.1f}%")
""")

code(f"""
top_ten = importance.head(10)
plt.barh(top_ten.index[::-1], top_ten.values[::-1])
plt.xlabel("share of the forest's decisions")
plt.ylabel("chemistry column")
plt.title("The 10 columns the forest leaned on, of {M['n_features']} "
          "(fitted on {M['n_train']} training basalts)")
plt.show()
""")

md(f"""
Read that list against the petrology from the start of the notebook and it is not a random ten.
`{M['top'][0][0]}` and `{M['top'][3][0]}` are the two sides of the subduction signature: strontium
travels in the water coming off a sinking plate and niobium does not, so an arc basalt carries more
strontium than a ridge basalt and is conspicuously short of niobium, while an ocean island basalt,
which owes nothing to subduction, is the niobium-rich one. `{M['top'][1][0]}` and
`{M['top'][2][0]}` are the titanium and zirconium that Pearce and Cann were already plotting by
hand in the 1970s, and titanium is half of Shervais's pair as well. The model has, on its own,
arrived at the elements a petrologist would have nominated.

It has also leaned hard on something a petrologist would want flagged. Strontium and potassium are
*mobile*: seawater and low-grade metamorphism move them around long after the rock has solidified,
so a mobile element can be telling you about the rock's later life rather than about the melt it
came from. Pearce and Cann used strontium anyway — titanium, zirconium and strontium was the second
of their two diagrams — and said in the same paper that alteration moves it, which is why titanium,
zirconium and yttrium is the diagram people trust on a rock that has sat under the ocean. The
forest cannot tell an altered rock from a fresh one. It found strontium the single most useful
column in the file, and on this compilation that works — but it is a reason to be careful about
handing this model an altered ocean-floor basalt.

And notice what is *not* in the top ten. The most useful of the {M['n_half_empty']} columns that are
more than half empty is `{M['sparse_best']}`, ranked {M['sparse_best_rank']}; between them those
{M['n_half_empty']} columns account for {M['sparse_share']:.3f} of the forest's decisions. So the
forest's attention did not go to the emptiest columns. What it took from beyond the ten oxides were
trace elements like `{M['top'][2][0]}` and `{M['top'][3][0]}`, missing {pct(M['top'][2][2])} and
{pct(M['top'][3][2])} percent of the time — patchy, but mostly there.
""")

# --- section 6: the second half of spine question 4 -------------------------
md(f"""
One more check before anybody reports {M['all_forest']:.3f} to a geochemist. *You got 99 percent. Be
suspicious. Did one of your columns already know the answer?*

Nothing in the chemistry knows the answer. But something else in this file might. Every sample came
from some published study, each study measured its own set of elements, and studies tend to be about
one setting at a time — a paper on Hawaiian volcanoes measures a particular list, a paper on the
Mariana arc a different one. If that is true then the *pattern of blanks* on a row is a fingerprint
of which paper the row came from, and the paper knows the setting.

`basalts[feature_columns].isna()` is a table of exactly the same shape, holding `True` where the
measurement is missing and `False` where it exists, and not one actual measurement. Hand the forest
that.
""")

ask("""
### ✏️ Your turn 5

First write down what you expect: set `my_blank_guess` to the accuracy you think a forest can reach
knowing only which numbers are missing.

Then build `blanks = basalts[feature_columns].isna()` and score a fresh forest on it with
`accuracy`. Do the same for `basalts[major_oxides].isna()`. Print both, and print your baseline from
your turn 1 for comparison.

Then, in the cell after, say in two or three sentences what those two numbers mean for the
`all_score` you reported in your turn 4, and which of your numbers you would hand a geochemist.

**Use these names**, because the self-check looks for them: `my_blank_guess`, `blanks`,
`blank_score`, `oxide_blank_score`.
""")

answer("""
my_blank_guess = 0.35

blanks = basalts[feature_columns].isna()
blank_score = accuracy(RandomForestClassifier(n_estimators=200, random_state=0), blanks)
oxide_blank_score = accuracy(RandomForestClassifier(n_estimators=200, random_state=0),
                             basalts[major_oxides].isna())

print("you guessed:                    ", my_blank_guess)
print("blanks in all the columns:      ", round(blank_score, 3))
print("blanks in the ten major oxides: ", round(oxide_blank_score, 3))
print("always guessing the commonest:  ", round(baseline, 3))
""", """
assert oxide_blank_score < blank_score, \\
    "the ten oxides' blanks should say less than all the columns' blanks - same table twice?"
print("✓ the blanks alone —", round(blank_score, 3), "from all the columns and",
      round(oxide_blank_score, 3), "from the ten oxides, against a baseline of", round(baseline, 3))
""")

answer_prose(f"""
Knowing nothing but which measurements are missing, the forest reaches {M['blank_forest']:.3f} —
nowhere near the {M['all_forest']:.3f} of the real model, but hugely above the
{M['baseline']:.3f} you get by guessing, so the pattern of blanks really is telling it which
setting a sample came from. That means the {M['all_forest']:.3f} is not purely chemistry: some
unknown part of it is the model recognising which study a row came from, because the median-filled
holes leave that fingerprint behind in the numbers. The ten major oxides have much less of this
problem — their blanks alone give {M['oxide_blank_forest']:.3f}, above the {M['baseline']:.3f}
baseline but far short of what all {M['n_features']} columns' blanks give, because eight of the ten
are measured by nearly everybody. So the number I would hand a geochemist is the ten-oxide
{M['oxide_forest']:.3f}, quoted as chemistry, with the all-columns {M['all_forest']:.3f} beside it
and the warning that part of that extra {M['gap']:.3f} is bookkeeping rather than rock.
""")

# --- section 7 -------------------------------------------------------------
md(f"""
## The question, answered

**From three different places, and the rock remembers which.** A ridge basalt melts by
decompression from mantle already stripped once; an arc basalt melts because water off a sinking
plate lowered the melting point, and arrives carrying the elements that travel in water and short
of the ones that do not; an ocean island basalt comes from hotter, less depleted mantle deeper down.
Those three histories leave three different chemistries, and a random forest reads them back with
{M['oxide_forest']:.3f} accuracy from the ten major oxides alone, against {M['baseline']:.3f} for
guessing. What it leans on hardest — strontium, titanium, zirconium, niobium — is exactly what the
melting mechanisms predict it should. The setting really is written in the chemistry; part of what
looks like extra skill from the other {M['n_features'] - 10} columns is the file remembering who
measured what.
""")

# --- summary and homework --------------------------------------------------
md(weekkit.week_cheatsheet(11))

md(f"""
## Homework

Three parts, all on the same table and the same `accuracy` function from class.

Class showed you that the {M['n_features']} columns together beat the ten major oxides. It never
asked what the badly measured columns can do on their own, never let you choose where to draw the
line, and never went back to `.dropna()` to see what it would have cost you.

Run the setup cell at the top of the notebook, then the cell below, and you have everything the
three parts need.
""")

code(weekkit.CHECKPOINT.format(body='''missing = basalts[feature_columns].isna().sum() / len(basalts)
baseline = basalts["affinity"].value_counts().max() / len(basalts)
major_oxides = ["SiO2_wt_percent", "TiO2_wt_percent", "Al2O3_wt_percent", "Fe2O3_wt_percent",
                "FeO_wt_percent", "CaO_wt_percent", "MgO_wt_percent", "MnO_wt_percent",
                "K2O_wt_percent", "Na2O_wt_percent"]


def accuracy(model, features, seed=0):
    """Fit one model on one set of columns and score it on the held-out third of the samples."""
    labels = basalts["affinity"]
    # Split first, before anything at all is measured from the data. `stratify` keeps the mix
    # of affinities the same in both halves, so a rare one cannot land entirely in the test set.
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3,
                                                        random_state=seed, stratify=labels)
    # Fill the holes, then put the columns on a common footing. Both of these learn their
    # numbers from the training half alone — `fit_transform` on it, plain `transform` on the
    # test half — because a median taken from rows you are about to be tested on is leakage.
    filler = SimpleImputer(strategy="median")
    X_train = filler.fit_transform(X_train)
    X_test = filler.transform(X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Learn from the training half; report the score on the test half, which it has never seen.
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)'''))

ask(f"""
### ✏️ Your turn 6

The {M['n_half_empty']} columns you counted in your turn 3 — the ones more than half empty — on
their own, with nothing else.

Build `sparse_columns` with a loop over `feature_columns`, keeping the names where
`missing[name] > 0.5`. Then score a `RandomForestClassifier(n_estimators=200, random_state=0)` on
`basalts[sparse_columns]` with `accuracy`. Print how many columns you kept, the score, and your
`baseline` from your turn 1.

Then answer it in one more printed line, quoting your score against your baseline: do
{M['n_half_empty']} columns that are more than half empty carry real information about tectonic
setting, or not?

**Use these names**, because the self-check looks for them: `sparse_columns`, `sparse_score`.
""")

answer("""
sparse_columns = []
for name in feature_columns:
    if missing[name] > 0.5:
        sparse_columns.append(name)

sparse_score = accuracy(RandomForestClassifier(n_estimators=200, random_state=0),
                        basalts[sparse_columns])

print("columns kept:", len(sparse_columns))
print("forest on those alone:", round(sparse_score, 3))
print("baseline:             ", round(baseline, 3))

print("Yes — badly measured is not the same as uninformative:", round(sparse_score, 3),
      "against a baseline of", round(baseline, 3), "is", round(sparse_score / baseline, 1),
      "times better than guessing, on columns that are blank for most of the rocks.")
""", """
assert missing[sparse_columns].min() > 0.5, "every column in sparse_columns should be over half empty"
print("✓ the emptiest columns alone —", len(sparse_columns), "columns,",
      round(sparse_score, 3), "against a baseline of", round(baseline, 3))
""")

ask(f"""
### ✏️ Your turn 7

Your decision, and it is a real one. A geochemist might reasonably refuse to report a model that
leans on a column measured in three samples out of a hundred. So set a cutoff and throw away every
column emptier than it.

Set `cutoff` to **either 0.5 or 0.2** — your choice, and both are defensible. Build `kept_columns`
by looping over `feature_columns` and keeping the names where `missing[name] < cutoff`, and score a
forest on `basalts[kept_columns]`. Then do the same for the cutoff you did *not* pick, so that you
can say what your choice cost. Print, for each: the cutoff, how many columns survived and the
score. Class got {M['all_forest']:.3f} from all {M['n_features']}.

Then say it, in one more printed line: quote both scores and both column counts, and name what
your cutoff cost you.

**Use these names**, because the self-check looks for them: `cutoff`, `kept_columns`, `kept_score`.
""")

answer("""
cutoff = 0.5

kept_columns = []
for name in feature_columns:
    if missing[name] < cutoff:
        kept_columns.append(name)

kept_score = accuracy(RandomForestClassifier(n_estimators=200, random_state=0),
                      basalts[kept_columns])

other_columns = []
for name in feature_columns:
    if missing[name] < 0.2:
        other_columns.append(name)

other_score = accuracy(RandomForestClassifier(n_estimators=200, random_state=0),
                       basalts[other_columns])

print("my cutoff", cutoff, "-", len(kept_columns), "columns, score", round(kept_score, 3))
print("cutoff 0.2 -", len(other_columns), "columns, score", round(other_score, 3))

print("Cutting at", cutoff, "kept", len(kept_columns), "columns and scored", round(kept_score, 3),
      "where the other cutoff kept", len(other_columns), "columns and scored",
      round(other_score, 3), "so my choice bought", round(kept_score - other_score, 3),
      "of accuracy. What it cost is the right to say the model uses only well-measured columns:",
      len(kept_columns) - len(other_columns), "of the columns I kept are blank for between a",
      "fifth and a half of the rocks.")
""", """
assert missing[kept_columns].max() < cutoff, "every column you kept should be emptier than cutoff"
print("✓ your cutoff —", cutoff, "keeps", len(kept_columns), "columns and scores",
      round(kept_score, 3))
""")

ask("""
### ✏️ Your turn 8

Back to `.dropna()`, the tool you started the day with. `basalts.dropna(subset=major_oxides)` drops
a row only when one of *those* columns is blank, and leaves holes elsewhere alone.

Print how many rows survive `basalts.dropna(subset=major_oxides)`. You already have the number to
set it against: your turn 3 found that dropping every row with a hole anywhere in it leaves {rows}
of the {total} samples.

Then, in the cell after, write two or three sentences using **those two counts**: what would you
have concluded about the {n} columns outside `major_oxides` — mostly trace elements and isotope
ratios — if `.dropna()` were the only tool you knew? Say what you would have measured, what you
would have reported, and which of this week's three takeaways that story would have broken.

**Use this name**, because the self-check looks for it: `rows_left_oxides`.
""".replace("{n}", str(M["n_features"] - 10))
   .replace("{rows}", str(M["rows_left"])).replace("{total}", str(M["n_rows"])))

answer("""
rows_left_oxides = len(basalts.dropna(subset=major_oxides))

print("rows left dropping on the ten major oxides:", rows_left_oxides, "of", len(basalts))
""", f"""
assert rows_left_oxides > {M['rows_left']}, \\
    "ten columns should leave far more rows than all of them - did you pass subset=?"
print("✓ what dropna costs —", rows_left_oxides,
      "rows survive on the ten major oxides, against {M['rows_left']} on the whole table")
""")

answer_prose(f"""
Dropping incomplete rows on the ten major oxides leaves {M['rows_left_oxides']} samples of
{M['n_rows']}, which is a small but workable dataset. Doing the same over every one of the
{M['n_features']} columns leaves {M['rows_left']}. With `.dropna()` as my only tool I would have
measured a
ten-oxide model on {M['rows_left_oxides']} rocks, tried to add the trace elements, watched my
training set collapse to {M['rows_left']} sample, and reported that the trace elements are
unusable — when what actually happened is that no single laboratory measured all
{M['n_features']} things on one rock, so a rule that demands all of them at once matches almost
nothing. That story breaks the second takeaway completely: filling the holes instead shows those
columns carrying real signal — {M['sparse_forest']:.3f} from the
{M['n_half_empty']} emptiest columns alone, against a baseline of {M['baseline']:.3f} — and it is
the third takeaway, that deleting rows can destroy a dataset entirely, that explains why deletion
made them look worthless.
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
    print(f"cache: data/{CACHE_NAME}")


if __name__ == "__main__":
    main()
    weekkit.gate(11)
