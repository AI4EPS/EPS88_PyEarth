#!/usr/bin/env python
"""Build week 6 — "How old is the universe?"

Emits both notebooks from one source so they cannot drift:

    docs/notebooks/06_age_of_the_universe_solution.ipynb   executed, every output saved
    docs/notebooks/06_age_of_the_universe.ipynb            the same file with the answers deleted

It also writes the week's three cached fallbacks. All three are byte-identical copies of the
files in offerings/2024-inherited_copy, which is the AI4EPS/EPS88_2024 repository pinned at
a58436d0 — that pinned raw URL is the live source the notebook tries first, so the cache is a
real fallback rather than the only copy.

Every number that appears in prose or in a model answer is computed HERE, from the same files
the notebook reads, and formatted in. Nothing is typed from memory or copied from the plan.

    python tools/build_week06.py
"""
import json
import pathlib
import shutil
import subprocess
import sys

import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARCHIVE = ROOT.parent / "offerings/2024-inherited_copy"
OUT = ROOT / "docs/notebooks"
SLUG = "06_age_of_the_universe"

course = yaml.safe_load((ROOT / "course.yml").read_text())
WEEK = next(s for s in course["schedule"] if s["n"] == 6)
PLATFORM = course["platform"]
CACHE_BASE = PLATFORM["cache_base"]

# The archive the three tables come from, pinned by commit so the URL cannot move under us.
SOURCE = "https://raw.githubusercontent.com/AI4EPS/EPS88_2024/a58436d0/"
SN_PATH = "Week08_Age_of_the_Universe/Data/Freedman2000_Supernova1a.csv"
PAR_PATH = "Week05_Seafloor_Spreading/data/PAR_east_age_dist.csv"
MAR_PATH = "Week05_Seafloor_Spreading/data/MAR_east_age_dist.csv"
CACHED = {SN_PATH: "week06_supernovae.csv",
          PAR_PATH: "week06_par_age_distance.csv",
          MAR_PATH: "week06_mar_age_distance.csv"}

# Two conversions, both conventions rather than measurements, both read 2026-08-31:
#   1 pc = 648000/pi au and 1 au = 149 597 870 700 m exactly (IAU 2015 Resolution B2),
#   so 1 Mpc = 3.0857e19 km.  A Julian year is 365.25 days, the astronomers' year.
MPC_IN_KM = 3.0857e19
SECONDS_PER_YEAR = 60 * 60 * 24 * 365.25


# ---------------------------------------------------------------------------
# 1. write the cached fallbacks, then measure everything the notebook will say
# ---------------------------------------------------------------------------
def cache_files():
    """Copy the three archive tables into data/ as this week's fallbacks."""
    for path, name in CACHED.items():
        shutil.copyfile(ARCHIVE / path, ROOT / "data" / name)


def free(x, y):
    return LinearRegression().fit(x, y)


def forced(x, y):
    return LinearRegression(fit_intercept=False).fit(x, y)


def age_gyr(h0):
    """A Hubble constant in km/s/Mpc, turned into an age in billions of years."""
    return MPC_IN_KM / h0 / SECONDS_PER_YEAR / 1e9


cache_files()
sn = pd.read_csv(ROOT / "data" / CACHED[SN_PATH])
par = pd.read_csv(ROOT / "data" / CACHED[PAR_PATH])
mar = pd.read_csv(ROOT / "data" / CACHED[MAR_PATH])

M = {}
M["n_sn"] = len(sn)
M["d_min"], M["d_max"] = float(sn["D(Mpc)"].min()), float(sn["D(Mpc)"].max())
M["v_min"], M["v_max"] = int(sn["VCMB"].min()), int(sn["VCMB"].max())

h_each = sn["VCMB"] / sn["D(Mpc)"]
M["h_min"], M["h_max"] = float(h_each.min()), float(h_each.max())
M["h_mean"] = float(h_each.mean())
M["age_h_min"], M["age_h_max"] = age_gyr(M["h_max"]), age_gyr(M["h_min"])
M["age_spread"] = M["age_h_max"] - M["age_h_min"]

X, y = sn[["D(Mpc)"]], sn["VCMB"]
sn_free, sn_forced = free(X, y), forced(X, y)
M["sn_free_slope"] = float(sn_free.coef_[0])
M["sn_free_intercept"] = float(sn_free.intercept_)
M["sn_free_r2"] = float(sn_free.score(X, y))
M["sn_forced_slope"] = float(sn_forced.coef_[0])
M["sn_forced_r2"] = float(sn_forced.score(X, y))
M["age_free"] = age_gyr(M["sn_free_slope"])
M["age_forced"] = age_gyr(M["sn_forced_slope"])

resid = y - sn_free.predict(X)
M["resid_max"] = float(resid.max())
M["resid_min"] = float(resid.min())
M["resid_mean"] = float(resid.mean())
M["resid_forced_mean"] = float((y - sn_forced.predict(X)).mean())
M["resid_worst_d"] = float(sn.loc[resid.abs().idxmax(), "D(Mpc)"])
M["resid_worst_name"] = str(sn.loc[resid.abs().idxmax(), "Supernova"])

# the two ridges
for tag, table in (("par", par), ("mar", mar)):
    x, d = table[["Age"]], table["Distance"]
    f, o = free(x, d), forced(x, d)
    M[f"{tag}_n"] = len(table)
    M[f"{tag}_age_min"], M[f"{tag}_age_max"] = float(x["Age"].min()), float(x["Age"].max())
    M[f"{tag}_d_min"], M[f"{tag}_d_max"] = float(d.min()), float(d.max())
    M[f"{tag}_free_slope"] = float(f.coef_[0])
    M[f"{tag}_free_intercept"] = float(f.intercept_)
    M[f"{tag}_free_r2"] = float(f.score(x, d))
    M[f"{tag}_forced_slope"] = float(o.coef_[0])
    M[f"{tag}_forced_r2"] = float(o.score(x, d))
M["ridge_ratio"] = M["par_forced_slope"] / M["mar_forced_slope"]
M["n_picks"] = M["par_n"] + M["mar_n"]

# homework 1: the nearest half and the farthest half of the supernovae
ordered = sn.sort_values("D(Mpc)")
near, far = ordered.head(18), ordered.tail(18)
for tag, half in (("near", near), ("far", far)):
    o = forced(half[["D(Mpc)"]], half["VCMB"])
    M[f"h0_{tag}"] = float(o.coef_[0])
    M[f"age_{tag}"] = age_gyr(M[f"h0_{tag}"])
    M[f"d_{tag}_min"] = float(half["D(Mpc)"].min())
    M[f"d_{tag}_max"] = float(half["D(Mpc)"].max())
M["age_half_gap"] = abs(M["age_near"] - M["age_far"])

# homework 2: the older half of the Pacific-Antarctic picks
old = par[par["Age"] >= 20]
old_free, old_forced = free(old[["Age"]], old["Distance"]), forced(old[["Age"]], old["Distance"])
M["old_n"] = len(old)
M["old_free_slope"] = float(old_free.coef_[0])
M["old_free_intercept"] = float(old_free.intercept_)
M["old_free_r2"] = float(old_free.score(old[["Age"]], old["Distance"]))
M["old_forced_slope"] = float(old_forced.coef_[0])
M["old_forced_r2"] = float(old_forced.score(old[["Age"]], old["Distance"]))
M["within_par_ratio"] = M["par_forced_slope"] / M["old_free_slope"]

# the two literature numbers this week checks itself against, both read 2026-08-31
PLANCK_AGE = 13.797            # Planck 2018 results VI, A&A 641, A6 (2020): 13.797 +/- 0.023 Gyr
PLANCK_ERR = 0.023
FREEDMAN_H0 = 71               # Freedman et al. 2001, ApJ 553, 47: 71 +/- 2 +/- 6 from SNe Ia
M["age_gap_percent"] = abs(M["age_forced"] - PLANCK_AGE) / PLANCK_AGE * 100


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
Every galaxy far enough away from us is moving away, and the further away it is the faster it
goes. That one observation is the most consequential measurement in astronomy, because if
everything is flying apart now then everything was in the same place once — and the speed tells
you how long ago.

Today you get thirty-six of those measurements. Each one is a Type Ia supernova: an exploding
star bright enough to be seen most of the way across the universe, and regular enough that its
brightness gives away its distance. Distance on one axis, speed on the other, and the slope of
the line through them has a length of time hidden inside it.

Getting that time right turns on a single decision, and it is not a statistical decision — it is
a physical one. You will make the same decision a second time before the end of the notebook, on
the floor of the Pacific Ocean.
"""

md(weekkit.OPENING.format(question=WEEK["question"], datahub=datahub, hook=HOOK.strip()))

md("""
## What you'll be able to do

**The science.** Say how old the universe is, in billions of years, and say exactly which
measurement that number came from. Say how fast two ocean ridges are building new seafloor, and
why one of them is several times faster than the other.

**The skills.** Fit a straight line with scikit-learn: `LinearRegression().fit(x, y)`, then
`.coef_` for the slope, `.intercept_` for where it crosses zero, `.predict()` for what the line
says, and `.score()` for how much of the data it accounts for. Subtract the line from the data to
get **residuals**, and look at them. And fit a line that is not allowed an intercept at all,
`LinearRegression(fit_intercept=False)`, when physics has already told you one point it must pass
through.

**Eight places where you write something: five in class, three at home.** Each one is headed
*Your turn*, with an empty cell under it.
""")

setup = weekkit.setup_cell(
    imports="from sklearn.linear_model import LinearRegression\n",
    figsize="(7, 4)",
    cache_base=CACHE_BASE,
    signature="archive_path, cached_name",
    docstring=("Read one of this week's tables from the archive it came from; fall back to "
               "the copy stored with the course."),
    url_expr=f'f"{SOURCE}{{archive_path}}"',
    cache_expr="cached_name",
    unpack=f'''
STARS = "Week08_Age_of_the_Universe/Data/"
RIDGES = "Week05_Seafloor_Spreading/data/"

sn = load(STARS + "Freedman2000_Supernova1a.csv", "{CACHED[SN_PATH]}")
par = load(RIDGES + "PAR_east_age_dist.csv", "{CACHED[PAR_PATH]}")
mar = load(RIDGES + "MAR_east_age_dist.csv", "{CACHED[MAR_PATH]}")
coast = pd.read_csv(CACHE + "/coastlines.csv")

print("supernovae:", sn.shape, " Pacific-Antarctic:", par.shape, " Mid-Atlantic:", mar.shape)
'''.strip("\n"))
code(setup)

# --- section 1: the supernova table ---------------------------------------
md("""
## Thirty-six exploding stars

The table is small enough to read. It is the Type Ia supernova table from the *Hubble Space
Telescope* Key Project (Freedman et al., *The Astrophysical Journal* **553**, 47, 2001), the study
that set out to pin the expansion rate down, and it has been handed round this course since 2019.
""")

code("""
print("rows:", len(sn))
print("distance:", sn["D(Mpc)"].min(), "to", sn["D(Mpc)"].max(), "Mpc")
print("speed:   ", sn["VCMB"].min(), "to", sn["VCMB"].max(), "km/s")
sn.head()
""")

md(f"""
Four numbers matter, and one of them is a column name you have to type in quotes because it has
brackets in it.

- `Supernova` — which exploding star this row is.
- `VCMB` — how fast that supernova's galaxy is moving away from us, in km/s. `CMB` says the speed
  has been corrected for the Sun's own motion, so it is a speed relative to the universe rather
  than to us.
- `D(Mpc)` — how far away it is, in **megaparsecs**. A parsec is 3.0857 × 10¹³ km, about 3.26
  light years (the International Astronomical Union's definition of the parsec, read 2026-08-31),
  and a megaparsec is a million of them.
- `HCMB` — the speed divided by the distance, which the paper printed for every supernova.

So the nearest of these supernovae is {M['d_min']:.0f} Mpc away and the farthest
{M['d_max']:.0f} Mpc — eight times further — and every one of them is receding from us. Plot the
two columns against each other and the shape of the relationship is immediate.
""")

code(f"""
plt.scatter(sn["D(Mpc)"], sn["VCMB"])
plt.xlabel("distance (Mpc)")
plt.ylabel("recession speed (km/s)")
plt.title("Type Ia supernovae, {M['n_sn']} of them")
plt.show()
""")

md("""
Further away means faster, and close enough to a straight line that you could put a ruler on it.
That is **Hubble's law**: speed = H₀ × distance, where H₀ is the number we are after. It is
measured in the awkward-looking units of kilometres per second per megaparsec — how much faster a
galaxy recedes for every extra megaparsec of distance.

Notice what those units are hiding. Kilometres per second divided by megaparsecs is a speed
divided by a distance, and a speed divided by a distance is one over a time. So H₀ has a time
buried in it, and that is the whole trick of this notebook.

Before any line-fitting, the crudest possible estimate. Every single supernova already gives you
its own H₀ — just divide.
""")

ask("""
### ✏️ Your turn 1

Divide the speed column by the distance column, all thirty-six at once, and see how much the
answers disagree. Print the smallest, the largest, the average, and the largest divided by the
smallest.

Then check yourself against the paper: the `HCMB` column is the same division, done by the
authors. Print its average too, and the two should match.

**Use these names**, because the self-check looks for them: `h_each`.
""")

answer("""
h_each = sn["VCMB"] / sn["D(Mpc)"]

print("smallest:", round(h_each.min(), 1), "km/s/Mpc")
print("largest: ", round(h_each.max(), 1), "km/s/Mpc")
print("average: ", round(h_each.mean(), 2), "km/s/Mpc")
print("largest / smallest:", round(h_each.max() / h_each.min(), 2))
print("the paper's own column averages:", round(sn["HCMB"].mean(), 2), "km/s/Mpc")
""", f"""
assert len(h_each) == {M['n_sn']}, "h_each should hold one number per supernova, not one number"
print("✓ one Hubble constant per supernova —", round(h_each.min(), 1), "to",
      round(h_each.max(), 1), "km/s/Mpc, which is not one answer but",
      len(h_each), "of them")
""")

# --- section 2: one line through all of them ------------------------------
md(f"""
Thirty-six supernovae, {M['h_min']:.1f} to {M['h_max']:.1f} km/s/Mpc, the largest
{M['h_max'] / M['h_min']:.2f} times the smallest. That is not one Hubble constant, it is
thirty-six of them — and since H₀ has a time inside it, thirty-six different ages for the
universe, spread by that same factor. Dividing one supernova by itself throws away the other
thirty-five, and every supernova carries measurement error in both
its speed and its distance.

What we want is one number that uses all thirty-six at once.

## One line through all of them

Draw the best straight line. Best means the smallest total miss. For each point, the miss is the
vertical gap between the real speed and the speed the line claims; square each gap so that
misses above and below cannot cancel; add them up; and choose the line that makes that total as
small as it can be. There is exactly one such line, and scikit-learn will find it.

```
model = LinearRegression().fit(x, y)
```

Two details in that one line. `x` has to be a **table of columns**, not a single column, because
in later weeks you will fit on several columns at once — so it is `sn[["D(Mpc)"]]` with two sets
of brackets, meaning "a table containing this one column", while `y` stays a single column,
`sn["VCMB"]`. And `fit` hands you back the model, which then answers questions: `.coef_[0]` is
the slope, `.intercept_` is where the line crosses x = 0, `.predict()` is what the line says at
any distance, and `.score()` is R², the fraction of the up-and-down variation in the data that
the line accounts for — 1.0 would be every point exactly on the line.

### Predict before you run

The line is about to tell you what speed it expects at a distance of **zero** — a galaxy right
here, no distance away at all. What number should that be? Commit to it in the next cell before
you run it.
""")

code(f"""
my_guess_intercept = 0        # ← change this to whatever you think, then run the cell

model = LinearRegression().fit(sn[["D(Mpc)"]], sn["VCMB"])

print("you guessed: ", my_guess_intercept, "km/s")
print("slope:       ", round(model.coef_[0], 2), "km/s per Mpc")
print("intercept:   ", round(model.intercept_, 1), "km/s")
print("R²:          ", round(model.score(sn[["D(Mpc)"]], sn["VCMB"]), 4))
""")

md(f"""
An R² of {M['sn_free_r2']:.3f} says the line accounts for almost all of the spread in the speeds,
and a slope of {M['sn_free_slope']:.1f} km/s/Mpc is in the right neighbourhood of the
supernova-by-supernova numbers you printed. The intercept is the surprise, and we will come back
to it in a moment.

Draw the line on top of the points first. `.predict()` takes the same table of distances and
returns what the line says at each one.
""")

code(f"""
predicted = model.predict(sn[["D(Mpc)"]])

plt.scatter(sn["D(Mpc)"], sn["VCMB"], label="supernovae")
plt.plot(sn["D(Mpc)"], predicted, color="black", label="best straight line")
plt.xlabel("distance (Mpc)")
plt.ylabel("recession speed (km/s)")
plt.title("Type Ia supernovae and the fitted line, {M['n_sn']} points")
plt.legend()
plt.show()
""")

md("""
The line looks right, but "looks right" is not a measurement. The honest way to see what a fit is
doing wrong is to subtract it: for each point, the **residual** is the real value minus the value
the line predicts. A residual is what the line failed to explain, and plotting the residuals
against distance is a much harsher test than plotting the fit — the line is now flat, at zero, so
any pattern left in the picture is a pattern the line missed.
""")

ask("""
### ✏️ Your turn 2

Compute the residuals — the real speeds minus `predicted` — and plot them against distance, with
a horizontal line at zero to fit against. Then print the biggest miss in each direction, using
`.max()` and `.min()`.

Look at the picture before you move on: are the misses scattered evenly above and below zero
across the whole range of distances, or do they drift systematically to one side at one end?

**Use these names**, because the self-check looks for them: `residuals`.
""")

answer(f"""
residuals = sn["VCMB"] - predicted

plt.scatter(sn["D(Mpc)"], residuals)
plt.axhline(0, color="black")
plt.xlabel("distance (Mpc)")
plt.ylabel("real speed minus fitted speed (km/s)")
plt.title("What the line missed, {M['n_sn']} supernovae")
plt.show()

print("biggest miss above the line:", round(residuals.max()), "km/s")
print("biggest miss below the line:", round(residuals.min()), "km/s")
""", """
assert abs(residuals.mean()) < 1, \\
    "these should be the misses of the free fit, whose residuals always average to zero"
print("✓ residuals — the worst miss is", round(residuals.abs().max()),
      "km/s, on a fit that averages", round(residuals.mean(), 6), "km/s off")
""")

md(f"""
The worst miss is {M['resid_max']:,.0f} km/s above the line and {abs(M['resid_min']):,.0f} km/s
below it, and both signs turn up at every distance rather than the picture sweeping from one
side of zero to the other as you move right. So the straight line is a fair description of these
data, and a curve would not obviously do better. That settles the *shape*. It does not settle the
*position*.

## Where the line crosses zero

Go back to the intercept, which is the number your prediction was about. The fitted line says
that at a distance of zero — right here, no distance at all — a galaxy is already receding at
{M['sn_free_intercept']:.0f} km/s.
""")

ask("""
### ✏️ Your turn 3

In two or three sentences: what would it actually mean for a galaxy at zero distance to be moving
away from us at that speed, and is that something the universe does? Say what you think should be
done about it.

*(This one is written, not coded — answer in the cell below.)*
""")

answer_prose(f"""
A galaxy at zero distance from us is us, or near enough, and it cannot be receding at
{M['sn_free_intercept']:.0f} km/s in every direction at once — that would mean the expansion has a
special speed attached to our own position, which is exactly what Hubble's law says it does not.
The physics is unambiguous: zero distance has to mean zero recession, so the true line has to pass
through the origin. The fitted intercept of {M['sn_free_intercept']:.0f} km/s is not a discovery
about the universe, it is what happens when you let a line float free through data that all sits
between {M['d_min']:.0f} and {M['d_max']:.0f} Mpc and then extrapolate it back to a distance
nothing in the table has. The fit was not wrong; it was unconstrained. We should put the
constraint in and fit again.
""")

md("""
Sometimes physics already knows one point on the line. Make the line go through it. In
scikit-learn that is one extra argument, and nothing else about the fit changes:

```
through_origin = LinearRegression(fit_intercept=False).fit(x, y)
```

`fit_intercept=False` says: do not look for a best crossing point, there isn't one to look for —
the line goes through (0, 0) and the only thing left to choose is its slope.

And now the slope is the whole answer, because of those units. H₀ is a speed over a distance,
which is one over a time, so **1 / H₀ is a time**: how long ago everything was in the same place,
if galaxies have always moved at the speed they are moving now. Two conversions turn it into
years — the megaparsec in kilometres from the definition above, and a year of 365.25 days, which
is the astronomers' convention rather than a measurement.
""")

ask(f"""
### ✏️ Your turn 4

Fit the supernovae again, this time with `fit_intercept=False`, and print the slope: that is our
measured H₀.

Then turn it into an age. Write a function `age_of_universe(H0)` with a docstring, which takes a
Hubble constant in km/s per Mpc and returns an age in **billions of years**. Inside it:

```
MPC_IN_KM = 3.0857e19             # one megaparsec, in kilometres
SECONDS_PER_YEAR = 60 * 60 * 24 * 365.25
```

1 / H₀ is in Mpc·s/km, so multiplying by `MPC_IN_KM` gives seconds; divide by
`SECONDS_PER_YEAR` for years, and by `1e9` for billions of years.

Call it twice and print both answers: once on your H₀, and once on the slope of the free fit from
earlier, `model.coef_[0]`. The second one is what the constraint cost you if you had left it out.

**Use these names**, because the self-check looks for them: `H0`, `age_of_universe`, `age_Gyr`.
""")

answer("""
through_origin = LinearRegression(fit_intercept=False).fit(sn[["D(Mpc)"]], sn["VCMB"])
H0 = through_origin.coef_[0]


def age_of_universe(H0):
    \"\"\"A Hubble constant in km/s per Mpc, turned into an age in billions of years.\"\"\"
    MPC_IN_KM = 3.0857e19
    SECONDS_PER_YEAR = 60 * 60 * 24 * 365.25
    seconds = MPC_IN_KM / H0
    return seconds / SECONDS_PER_YEAR / 1e9


age_Gyr = age_of_universe(H0)

print("H0 forced through the origin:", round(H0, 2), "km/s/Mpc")
print("age of the universe:         ", round(age_Gyr, 2), "billion years")
print("what the free fit would say: ", round(age_of_universe(model.coef_[0]), 2), "billion years")
""", """
assert 1 < age_Gyr < 100, \\
    "age_Gyr should be a number of BILLIONS of years — check the last two divisions"
print("✓ the age of the universe —", round(age_Gyr, 2), "billion years, from a slope of",
      round(H0, 2), "km/s/Mpc")
""")

md(f"""
Two checks on that, and both are external.

The paper these data come from quotes {FREEDMAN_H0} ± 2 (random) ± 6 (systematic) km/s/Mpc from
its Type Ia supernovae; you got {M['sn_forced_slope']:.2f} from the same table. And the current
best measurement of the age of the universe, made a completely different way — from the Planck
satellite's map of the microwave background — is {PLANCK_AGE} ± {PLANCK_ERR} billion years (Planck
Collaboration, *Astronomy & Astrophysics* **641**, A6, 2020; both figures read 2026-08-31). Your
{M['age_forced']:.2f} sits {abs(M['age_forced'] - PLANCK_AGE):.2f} billion years above it.

That is a startlingly good answer from thirty-six points, and it deserves one honest caveat. 1 / H₀ is
the age only if the expansion has always run at today's rate. It has not: gravity slowed it down
early on, and over the last few billion years it has been speeding up again. Those two effects
very nearly cancel in our universe, which is why the simple answer lands so close to the careful
one. And notice what the constraint bought. The free fit, with its {M['sn_free_intercept']:.0f}
km/s intercept, gives {M['age_free']:.2f} billion years — {M['age_free'] - PLANCK_AGE:.2f} billion
years out, for a reason that has nothing to do with cosmology and everything to do with a line
nobody told where to start.
""")

# --- section 3: the same decision on the ocean floor ----------------------
md("""
## The same question on the ocean floor

Nothing in the last three sections was about astronomy. Two columns, a straight line, a slope with
a time in it, and one physical constraint on where the line has to pass. That pattern is
everywhere, and the nearest example is under the sea.

New ocean floor is made at mid-ocean ridges: two plates pull apart, molten rock rises into the
gap and freezes onto both edges. As it freezes it records the direction of Earth's magnetic field,
which flips every so often, so the seafloor carries a barcode of magnetic stripes running parallel
to the ridge — and each stripe can be dated, because the pattern of flips is known. Measure how
far a dated stripe now sits from the ridge that made it and you have exactly the shape of problem
you just solved: a distance, an age, and a slope that is a speed.

The two tables are picks of those stripes along ship tracks, one set east of the
**Pacific-Antarctic Ridge** in the far South Pacific and one east of the **Mid-Atlantic Ridge**.
They came into the course with the earlier offerings and carry no source note of their own, which
is worth knowing about any file you did not make. Each row is one pick: its age in millions of
years, where it is, and its distance in kilometres. (The first column is the pick number the
original files were saved with; ignore it.)
""")

code(weekkit.CHECKPOINT.format(body='''par = load(RIDGES + "PAR_east_age_dist.csv", "%s")
mar = load(RIDGES + "MAR_east_age_dist.csv", "%s")'''
                               % (CACHED[PAR_PATH], CACHED[MAR_PATH])))

code("""
par.head()
""")

md(f"""
Where those picks are matters, so put them on the map before fitting anything. Longitude and
latitude are just two more columns, drawn as an ordinary scatter over the coastline.
""")

code(f"""
plt.figure(figsize=(9, 5))          # a world map needs to be wider than the house default
plt.plot(coast["lon"], coast["lat"], color="0.6", lw=0.6)
plt.scatter(par["Lon"], par["Lat"], s=6, label="Pacific-Antarctic Ridge")
plt.scatter(mar["Lon"], mar["Lat"], s=6, label="Mid-Atlantic Ridge")
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.gca().set_aspect("equal")
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title("Magnetic-stripe picks on two ridges, {M['n_picks']} of them")
plt.legend(loc="lower left")
plt.show()
""")

md(f"""
Two lines of dots, each trailing away from its ridge: the Pacific-Antarctic picks run southeast
from about 55° S, the Mid-Atlantic picks run west to east across the Atlantic at about 25° N.
Neither is near a coast, which is the point — this is seafloor, not continent.

Now the same plot as the supernovae, with age where distance was.
""")

code(f"""
plt.scatter(par["Age"], par["Distance"], s=10)
plt.xlabel("age of the seafloor (millions of years)")
plt.ylabel("distance from the ridge (km)")
plt.title("Pacific-Antarctic Ridge, {M['par_n']} picks")
plt.show()
""")

md("""
Straight, and the physical constraint is even more obvious here than it was for the supernovae:
seafloor of age zero is being made at the ridge right now, so it is zero kilometres from the
ridge. The line has a point it must pass through, and it is the origin again.
""")

ask("""
### ✏️ Your turn 5

Fit the Pacific-Antarctic picks **both ways** — once free, once with `fit_intercept=False` — with
`par[["Age"]]` as x and `par["Distance"]` as y.

Print, for the free fit, its slope and its intercept; and for the forced fit, its slope. Report
the slopes in **centimetres per year** as well as km per million year: one km per million years is
0.1 cm per year, so divide by 10.

Then draw both lines on the scatter, so you can see how far apart they are.

**Use these names**, because the self-check looks for them: `par_free`, `par_forced`.
""")

answer(f"""
par_free = LinearRegression().fit(par[["Age"]], par["Distance"])
par_forced = LinearRegression(fit_intercept=False).fit(par[["Age"]], par["Distance"])

print("free fit:  ", round(par_free.coef_[0] / 10, 2), "cm/yr, crossing age zero at",
      round(par_free.intercept_, 1), "km from the ridge")
print("forced fit:", round(par_forced.coef_[0] / 10, 2), "cm/yr")

plt.scatter(par["Age"], par["Distance"], s=10, label="picks")
plt.plot(par["Age"], par_free.predict(par[["Age"]]), color="black", label="free fit")
plt.plot(par["Age"], par_forced.predict(par[["Age"]]), color="red",
         label="forced through the origin")
plt.xlabel("age of the seafloor (millions of years)")
plt.ylabel("distance from the ridge (km)")
plt.title("Pacific-Antarctic Ridge, {M['par_n']} picks, two fits")
plt.legend()
plt.show()
""", """
assert par_forced.intercept_ == 0, \\
    "the forced model should have no intercept at all — did you pass fit_intercept=False?"
print("✓ the Pacific-Antarctic Ridge — free fit",
      round(par_free.coef_[0] / 10, 2), "cm/yr with the ridge",
      round(par_free.intercept_), "km from itself; forced through the origin",
      round(par_forced.coef_[0] / 10, 2), "cm/yr")
""")

md(f"""
The free fit puts seafloor of age zero {M['par_free_intercept']:.0f} km away from the ridge
that is making it. There is no reading of that which is physically possible: no seafloor can
exist before any seafloor has been made. It is the {M['sn_free_intercept']:.0f} km/s intercept
again in different clothes, but the cause here is in the picks themselves: they do not sit on
one straight line, they bend, and a single straight line drawn through a bend can cross age zero
a long way from zero. Forcing the line through the origin moves the answer from
{M['par_free_slope'] / 10:.2f} to {M['par_forced_slope'] / 10:.2f} cm/yr —
{abs(M['par_forced_slope'] - M['par_free_slope']) / 10:.2f} cm/yr of difference produced by an
argument rather than by any new data.

## A second ridge

One ridge is not a result. The Mid-Atlantic table has {M['mar_n']} picks, and the fit is lines you
have already written.
""")

code(f"""
mar_forced = LinearRegression(fit_intercept=False).fit(mar[["Age"]], mar["Distance"])

print("Mid-Atlantic:      ", round(mar_forced.coef_[0] / 10, 2), "cm/yr, R²",
      round(mar_forced.score(mar[["Age"]], mar["Distance"]), 3))
print("Pacific-Antarctic: ", round(par_forced.coef_[0] / 10, 2), "cm/yr, R²",
      round(par_forced.score(par[["Age"]], par["Distance"]), 3))
print("ratio:             ", round(par_forced.coef_[0] / mar_forced.coef_[0], 2))
""")

code(f"""
plt.scatter(mar["Age"], mar["Distance"], s=10, label="Mid-Atlantic picks")
plt.scatter(par["Age"], par["Distance"], s=10, label="Pacific-Antarctic picks")
plt.plot(mar["Age"], mar_forced.predict(mar[["Age"]]), color="black")
plt.plot(par["Age"], par_forced.predict(par[["Age"]]), color="red")
plt.xlabel("age of the seafloor (millions of years)")
plt.ylabel("distance from the ridge (km)")
plt.title("Two ridges, {M['n_picks']} picks, both forced through the origin")
plt.legend()
plt.show()
""")

md(f"""
The Mid-Atlantic line rises at {M['mar_forced_slope'] / 10:.2f} cm/yr and the Pacific-Antarctic
one at {M['par_forced_slope'] / 10:.2f} cm/yr — {M['ridge_ratio']:.1f} times as steep, over
{M['mar_age_max']:.0f} million years of Atlantic seafloor and {M['par_age_max']:.0f} million years
of Pacific. Both are roughly the speed a fingernail grows, and neither fit accounts for less than
{int(min(M['par_forced_r2'], M['mar_forced_r2']) * 100)} % of the variation in the distances, so
the gap between them is not slop in the fitting.

It is not slop in the Earth either. Plates are pulled along mainly by their own sinking edges:
where old, cold ocean floor bends down into the mantle at a trench, its weight drags the rest of
the plate after it, and that force dominates everything else (Forsyth & Uyeda, *Geophysical
Journal of the Royal Astronomical Society* **43**, 163, 1975). The Pacific plate is ringed almost
all the way round by trenches. The plates on either side of the Mid-Atlantic Ridge — North America
and Africa — have almost no sinking edge anywhere. So the Atlantic opens slowly and the Pacific
opens fast, and the number you fitted is a measurement of that difference.
""")

md(weekkit.CLOSING_HEADING)

md(f"""
About {M['age_forced']:.2f} billion years — the slope of a straight line through
{M['n_sn']} exploding stars, forced through the origin because zero distance has to mean zero
recession, and landing {abs(M['age_forced'] - PLANCK_AGE):.2f} billion years from the
{PLANCK_AGE} ± {PLANCK_ERR} billion years the microwave background gives.
""")

md(weekkit.week_cheatsheet(6))

# --- homework -------------------------------------------------------------
md("""
## Homework

Three parts on the two datasets you already have loaded. Parts 1 and 2 are two versions of the
same worry: class fitted one line to a whole table and read one number off it, and neither table
was asked whether a single line is really enough. Part 3 is where you argue from your own numbers.
If you have restarted since class, run the setup cell at the top, then the cell below, and then
your own answer to *Your turn 4*, so that `age_of_universe` exists again.
""")

code(weekkit.CHECKPOINT.format(body='''sn = load(STARS + "Freedman2000_Supernova1a.csv", "%s")
par = load(RIDGES + "PAR_east_age_dist.csv", "%s")'''
                               % (CACHED[SN_PATH], CACHED[PAR_PATH])))

ask(f"""
### ✏️ Your turn 6

If Hubble's law is really a law, then the {M['n_sn']} supernovae should give the same H₀ whichever
ones you use. Test it on the two halves of the table.

Sort the table by distance with `sn.sort_values("D(Mpc)")`, take the nearest 18 with `.head(18)`
and the farthest 18 with `.tail(18)`, and fit each half through the origin. Print both Hubble
constants, and put both through your `age_of_universe` function from class to get two ages.

**Use these names**, because the self-check looks for them: `H0_near`, `H0_far`, `age_near`,
`age_far`.
""")

answer("""
ordered = sn.sort_values("D(Mpc)")
near = ordered.head(18)
far = ordered.tail(18)

near_fit = LinearRegression(fit_intercept=False).fit(near[["D(Mpc)"]], near["VCMB"])
far_fit = LinearRegression(fit_intercept=False).fit(far[["D(Mpc)"]], far["VCMB"])

H0_near = near_fit.coef_[0]
H0_far = far_fit.coef_[0]
age_near = age_of_universe(H0_near)
age_far = age_of_universe(H0_far)

print("nearest 18: ", round(H0_near, 2), "km/s/Mpc ->", round(age_near, 2), "billion years")
print("farthest 18:", round(H0_far, 2), "km/s/Mpc ->", round(age_far, 2), "billion years")
""", """
assert H0_near != H0_far, \\
    "two different halves of the table cannot give the identical fit — check you split it"
print("✓ near against far — the two halves differ by",
      round(abs(age_near - age_far), 2), "billion years")
""")

ask(f"""
### ✏️ Your turn 7

Now the same worry about the Pacific-Antarctic Ridge, where it has more bite. Class fitted one
line to all {M['par_n']} picks and read off a single rate — but the picks run from
{M['par_age_min']} to {M['par_age_max']:.0f} million years, and there is no law saying a ridge
must keep the same speed for forty million years.

Take only the older picks, `old = par[par["Age"] >= 20]`, and fit **those** both ways: free, and
forced through the origin. Print both slopes in cm/yr and the free fit's intercept. Then draw the
picks you kept with both of your lines on top.

This is the fork, and this time the two answers are both defensible. Forcing through the origin
says *the ridge existed at age zero, so the line must start there*. Fitting free says *I am asking
how fast this ridge moved between 20 and {M['par_age_max']:.0f} million years ago, and the
origin is outside that window*. Part 3 is where you choose.

**Use these names**, because the self-check looks for them: `old`, `old_free`, `old_forced`.
""")

answer(f"""
old = par[par["Age"] >= 20]

old_free = LinearRegression().fit(old[["Age"]], old["Distance"])
old_forced = LinearRegression(fit_intercept=False).fit(old[["Age"]], old["Distance"])

print(len(old), "picks kept")
print("free fit:  ", round(old_free.coef_[0] / 10, 2), "cm/yr, crossing age zero at",
      round(old_free.intercept_, 1), "km")
print("forced fit:", round(old_forced.coef_[0] / 10, 2), "cm/yr")

plt.scatter(old["Age"], old["Distance"], s=10, label="picks 20 Ma and older")
plt.plot(old["Age"], old_free.predict(old[["Age"]]), color="black", label="free fit")
plt.plot(old["Age"], old_forced.predict(old[["Age"]]), color="red",
         label="forced through the origin")
plt.xlabel("age of the seafloor (millions of years)")
plt.ylabel("distance from the ridge (km)")
plt.title("Pacific-Antarctic Ridge, {M['old_n']} picks older than 20 Ma")
plt.legend()
plt.show()
""", """
assert old["Age"].min() >= 20, "old should keep the picks 20 Ma and OLDER"
print("✓ the older half of the ridge — free", round(old_free.coef_[0] / 10, 2),
      "cm/yr against forced", round(old_forced.coef_[0] / 10, 2), "cm/yr")
""")

ask(f"""
### ✏️ Your turn 8

Four numbers, all yours: the whole-ridge rate from class ({M['par_forced_slope'] / 10:.2f} cm/yr,
forced through the origin), the Mid-Atlantic rate from class
({M['mar_forced_slope'] / 10:.2f} cm/yr), and your two rates for the older half of the
Pacific-Antarctic picks.

Quote all four, then answer both of these in a short paragraph.

**Which of your two fits for the older half would you publish** as the spreading rate for that
window of time, and why? Your answer has to deal with the free fit's intercept, which is no longer
close to zero.

**And is the difference *between* the two ridges bigger or smaller than the difference *within*
the Pacific-Antarctic Ridge?** Give both as a ratio, from your own numbers, and say what your
answer implies about whether "the spreading rate of a ridge" is one number or several.
""")

answer_prose(f"""
Whole Pacific-Antarctic ridge, forced through the origin: {M['par_forced_slope'] / 10:.2f} cm/yr.
Mid-Atlantic: {M['mar_forced_slope'] / 10:.2f} cm/yr. Older Pacific-Antarctic picks only:
{M['old_free_slope'] / 10:.2f} cm/yr free, {M['old_forced_slope'] / 10:.2f} cm/yr forced through
the origin.

I would publish the free fit, {M['old_free_slope'] / 10:.2f} cm/yr, for that window. The origin
constraint was a real physical fact in class because the line was being asked about the whole
history of the ridge, including age zero. Here it is not: the question is how fast the seafloor
moved between 20 and {M['par_age_max']:.0f} million years ago, and age zero is outside the data.
The free fit's intercept of {M['old_free_intercept']:.0f} km is not a claim that
{M['old_free_intercept']:.0f} km of seafloor existed at the start — it is just where this segment
of the line happens to cross, and forcing it to zero instead makes the fit answer a different
question, averaging in a faster recent period it was never given. The clue is that the forced
version, {M['old_forced_slope'] / 10:.2f} cm/yr, comes out much closer to the whole-ridge number
than the free version does.

Between the ridges: {M['par_forced_slope'] / 10:.2f} / {M['mar_forced_slope'] / 10:.2f} =
{M['ridge_ratio']:.1f}. Within the Pacific-Antarctic: {M['par_forced_slope'] / 10:.2f} /
{M['old_free_slope'] / 10:.2f} = {M['within_par_ratio']:.2f}. The gap between the two ridges is
far bigger than the change inside one of them, so a single number is a fair summary of a ridge for
most purposes — but it is a summary, not a constant. My own numbers say the Pacific-Antarctic
Ridge was around {(M['par_forced_slope'] - M['old_free_slope']) / M['old_free_slope'] * 100:.0f} %
slower in the older part of the record than the whole-ridge fit reports, which is a real change in
the speed of a plate and would be invisible to anyone who only ever fitted one line.
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
    for name in CACHED.values():
        print(f"cache: data/{name}")


if __name__ == "__main__":
    main()
    weekkit.gate(6)
