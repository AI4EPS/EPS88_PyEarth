#!/usr/bin/env python
"""Build week 7 — "How often does a Tambora happen?"

Emits both notebooks from one source so they cannot drift:

    docs/notebooks/07_how_often_tambora_solution.ipynb   executed, every output saved
    docs/notebooks/07_how_often_tambora.ipynb            the same file with the answers deleted

It also writes the three cached fallbacks this week reads: two USGS catalogue slices and the
Smithsonian eruption table.

Every number that appears in prose or in a model answer is computed HERE, from the same files
the notebook reads, and formatted in. Nothing is typed from memory or copied from the plan.

    python tools/build_week07.py
"""
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/notebooks"
SLUG = "07_how_often_tambora"

course = yaml.safe_load((ROOT / "course.yml").read_text())
WEEK = next(s for s in course["schedule"] if s["n"] == 7)
PLATFORM = course["platform"]
CACHE_BASE = PLATFORM["cache_base"]

# The three live reads. Pinned here so the cached CSVs, the notebook and the prose below cannot
# drift apart. The box is course.yml's pinned California slice.
BOX = "&minlatitude=32&maxlatitude=42&minlongitude=-125&maxlongitude=-114"
USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&orderby=time-asc"
EQ_START, EQ_END, EQ_FLOOR = "1990-01-01", "2026-01-01", 3.5
HIST_START, HIST_END, HIST_FLOOR = "1810-01-01", "2026-01-01", 7.0
GVP = ("https://webservices.volcano.si.edu/geoserver/GVP-VOTW/ows?service=WFS&version=1.0.0"
       "&request=GetFeature&typeName=GVP-VOTW:Smithsonian_VOTW_Holocene_Eruptions"
       "&outputFormat=csv")

EQ_CACHE = f"week07_california_{EQ_START}_{EQ_END}_M{EQ_FLOOR}.csv"
HIST_CACHE = f"week07_california_{HIST_START}_{HIST_END}_M{HIST_FLOOR}.csv"
GVP_CACHE = "week07_gvp_eruptions.csv"

EQ_URL = f"{USGS}&starttime={EQ_START}&endtime={EQ_END}&minmagnitude={EQ_FLOOR}{BOX}"
HIST_URL = f"{USGS}&starttime={HIST_START}&endtime={HIST_END}&minmagnitude={HIST_FLOOR}{BOX}"

VOLCANO_FROM = 1800          # the window the class fits
WINDOW = 36                  # years per window, both for the prediction and for the homework
FORK_YEARS = [1700, 1900]    # the homework's two other start years

# USGS, "Earthquake Magnitude, Energy Release, and Shaking Intensity", read 2026-08-31:
# each whole number of magnitude is about 32 times more energy released.
ENERGY_PER_STEP = 32


# ---------------------------------------------------------------------------
# 1. measure everything the notebook will say
# ---------------------------------------------------------------------------
def fetch(url, name):
    """Run one live query once, cache it beside the course, and return the cached copy."""
    out = ROOT / "data" / name
    if not out.exists():
        pd.read_csv(url).to_csv(out, index=False)
    return pd.read_csv(out)


def levels_between(lowest, highest):
    """The magnitudes lowest, lowest + 0.1, ... highest, built from whole numbers."""
    return np.arange(round(lowest * 10), round(highest * 10) + 1) / 10


def count_at_least(values, level):
    """How many of these values are at or above the level."""
    return (values >= level).sum()


def fit_line(values, levels):
    """Least squares through log10 of the cumulative counts. Returns the fitted model."""
    counts = [count_at_least(values, level) for level in levels]
    model = LinearRegression()
    model.fit(levels.reshape(-1, 1), np.log10(counts))
    return model, np.array(counts)


def predict_count(values, levels, target):
    """The slope of that line, and the count it predicts at target."""
    model, _ = fit_line(values, levels)
    return model.coef_[0], 10 ** model.predict([[target]])[0]


quakes = fetch(EQ_URL, EQ_CACHE)
big_history = fetch(HIST_URL, HIST_CACHE)
eruptions = fetch(GVP, GVP_CACHE)

# `mags` is EVERY magnitude the slice returned, the five magnitude 7s included. A count labelled
# "at or above this magnitude" has to be that count: dropping the large events made the level-6.0
# point read 16 where the catalogue holds 21, steepened the line, and manufactured part of the
# very shortfall this week argues about. What is held back is the RANGE the line is fitted over
# (course.yml, pinned: training_range), never the events. `big` is the same five events, picked
# out so the reveal has something to print.
big = quakes[quakes["mag"] >= 7.0]
mags = quakes["mag"].values

M = {}
M["n_quakes"] = len(quakes)
M["n_big"] = len(big)
M["mean_mag"] = round(float(mags.mean()), 3)
M["median_mag"] = round(float(np.median(mags)), 2)
M["max_mag"] = float(mags.max())
M["energy_ratio"] = round(float(ENERGY_PER_STEP ** (M["max_mag"] - mags.mean())))
M["places"] = list(big["place"])
M["big_years"] = [t[:4] for t in big["time"]]

M["n4"] = int(count_at_least(mags, 4.0))
M["n5"] = int(count_at_least(mags, 5.0))
M["n6"] = int(count_at_least(mags, 6.0))
M["ratio45"] = round(M["n4"] / M["n5"], 2)
M["ratio56"] = round(M["n5"] / M["n6"], 2)

MAG_LEVELS = levels_between(3.5, 5.0)
eq_model, eq_counts = fit_line(mags, MAG_LEVELS)
M["n_at_3_5"] = int(eq_counts[0])
M["n_at_4_5"] = int(eq_counts[10])
M["n_at_5_0"] = int(eq_counts[-1])
M["tail_share"] = round(100 * M["n_at_4_5"] / M["n_at_3_5"])
M["slope"] = round(float(eq_model.coef_[0]), 3)
M["intercept"] = round(float(eq_model.intercept_), 3)
M["step_factor"] = round(float(10 ** -eq_model.coef_[0]), 2)
M["r2"] = round(float(eq_model.score(MAG_LEVELS.reshape(-1, 1), np.log10(eq_counts))), 5)

# The control that shows what that R squared is worth. Cumulative counts can only fall, so the
# fit is very nearly guaranteed a straight line whatever the magnitudes are: run it on magnitudes
# drawn UNIFORMLY across the catalogue's own range — the least power-law-like thing there is —
# and it still scores near 1. Measured here, and again in the notebook, rather than asserted.
FLAT_SEED = 88                                  # the course number, as everywhere else
FLAT_LOW = int(round(EQ_FLOOR * 10))            # the catalogue's own range, counted in tenths,
FLAT_TOP = int(round(M["max_mag"] * 10)) + 1    # so every magnitude on the 0.1 grid is drawable
flat = np.random.default_rng(FLAT_SEED).integers(FLAT_LOW, FLAT_TOP, len(mags)) / 10
flat_model, flat_counts = fit_line(flat, MAG_LEVELS)
M["r2_flat"] = round(float(flat_model.score(MAG_LEVELS.reshape(-1, 1), np.log10(flat_counts))), 5)

# the float trap the notebook warns about, measured rather than asserted
naive = np.arange(3.5, 5.05, 0.1)
M["n_at_3_8"] = int(count_at_least(mags, 3.8))
M["n_naive_3_8"] = int(count_at_least(mags, naive[3]))
M["lost_to_float"] = M["n_at_3_8"] - M["n_naive_3_8"]

EQ_RANGES = [(3.5, 5.0), (4.0, 5.5), (4.5, 6.0)]
M["eq_predictions"] = [round(float(predict_count(mags, levels_between(lo, hi), 7.0)[1]), 2)
                       for lo, hi in EQ_RANGES]
M["pred_main"] = M["eq_predictions"][0]
M["under_by"] = round(M["n_big"] / M["pred_main"], 1)

# --- volcanoes -------------------------------------------------------------
rated = eruptions.dropna(subset=["ExplosivityIndexMax"])
recent = rated[rated["StartDateYear"] >= VOLCANO_FROM]
vei = recent["ExplosivityIndexMax"].values
M["n_eruption_rows"] = len(eruptions)
M["n_rated"] = len(rated)
M["n_recent"] = len(recent)
M["vei_counts"] = [int(count_at_least(vei, k)) for k in range(8)]
M["vei_exact2"] = int((vei == 2).sum())
M["vei_exact3"] = int((vei == 3).sum())
M["span"] = 2026 - VOLCANO_FROM

VEI_LEVELS = np.array([2, 3, 4])
vei_model, _ = fit_line(vei, VEI_LEVELS)
M["vei_slope"] = round(float(vei_model.coef_[0]), 3)
M["vei_step_factor"] = round(float(10 ** -vei_model.coef_[0]), 2)
M["vei_line"] = [round(float(10 ** vei_model.predict([[k]])[0]), 2) for k in range(8)]
M["vei_pred7"] = round(float(10 ** vei_model.predict([[7]])[0]), 2)
M["vei_obs7"] = M["vei_counts"][7]
M["one_per"] = round(M["span"] / float(10 ** vei_model.predict([[7]])[0]))

top = rated[rated["ExplosivityIndexMax"] >= 7]
M["tambora_year"] = int(top[top["Volcano_Name"] == "Tambora"]["StartDateYear"].iloc[0])
M["n_top_all"] = len(top)
M["oldest_top"] = int(top["StartDateYear"].min())
M["top_span"] = 2026 - M["oldest_top"]
M["top_rate"] = round(M["top_span"] / M["n_top_all"])

# --- how much one observation can settle -----------------------------------
POISSON_RATES = [0.1, 0.5, 1.0, 2.0, 4.0]
rng = np.random.default_rng(88)
M["poisson"] = [round(float((rng.poisson(rate, size=20000) == 1).mean()), 3)
                for rate in POISSON_RATES]
M["poisson_lo"] = min(M["poisson"])
M["poisson_hi"] = max(M["poisson"])

# --- homework --------------------------------------------------------------
hist_years = big_history["time"].str[:4].astype(int)
WINDOW_STARTS = list(range(2026 - 6 * WINDOW, 2026, WINDOW))
M["window_counts"] = [int(((hist_years >= s) & (hist_years < s + WINDOW)).sum())
                      for s in WINDOW_STARTS]
M["n_history"] = len(big_history)
M["window_mean"] = round(float(np.mean(M["window_counts"])), 2)
M["busiest_window"] = WINDOW_STARTS[int(np.argmax(M["window_counts"]))]

M["fork"] = {}
for year in FORK_YEARS:
    sub = rated[rated["StartDateYear"] >= year]["ExplosivityIndexMax"].values
    slope, pred = predict_count(sub, VEI_LEVELS, 7.0)
    M["fork"][year] = {"n": len(sub), "slope": round(float(slope), 3),
                       "pred": round(float(pred), 2),
                       "obs": int(count_at_least(sub, 7))}


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
In 1815 the Indonesian volcano Tambora blew its summit apart in the largest eruption anybody has
watched happen. Ash and sulphur reached the stratosphere and stayed there, and the following year
went down in Europe and eastern North America as the Year Without a Summer: frost in June, failed
harvests, bread riots. The obvious question is how often the Earth does that.

You cannot answer it by counting. There is one Tambora in the record, and one event is not a rate.
But the same catalogue holds thousands of small eruptions, California's catalogue holds thousands
of small earthquakes, and both turn out to sit on a single straight line — once you plot them the
right way. Today you will find that line where the data is thick, use it to predict something the
record has never shown you, check whether it was right, and only then take it to Tambora.
"""

md(weekkit.OPENING.format(question=WEEK["question"], datahub=datahub, hook=HOOK.strip()))

md("""
## What you'll be able to do

**The science.** Say how many times more small earthquakes than large ones California has, and why
that ratio is a law rather than an accident. Give a number for how often a Tambora-sized eruption
happens, and say honestly how much that number is worth.

**The skills.** Turn a column of sizes into a **cumulative count** — how many events at or above
each level. Put those counts on a **log axis** so a hopeless curve becomes a straight line. Fit
that line with `LinearRegression`, read its slope, and use `predict` to ask it about a size that is
not in your data at all.

**Eight places where you write something: five in class, three at home.** Each one is headed
*Your turn*, with an empty cell under it.
""")

setup = weekkit.setup_cell(
    imports="import numpy as np\nfrom sklearn.linear_model import LinearRegression\n",
    figsize="(6.5, 4)",
    cache_base=CACHE_BASE,
    signature="url, cache_name",
    docstring="Read one live catalogue; fall back to the copy stored with the course.",
    url_expr="url",
    cache_expr="cache_name",
    unpack=f'''
BOX = "{BOX}"
USGS = "{USGS}"
GVP = ("https://webservices.volcano.si.edu/geoserver/GVP-VOTW/ows?service=WFS&version=1.0.0"
       "&request=GetFeature&typeName=GVP-VOTW:Smithsonian_VOTW_Holocene_Eruptions"
       "&outputFormat=csv")

quakes = load(USGS + "&starttime={EQ_START}&endtime={EQ_END}&minmagnitude={EQ_FLOOR}" + BOX,
              "{EQ_CACHE}")
big_history = load(USGS + "&starttime={HIST_START}&endtime={HIST_END}&minmagnitude={HIST_FLOOR}" + BOX,
                   "{HIST_CACHE}")
eruptions = load(GVP, "{GVP_CACHE}")

# mags is every magnitude in the box, the largest included: "how many at magnitude 5 or above"
# has to count the magnitude 7s too, or it is not that count. What we hold back today is the
# RANGE of magnitudes the line gets fitted over, never the earthquakes themselves.
mags = quakes["mag"].values
big = quakes[quakes["mag"] >= 7.0]      # the same events again, gathered so we can look at them

print("California catalogue:", quakes.shape, "  eruption catalogue:", eruptions.shape)
print("magnitude 7 and above in the same box since 1810:", len(big_history))
'''.strip("\n"))
code(setup)

# --- section 1 -------------------------------------------------------------
md(f"""
## One eruption is not a rate

The Smithsonian Institution's Global Volcanism Program keeps a table of every eruption it can
document, {M['n_eruption_rows']:,} of them, and rates most of those on the **Volcanic Explosivity
Index** — VEI, a whole number from 0 to 8. It works like an earthquake magnitude: each step up is
roughly ten times the volume of material thrown out, so VEI 2 is a nuisance and VEI 7 rearranges
the climate.

Ask the catalogue how many eruptions of each size it holds since {VOLCANO_FROM}.
""")

code(f"""
rated = eruptions.dropna(subset=["ExplosivityIndexMax"])
recent = rated[rated["StartDateYear"] >= {VOLCANO_FROM}]

print(recent["ExplosivityIndexMax"].value_counts().sort_index())
print(recent[recent["ExplosivityIndexMax"] == 7][["Volcano_Name", "StartDateYear"]])
""")

md(f"""
There it is, and there is only one of it. In the {M['span']} years since {VOLCANO_FROM} the
catalogue holds exactly {M['vei_obs7']} eruption at VEI 7 — Tambora, whose eruptive episode the
catalogue dates from {M['tambora_year']}, three years before the explosion that made 1816 cold.

One event gives you no rate. Divide 1 by {M['span']} years and you get a number, but you would have
got a different number from any other stretch of history, and nothing tells you which is right.
Counting worked when we had hundreds of events to count; here it dies.

What the catalogue does have is {M['vei_exact2']:,} eruptions rated VEI 2 and
{M['vei_exact3']:,} rated VEI 3. If the small ones and the large ones are connected by something
regular, that is thousands of measurements of the thing we want. Finding out whether they are
connected needs a catalogue where even the small events get counted properly — and the best such
catalogue on Earth is a seismic network, so we go there first.
""")

# --- section 2 -------------------------------------------------------------
md(f"""
## The shape of a catalogue of earthquakes

`quakes` is every earthquake of magnitude {EQ_FLOOR} and above that the USGS recorded between
{EQ_START} and {EQ_END} inside a box around California: latitude 32 to 42 north, longitude 125 to
114 west. That box is a rectangle, not a state — it reaches into Nevada and into Baja California —
and it holds {M['n_quakes']:,} earthquakes, whose magnitudes are all in `mags`.

Start with the plainest possible picture of them.
""")

code(f"""
plt.hist(mags, bins=np.arange(3.5, 8.0, 0.5))
plt.xlabel("magnitude")
plt.ylabel("number of earthquakes")
plt.title("California, {M['n_quakes']:,} earthquakes at magnitude {EQ_FLOOR} and above, "
          "{EQ_START[:4]}-{EQ_END[:4]}")
plt.show()
""")

md("""
A wall on the left and nothing on the right. There is no hump anywhere: almost every event sits
close to the magnitude 3.5 floor we asked for, and it would sit close to any other floor we chose,
because there are always more small ones. The events anybody actually cares about are off in a
region of the axis that looks empty.

Which is why the usual summary of a column — its middle — is worse than useless here.
""")

code(f"""
print("mean magnitude:  ", round(mags.mean(), 3))
print("median magnitude:", round(np.median(mags), 2))
print("largest in the box:", mags.max())

# USGS, "Earthquake Magnitude, Energy Release, and Shaking Intensity", read 2026-08-31:
# one whole step of magnitude is about {ENERGY_PER_STEP} times more energy released.
energy_ratio = {ENERGY_PER_STEP} ** (mags.max() - mags.mean())
print(f"the largest released about {{round(energy_ratio):,}} times the energy of an average one")
""")

md(f"""
The average California earthquake in this file is magnitude {M['mean_mag']}. People nearby feel one
of those, and it damages nothing; no plan for the state's next hundred years turns on it. Meanwhile
the largest in the box released roughly {M['energy_ratio']:,} times as much energy as that average.
When a distribution is shaped like this one, the mean describes the crowd
and the crowd is irrelevant: everything that matters is in the part of the axis where the histogram
looks like zero.

So stop asking what is typical, and start asking how the count changes as the size goes up.
""")

ask(f"""
### ✏️ Your turn 1

`mags` holds the magnitude of every one of the {M['n_quakes']:,} earthquakes in the box.
Count how many are at magnitude 4.0 or above, at 5.0 or above, and at 6.0 or above — a comparison
gives you True and False, and adding those up counts the Trues. Then print each count divided by
the next one.

**Use these names**, because the self-check looks for them: `n4`, `n5`, `n6`.
""")

answer("""
n4 = (mags >= 4.0).sum()
n5 = (mags >= 5.0).sum()
n6 = (mags >= 6.0).sum()

print("magnitude 4 and above:", n4)
print("magnitude 5 and above:", n5)
print("magnitude 6 and above:", n6)
print("times more 4s than 5s:", round(n4 / n5, 2))
print("times more 5s than 6s:", round(n5 / n6, 2))
""", """
assert n6 < n5 < n4, "the counts must fall as the magnitude rises — check which way your >= points"
print("✓ counting up the catalogue —", n4, "at M4+,", n5, "at M5+,", n6, "at M6+, so",
      round(n4 / n5, 2), "and", round(n5 / n6, 2), "times fewer at each step")
""")

md(f"""
One step up in magnitude and there are {M['ratio45']} times fewer earthquakes; one more step and
there are {M['ratio56']} times fewer again. The same factor twice is the sort of thing that stops
being a coincidence, so the next job is to measure it properly instead of at three points.

## Counting upwards, at every level

The histogram above counted earthquakes **in** each magnitude bin. What Your turn 1 counted is
different and more useful: how many are **at or above** a level. That is a *cumulative* count, and
it is the natural thing to ask of a hazard — nobody wants to know how many earthquakes were between
5.9 and 6.0, they want to know how many were 6 or worse.

Doing it at every level from 3.5 to 5.0 needs the list of levels first. There is a trap in building
it, so we build it in a function and use that function all week.
""")

code("""
def levels_between(lowest, highest):
    \"\"\"The magnitudes lowest, lowest + 0.1, ... highest, built from whole numbers.\"\"\"
    return np.arange(round(lowest * 10), round(highest * 10) + 1) / 10


mag_levels = levels_between(3.5, 5.0)
print(mag_levels)

the_obvious_way = np.arange(3.5, 5.05, 0.1)      # looks the same, and is not
print("is its fourth entry equal to 3.8?", the_obvious_way[3] == 3.8)
print("earthquakes at 3.8 and above:", (mags >= 3.8).sum(),
      "  using its fourth entry instead:", (mags >= the_obvious_way[3]).sum())
""")

md(f"""
Whole numbers first, then divide. The obvious way looks identical on screen and is not: its fourth
entry is a hair above 3.8, so it excludes every earthquake recorded as exactly 3.8 and the count
falls from {M['n_at_3_8']:,} to {M['n_naive_3_8']:,} — {M['lost_to_float']} earthquakes gone, with
no error and no warning. Decimals are stored as approximations, so two that ought to be equal often
are not. Build the levels from integers and the problem never arises.
""")

ask(f"""
### ✏️ Your turn 2

Write `count_at_least(values, level)`: one line, returning how many of `values` are at or above
`level`. Give it a docstring.

Then use it in a loop over `mag_levels` to build a list called `counts`, and print the first and
last entries with the level each belongs to.

**Use these names**, because the self-check looks for them: `count_at_least`, `counts`.
""")

answer("""
def count_at_least(values, level):
    \"\"\"How many of these values are at or above the level.\"\"\"
    return (values >= level).sum()


counts = []
for level in mag_levels:
    counts.append(count_at_least(mags, level))

print("at magnitude", mag_levels[0], "and above:", counts[0])
print("at magnitude", mag_levels[-1], "and above:", counts[-1])
""", """
assert len(counts) == len(mag_levels), "one count per level — is the append inside the loop?"
assert counts[0] > counts[-1], "the counts must fall as the level rises"
print("✓ cumulative counts —", len(counts), "levels, from", counts[0], "down to",
      counts[-1])
""")

md("""
Now draw them. Level along the bottom, count up the side, one dot per level.
""")

code(f"""
plt.scatter(mag_levels, counts)
plt.xlabel("magnitude")
plt.ylabel("number of earthquakes at or above this magnitude")
plt.title("California, {M['n_quakes']:,} earthquakes counted at {len(MAG_LEVELS)} levels")
plt.show()
""")

md(f"""
A curve, and one that hides the half we came for. The left-hand point is {M['n_at_3_5']:,} and the
right-hand one is {M['n_at_5_0']}, so the axis has to reach {M['n_at_3_5']:,} and the whole right
half of the plot is packed into a thin band along the bottom. You cannot tell by looking whether
those dots fall along a line, along a curve, or along nothing in particular — and they are the ones
nearest the sizes that matter.

The fix is the one the plotting week introduced. When the values span factors of a thousand, plot
the exponents instead and a curve becomes a line. `plt.yscale("log")` does exactly that: the
distance from 10 to 100 on the axis becomes the same as the distance from 100 to 1000.
""")

code(f"""
plt.scatter(mag_levels, counts)
plt.yscale("log")
plt.xlabel("magnitude")
plt.ylabel("number of earthquakes at or above this magnitude")
plt.title("California, the same {len(MAG_LEVELS)} counts of {M['n_quakes']:,} earthquakes, log axis")
plt.show()
""")

md("""
The same numbers, and now they are a straight line.

A straight line on a log count axis has a name and a meaning. Every step up in size divides the
count by the same factor — which lets you predict the sizes you have never seen. Seismologists call
this the Gutenberg–Richter relation, after the two Caltech seismologists who described it in the
1940s, and it holds with the same shape in essentially every region anybody has looked at.

## Measuring the line

A straight line is what least squares is for. The log axis was a change of drawing and not of
data, so the counts themselves are still a curve; what is straight is `np.log10(counts)` plotted
against magnitude, and that is the pair least squares gets. Draw the best straight line. Best means
the smallest total miss.

`LinearRegression` wants one row per data point rather than one long row, which is what
`.reshape(-1, 1)` is doing below: sixteen numbers become sixteen rows of one number each.
""")

ask("""
### ✏️ Your turn 3

Fit a straight line through `mag_levels` and `np.log10(counts)`:

```
model = LinearRegression()
model.fit(mag_levels.reshape(-1, 1), np.log10(counts))
```

Then print `model.coef_[0]` as `slope`, print `model.intercept_`, print the R-squared from
`model.score(...)` on the same two arguments, and finally print `10 ** -slope` — which is how many
times fewer earthquakes there are for each whole step up in magnitude.

**Use these names**, because the self-check looks for them: `model`, `slope`.
""")

answer("""
model = LinearRegression()
model.fit(mag_levels.reshape(-1, 1), np.log10(counts))
slope = model.coef_[0]

print("slope:    ", round(slope, 3))
print("intercept:", round(model.intercept_, 3))
print("R squared:", round(model.score(mag_levels.reshape(-1, 1), np.log10(counts)), 5))
print("times fewer per whole magnitude:", round(10 ** -slope, 2))
""", """
assert -1.5 < slope < -0.5, ("a slope near -1 is expected here; if yours is far from that, "
                             "check that you fitted np.log10(counts) and not counts")
print("✓ the fitted line — slope", round(slope, 3), "so each whole magnitude step "
      "divides the count by", round(10 ** -slope, 2))
""")

md(f"""
A slope of {M['slope']}, with R squared {M['r2']}. Seismologists quote the size of that slope and
call it the **b-value**; a b-value near 1 is what Gutenberg and Richter found and what most of the
world's crust gives. It says that earthquakes of magnitude 4 and above outnumber those of
magnitude 5 and above by about {M['step_factor']} to one, and that the same factor holds between
any two levels a whole magnitude apart, with no favoured size anywhere.

That is a strong statement about how the crust breaks. Rock does not have a characteristic
earthquake size the way a person has a characteristic height — the same physics of a rupture
running along a fault and stopping produces every size, and only the chance of running further
decides which one you get.

The one number in that output you should not be impressed by is the R squared. A cumulative count
can only fall as the level rises — that is what "at or above" means — so these dots were going to
descend smoothly whatever the magnitudes were, and a fit to something already smooth and already
falling scores near 1 almost regardless. Test it: give the identical fit magnitudes with no pattern
in them at all, spread evenly from one end of the catalogue's range to the other.
""")

code(f"""
# the same number of magnitudes, with no law in them: every tenth from {EQ_FLOOR} to
# {M['max_mag']} equally likely, drawn in tenths so they land on the same grid as real magnitudes
flat = np.random.default_rng({FLAT_SEED}).integers({FLAT_LOW}, {FLAT_TOP}, len(mags)) / 10

flat_counts = []
for level in mag_levels:
    flat_counts.append(count_at_least(flat, level))

flat_model = LinearRegression()
flat_model.fit(mag_levels.reshape(-1, 1), np.log10(flat_counts))
print("R squared on magnitudes with no pattern at all:",
      round(flat_model.score(mag_levels.reshape(-1, 1), np.log10(flat_counts)), 5))
""")

md(f"""
{M['r2_flat']} — from magnitudes that hold no law whatever, against {M['r2']} from the real
catalogue. The check could hardly have failed, so passing it says almost nothing, and any argument
resting on that {M['r2']} is resting on the shape of a cumulative count rather than on California.
What would be worth something is the line holding up somewhere it was never fitted. That is
testable, so test it.

## Reading the line off past the end of the data

The line was measured between magnitude 3.5 and 5.0. Nothing stops us evaluating it somewhere else,
and `model.predict` will do it without complaint — so the next cell draws the line right across the
plot, well past the last dot it was fitted to.
""")

code(f"""
line_x = levels_between(3.5, 7.5)
line_y = 10 ** model.predict(line_x.reshape(-1, 1))

plt.scatter(mag_levels, counts, label="counted")
plt.plot(line_x, line_y, color="firebrick", label="the fitted line, extended")
plt.yscale("log")
plt.xlabel("magnitude")
plt.ylabel("number of earthquakes at or above this magnitude")
plt.title("California, {M['n_quakes']:,} earthquakes and the line fitted to {len(MAG_LEVELS)} levels")
plt.legend()
plt.show()
""")

code("""
predicted_7 = 10 ** model.predict([[7.0]])[0]
print("the line expects", round(predicted_7, 2), "earthquakes at magnitude 7 or above")
""")

md(f"""
### Predict before you run

The line, which was measured between magnitude 3.5 and 5.0 and never asked about anything larger,
says to expect about {M['pred_main']} earthquakes of magnitude 7 or above in this box in these
{WINDOW} years.

How many actually happened? Write your guess into `my_guess` below before you run the cell. `big`
holds them, gathered in the setup cell and not looked at since.
""")

code("""
my_guess = 2

print("you guessed:", my_guess)
print("the catalogue holds:", len(big))
print(big[["time", "mag", "place"]].to_string(index=False))
""")

md(f"""
{M['n_big']} of them — about {M['under_by']} times what the line expected. They are real and they
are famous: two in 1992, then {', '.join(M['big_years'][2:])}. Notice also that one of the five is in
Baja California, Mexico: the box is a rectangle drawn on a map, and rectangles do not respect
borders. Whatever this section concludes is about that rectangle, not about the state.

Before blaming the line, check whether the line was ever a single thing. We chose to fit it between
magnitude 3.5 and 5.0. Somebody else would have chosen differently, and the honest question is
whether that choice is doing the work.
""")

ask("""
### ✏️ Your turn 4

Write `predict_count(values, levels, target)`, which packages up what you did in Your turn 2 and
Your turn 3:

- build the cumulative count of `values` at each level in `levels`, using your `count_at_least`
- fit a `LinearRegression` to `levels.reshape(-1, 1)` and `np.log10` of those counts
- **return two things**: the slope, and `10 ** model.predict([[target]])[0]`

Give it a docstring. Then loop over the three fitting ranges `(3.5, 5.0)`, `(4.0, 5.5)` and
`(4.5, 6.0)`, building the levels for each with `levels_between`, and print the slope and the
predicted number at magnitude 7 for each. Collect the three predictions in a list called
`predictions`.

The first range is the one you already did by hand, so its answer is a check on your function.

**Use these names**, because the self-check looks for them: `predict_count`, `predictions`.
""")

answer("""
def predict_count(values, levels, target):
    \"\"\"The slope of the fitted line, and the count it predicts at target.\"\"\"
    counts = []
    for level in levels:
        counts.append(count_at_least(values, level))
    model = LinearRegression()
    model.fit(levels.reshape(-1, 1), np.log10(counts))
    return model.coef_[0], 10 ** model.predict([[target]])[0]


predictions = []
for lowest, highest in [(3.5, 5.0), (4.0, 5.5), (4.5, 6.0)]:
    slope, predicted = predict_count(mags, levels_between(lowest, highest), 7.0)
    predictions.append(predicted)
    print("fitted on", lowest, "to", highest,
          "  slope", round(slope, 3),
          "  expects", round(predicted, 2), "at magnitude 7 or above")
""", """
assert len(predictions) == 3, "three fitting ranges, three predictions"
assert min(predictions) > 0.5, ("a prediction below 0.5 usually means the 10 ** was left off, so "
                                "the function is returning the log of the count")
print("✓ three defensible choices — the line expects",
      round(min(predictions), 2), "to", round(max(predictions), 2),
      "at magnitude 7, against", len(big), "that happened")
""")

md(f"""
The three defensible ranges land between {min(M['eq_predictions'])} and
{max(M['eq_predictions'])}, and {M['n_big']} happened. The choice does matter: the highest of the
three expects about {round(max(M['eq_predictions']) / min(M['eq_predictions']), 1)} times what the
lowest does, which is worth remembering the next time somebody quotes a single number off a fit like
this one. What it does not do is close the gap. Every range falls short of {M['n_big']} by a factor
of between {round(M['n_big'] / max(M['eq_predictions']), 1)} and
{round(M['n_big'] / min(M['eq_predictions']), 1)}, so whatever is missing is missing from all three.

Three things could be true, and this notebook cannot tell them apart. {WINDOW} years may simply be
too short a window for an event this rare, so we are looking at an unlucky draw. Or California may
genuinely make more large earthquakes than its small ones imply: the small events come from cracked
crust everywhere, while the magnitude 7s come from a handful of very long faults rupturing along
most of their length, and whether those long faults deliver more big earthquakes than the
small-event line allows is a live argument in seismology rather than a settled question. Or the
counting could be off. Hold the question open; the first part of the homework goes after the first
of those three.

## The same line, drawn for volcanoes

VEI is a magnitude scale, so the same machinery applies without changing a thing: cumulative counts
at each level, log axis, straight line, and then read the line off at VEI 7 where the catalogue has
almost nothing.

The counts below are every rated eruption since {VOLCANO_FROM}, at each level from 0 upward.
""")

code(weekkit.CHECKPOINT.format(body=f"""# Re-run your own count_at_least (Your turn 2) and predict_count (Your turn 4) cells as well.
# Those two are your code, so this cell cannot rebuild them for you; the rest of the week uses them.
rated = eruptions.dropna(subset=["ExplosivityIndexMax"])
recent = rated[rated["StartDateYear"] >= {VOLCANO_FROM}]
vei = recent["ExplosivityIndexMax"].values"""))

code("""
for level in range(0, 7):
    print("VEI", level, "and above:", count_at_least(vei, level))
""")

md(f"""
Read those from the bottom up. From VEI 2 onwards each step divides the count by roughly the same
factor — {M['vei_counts'][2]:,}, {M['vei_counts'][3]}, {M['vei_counts'][4]}, {M['vei_counts'][5]},
{M['vei_counts'][6]} — but the bottom two steps do not play along at all. Going from VEI 2 down to
VEI 1 the count should multiply by about five, and it barely moves.

That is not volcanology, it is bookkeeping. A catalogue lists what somebody's instruments recorded,
not what happened. Where there are no seismometers there are no earthquakes in the file — and
nobody files a report on a VEI 1 eruption in an uninhabited part of the Andes in 1840. So the fit
starts at VEI 2, where the record is thick enough to trust.
""")

ask(f"""
### ✏️ Your turn 5

Fit the line to VEI 2, 3 and 4 only, and ask it about VEI 7.

Build `vei_levels = np.array([2, 3, 4])`, call your `predict_count` on `vei` with `target` 7, and
print the slope, the predicted number of VEI 7 eruptions since {VOLCANO_FROM}, and — using
`count_at_least` — how many the catalogue actually holds.

**Use these names**, because the self-check and the cells below look for them: `vei_levels`,
`vei_slope`, `vei_predicted`.
""")

answer(f"""
vei_levels = np.array([2, 3, 4])
vei_slope, vei_predicted = predict_count(vei, vei_levels, 7)

print("slope:", round(vei_slope, 3))
print("the line expects", round(vei_predicted, 2), "eruptions at VEI 7 since {VOLCANO_FROM}")
print("the catalogue holds", count_at_least(vei, 7))
""", """
assert not isinstance(vei_predicted, tuple), \\
    "predict_count hands back two things — unpack both: vei_slope, vei_predicted = predict_count(...)"
assert 0.5 < vei_predicted < 2, \\
    ("a VEI 7 count outside 0.5 to 2 means something went into predict_count in the wrong slot — "
     "the order is the values, then the levels, then the target")
print("✓ the volcano line — it expects", round(vei_predicted, 2),
      "at VEI 7 and the catalogue holds", count_at_least(vei, 7))
""")

code("""
all_levels = np.arange(0, 8)

vei_counted = []
vei_on_the_line = []
for level in all_levels:
    vei_counted.append(count_at_least(vei, level))
    vei_on_the_line.append(predict_count(vei, vei_levels, level)[1])
    print("VEI", level, " counted", vei_counted[-1],
          "  the line says", round(vei_on_the_line[-1], 2))
""")

code(f"""
plt.scatter(all_levels, vei_counted, label="counted")
plt.plot(all_levels, vei_on_the_line, color="firebrick", label="the line fitted to VEI 2, 3, 4")
plt.yscale("log")
plt.xlabel("Volcanic Explosivity Index")
plt.ylabel("number of eruptions at or above this VEI")
plt.title("Eruptions since {VOLCANO_FROM}, {M['n_recent']:,} of them rated")
plt.legend()
plt.show()
""")

md(f"""
Above VEI 4 the line is uncanny. It was fitted to three points and never saw the rest, and it puts
{M['vei_line'][5]} at VEI 5 where there are {M['vei_counts'][5]}, {M['vei_line'][6]} at VEI 6 where
there are {M['vei_counts'][6]}, and {M['vei_pred7']} at VEI 7 where there is {M['vei_obs7']}.

Below VEI 2 it is a disaster, and the log axis is what makes the size of the disaster visible: the
line calls for {M['vei_line'][0]:,.0f} eruptions at VEI 0 or above and the catalogue holds
{M['vei_counts'][0]:,}. Taken literally that says more than nine in ten of the world's smallest
eruptions never reached anybody's records; taken carefully it says the record cannot be trusted
down there at all, which is the same conclusion and a safer way to say it. Either way the missing
thing is the data, not the line.

## What one observation can settle

{M['vei_pred7']} predicted against {M['vei_obs7']} observed looks like a triumph, and this is
exactly the moment to be suspicious. A prediction is only impressive if it could have been
embarrassed, so ask what would have counted as a failure here: if the true rate of VEI 7 eruptions
were something quite different, how often would {M['span']} years still hand you exactly one?

Rare independent events arrive with Poisson counts, so simulate. Twenty thousand imaginary
histories at each of several rates, counting how often each one produces exactly {M['vei_obs7']}.
""")

code(f"""
rng = np.random.default_rng(88)

for rate in {POISSON_RATES}:
    histories = rng.poisson(rate, size=20000)
    print("if the true rate were", rate, "per", {M['span']}, "years, exactly 1 happens in",
          round((histories == 1).mean() * 100), "% of histories")

top = rated[rated["ExplosivityIndexMax"] >= 7]
oldest = int(top["StartDateYear"].min())
print()
print("the fitted rate is one VEI 7 every", round({M['span']} / vei_predicted), "years")
print("the whole catalogue holds", len(top), "of them, the oldest starting", oldest,
      "— one every", round((2026 - oldest) / len(top)), "years")
""")

md(f"""
Every rate from {POISSON_RATES[0]} to {POISSON_RATES[-1]} produces exactly one eruption in a
respectable fraction of histories — between {M['poisson_lo'] * 100:.0f} and
{M['poisson_hi'] * 100:.0f} per cent. A single count cannot tell those apart, so a model predicting
{M['vei_pred7']} and a model predicting {POISSON_RATES[-1]} both "pass" this test, and so would a
model wrong by a factor of forty. The check has almost no power to fail, which means passing it
carries almost no information.

And the wider catalogue says the same thing from the other direction. It holds {M['n_top_all']} VEI
7 eruptions in total, the oldest beginning {abs(M['oldest_top']):,} BCE, which over that span is one
every {M['top_rate']:,} years — around {round(M['top_rate'] / M['one_per'])} times rarer than the
post-{VOLCANO_FROM} fit says. The old record is certainly missing some, so the truth is somewhere
between. Notice which case is which: the earthquake prediction failed loudly and the volcano
prediction passed quietly, and it is the loud failure that told us something.
""")

md(f"""
{weekkit.CLOSING_HEADING}

**Roughly once every few hundred to a couple of thousand years, and this week's data cannot do
better than that.** Fitting the Gutenberg–Richter line to eruptions of VEI 2 to 4 since
{VOLCANO_FROM} predicts {M['vei_pred7']} eruptions at VEI 7 in that window — one every
{M['one_per']} years — and exactly {M['vei_obs7']} occurred. Across the whole catalogue the same
kind of event has come every {M['top_rate']:,} years. Both numbers are defensible and they disagree
by a factor of {round(M['top_rate'] / M['one_per'])}, because one rests on a single event and the
other on a record that thins as it goes back. The line itself is the solid part, and for one reason
only: fitted to VEI 2, 3 and 4, it reproduces VEI 5 and VEI 6 without ever being shown them, which
is a test it could have failed. Its R squared is not such a test — magnitudes with no pattern in
them scored {M['r2_flat']} on the same fit — and neither is landing near a count of
{M['vei_obs7']}. Where the line was tested by something that could have refuted it, it duly was
refuted: California's magnitude 7s outnumber what it expects by a factor of {M['under_by']}.
""")

# --- summary and homework --------------------------------------------------
md(weekkit.week_cheatsheet(7))

md(f"""
## Homework

Three parts, on the same two catalogues. Part 1 goes after the first of the three explanations class
left open for California; part 2 makes you choose a window for the volcanoes and live with the
consequence; part 3 puts the two together. If you have restarted since class, run the setup cell at
the top first, then the checkpoint below.
""")

code(weekkit.CHECKPOINT.format(body=f"""# Re-run your own count_at_least (Your turn 2) and predict_count (Your turn 4) cells as well.
# Those two are your code, so this cell cannot rebuild them for you; every part below uses them.
rated = eruptions.dropna(subset=["ExplosivityIndexMax"])
vei_levels = np.array([2, 3, 4])"""))

ask(f"""
### ✏️ Your turn 6

Class found {M['n_big']} earthquakes of magnitude 7 or above in {WINDOW} years, against a line that
expected {M['pred_main']}. The first explanation on the list was that {WINDOW} years is too short a
window and we drew an unlucky one. That is checkable: count the other windows.

`big_history` holds every magnitude 7 and above the catalogue has for the same box back to
{HIST_START[:4]} — {M['n_history']} of them. Get the year of each with
`big_history["time"].str[:4].astype(int)`, then loop over
`window_starts = range({WINDOW_STARTS[0]}, 2026, {WINDOW})` and for each one count the events with
year at or after `start` and before `start + {WINDOW}`. Collect the counts in a list called
`window_counts`, print each window with its count, and print the mean.

**Use these names**, because the self-check looks for them: `window_counts`, `big_history`.
""")

answer(f"""
years = big_history["time"].str[:4].astype(int)

window_counts = []
for start in range({WINDOW_STARTS[0]}, 2026, {WINDOW}):
    in_window = (years >= start) & (years < start + {WINDOW})
    window_counts.append(in_window.sum())
    print(start, "to", start + {WINDOW}, ":", in_window.sum())

print("mean over the windows:", round(np.mean(window_counts), 2))
""", """
assert sum(window_counts) == len(big_history), \\
    "every event should land in exactly one window — check for < rather than <= at the top edge"
print("✓ six windows — counts", np.array(window_counts), "with a mean of",
      round(np.mean(window_counts), 2))
""")

ask(f"""
### ✏️ Your turn 7

Class fitted the volcano line to eruptions since {VOLCANO_FROM}. That start year was a choice, and
two others are just as defensible: {FORK_YEARS[0]}, which buys more eruptions at the cost of a
thinner record, and {FORK_YEARS[1]}, which buys a better record at the cost of fewer eruptions.

For each of `{FORK_YEARS}`: cut `rated` down to eruptions with `StartDateYear` at or after that
year, call `predict_count` on their `ExplosivityIndexMax` values with `vei_levels` and target 7, and
print the start year, how many eruptions the window holds, the predicted number at VEI 7, and — with
`count_at_least` — how many actually occurred in that window. Collect the predictions in a list
called `fork_predictions`.

**Use these names**, because the self-check looks for them: `fork_predictions`, `rated`,
`vei_levels`.
""")

answer(f"""
fork_predictions = []
for start_year in {FORK_YEARS}:
    window = rated[rated["StartDateYear"] >= start_year]
    window_vei = window["ExplosivityIndexMax"].values
    fork_slope, fork_predicted = predict_count(window_vei, vei_levels, 7)
    fork_predictions.append(fork_predicted)
    print("since", start_year, ":", len(window_vei), "eruptions,",
          "the line expects", round(fork_predicted, 2), "at VEI 7,",
          "and", count_at_least(window_vei, 7), "occurred")
""", """
assert len(fork_predictions) == 2, "two start years, two predictions"
assert fork_predictions[0] != fork_predictions[1], \\
    "identical predictions mean the same eruptions went in twice — check the cut inside the loop"
print("✓ the window is a choice — the two start years predict",
      round(fork_predictions[0], 2), "and", round(fork_predictions[1], 2), "VEI 7 eruptions")
""")

ask(f"""
### ✏️ Your turn 8

Two or three sentences, using your own numbers from parts 1 and 2, on this question: **does anything
you computed rescue the California line, and does anything you computed threaten the volcano line?**

Quote the six window counts and their mean against the {M['pred_main']} the line expected, and quote
both of your part 2 predictions against the counts that went with them. Do not answer from the
summary table; answer from your output.
""")

answer_prose(f"""
The six windows hold {', '.join(str(c) for c in M['window_counts'])} earthquakes of magnitude 7 and
above, a mean of {M['window_mean']}. So {M['busiest_window']}-{M['busiest_window'] + WINDOW} really
was the busiest of the six, and part of class's shortfall was an unlucky window — but only part of
it, because the long-run mean of {M['window_mean']} is still about
{round(M['window_mean'] / M['pred_main'], 1)} times the {M['pred_main']} the line expected, and the
older windows lean on historical accounts rather than instruments, so if anything they undercount.
Something beyond bad luck is missing from the extrapolation. The volcano line goes the other way and
is worse off for it: starting from {FORK_YEARS[0]} it expects {M['fork'][FORK_YEARS[0]]['pred']} VEI
7 eruptions and {M['fork'][FORK_YEARS[0]]['obs']} occurred, starting from {FORK_YEARS[1]} it expects
{M['fork'][FORK_YEARS[1]]['pred']} and {M['fork'][FORK_YEARS[1]]['obs']} occurred, and both of those
"agree" as comfortably as the {M['vei_pred7']} did in class. A prediction that agrees with every
window, including the one holding no eruptions at all, is not being tested by any of them.
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
        print(r.stderr[-4000:])
        sys.exit("the solution did not execute")

    (OUT / f"{SLUG}.ipynb").write_text(json.dumps(stu, indent=1) + "\n")
    print(f"wrote {SLUG}.ipynb ({len(CELLS)} cells) and the executed solution")
    for name in (EQ_CACHE, HIST_CACHE, GVP_CACHE):
        print(f"cache: data/{name}")


if __name__ == "__main__":
    main()
    weekkit.gate(7)
