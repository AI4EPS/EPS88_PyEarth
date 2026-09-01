#!/usr/bin/env python
"""Build week 3 — "Earth's elevation has two peaks. So does Mars's. Same reason?"

Emits both notebooks from one source so they cannot drift:

    docs/notebooks/03_two_peaks_solution.ipynb   executed, every output saved
    docs/notebooks/03_two_peaks.ipynb            the same file with the answers deleted

It also writes the week's cached fallback for the one live query (the USGS catalogue). The two
elevation grids were pre-built by tools/make_elevation_grids.py and are already in data/; this
script never rebuilds them.

Every number that appears in prose or in a model answer is computed HERE, from the same files
the notebook reads, and formatted in. Nothing is typed from memory or copied from the plan.

    python tools/build_week03.py
"""
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/notebooks"
SLUG = "03_two_peaks"

course = yaml.safe_load((ROOT / "course.yml").read_text())
WEEK = next(s for s in course["schedule"] if s["n"] == 3)
PLATFORM = course["platform"]
CACHE_BASE = PLATFORM["cache_base"]

# The one live query this week runs. Pinned here so the cached CSV, the notebook and the prose
# below cannot drift apart.
START, END, MINMAG = "2000-01-01", "2026-01-01", 5.5
CACHE_NAME = f"week03_{START}_{END}_M{MINMAG}.csv"
FDSN = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&orderby=time-asc"
        f"&starttime={START}&endtime={END}&minmagnitude={MINMAG}")

BINS = np.arange(-10000, 21000, 250)      # 250-metre bins, the same ones for both planets


# ---------------------------------------------------------------------------
# 1. measure everything the notebook will say
# ---------------------------------------------------------------------------
def read_grid(planet):
    """Read one shipped elevation CSV the same way the notebook does."""
    return pd.read_csv(ROOT / f"data/{planet}_elevation.csv", header=None).values


def peak_position(grid, lowest, highest):
    """The centre of the tallest 250-metre bin between two elevations."""
    counts, edges = np.histogram(grid.ravel(), bins=BINS)
    centres = (edges[:-1] + edges[1:]) / 2
    inside = (centres >= lowest) & (centres <= highest)
    return centres[inside][counts[inside].argmax()]


def humps(grid, n_bins):
    """How many separate humps a histogram of this grid shows at n_bins bins."""
    counts, _ = np.histogram(grid.ravel(), bins=n_bins)
    return sum(1 for i in range(len(counts))
               if (i == 0 or counts[i] > counts[i - 1])
               and (i == len(counts) - 1 or counts[i] > counts[i + 1]))


def fetch_catalogue():
    """Run the live query once, cache it, and return it."""
    out = ROOT / "data" / CACHE_NAME
    if not out.exists():
        pd.read_csv(FDSN).to_csv(out, index=False)
    return pd.read_csv(out)


earth = read_grid("earth")
mars = read_grid("mars")
coast = pd.read_csv(ROOT / "data/coastlines.csv")
quakes_full = fetch_catalogue()

M = {}                                     # every measured number the notebook uses
M["n_cells"] = int(earth.size)
M["earth_min"], M["earth_max"] = int(earth.min()), int(earth.max())
M["mars_min"], M["mars_max"] = int(mars.min()), int(mars.max())
M["mars_max_km"] = round(M["mars_max"] / 1000, 3)
M["earth_deep"] = int(peak_position(earth, -10000, -1000))
M["earth_high"] = int(peak_position(earth, -1000, 21000))
M["mars_deep"] = int(peak_position(mars, -10000, -1000))
M["mars_high"] = int(peak_position(mars, -1000, 21000))
M["earth_gap"] = M["earth_high"] - M["earth_deep"]
M["mars_gap"] = M["mars_high"] - M["mars_deep"]

below = earth < 0
M["n_below"] = int(below.sum())
M["frac_below"] = round(float(M["n_below"] / earth.size), 3)
band = (earth >= -5000) & (earth <= -3000)
M["frac_band"] = round(float(band.sum() / earth.size), 3)

lats = np.arange(89.5, -90, -1)
cell_width = np.cos(np.deg2rad(lats))
M["frac_below_weighted"] = round(
    float((below.sum(axis=1) * cell_width).sum() / (360 * cell_width.sum())), 3)

BIN_SWEEP = [3, 4, 5, 6, 8, 10, 20]
M["earth_fewest_bins"] = min(n for n in BIN_SWEEP if humps(earth, n) >= 2)
M["mars_fewest_bins"] = min(n for n in BIN_SWEEP if humps(mars, n) >= 2)


def bin_width(grid, n_bins):
    """How wide one bin is, in metres, when a grid is cut into n_bins of them."""
    _, edges = np.histogram(grid.ravel(), bins=n_bins)
    return int(round(edges[1] - edges[0]))


# The homework's part-1 answer turns on bin WIDTH rather than bin count, so the widths either
# side of each planet's merge are measured here instead of read off a printout by hand.
M["earth_range"] = M["earth_max"] - M["earth_min"]
M["mars_range"] = M["mars_max"] - M["mars_min"]
M["earth_width_keeps"] = bin_width(earth, M["earth_fewest_bins"])
M["earth_width_loses"] = bin_width(earth, BIN_SWEEP[BIN_SWEEP.index(M["earth_fewest_bins"]) - 1])
M["mars_width_keeps"] = bin_width(mars, M["mars_fewest_bins"])
M["mars_width_loses"] = bin_width(mars, BIN_SWEEP[BIN_SWEEP.index(M["mars_fewest_bins"]) - 1])

# The third, much smaller bump on the right of Earth's histogram. The prose beside the class
# figure used to say the histogram has two humps; this measures the third one instead.
counts_250, edges_250 = np.histogram(earth.ravel(), bins=BINS)
centres_250 = (edges_250[:-1] + edges_250[1:]) / 2
high_ground = (centres_250 >= 2000) & (centres_250 <= 4000)
i_third = int(np.arange(len(centres_250))[high_ground][counts_250[high_ground].argmax()])
M["third_centre"] = int(centres_250[i_third])
M["third_count"] = int(counts_250[i_third])
M["n_exact_zero"] = int((earth == 0).sum())

lat_grid = np.repeat(lats[:, None], 360, axis=1)                 # centre latitude of every cell
lon_grid = np.repeat(np.arange(-179.5, 180)[None, :], 180, axis=0)
in_third = (earth >= edges_250[i_third]) & (earth < edges_250[i_third + 1])
M["third_antarctica"] = int((in_third & (lat_grid <= -60)).sum())
M["third_greenland"] = int((in_third & (lat_grid >= 59.5) & (lat_grid <= 84)
                            & (lon_grid >= -75) & (lon_grid <= -10)).sum())

# What area weighting does to that bump, and to the two peak positions — the homework's model
# answer quotes both, so both are measured rather than asserted.
weights = np.repeat(cell_width[:, None], 360, axis=1).ravel()
wcounts, _ = np.histogram(earth.ravel(), bins=BINS, weights=weights)
M["third_weighted"] = int(round(wcounts[i_third]))
M["third_shrink"] = int(round(M["third_count"] / wcounts[i_third]))


def weighted_peak(lowest, highest):
    """The same peak position, counting each cell by its width instead of one apiece."""
    inside = (centres_250 >= lowest) & (centres_250 <= highest)
    return int(centres_250[inside][wcounts[inside].argmax()])


M["earth_deep_weighted"] = weighted_peak(-10000, -1000)
M["earth_high_weighted"] = weighted_peak(-1000, 21000)
BIN_WIDTH = int(BINS[1] - BINS[0])
assert abs(M["earth_deep_weighted"] - M["earth_deep"]) <= BIN_WIDTH, "peaks move more than a bin"
assert abs(M["earth_high_weighted"] - M["earth_high"]) <= BIN_WIDTH, "peaks move more than a bin"
assert M["third_shrink"] == 7, "the homework prose says sevenfold; it no longer is"
assert (wcounts[i_third] > wcounts[i_third - 1]
        and wcounts[i_third] > wcounts[i_third + 1]), "the third bump is no longer a local max"

M["n_quakes"] = len(quakes_full)
M["n_columns"] = len(quakes_full.columns)
M["n_dmin_missing"] = int(quakes_full["dmin"].isna().sum())
M["n_after_dropna"] = len(quakes_full.dropna())
kinds = quakes_full["type"].value_counts()
M["n_earthquake_rows"] = int(kinds["earthquake"])
M["n_other_rows"] = int(len(quakes_full) - kinds["earthquake"])
M["n_big"] = int((quakes_full["mag"] >= 8.0).sum())
M["biggest_mag"] = float(quakes_full["mag"].max())
M["n_deep"] = int((quakes_full["depth"] > 500).sum())
M["deepest"] = round(float(quakes_full["depth"].max()), 1)

years = quakes_full["time"].str[:4]
per_year = quakes_full.assign(year=years).groupby("year")["mag"].count()
M["n_years"] = int(len(per_year))
M["busiest_year"] = str(per_year.idxmax())
M["busiest_count"] = int(per_year.max())
M["year_before"] = int(per_year.loc[str(int(M["busiest_year"]) - 1)])
M["year_after"] = int(per_year.loc[str(int(M["busiest_year"]) + 1)])
M["median_year"] = float(per_year.median())
# The year used as the syntax example in Your turn 8 must be an unremarkable one, or it reads as
# the answer to the question it sits in: the ordinary year closest to the median.
ordinary = per_year.drop([M["busiest_year"], str(int(M["busiest_year"]) - 1),
                          str(int(M["busiest_year"]) + 1)])
M["example_year"] = str((ordinary - per_year.median()).abs().idxmin())


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
    """A code answer cell. The solution carries the model answer; the student gets the stub.

    A self-check, where one is possible, lives in the SAME cell after the answer, so that it
    survives into the student copy without costing a third cell per question.
    """
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
Ask how high every point on a planet's solid surface is, and you might expect the answers to
pile up around one typical height, the way people's heights do. Earth's do not. Earth's surface
has two levels, with a gap between them that almost nothing sits in. That is a strange thing for
a planet to be, and it wants an explanation.

Mars has two levels as well. Mars has no ocean, no plate tectonics and no mid-ocean ridge — so
whatever built its two levels cannot be the thing that built Earth's. Or can it?

Today both planets arrive as a grid of numbers: one elevation for every one-degree square of the
surface. You will measure where each planet's two levels sit, draw them on a map, and decide
whether one story explains both. Then you will meet the other container Python keeps data in —
the table with named columns — on a catalogue of earthquakes.
"""

md(weekkit.OPENING.format(question=WEEK["question"], datahub=datahub, hook=HOOK.strip()))

md("""
## What you'll be able to do

**The science.** Say where each planet's two levels sit, in metres, and how far apart they are.
Explain Earth's two levels from the two kinds of crust it is made of, and say what is settled
and what is still argued about Mars's.

**The skills.** A grid of numbers is a **numpy array**: `.shape` to see how big it is,
`.ravel()` to lay it out in one line, `earth < 0` to ask one question of every cell at once,
`np.histogram` to count what falls where, and `plt.imshow` to draw the whole grid as a picture.
A table with named columns is a **pandas DataFrame**: `.info()`, `.head()`, `.isna()`,
`.value_counts()`, `.sort_values()` and `.groupby()`.

**Eight places where you write something: five in class, three at home.** Each one is headed
*Your turn*, with an empty cell under it. All three homework parts ask for a number and then for a
sentence about it, so those have a second cell as well.

1. Where do Earth's two levels sit?
2. What are those two levels made of?
3. Does the same story work for Mars?
4. Where does old ocean floor end up?
""")

setup = weekkit.setup_cell(
    imports="import numpy as np\n",
    figsize="(7, 4)",
    cache_base=CACHE_BASE,
    # NO PARAMETERS. This week reads ONE window and reads it at one call site, so `start`, `end`
    # and `minmag` were three names a student had to carry for a query that never varies -- and
    # week 3 is early enough that every name costs. Weeks 1, 5 and 12 keep the parameter form
    # because the STUDENT changes the query there, which is the teaching; here nobody does.
    signature="",
    docstring="Fetch this week's window of the USGS earthquake catalogue; fall back to the "
              "cached copy.",
    url_expr=('"https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&orderby=time-asc"\n'
              f'                       "&starttime={START}&endtime={END}&minmagnitude={MINMAG}"'),
    cache_expr=f'"{CACHE_NAME}"',
    unpack=f'''
def elevation(planet):
    """Read one planet's 1-degree elevation grid: row 0 is the north, column 0 is -180 degrees."""
    return pd.read_csv(CACHE + "/" + planet + "_elevation.csv", header=None).values


# These three files live in this repository, so CACHE is their home rather than their fallback:
# there is no live server to try first. The catalogue below is the live read.
earth = elevation("earth")
mars = elevation("mars")
coast = pd.read_csv(CACHE + "/coastlines.csv")

quakes = load()
bins = np.arange(-10000, 21000, 250)       # 250-metre bins, the same ones for both planets
print("elevation grids:", earth.shape, mars.shape, " catalogue:", quakes.shape)
'''.strip("\n"))
code(setup)

# --- section 1a: a grid, not a list ------------------------------------------
md("""
## 1. Where do Earth's two levels sit?

Every elevation on this planet, one number per one-degree square, is 64,800 numbers. (Earth's
come from NOAA's ETOPO global relief model, Mars's from the MOLA laser altimeter that flew on
Mars Global Surveyor; both were averaged down to one degree so that the files are small enough to
hand round.) A Python list can hold them. But watch what a list does when you ask it for
arithmetic.
""")

code("""
heights_list = [0, 1000, 2000]
heights_array = np.array([0, 1000, 2000])

print("list  times 2:", heights_list * 2)
print("array times 2:", heights_array * 2)
""")

md("""
The list did not double anything — it made a longer list, with the same three numbers twice.
That is what `*` means for a list. What you meant is what an **array** does. An array is
a grid of numbers where every cell is the same kind of thing, so one line of arithmetic changes
all of them at once.

That is why the elevation data is an array. And because an array can be two-dimensional, it
keeps the *shape* of the planet: one row per line of latitude, one column per line of longitude.
Row 0 of both files is the northernmost band and column 0 is the westernmost, so the grid is
already the right way up — nothing needs turning over.
""")

code("""
print("shape:", earth.shape)
print("cells:", earth.size)
print("lowest cell: ", earth.min(), "m")
print("highest cell:", earth.max(), "m")
""")

ask("""
### ✏️ Your turn 1

The Mars grid is already loaded as `mars`. Print its shape, and print its highest cell in
**kilometres** rather than metres — one line of array arithmetic, then `.max()`.

**Use these names**, because the self-check looks for them: `mars_km` for the grid in kilometres.
""")

answer("""
mars_km = mars / 1000

print("shape:", mars.shape)
print("highest cell:", round(mars_km.max(), 3), "km")
""", """
assert mars_km.shape == mars.shape, "mars_km should be the whole grid, not one number"
print("\u2713 Mars in kilometres \u2014 the highest cell is",
      round(mars_km.max(), 3), "km above the zero level")
""")

# --- section 1b: where the two levels sit ------------------------------------
md("""
A histogram wants one long line of numbers, not a grid. `.ravel()` reads the grid row by row and
lays it out flat — the same 64,800 numbers, in one line instead of 180 of them.
""")

code("""
flat = earth.ravel()
print(flat.shape)
""")

md("""
Before the histogram, one number. Comparing an array with a number asks the same question of
every cell at once and hands back a grid of True and False. That is a **mask**, and adding one up
counts the Trues, because Python counts True as 1.

### Predict before you run

What fraction of Earth's solid surface lies below sea level? Commit to a number before you run
the next cell — change `my_guess` to whatever you think, then run it.
""")

CELLS.extend(("code", s, a) for s, a in
             weekkit.predict_cell("0.70", "of Earth's solid surface lies below sea level"))

code("""
below = earth < 0
fraction_below = below.sum() / earth.size

print("you guessed:", my_guess)
print("this grid says:", f"{fraction_below:.3f}")
""")

md(f"""
{M['frac_below']:.3f} of the cells. Write that down, and be suspicious of it: what you measured is
a fraction of *cells*, and whether a fraction of cells is the same as a fraction of the planet is
a real question. The second part of the homework settles it, and the answer moves.

Now the whole distribution rather than one number. We fix the bins at 250 metres wide and use the
same ones for both planets, so that a peak position means the same thing every time we quote one.
Left to itself `plt.hist` picks its own bins, and the answer you read off moves when it does —
which is the first part of the homework.
""")

code(f"""
plt.hist(flat, bins=bins)
plt.xlim(-9500, 7000)
plt.xlabel("elevation (m)")
plt.ylabel("number of 1-degree cells")
plt.title("Earth, {M['n_cells']:,} cells, 250 m bins")
plt.show()
""")

md(f"""
Two big humps with a thinly populated gap between them: a broad one a few kilometres down, and a
taller, narrower one sitting right at zero. (Only {M['n_exact_zero']} of the {M['n_cells']:,} cells
are exactly 0 m, so that spike is real low ground, not the grid rounding anything to sea level.)
There is a third, far smaller bump out on the right, near {M['third_centre']} m, and it is worth
knowing what it is: {M['third_antarctica']:,} of the {M['third_count']:,} cells in that bin lie
south of 60 degrees, so it is the Antarctic ice sheet, with {M['third_greenland']} more from
Greenland. Two levels is the story of this planet's crust; that bump is the story of where its ice
is, and the homework asks how much of its size is the planet and how much is the grid.

Reading the two main positions off the picture by eye is guesswork, so count instead:
`np.histogram` does the same counting `plt.hist` does but hands you the numbers. `counts[i]` is how many cells fell in bin `i`, and `edges` holds the bin
boundaries, so the middle of bin `i` is halfway between `edges[i]` and `edges[i+1]`.

`.argmax()` gives the *position* of the largest value — the same move as week one's
`list.index(max(list))`, in one word.
""")

code("""
counts, edges = np.histogram(flat, bins=bins)
centres = (edges[:-1] + edges[1:]) / 2

print("tallest bin is centred at", centres[counts.argmax()], "m")
print("and holds", counts.max(), "cells")
""")

ask("""
### ✏️ Your turn 2

That found the taller hump. The other one needs the same three lines applied to a *slice* of the
bins, so write it once as a function and use it twice.

Write `peak_position(grid, lowest, highest)`: it should histogram `grid` on `bins`, keep only the
bins whose centre lies between `lowest` and `highest`, and return the centre of the tallest one
that is left. Give it a docstring. Then print Earth's deep peak (search between -10000 and -1000)
and Earth's shallow peak (search between -1000 and 21000).

**Use these names**, because the self-check looks for them: `peak_position`, `earth_deep`,
`earth_high`.
""")

answer("""
def peak_position(grid, lowest, highest):
    \"\"\"The centre of the tallest 250-metre bin between two elevations.\"\"\"
    counts, edges = np.histogram(grid.ravel(), bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    inside = (centres >= lowest) & (centres <= highest)
    return centres[inside][counts[inside].argmax()]


earth_deep = peak_position(earth, -10000, -1000)
earth_high = peak_position(earth, -1000, 21000)

print("Earth's deep level:   ", earth_deep, "m")
print("Earth's shallow level:", earth_high, "m")
""", """
assert -6000 < earth_deep < -3000, \\
    "earth_deep should be an elevation in metres, not the height of a bin"
assert -500 < earth_high < 1000, \\
    "earth_high should be an elevation in metres, not the height of a bin"
print("\u2713 Earth's two levels \u2014", earth_deep, "m and", earth_high, "m,",
      earth_high - earth_deep, "m apart")
""")

# --- section 2 -------------------------------------------------------------
md("""
## 2. What are those two levels made of?

The mask you built two cells ago has one `True` or `False` per one-degree square — which is to
say, it is already a map. `plt.imshow` draws any grid as a picture, one pixel per cell, and
`extent` tells it what the corners mean in degrees. The coastline goes on top from
`data/coastlines.csv`, exactly as in week one, so you can see whether the mask agrees with it.
""")

code(weekkit.CHECKPOINT.format(body="""earth = elevation("earth")
below = earth < 0
# Re-run your own Your turn 2 cell as well, the one that defines peak_position and
# sets earth_deep and earth_high. That code is yours, so this cell cannot rebuild it
# for you."""))

code(f"""
plt.imshow(below, extent=[-180, 180, -90, 90], cmap="Greys")
plt.plot(coast.lon, coast.lat, color="firebrick", lw=0.6)
plt.xlabel("degrees east")
plt.ylabel("degrees north")
plt.title("Earth: the {M['n_below']:,} cells below sea level, of {M['n_cells']:,}")
plt.show()
""")

md(f"""
The dark region is not a shape anyone had to draw: it is one comparison, `earth < 0`, applied to
every cell. It comes out as the oceans, and the coastline lands on its edge.

So the deep hump in the histogram, centred near {M['earth_deep']} m, is the ocean floor — the
abyssal plains — and the shallow hump near {M['earth_high']} m is low-lying land. The reason
those are two levels rather than one is that Earth is made of two kinds of crust. Ocean crust is
basalt, thin (a few kilometres) and dense; continental crust is granitic, far thicker (tens of
kilometres) and less dense. Both float on the mantle, and a thick light raft floats higher than a
thin heavy one, so the two kinds settle at two different heights. Plate tectonics keeps the
arrangement going: ocean crust is manufactured at mid-ocean ridges and destroyed at subduction
zones within a couple of hundred million years, while the light continental crust is too buoyant
to sink and stays. Water then fills the low level, which is why the boundary between the two
humps sits so close to sea level.

Note what that map is *not* good at. Every cell is drawn the same size, but a one-degree square
at 60° north is only half as wide, east to west, as one at the equator — `cos(60°) = 0.5` — and
at the poles the width goes to nothing. Antarctica along the bottom of the map is stretched
across far more pixels than it deserves. Hold that thought too.
""")

# --- section 3a: Mars -------------------------------------------------------
md("""
## 3. Does the same story work for Mars?

The Mars grid is the same shape as Earth's — 180 by 360, one cell per one-degree square — and it
was measured by a laser altimeter in orbit, which is why it exists at all. Zero on Mars is not a
sea level; there is no sea. It is a reference surface chosen by geodesists, so instead of a
below-and-above mask we colour the whole range. The colour scale is the one normally used for
topography, so the blue end means nothing more than *low* — there is no water on this map.
""")

code(weekkit.CHECKPOINT.format(body="""earth = elevation("earth")
mars = elevation("mars")
flat = earth.ravel()
# Re-run your own Your turn 2 cell as well, the one that defines peak_position and
# sets earth_deep and earth_high. That code is yours, so this cell cannot rebuild it
# for you."""))

code(f"""
# the colour scale stops at 4000 m, or the step between the two halves is invisible
plt.imshow(mars, extent=[-180, 180, -90, 90], cmap="terrain", vmin=-4000, vmax=4000)
plt.colorbar(label="elevation (m)")
plt.xlabel("degrees east")
plt.ylabel("degrees north")
plt.title("Mars, {M['n_cells']:,} cells")
plt.show()
""")

md("""
There is nothing subtle about that map. The north of the planet is low and smooth, the south is
high and rough, and the step between them runs most of the way round the planet. Planetary scientists call it the **crustal dichotomy**, and the crust under the northern
lowlands is measurably thinner than the crust under the southern highlands.

What Mars does not have is an ocean to fill the low half, or plates to make and destroy crust.
Whatever put that step there did it long ago and left it, and the surface has kept it ever since.

So: two levels here as well. The next cell puts both planets on the same axis, using the same
250-metre bins, so the shapes can be compared rather than described.
""")

code(f"""
plt.hist(flat, bins=bins, label="Earth")
plt.hist(mars.ravel(), bins=bins, label="Mars", alpha=0.6)   # see-through, or Mars hides Earth
plt.xlim(-9500, 7000)
plt.xlabel("elevation (m)")
plt.ylabel("number of 1-degree cells")
plt.title("Earth and Mars, {M['n_cells']:,} cells each, 250 m bins")
plt.legend()
plt.show()
""")

ask(f"""
### ✏️ Your turn 3

Mars runs off the right of that plot: its highest cell is the {M['mars_max_km']} km you printed in
your turn 1, and the axis stops at 7000 m so that the two distributions are both readable.

Use your `peak_position` function on `mars` — the same two searches you ran for Earth — and then
print how far apart each planet's two levels are.

**Use these names**, because the self-check looks for them: `mars_deep`, `mars_high`.
""")

answer("""
mars_deep = peak_position(mars, -10000, -1000)
mars_high = peak_position(mars, -1000, 21000)

print("Mars's two levels: ", mars_deep, "m and", mars_high, "m")
print("Mars step: ", mars_high - mars_deep, "m")
print("Earth step:", earth_high - earth_deep, "m")
""", """
assert -6000 < mars_deep < -3000, \\
    "mars_deep should be an elevation in metres, not the height of a bin"
assert 500 < mars_high < 2500, \\
    "mars_high should be an elevation in metres, not the height of a bin"
print("\u2713 Mars's two levels \u2014", mars_deep, "m and", mars_high, "m, a step",
      (mars_high - mars_deep) - (earth_high - earth_deep), "m bigger than Earth's")
""")

ask("""
### ✏️ Your turn 4

Same reason? Two or three sentences. Use the four peak positions you measured and the two maps —
what the Earth mask looked like, what the Mars map looked like — and say whether one explanation
covers both planets. If it does not, say what Earth has that Mars does not.
""")

answer_prose(f"""
Both planets really do have two levels: Earth's sit at {M['earth_deep']} m and
{M['earth_high']} m, {M['earth_gap']} m apart, and Mars's at {M['mars_deep']} m and
{M['mars_high']} m, {M['mars_gap']} m apart — a bigger step than Earth's. But the maps do not
match. Earth's low level came out as the oceans, scattered in basins between continents, and
Earth's two levels are two kinds of crust that plate tectonics keeps making and destroying. Mars's
low level is one connected cap over the north, with a single step running round the planet, and
Mars has neither ocean nor plates. So the same explanation cannot cover both: the shape of the
histogram is the same, and the cause is not.
""")

# --- section 3b: where the argument stands ----------------------------------
md("""
Earth's two levels have a settled explanation, the one the map and the histogram gave you: two
kinds of crust, floating at two heights, made and destroyed by plate tectonics. Mars's do not.
Which process put that hemisphere-wide step in the crust is genuinely unsettled, and the two
families of explanation under discussion are a single enormous impact that excavated the northern
lowlands, and a pattern of convection inside the young planet that thinned the crust on one side.
Nobody has closed the argument, so if you found the Mars half less satisfying than the Earth half,
that is not because the notebook left something out.
""")

# --- section 4 -------------------------------------------------------------
md("""
## 4. Where does old ocean floor end up?

Section 2 left half a sentence hanging. Ocean crust is manufactured at the mid-ocean ridges and
destroyed within a couple of hundred million years — destroyed *where*, and how would anybody
watch it happen? Not in an elevation grid. A grid of elevations holds the surface, and this is a
question about what goes on underneath it. The record that can answer it is an earthquake
catalogue, and the last question of this notebook is where you go looking in one.

A catalogue is not a grid. An array is the right container when every number means the same
thing — here, metres of elevation — and position is what tells them apart. A catalogue is not
like that: it has a time, a latitude, a depth, a magnitude and a place name on every row, and
those are five different kinds of thing. What that wants is a **table**: a table with a name on
every column, so you ask for data by name instead of by position. Pandas calls one a DataFrame.

The catalogue below is every earthquake of magnitude 5.5 and above that the USGS has recorded
since the start of 2000. Remember what such a file is: *A catalogue lists what somebody's
instruments recorded, not what happened. Where there are no seismometers there are no earthquakes
in the file.*

`.info()` is the first thing to run on a table you have not seen. It names every column, says
what type it holds, and — the part that matters — says how many rows are not blank.
""")

code("""
print(quakes.shape)
quakes.info()

print("rows with no dmin:", quakes["dmin"].isna().sum())
print("rows left if you drop every row with any blank:", len(quakes.dropna()))
""")

md(f"""
{M['n_quakes']:,} rows and {M['n_columns']} columns. Read down the non-null counts: `time`,
`latitude`, `depth` and `mag` are complete, but several columns are not. Where the file had
nothing at all, pandas puts NaN. A NaN is a hole, not a zero. The difference matters: a depth of
0 km is a real, shallow earthquake, while a NaN depth is an earthquake whose depth nobody
recorded.

`.isna()` marks the holes — it is exactly the mask move from the first half of the notebook, with
`is this blank?` as the question instead of `is this below zero?`. `.dropna()` throws away every
row with a hole anywhere in it, which sounds tidy and is usually a disaster: only
{M['n_after_dropna']:,} rows survive out of {M['n_quakes']:,}, not because those earthquakes are
bad but because a column nothing here needs happens to be patchy. So keep the columns you
actually want, and leave the rest alone.

A new column is made by assigning to a name that does not exist yet. `.str[:4]` slices every
string in a column the way `[:4]` slices one string, so the four characters at the front of
`time` give the year. Then the columns worth keeping are chosen by name, in a list, inside the
square brackets, and `.head()` shows the first five rows.
""")

code("""
quakes["year"] = quakes["time"].str[:4]
quakes = quakes[["year", "depth", "mag", "type", "place"]]

print(quakes.head())
""")

md("""
`.value_counts()` counts how often each value appears in one column — the table version of
counting `True`s in a mask. Run it on `type` and the catalogue tells you something about itself.
""")

code("""
print(quakes["type"].value_counts())
print(quakes[quakes["type"] != "earthquake"])
""")

md(f"""
Not everything in the earthquake catalogue is an earthquake. {M['n_other_rows']} of the
{M['n_quakes']:,} rows are not: seismometers record whatever shakes the ground, and a large enough
explosion or eruption shakes it in much the way an earthquake does. Telling those apart is a
question this course comes back to.

That second line is **boolean filtering**: `quakes["type"] != "earthquake"` is a mask, one True or
False per row, and putting a mask inside the square brackets keeps the rows where it is True. The
grid and the table use the same move. `.sort_values()` then puts the rows in order by a column.
""")

code(f"""
big = quakes[quakes["mag"] >= 8.0]
print(len(big), "earthquakes at magnitude 8.0 or above")
print(big.sort_values("mag", ascending=False).head()[["year", "mag", "place"]])
""")

ask("""
### ✏️ Your turn 5

Earthquakes deeper than 500 km are strange animals: the rock at that depth is far too hot and
squeezed to snap the way it does near the surface, and they only happen where cold ocean floor is
sinking back into the mantle — the far end of the story the first half of this notebook told.

Filter `quakes` to the rows deeper than 500 km, print how many there are, and print the five
deepest in order, deepest first, showing the year, the depth and the place.

**Use these names**, because the self-check looks for them: `deep_quakes`.
""")

answer("""
deep_quakes = quakes[quakes["depth"] > 500]

print(len(deep_quakes), "earthquakes deeper than 500 km")
print(deep_quakes.sort_values("depth", ascending=False).head()[["year", "depth", "place"]])
""", """
assert deep_quakes["depth"].min() > 500, "deep_quakes should hold only the rows below 500 km"
print("\u2713 deep earthquakes \u2014", len(deep_quakes), "of them, the deepest at",
      deep_quakes["depth"].max(), "km")
""")

# --- the closing -----------------------------------------------------------
md(f"""
{weekkit.CLOSING_HEADING}

**No — two levels, two different causes.** Earth's two levels are two kinds of crust, thin dense
ocean floor and thick light continent, floating at two heights and continuously remade by plate
tectonics; Mars's are one hemisphere-wide step in crustal thickness, made once and never reworked,
on a planet with neither ocean nor plates. The table half is the same story from its other end:
the earthquakes you picked out, the ones deeper than 500 km, happen where ocean floor is sinking
back into the mantle, which is the half of the cycle that keeps Earth's low level low.
""")

# --- summary and homework --------------------------------------------------
md(weekkit.week_cheatsheet(3))

md("""
## Homework

Three parts, all on the two grids and the catalogue you already have loaded. Part 1 and part 2 go
back to the elevation grids and finish two arguments class deliberately left open; part 3 stays
with the table. If you have restarted since class, run the setup cell at the top and then the
checkpoint below: between them they rebuild everything the three parts read.
""")

code(weekkit.CHECKPOINT.format(body=f"""earth = elevation("earth")
mars = elevation("mars")
below = earth < 0
fraction_below = below.sum() / earth.size

quakes = load()
quakes["year"] = quakes["time"].str[:4]
quakes = quakes[["year", "depth", "mag", "type", "place"]]"""))

ask(f"""
### ✏️ Your turn 6

Class fixed the bins at 250 metres wide. Loosen that, and the two peaks eventually merge into one
hump — but which way? Find out.

Loop over `bin_counts = {BIN_SWEEP}` and for each one print the bin count and the counts
`np.histogram(earth.ravel(), bins=n)` returns. You are looking for a **dip** between two larger
numbers: that dip is the gap between the two levels, and when it disappears the peaks have
merged. Report the smallest number of bins that still shows two humps for Earth, and then do the
same for Mars.

Then, in one or two sentences in the cell after, say which planet keeps its two humps at the
**smaller** number of bins — and why it is not the planet whose two levels are further apart.
Quote `earth_fewest_bins` and `mars_fewest_bins`, the bin widths in metres your loop printed
beside them, and the two peak separations you measured in class, and say which of those numbers
decides when two humps merge into one.

**Use these names**, because the self-check looks for them: `earth_fewest_bins`,
`mars_fewest_bins`. The check writes its own `still_two_humps` to test the two numbers you report
against the sweep; it does not go looking for them, because the looking is the question.
""")

answer(f"""
bin_counts = {BIN_SWEEP}

for n in bin_counts:
    counts, edges = np.histogram(earth.ravel(), bins=n)
    print("Earth,", n, "bins of", round(edges[1] - edges[0]), "m")
    print(counts)

for n in bin_counts:
    counts, edges = np.histogram(mars.ravel(), bins=n)
    print("Mars,", n, "bins of", round(edges[1] - edges[0]), "m")
    print(counts)

earth_fewest_bins = {M['earth_fewest_bins']}
mars_fewest_bins = {M['mars_fewest_bins']}
""", """

# This CHECKS an answer; it does not find one. The searching is the question, and it is yours.
def still_two_humps(grid, n_bins):
    \"\"\"Does this grid's histogram still dip between two humps at this many bins?\"\"\"
    counts, edges = np.histogram(grid.ravel(), bins=n_bins)
    for i in range(1, len(counts) - 1):
        if counts[i] < counts[:i].max() and counts[i] < counts[i + 1:].max():
            return True
    return False


assert still_two_humps(earth, earth_fewest_bins), \\
    "Earth's dip has already gone at that many bins — read the counts again"
assert not still_two_humps(earth, bin_counts[bin_counts.index(earth_fewest_bins) - 1]), \\
    "the next coarser binning in the sweep still dips, so that is not the SMALLEST that does"
assert still_two_humps(mars, mars_fewest_bins), \\
    "Mars's dip has already gone at that many bins — read the counts again"
assert not still_two_humps(mars, bin_counts[bin_counts.index(mars_fewest_bins) - 1]), \\
    "the next coarser binning in the sweep still dips, so that is not the SMALLEST that does"
print("\u2713 where the peaks merge \u2014 Earth keeps two humps down to", earth_fewest_bins,
      "bins, Mars down to", mars_fewest_bins)
""")

answer_prose(f"""
Earth keeps its two humps down to {M['earth_fewest_bins']} bins and Mars needs
{M['mars_fewest_bins']}, which is the wrong way round if the distance between the levels were what
mattered: Mars's two levels are {M['mars_gap']:,} m apart and Earth's only {M['earth_gap']:,} m.
The number that decides it is the bin **width** my loop printed beside each count, not the count.
Earth's dip survives {M['earth_width_keeps']:,} m bins and is gone by {M['earth_width_loses']:,} m;
Mars's survives {M['mars_width_keeps']:,} m bins and is gone by {M['mars_width_loses']:,} m — the
same window of widths for both planets, which is what you would expect if a gap disappears once a
single bin is wide enough to swallow it. Mars needs twice as many bins to reach that width because
`np.histogram` spreads its bins across the whole range of the data, and Mars's range is
{M['mars_range']:,} m against Earth's {M['earth_range']:,} m — Olympus Mons and the Tharsis rise
reach {M['mars_max_km']} km above the reference surface, so most of those extra bins are spent on
ground that has nothing to do with either level.
""")

ask(f"""
### ✏️ Your turn 7

Class measured that {M['frac_below']:.3f} of the *cells* in the Earth grid are below sea level,
and parked the question of whether that is the fraction of the *planet*. Here is the missing
piece, and it has a name — **area weighting**. A longitude-latitude grid counts every square
once, but a square near the pole is a sliver; weight each row by cos(latitude) before quoting a
percentage. A square at latitude *lat* is `cos(lat)` times as wide, east to west, as one at the
equator.

So count each row of the grid by how wide its cells are instead of by how many there are:

```
lats = np.arange(89.5, -90, -1)          # the centre latitude of each of the 180 rows
cell_width = np.cos(np.deg2rad(lats))    # 1.0 at the equator, nearly 0 at the poles
rows_below = below.sum(axis=1)           # below-sea-level cells in each row
```

`rows_below` and `cell_width` are both 180 numbers long, so `rows_below * cell_width` weights
each row. Divide by what all 180 full rows would weigh — `360 * cell_width.sum()` — to get the
weighted fraction. Then say, in the cell after, which of the two numbers is the honest one and
what the longitude-latitude grid was doing to the poles to produce the other.

**Use these names**, because the self-check looks for them: `earth_weighted`.
""")

answer("""
lats = np.arange(89.5, -90, -1)
cell_width = np.cos(np.deg2rad(lats))
rows_below = below.sum(axis=1)

earth_weighted = (rows_below * cell_width).sum() / (360 * cell_width.sum())

print("counting cells:      ", f"{fraction_below:.3f}")
print("weighting by width:  ", f"{earth_weighted:.3f}")
""", """
assert earth_weighted > fraction_below, \\
    "the weighted fraction should be the bigger one — check what you divided by"
print("\u2713 area weighting \u2014 the fraction below sea level moved from",
      f"{fraction_below:.3f}", "to", f"{earth_weighted:.3f}")
""")

answer_prose(f"""
The weighted figure, {M['frac_below_weighted']:.3f}, is the honest one, because the question was
about the planet's surface and not about the file's rows. Counting cells treats a one-degree
square at the pole as worth the same as one at the equator, and it is not — `cos(89.5°)` is under
0.01, so the top row of cells is a hundredth as wide as the equatorial row. The grid therefore
over-counts high latitudes, and high latitudes are exactly where the two largest pieces of
high-standing land sit: Antarctica and Greenland. Over-counting them pulls the below-sea-level
share down, from {M['frac_below_weighted']:.3f} to {M['frac_below']:.3f}, about
{abs(M['frac_below_weighted'] - M['frac_below']) * 100:.0f} percentage points. The same weights
explain the small third bump near {M['third_centre']} m in the class histogram, the one that is
nearly all Antarctica: weighting shrinks it about sevenfold, from {M['third_count']:,} cells to a
weighted {M['third_weighted']}, and leaves it a local maximum but a much less impressive one.
The two peak positions survive to within one {BIN_WIDTH}-metre bin, because a peak position is an
elevation and not a share of area, so the two-levels story from class is unaffected. Only the
percentage was wrong.
""")

ask(f"""
### ✏️ Your turn 8

Back to the table. `quakes` has a `year` column, so `.groupby("year")` will split the catalogue
into one group per year, and `["mag"].count()` will count the rows in each group.

Build `per_year`, print the three largest years using `.sort_values(ascending=False).head(3)`,
and then print the year before and the year after the busiest one — `per_year.loc["{M['example_year']}"]`
reads one year out — together with `per_year.median()`.

Then, in the cell after, answer this in two or three sentences using your own numbers: was the
planet busier in that year, or did one thing happen? Whatever you claim, the neighbouring years
and the median have to be consistent with it.

**Use these names**, because the self-check looks for them: `per_year`.
""")

answer(f"""
per_year = quakes.groupby("year")["mag"].count()

print(per_year.sort_values(ascending=False).head(3))
print("the year before:", per_year.loc["{int(M['busiest_year']) - 1}"])
print("the year after: ", per_year.loc["{int(M['busiest_year']) + 1}"])
print("median year:    ", per_year.median())
""", f"""
assert per_year.sum() == len(quakes), \\
    "the yearly counts should add up to every row in the catalogue"
print("\u2713 earthquakes by year \u2014 busiest is",
      per_year.sort_values(ascending=False).index[0], "with", per_year.max(),
      "against a median of", per_year.median())
""")

answer_prose(f"""
{M['busiest_year']} holds {M['busiest_count']} earthquakes of magnitude 5.5 and above, against a
median year of {M['median_year']:.1f}, and its immediate neighbours are ordinary:
{M['year_before']} the year before and {M['year_after']} the year after. A planet that had genuinely
become more active would not go back to normal within twelve months. The catalogue itself says
what happened instead: the largest event in the whole file, magnitude {M['biggest_mag']}, is in
{M['busiest_year']}, and a great earthquake is followed by thousands of aftershocks, a few hundred
of which are large enough to clear the magnitude 5.5 floor of this query. So one earthquake
sequence, not a busier planet.
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
    r = weekkit.execute(sol_path, timeout=600)
    if r.returncode:
        print(r.stderr[-4000:])
        sys.exit("the solution did not execute")

    (OUT / f"{SLUG}.ipynb").write_text(json.dumps(stu, indent=1) + "\n")
    print(f"wrote {SLUG}.ipynb ({len(CELLS)} cells) and the executed solution")
    print(f"cache: data/{CACHE_NAME}")


if __name__ == "__main__":
    main()
    weekkit.gate(3)
