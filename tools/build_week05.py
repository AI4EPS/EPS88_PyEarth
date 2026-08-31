#!/usr/bin/env python
"""Build week 5 — "Do earthquakes cluster — or is that just what randomness looks like?"

Emits both notebooks from one source so they cannot drift:

    docs/notebooks/05_clustered_or_random_solution.ipynb   executed, every output saved
    docs/notebooks/05_clustered_or_random.ipynb            the same file with the answers deleted

It also writes the week's two cached fallbacks: the global catalogue class works on, and the
Bay Area catalogue the homework forecasts from.

Every number that appears in prose or in a model answer is computed HERE, from the same files
the notebook reads, and formatted in. Nothing is typed from memory or copied from the plan.

    python tools/build_week05.py
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
SLUG = "05_clustered_or_random"

course = yaml.safe_load((ROOT / "course.yml").read_text())
WEEK = next(s for s in course["schedule"] if s["n"] == 5)
PLATFORM = course["platform"]
CACHE_BASE = PLATFORM["cache_base"]

# The two live queries. Pinned here so the cached CSVs, the notebook and the prose below cannot
# drift apart. The global slice is course.yml's pinned: slice, verbatim.
G_START, G_END, G_MAG = "2000-01-01", "2026-01-01", 5.5
B_START, B_END, B_MAG = "1900-01-01", "2026-01-01", 5.0
BERK_LAT, BERK_LON = 37.87, -122.27          # Berkeley campus
BAY_HALF = 2.0                               # the query box: two degrees each side of campus
BAY_BOX = (f"&minlatitude={BERK_LAT - BAY_HALF}&maxlatitude={BERK_LAT + BAY_HALF}"
           f"&minlongitude={BERK_LON - BAY_HALF}&maxlongitude={BERK_LON + BAY_HALF}")
BAY_YEARS = 126                              # 1900-01-01 to 2026-01-01

SEED = 88                                    # the course number; fixed before anything was run
N_WORLDS = 2000                              # the audit's number of Monte Carlo runs
FDSN = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&orderby=time-asc"


# ---------------------------------------------------------------------------
# 1. measure everything the notebook will say
# ---------------------------------------------------------------------------
def fetch(name, start, end, minmag, region):
    """Run one live query once, cache it beside the course, and return it."""
    out = ROOT / "data" / f"week05_{name}_{start}_{end}_M{minmag}.csv"
    if not out.exists():
        pd.read_csv(f"{FDSN}&starttime={start}&endtime={end}&minmagnitude={minmag}{region}"
                    ).to_csv(out, index=False)
    return pd.read_csv(out)


quakes = fetch("global", G_START, G_END, G_MAG, "")
bay = fetch("bay", B_START, B_END, B_MAG, BAY_BOX)

quakes["time"] = pd.to_datetime(quakes["time"])
daily = quakes.set_index("time")["mag"].resample("D").count()

n_days, n_quakes = len(daily), len(quakes)
day_edges = np.arange(n_days + 1)


def random_world(rng):
    """Scatter n_quakes earthquakes at random over n_days days and count each day."""
    counts, edges = np.histogram(rng.integers(0, n_days, n_quakes), bins=day_edges)
    return counts


M = {}
M["n_quakes"] = n_quakes
M["n_days"] = n_days
M["lam"] = float(daily.mean())
M["busiest"] = int(daily.max())
M["busiest_day"] = str(daily.idxmax())[:10]
M["second"] = int(daily.sort_values(ascending=False).iloc[1])
M["second_day"] = str(daily.sort_values(ascending=False).index[1])[:10]

# one random world, the same seed the notebook uses
M["one_busiest"] = int(random_world(np.random.default_rng(SEED)).max())
M["one_quiet_frac"] = float((random_world(np.random.default_rng(SEED)) == 0).mean())

# the four seeds the model answer to question 2 uses
M["q2_seeds"] = [1, 2, 3, 4]
M["q2_busiest"] = [int(random_world(np.random.default_rng(s)).max()) for s in M["q2_seeds"]]

rng = np.random.default_rng(SEED)
worlds = np.array([random_world(rng).max() for _ in range(N_WORLDS)])
M["sim_mean"] = float(worlds.mean())
M["sim_p95"] = float(np.percentile(worlds, 95))
M["sim_max"] = int(worlds.max())
M["sim_reached"] = int((worlds >= M["busiest"]).sum())
M["ratio"] = M["busiest"] / M["sim_mean"]

# the busiest day itself
day = quakes[(quakes["time"] >= "2011-03-11") & (quakes["time"] < "2011-03-12")]
top = day.sort_values("mag", ascending=False).iloc[0]
M["top_mag"] = float(top["mag"])
M["top_place"] = str(top["place"])
M["top_lat"], M["top_lon"] = round(float(top["latitude"]), 1), round(float(top["longitude"]), 1)
M["near_epicentre"] = int(((abs(day["latitude"] - M["top_lat"]) < 5)
                           & (abs(day["longitude"] - M["top_lon"]) < 5)).sum())
M["after30"] = int(daily.loc["2011-03-11":"2011-04-09"].sum())
M["before30"] = int(daily.loc["2011-02-09":"2011-03-10"].sum())

running = daily.rolling(365).sum()
M["run_min"], M["run_max"] = int(running.min()), int(running.max())
M["run_median"] = int(running.median())
M["run_max_day"] = str(running.idxmax())[:10]

by_year = daily.groupby(daily.index.year).max()
M["n_years"] = int(len(by_year))
M["years_over"] = int((by_year > M["sim_p95"]).sum())
M["quietest_year_max"] = int(by_year.min())

# the Poisson section
M["p_zero"] = float(np.exp(-M["lam"]))
M["expected_quiet"] = round(M["p_zero"] * n_days)
M["real_quiet"] = int((daily == 0).sum())
M["excess_quiet"] = M["real_quiet"] - M["expected_quiet"]
M["p_one_day"] = 1 - M["p_zero"]
M["p_one_hour"] = float(1 - np.exp(-M["lam"] / 24))
M["gap_hours"] = 24 / M["lam"]

# the homework, both sides of the fork
def forecast(half):
    near = bay[(abs(bay["latitude"] - BERK_LAT) <= half)
               & (abs(bay["longitude"] - BERK_LON) <= half)]
    big = near[near["mag"] >= 6.0].sort_values("time")
    rate = len(big) / BAY_YEARS
    gaps = sorted(pd.to_datetime(big["time"]).diff().dt.days.dropna())
    return {"n_near": len(near), "n": len(big), "rate": rate, "recur": 1 / rate,
            "p4": float(1 - np.exp(-rate * 4)), "p30": float(1 - np.exp(-rate * 30)),
            "gaps": [int(g) for g in gaps],
            "places": list(big["place"])}


M["small"] = forecast(1.0)
M["big"] = forecast(2.0)
M["n_bay"] = len(bay)
# the two events the wider box adds that the model answer names; both are printed by forecast()
M["extra"] = [p.replace("The ", "").replace(", California Earthquake", "").strip()
              for p in M["big"]["places"] if p not in M["small"]["places"]]


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
On a map, earthquakes obviously cluster: they draw the plate boundaries, which is what the first
week of this course found. In time it is much harder to tell. Plot every large earthquake of the
last twenty-six years day by day and the record looks bunched — long quiet stretches, then several
in a week — and it is tempting to say so and move on.

The trouble is that bunching is what randomness does. Scatter twelve thousand earthquakes at
random across nine thousand days and you get quiet stretches and busy weeks too, because *random*
does not mean *evenly spread*. So the eye cannot settle this, and no stronger adjective will
either.

Today you build the world where earthquakes happen at random at the real rate, measure how clumpy
that world gets, and compare it with the one we live on. Whatever is left over is physics.
"""

md(weekkit.OPENING.format(question=WEEK["question"], datahub=datahub, hook=HOOK.strip()))

md("""
## What you'll be able to do

**The science.** Say whether earthquakes really are more bunched in time than chance would make
them, by how much, and what the excess is made of. Then turn a catalogue into a forecast: the
chance of an earthquake near campus in the next few years, and what that number quietly assumes.

**The skills.** Dates that Python understands: `pd.to_datetime`, `set_index`, and `resample` to
count events per day. Simulation: `np.random.default_rng` and a seed, so a random experiment gives
the same answer twice, and a loop that runs it two thousand times. And the Poisson formula,
`np.exp`, which gets the same answers without the simulation.

**Eight places where you write something: five in class, three at home.** Each one is headed
*Your turn*, with an empty cell under it.
""")

setup = weekkit.setup_cell(
    imports="import numpy as np\n",
    figsize="(7, 3.6)",
    cache_base=CACHE_BASE,
    signature="name, start, end, minmag, region",
    docstring=("Fetch one window of the USGS catalogue; fall back to the copy stored with the "
               "course.\n\n    `region` is extra query text limiting the search to a box of "
               "latitude and longitude;\n    pass \"\" for the whole world.\n    "),
    url_expr=(f'f"{FDSN}"\n'
              '                       f"&starttime={start}&endtime={end}&minmagnitude={minmag}"\n'
              '                       + region'),
    cache_expr='f"week05_{name}_{start}_{end}_M{minmag}.csv"',
    unpack=f'''
# Two degrees of latitude and longitude each side of Berkeley, which sits at
# {BERK_LAT} north, {BERK_LON} east. You need this one at the end of the notebook.
BAY_BOX = ("{BAY_BOX[:36]}"
           "{BAY_BOX[36:]}")

quakes = load("global", "{G_START}", "{G_END}", {G_MAG}, "")
bay = load("bay", "{B_START}", "{B_END}", {B_MAG}, BAY_BOX)
print("global catalogue:", quakes.shape, " Bay Area catalogue:", bay.shape)
'''.strip("\n"))
code(setup)

# --- section 1 -------------------------------------------------------------
md(f"""
## From a list of earthquakes to a count per day

`quakes` holds every earthquake of magnitude {G_MAG} and above that the USGS recorded worldwide
between the start of {G_START[:4]} and the start of {G_END[:4]}. One row is one earthquake, and the
`time` column says when — but only as far as pandas is concerned it is text, and you cannot count
text by the day.

`pd.to_datetime` turns that column into real dates. `set_index` then makes the time column the row
labels, so the table knows *when* each row happened rather than just holding a column about it.
Once it does, `resample("D")` regroups the rows into fixed windows — `"D"` for one day each — and
`.count()` says how many fell in each.
""")

code("""
quakes["time"] = pd.to_datetime(quakes["time"])
daily = quakes.set_index("time")["mag"].resample("D").count()

print(daily.head())
print("days covered:      ", len(daily))
print("earthquakes:       ", daily.sum())
print("average per day:   ", round(daily.mean(), 3))
""")

md(f"""
{M['n_quakes']:,} earthquakes over {M['n_days']:,} days, so a bit over one a day on average. Every
day is in there, including the ones with nothing on them — that is what `resample` gives you and
it is what we want, because a day with no earthquake is data too.

Now draw it. One point per day, {M['n_days']:,} of them.
""")

code(f"""
plt.plot(daily.index, daily.values, lw=0.5)
plt.xlabel("date")
plt.ylabel("earthquakes M{G_MAG}+ per day")
plt.title("{M['n_quakes']:,} earthquakes over {M['n_days']:,} days")
plt.show()
""")

ask("""
### ✏️ Your turn 1

One day in that plot towers over the rest. Which day was it, and how many earthquakes did it hold?

`daily.max()` gives the largest count. `daily.idxmax()` gives the *label* of the largest value —
here, the date — the way `.argmax()` gave you the position of the largest cell in a grid.
`str(...)[:10]` trims the timestamp down to just the date.

**Use these names**, because the self-check looks for them: `busiest_count`, `busiest_day`.
""")

answer("""
busiest_count = daily.max()
busiest_day = str(daily.idxmax())[:10]

print("the busiest day held", busiest_count, "earthquakes")
print("and it was", busiest_day)
""", """
assert "-" in busiest_day, \\
    "busiest_day should read like a date — .idxmax() gives the label, .argmax() gives a row number"
print("✓ the busiest day —", busiest_day, "with", busiest_count, "earthquakes, against",
      round(daily.mean(), 3), "on an average day")
""")

# --- section 2 -------------------------------------------------------------
md(f"""
## What a world without clustering looks like

{M['busiest']} in one day against {M['lam']:.3f} on an average day is a startling ratio, and the
temptation is to declare the catalogue clustered and stop. Resist it for one section, because you
have nothing to compare against. Startling *compared with what?*

So build the comparison. Take the same {M['n_quakes']:,} earthquakes and the same
{M['n_days']:,} days, and throw each earthquake onto a day picked at random. That is a world with
the same rate as ours and no clustering of any kind, because nothing in it knows about anything
else. Then count its days the way you counted the real ones.

`np.random.default_rng(seed)` makes a random-number generator. The `seed` is the only unusual part:
a computer's random numbers are produced by a recipe, and the seed is where the recipe starts, so
the same seed gives the same "random" numbers every time. That is what makes a simulation
something you can hand to somebody else. `rng.integers(0, n_days, n_quakes)` then draws that many
whole numbers between 0 and `n_days`, one day number per earthquake, and `np.histogram` with one
bin per day counts how many landed on each.
""")

code("""
n_days = len(daily)
n_quakes = len(quakes)
day_edges = np.arange(n_days + 1)          # one bin per day


def random_world(rng):
    \"\"\"Scatter n_quakes earthquakes at random over n_days days, and count each day.\"\"\"
    day_numbers = rng.integers(0, n_days, n_quakes)
    counts, edges = np.histogram(day_numbers, bins=day_edges)
    return counts
""")

md(f"""
### Predict before you run

In that world nothing is clustered — every earthquake landed on a day chosen by a coin the
universe has no memory of. Out of {M['n_days']:,} such days, how many earthquakes do you think the
busiest single one will hold? The average day holds {M['lam']:.3f}.

Change `my_guess` to whatever you think, then run the cell.
""")

code(f"""
my_guess = 3

rng = np.random.default_rng({SEED})
one_world = random_world(rng)

print("you guessed:                   ", my_guess)
print("the random world's busiest day:", one_world.max())
""")

md(f"""
{M['one_busiest']}, from a process with no clustering in it at all. If you guessed three or four
you are in good company, and the reason is worth holding on to: {M['n_quakes']:,} things scattered
over {M['n_days']:,} days will not lay themselves out one-per-day-and-a-bit. Some days get none,
some get five, and out of nine and a half thousand tries something is going to get
{M['one_busiest']}.

Here is the same point as a picture. The first two hundred days of the real catalogue, then the
first two hundred days of the random world, on the same axes.
""")

code(f"""
plt.plot(daily.index[:200], daily.values[:200], lw=1)
plt.ylim(0, 9)
plt.xlabel("date")
plt.ylabel("earthquakes M{G_MAG}+ per day")
plt.title("the real catalogue, 200 days")
plt.show()

plt.plot(daily.index[:200], one_world[:200], lw=1)
plt.ylim(0, 9)
plt.xlabel("date")
plt.ylabel("earthquakes M{G_MAG}+ per day")
plt.title("a random world, the same 200 days")
plt.show()
""")

md("""
Cover the titles and you would struggle to say which is which. Both have runs of empty days, both
have spikes, both have stretches that look busier than the rest. **Randomness is already clumpy, so
"it looks clustered" proves nothing on its own** — and that is not a lesson about earthquakes, it
is a lesson about eyes.
""")

ask(f"""
### ✏️ Your turn 2

One random world is an anecdote. Run four more, with four seeds of your own choosing — your
birthday, the last digits of your student ID, anything — and collect the busiest day of each.

Build the list with the accumulator pattern: start `busiest_by_seed` empty, then inside a `for`
loop over your seeds, make a generator with `np.random.default_rng(seed)`, call
`random_world(rng)`, and append `int(...)` of `.max()` of what comes back — `int` because numpy's
own whole numbers print with their type attached, which is noise here.

**Use these names**, because the self-check looks for them: `my_seeds`, `busiest_by_seed`.
""")

answer(f"""
my_seeds = {M['q2_seeds']}
busiest_by_seed = []

for seed in my_seeds:
    rng = np.random.default_rng(seed)
    busiest_by_seed.append(int(random_world(rng).max()))

print("seeds:              ", my_seeds)
print("busiest day of each:", busiest_by_seed)
""", """
assert len(busiest_by_seed) == len(my_seeds), \\
    "one busiest day per seed, so the two lists should be the same length"
assert max(busiest_by_seed) < daily.max(), \\
    "a random world reached the real busiest day — check that you called random_world"
print("✓ four more random worlds — busiest days", busiest_by_seed,
      "against", daily.max(), "in the real catalogue")
""")

# --- section 3 -------------------------------------------------------------
md(f"""
## Two thousand worlds instead of one

Five worlds is better than one and still not an answer. What you want is the whole *range* of
busiest days a world without clustering produces, so that you can say where the real catalogue
falls in it. That is a **Monte Carlo** simulation: Make up a world where the effect is absent, a
thousand times, and see how often chance alone beats what you measured.

You already have the machinery. Run `random_world` {N_WORLDS:,} times and keep only the number you
care about each time — the busiest day.
""")

code(f"""
rng = np.random.default_rng({SEED})
busiest_days = []

for run in range({N_WORLDS}):
    busiest_days.append(random_world(rng).max())

print("worlds simulated:", len(busiest_days))
""")

code(f"""
plt.hist(busiest_days, bins=np.arange(min(busiest_days) - 0.5, {M['busiest']} + 5))
plt.axvline(daily.max(), color="firebrick")
plt.xlabel("busiest single day of a whole world (earthquakes M{G_MAG}+)")
plt.ylabel("number of simulated worlds")
plt.title("{N_WORLDS:,} worlds without clustering; the red line is the real catalogue")
plt.show()
""")

md(f"""
Every one of the {N_WORLDS:,} simulated worlds is in that little pile on the left. The red line is
where the catalogue we actually live in sits. There is nothing in between.

That figure is the week's argument in one picture, but a picture is not a number. Put numbers on it.
""")

ask(f"""
### ✏️ Your turn 3

`busiest_days` is a plain Python list. `np.array(...)` turns it into an array so that you can ask
it array questions.

Print four things: the mean of the {N_WORLDS:,} busiest days, their 95th percentile
(`np.percentile(worlds, 95)` — the value 95% of them fall below), how many of the {N_WORLDS:,}
worlds reached the real busiest day, and the real busiest day divided by the simulated mean.

**Use these names**, because the self-check looks for them: `worlds`.
""")

answer(f"""
worlds = np.array(busiest_days)

print("mean busiest day of a random world:  ", round(worlds.mean(), 2))
print("95th percentile:                     ", np.percentile(worlds, 95))
print("worlds that reached the real busiest:", (worlds >= daily.max()).sum())
print("real divided by random:              ", round(daily.max() / worlds.mean(), 1))
""", """
assert worlds.max() < daily.max(), \\
    "if a simulated world reached the real busiest day, something is wrong with the simulation"
print("✓ the comparison — the busiest day of a random world averages",
      round(worlds.mean(), 2), "and the real one is", daily.max(), "—",
      round(daily.max() / worlds.mean(), 1), "times larger")
""")

# --- section 4 -------------------------------------------------------------
md(f"""
## The day itself, and the year around it

A factor of {M['ratio']:.1f} is not a near miss. Nothing chance produces looks like the catalogue,
so something other than chance is putting earthquakes on the same day as each other. The catalogue
can say what.
""")

code(weekkit.CHECKPOINT.format(body="""quakes["time"] = pd.to_datetime(quakes["time"])
daily = quakes.set_index("time")["mag"].resample("D").count()"""))

code(f"""
day = quakes[(quakes["time"] >= "{M['busiest_day']}") & (quakes["time"] < "2011-03-12")]

print(len(day), "earthquakes on", "{M['busiest_day']}")
print(day.sort_values("mag", ascending=False).head(1)[["latitude", "longitude", "mag", "place"]])
""")

code(f"""
close = (abs(day["latitude"] - {M['top_lat']}) < 5) & (abs(day["longitude"] - {M['top_lon']}) < 5)

print(close.sum(), "of the", len(day), "were within 5 degrees of that one")
""")

md(f"""
Magnitude {M['top_mag']}, off the Pacific coast of northern Japan — the largest earthquake in this
catalogue. And {M['near_epicentre']} of the day's {M['busiest']} earthquakes happened within five
degrees of it.

That is an **aftershock** sequence. A large earthquake does not release the stress in the crust
tidily: it slips over a patch of fault hundreds of kilometres long, and in doing so it loads the
rock around the edges of that patch and on every neighbouring fault. Those places then fail in
turn — over hours, then weeks, then years, at a rate that dies away. So the events are not
independent. One earthquake makes the next one more likely, in a particular place and at a
particular time, and that is exactly the assumption the random world was built without.

How long does it last? `daily.loc[a:b]` reads out a stretch of days between two dates, and
`.sum()` adds them up.
""")

code(f"""
print("earthquakes in the 30 days from {M['busiest_day']}:", daily.loc["{M['busiest_day']}":"2011-04-09"].sum())
print("earthquakes in the 30 days before it: ", daily.loc["2011-02-09":"2011-03-10"].sum())
""")

md(f"""
{M['after30']} against {M['before30']} — the month after the mainshock held about four times the
month before it. So the excess is not confined to one day.

One more thing worth checking before drawing a conclusion, because the simulation assumed the rate
was the same every day for twenty-six years. Was it? `rolling(365).sum()` slides a 365-day window
along the series and adds up what is inside it, which turns a spiky daily count into a running
year.
""")

code(f"""
running = daily.rolling(365).sum()

plt.plot(running.index, running.values)
plt.xlabel("date")
plt.ylabel("earthquakes M{G_MAG}+ in the previous 365 days")
plt.title("a running year, {M['n_days']:,} daily windows")
plt.show()

print("lowest running year: ", running.min())
print("highest running year:", running.max(), "on", str(running.idxmax())[:10])
print("median:              ", running.median())
""")

md(f"""
The rate is not a constant: a running year holds anywhere from {M['run_min']} to {M['run_max']}
earthquakes around a median of {M['run_median']}. But there is no long climb across the record, and
the highest running year of all ends on {M['run_max_day']} — which is to say the biggest wobble in
the rate is the same aftershock sequence, seen through a wider window. A flat rate is an
approximation, and a fair one for this comparison, because the simulation was handed exactly the
{M['n_quakes']:,} earthquakes the catalogue holds.

Which leaves the obvious worry: is this whole result one earthquake in Japan?
""")

ask(f"""
### ✏️ Your turn 4

Answer it with the neighbouring windows. `daily.index.year` gives the year of every day in the
series, so `daily.groupby(daily.index.year).max()` splits the {M['n_years']} years apart and gives
you the busiest day of each.

Build that, draw it with `plt.scatter`, and put a horizontal line at the 95th percentile you
printed above with `plt.axhline({M['sim_p95']:.0f}, color="firebrick")` — that is the bar a world
without clustering clears only one time in twenty. Then print the series itself, and count how many
years are above the line.

**Use these names**, because the self-check looks for them: `by_year`.
""")

answer(f"""
by_year = daily.groupby(daily.index.year).max()

plt.scatter(by_year.index, by_year.values)
plt.axhline({M['sim_p95']:.0f}, color="firebrick")
plt.xlabel("year")
plt.ylabel("busiest single day (earthquakes M{G_MAG}+)")
plt.title("the busiest day of each of {M['n_years']} years")
plt.show()

print(by_year)
print("years whose busiest day is above {M['sim_p95']:.0f}:", (by_year > {M['sim_p95']:.0f}).sum())
""", f"""
assert len(by_year) == {M['n_years']}, "one number per year, so there should be {M['n_years']}"
print("✓ every year, not one —", (by_year > {M['sim_p95']:.0f}).sum(), "of {M['n_years']} years hold a day above",
      {M['sim_p95']:.0f}, "and", (by_year > {M['sim_p95']:.0f}).sum() - 1, "of those are not 2011")
""")

md(f"""
{M['years_over']} of the {M['n_years']} years clear a bar that a world without clustering clears
once in twenty, and even the quietest year in the record reaches {M['quietest_year_max']}. Take
2011 out entirely and {M['years_over'] - 1} years still do it. This is not one earthquake in Japan;
it is what the record looks like everywhere you cut it.
""")

# --- section 5 -------------------------------------------------------------
md(f"""
## A formula instead of a simulation

Simulating {N_WORLDS:,} worlds took a moment and a screenful of code. For some questions about a
world with no clustering there is a formula that gives the answer exactly, with no simulation at
all, and it is worth having because it is what a real forecast is built from.

If events happen at random at an average rate of λ per interval, then the chance that a given
interval holds **none at all** is e to the power of −λ — in Python, `np.exp(-lam)`. That is the
**Poisson** formula, and λ is the only thing it needs. The simulation and the formula are two
routes to the same place: one *empirical*, got by counting what happened, and one *theoretical*,
got by evaluating an expression. If they disagree, one of them is wrong.
""")

code(weekkit.CHECKPOINT.format(body=f"""quakes["time"] = pd.to_datetime(quakes["time"])
daily = quakes.set_index("time")["mag"].resample("D").count()
n_days, n_quakes = len(daily), len(quakes)
day_edges = np.arange(n_days + 1)


def random_world(rng):
    \"\"\"Scatter n_quakes earthquakes at random over n_days days, and count each day.\"\"\"
    counts, edges = np.histogram(rng.integers(0, n_days, n_quakes), bins=day_edges)
    return counts


one_world = random_world(np.random.default_rng({SEED}))"""))

code("""
lam = daily.mean()

print("rate, earthquakes per day:      ", round(lam, 3))
print("formula says quiet days:        ", round(np.exp(-lam), 4))
print("the simulated world actually had:", round((one_world == 0).mean(), 4))
""")

md(f"""
{M['p_zero']:.4f} against {M['one_quiet_frac']:.4f}. The formula and the simulation agree, which is
the point: the simulation was never doing anything magical, it was drawing from the world the
formula describes.

Two more things fall straight out of it. The chance of **at least one** event in an interval is
whatever is left over, `1 - np.exp(-lam)`. And λ can be scaled to any interval you like: a day is
{M['lam']:.3f} earthquakes, so an hour is λ divided by 24. Finally, `1 / lam` is the average wait
between events — the **recurrence interval**.
""")

code("""
print("chance of at least one somewhere on Earth today:      ", round(1 - np.exp(-lam), 4))
print("chance of at least one in the next hour:              ", round(1 - np.exp(-lam / 24), 4))
print("average hours between them:                           ", round(24 / lam, 1))
""")

ask("""
### ✏️ Your turn 5

The formula describes a world without clustering. The catalogue is not that world, and the busiest
day already showed one way it differs. Here is the other end of the same distribution.

Count the days in the real catalogue that held no earthquake at all — `(daily == 0).sum()` — and
work out how many the formula expects, which is `np.exp(-lam) * n_days`. Print both, and print the
difference.

**Use these names**, because the self-check looks for them: `real_quiet`, `expected_quiet`.
""")

answer("""
real_quiet = (daily == 0).sum()
expected_quiet = np.exp(-lam) * n_days

print("quiet days in the real catalogue:", real_quiet)
print("quiet days the formula expects:  ", round(expected_quiet))
print("difference:                      ", real_quiet - round(expected_quiet), "days")
""", """
assert real_quiet > 1, "real_quiet should be a COUNT of days, not a fraction — .sum(), not .mean()"
print("✓ quiet days —", real_quiet, "in the catalogue against", round(expected_quiet),
      "from the formula:", real_quiet - round(expected_quiet), "more silence than chance allows")
""")

md(f"""
So the real Earth has {M['excess_quiet']} *more* empty days than a world without clustering, as
well as a day holding {M['busiest']}. Both at once, and they are the same fact: piling earthquakes
into a few days has to leave other days emptier, because the total is fixed. That is what
clustering does to a record, at both ends.
""")

# --- section 6 -------------------------------------------------------------
md(f"""
{weekkit.CLOSING_HEADING}

**They cluster, and it is not an illusion — but you could not have known that by looking.** A world
with no clustering at all, running at Earth's own rate, produces a busiest day of about
{M['sim_mean']:.1f} and never once in {N_WORLDS:,} tries got past {M['sim_max']}; the real
catalogue's busiest day holds {M['busiest']}, about {M['ratio']:.1f} times chance, and
{M['years_over']} of {M['n_years']} separate years break the same bar. The excess is aftershocks:
{M['near_epicentre']} of the {M['busiest']} earthquakes on {M['busiest_day']} were within five
degrees of a single magnitude-{M['top_mag']} rupture, and the month after it held {M['after30']}
events against {M['before30']} in the month before. That is physics — stress transferred to
neighbouring rock, which fails in its turn — not noise.

What did the work was not a cleverer look at the data. It was building the world where the effect
is absent and measuring how often chance alone beats what you measured.
""")

# --- summary and homework --------------------------------------------------
md(weekkit.week_cheatsheet(5))

md(f"""
## Homework

Three parts, and a change of scale. Class asked whether earthquakes cluster; the homework asks the
question people actually want answered, which is what the chance is of one happening **here**,
while you are here.

`bay` is already loaded: every earthquake of magnitude {B_MAG} and above the USGS records within
two degrees of Berkeley, from {B_START[:4]} to {B_END[:4]} — {M['n_bay']} of them over
{BAY_YEARS} years. If you have restarted since class, run the setup cell at the top first. The
oldest events in it predate modern instruments, and their magnitudes were reconstructed later from
written accounts of the shaking, so treat them as approximate.
""")

ask(f"""
### ✏️ Your turn 6

Berkeley sits at latitude {BERK_LAT}, longitude {BERK_LON}. So
`abs(bay["latitude"] - {BERK_LAT})` is how far north or south a row is from campus in degrees, and
`abs(bay["longitude"] - ({BERK_LON}))` is how far east or west.

Keep the rows within **1.0 degree** of campus in both directions, call that `near`, and from it
keep the rows of magnitude 6.0 and above, called `big`. Then:

- a **rate**: how many of them per year, over the {BAY_YEARS} years the query covers;
- a **recurrence interval**: `1 / rate`, the average wait in years;
- and the chance of **at least one** during four years at Berkeley, `1 - np.exp(-rate * 4)`.

Print all four.

**Use these names**, because the self-check looks for them: `near`, `big`, `rate`.
""")

answer(f"""
near = bay[(abs(bay["latitude"] - {BERK_LAT}) <= 1.0)
           & (abs(bay["longitude"] - ({BERK_LON})) <= 1.0)]
big = near[near["mag"] >= 6.0]
rate = len(big) / {BAY_YEARS}

print("magnitude 6+ within 1 degree of campus:", len(big))
print("rate, per year:                        ", round(rate, 4))
print("recurrence interval, years:            ", round(1 / rate, 1))
print("chance of at least one in four years:  ", round(1 - np.exp(-rate * 4), 3))
""", """
assert len(big) < len(near), "big is the magnitude-6 part of near, so it must be the smaller one"
assert rate < 1, "that is a rate per YEAR — above 1 means you divided by the wrong number"
print("✓ the four-year forecast —", len(big), "earthquakes,", round(1 / rate, 1),
      f"years apart on average, and a {round(100 * (1 - np.exp(-rate * 4)))}% chance in four years")
""")

ask(f"""
### ✏️ Your turn 7

"Near Berkeley" was your choice, and it changed the answer. Find out by how much.

Write a function `forecast(half)` that does everything part 1 did, for a box `half` degrees on each
side of campus, and **returns** the four-year chance. Inside it, also print:

- the events themselves — `print(big[["time", "mag", "place"]])`;
- the gaps between them in days. `pd.to_datetime(big["time"]).diff()` gives the interval between
  each event and the one before it — the first is blank, so `.dropna()` — and `.dt.days` turns each
  interval into a whole number of days. `sorted(gaps)` then puts them in order, shortest first.
  Sort `big` by `"time"` first, or the gaps are meaningless;
- and the chance of at least one in **30 years**, which is the window published earthquake
  forecasts conventionally use.

Then call it twice, at `half = 1.0` and `half = 2.0`, keeping the two answers.

**Use these names**, because the self-check looks for them: `forecast`, `p1`, `p2`.
""")

answer(f"""
def forecast(half):
    \"\"\"The Poisson forecast for a box `half` degrees each side of campus; returns the 4-year chance.\"\"\"
    near = bay[(abs(bay["latitude"] - {BERK_LAT}) <= half)
               & (abs(bay["longitude"] - ({BERK_LON})) <= half)]
    big = near[near["mag"] >= 6.0].sort_values("time")
    rate = len(big) / {BAY_YEARS}
    gaps = pd.to_datetime(big["time"]).diff().dropna().dt.days

    print("half-width", half, "degrees:", len(big), "earthquakes,",
          round(1 / rate, 1), "years apart on average")
    print(big[["time", "mag", "place"]])
    print("gaps in days, shortest first:", sorted(gaps))
    print("chance of at least one in 4 years: ", round(1 - np.exp(-rate * 4), 3))
    print("chance of at least one in 30 years:", round(1 - np.exp(-rate * 30), 3))
    return 1 - np.exp(-rate * 4)


p1 = forecast(1.0)
p2 = forecast(2.0)
""", """
assert p2 > p1, "the bigger box holds more earthquakes, so its four-year chance must be the larger"
print(f"✓ the two boxes — {round(100 * p1)}% within one degree of campus "
      f"and {round(100 * p2)}% within two")
""")

ask(f"""
### ✏️ Your turn 8

You now have two numbers for the same question and, under each of them, the earthquakes they were
built from and the gaps between those earthquakes. Three or four sentences, using **your own
printed output**:

1. Which of the two four-year numbers would you quote, and what does the bigger box buy and cost?
   Name at least one earthquake it adds.
2. The Poisson formula assumes events arrive independently at a steady rate. Quote your shortest
   gap and your recurrence interval, and say whether those two numbers are consistent with that
   assumption.
3. Given what class measured about clustering, say in which direction you distrust your number.
""")

answer_prose(f"""
I would quote the one-degree box: {M['small']['p4'] * 100:.0f}% in four years, from
{M['small']['n']} earthquakes with a recurrence interval of {M['small']['recur']:.1f} years.
Widening to two degrees raises it to {M['big']['p4'] * 100:.0f}% and shortens the recurrence to
{M['big']['recur']:.1f} years, but it buys those extra {M['big']['n'] - M['small']['n']} events by
reaching two degrees away: the {M['extra'][0]} and {M['extra'][-1]} earthquakes are both in the
wider box, and shaking falls off with distance, so an earthquake that far from campus is not the
same hazard as one underneath it. More data is not automatically better data when the extra data
is answering a different question.

The gaps say the independence assumption is wrong. In the one-degree box the shortest gap between
consecutive magnitude-6 earthquakes is {M['small']['gaps'][0]} days, against a recurrence interval
of {M['small']['recur']:.1f} years — and in the two-degree box the shortest gap is
{M['big']['gaps'][0]} days, because two magnitude-6 earthquakes happened within an hour of each
other. Then the record swings the other way: the one-degree box then waits
{M['small']['gaps'][-1]:,} days, {M['small']['gaps'][-1] / 365.25:.1f} years, for the next one. Two
events in {M['small']['gaps'][0]} days and then nothing for
{M['small']['gaps'][-1] / 365.25:.1f} years is not a steady rate, and class showed why: after a
large earthquake the surrounding crust is loaded and fails again, so events arrive in bursts. That
is the same clustering that made the real catalogue's busiest day {M['ratio']:.1f} times what
chance produces.

Which direction do I distrust it in? Both, which is the honest answer. The count includes
aftershocks, so the rate is higher than the rate at which *independent* earthquakes begin, and that
pushes the number up; but the whole estimate rests on {M['small']['n']} events in
{BAY_YEARS} years, and one event more or fewer moves it by several points. It is a single number
with nothing attached to say how firm it is, and putting an interval around a number like this one
is the next thing this course has to learn to do.
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
    for p in sorted((ROOT / "data").glob("week05_*.csv")):
        print(f"cache: data/{p.name} ({p.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
    weekkit.gate(5)
