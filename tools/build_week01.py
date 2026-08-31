#!/usr/bin/env python
"""Build week 1 — "What was your birthquake?" — from one source.

Emits BOTH notebooks so they cannot drift:

    docs/notebooks/01_birthquake_solution.ipynb   executed, every output and figure saved
    docs/notebooks/01_birthquake.ipynb            the same file with the answer cells emptied

and writes the cached fallback CSVs into data/, one per live query the class runs.

    python tools/build_week01.py             build, cache, execute
    python tools/build_week01.py --no-exec   build and cache only
    python tools/build_week01.py --no-shim   execute with NO cache redirect (see below)

THE CACHE REDIRECT, AND WHY IT EXISTS.
`platform: cache_base:` points at raw.githubusercontent.com/AI4EPS/EPS88_PyEarth/main/data, and
data/ HAS NEVER BEEN PUSHED, so every cache URL 404s today. The USGS queries all run live and are
unaffected; only coastlines.csv has no live source. The notebook now degrades gracefully — it
prints a message and draws the maps without coastlines — so it executes clean either way. But
shipping five maps with no coastline would ship figures no student will ever see. So this script
executes the kernel with a `sitecustomize.py` on PYTHONPATH that redirects cache_base reads to the
identical files in ./data. NOTHING is inserted into the notebook: execution counts run 1..N and no
build scaffolding exists to remove. The saved outputs are exactly what the shipped code produces
once data/ is on main. AFTER THE PUSH, run with --no-shim and the outputs must be unchanged.
"""
import argparse
import os
import pathlib
import tempfile
import urllib.parse as up

import nbformat as nbf
import pandas as pd
import yaml
from nbclient import NotebookClient

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import weekkit  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "docs" / "notebooks"

COURSE = yaml.safe_load((ROOT / "course.yml").read_text())
PLATFORM = COURSE["platform"]
WEEK = next(s for s in COURSE["schedule"] if s["n"] == 1)
CACHE_BASE = PLATFORM["cache_base"]
USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query"

ANSWER_STUB = "# ← your answer here"
PROSE_STUB = "*(Double-click this cell and replace this line with your answer.)*"

# Every live query the class cells run, and therefore every file the fallback needs.
# The homework is deliberately absent: 46 unknown birthdays cannot be cached.
CACHED_QUERIES = [
    ("1983-12-02", "1983-12-03", 4.5),
    ("1983-01-01", "1984-01-01", 4.5),
    ("1983-01-01", "1984-01-01", 5.5),
    ("1983-01-01", "1984-01-01", 6.5),
    ("1983-01-01", "1984-01-01", 7.5),
    ("1982-01-01", "1983-01-01", 7.5),
    ("1984-01-01", "1985-01-01", 7.5),
    ("1980-01-01", "1990-01-01", 6.5),
    ("1980-01-01", "1990-01-01", 7.5),
    ("1990-01-01", "2000-01-01", 6.5),
    ("1990-01-01", "2000-01-01", 7.5),
    ("2000-01-01", "2010-01-01", 6.5),
    ("2000-01-01", "2010-01-01", 7.5),
    ("2010-01-01", "2020-01-01", 6.5),
    ("2010-01-01", "2020-01-01", 7.5),
    ("1976-01-01", "2026-01-01", 6.5),
    ("1976-01-01", "2026-01-01", 7.5),
    ("1940-10-03", "1940-10-04", 4.5),
    ("1940-01-01", "1941-01-01", 4.5),
]


def query_url(start, end, minmag):
    return (f"{USGS}?format=csv&orderby=time-asc"
            f"&starttime={start}&endtime={end}&minmagnitude={minmag}")


def cache_name(start, end, minmag):
    return f"week01_{start}_{end}_M{minmag}.csv"


def write_cache():
    DATA.mkdir(exist_ok=True)
    total = 0
    for start, end, minmag in CACHED_QUERIES:
        path = DATA / cache_name(start, end, minmag)
        df = pd.read_csv(query_url(start, end, minmag))
        df.to_csv(path, index=False)
        total += path.stat().st_size
        print(f"  {path.name}: {len(df)} rows, {path.stat().st_size / 1e3:.0f} kB")
    print(f"  ({len(CACHED_QUERIES)} files, {total / 1e6:.1f} MB)")


def datahub_link(slug):
    repo_name = PLATFORM["repo"].rstrip("/").split("/")[-1]
    q = up.urlencode({"repo": PLATFORM["repo"],
                      "urlpath": f"lab/tree/{repo_name}/{PLATFORM['notebook_dir']}/{slug}.ipynb",
                      "branch": PLATFORM["branch"]})
    return f"{PLATFORM['datahub']}/hub/user-redirect/git-pull?{q}"


# ----------------------------------------------------------------------------------------------
# The notebook. md(...) and code(...) build cells; answer=True marks a cell the student fills in.
# ----------------------------------------------------------------------------------------------
CELLS = []


def md(src, answer=False):
    CELLS.append({"t": "md", "src": src.strip("\n"), "answer": answer})


def code(src, answer=False):
    CELLS.append({"t": "code", "src": src.strip("\n"), "answer": answer})


# ── front matter ──────────────────────────────────────────────────────────────────────────────
md(f"""
# Week 1 — What was your birthquake?

[**Open this notebook in DataHub**]({datahub_link(WEEK['slug'])})

## The question

Somewhere on the day you were born, the ground moved. It moved several times, in fact, and one of
those was bigger than all the others. That one is your **birthquake**.

The United States Geological Survey keeps a catalogue of every earthquake its instruments have
managed to locate — where it was, how deep, how big, to the second. Anyone can ask it a question
over the internet, for free, without an account. In the next hour you will ask it for a single
day and find a birthquake in four lines of Python.

Then we will ask it for something harder. A single day of earthquakes turns out to look like
nothing at all: a scatter of dots on a blank world. A single *year* of the same data draws the
outline of the planet's tectonic plates. Somewhere between one day and one year, a pattern that
was always there becomes visible — and deciding how much data you need to see it is a scientific
question in its own right, not a technicality.
""")

md("""
## What you'll be able to do

**The Earth science**

- find the largest earthquake the world recorded on any date you choose;
- say what shape the world's earthquakes make, and what that shape is;
- check for yourself whether small earthquakes really do outnumber large ones ten to one, and say
  how much data it takes before that ratio means anything;
- say what an earthquake catalogue actually is, and what it is not.

**The Python**

- run a cell, read an error, restart the kernel and run everything again;
- `print`, the four basic kinds of value, variables, arithmetic, and f-strings;
- lists: `len`, `list[i]`, `list[-1]`, `list[a:b]`, `max`, `min`, `list.index(v)`;
- three plots: `plt.hist`, `plt.scatter` and `plt.plot`, with labelled axes and a title.

## How this notebook works

A notebook is a stack of **cells**. Grey cells hold Python; white cells (like this one) hold text.
Click a cell and press **Shift + Enter** to run it. Everything runs inside a **kernel** — a Python
session that remembers what you have run so far, in the order you ran it. When something stops
making sense, use **Kernel ▸ Restart Kernel and Run All Cells**; that clears the memory and runs
the notebook top to bottom, which is also how it will be marked.

**Twelve places where you write something: eight in class, four at home.** Ten of them are grey
cells holding one line, and nothing else in the notebook looks like it:

```python
# ← your answer here
```

The other two ask for a paragraph instead of code, and are white cells reading *"Double-click this
cell and replace this line with your answer."*

The eight in class are six questions and two one-line predictions. The four at home are the three
homework parts, the last of which wants a short paragraph as well as code. All of it is your work
and all of it is graded.

If you fall behind, look for a **Checkpoint** cell — running it rebuilds everything the cells
after it need, so a broken cell costs you a few minutes rather than the day.
""")

md("""
## Setup

Run this cell. You do not need to follow it yet.

**Coming later:** it uses **pandas** (week 3) to fetch a table off the web, and `def` to give a job
a name (week 2). What comes back out is all things you meet today: `load()` hands you six plain
lists, `how_many()` hands you one number, and `coast_lon`/`coast_lat` are two more lists holding
the world's coastlines.
""")

code(f'''
import pandas as pd
import matplotlib.pyplot as plt

# house style, set once, so every plot cell below holds only what matters
plt.rcParams.update({{"figure.figsize": (7, 4), "figure.dpi": 110,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True}})

USGS = "{USGS}"
CACHE = "{CACHE_BASE}"


def load(start, end, minmag=4.5):
    """Ask the USGS catalogue for one slice of time; hand back six plain lists."""
    if end <= start:
        raise ValueError(f'"{{end}}" is not after "{{start}}". The second date is the END of the '
                         'span, so it has to be the later one.')
    live = (f"{{USGS}}?format=csv&orderby=time-asc"
            f"&starttime={{start}}&endtime={{end}}&minmagnitude={{minmag}}")
    try:
        table = pd.read_csv(live)
    except Exception as problem:
        if "400" in str(problem):
            raise ValueError(f'The catalogue rejected "{{start}}" or "{{end}}". Dates must be '
                             'written as "YYYY-MM-DD", for example "2008-03-12".') from None
        print("live source unreachable, using the cached copy:", type(problem).__name__)
        try:
            table = pd.read_csv(f"{{CACHE}}/week01_{{start}}_{{end}}_M{{minmag}}.csv")
        except Exception:
            raise ConnectionError(
                f"No network, and no cached copy of {{start}} to {{end}}. Your own birthday is "
                "only ever fetched live, so nobody could cache it in advance — get back online "
                "and run this cell again.") from None
    return (list(table.time), list(table.latitude), list(table.longitude),
            list(table.depth), list(table.mag), list(table.place))


def how_many(start, end, minmag=4.5):
    """Just the number of earthquakes in that slice of the catalogue."""
    times = load(start, end, minmag)[0]   # counting needs one of the six lists, not all six
    return len(times)


def big_quake_ratio(start, end):
    """How many M6.5+ earthquakes there are for each M7.5+ one, over any span of time."""
    return round(how_many(start, end, 6.5) / how_many(start, end, 7.5), 1)


def load_coastlines():
    """The world's coastlines as two plain lists — every map below draws them."""
    try:
        shape = pd.read_csv(CACHE + "/coastlines.csv")
    except Exception as problem:
        print("coastline file unreachable:", type(problem).__name__,
              "- the maps below will draw without it")
        return [], []
    return list(shape.lon), list(shape.lat)


coast_lon, coast_lat = load_coastlines()
''')

# ── 1. cells, values, names ───────────────────────────────────────────────────────────────────
md("""
## 1. A cell, a value, a name

Python is a calculator that can also hold on to things.

`print()` shows you a value. `type()` says what kind of value it is — there are four basic kinds,
and confusing them is the commonest first-week error: `4` (a whole number, `int`), `4.5` (a
decimal, `float`), `"Berkeley"` (text, `str`, always in quotes) and `True` (yes-or-no, `bool`).

A **variable** is a name stuck onto a value with `=`. An **f-string** is a piece of text with an
`f` in front of the opening quote, which lets you drop a variable into the middle of a sentence by
putting its name in `{ }`.
""")

code('''
print("Hello from cell one")
print(2 + 2, 7 / 2, 2 ** 10)          # ** means "to the power of"
print(type(4), type(4.5), type("Berkeley"), type(True))
''')

code('''
quarterback = "Aaron Rodgers"
jersey = 12
print(f"{quarterback} wore number {jersey}, and he was born on 1983-12-02.")
''')

md("""
### ✏️ Question 1

Make two variables of your own: `my_name`, holding your name as text, and `my_number`, holding any
whole number you like. Then print one sentence that uses both, with an f-string.

**Use these names** — `my_name` and `my_number`.
""")

code("", answer=True)

# ── 2. one day of earthquakes ─────────────────────────────────────────────────────────────────
md("""
## 2. One day of earthquakes

`load(start, end)` fetches every earthquake the catalogue holds between two dates and hands back
six lists, all the same length and all in the same order: the times, the latitudes, the longitudes,
the depths in kilometres, the magnitudes, and the place names.

Six names on the left of one `=` is how Python catches six things at once: the first name gets the
first list, the second name the second, and so on.

By default it asks for magnitude 4.5 and up. That floor is not innocent, and we come back to it.

We will work on **1983-12-02**, the day Aaron Rodgers was born. `end` is the day *after* the one
you want, because the catalogue counts up to it but not including it.
""")

code('''
times, lats, lons, depths, mags, places = load("1983-12-02", "1983-12-03")
print(len(mags), "earthquakes of magnitude 4.5 or larger, worldwide, on 1983-12-02")
''')

md("""
A **list** is an ordered row of values. `len(mags)` is how many there are. To get one of them you
give its position in square brackets — and **positions count from 0**, so the first earthquake of
the day is `mags[0]`, the second is `mags[1]`, and `mags[-1]` counts back from the end and gives
you the last.

The six lists line up. Position 1 in `times` and position 1 in `places` describe the *same*
earthquake, which is what makes reading across them useful.

`max(mags)` and `min(mags)` give the largest and smallest values in a list.
""")

code('''
print(times[1])
print(places[1])
print(mags[1], "at", depths[1], "km deep")
print("the smallest magnitude of the day was", min(mags))
''')

md("""
### ✏️ Question 2

The catalogue came back in time order, earliest first. Print two things:

1. the places of the **first four** earthquakes of the day — one slice, `places[a:b]`, where the
   slice runs from `a` up to but *not including* `b`;
2. the magnitude of the **very last** earthquake of the day, using a negative index.
""")

code("", answer=True)

md("""
### ✏️ Question 3

**Find the birthquake.** It is the largest earthquake of the day, and you now have everything you
need to describe it.

`max(mags)` gives you the largest magnitude. `mags.index(v)` gives you the *position* of the value
`v` in the list — and because the six lists line up, that same position in `places` and `depths`
tells you where it happened and how deep it was. (The dot in `mags.index(...)` is new: it means
"the `index` that belongs to `mags`".)

Store the position in a variable called `biggest`, then print one f-string sentence giving the
magnitude, the place and the depth.
""")

code("", answer=True)

md("""
Now look back at the four place names you printed in question 2, and at where `biggest` turned out
to be. Two of those first four are the same stretch of Guatemalan coast, and if you print more of
the list you find two more: positions 0, 2, 4 and 6 are all Champerico.

That is not a coincidence and it is not a mistake in the file. The smaller shocks that follow a
large rupture in the same place are its **aftershocks**, the crust around the break settling.
Further down the list are five more in the Indian Ocean, all between 4.5 and 5.3 with no large one
among them — a different pattern, and one nobody has to be first.

So the day is not fourteen unrelated events. It is a handful of places, each producing several.
""")

# ── 3. something on screen ────────────────────────────────────────────────────────────────────
md("""
## 3. Something on screen

Fourteen rows of text is not a way to see anything. Two plots, both one line of real content:
`plt.hist(x, bins=n)` chops one list of numbers into `n` bars and shows how they are spread out,
and `plt.scatter(x, y)` puts a dot at every pair.
""")

code('''
# ── Checkpoint ── run this if you are behind ──
times, lats, lons, depths, mags, places = load("1983-12-02", "1983-12-03")
''')

code('''
plt.hist(mags, bins=10)
plt.xlabel("magnitude")
plt.ylabel("number of earthquakes")
plt.locator_params(axis="y", integer=True)      # half an earthquake is not a thing
plt.title(f"Magnitudes on 1983-12-02 (n = {len(mags)})")
plt.show()
''')

md("""
Fourteen numbers spread over ten bars: four bars at the left hold thirteen of the earthquakes,
five in the middle hold nothing at all, and one lone bar out on the right holds the 7.0. You can see that small ones
are commoner than large ones, and that is all you can see — you certainly cannot read a ratio off
it. **Hold that thought.**

Now the same day in space. Longitude runs −180 to 180 across the map, latitude −90 to 90 up it.
`plt.plot(coast_lon, coast_lat)` draws the coastlines, so the dots have a world to sit on. Those
two lists came out of the setup cell, and the gaps in them are what make matplotlib lift the pen
between one island and the next.
""")

code('''
plt.plot(coast_lon, coast_lat, color="0.6", lw=0.6)
plt.scatter(lons, lats, s=18, color="crimson")
plt.gca().set_aspect("equal")                   # gca() is "the plot I am drawing now"
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title(f"Earthquakes on 1983-12-02 (n = {len(mags)})")
plt.show()
''')

md("""
Fourteen dots on a whole planet: Guatemala, Svalbard up in the Arctic, Iran, Ethiopia, Tonga at
the very edge of the map, Indonesia, and a cluster in the middle of the Indian Ocean — with
enormous empty spaces between them. If somebody handed you this map and told you earthquakes
happen in narrow lines, you would be entitled to disbelieve them.

Both figures fail for the same reason, and it is not a coding mistake: **fourteen numbers.**
""")

# ── 4. a whole year ───────────────────────────────────────────────────────────────────────────
md("""
## 4. A whole year

Same code, more data. Before you run anything:

### Predict before you run

Set `guess_year` to a whole number — how many magnitude-4.5-and-up earthquakes do you think the
catalogue holds for the *whole* of 1983? Write it down before you look. A wrong number you
committed to teaches you more than a right one you were handed.
""")

code("", answer=True)

code('''
# ── Checkpoint ── run this if you are behind ──
coast_lon, coast_lat = load_coastlines()
''')

code('''
year_times, year_lats, year_lons, year_depths, year_mags, year_places = load(
    "1983-01-01", "1984-01-01")
print(len(year_mags), "M4.5+ earthquakes worldwide in 1983.  You guessed", guess_year)
''')

md("""
### ✏️ Question 4

Draw the map again, for the whole year. It is the same cell you just typed, with the year's lists
instead of the day's — `year_lons` and `year_lats` — a smaller dot size (`s=2`, because there are
a lot of them) and a title that says 1983 and gives the count.

Before you run it, decide what you expect: dots sprinkled fairly evenly over the globe, or
something else?
""")

code("", answer=True)

md("""
They are not evenly spread, and they are not random. The dots fall into **narrow lines**: a ring
right round the Pacific, a line down the very middle of the Atlantic, another through the middle
of the Indian Ocean, and a broad band across southern Asia from Indonesia to the Mediterranean.
Whole continents — Canada, Brazil, Australia, most of Africa — are nearly blank.

The coastlines you drew show that these lines are not a property of *land*. The Atlantic line runs
down open ocean thousands of kilometres from any shore; the South American line hugs the coast on
one side of the continent and nothing on the other.

These lines are the edges of the Earth's tectonic plates: the plates move past each other only at
their boundaries, so that is where the crust breaks. That identification is not something this one
map proves — it is the conclusion seismologists drew from exactly this kind of map in the 1960s
(Isacks, Oliver and Sykes, 1968), once there were finally enough recorded earthquakes to see the
lines. What the map does show, on its own, is that the pattern is real and extremely sharp.

And note what changed between the last figure and this one: **nothing but the amount of data.**
""")

# ── 5. how many of each size ──────────────────────────────────────────────────────────────────
md("""
## 5. How many of each size

The day's histogram was too thin to read. Try it again with the year.
""")

code('''
# ── Checkpoint ── run this if you are behind ──
year_times, year_lats, year_lons, year_depths, year_mags, year_places = load(
    "1983-01-01", "1984-01-01")
''')

code('''
plt.hist(year_mags, bins=31)
plt.xlabel("magnitude")
plt.ylabel("number of earthquakes")
plt.title(f"Magnitudes worldwide in 1983 (n = {len(year_mags)})")
plt.show()
''')

md("""
Two things to see, and the second is the more interesting.

The obvious one: from about magnitude 4.9 rightwards the bars collapse. Around five hundred
earthquakes near magnitude 4.8, a couple of dozen near 6, single figures near 7. Small earthquakes
are enormously commoner than large ones, and the drop looks steady rather than sudden.

The odd one: the far **left** of the histogram turns over. There are fewer earthquakes in the file
at magnitude 4.5 than at 4.8. Whatever else is true of the Earth, it does not make fewer magnitude
4.5 earthquakes than magnitude 4.8 ones. Park that; section 7 is about where it comes from.

A shape is not a number, though, and the tail is exactly where the picture is least readable — the
bars out at magnitude 7 are a pixel high. To put a number on it, count instead of looking. Raising
the magnitude floor and counting what survives is the whole measurement.
""")

md("""
### ✏️ Question 5

`how_many(start, end, minmag)` returns just the count. Run it three times on 1983 with floors of
4.5, 5.5 and 6.5 — for example `how_many("1983-01-01", "1984-01-01", 5.5)` — storing the answers as
`n45`, `n55` and `n65`. Print the three counts, then print the two ratios `n45 / n55` and
`n55 / n65` — wrap each one in `round(x, 1)` so you get 9.1 rather than sixteen decimal places.

**Use those names**: two later cells read `n65` and `n45` back.
""")

code("", answer=True)

# ── 6. can one earthquake carry a ratio? ──────────────────────────────────────────────────────
md("""
## 6. Can one earthquake carry a ratio?

Both of your ratios came out close to nine, against a rule of thumb that says the answer should be
about ten. Before asking where that rule comes from, take one more step up the magnitude scale —
because the top of a catalogue is where counting gets hard.

The cell below does that, then widens the window: the two years either side of 1983, then four
whole decades, then fifty years. It takes a few seconds; it is asking the catalogue eleven
questions.
""")

code('''
# ── Checkpoint ── run this if you are behind ──
n45 = how_many("1983-01-01", "1984-01-01", 4.5)
n65 = how_many("1983-01-01", "1984-01-01", 6.5)
''')

code('''
n75 = how_many("1983-01-01", "1984-01-01", 7.5)
print("1983:", n65, "at M6.5+ and", n75, "at M7.5+, a ratio of", round(n65 / n75, 1))
print("M7.5+ in 1982:", how_many("1982-01-01", "1983-01-01", 7.5),
      "  in 1984:", how_many("1984-01-01", "1985-01-01", 7.5))
print("the same ratio by decade —",
      "1980s:", big_quake_ratio("1980-01-01", "1990-01-01"),
      " 1990s:", big_quake_ratio("1990-01-01", "2000-01-01"),
      " 2000s:", big_quake_ratio("2000-01-01", "2010-01-01"),
      " 2010s:", big_quake_ratio("2010-01-01", "2020-01-01"))

n65_all = how_many("1976-01-01", "2026-01-01", 6.5)
n75_all = how_many("1976-01-01", "2026-01-01", 7.5)
print("fifty years:", n65_all, "at M6.5+ and", n75_all, "at M7.5+, a ratio of",
      round(n65_all / n75_all, 1))
''')

md("""
### ✏️ Question 6

1983 gives a ratio nowhere near ten. Two readings are on the table, and both are things a
scientist might say:

- **A.** The ten-to-one rule breaks down at the top of the scale — the very largest earthquakes are
  rarer than it predicts.
- **B.** These counts are too small to measure a ratio with at all, and the spread you see is what
  small counts look like.

**Which does the evidence support, and which number are you leaning on?** Two or three sentences,
in the cell below.

If your honest answer is that you cannot tell one decade from another, say so, and say what would
let you tell. That is a full answer, not a hedge — knowing when your data cannot settle a question
is most of the job.
""")

md(PROSE_STUB, answer=True)

md("""
Two things are now worth naming.

**The rule of thumb.** It is not something this notebook derived. Beno Gutenberg and Charles
Richter measured it in southern California and published it in 1944, as a straight line on a plot
of magnitude against the logarithm of the count. The slope of that line is called the *b-value*,
and "about ten per magnitude step" is the statement that b is close to 1. We are borrowing a
convention, and your numbers are a check on how far it travels — from Richter's California to the
whole world across fifty years, it travels well.

**Which way your own ratios are likely to be wrong.** Question 5 gave numbers a little under nine,
and the turnover at the left of the 1983 histogram says the catalogue is already missing some of
its smallest earthquakes. Missing small earthquakes makes `n45` too small, which drags `n45 / n55`
**down**. Nine is more likely a floor on the real ratio than a ceiling — and where those missing
earthquakes went is the next section.
""")

# ── 7. what a catalogue is ────────────────────────────────────────────────────────────────────
md("""
## 7. What a catalogue is

Walter Alvarez taught geology at Berkeley, a few hundred metres from this room, and worked out with
his father that an asteroid impact killed the dinosaurs. He was born on **1940-10-03**.

### Predict before you run

Set `guess_1940` to how many M4.5-and-up earthquakes you think the catalogue holds for the whole of
**1940**. The Earth was the same size and the plates were moving at the same speed, so commit to a
number before you run the next cell.
""")

code("", answer=True)

code('''
# ── Checkpoint ── run this if you are behind ──
n45 = how_many("1983-01-01", "1984-01-01", 4.5)

print("M4.5+ on Alvarez's birthday, 1940-10-03:", how_many("1940-10-03", "1940-10-04"))
print("M4.5+ in all of 1940:", how_many("1940-01-01", "1941-01-01"), " you guessed", guess_1940)
print("M4.5+ in all of 1983:", n45)
''')

md("""
One earthquake on the day he was born, against fourteen on the day Rodgers was born. Two hundred
and thirty-six in his birth year, against four thousand one hundred and twenty-four in Rodgers's.

The planet did not get seventeen times busier between 1940 and 1983. Plate motion is measured in
centimetres per year and does not change on that timescale. What changed is the listening, and the
bookkeeping. Seismic stations in 1940 were sparse and clustered in Europe, Japan and North America,
and their readings were collected and compared by hand long after the event; by 1983 there was a
standardised global network reporting continuously. A magnitude 5 in the middle of the Pacific in
1940 happened, and is simply not in the file.

> **Catalogue completeness.** A catalogue lists what somebody's instruments recorded, not what
> happened. Where there are no seismometers there are no earthquakes in the file.

This is the single most important thing on this page, and it explains three things at once.

It explains the 1940 numbers, which therefore cannot be compared with the 1983 numbers: any
"earthquakes are increasing" claim built that way is a fact about instruments, not about the Earth.

It explains the turnover on the left of the histogram in section 5. 1983 is not 1940, but it is not
today either, and a catalogue that is complete at magnitude 5 can still be missing magnitude 4.5s
from the parts of the world nobody was listening to closely.

And it explains why the floor of 4.5 has been in every query today. A floor is a decision about
*which* earthquakes you are willing to believe the catalogue really has all of. Below it, things
start going missing, and they go missing by different amounts in different places. The homework
walks straight into that.
""")

# ── the question, answered ────────────────────────────────────────────────────────────────────
md("""
## The question, answered

**What was your birthquake?** It is the largest earthquake the world's seismometers recorded on the
day you were born — for 2 December 1983, a magnitude 7.0 off Champerico, Guatemala — and the
catalogue you pulled it out of also shows you, in one year of the same data, that earthquakes fall
along the narrow lines where tectonic plates meet, that each step up in magnitude cuts the count by
roughly ten once you have enough years to measure it, and that what is in the file depends on who
was listening.
""")

md(weekkit.week_cheatsheet(1))

# ── homework ──────────────────────────────────────────────────────────────────────────────────
md("""
## Homework

Three parts, on **your own** birthday. Class did all of this on someone else's.

Nobody knew your birthday in advance, so unlike every cell above there is no cached copy of it. If
you are offline, `load()` will stop with a message saying so — do the homework somewhere with a
network connection rather than working around it.

Each part has a **self-check** cell: run it, and it will tell you if something is missing. It
prints your own numbers back at you when it passes. (Those cells use `assert` and comparisons like
`!=`, which are next week's — you run them, you never have to write them.)
""")

md("""
### ✏️ Homework 1 — your own birthquake, and your own day

**First, answer the week's title question for yourself.**

1. Set `MY_BIRTHDAY` to your own birth date and `MY_NEXT_DAY` to the day after it, both as text in
   `"YYYY-MM-DD"` form.
2. Load your birthday into six lists, exactly as class did, named `my_times`, `my_lats`, `my_lons`,
   `my_depths`, `my_mags`, `my_places`. Print **your birthquake** — its magnitude, where it was and
   how deep — using `max()` and `.index()` the way question 3 did.
3. Now take the magnitude floor off altogether. `how_many(start, end, -9)` does it: −9 is lower
   than any magnitude any seismometer has ever reported, so nothing is excluded. **Before you run
   it**, set `guess_all` to how many you think one day holds with no floor at all.
4. Then set `n_45` to the M4.5-and-up count for your day and `n_all` to the count with no floor,
   and print both alongside your guess.

**Use these names**, because the self-check looks for them: `MY_BIRTHDAY`, `MY_NEXT_DAY`,
`my_mags`, `my_places`, `guess_all`, `n_45`, `n_all`.
""")

code("", answer=True)

code('''
assert MY_BIRTHDAY != "1983-12-02", "use your own birthday, not the one from class"
assert MY_NEXT_DAY != MY_BIRTHDAY, "MY_NEXT_DAY is the day AFTER MY_BIRTHDAY"
assert n_45 > 0, "nothing came back at M4.5+ — check MY_NEXT_DAY is the later of the two dates"
assert n_all >= n_45, "taking the floor off cannot give you fewer earthquakes than leaving it on"
print(f"Your birthquake: magnitude {max(my_mags)}, {my_places[my_mags.index(max(my_mags))]}.")
print(f"{n_all} earthquakes on {MY_BIRTHDAY}, of which {n_45} were M4.5 or larger: "
      f"{n_all / n_45:.1f} times as many. You guessed {guess_all}.")
''')

md("""
### ✏️ Homework 2 — the fork

Question 5 measured about nine times as many earthquakes for each step *down* in magnitude. You
just took the floor off your own birthday altogether, so the ten-to-one rule ought to be able to
predict `n_all` from `n_45` — and you get to decide how far down to run it. Two defensible choices,
because the rule has to stop somewhere and the catalogue does not tell you where:

- **all the way down to 0**, near the bottom of what the catalogue records anywhere, which is
  `4.5 - 0 = 4.5` steps;
- **only down to 2.5**, on the grounds that a global catalogue records very little below that
  outside a few densely instrumented countries, which is `4.5 - 2.5 = 2` steps.

Pick one. Set `low_mag` to the magnitude you are extrapolating down to, set `steps = 4.5 - low_mag`,
and set `predicted_all` to `n_45 * 10 ** steps`. Then set `miss_factor` to how many times bigger
your prediction is than the `n_all` you actually measured, and print your choice, the prediction
and the miss.

**Use these names**: `low_mag`, `steps`, `predicted_all`, `miss_factor`.

*(If homework 1 did not run, set `n_45` and `n_all` by hand from any single day you can load — the
arithmetic is the point, not the date.)*
""")

code("", answer=True)

code('''
assert low_mag < 4.5, "low_mag is the floor you extrapolate DOWN to, so it is below 4.5"
assert steps > 0, "steps counts how far down you are going: 4.5 - low_mag, not the other way round"
assert miss_factor > 1, ("the rule should overshoot here. If yours does not: predicted_all is "
                         "n_45 * 10 ** steps — ** is 'to the power of', not 'times' — and "
                         "miss_factor is the prediction divided by n_all, not the reverse")
print(f"Extrapolating to M{low_mag} predicts {predicted_all:.0f} earthquakes; the catalogue holds "
      f"{n_all}. The rule overshoots by a factor of {miss_factor:.1f}.")
''')

md("""
### ✏️ Homework 3 — how much data is enough?

One day of earthquakes showed you nothing. One year drew the plate boundaries. The lines must
become visible somewhere in between — and *where* is a question class did not answer.

Find out for your own birthday. Draw **three maps**, all at magnitude 4.5 and above, all with the
coastline, all in the same style as the class maps:

- your birthday alone;
- the seven days beginning on your birthday;
- the whole calendar month your birthday falls in.

You have no date arithmetic yet, so write the six dates out by hand as text. Put the earthquake
count in each title, and store the three counts as `n_day`, `n_week` and `n_month`.

Then, in the **markdown cell below the self-check**, answer this: which of the three maps is the
first one where you can see *lines* rather than scattered dots, and how many earthquakes did that
take? Two or three sentences, quoting your own counts. There is no single right answer here — the
week's third takeaway is that this is a judgement, so a defensible call with a number attached is a
full answer, and "it is somewhere between the week and the month" is a fine thing to say if you say
what you saw.
""")

code("", answer=True)

code('''
assert n_day < n_week < n_month, "a week holds more than a day and a month more than a week; check your six dates"
print(f"{n_day} earthquakes in a day, {n_week} in a week, {n_month} in a month.")
''')

md(PROSE_STUB, answer=True)


# ----------------------------------------------------------------------------------------------
# Model answers, in the order the answer cells appear.
# ----------------------------------------------------------------------------------------------
ANSWERS = [
    # Q1
    '''
my_name = "Alex"
my_number = 7
print(f"{my_name} picked the number {my_number}.")
''',
    # Q2
    '''
print(places[0:4])
print(mags[-1])
''',
    # Q3
    '''
biggest = mags.index(max(mags))
print(f"The birthquake was a magnitude {mags[biggest]} at {places[biggest]}, "
      f"{depths[biggest]} km deep.")
''',
    # prediction: guess_year
    '''
guess_year = 200
''',
    # Q4
    '''
plt.plot(coast_lon, coast_lat, color="0.6", lw=0.6)
plt.scatter(year_lons, year_lats, s=2, color="crimson")
plt.gca().set_aspect("equal")
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title(f"M4.5+ earthquakes worldwide in 1983 (n = {len(year_mags)})")
plt.show()
''',
    # Q5
    '''
n45 = how_many("1983-01-01", "1984-01-01", 4.5)
n55 = how_many("1983-01-01", "1984-01-01", 5.5)
n65 = how_many("1983-01-01", "1984-01-01", 6.5)
print(n45, n55, n65)
print(round(n45 / n55, 1), round(n55 / n65, 1))
''',
    # Q6 (prose)
    '''
**B**, and the fifty-year number is what convinces me. 1983 has one earthquake at M7.5 and above,
1982 has none and 1984 has two — a ratio resting on a single event is not a measurement of
anything, and 53 tells me about the size of my sample, not about the Earth.

The decades do not settle it either: 19.9, 11.6, 7.8, 8.9. If I had only been shown the 1980s I
would have said the rule fails badly at the top of the scale; if I had only been shown the 2000s I
would have said it overshoots. I cannot tell one of those decades from another on this evidence —
four numbers scattered between 7.8 and 19.9 is what a ratio looks like when each one rests on
twenty to sixty of the rarest events in the file.

Over all fifty years, 2192 at M6.5+ against 218 at M7.5+ is a ratio of 10.1, which is the rule of
thumb almost exactly. So I do not think A is supported: the rule holds at the very place 1983 made
it look broken, and what I was looking at was a shortage of data. What would let me test A properly
is a way of putting an uncertainty on each decade's ratio, so I could say whether 19.9 is further
from 10 than chance would explain — which is more statistics than I have yet.
''',
    # prediction: guess_1940
    '''
guess_1940 = 4000
''',
    # HW1
    '''
MY_BIRTHDAY = "2008-03-12"      # <- your own birthday goes here
MY_NEXT_DAY = "2008-03-13"

my_times, my_lats, my_lons, my_depths, my_mags, my_places = load(MY_BIRTHDAY, MY_NEXT_DAY)
mine = my_mags.index(max(my_mags))
print(f"My birthquake was a magnitude {my_mags[mine]} at {my_places[mine]}, "
      f"{my_depths[mine]} km deep.")

guess_all = 45

n_45 = how_many(MY_BIRTHDAY, MY_NEXT_DAY, 4.5)
n_all = how_many(MY_BIRTHDAY, MY_NEXT_DAY, -9)
print("guessed", guess_all, " M4.5+:", n_45, " everything:", n_all)
''',
    # HW2
    '''
low_mag = 2.5
steps = 4.5 - low_mag
predicted_all = n_45 * 10 ** steps
miss_factor = predicted_all / n_all
print(f"Extrapolating {steps} steps down to M{low_mag} predicts {predicted_all:.0f}, "
      f"against {n_all} measured, so the rule overshoots {miss_factor:.1f} times.")

# the other defensible choice, worked for comparison: run the rule all the way down to 0
other = n_45 * 10 ** 4.5
print(f"Taking it all the way to M0 predicts {other:.0f} instead, which is "
      f"{other / n_all:.0f} times the real count. Both choices overshoot, and the further down "
      f"the rule is pushed the worse it gets.")
''',
    # HW3
    '''
day_times, day_lats, day_lons, day_depths, day_mags, day_places = load("2008-03-12", "2008-03-13")
week_times, week_lats, week_lons, week_depths, week_mags, week_places = load(
    "2008-03-12", "2008-03-19")
month_times, month_lats, month_lons, month_depths, month_mags, month_places = load(
    "2008-03-01", "2008-04-01")
n_day, n_week, n_month = len(day_mags), len(week_mags), len(month_mags)

plt.plot(coast_lon, coast_lat, color="0.6", lw=0.6)
plt.scatter(day_lons, day_lats, s=18, color="crimson")
plt.gca().set_aspect("equal")
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title(f"One day: 2008-03-12 (n = {n_day})")
plt.show()

plt.plot(coast_lon, coast_lat, color="0.6", lw=0.6)
plt.scatter(week_lons, week_lats, s=10, color="crimson")
plt.gca().set_aspect("equal")
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title(f"One week: 2008-03-12 to 2008-03-18 (n = {n_week})")
plt.show()

plt.plot(coast_lon, coast_lat, color="0.6", lw=0.6)
plt.scatter(month_lons, month_lats, s=6, color="crimson")
plt.gca().set_aspect("equal")
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title(f"One month: March 2008 (n = {n_month})")
plt.show()
''',
    # HW3 prose
    '''
March 2008, with 527 earthquakes, is the first of my three maps where I would honestly call them
lines — and even then only some of them.

One day gave 15 dots, and four of those were the same spot in Vanuatu, so it is really about a
dozen places: Indonesia, Vanuatu, the Kuril Islands, one off Mexico, one in the middle of the
Atlantic. It could have been anything. The week, at 100, is better than I expected: the arc from
the Kuril Islands down through Japan, the Izu Islands, Indonesia and New Guinea to Vanuatu is
unmistakable. But the whole of South America is three dots, and the Atlantic is three — the Azores,
Bouvet and the South Sandwich Islands — so if that week were all I had I would have concluded that
earthquakes happen in the western Pacific.

By the month the Pacific ring closes: 31 in South America make a line down the Chilean and Peruvian
coast, 20 across the Aleutians join Asia to Alaska, and Central America fills in. The mid-ocean
ridges are the last to arrive — 18 down the whole Atlantic is still a dotted line rather than a
solid one, and those only became continuous on the year map in class, with its 4124.

So my number is about 500. A few hundred earthquakes buy you the busiest plate boundaries and a few
thousand are needed for the quiet ones, which is really the same lesson as the day map: the pattern
was there the whole time, and what more data bought was being able to see the parts of it that
produce the fewest earthquakes.
''',
]


def build():
    sol = nbf.v4.new_notebook()
    stu = nbf.v4.new_notebook()
    ai = 0
    for c in CELLS:
        if c["answer"]:
            body = ANSWERS[ai].strip("\n")
            ai += 1
            if c["t"] == "md":
                sol.cells.append(nbf.v4.new_markdown_cell(body))
                stu.cells.append(nbf.v4.new_markdown_cell(PROSE_STUB))
            else:
                sol.cells.append(nbf.v4.new_code_cell(body))
                stu.cells.append(nbf.v4.new_code_cell(ANSWER_STUB))
        else:
            make = nbf.v4.new_markdown_cell if c["t"] == "md" else nbf.v4.new_code_cell
            sol.cells.append(make(c["src"]))
            stu.cells.append(make(c["src"]))
    if ai != len(ANSWERS):
        raise SystemExit(f"{ai} answer cells but {len(ANSWERS)} model answers")
    for nb in (sol, stu):
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                     "name": "python3"}
    return sol, stu


SITECUSTOMIZE = f'''
"""Build-time only: redirect cache_base reads to ./data until the repo is pushed.

Lives on PYTHONPATH for the build kernel and nowhere else. It touches no notebook cell, so
execution counts stay 1..N. Delete the need for it by pushing data/ and running --no-shim.
"""
import pandas as _pd

_real_read_csv = _pd.read_csv
_BASE = "{CACHE_BASE}"
_LOCAL = "{DATA}"


def _read_csv(target, *a, **k):
    if isinstance(target, str) and target.startswith(_BASE):
        target = _LOCAL + target[len(_BASE):]
    return _real_read_csv(target, *a, **k)


_pd.read_csv = _read_csv
'''


def execute(sol, shim=True):
    """Run the solution on a fresh kernel and keep every output."""
    saved = os.environ.get("PYTHONPATH")
    tmp = None
    if shim:
        tmp = tempfile.mkdtemp(prefix="eps88-week01-shim-")
        (pathlib.Path(tmp) / "sitecustomize.py").write_text(SITECUSTOMIZE)
        os.environ["PYTHONPATH"] = tmp + (os.pathsep + saved if saved else "")
    try:
        NotebookClient(sol, timeout=900, kernel_name="python3",
                       resources={"metadata": {"path": str(OUT)}}).execute()
    finally:
        if saved is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = saved
    return sol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-exec", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-shim", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if not args.no_cache:
        print("cached fallbacks:")
        write_cache()
    if not (DATA / "coastlines.csv").exists():
        raise SystemExit("data/coastlines.csv missing — run tools/make_coastlines.py")

    sol, stu = build()
    nbf.write(stu, OUT / "01_birthquake.ipynb")
    if not args.no_exec:
        sol = execute(sol, shim=not args.no_shim)
    nbf.write(sol, OUT / "01_birthquake_solution.ipynb")
    n_q = sum(1 for c in CELLS if c["answer"])
    print(f"\n{len(CELLS)} cells, {n_q} answer cells "
          f"-> {OUT / '01_birthquake.ipynb'}\n{'':21}-> {OUT / '01_birthquake_solution.ipynb'}")


if __name__ == "__main__":
    main()
