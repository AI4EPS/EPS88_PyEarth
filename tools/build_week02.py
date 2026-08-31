#!/usr/bin/env python
"""Build week 2 — "Which of these worlds could have liquid water?" — from one source.

Emits BOTH notebooks so they cannot drift:

    docs/notebooks/02_liquid_water_solution.ipynb   executed, every output and figure saved
    docs/notebooks/02_liquid_water.ipynb            the same file with the answer cells emptied

and writes the cached fallback CSV into data/.

    python tools/build_week02.py            build, cache, execute
    python tools/build_week02.py --no-exec  build and cache only

ONE NOTE ON EXECUTION. The notebook's only fetch is the live NASA Exoplanet Archive query, so the
solution executes end to end without touching the cache; the cached read points at
`platform: cache_base:` on GitHub and 404s until the repo is pushed. The archive stalls or 503s on
roughly one request in four, so the build retries the cache write rather than failing the run.
"""
import argparse
import pathlib
import sys
import urllib.parse as up

import nbformat as nbf
import pandas as pd
import yaml
from nbclient import NotebookClient

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import weekkit  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "docs" / "notebooks"

COURSE = yaml.safe_load((ROOT / "course.yml").read_text())
PLATFORM = COURSE["platform"]
WEEK = next(s for s in COURSE["schedule"] if s["n"] == 2)
CACHE_BASE = PLATFORM["cache_base"]

ANSWER_STUB = "# ← your answer here\n"
PROSE_STUB = "*(Double-click this cell and replace this line with your answer.)*"

CACHE_FILE = "week02_exoplanets.csv"
COLUMNS = "pl_name,st_teff,st_rad,pl_orbsmax,pl_rade,pl_eqt,discoverymethod"
ARCHIVE_URL = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
               f"select+{COLUMNS}+from+ps+where+default_flag=1&format=csv")


def write_cache(attempts=4):
    """Fetch the archive and write the fallback CSV, retrying: 1 request in 4 fails."""
    DATA.mkdir(exist_ok=True)
    path = DATA / CACHE_FILE
    for attempt in range(attempts):
        try:
            df = pd.read_csv(ARCHIVE_URL)
            break
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {type(e).__name__}")
    else:
        raise SystemExit("the archive refused every attempt; run again")
    df.to_csv(path, index=False)
    print(f"  {path.name}: {len(df)} rows, {path.stat().st_size / 1e3:.0f} kB")


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


# ── 0. front matter ───────────────────────────────────────────────────────────────────────────
md(f"""
# Week 2 — Which of these worlds could have liquid water?

[**Open this notebook in DataHub**]({datahub_link(WEEK['slug'])})
""")

md("""
## The question

Liquid water is the one thing every search for life beyond Earth agrees on looking for, and there
is a temperature range where water is liquid: between freezing and boiling. So here is a test you
can write in four lines. Work out how hot a planet's star makes it. If the answer lands between
273 K and 373 K, keep the planet. Otherwise throw it away.

NASA's Exoplanet Archive holds thousands of planets around other stars, and enough of them carry
the three numbers this test needs — the star's temperature, the star's size, and how far out the
planet orbits — that you can run it on all of them at once. You will.

Then you will point the same test at Earth, and it will throw Earth away. That failure is the
whole point of the week: it is not a bug in your arithmetic, and finding out what is missing tells
you more about this planet than a list of candidate worlds ever could.
""")

md("""
## What you'll be able to do

**The Earth and planetary science**

- work out the temperature starlight alone would give any planet, from three published numbers;
- say what a planet's *albedo* is and what changing it does to that answer;
- measure the greenhouse effect, as the gap between the temperature a world should have and the
  one it does have — for Venus, Earth and Mars, three thicknesses of air;
- say why a list of "habitable" worlds built this way is not to be believed, and what it is
  missing.

**The Python**

- `for` loops, `range`, and `list.append` — doing the same thing to every item in a list;
- `if` / `elif` / `else` and the comparison operators, including what happens when the number you
  wanted to compare is not there;
- the accumulator pattern, for counting and collecting as a loop runs;
- `def`, arguments, `return`, and a docstring you can read back with `help()`.

## How this notebook works

A notebook is a stack of **cells**. Grey cells hold Python; white cells (like this one) hold text.
Click a cell and press **Shift + Enter** to run it. Everything runs inside a **kernel**, which
remembers what you have run so far. When things stop making sense, use
**Kernel ▸ Restart Kernel and Run All Cells**.

**Ten places where you write something: seven in class, three at home.** Nine of them are grey
cells, and nothing else in the notebook looks like this:

```python
# ← your answer here
```

The tenth wants a paragraph instead of code, and is a white cell reading *"Double-click this cell
and replace this line with your answer."*

The seven in class are six questions and one prediction. The three at home are the homework parts,
the last of which is the paragraph. All of it is your work and all of it is graded.

If you fall behind, look for a **Checkpoint** cell — running it rebuilds everything the next
section needs.
""")

md("""
## Setup

Run this cell. You do not need to follow it yet.

**Coming later:** it uses **pandas** (week 3) to fetch a table off the web, and `def` (later
today) to give that job a name. What it hands you is seven plain lists of the kind you met last
week, all the same length and all in the same order.

**If this cell is still spinning after half a minute**, the archive has stalled. It does that
often, and it stalls silently rather than raising an error, so there is nothing to read. Press the
■ (stop) button in the toolbar to interrupt it, then run the cell again — each attempt is a fresh
roll of the dice, and a second or third usually gets through.
""")

code(weekkit.SETUP_CELL.format(
    figsize="(7, 4)",
    cache_base=CACHE_BASE,
    signature="",
    docstring="Ask the NASA Exoplanet Archive for one row per known planet.",
    url_expr=('(\n            "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="\n'
              f'            "select+{COLUMNS}"\n'
              '            "+from+ps+where+default_flag=1&format=csv")'),
    cache_expr=f'"{CACHE_FILE}"',
    unpack='''SUN_TEMP = 5772.0             # the Sun's surface temperature in kelvin (IAU 2015 nominal value)
SUN_RADIUS_IN_AU = 0.00465047 # the Sun's radius, in astronomical units

archive = load()
usable = archive.dropna(subset=["st_teff", "st_rad", "pl_orbsmax"])

PLANETS_IN_ARCHIVE = len(archive)
names = list(usable.pl_name)
star_temps = list(usable.st_teff)                # the star's surface temperature, kelvin
star_radii = list(usable.st_rad)                 # the star's radius, in Suns
distances = list(usable.pl_orbsmax)              # how far the planet orbits out, in AU
radii = list(usable.pl_rade.astype(object).where(usable.pl_rade.notnull(), None))
archive_temps = list(usable.pl_eqt.astype(object).where(usable.pl_eqt.notnull(), None))
methods = list(usable.discoverymethod)         # how each planet was found''',
))

# ── 1. the temperature starlight alone gives a world ──────────────────────────────────────────
md("""
## 1. The temperature starlight alone would give a world

A planet has no worthwhile furnace of its own. It catches a disc of its star's light and it
radiates heat away from its whole sphere, and it settles at whatever temperature makes those two
equal. Write that balance down and the planet's own size cancels out of it — which is why a test
for liquid water needs nothing about the planet except how far out it orbits:

$$T \\;=\\; T_{\\star}\\,\\sqrt{\\frac{R_{\\star}}{2d}}$$

$T_{\\star}$ is the star's surface temperature, $R_{\\star}$ is the star's radius and $d$ is the
planet's distance from it. The star's radius and the distance have to be in the same units, and
the archive gives us the first in Suns and the second in AU, so `SUN_RADIUS_IN_AU` from the setup
cell converts one to the other. In Python, `** 0.5` is a square root.

Two things this leaves out, and we will come back to both. It assumes the planet absorbs every
scrap of starlight that reaches it — real planets reflect some straight back. And it assumes the
planet has no air. So the number it gives has a name of its own. **Equilibrium temperature.** The
temperature a planet would sit at if starlight were the only thing heating it and it had no air.

It is a convention rather than a measurement: astronomers publish it for almost every planet found,
which is exactly what makes it comparable across thousands of worlds at once.

Let us do Venus, then Mars. Venus orbits at 0.723332 AU and Mars at 1.523679 AU (NASA Planetary
Fact Sheet).
""")

code('''
venus_temp = SUN_TEMP * (1.0 * SUN_RADIUS_IN_AU / (2 * 0.723332)) ** 0.5
print(round(venus_temp, 1), "K")

mars_temp = SUN_TEMP * (1.0 * SUN_RADIUS_IN_AU / (2 * 1.523679)) ** 0.5
print(round(mars_temp, 1), "K")
''')

md("""
Those two blocks differ by one number. That is the signal to stop typing and start looping.

A **`for` loop** takes a list and runs the same indented lines once for each item in it:

```python
for distance in world_distances:
    ...one indented line...
    ...another...
```

The name after `for` is yours to choose; on each pass through the loop it holds the next item.
The indentation is not decoration — it is how Python knows which lines belong to the loop.

`list.append(x)` adds one item to the end of a list. Starting from an empty list `[]` and
appending inside a loop is how you build a result up one item at a time.
""")

code('''
world_names = ["Mercury", "Venus", "Mars", "Jupiter"]
world_distances = [0.387098, 0.723332, 1.523679, 5.204400]   # AU, NASA Planetary Fact Sheet

world_temps = []
for distance in world_distances:
    world_temps.append(SUN_TEMP * (1.0 * SUN_RADIUS_IN_AU / (2 * distance)) ** 0.5)

print(world_temps)
''')

md("""
Four temperatures, and sixteen digits of noise on the end of each — `round()` from last week fixes
that. But the bigger problem is that they have arrived with no names attached, and reading them off
by counting is exactly the kind of thing you get wrong at 2 a.m.

`range(n)` gives you the whole numbers from 0 up to but not including `n`, so
`for i in range(len(world_names)):` walks `i` through every position in the list. Because
`world_names` and `world_temps` line up, position `i` in one describes the same world as position
`i` in the other.

### ✏️ Question 1

Print one line per world, reading from both lists, so that the first line reads

```
Mercury: 447.4 K
```

Use `range` and `len`, and round each temperature to one decimal place.
""")

code("", answer=True)

# ── 2. counting with if ───────────────────────────────────────────────────────────────────────
md("""
## 2. Asking a question of every world at once

Water is liquid between **273 K** (freezing) and **373 K** (boiling) at Earth's sea-level air
pressure. Those two numbers are the whole test.

`if` runs its indented block only when a comparison is true, and the comparisons are the ones you
would write by hand: `<` `>` `<=` `>=` `==` (equal) and `!=` (not equal). `and` joins two of them
into one, and is true only when both halves are. There is `or` too, true when either half is.

The **accumulator pattern** is the other half of a loop. Set a counter to 0 before the loop starts
and add to it inside; same idea for a list — start it empty and append inside.
""")

code('''
too_cold = []
for i in range(len(world_names)):
    if world_temps[i] < 273:
        too_cold.append(world_names[i])

print(len(too_cold), "of the four are below freezing:", too_cold)
''')

md("""
### ✏️ Question 2

Now the test itself. Build a list called `liquid_water_worlds` holding the names of the worlds
whose temperature is **between 273 and 373 K inclusive** — one `if` with an `and` in it — and
print how many there are and which.

**Use that name**, `liquid_water_worlds`, because the sentence below reads it back.
""")

code("", answer=True)

md("""
One world out of four, and it is Venus. Hold on to that; section 7 comes back to it with a
thermometer.

You may also have noticed which world is missing from `world_names`. That was deliberate. Earth
comes back in section 6, and by then you will have built the test properly.
""")

# ── 3. three thousand worlds ──────────────────────────────────────────────────────────────────
md("""
## 3. Three thousand worlds

Four planets do not tell you whether the test is any good. The setup cell fetched NASA's Exoplanet
Archive, which holds one row per known planet around another star, and kept every planet carrying
all three numbers the formula needs.

Last week's idea applies here too, word for word: **a catalogue lists what somebody's instruments
recorded, not what happened.** A planet is in this file because a telescope could detect it, and
the numbers it carries are the ones that particular detection method can measure.
""")

code('''
print(PLANETS_IN_ARCHIVE, "planets in the archive")
print(len(names), "of them carry a star temperature, a star radius and an orbital distance")
print(names[0], star_temps[0], "K,", star_radii[0], "Suns,", distances[0], "AU")

by_transit = 0
for method in methods:
    if method == "Transit":
        by_transit = by_transit + 1
print(by_transit, "of the", len(names), "were found by watching their star dim as they crossed it")
''')

md("""
The loop you wrote for four worlds is the same loop for three thousand. That is the whole reason
loops exist: the number of items stops being your problem.
""")

code('''
temps = []
for i in range(len(names)):
    temps.append(star_temps[i] * (star_radii[i] * SUN_RADIUS_IN_AU / (2 * distances[i])) ** 0.5)

hotter_than_1500 = 0
for temperature in temps:
    if temperature > 1500:
        hotter_than_1500 = hotter_than_1500 + 1

print(len(temps), "temperatures, from", round(min(temps), 1), "K to", round(max(temps), 1), "K")
print(hotter_than_1500, "of them are above 1500 K")
''')

code('''
plt.hist(temps, bins=range(0, 1550, 50))
plt.axvline(273, color="C1")        # water freezes
plt.axvline(373, color="C1")        # water boils
plt.xlabel("equilibrium temperature (K)")
plt.ylabel("number of planets")
plt.title(f"{len(temps)} planets; the {hotter_than_1500} above 1500 K are off the right edge")
plt.show()
''')

md("""
`plt.axvline(x)` draws a vertical line at `x` — here the two edges of the window. They are only
100 K apart while the temperatures run past 1500 K, so the window is a thin slice of what is out
there, and the bulk of the distribution sits to the right of it, hotter. That is mostly a fact
about the telescopes: three quarters of these planets were found by watching a star dim as the
planet crossed in front of it, a planet close in crosses more often and lines up more often, and
close in means hot.

Before trusting our own arithmetic, check it against somebody else's. The archive publishes its own
equilibrium temperature for some of these planets. `abs(x)` gives the size of a number ignoring its
sign, which is how you ask "how far apart are these two" without caring which is bigger.
""")

code('''
ours = []
published = []
for i in range(len(names)):
    if archive_temps[i] is not None:
        ours.append(temps[i])
        published.append(archive_temps[i])

close = 0
for i in range(len(ours)):
    if abs(ours[i] - published[i]) <= 10:
        close = close + 1

print(len(ours), "planets have a published equilibrium temperature to compare against")
print(close, "of those agree with ours to within 10 K —", round(100 * close / len(ours)), "percent")
''')

code('''
plt.scatter(published, ours, s=4, alpha=0.3)
plt.plot([0, 4000], [0, 4000], color="0.4", lw=1)   # the line where the two would agree exactly
plt.xlabel("the archive's published equilibrium temperature (K)")
plt.ylabel("our equilibrium temperature (K)")
plt.title(f"our arithmetic against the archive's, {len(ours)} planets")
plt.show()
''')

md("""
The cloud lies along the grey line and three in five agree to within 10 K, which is enough to say
our formula is the standard formula. It is not enough to say the two always agree. The archive's
number comes from whichever paper reported that planet, and papers differ in the albedo they assume
and the orbital distance they adopt, so where the two part company nothing in this file tells you
which of them is right. Look at the handful of points along the bottom of the plot: our formula
returns almost nothing for a planet the archive calls hot, which means the distance in that row and
the temperature beside it cannot both be describing the same orbit.

### ✏️ Question 3

Run the liquid-water test on the whole archive. Count how many of the temperatures in `temps` are
between 273 and 373 K inclusive, and print the count and how many planets you tested.

**Use the name** `in_window` for the count.
""")

code("", answer=True)

# ── 4. rocky or not ───────────────────────────────────────────────────────────────────────────
md("""
## 4. Rocky, gassy, or nobody knows

A planet the size of Neptune has no surface for an ocean to sit on. The usual dividing line is
**1.6 Earth radii**: below it planets are almost all dense enough to be rock, above it almost all
have thick hydrogen envelopes. That is a convention drawn from measured masses and radii, not a
law, and planets near the line go either way.

The archive has `pl_rade`, the planet's radius in Earths — but not for every planet. Some planets
were found by a method that cannot measure a radius at all. Where the number is missing, `load()`
put **`None`** there, which is Python's word for "nothing here". You cannot compare `None` with a
number: `None < 1.6` is an error, not an answer. So the test needs three branches rather than two,
and `if` / `elif` / `else` is how you write them: Python tries each in order and runs the first one
that is true.
""")

code('''
rocky_worlds = []
too_big = 0
unknown_radius = 0
for i in range(len(names)):
    if temps[i] >= 273 and temps[i] <= 373:
        if radii[i] is None:
            unknown_radius = unknown_radius + 1
        elif radii[i] < 1.6:
            rocky_worlds.append(names[i])
        else:
            too_big = too_big + 1

print(len(rocky_worlds), "rocky,", too_big, "too big,", unknown_radius, "with no measured radius")
print("adding up to", len(rocky_worlds) + too_big + unknown_radius, "planets in the window")
print(rocky_worlds)
''')

code('''
for planet in ["TRAPPIST-1 c", "TRAPPIST-1 d", "TRAPPIST-1 e", "TRAPPIST-1 f"]:
    i = names.index(planet)
    print(planet, round(temps[i], 1), "K")
''')

md("""
Look at the names on the list, and then at those four planets of the TRAPPIST-1 system, which all
orbit the same star at increasing distances. The test keeps c and d and throws out e and f — and
that is the wrong way round from how the exoplanet literature reads that system, where e is the
planet most often named as its best candidate for liquid water and c is usually discussed as a
likely Venus analogue. Your test has kept the one people doubt and rejected the one people like,
and the arithmetic that did it is not wrong.

So the test accepts twelve worlds — and the third branch is the interesting one. 102 of the 189
planets in the window have no measured radius at all, more than half of them, and any of those
could be rocky. Twelve is a floor on the answer, not the answer. **A catalogue lists what
somebody's instruments recorded, not what happened**, and here what was not recorded is most of
the evidence.

Keep `rocky_worlds`. The homework compares its own list against it.
""")

# ── 5. a function ─────────────────────────────────────────────────────────────────────────────
md("""
## 5. Writing the question down once
""")

code('''
# ── Checkpoint ── run this if you are behind ──
temps = []
for i in range(len(names)):
    temps.append(star_temps[i] * (star_radii[i] * SUN_RADIUS_IN_AU / (2 * distances[i])) ** 0.5)

rocky_worlds = []
for i in range(len(names)):
    if temps[i] >= 273 and temps[i] <= 373 and radii[i] is not None and radii[i] < 1.6:
        rocky_worlds.append(names[i])
''')

md("""
We are about to ask the same question with a different number in it, several times over. Copying
the loop and editing one value is what you would do next, and it is exactly what goes wrong: five
copies of a formula are five chances to mistype it and no way to tell which copy is the one that
is right.

A **function** is a recipe written once and given a name. `def` starts it, the names in brackets
are what it needs, the indented lines are what it does, and `return` hands one value back. The
first thing inside should be a **docstring**: one line in triple quotes saying what the function is
for. `help()` prints it back at you, which is why writing one pays.
""")

code('''
def to_celsius(kelvin):
    """Convert a temperature in kelvin to degrees Celsius."""
    return kelvin - 273.15


print(round(to_celsius(288.0), 2))
help(to_celsius)
''')

md("""
Now the real one — and this is where the missing physics goes in.

**Albedo.** The fraction of the starlight falling on a world that it reflects straight back into
space; only the rest is absorbed and turned into heat. Everything so far assumed an albedo of 0, a
perfectly black world. Putting it in multiplies the temperature by $(1-A)^{1/4}$:

$$T \\;=\\; T_{\\star}\\,(1-A)^{1/4}\\,\\sqrt{\\frac{R_{\\star}}{2d}}$$

### ✏️ Question 4

Write that formula down once, as a function:

```python
def equilibrium_temperature(star_temp, star_radius, distance, albedo):
```

It returns `star_temp` × `(1 - albedo) ** 0.25` × the square root of
`star_radius * SUN_RADIUS_IN_AU / (2 * distance)`. Give it a one-line docstring.

Then check it against something you already know: call it for Mars — `SUN_TEMP`, a star radius of
`1.0`, a distance of `1.523679` and an albedo of `0.0` — and print the answer rounded to one
decimal. You should get back the number section 1 printed for Mars.

**Use this name**, `equilibrium_temperature`, because every cell below calls it.
""")

code("", answer=True)

# ── 6. the reveal ─────────────────────────────────────────────────────────────────────────────
md("""
## 6. Now point it at Earth

Two worlds have been missing from every list in this notebook. Earth is one of them; Venus is the
other, and Venus is the only solar-system world your test has accepted so far.

Both have had their albedo measured from orbit: **Earth 0.306**, **Venus 0.770**. Venus's is the
NASA Planetary Fact Sheet value. Earth's is the long-standing NSSDC figure — that sheet now prints
0.294 — and we use 0.306 throughout, because it is the one the sheet's own black-body temperature
row still reproduces. Earth orbits at 1.000 AU and Venus at 0.723332 AU.

Earth's average surface temperature is 288 K — 15 °C, the number you have known since school.
Before you run anything, commit to a guess: set `guess_earth` to the temperature in kelvin you
think `equilibrium_temperature` will hand back for Earth.
""")

code("", answer=True)

md("""
### ✏️ Question 5

Call `equilibrium_temperature` four times, and print each answer rounded to one decimal along with
whether it lands inside the 273–373 K window:

- Earth with albedo `0.0`, then Earth with its measured `0.306`;
- Venus with albedo `0.0`, then Venus with its measured `0.770`.

Then print how far your `guess_earth` was from the answer at Earth's measured albedo.
""")

code("", answer=True)

md("""
Read those four lines again, because between them they demolish the test you just built.

Assume both worlds are perfectly black and the test **accepts both** — Venus at 327.3 K and Earth
at 278.3 K, five degrees above freezing. Use the albedos somebody actually measured and it
**rejects both** — Venus at 226.6 K and Earth at 254.0 K, nineteen degrees below freezing.

There is no third run where it gets the answer right. One of these two worlds has oceans and the
other does not, and the test never separates them: it puts them 49 K apart at albedo 0, with Venus
the hotter, and 27 K apart at their measured albedos, with Venus the cooler. Which way it is wrong
turns on the albedo, a number that says nothing about whether anybody could live there. And if you
guessed Earth's real 288 K a moment ago, your miss was 34.0 K. Hold on to it. The next section
measures how large the thing this test is missing actually is.
""")

# ── 7. the greenhouse effect ──────────────────────────────────────────────────────────────────
md("""
## 7. The size of what is missing

You have a number for the temperature a world should have. For the worlds next door somebody has
been and measured the temperature they do have. The difference is not an error term — it is a
quantity, and one you can now put a size on.

The mean surface temperatures and surface pressures below are measured values from the NASA
Planetary Fact Sheet; the albedos are Bond albedos, Earth's the 0.306 discussed above. The plot
after them
uses one new drawing command, `plt.text(x, y, "Venus")`, which writes a word at a point on the
axes — with three dots and no labels the figure would say nothing.
""")

code('''
solar_names = ["Venus", "Earth", "Mars"]
solar_distances = [0.723332, 1.000000, 1.523679]   # AU
solar_albedos = [0.770, 0.306, 0.250]              # Bond albedo
surface_temps = [737.0, 288.0, 210.0]              # measured mean surface temperature, K
pressures = [92.0, 1.014, 0.006]                   # surface pressure, bar

no_air_temps = []
for i in range(len(solar_names)):
    no_air = equilibrium_temperature(SUN_TEMP, 1.0, solar_distances[i], solar_albedos[i])
    no_air_temps.append(no_air)
    print(f"{solar_names[i]:6s} starlight alone {no_air:6.1f} K   measured {surface_temps[i]:6.1f} K"
          f"   gap {surface_temps[i] - no_air:+7.1f} K   air {pressures[i]:6.3f} bar")
''')

code('''
plt.scatter(no_air_temps, surface_temps)
plt.plot([150, 800], [150, 800], color="0.4", lw=1)   # a world with no air would sit on this line
for i in range(len(solar_names)):
    plt.text(no_air_temps[i] + 15, surface_temps[i], solar_names[i])
plt.xlim(150, 800)                     # the same range on both axes, so the grey line is a fair
plt.ylim(150, 800)                     # comparison rather than an accident of scaling
plt.gca().set_aspect("equal")
plt.xlabel("equilibrium temperature, using each world's measured albedo (K)")
plt.ylabel("measured mean surface temperature (K)")
plt.title("what the air adds: 3 worlds")
plt.show()
''')

md("""
All three should be on the grey line. Venus is far above it, Earth is a little above it, and Mars
sits on it — and that last one needs care in a moment. The plot makes the reason visible: along
the bottom axis the three worlds sit between 209.8 K and 254.0 K, inside forty-five degrees of
each other, while up the side they run from 210.0 K to 737.0 K. Whatever separates
Venus from Mars, the starlight arithmetic cannot see it.

The gap has a name. **The greenhouse effect is the difference between the temperature a world
would have with no air and the temperature it actually has** — the atmosphere lets sunlight down
and slows the infrared going back out. It is not a definition we imposed here; it is what is left
over after the arithmetic: +510.4 K for Venus under 92 bar of air and +34.0 K for Earth under
1.014 bar. Venus has ninety times Earth's surface pressure and fifteen times its warming, so how
much air a world carries is not the whole story — what the air is made of matters too.

Mars needs a caveat, and it is the albedo's lesson over again. Our +0.2 K is not a measurement of
anything: move the 210 K by two degrees and the sign flips, and NASA prints that 210 K with a tilde
in front of it. It is also not quite the right subtraction. An equilibrium temperature balances the
*fourth power* of temperature, and Mars's surface swings through tens of kelvin between day and
night, so its average temperature sits several kelvin below the temperature that matches its
average radiation — and it is the second one this formula is about. That correction is negligible
for Venus, small for Earth and decisive for Mars. Estimates that handle it properly put Mars's
greenhouse effect at a few kelvin rather than nothing (Haberle 2013, *Icarus* 223, 619). Read our
Mars row as *too small for these inputs to resolve*, not as *zero*.

That is why the test threw Earth away. It was never a test of habitability. It was a test of how
much starlight arrives, and the 34 K that makes this planet habitable is not in it.
""")

# ── 8. asking the question again ──────────────────────────────────────────────────────────────
md("""
## 8. Asking again, and again

The window edges were a choice too. 273 K and 373 K are water's freezing and boiling points at
Earth's sea-level pressure — a planet with thinner air boils water lower, and a planet with more
holds it liquid higher. Nobody has measured the air pressure on any of these worlds.

So we should ask the question several ways. That is what the function was for: `count_worlds`
below is the whole of sections 3 and 4 written down once, and now the window and the albedo are
things you hand it rather than things you retype.
""")

code('''
def count_worlds(low, high, albedo):
    """How many archive planets with a measured radius under 1.6 Earths sit between low and high K."""
    found = 0
    for i in range(len(names)):
        temperature = equilibrium_temperature(star_temps[i], star_radii[i], distances[i], albedo)
        if radii[i] is not None and radii[i] < 1.6 and temperature >= low and temperature <= high:
            found = found + 1
    return found


print(count_worlds(180, 273, 0.0), "rocky worlds are below freezing but above 180 K")
''')

md("""
### ✏️ Question 6

Call `count_worlds` three times, all with albedo `0.0`, and print the count each window gives:

- **273 to 373 K** — water's liquid range at Earth's sea-level pressure;
- **250 to 350 K** — the same width, shifted down, on the grounds that any atmosphere at all warms
  a planet above its equilibrium temperature;
- **273 to 323 K** — freezing to 50 °C, if you want somewhere merely uncomfortable.

Print the window and its count on each line. The first should hand back the twelve worlds section
4 found — that is your check that `count_worlds` is doing what the loops did.
""")

code("", answer=True)

md("""
Three defensible windows, three different answers, from the same data and the same code. Nothing
in the archive decides between them — you decide, and then you report which one you chose. The
homework hands you the other choice, the albedo, and that one moves the answer much further.

## The question, answered

**Which of these worlds could have liquid water — and why does your test reject Earth?** On the
archive's own numbers the test accepts twelve rocky worlds, with a hundred and two more in the
window whose size nobody has measured; but it rejects Earth, because equilibrium temperature
counts the starlight arriving and nothing else, and the 34.0 K that keeps this planet's oceans
liquid comes from its air. Run the same test on Venus and Earth with an albedo of 0 and it accepts
both; run it with the albedos we have measured and it rejects both. It never gets the answer right,
and the size of how wrong it is — 34 K here, 510 K on Venus, too small for our numbers to resolve
on Mars — is the greenhouse effect, measured.
""")

md(weekkit.week_cheatsheet(2))

# ── homework ──────────────────────────────────────────────────────────────────────────────────
md("""
## Homework

Three parts. Class ran the test at one albedo and asked which worlds passed; these ask where the
window actually is, what happens when you pick a different albedo, and whether you believe the
answer.

Each of the first two ends with a **self-check** cell: run it, and it tells you if something is
missing, then prints your own numbers back. (Those cells use `assert`, which stops with a message
when what follows it is false. You run them; you never have to write them.)
""")

md("""
### ✏️ Homework 1 — where is the window?

Class asked *is this planet in the window?* Ask the other question: *where is the window?* For a
given star there are two distances, one where a planet would be 373 K and one where it would be
273 K, and every planet between them passes the test.

Turn the formula round. If

$$T \\;=\\; T_{\\star}(1-A)^{1/4}\\sqrt{\\frac{R_{\\star}}{2d}} \\qquad\\text{then}\\qquad
d \\;=\\; \\frac{R_{\\star}}{2\\,(T / T_{\\text{eff}})^{2}} \\quad\\text{where}\\quad
T_{\\text{eff}} = T_{\\star}(1-A)^{1/4}$$

Write it as a function:

```python
def window_edges(star_temp, star_radius, albedo):
```

returning **two** numbers — the inner edge (where a planet would be 373 K) and the outer edge
(where it would be 273 K), both in AU. `return inner, outer` hands back two values at once, and
you catch them with `inner, outer = window_edges(...)`, the same way the setup cell caught seven
lists from `load()`. Remember `SUN_RADIUS_IN_AU`, since the archive measures star radii in Suns
and distances in AU.

Then call it twice for the Sun — `SUN_TEMP`, a radius of `1.0` — once with albedo `0.0` and once
with Earth's measured `0.306`, and print both windows. Earth orbits at 1.000 AU: say for each
window whether Earth is inside it.

**Use these names**, because the self-check looks for them: `sun_inner_0`, `sun_outer_0`,
`sun_inner_earth`, `sun_outer_earth`.
""")

code("", answer=True)

code('''
assert sun_inner_0 < sun_outer_0, "the 373 K edge is the inner one — it is closer to the star"
assert sun_outer_earth < sun_outer_0, "a planet that reflects light is colder, so its window sits closer in"
print(f"With no reflection the Sun's window runs {sun_inner_0:.3f} to {sun_outer_0:.3f} AU.")
print(f"With Earth's albedo it runs {sun_inner_earth:.3f} to {sun_outer_earth:.3f} AU.")
print("Earth orbits at 1.000 AU.")
''')

md("""
### ✏️ Homework 2 — the fork

Nobody has measured the albedo of any planet in the archive. Class used 0, a perfectly black
world, which is the one value we know is wrong. Two defensible substitutes, both measured, both in
this notebook:

- **Earth's, 0.306** — use the one rocky world we know has liquid water;
- **Venus's, 0.770** — most of the worlds this test accepts orbit close in and hot, and Venus is
  the rocky world we have that ended up that way; a thick, bright atmosphere is at least as
  ordinary an outcome as a thin, clear one.

Pick one. Set `my_albedo` to it, then rebuild the accepted list at that albedo: loop over the
archive, and for every planet whose radius is known and under 1.6 and whose temperature is between
273 and 373 K inclusive, append the name to `my_worlds` and the temperature to `my_temps`.

Then compare against class's list. `if name in rocky_worlds:` is true when that name is somewhere
in the list. Count into `stayed` the worlds on your list that were on class's twelve, and into
`moved_in` the ones that were not.

**Use these names**: `my_albedo`, `my_worlds`, `my_temps`, `stayed`, `moved_in`.
""")

code("", answer=True)

code('''
assert my_albedo == 0.306 or my_albedo == 0.770, "pick Earth's albedo or Venus's — those are the two anyone has measured"
assert len(my_worlds) == len(my_temps), "one temperature per world; append to both lists in the same pass"
assert len(my_worlds) > len(rocky_worlds), "a higher albedo cools every planet, so worlds that were too hot for class's window drop into yours — expect more than twelve"
assert stayed + moved_in == len(my_worlds), "every world on your list either was on class's list or was not"
print(f"At albedo {my_albedo} the test accepts {len(my_worlds)} worlds: {stayed} of class's "
      f"{len(rocky_worlds)} survived and {moved_in} are new.")
print("The first ten, with the numbers your test used on them:")
for i in range(10):
    j = names.index(my_worlds[i])
    print(f"  {my_worlds[i]:16s} {my_temps[i]:6.1f} K   star {star_temps[j]:6.0f} K"
          f"   orbit {distances[j]:.3f} AU")
''')

md("""
### ✏️ Homework 3 — one you do not believe

Your test now calls a list of worlds habitable. Pick one of them you do not believe, and say what
the test is missing.

Three or four sentences, and make them specific to numbers you produced:

- name the world, and quote the equilibrium temperature your own run gave it;
- say what your test measured about that world and what it did not;
- class measured Earth's air adding 34.0 K and Venus's adding 510.4 K, to worlds whose starlight
  temperatures were 254.0 K and 226.6 K. Use those numbers to say how far wrong your figure for
  your world could be, in both directions.

Do not stop at "it ignores the atmosphere". Say what that world's atmosphere would have to be
doing for it to be habitable, and what it would have to be doing for it not to be.
""")

md(PROSE_STUB, answer=True)


# ----------------------------------------------------------------------------------------------
# Model answers — filled into the answer cells of the solution only.
# Keyed by the index of the answer cell, counted in order.
# ----------------------------------------------------------------------------------------------
ANSWERS = [
    # Q1
    '''
for i in range(len(world_names)):
    print(f"{world_names[i]}: {round(world_temps[i], 1)} K")
''',
    # Q2
    '''
liquid_water_worlds = []
for i in range(len(world_names)):
    if world_temps[i] >= 273 and world_temps[i] <= 373:
        liquid_water_worlds.append(world_names[i])

print(len(liquid_water_worlds), "of the four are in the window:", liquid_water_worlds)
''',
    # Q3
    '''
in_window = 0
for temperature in temps:
    if temperature >= 273 and temperature <= 373:
        in_window = in_window + 1

print(in_window, "of the", len(temps), "planets are between 273 K and 373 K")
''',
    # Q4
    '''
def equilibrium_temperature(star_temp, star_radius, distance, albedo):
    """The temperature starlight alone would give a planet, in kelvin."""
    return (star_temp * (1 - albedo) ** 0.25
            * (star_radius * SUN_RADIUS_IN_AU / (2 * distance)) ** 0.5)


print(round(equilibrium_temperature(SUN_TEMP, 1.0, 1.523679, 0.0), 1), "K")
''',
    # prediction: guess_earth
    '''
guess_earth = 288
''',
    # Q5
    '''
labels = ["Earth, albedo 0", "Earth, albedo 0.306", "Venus, albedo 0", "Venus, albedo 0.770"]
answers = [equilibrium_temperature(SUN_TEMP, 1.0, 1.000000, 0.0),
           equilibrium_temperature(SUN_TEMP, 1.0, 1.000000, 0.306),
           equilibrium_temperature(SUN_TEMP, 1.0, 0.723332, 0.0),
           equilibrium_temperature(SUN_TEMP, 1.0, 0.723332, 0.770)]

for i in range(len(labels)):
    if answers[i] >= 273 and answers[i] <= 373:
        verdict = "inside the window"
    else:
        verdict = "outside it"
    print(f"{labels[i]:22s} {round(answers[i], 1):6} K   {verdict}")

print(f"You guessed {guess_earth} K; with its measured albedo Earth comes out at "
      f"{round(answers[1], 1)} K, a miss of {round(answers[1] - guess_earth, 1)} K.")
''',
    # Q6
    '''
print("273 to 373 K:", count_worlds(273, 373, 0.0), "worlds")
print("250 to 350 K:", count_worlds(250, 350, 0.0), "worlds")
print("273 to 323 K:", count_worlds(273, 323, 0.0), "worlds")
''',
    # HW1
    '''
def window_edges(star_temp, star_radius, albedo):
    """The two distances in AU between which a planet round this star would be 273-373 K."""
    effective = star_temp * (1 - albedo) ** 0.25
    inner = star_radius * SUN_RADIUS_IN_AU / (2 * (373 / effective) ** 2)
    outer = star_radius * SUN_RADIUS_IN_AU / (2 * (273 / effective) ** 2)
    return inner, outer


sun_inner_0, sun_outer_0 = window_edges(SUN_TEMP, 1.0, 0.0)
sun_inner_earth, sun_outer_earth = window_edges(SUN_TEMP, 1.0, 0.306)

print(f"albedo 0.000: {sun_inner_0:.3f} to {sun_outer_0:.3f} AU")
print(f"albedo 0.306: {sun_inner_earth:.3f} to {sun_outer_earth:.3f} AU")

if sun_inner_0 < 1.0 and 1.0 < sun_outer_0:
    print("Earth is inside the albedo-0 window")
else:
    print("Earth is outside the albedo-0 window")

if sun_inner_earth < 1.0 and 1.0 < sun_outer_earth:
    print("Earth is inside the albedo-0.306 window")
else:
    print("Earth is outside the albedo-0.306 window")
''',
    # HW2
    '''
my_albedo = 0.770

my_worlds = []
my_temps = []
for i in range(len(names)):
    temperature = equilibrium_temperature(star_temps[i], star_radii[i], distances[i], my_albedo)
    if radii[i] is not None and radii[i] < 1.6 and temperature >= 273 and temperature <= 373:
        my_worlds.append(names[i])
        my_temps.append(temperature)

stayed = 0
moved_in = 0
for name in my_worlds:
    if name in rocky_worlds:
        stayed = stayed + 1
    else:
        moved_in = moved_in + 1

print(len(my_worlds), "worlds at albedo", my_albedo, "-", stayed, "stayed and", moved_in, "are new")
''',
    # HW3 prose
    '''
I took **Venus's albedo, 0.770**, and the test came back with 59 worlds, not one of which was on
class's list of twelve — 0 stayed, 59 are new. The first name on my list is **K2-415 b at 285.8 K**,
and that is the one I do not believe, precisely because it is the most believable: 285.8 K is two
degrees off the 288.0 K this planet actually sits at.

What the test measured about K2-415 b is three numbers and a size: a star at 3173 K, an orbit at
0.027 AU, and a radius under 1.6 Earths. What it did not measure is anything about the planet's
air — whether it has any, what it is made of, or how much. That is the whole of the difference
between the two worlds class weighed. Earth and Venus came out of the same formula at 254.0 K and
226.6 K, 27 K apart and with Venus the cooler one, and then reality put them at 288.0 K and
737.0 K. Earth's air added 34.0 K; Venus's added 510.4 K.

So 285.8 K is not a temperature, it is a starting point. Add the two corrections class measured and
the real surface lands at 319.8 K if K2-415 b's air behaves like Earth's and 796.2 K if it behaves
like Venus's — habitable at one end, hotter than Venus's own 737.0 K at the other. For K2-415 b to
be habitable its
atmosphere would have to be thin and transparent in the infrared, adding a few tens of kelvin the
way Earth's does. For it not to be, the atmosphere would only have to be thick — and at 0.027 AU
from a cool star, close enough that tidal locking is the expected outcome and the planet has been
soaking in stellar wind and ultraviolet for its whole life, either a thick CO2 blanket or no
atmosphere left at all is at least as likely a history as ours.

The part that bothers me most is not any one planet. Choosing Venus's albedo rather than 0 threw
out every single world class accepted and handed me 59 different ones. If which number I assume
decides the entire membership of the list, the list is measuring my assumption and not the planets.
''',

]


def build():
    sol = nbf.v4.new_notebook()
    stu = nbf.v4.new_notebook()
    ai = 0
    for c in CELLS:
        if c["t"] == "md":
            if c["answer"]:
                sol.cells.append(nbf.v4.new_markdown_cell(ANSWERS[ai].strip("\n")))
                stu.cells.append(nbf.v4.new_markdown_cell(PROSE_STUB))
                ai += 1
            else:
                sol.cells.append(nbf.v4.new_markdown_cell(c["src"]))
                stu.cells.append(nbf.v4.new_markdown_cell(c["src"]))
        else:
            if c["answer"]:
                sol.cells.append(nbf.v4.new_code_cell(ANSWERS[ai].strip("\n")))
                stu.cells.append(nbf.v4.new_code_cell(ANSWER_STUB.rstrip("\n")))
                ai += 1
            else:
                sol.cells.append(nbf.v4.new_code_cell(c["src"]))
                stu.cells.append(nbf.v4.new_code_cell(c["src"]))
    if ai != len(ANSWERS):
        raise SystemExit(f"{ai} answer cells but {len(ANSWERS)} model answers")
    for nb in (sol, stu):
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                     "name": "python3"}
    return sol, stu


def execute(sol):
    """Run the solution on a fresh kernel and keep every output."""
    NotebookClient(sol, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(OUT)}}).execute()
    return sol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-exec", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if not args.no_cache:
        print("cached fallback:")
        write_cache()

    sol, stu = build()
    nbf.write(stu, OUT / "02_liquid_water.ipynb")
    if args.no_exec:
        nbf.write(sol, OUT / "02_liquid_water_solution.ipynb")
    else:
        nbf.write(execute(sol), OUT / "02_liquid_water_solution.ipynb")
    n_q = sum(1 for c in CELLS if c["answer"])
    print(f"\n{len(CELLS)} cells, {n_q} answer cells "
          f"-> {OUT / '02_liquid_water.ipynb'}\n{'':21}-> "
          f"{OUT / '02_liquid_water_solution.ipynb'}")


if __name__ == "__main__":
    main()

# The two gates. A build that has not passed these is not a build.
weekkit.gate(2)
