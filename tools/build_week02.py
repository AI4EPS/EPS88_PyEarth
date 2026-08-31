#!/usr/bin/env python
"""Build week 2 — "Which of these worlds could have liquid water?" — as both notebooks.

The SOLUTION is written here in full; the STUDENT copy is derived by replacing every cell
tagged `answer` with the standard stub, so the two cannot drift apart. Run:

    python tools/build_week02.py

It writes docs/notebooks/02_liquid_water_solution.ipynb (executed) and
docs/notebooks/02_liquid_water.ipynb (clean), then calls weekkit.gate(2).
"""
import pathlib, sys

import nbformat
import yaml
from nbclient import NotebookClient

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
COURSE = yaml.safe_load((ROOT / "course.yml").read_text())
PLATFORM = COURSE["platform"]
WEEK = next(s for s in COURSE["schedule"] if s["n"] == 2)

DATAHUB = (f"{PLATFORM['datahub']}/hub/user-redirect/git-pull"
           f"?repo={PLATFORM['repo'].replace(':', '%3A')}"
           f"&branch={PLATFORM['branch']}"
           f"&urlpath=lab/tree/EPS88_PyEarth/{PLATFORM['notebook_dir']}/{WEEK['slug']}.ipynb")

CODE_STUB = "# ← your answer here\n"
PROSE_STUB = "*(Double-click this cell and replace this line with your answer.)*"

cells = []


def md(text):
    cells.append(("markdown", text.strip("\n"), False))


def code(text):
    cells.append(("code", text.strip("\n"), False))


def answer_code(text):
    cells.append(("code", text.strip("\n"), True))


def answer_md(text):
    cells.append(("markdown", text.strip("\n"), True))


def check_print(label, *parts):
    """one weekkit.CHECK_LINE print, wrapped so no source line runs past the page"""
    line = weekkit.CHECK_LINE.format(label=label, summary=parts[0])
    if len(parts) == 1:
        return f'print(f"{line}")'
    rest = "".join(f'\n      f"{q}"' for q in parts[1:])
    return f'print(f"{line} "{rest})'


# ---------------------------------------------------------------- 0. front matter
md(weekkit.OPENING.format(
    question=WEEK["question"],
    datahub=DATAHUB,
    hook="Astronomers have found six thousand planets around other stars. For most of them we "
         "know three numbers and nothing else: how hot the star is, how big it is, and how far "
         "out the planet orbits. Today you turn those three numbers into a temperature, and a "
         "temperature into a verdict."))

md("""
## The question

Liquid water is the one thing every living system we know of needs. So the first question anyone
asks about a newly found planet is whether it could hold any — and the honest answer usually has
to come from almost nothing. Nobody has photographed the surface of a planet around another star.
For most of the six thousand we have found, the archive holds the star's temperature, the star's
size, and the width of the planet's orbit.

That is enough to compute something: the temperature a planet would sit at if starlight were the
only thing heating it. Today you write that calculation as a **function**, run it over three
thousand worlds with a **loop**, and let an **if** statement sort them into ones that could hold
liquid water and ones that could not.

Then you run the same test on Earth, whose answer we already know. **Which of these worlds could
have liquid water — and why does your test reject Earth?**
""")

md("""
## What you'll be able to do

**The science.** Say what a planet's equilibrium temperature is, and compute it from three numbers
an archive will give you for almost any planet. Measure the difference between that temperature and
the real one, on the three worlds where we have both. And say what that difference is made of.

**The code.** `for` loops over a list and over `range(n)` · the accumulator pattern, with
`list.append` · `if` / `elif` / `else` and the comparison operators · `and`, `or`, `not` · `None`,
and why you cannot compare it with a number · `abs` · writing your own functions with `def`,
arguments, `return` and a docstring · `help` · and marking a plot up with `plt.axvline` and
`plt.text`.
""")

# ---------------------------------------------------------------- setup
md("""
## Setup

Run the next cell once. You are not expected to follow it — it is here so that everything below it
is short.

**Coming later:** it uses **pandas**, the tables library we meet properly in the tables week, to
fetch the NASA Exoplanet Archive's list of confirmed planets and hand it back as six ordinary
**lists** — one per column, all in the same order, so position 40 of each list is the same planet.
Lists and loops are this week's subject; pandas is not.

The archive is a live catalogue: it grows, and rows get revised. Every count printed in this
notebook came from the copy stored with the course, so if you run it live and get a slightly
different number, that is the archive having moved, not you having made a mistake.
""")

code(weekkit.setup_cell(
    figsize="(7, 4.5)",
    cache_base=PLATFORM["cache_base"],
    docstring=("the NASA Exoplanet Archive's confirmed planets: live if the network is up, "
               "the copy stored with the course if not"),
    url_expr=('"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="\n'
              '                           "select+pl_name,st_teff,st_rad,pl_orbsmax,pl_rade,'
              'pl_eqt"\n'
              '                           "+from+ps+where+default_flag=1&format=csv"'),
    cache_expr='"week02_exoplanets.csv"',
    unpack='''
archive = load()
archive = archive.astype(object).where(archive.notna(), None)


def column(name):
    """one column of the archive as an ordinary list, with None wherever it has no value"""
    return list(archive[name])


planet_names = column("pl_name")     # what the planet is called
star_temps = column("st_teff")       # its star's surface temperature, in K
star_radii = column("st_rad")        # its star's radius, in radii of the Sun
distances = column("pl_orbsmax")     # the width of the planet's orbit, in AU
planet_radii = column("pl_rade")     # the planet's radius, in radii of the Earth
archive_temps = column("pl_eqt")     # the archive's own equilibrium temperature, where it has one

print("planets in the archive:", len(planet_names))
'''.strip("\n")))

# ---------------------------------------------------------------- 1. loops
md("""
## 1. Doing the same thing to every world

Before three thousand planets, three. Venus, Earth and Mars are the three rocky worlds we have
measurements from, and they are the only planets anywhere whose surface temperature we know from
having been there.

The numbers below are from NASA's NSSDC planetary fact sheets. Those sheets are no longer on the
live web — `nssdc.gsfc.nasa.gov/planetary/factsheet/` redirects to a NASA landing page that does
not carry them — so what is cited here is the last archived copy: the Internet Archive's capture
of [the Earth sheet](https://web.archive.org/web/20250820/https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html),
[Venus](https://web.archive.org/web/20250820/https://nssdc.gsfc.nasa.gov/planetary/factsheet/venusfact.html)
and [Mars](https://web.archive.org/web/20250820/https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html),
taken **2025-08-20**, read 2026-08-31. Those links open; the original address does not.

**Bond albedo** is the fraction of sunlight a world reflects, and Earth's is the one number here
that needs an argument, because the sheet does not agree with itself. It prints a Bond albedo of
0.294 — and, four rows down, a black-body temperature of 254.0 K. Those two cannot both be right:
0.294 gives 255.1 K, and only 0.306 reproduces the 254.0 the sheet itself prints. This notebook
takes 0.306, the value that makes the page self-consistent, and once you have the formula and the
albedo term — section 4 — you can check that for yourself. Every temperature below moves if you
change it, which is why the value, where it came from and when it was read all have to be written
down.

Five lists, in the same order — Venus, Earth, Mars, going outwards from the Sun. A **`for` loop**
does the same thing once for each item in a list.
""")

code('''
worlds = ["Venus", "Earth", "Mars"]
sun_distances = [0.723, 1.000, 1.524]      # width of the orbit, in AU
bond_albedos = [0.770, 0.306, 0.250]       # fraction of sunlight reflected
surface_temps = [737.0, 288.0, 214.0]      # measured surface temperature, in K
air_pressures = [92.0, 1.014, 0.006]       # atmospheric pressure at the surface, in bar

for world in worlds:
    print(world)
''')

md("""
`world` is a name the loop invents for you: on the first pass it holds `"Venus"`, on the second
`"Earth"`, on the third `"Mars"`. The indented lines run once per item.

That form is fine when you need one list. With five lists you need the **position** instead, so
that you can read the same position out of all of them — the trick from last week, where the
position of the largest magnitude also found the place name. `range(n)` gives you the whole run of
positions: `range(3)` is 0, 1, 2. And `len(worlds)` is 3, so `range(len(worlds))` is every position
in the list without your having to count them.
""")

code('''
for i in range(len(worlds)):
    print(worlds[i], "orbits at", sun_distances[i], "AU and its surface sits at",
          surface_temps[i], "K under", air_pressures[i], "bar of air")
''')

md("""
Three lines of output, and the third column already carries a puzzle: Venus's air is 92 bar against
Earth's 1.014, and Mars's is 0.006. Hold on to that.

The other thing a loop is for is building something up as it goes. **Set a counter to 0 before the loop starts and add to it inside; same idea for a list — start
it empty and append inside.** `results.append(x)` adds one item to the end of the list `results`.
""")

md("""
✏️ **Your turn.** Sunlight thins out with distance: a planet twice as far from its star catches a
quarter as much. So the sunlight reaching a world, compared with Earth, is `1 / distance ** 2` —
where `**` means "to the power of", so `distance ** 2` is the distance squared.

Start an empty list called `sunlight`, loop over the three worlds, and append each one's share of
sunlight compared with Earth, rounded to two decimal places with `round(value, 2)`. Print the list
when the loop has finished.

**Use these names**: `sunlight`.
""")

answer_code('''
sunlight = []
for i in range(len(worlds)):
    sunlight.append(round(1 / sun_distances[i] ** 2, 2))

print("sunlight compared with Earth, for", worlds, "-", sunlight)
''')

# ---------------------------------------------------------------- 2. equilibrium temperature
md("""
## 2. The temperature starlight alone would give

A planet sits in a rain of starlight. Some of that light bounces straight off; the rest is absorbed
and warms the planet up. A warm planet glows — in the infrared, not visibly — and the warmer it
gets the harder it glows, until it is losing exactly as much heat as it is catching. The
temperature at which those two rates balance is the planet's **equilibrium temperature**.

**Equilibrium temperature: the temperature a planet would sit at if starlight were the only thing heating it and it had
no air.**

Balancing those two rates gives one formula. It is standard — deriving it needs the
Stefan–Boltzmann law, which is not today's business — and it needs three numbers about the star and
the orbit, plus the fraction of light that bounces off:

```
temperature = star_temp * (star_radius / (2 * distance)) ** 0.5 * (1 - albedo) ** 0.25
```

- `star_temp` is the star's surface temperature in kelvin;
- `star_radius / distance` is how big the star looks from the planet, with both lengths in the
  same units; the extra `2` is not part of that view. It is there because the planet catches
  starlight on one circular face but glows from its whole surface — four times the area — so the
  heat it caught is spread over day and night alike;
- `** 0.5` is a square root and `** 0.25` a fourth root, so `(1 - albedo) ** 0.25` is the fourth
  root of the fraction of light the planet keeps.

For the Sun the two numbers we need are its surface temperature, 5,772 K, and its radius,
0.00465047 AU — the IAU's 2015 nominal solar values (Resolution B3), checked 2026-08-31. Start with
Earth, and with `albedo` left out entirely, as if the planet reflected nothing at all.
""")

code('''
sun_temp = 5772             # K, the Sun's surface temperature
sun_radius_au = 0.00465047  # the Sun's radius, in AU

earth_temp = sun_temp * (sun_radius_au / (2 * 1.000)) ** 0.5
print("Earth, reflecting nothing and with no air:", round(earth_temp, 1), "K")
''')

md("""
✏️ **Your turn.** Do the same for all three worlds. Loop over the positions in `worlds`, compute
each one's temperature the way the cell above did — reflecting nothing, `sun_distances[i]` in place
of the `1.000` — and print the world's name and its temperature rounded to one decimal place.

Nothing here is new; it is the range loop from section 1 with the arithmetic from the cell above
dropped inside it.
""")

answer_code('''
for i in range(len(worlds)):
    temperature = sun_temp * (sun_radius_au / (2 * sun_distances[i])) ** 0.5
    print(worlds[i], round(temperature, 1), "K")
''')

# ---------------------------------------------------------------- 3. conditionals
md("""
## 3. Deciding, in code

Now turn a temperature into a verdict. Water is liquid between its melting point and its boiling
point, and at Earth's sea-level pressure those are 273 K and 373 K. That is the window this
notebook will use throughout. It is a convention, and a shaky one: both ends move with pressure,
and on Mars's 0.006 bar of air water boils only a few degrees above freezing. We are adopting it
anyway, and noting what we adopted.

An **`if` statement** runs its indented block only when a comparison comes out true. `elif` (short
for "else if") offers another comparison to try if the first failed, and `else` catches everything
left. The comparisons themselves are `<`, `>`, `<=`, `>=`, `==` for "is equal to" and `!=` for "is
not". You can chain two of them: `273 <= temperature <= 373` is true when the temperature is inside
the window, which is shorter than saying it twice.

Here is the whole test, over the three worlds, with a counter accumulating alongside it.
""")

code('''
in_window = 0
for i in range(len(worlds)):
    temperature = sun_temp * (sun_radius_au / (2 * sun_distances[i])) ** 0.5
    if temperature < 273:
        verdict = "too cold"
    elif temperature > 373:
        verdict = "too hot"
    else:
        verdict = "liquid water possible"
        in_window = in_window + 1
    print(worlds[i], round(temperature, 1), "K -", verdict)

print("worlds the test accepts:", in_window)
''')

# ---------------------------------------------------------------- 4. albedo
md("""
## 4. The light a world throws away

Two of the three pass. One of the two is Earth, which is reassuring. The other is Venus — whose
surface temperature, in the first table of this notebook, is 737 K. That is 364 degrees above the
boiling point of water, and our test just said liquid water was possible there. Something is
missing, and section 1 already put it on screen: 92 bar of air.

The formula also has a term we have not used yet.

**Albedo: the fraction of the starlight falling on a world that it reflects straight back into space; only
the rest is absorbed and turned into heat.**

A world that reflects more of its starlight keeps less of it and runs cooler, and the three are
very different: Mars reflects a quarter of what reaches it, Earth 0.306 of it, and Venus — wrapped
in cloud from pole to pole — 0.770. Those are the `bond_albedos` from section 1, and putting them
into the formula means multiplying by `(1 - bond_albedos[i]) ** 0.25`.

### Predict before you run

Write a number down before you run the next cell. With each world's real albedo in the formula, how
many of the three end up inside the 273–373 K window: **0, 1, 2 or 3?** Venus reflects far more
than Earth does, so Venus should cool by more than Earth does. Commit to an answer, out loud, to
whoever is sitting next to you.
""")

code('''
albedo_temps = []
for i in range(len(worlds)):
    starlight = sun_temp * (sun_radius_au / (2 * sun_distances[i])) ** 0.5
    albedo_temps.append(starlight * (1 - bond_albedos[i]) ** 0.25)
    print(worlds[i], round(albedo_temps[i], 1), "K")
''')

md("""
None of them. Venus falls to 226.7 K and Earth to 254.0 K — nineteen degrees below freezing — and
Mars was never in. Run the test one way and it accepts Earth and Venus together; run it the other
way and it rejects Earth, Venus and Mars alike. It has not been right once.

So how wrong is it? We know what these three worlds are actually like.
""")

md("""
✏️ **Your turn.** Loop over the three worlds and print, for each, its measured surface temperature
from `surface_temps`, the temperature the test just gave it from `albedo_temps`, the difference
between them rounded to one decimal place, and its `air_pressures` value. Four numbers on a line,
and the third one is `surface_temps[i] - albedo_temps[i]`.

**Use these names**: `gap`, for the difference.
""")

answer_code('''
for i in range(len(worlds)):
    gap = surface_temps[i] - albedo_temps[i]
    print(f"{worlds[i]}: really {surface_temps[i]} K, test says {round(albedo_temps[i], 1)} K, "
          f"{round(gap, 1)} K warmer than the test, under {air_pressures[i]} bar of air")
''')

md("""
Mars, 0.006 bar of air: the test is out by 4.2 K. Earth, 1.014 bar: out by 34.0 K. Venus, 92 bar:
out by 510.3 K.

Same physics, same formula, three thicknesses of air, and the error tracks the air. That
difference has a name.

**The greenhouse effect: the difference between the temperature a world would have with no air and the temperature it
actually has.**

It is worth being clear about what just happened, because it is the opposite of how this usually
gets taught. Nobody defined the greenhouse effect and then went looking for it. We computed what
starlight alone would do, compared that with three thermometers, and the leftovers came out at 4.2,
34.0 and 510.3 K — in the order of how much air each world has. Mars to Venus is four orders of
magnitude in pressure, 0.006 bar to 92, and it comes out as two orders of magnitude in the
leftover. The greenhouse effect is that leftover, measured.

The figure below plots it. The diagonal is where a world with no air would sit, and each planet
stands above it by exactly the amount its own air adds — Mars by 4.2 K, so close to the line you
have to look for the gap, Venus by 510.3 K, which is almost the whole height of the plot. All
three sit to the left of the window.
""")

code('''
plt.plot([195, 390], [195, 390], color="0.6", lw=1)     # where a world with no air would sit
plt.scatter(albedo_temps, surface_temps, s=45)
plt.axvline(273, color="0.8")
plt.axvline(373, color="0.8")
for i in range(len(worlds)):
    plt.text(albedo_temps[i] + 4, surface_temps[i], worlds[i])
plt.text(285, 720, "liquid water window")
plt.text(330, 305, "no air at all")
plt.xlabel("equilibrium temperature, at the world's own albedo (K)")
plt.ylabel("measured surface temperature (K)")
plt.title(f"Starlight alone against reality (n = {len(worlds)})")
plt.show()
''')

# ---------------------------------------------------------------- 5. functions
md("""
## 5. Writing the test down once

You have now typed that formula four times. The next section runs it on three thousand planets, and
typing it a fifth time inside that loop would mean any correction had to be made in five places.

A **function** is the fix: write the recipe once, give it a name, and from then on ask for it by
name. `def` starts the definition, the names in brackets are what the function needs to be given,
and `return` hands one value back to whoever called it. The triple-quoted line just inside is a
**docstring**, and it says what the function is for.
""")

code('''
def equilibrium_temperature(star_temp, star_radius, distance, albedo):
    """the temperature starlight alone would hold a planet at, in kelvin"""
    return star_temp * (star_radius * sun_radius_au / (2 * distance)) ** 0.5 * (1 - albedo) ** 0.25


print("Earth at its own albedo:", round(equilibrium_temperature(5772, 1.0, 1.000, 0.306), 1), "K")
help(equilibrium_temperature)
''')

md("""
Two things changed on the way in. `star_radius` is now measured in radii of the Sun, because that
is the unit the archive uses — the function multiplies by `sun_radius_au` itself, so the Sun goes
in as `1.0`. And `albedo` is an argument rather than something baked in, so the same function
answers the albedo-0 question and the albedo-0.306 question without being rewritten.

`help(equilibrium_temperature)` printed the docstring back. That is what writing one buys you.

A function can also hand back **two** values at once: `return temperature, verdict` gives both, and
`t, v = check_planet(...)` catches them in that order.
""")

md("""
✏️ **Your turn.** Write a function called `check_planet` that takes the same four arguments —
`star_temp`, `star_radius`, `distance`, `albedo` — and returns two things: the temperature, and a
verdict string that is `"too cold"` below 273 K, `"too hot"` above 373 K, and
`"liquid water possible"` in between.

Give it a docstring. Inside it, call `equilibrium_temperature` rather than retyping the formula,
then use the same `if` / `elif` / `else` as section 3. Finish by calling it on Venus twice — once
with albedo `0.0` and once with albedo `0.770` — and printing both results.

**Use these names**: `check_planet`.
""")

answer_code('''
def check_planet(star_temp, star_radius, distance, albedo):
    """a planet's equilibrium temperature, and whether liquid water is possible there"""
    temperature = equilibrium_temperature(star_temp, star_radius, distance, albedo)
    if temperature < 273:
        verdict = "too cold"
    elif temperature > 373:
        verdict = "too hot"
    else:
        verdict = "liquid water possible"
    return temperature, verdict


print("Venus, reflecting nothing:", check_planet(5772, 1.0, 0.723, 0.0))
print("Venus, at its own albedo:", check_planet(5772, 1.0, 0.723, 0.770))
''')

# ---------------------------------------------------------------- 6. the archive
md("""
## 6. Three thousand other worlds

The setup cell loaded the NASA Exoplanet Archive: every confirmed planet around another star, one
row each. Six lists, all in the same order.

Most of those rows are incomplete, and that is the normal condition of an astronomical catalogue
rather than a flaw in it. Different discovery methods measure different things: a planet found by
watching it cross its star gives a radius, a planet found by watching its star wobble gives a mass,
and neither gives both. Where the archive has no value, the setup cell put **`None`** in the list —
Python's word for "nothing here".

`None` is not zero and it is not a small number: it is the absence of one, and Python refuses to
compare it with a number at all. `None < 1.6` is an error, not a `False`. That refusal is a gift.
The test for it is `is None`, or `is not None` for the other way round, and `and` joins two
conditions so that both have to hold.

The cell below keeps only the planets that carry all three numbers the formula needs.
""")

code('''
usable_names = []
usable_star_temps = []
usable_star_radii = []
usable_distances = []
usable_radii = []
usable_archive_temps = []
for i in range(len(planet_names)):
    if star_temps[i] is not None and star_radii[i] is not None and distances[i] is not None:
        usable_names.append(planet_names[i])
        usable_star_temps.append(star_temps[i])
        usable_star_radii.append(star_radii[i])
        usable_distances.append(distances[i])
        usable_radii.append(planet_radii[i])
        usable_archive_temps.append(archive_temps[i])

print("planets with all three numbers:", len(usable_names), "out of", len(planet_names))
''')

md("""
More than half the archive is gone before we start, and no amount of cleverness gets it back: those
planets were never measured that way. (Six appends to keep six lists in step is also the clearest
possible argument for the tables library we meet in a few weeks, which does this in one line.)

Before trusting our own arithmetic on three thousand worlds, check it against somebody else's. The
archive publishes its own equilibrium temperature, `pl_eqt`, for some planets — computed by whoever
wrote the discovery paper, with their own assumptions. Where both exist we can compare. `abs(x)`
gives the size of a difference without caring which way round it went, and the last three lines of
the loop are the other thing an accumulator does: hold on to the biggest one seen so far.
""")

code('''
ours = []
theirs = []
close = 0
above = 0
below = 0
worst_gap = 0
worst_planet = ""
worst_distance = 0
for i in range(len(usable_names)):
    if usable_archive_temps[i] is not None:
        temperature = equilibrium_temperature(usable_star_temps[i], usable_star_radii[i],
                                              usable_distances[i], 0.0)
        gap = abs(temperature - usable_archive_temps[i])
        ours.append(temperature)
        theirs.append(usable_archive_temps[i])
        if gap < 10:
            close = close + 1
        elif temperature > usable_archive_temps[i]:
            above = above + 1
        else:
            below = below + 1
        if gap > worst_gap:
            worst_gap = gap
            worst_planet = usable_names[i]
            worst_distance = usable_distances[i]

print("planets we can compare against:", len(ours))
print("of those, within 10 K of the published value:", close)
print("the rest, ours hotter than theirs:", above, " ours colder:", below)
print("the largest disagreement:", round(worst_gap), "K, on", worst_planet, "-",
      worst_distance, "AU from its star")
''')

code('''
plt.scatter(theirs, ours, s=3)
plt.plot([0, 4000], [0, 4000], color="0.6", lw=1)       # where perfect agreement would sit
plt.xlabel("the archive's published equilibrium temperature (K)")
plt.ylabel("our equilibrium temperature, albedo 0 (K)")
plt.title(f"Our arithmetic against the archive's (n = {len(ours)})")
plt.show()
''')

md("""
952 of 1525 agree within 10 K, and the cloud hugs the diagonal, so this is the calculation the
field uses. The loop counted the rest rather than leaving it to the eye: of the 573 that miss by
10 K or more, **456 sit above the line and 117 below**. The 456 are our own doing — we let every
planet absorb all the light it catches, and the discovery papers each used an albedo of their own,
so our answers run hot.

The 117 below the line are a different animal, and they are the ones your eye goes to, because they
run far off the diagonal instead of crowding beside it. The worst of them says what they are.
HD 284149 AB b orbits 431 AU out — four hundred times the width of Earth's orbit — and at that
distance the starlight our formula is built on has almost nothing left in it, so our answer
comes out near zero and the two disagree by 2379 K. The archive's number for that planet is not a
response to its star at all: a young planet photographed that far out is still glowing with the
heat of its own formation. The formula is not wrong about it. It is answering a question that does
not apply.

Agreement with the professionals is not agreement with reality, either. Everyone in that figure is
computing the same quantity, and section 4 already showed what that quantity leaves out.
""")

# ---------------------------------------------------------------- 7. the survey
md("""
## 7. Running the test on all of them

One more number, and it is a radius. A planet's temperature says nothing about whether it has a
surface: at about 1.6 Earth radii, planets stop being rock and start being small versions of
Neptune, with atmospheres thousands of kilometres deep and no ground under them. That line comes
from Rogers, *The Astrophysical Journal* **801**, 41 (2015) — "Most 1.6 Earth-radius planets are
not rocky", reference checked 2026-08-31. It is a convention, like the 273–373 K window, and worth
holding loosely.

So a planet inside the temperature window falls into one of three cases, and `None` makes the third
one unavoidable: rocky, too big to be rocky, or the archive never measured its radius.
""")

md("""
✏️ **Your turn.** Loop over the usable planets. For each one, call `check_planet` with albedo
`0.0` — the same "reflecting nothing" pass the class started with — and collect every temperature
into a list called `all_temps`, which the next cell plots.

When the verdict is `"liquid water possible"`, add one to `n_window`, and then sort that planet
into one of three:

- `usable_radii[i] is None` — add one to `unknown_radius`;
- `usable_radii[i] < 1.6` — append the planet's name to `rocky_names`;
- anything else — add one to `too_big`.

Then print the four counts, and print each rocky candidate's name, temperature and radius, one per
line. You will need that printout for the homework.

**Use these names**, because the self-check looks for them: `all_temps`, `n_window`,
`unknown_radius`, `rocky_names` and `too_big`.
""")

answer_code('''
all_temps = []
n_window = 0
unknown_radius = 0
too_big = 0
rocky_names = []
for i in range(len(usable_names)):
    temperature, verdict = check_planet(usable_star_temps[i], usable_star_radii[i],
                                        usable_distances[i], 0.0)
    all_temps.append(temperature)
    if verdict == "liquid water possible":
        n_window = n_window + 1
        if usable_radii[i] is None:
            unknown_radius = unknown_radius + 1
        elif usable_radii[i] < 1.6:
            rocky_names.append(usable_names[i])
        else:
            too_big = too_big + 1

print("inside the window:", n_window)
print("  of those, rocky:", len(rocky_names), " too big:", too_big,
      " radius never measured:", unknown_radius)
for i in range(len(usable_names)):
    if usable_names[i] in rocky_names:
        print(f"  {usable_names[i]}: {round(all_temps[i], 1)} K, {usable_radii[i]} Earth radii")
''')

code(f'''
assert len(all_temps) == len(usable_names), "all_temps needs one entry per planet, not per hit"
assert unknown_radius > 0, "unknown_radius never moved — add one to it inside the `is None` branch"
{check_print("Section 7",
             "{n_window} planets in the window: {len(rocky_names)} rocky,",
             "{too_big} too big, {unknown_radius} with no measured radius")}
''')

md("""
102 of the 189 have no published radius at all. That is not a rounding detail: it is more than half
the answer, and it is the difference between "there are 12 candidates" and "there are 12 candidates
and 102 unknowns". A test that had silently counted the unknowns as "not rocky" would have reported
the same 12 and looked far more confident than the data allows.

The figure below is every one of the three thousand temperatures, with the window marked.
""")

code('''
plt.hist(all_temps, bins=250)
plt.axvline(273, color="0.4")
plt.axvline(373, color="0.4")
plt.text(323, 105, "liquid water window", ha="center")   # ha= centres both on the band
plt.text(323, 94, "↓", ha="center")
plt.xlim(0, 2500)
plt.ylim(0, 115)                       # room above the bars for the label
plt.xlabel("equilibrium temperature at albedo 0 (K)")
plt.ylabel("number of planets")
plt.title(f"Every planet with the three numbers (n = {len(all_temps)})")
plt.show()
''')

md("""
Most known planets are far hotter than the window, because a planet close to its star is the
easiest kind to find — a bias in the catalogue, not in the galaxy.

Now the interesting part. Astronomers keep an informal list of the planets thought most likely to
be habitable, and it is short: the outer TRAPPIST-1 planets, Proxima Centauri b, TOI-700 d,
Kepler-186 f, Kepler-442 b. The cell below asks our test about those, and about two more that are
not on it.
""")

code('''
famous = ["TRAPPIST-1 c", "TRAPPIST-1 e", "TRAPPIST-1 f", "TRAPPIST-1 g",
          "Proxima Cen b", "TOI-700 d", "Kepler-186 f", "Kepler-442 b", "K2-18 b"]
for i in range(len(usable_names)):
    if usable_names[i] in famous:
        temperature, verdict = check_planet(usable_star_temps[i], usable_star_radii[i],
                                            usable_distances[i], 0.0)
        print(f"{usable_names[i]}: {round(temperature, 1)} K, radius {usable_radii[i]} - {verdict}")
''')

md("""
It rejects almost all of them. TRAPPIST-1 e, f and g, Proxima Cen b, TOI-700 d, Kepler-186 f and
Kepler-442 b all come out too cold, between 197.4 K and 267.8 K, on planets whose whole claim to
interest is that they might have air. That is the same failure the test made on Earth — but not
the same number, and the difference is worth being careful about. Every figure on that line is an
albedo-0 figure, and Earth at albedo 0 is 278.3 K and *passes*; it was Earth's own albedo, 0.306,
that pushed it down to 254.0 K. Two numbers computed at two different albedos are not comparable,
and the temptation to line them up anyway is exactly what makes this test easy to misread. What
does repeat, at either albedo, is the reason: a formula that knows nothing about air, asked about
worlds whose whole interest is their air. (Proxima Cen b prints its radius as `None`: it was found
by watching its star wobble, and that method never measures a size.)

It accepts two. TRAPPIST-1 c comes out at 339.9 K, hotter than any of its siblings we looked at,
and K2-18 b at 282.7 K — and K2-18 b is 2.37 Earth radii, a size our own radius branch had already
counted as too big to be rock, so the verdict string and the radius disagree about the same planet.
K2-18 b is also the planet the field has argued over hardest in recent years, water vapour in its
atmosphere and then a series of contested readings of its JWST spectrum, so "liquid water possible"
is the one verdict on that line nobody should read as a finding.

That accept-and-reject set is worth staring at, because it is strange without being random. Look
back at section 7's twelve rocky candidates: TRAPPIST-1 d, TOI-700 e, Kepler-438 b and K2-72 e are
all planets the field itself puts forward, so the test is not picking out some different set of
worlds from the ones astronomers care about. What it is doing is sorting **within** those systems
by temperature alone, keeping the warmer planet and dropping the cooler one — TRAPPIST-1 c and d
in, e, f and g out. That is the only thing a bare rock with no air can be sorted by, and it is
exactly the sorting that put Earth on the wrong side of the line.
""")

# ---------------------------------------------------------------- the question, answered
md("""
## The question, answered

Reflecting nothing, the test accepts 189 planets — 12 of them small enough to be rock — and it
accepts Earth at 278.3 K and Venus at 327.3 K together. At each world's measured albedo it rejects
Earth at 254.0 K, Venus at 226.7 K and Mars at 209.8 K alike. It rejects Earth because equilibrium
temperature is the temperature of a bare rock in sunlight, and Earth is not one: 34.0 K of what
Earth has comes from its air, and 510.3 K of what Venus has comes from its own. The test is not
broken. It is answering a narrower question than the one we asked it, and the gap between the two
is the greenhouse effect.
""")

md(weekkit.week_cheatsheet(2))

# ---------------------------------------------------------------- homework
md("""
## Homework

Three parts, on the same archive and the same tools. Part 1 is the one everybody finishes; part 3
is the one that takes thinking rather than typing.

Run the cell below first if you have restarted the kernel — after the setup cell at the top, it
rebuilds everything parts 1 and 2 use: the two constants, `equilibrium_temperature`, and the five
`usable_` lists from section 6. Then you do not have to hunt back through the notebook.
""")

code(weekkit.CHECKPOINT.format(body='''
sun_temp = 5772
sun_radius_au = 0.00465047


def equilibrium_temperature(star_temp, star_radius, distance, albedo):
    """the temperature starlight alone would hold a planet at, in kelvin"""
    return star_temp * (star_radius * sun_radius_au / (2 * distance)) ** 0.5 * (1 - albedo) ** 0.25


usable_names = []
usable_star_temps = []
usable_star_radii = []
usable_distances = []
usable_radii = []
for i in range(len(planet_names)):
    if star_temps[i] is not None and star_radii[i] is not None and distances[i] is not None:
        usable_names.append(planet_names[i])
        usable_star_temps.append(star_temps[i])
        usable_star_radii.append(star_radii[i])
        usable_distances.append(distances[i])
        usable_radii.append(planet_radii[i])
'''.strip("\n")))

md("""
✏️ **Part 1 — what kind of stars are these?**

Class asked which planets could hold liquid water and never asked what they orbit. A star's
temperature is the first number the formula takes, and it has been sitting in `usable_star_temps`
all along.

It carries more than it looks. A star much cooler than the Sun is also much smaller and fainter,
so a planet warm enough for liquid water has to sit very close in — close enough to be locked with
one face to its star for good, and close enough that a stellar flare arrives at full strength.
Whether these candidates orbit Sun-like stars or cool ones is the most consequential thing about
the list, and nobody has looked yet.

Loop over the usable planets and, for every one that the albedo-0 test calls a rocky candidate —
inside the 273–373 K window **and** with a radius that is not `None` **and** below 1.6 — append its
name to `candidate_names` and its star's temperature to `candidate_star_temps`. Two lists growing
together inside the same `if`.

Then print: how many candidates there are; how many of them orbit a star cooler than the Sun's
5772 K, counted into `cooler_than_sun` with a second loop; and the value of the coolest star in
the set, `coolest`, together with the planet it belongs to. `min(candidate_star_temps)` gives you
the coolest value and `.index(...)` gives you its position, exactly as last week's `max` and
`.index` found the largest earthquake and then its place name.

One thing about `.index` that last week never made you face: it hands back the **first** position
holding that value and stops looking. Two planets of the same system orbit the same star, so they
carry the same star temperature to the last decimal — and if that star is the coolest one, `.index`
will name one of those planets and say nothing about the other. So finish by looping once more over
the candidates and printing each name beside its star's temperature, one per line. That printout is
what tells you whether the planet `.index` picked was alone, and it is the one to read before you
believe any single line above it.

**Use these names**, because the self-check looks for them: `candidate_names`,
`candidate_star_temps`, `cooler_than_sun` and `coolest`.
""")

answer_code('''
candidate_names = []
candidate_star_temps = []
for i in range(len(usable_names)):
    temperature = equilibrium_temperature(usable_star_temps[i], usable_star_radii[i],
                                          usable_distances[i], 0.0)
    if 273 <= temperature <= 373 and usable_radii[i] is not None and usable_radii[i] < 1.6:
        candidate_names.append(usable_names[i])
        candidate_star_temps.append(usable_star_temps[i])

cooler_than_sun = 0
for i in range(len(candidate_star_temps)):
    if candidate_star_temps[i] < 5772:
        cooler_than_sun = cooler_than_sun + 1

coolest = min(candidate_star_temps)
print("rocky candidates:", len(candidate_names))
print("of those, orbiting a star cooler than the Sun:", cooler_than_sun)
print("the coolest star of the set:", coolest, "K, around",
      candidate_names[candidate_star_temps.index(coolest)])

# .index stopped at the first match, so print them all and see what it passed over
for i in range(len(candidate_names)):
    print(f"  {candidate_names[i]}: star at {candidate_star_temps[i]} K")
''')

code(f'''
assert len(candidate_names) == len(candidate_star_temps), \\
    "the two lists must grow together — append to both inside the same if"
assert len(candidate_names) < 100, "that is everything in the window; the radius test is missing"
{check_print("Homework 1",
             "{len(candidate_names)} candidates, {cooler_than_sun} around stars",
             "cooler than the Sun, coolest {coolest} K")}
''')

md("""
✏️ **Part 2 — move one knob, and count what moves.**

Both of the choices class made were arguable. Pick **one** of these two changes, and only one.

- **Option A — the mirror.** Venus reflects 0.770 of the light that reaches it, and there is no
  reason a distant rocky planet should reflect nothing. Set `albedo = 0.770`, leave `low = 273` and
  `high = 373`.
- **Option B — the floor.** Air can only warm a planet, never cool it, so the albedo-0 equilibrium
  temperature is a floor rather than an estimate: a world 20 K below freezing on paper could still
  have liquid water underneath a thick atmosphere. Set `albedo = 0.0`, `low = 250` and
  `high = 373`.

Then rerun part 1's loop with your three values in place of the fixed ones, into a list called
`new_candidates`, and report what moved: how many candidates there are now, how many of part 1's
`candidate_names` are still on the list, how many dropped off and how many are new. `name in
candidate_names` is true when that name is somewhere in the list. (The first number stands on its
own; the other three need part 1 to have run.)

Say which option you took in a comment on the first line.

**Use these names**, because the self-check looks for them: `albedo`, `low`, `high`,
`new_candidates` and `stayed`.
""")

answer_code('''
# Option A — Venus's albedo, class's window
albedo = 0.770
low = 273
high = 373

new_candidates = []
for i in range(len(usable_names)):
    temperature = equilibrium_temperature(usable_star_temps[i], usable_star_radii[i],
                                          usable_distances[i], albedo)
    if low <= temperature <= high and usable_radii[i] is not None and usable_radii[i] < 1.6:
        new_candidates.append(usable_names[i])

stayed = 0
for i in range(len(new_candidates)):
    if new_candidates[i] in candidate_names:
        stayed = stayed + 1

print("candidates now:", len(new_candidates), "against", len(candidate_names), "before")
print("still on the list:", stayed)
print("dropped off:", len(candidate_names) - stayed, " newly on:", len(new_candidates) - stayed)
''')

code(f'''
assert albedo == 0.770 or low == 250, "neither knob moved — set up option A or option B"
assert len(new_candidates) != len(candidate_names), "the count did not move; check what you changed"
{check_print("Homework 2",
             "albedo {albedo}, window {low}-{high} K:",
             "{len(new_candidates)} candidates, {stayed} of part 1's still there")}
''')

md("""
✏️ **Part 3 — one planet you do not believe.**

Section 7's printout lists every planet this week's test calls a rocky candidate, with its
temperature and its radius. Pick **one** of them that you do not believe could have liquid water,
and make the case against it in two or three sentences, in the cell below.

Quote real numbers, not impressions: that planet's own temperature and radius from the printout, and
how far the same test was wrong about Earth and about Venus — both of those differences were printed
in section 4. Then finish with one sentence naming what the test does not know about your planet
that would settle it.

There is no single right answer. There is a defensible one with three numbers in it.
""")

answer_md("""
I do not believe TRAPPIST-1 c. The test gives it 339.9 K and the archive gives it 1.097 Earth
radii, so it reads as a rock sitting comfortably inside the liquid-water window — but the same test,
on the same formula, put Earth at 254.0 K when Earth is really 288.0 K, and put Venus at 226.7 K
when Venus is really 737.0 K. Being wrong by 34.0 K is survivable; being wrong by 510.3 K, as it was
for Venus, is not. And TRAPPIST-1 c is the hottest planet in its own system that our test accepted,
at 339.9 K against TRAPPIST-1 d's 286.3 K, so it is the one in that system with the least room
before it goes the way of Venus.

What the test does not know is how much air TRAPPIST-1 c has. Its answer would be identical for a
bare rock, for something Earth-like under 1.014 bar, and for something under 92 bar the way Venus
is — and those three would sit hundreds of degrees apart. It knows nothing about the star either,
and part 1 is where that shows. All twelve candidates orbit stars cooler than the Sun's 5772 K,
but that count flatters the pattern: Kepler-1126 c's star is 5678 K, which is barely cooler at all.
The other eleven run from 2566 K to 4262 K, and TRAPPIST-1's 2566 K is less than half the Sun's. A
star that cool is small and faint, so the only way TRAPPIST-1 c reaches 339.9 K is by orbiting very
close in — close enough to be tidally locked and to take its star's flares at full strength. My
part-1 printout also shows TRAPPIST-1 d sharing that same 2566 K, which is `.index` reporting one
planet where there were two. None of that is a number the formula ever asks for.
""")


# ---------------------------------------------------------------- emit
def build():
    solution = nbformat.v4.new_notebook()
    student = nbformat.v4.new_notebook()
    for kind, text, is_answer in cells:
        if kind == "markdown":
            solution.cells.append(nbformat.v4.new_markdown_cell(text))
            student.cells.append(nbformat.v4.new_markdown_cell(
                PROSE_STUB if is_answer else text))
        else:
            solution.cells.append(nbformat.v4.new_code_cell(text))
            student.cells.append(nbformat.v4.new_code_cell(
                CODE_STUB if is_answer else text))

    out = ROOT / PLATFORM["notebook_dir"]
    out.mkdir(parents=True, exist_ok=True)

    print(f"executing {len(solution.cells)} cells ...")
    NotebookClient(solution, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(out)}}).execute()

    for nb in (solution, student):
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                     "name": "python3"}
    nbformat.write(solution, out / f"{WEEK['slug']}_solution.ipynb")
    nbformat.write(student, out / f"{WEEK['slug']}.ipynb")
    print(f"wrote {WEEK['slug']}_solution.ipynb and {WEEK['slug']}.ipynb")


if __name__ == "__main__":
    build()
    weekkit.gate(2)
