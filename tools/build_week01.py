#!/usr/bin/env python
"""Build week 1 — "What was your birthquake?" — as both notebooks from one source.

The SOLUTION is written here in full; the STUDENT copy is derived by replacing every cell
tagged `answer` with the standard stub, so the two cannot drift apart. Run:

    python tools/build_week01.py

It writes docs/notebooks/01_birthquake_solution.ipynb (executed) and
docs/notebooks/01_birthquake.ipynb (clean), then calls weekkit.gate(1).
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
WEEK = next(s for s in COURSE["schedule"] if s["n"] == 1)

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
    hook="Somewhere in the world, on the day you were born, the ground broke. This notebook finds "
         "that earthquake — yours, not an example — and then asks what the whole catalogue of "
         "them looks like."))

md("""
## The question

Every day, somewhere, the ground breaks. The United States Geological Survey keeps a running list
of it: for every earthquake anybody has located, a time, a place name, a latitude and longitude, a
depth and a magnitude. It is one of the largest openly published records of anything that happens
on this planet, and anyone can read it — including you, from this notebook, in one line.

So: **what was your birthquake?** On the day you were born, what was the largest earthquake in the
world, and where was it?

That question is the way in rather than the point. Once you can pull one day out of the catalogue
you can pull a year, and a year of earthquakes turns out to draw a map that nobody drew: the edges
of the tectonic plates, in dots. By the end of today you will have found your own birthquake, and
you will have an argument — with a number attached — about how much of the catalogue you have to
read before it tells you anything at all.
""")

md("""
## What you'll be able to do

**The science.** Read a real earthquake catalogue. Find the largest earthquake in any span of
days. See that earthquakes fall along narrow lines rather than spreading out. Count how many small
earthquakes there are for each large one, and say where that count stops being trustworthy. And
say why a catalogue is a record of what was *recorded*, not of what happened.

**The code.** `print` · `type` · variables and arithmetic · f-strings · lists, `len`, `list[i]`,
`list[-1]` and `list[a:b]` · `max` and `min` · `list.index` · `round` · and your first figures,
with `plt.scatter`, `plt.hist` and `plt.plot`.
""")

# ---------------------------------------------------------------- setup
md("""
## Setup

Run the next cell once. You are not expected to follow it — it is here so that everything below it
is short.

**Coming later:** it uses **pandas**, the tables library we meet properly in the tables week, to
fetch earthquakes from the USGS and hand them back as six ordinary **lists**: times, latitudes,
longitudes, depths, magnitudes and place names. Lists are this week's subject; pandas is not. Each
of the five small helpers it defines says what it is for on its first line, so you can read what
they do without reading how.

One line below does something you have not seen: `times, lats, lons, depths, mags, places =
columns(...)` gives six names their values in one go, in the order they are written.
""")

code(weekkit.SETUP_CELL.format(
    figsize="(7, 4)",
    cache_base=PLATFORM["cache_base"],
    signature="start, end, minmag",
    docstring=("one slice of the USGS earthquake catalogue: live if the network is up, "
               "the cached copy if not"),
    url_expr=('"https://earthquake.usgs.gov/fdsnws/event/1/query"\n'
              '                           "?format=csv&orderby=time-asc"\n'
              '                           f"&starttime={start}&endtime={end}'
              '&minmagnitude={minmag}"'),
    cache_expr='f"week01_{start}_{end}_M{minmag}.csv"',
    unpack='''
def count(start, end, minmag):
    """how many earthquakes of at least this magnitude the catalogue lists in this window"""
    return len(load(start, end, minmag))


def report_window(label, start, end):
    """print how many M6.5+ and M7.5+ a window holds, and the ratio between them"""
    big = count(start, end, 6.5)
    huge = count(start, end, 7.5)
    print(label, big, "at M6.5+ and", huge, "at M7.5+, ratio", round(big / huge, 1))


def columns(quakes):
    """the six columns we use, handed back as six ordinary lists in the same order"""
    return (list(quakes["time"]), list(quakes["latitude"]), list(quakes["longitude"]),
            list(quakes["depth"]), list(quakes["mag"]), list(quakes["place"]))


def load_yours(start, end, minmag):
    """the same query for dates only you know — your own birthday has no cached copy"""
    try:
        return pd.read_csv("https://earthquake.usgs.gov/fdsnws/event/1/query"
                           "?format=csv&orderby=time-asc"
                           f"&starttime={start}&endtime={end}&minmagnitude={minmag}")
    except Exception as e:
        raise RuntimeError("Could not read the catalogue for those dates. Check that they are "
                           "real dates written as YYYY-MM-DD and that you are online, then run "
                           "this cell again.") from e


coast = pd.read_csv(CACHE + "/coastlines.csv")
'''.strip("\n")))

code('''
times, lats, lons, depths, mags, places = columns(load("1983-12-02", "1983-12-03", 4.5))
print("earthquakes the catalogue lists for 2 December 1983:", len(mags))
''')

# ---------------------------------------------------------------- 1. cells, values, names
md("""
## 1. Cells, values and names

The grey box above is a **cell**. Click it, press **Shift+Enter**, and Python runs what is inside.
The notebook keeps a memory of everything it has run — that memory is the **kernel** — which is
why the cell below can use names the cell above created.

A **value** is one piece of data. `14` is a whole number, an `int`. `4.5` is a decimal, a `float`.
`"Champerico, Guatemala"` is text, a `str`, and the quotes are part of how you write it. A **name**
is a label you stick on a value with `=`, so you can use it again without retyping it.
""")

code('''
n_quakes = len(mags)
minutes_in_a_day = 24 * 60
print(n_quakes, type(n_quakes))
print(minutes_in_a_day, type(minutes_in_a_day))
''')

md("""
✏️ **Your turn.** Make a name called `minutes_each` holding `minutes_in_a_day` divided by
`n_quakes`: how many minutes of that day there were for each earthquake in it. (That is a rate,
not a stopwatch reading — the earthquakes did not arrive politely spaced out, and we have not
looked at when they arrived yet.)

Then print one sentence reporting it to **one** decimal place. Two tools for that.
`round(minutes_each, 1)` trims the decimals. And an **f-string** is a piece of text with an `f` in
front of the opening quote, where anything in `{curly braces}` is replaced by its value:

```python
print(f"there were {n_quakes} earthquakes")
```

**Use these names**: `minutes_each`.
""")

answer_code('''
minutes_each = minutes_in_a_day / n_quakes
print(f"That day held one earthquake for every {round(minutes_each, 1)} minutes of it.")
''')

# ---------------------------------------------------------------- 2. a day, as lists
md("""
## 2. A day of earthquakes, as lists

`mags` is a **list**: fourteen numbers in a row, in the order the earthquakes happened. The other
five lists hold the other five columns, in that same order — so position 3 of `mags` and position 3
of `places` are the same earthquake. That is the whole trick of this section.

Counting starts at **0**: `mags[0]` is the first, `mags[1]` the second. `mags[-1]` counts from the
far end and gives you the last. And `mags[0:3]` is a **slice** — items 0, 1 and 2, up to but not
including 3.
""")

code('''
print("the first magnitude of the day:", mags[0])
print("the last magnitude of the day:", mags[-1])
print("the first three:", mags[0:3])
''')

md("""
Two more tools, and they work together. `max(mags)` gives the largest number in a list — `min`
gives the smallest. `mags.index(v)` gives the **position** of the value `v`; and because all six
lists are in the same order, that position reads the same earthquake out of any of the others.

The dot in `mags.index(...)` is new. It means "ask this list to do something", and `index` is the
thing being asked for.
""")

code('''
biggest = max(mags)
where = mags.index(biggest)
print("largest magnitude that day:", biggest)
print("its position in the list:", where)
print("it happened at:", times[where])
''')

md("""
✏️ **Your turn.** `where` is the position of the largest earthquake of the day. Read that same
position out of `places` and out of `depths`, and print both — so that we know where the day's
biggest earthquake was, and how far below the surface it broke.

**Use these names**: `where`, `places`, `depths`.
""")

answer_code('''
print("place:", places[where])
print("depth in km:", depths[where])
''')

md("""
Now look again at what `where` came out as: **0**. The largest earthquake of the day was also the
*first* of the day. The list is in time order, and nothing bigger followed.

That matters, because the fourteen are not fourteen separate places. Three others name the same
town in Guatemala, later the same day, and five more share one spot in the Indian Ocean. There is
no way yet to search a list for a word — searching arrives with loops — so here are both groups by
hand.
""")

code('''
print(times[0], mags[0], places[0])
print(times[2], mags[2], places[2])
print(times[4], mags[4], places[4])
print(times[6], mags[6], places[6])

print(times[5], mags[5], places[5])
print(times[8], mags[8], places[8])
print(times[9], mags[9], places[9])
print(times[11], mags[11], places[11])
print(times[12], mags[12], places[12])
''')

md("""
Two very different shapes. The Guatemala four open with the largest and get smaller: one earthquake
broke, and the ground around it kept adjusting — those are its aftershocks. The Indian Ocean five
have no such leader; the biggest of them is the fourth to arrive. A sequence with no dominant
earthquake is called a **swarm**, and what causes one is a live question rather than a settled
answer.
""")

# ---------------------------------------------------------------- 3. maps
md("""
## 3. Putting earthquakes on a map

A map is a scatter plot: longitude across, latitude up. `plt.scatter(lons, lats)` puts one dot at
every pair. On its own that would be dots in a white rectangle, so every map in this course first
draws `coast` — a long list of coastline points that the setup cell loaded — with `plt.plot`, which
joins points up with a line.
""")

code('''
# ── Checkpoint ── run this if you restarted the kernel or fell behind ──
times, lats, lons, depths, mags, places = columns(load("1983-12-02", "1983-12-03", 4.5))
''')

code('''
plt.plot(coast.lon, coast.lat, color="0.6", lw=0.6)
plt.scatter(lons, lats, s=25)
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title(f"Earthquakes M4.5+ on 2 December 1983 (n = {len(mags)})")
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.gca().set_aspect("equal")
plt.show()
''')

md("""
Fourteen earthquakes, and fewer dots than that: the Guatemala four sit on top of one another at
this scale, the Indian Ocean five very nearly do, so ten places is all the map has to work with.
Ten places scattered over a globe cannot show you a shape, and no amount of staring will make them.
A thin figure is not a failed figure as long as you say out loud that it is thin. Hold on to this
one.

The only fix is more data, and the only thing that has to change is the two dates. Here is a whole
year, at the same magnitude floor.
""")

code('''
times, lats, lons, depths, mags, places = columns(load("1983-01-01", "1984-01-01", 4.5))
print("earthquakes M4.5 and above in that whole year:", len(mags))
print("the largest of them:", max(mags))
''')

md("""
✏️ **Your turn.** Before you write anything, commit to one of these. When those earthquakes go on
the map, do they come out **(a)** sprinkled fairly evenly over the globe, **(b)** crowded around
the rim of the Pacific and almost nowhere else, or **(c)** strung along narrow lines all over the
world, oceans included? Say it out loud to whoever is next to you. A guess you committed to is
worth more than one you kept your options open on.

Now draw it: the same map as the cell above, using the year's `lons` and `lats`. Two things need
to change and no more — a smaller dot (`s=3`, because four thousand dots at `s=25` is a blot) and
a title that says what is on the map and how many earthquakes drew it.
""")

answer_code('''
plt.plot(coast.lon, coast.lat, color="0.6", lw=0.6)
plt.scatter(lons, lats, s=3)
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title(f"Earthquakes M4.5+ in 1983 (n = {len(mags)})")
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.gca().set_aspect("equal")
plt.show()
''')

md("""
Lines. Not a sprinkle, and not one crowded rim: narrow curved lines, running down the middle of the
Atlantic, up the western edge of the Americas, along the southern rim of Asia and all the way round
the western Pacific — with Australia, most of Siberia and Canada, and the middle of the Pacific
carrying almost nothing.

Those lines are the edges of the tectonic plates. The plate interiors are comparatively quiet;
nearly all of this breaking happens where two plates meet. Nobody drew that map from a theory. It
came out of four thousand dots, and it is the reason a catalogue is worth reading.
""")

# ---------------------------------------------------------------- 5. counting by magnitude
md("""
## 4. Counting by magnitude

Magnitude is a step scale, not a count: an M6.5 is not slightly bigger than an M5.5, it is in a
different league. The year you just loaded runs from 4.5 at the bottom to 7.6 at the top. So are
the big ones rare, and if they are, how rare?

There is a rule of thumb for this, and it is old: Gutenberg and Richter, *Bulletin of the
Seismological Society of America* **34**, 185–188 (1944) — reference checked 2026-08-31. Their
observation is that the number of earthquakes falls by roughly a **factor of ten for every step up
in magnitude**. That is a convention we are about to test against a real year, not something this
notebook derived, and the word "roughly" is doing real work in it.

Start by looking. `plt.hist(mags, bins=31)` chops the magnitude range into 31 slices and draws how
many earthquakes fall in each.
""")

code('''
# ── Checkpoint ── run this if you restarted the kernel or fell behind ──
times, lats, lons, depths, mags, places = columns(load("1983-01-01", "1984-01-01", 4.5))
''')

code('''
plt.hist(mags, bins=31)
plt.xlabel("magnitude")
plt.ylabel("number of earthquakes")
plt.title(f"Magnitudes in one year, M4.5 and above (n = {len(mags)})")
plt.locator_params(axis="y", integer=True)
plt.show()
''')

md("""
The bars fall away fast: hundreds per slice near the bottom, single figures at the top. Two things
are worth noticing. The very first bars are *lower* than the ones just above them, which is a loose
thread we will pull in a moment. And a histogram cannot tell you "ten times" by eye.

For that, count. `count(start, end, minmag)` asks how many earthquakes of at least that magnitude
a window of the catalogue holds.
""")

code('''
n_45 = count("1983-01-01", "1984-01-01", 4.5)
n_55 = count("1983-01-01", "1984-01-01", 5.5)
print("M4.5 and above:", n_45)
print("M5.5 and above:", n_55)
''')

md("""
✏️ **Your turn.** These counts are **cumulative**: `n_45` is everything at 4.5 and above, `n_55`
everything at 5.5 and above. So `n_45 / n_55` is how many earthquakes of at least 4.5 there are
for every one of at least 5.5 — one step up the scale.

Get the third count, at magnitude 6.5, then print both ratios to one decimal place with `round`.
`n_45` and `n_55` already exist.

**Use these names**: `n_65`.
""")

answer_code('''
n_65 = count("1983-01-01", "1984-01-01", 6.5)
print("M6.5 and above:", n_65)
print("one step, 4.5 to 5.5:", round(n_45 / n_55, 1))
print("one step, 5.5 to 6.5:", round(n_55 / n_65, 1))
''')

# ---------------------------------------------------------------- 6. wider windows
md("""
## 5. The same ratio in other windows

Nine, and eight and a half, against a rule of thumb that says ten. Close enough to be encouraging —
and one year is one year. Before believing any number measured in one window of a catalogue, the
next move is always the same: measure the windows either side of it.

Going wider means going further up the magnitude scale, for a dull but hard reason. The USGS server
refuses any single query above 20,000 earthquakes, and fifty years at M5.5 and above is more than
that, so from here on the pair is M6.5 and M7.5. Start with the year we have and its two
neighbours, at the top of the scale.
""")

code('''
print("M7.5 and above in 1982:", count("1982-01-01", "1983-01-01", 7.5))
print("M7.5 and above in 1983:", count("1983-01-01", "1984-01-01", 7.5))
print("M7.5 and above in 1984:", count("1984-01-01", "1985-01-01", 7.5))
''')

md("""
Zero, one, two. At this end of the scale a single year is not a measurement at all: 1982 gives us
nothing to divide by, and one earthquake either way moves the answer by a factor of two. So widen
the window — whole decades, and then fifty years. `report_window(label, start, end)` does the two
counts and the division in one go, and prints all three.
""")

code('''
report_window("1980s:", "1980-01-01", "1990-01-01")
report_window("1990s:", "1990-01-01", "2000-01-01")
report_window("2000s:", "2000-01-01", "2010-01-01")
report_window("2010s:", "2010-01-01", "2020-01-01")
report_window("fifty years:", "1976-01-01", "2026-01-01")
''')

md("""
Over the full fifty years, on 2192 earthquakes at M6.5 and above and 218 at M7.5 and above, the
ratio is 10.1 — the rule of thumb, near enough, at the scale where there is enough data to test it.

The four decades move between 7.8 and 19.9, a factor of two and a half, and some of that is simply
that the counts are small: a ratio built on 20 earthquakes moves a long way when a handful of them
cross the line. But not all of it. If the fifty-year ratio held in the 1980s, that decade's 397
earthquakes at M6.5 and above would have come with about forty at M7.5 and above. Twenty are
listed. That gap is too big to blame on the smallness of the count, and this notebook cannot tell
you what closes it: either the 1980s genuinely had half as many of the very largest earthquakes, or
their magnitudes were measured in a way that put some of them below 7.5. Nobody in this room can
settle that from these five lines, and saying so is the honest end of the section.
""")

# ---------------------------------------------------------------- 7. 1940
md("""
## 6. The catalogue in 1940

Walter Alvarez is a geologist at Berkeley. He and his father found the thin worldwide layer of
iridium that is the evidence for an asteroid striking the Earth at the end of the Cretaceous, and
ending the dinosaurs with it. He was born on 3 October 1940.

Ask the catalogue for his birthday, and then for his whole birth year, exactly the way we asked it
for 1983.
""")

code('''
# ── Checkpoint ── run this if you restarted the kernel or fell behind ──
n_45 = count("1983-01-01", "1984-01-01", 4.5)
''')

md("""
### Predict before you run

1983 gave us 4124 earthquakes at M4.5 and above. Write a number down now, before the next cell
runs: how many do you think the catalogue lists for the whole of 1940? Same planet, same magnitude
floor, forty-three years earlier.
""")

md("""
✏️ **Your turn.** Now find out. Count the earthquakes at M4.5 and above on 3 October 1940, and
then in the whole of 1940, and print both — then print how many times as many 1983 had, to one
decimal place. `n_45` is still the 1983 count from section 4.

**Use these names**: `n_1940`.
""")

answer_code('''
n_1940 = count("1940-01-01", "1941-01-01", 4.5)
print("earthquakes on 3 October 1940:", count("1940-10-03", "1940-10-04", 4.5))
print("earthquakes in all of 1940:", n_1940)
print("times as many in 1983:", round(n_45 / n_1940, 1))
''')

code('''
times, lats, lons, depths, mags, places = columns(load("1940-01-01", "1941-01-01", 4.5))

plt.plot(coast.lon, coast.lat, color="0.6", lw=0.6)
plt.scatter(lons, lats, s=3)
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title(f"Earthquakes M4.5+ in 1940 (n = {len(mags)})")
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.gca().set_aspect("equal")
plt.show()
''')

md("""
One earthquake on the day Walter Alvarez was born, and 236 in his whole birth year, against 4124
in 1983. Seventeen and a half times as many, on the same planet.

The planet did not get seventeen times busier. Put the two maps side by side. The 1983 map has dots
running down the middle of the Atlantic, across the southern oceans and out into the eastern
Pacific. On the 1940 map those lines are all but blank, while the Mediterranean, the Himalaya and
the western Pacific are populated. Those are the places that had seismometers near them in 1940.
The mid-ocean ridges did not, so their earthquakes are not in the file.

**Catalogue completeness.** A catalogue lists what somebody's instruments recorded, not what happened. Where there are no seismometers there are no earthquakes in the file.

Which puts section 4 in a different light. Those first histogram bars, lower than the ones just
above them, are the same effect at a smaller scale: not fewer M4.5 earthquakes in 1983, just fewer
of them written down.
""")

# ---------------------------------------------------------------- the question, answered
md("""
## The question, answered

For the one day the whole room loaded together, the birthquake was the magnitude 7.0 twenty-five
kilometres south of Champerico, Guatemala, at 03:09 UTC, sixty-seven kilometres down — and the
three later earthquakes that named the same town look like its aftershocks. Yours is the first
thing the homework asks for, and it will be a different earthquake on every screen in the room.
""")

md(weekkit.week_cheatsheet(1))

# ---------------------------------------------------------------- homework
md("""
## Homework

Three parts, all on your own birthday. Everything you need is above: the same loading move, the
same `max` and `.index`, the same map recipe. Part 1 is the one everybody finishes.

Your birthday has no cached copy, so if the network is down when you sit down to this, nothing
here will run. Come back to it when you are online.
""")

md("""
✏️ **Part 1 — your own birthquake, and how many earthquakes there really were.**

Class ran on one shared day so that forty-six screens printed the same numbers. This one is yours,
and nobody else in the room will get your answer.

Your birthday has no cached copy anywhere in this repository, so `load_yours` goes to the catalogue
live and, if it cannot get there, says so and stops rather than quietly handing you somebody else's
day. It wants a start date, an end date and a magnitude floor, and the end date is the **next** day:

```python
my_day = load_yours("2007-09-15", "2007-09-16", 4.5)
```

**First**, load your own birthday at magnitude 4.5 and above, unpack it with `columns`, and print
your birthquake — its magnitude, where it was and how deep — with the same `max` and `.index` move
class used in section 2.

**Then** commit to a guess, and write it down before you run anything else: with no magnitude floor
at all, how many earthquakes do you think the catalogue lists for that same one day? Load the day
again with the floor removed — `load_yours` takes a magnitude floor, so pass `-10`; no catalogue on
Earth holds anything that small, so nothing gets cut — and print how many there are and the
smallest magnitude among them.

**Use these names**, because the self-check looks for them: `my_mags`, `my_places`, `my_depths`,
`my_guess` and `all_mags`.
""")

answer_code('''
my_day = load_yours("2008-03-12", "2008-03-13", 4.5)
my_times, my_lats, my_lons, my_depths, my_mags, my_places = columns(my_day)
my_biggest = max(my_mags)
mine = my_mags.index(my_biggest)
print(f"My birthquake: M{my_biggest}, {my_places[mine]}, {my_depths[mine]} km deep.")

my_guess = 40
all_day = load_yours("2008-03-12", "2008-03-13", -10)
all_times, all_lats, all_lons, all_depths, all_mags, all_places = columns(all_day)
print("with no magnitude floor:", len(all_mags))
print("the smallest magnitude of the day:", min(all_mags))
''')

code(f'''
assert len(all_mags) > len(my_mags), "the no-floor day should hold more than the M4.5+ day"
assert min(all_mags) < 4.5, "the floor is still on — pass -10 as the magnitude floor"
{check_print("Homework 1",
             "birthquake M{max(my_mags)}, guessed {my_guess},",
             "catalogue lists {len(all_mags)}, smallest M{min(all_mags)}")}
''')

md("""
✏️ **Part 2 — what the ten-to-one rule predicts, and how wrong it is.**

Class measured about nine earthquakes at M4.5 and above for every one at M5.5 and above, and the
rule of thumb rounds that to ten per step. Run the rule the other way — downwards — and it predicts
how many small earthquakes your day should have had.

Here is the decision, and it is yours to make. **How far down do you extrapolate?**

- **Two steps**, from M4.5 down to M2.5: multiply your M4.5+ count by 10, and then by 10 again.
- **Four steps**, from M4.5 down to M0.5: multiply it by 10 four times.

Both are defensible — the day you loaded in part 1 does have earthquakes below magnitude 1 in it.
Pick one, say which in a comment, and print three things: the prediction, the number you actually
counted (`all_mags`), and how many times too big the prediction turned out to be. Your M4.5+ count
is `len(my_mags)`, from part 1.

**Use these names**: `predicted` and `actual`.
""")

answer_code('''
# two steps down, from M4.5 to M2.5
predicted = len(my_mags) * 10 * 10
actual = len(all_mags)
print("the ten-to-one rule predicts:", predicted)
print("the catalogue actually lists:", actual)
print("too big by a factor of:", round(predicted / actual, 1))
''')

code(f'''
assert predicted != actual, "predicted is the rule's number, not the one you counted"
{check_print("Homework 2",
             "the rule predicts {predicted}, the catalogue lists {actual}:",
             "out by a factor of {round(predicted / actual, 1)}")}
''')

md("""
✏️ **Part 3 — how many days of data do you need?**

One day of earthquakes showed you nothing. One year drew the plate boundaries. Somewhere between
the two is the smallest amount of data that would have convinced you, and nobody has told you
where — this part is you finding out, on your own birthday.

Draw three maps, all at magnitude 4.5 and above, all with the coastline, all the same recipe as
section 3 with the dates and the title changed:

1. your birthday alone;
2. your birthday and the six days after it — a week;
3. your birthday and the thirty days after it — a month.

Put the number of earthquakes in each title, the way class did. **Use these names**: `n_day`,
`n_week` and `n_month`.

Then use the **last cell of the notebook** for two or three sentences: which of the three is the
first map where you would say you can see *lines* rather than scattered dots, how many earthquakes
that took, and what you would have concluded about the Earth if the one-day map had been all you
ever saw. There is no single right answer here. There is a defensible one with a number attached.
""")

answer_code('''
my_day = load_yours("2008-03-12", "2008-03-13", 4.5)
day_times, day_lats, day_lons, day_depths, day_mags, day_places = columns(my_day)
n_day = len(day_mags)

plt.plot(coast.lon, coast.lat, color="0.6", lw=0.6)
plt.scatter(day_lons, day_lats, s=8)
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title(f"My birthday, M4.5+ (n = {n_day})")
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.gca().set_aspect("equal")
plt.show()

my_week = load_yours("2008-03-12", "2008-03-19", 4.5)
week_times, week_lats, week_lons, week_depths, week_mags, week_places = columns(my_week)
n_week = len(week_mags)

plt.plot(coast.lon, coast.lat, color="0.6", lw=0.6)
plt.scatter(week_lons, week_lats, s=8)
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title(f"My birth week, M4.5+ (n = {n_week})")
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.gca().set_aspect("equal")
plt.show()

my_month = load_yours("2008-03-12", "2008-04-12", 4.5)
month_times, month_lats, month_lons, month_depths, month_mags, month_places = columns(my_month)
n_month = len(month_mags)

plt.plot(coast.lon, coast.lat, color="0.6", lw=0.6)
plt.scatter(month_lons, month_lats, s=8)
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title(f"My birth month, M4.5+ (n = {n_month})")
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.gca().set_aspect("equal")
plt.show()
''')

code(f'''
assert n_day < n_month, "a month should hold more earthquakes than one day — check your dates"
{check_print("Homework 3", "day {n_day}, week {n_week}, month {n_month} earthquakes")}
''')

answer_md("""
The month is the first map I would call lines rather than dots. My day put 15 earthquakes on the
globe: a loose knot near Indonesia and half a dozen single dots elsewhere, nothing I could have
called a line. My week had 100 and the western Pacific had begun to fill in, but most of the world
was still empty and I would not have argued for a pattern from it.
My month had 546, and the arc through the western Pacific and the whole west side of South America
read as continuous lines, while the mid-ocean ridges were still only a dotted suggestion — so a
month is enough to see the pattern but not enough to see all of it. It took a few hundred
earthquakes, not a dozen. If the one-day map had been all I ever saw, I would have said earthquakes
happen more or less anywhere, and nothing in that map would have told me I was wrong.
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
    weekkit.gate(1)
