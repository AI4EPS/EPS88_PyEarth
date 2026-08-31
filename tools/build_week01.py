#!/usr/bin/env python
"""Build week 1 — "What was your birthquake?" — as both notebooks from one source.

The SOLUTION is written here in full; the STUDENT copy is derived by replacing every cell
tagged `answer` with the standard stub, so the two cannot drift apart. Run:

    python tools/build_week01.py

It writes docs/notebooks/01_birthquake_solution.ipynb (executed) and
docs/notebooks/01_birthquake.ipynb (clean), then calls weekkit.gate(1).

THREE HELPERS, ONE SHAPE EACH. `column(name, start, end, minmag)` is the only door to the
catalogue; `count` is `len` of the "mag" column without keeping it, and `report_window` prints
one window's two counts and their ratio. Nothing returns a tuple, so no call site unpacks, and
the same three arguments — a start date, an end date and a magnitude floor — say which
earthquakes you want everywhere in the week, class and homework alike.

THE SPINE comes from course.yml. The four questions are printed near the top as a bare list and
are also the four section headings, so a heading always says what its section answers and the
two cannot drift apart.
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
SPINE = WEEK["spine"]

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


def heading(i):
    """Section i's heading IS spine question i — the two cannot drift."""
    return f"## {i}. {SPINE[i - 1]}"


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
    hook="Every day, somewhere, the ground breaks. The United States Geological Survey keeps a "
         "running list of it: for every earthquake anybody has located, a time, a place name, a "
         "latitude and longitude, a depth and a magnitude. It is one of the largest openly "
         "published records of anything that happens on this planet, and anyone can read it — "
         "including you, from this notebook, in one line.\n\n"
         "So: **what was your birthquake?** On the day you were born, what was the largest "
         "earthquake in the world, and where was it?\n\n"
         "That question is the way in rather than the point. Once you can pull one day out of the "
         "catalogue you can pull a year, and a year of earthquakes turns out to draw a map that "
         "nobody drew: the edges of the tectonic plates, in dots. By the end of today you will "
         "have found your own birthquake, and you will have an argument — with a number attached "
         "— about how much of the catalogue you have to read before it tells you anything at "
         "all."))

md(f"""
## What you'll be able to do

**The science.** Read a real earthquake catalogue. Find the largest earthquake in any span of
days. See that earthquakes fall along narrow lines rather than spreading out. Count how many small
earthquakes there are for each large one, and say where that count stops being trustworthy. And
say why a catalogue is a record of what was *recorded*, not of what happened.

**The code.** `print` · `type` · variables and arithmetic · f-strings · lists, `len`, `list[i]`,
`list[-1]` and `list[a:b]` · `max` and `min` · `list.index` · `round` · and your first figures,
with `plt.scatter`, `plt.hist` and `plt.plot`.

**Today, in four questions.**

{chr(10).join(f'{i}. {q}' for i, q in enumerate(SPINE, 1))}
""")

# ---------------------------------------------------------------- setup
md("""
## Setup

Run the next cell once. You are not expected to follow it — it is here so that everything below it
is short.

**Coming later:** it uses **pandas**, the tables library we meet properly in the tables week, to
fetch earthquakes from the USGS. What it hands you back is ordinary **lists**, which are this
week's subject. It defines three small helpers, and you will use all three today:

- `column(name, start, end, minmag)` — **one** column of the catalogue, as a list. The names it
  takes are `"time"`, `"latitude"`, `"longitude"`, `"depth"`, `"mag"` and `"place"`.
- `count(start, end, minmag)` — how many earthquakes a window of the catalogue holds.
- `report_window(label, start, end)` — prints how many M6.5+ and M7.5+ a window holds, and the
  ratio between them.

Two dates and a magnitude floor say which earthquakes you want, every single time: the day to
start, the day to stop *before*, and the smallest magnitude to include.

The day we work through together is **31 December 1990** — New Year's Eve, and a real person's
birthday: the person teaching this course. It is one shared day so that the whole room starts from
the same numbers. Yours comes in the homework, and it will not be this one.
""")

code(weekkit.setup_cell(
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
def column(name, start, end, minmag):
    """one column of the catalogue for one window of days, handed back as an ordinary list"""
    try:
        quakes = load(start, end, minmag)
    except Exception as e:
        raise RuntimeError("Could not read the catalogue for those dates. Check that they are "
                           "real dates written as YYYY-MM-DD and that you are online, then run "
                           "the cell again — a day only you have asked for has no copy stored "
                           "with the course.") from e
    return list(quakes[name])


def count(start, end, minmag):
    """how many earthquakes of at least this magnitude the catalogue lists in this window"""
    return len(column("mag", start, end, minmag))


def report_window(label, start, end):
    """print how many M6.5+ and M7.5+ a window holds, and the ratio between them"""
    big = count(start, end, 6.5)
    huge = count(start, end, 7.5)
    print(label, big, "at M6.5+ and", huge, "at M7.5+, ratio", round(big / huge, 1))


coast = pd.read_csv(CACHE + "/coastlines.csv")
'''.strip("\n")))

code('''
mags = column("mag", "1990-12-31", "1991-01-01", 4.5)
times = column("time", "1990-12-31", "1991-01-01", 4.5)
places = column("place", "1990-12-31", "1991-01-01", 4.5)
depths = column("depth", "1990-12-31", "1991-01-01", 4.5)
print("earthquakes the catalogue lists for 31 December 1990:", len(mags))
''')

# ---------------------------------------------------------------- 1. one day
md(f"""
{heading(1)}

The grey box above is a **cell**. Click it, press **Shift+Enter**, and Python runs what is inside.
The notebook keeps a memory of everything it has run — that memory is the **kernel** — which is
why the cell below can use names the cell above created.

A **value** is one piece of data. `12` is a whole number, an `int`. `4.5` is a decimal, a `float`.
`"mag"` is text, a `str`, and the quotes are part of how you write it — which is why the column
you want goes in quotes. A **name** is a label you stick on a value with `=`, so you can use it
again without retyping it.
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

md("""
`mags` is a **list**: twelve numbers in a row, in the order the earthquakes happened. `times`,
`places` and `depths` are three more columns of those same twelve earthquakes, in that same order
— so position 3 of `mags` and position 3 of `places` are the same earthquake. That is the whole
trick of this section, and it is why `column` hands back one list at a time: you ask for the
columns you need, with the same two dates and the same floor, and they line up.

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
gives the smallest. `mags.index(v)` gives the **position** of the value `v`; and because the four
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
✏️ **Your turn.** Two goes at the same move.

**First**, `where` is the position of the largest earthquake of the day. Read that same position
out of `places` and out of `depths`, and print both — so that we know where the day's biggest
earthquake was, and how far below the surface it broke.

**Then** do it again on a different list. `depths` is in kilometres; find the day's *deepest*
earthquake with `max(depths)`, use `.index` on `depths` to get its position, and print how deep it
was and the place it happened.

**Use these names**: `where`, `deepest`.
""")

answer_code('''
print("place:", places[where])
print("depth in km:", depths[where])

deepest = max(depths)
print("the deepest earthquake of the day:", deepest, "km down")
print("it happened at:", places[depths.index(deepest)])
''')

md("""
Two things fell out of that. Look again at what `where` came out as — **11**, and there are twelve
earthquakes, so it is the last one. The largest earthquake of the day was also the final one of
it, and since this is 31 December, the final one of the year: the list is in time order, and
whatever adjusting the ground did afterwards happened in a window we did not ask for.

And the deepest broke more than two hundred kilometres down. Nothing about an earthquake requires
it to be near the surface, and how the ground can break that far down is a question this course
comes back to. For now it is a reminder that `depths` is a column worth reading.
""")

# ---------------------------------------------------------------- 2. where
md(f"""
{heading(2)}

A map is a scatter plot: longitude across, latitude up. `plt.scatter(lons, lats)` puts one dot at
every pair. On its own that would be dots in a white rectangle, so every map in this course first
draws `coast` — a long list of coastline points that the setup cell loaded — with `plt.plot`,
which joins points up with a line.

Two more columns of the same twelve earthquakes, then.
""")

code(weekkit.CHECKPOINT.format(body='''
mags = column("mag", "1990-12-31", "1991-01-01", 4.5)
lats = column("latitude", "1990-12-31", "1991-01-01", 4.5)
lons = column("longitude", "1990-12-31", "1991-01-01", 4.5)
'''.strip("\n")))

code('''
plt.plot(coast.lon, coast.lat, color="0.6", lw=0.6)
plt.scatter(lons, lats, s=25)
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title(f"Earthquakes M4.5+ on 31 December 1990 (n = {len(mags)})")
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.gca().set_aspect("equal")
plt.show()
''')

md("""
Count the dots and you will find eleven, not twelve: two of those earthquakes are about half a
degree apart in `lats` and `lons`, some fifty kilometres, and at this scale the map draws them as
one speck. Either way, a dozen earthquakes scattered over a whole globe cannot show you a shape,
and no amount of staring will make them. A thin figure is not a failed figure as long as you say
out loud that it is thin. Hold on to this one.

The only fix is more data, and the only thing that has to change is the two dates: the same three
lines, asking for the whole of 1990 instead of its last day, at the same magnitude floor.
""")

code('''
mags = column("mag", "1990-01-01", "1991-01-01", 4.5)
lats = column("latitude", "1990-01-01", "1991-01-01", 4.5)
lons = column("longitude", "1990-01-01", "1991-01-01", 4.5)
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
plt.title(f"Earthquakes M4.5+ in 1990 (n = {len(mags)})")
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.gca().set_aspect("equal")
plt.show()
''')

md("""
Lines. Not a sprinkle, and not one crowded rim: narrow curved lines, running down the middle of
the Atlantic, up the western edge of the Americas, along the southern rim of Asia and all the way
round the western Pacific — with Australia, most of Siberia and Canada, and the middle of the
Pacific carrying almost nothing.

Those lines are the edges of the tectonic plates. The plate interiors are comparatively quiet;
nearly all of this breaking happens where two plates meet. Nobody drew that map from a theory. It
came out of four thousand dots, and it is the reason a catalogue is worth reading.
""")

# ---------------------------------------------------------------- 3. how many small ones
md(f"""
{heading(3)}

Magnitude is a step scale, not a count: an M6.5 is not slightly bigger than an M5.5, it is in a
different league. The year you just loaded runs from 4.5 at the bottom to 7.8 at the top. So are
the big ones rare, and if they are, how rare?

There is a rule of thumb for this, and it is old: Gutenberg and Richter, *Bulletin of the
Seismological Society of America* **34**, 185–188 (1944) — reference checked 2026-08-31. Their
observation is that the number of earthquakes falls by roughly a **factor of ten for every step up
in magnitude**. That is a convention we are about to test against a real year, not something this
notebook derived, and the word "roughly" is doing real work in it.

Start by looking. `plt.hist(mags, bins=11)` chops the magnitude range into 11 slices and draws how
many earthquakes fall in each.
""")

code(weekkit.CHECKPOINT.format(body='''
mags = column("mag", "1990-01-01", "1991-01-01", 4.5)
'''.strip("\n")))

code('''
plt.hist(mags, bins=11)
plt.xlabel("magnitude")
plt.ylabel("number of earthquakes")
plt.title(f"Magnitudes in one year, M4.5 and above (n = {len(mags)})")
plt.locator_params(axis="y", integer=True)
plt.show()
''')

md("""
The bars fall away fast: well over a thousand earthquakes in the lowest slice, single figures at
the top. But a histogram cannot tell you "ten times" by eye, and the rule of thumb is a claim
about a number.

For that, count. `count(start, end, minmag)` asks how many earthquakes of at least that magnitude
a window of the catalogue holds — it is `len` of the `"mag"` column you have been loading, without
keeping the list.
""")

code('''
n_45 = count("1990-01-01", "1991-01-01", 4.5)
n_55 = count("1990-01-01", "1991-01-01", 5.5)
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
n_65 = count("1990-01-01", "1991-01-01", 6.5)
print("M6.5 and above:", n_65)
print("one step, 4.5 to 5.5:", round(n_45 / n_55, 1))
print("one step, 5.5 to 6.5:", round(n_55 / n_65, 1))
''')

md("""
Two measurements of one rule, in one year, and they do not agree with each other: 8.4 at the lower
step and 10.0 at the higher one. Do not read anything into the second landing exactly on ten. This
year was chosen for whose birthday it is, not for its ratios, and a rule of thumb that comes out
at 8.4 and 10.0 in the same twelve months of the same catalogue is telling you how much slack the
word "roughly" is carrying.

One year is also one year. Before believing any number measured in one window of a catalogue, the
next move is always the same: measure the windows either side of it.

Going wider means going further up the magnitude scale, for a dull but hard reason. The USGS
server refuses any single query above 20,000 earthquakes, and fifty years at M5.5 and above is
more than that, so from here on the pair is M6.5 and M7.5. Start with the year we have and its two
neighbours, at the top of the scale.
""")

code('''
print("M7.5 and above in 1989:", count("1989-01-01", "1990-01-01", 7.5))
print("M7.5 and above in 1990:", count("1990-01-01", "1991-01-01", 7.5))
print("M7.5 and above in 1991:", count("1991-01-01", "1992-01-01", 7.5))
''')

md("""
Two, five, three. At this end of the scale a single year is not a measurement: the number you
would be dividing by more than doubles from one year to the next, so a ratio built on it depends
mostly on which year you happened to be born in.

So widen the window — whole decades, and then fifty years. `report_window(label, start, end)` does
the two counts and the division in one go, and prints all three.
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

# ---------------------------------------------------------------- 4. any birthday?
md(f"""
{heading(4)}

Walter Alvarez is a geologist at Berkeley. He and his father found the thin worldwide layer of
iridium that is the evidence for an asteroid striking the Earth at the end of the Cretaceous, and
ending the dinosaurs with it. He was born on 3 October 1940 — fifty years, near enough, before the
day this class ran on.

Ask the catalogue for his birthday, and then for his whole birth year, exactly the way we asked it
for 1990.
""")

code(weekkit.CHECKPOINT.format(body='''
n_45 = count("1990-01-01", "1991-01-01", 4.5)
'''.strip("\n")))

md("""
### Predict before you run

1990 gave us 4430 earthquakes at M4.5 and above. Write a number down now, before the next cell
runs: how many do you think the catalogue lists for the whole of 1940? Same planet, same magnitude
floor, fifty years earlier.
""")

md("""
✏️ **Your turn.** Now find out. Count the earthquakes at M4.5 and above on 3 October 1940, and
then in the whole of 1940, and print both — then print how many times as many 1990 had, to one
decimal place. `n_45` is still the 1990 count from the section above.

**Use these names**: `n_1940`.
""")

answer_code('''
n_1940 = count("1940-01-01", "1941-01-01", 4.5)
print("earthquakes on 3 October 1940:", count("1940-10-03", "1940-10-04", 4.5))
print("earthquakes in all of 1940:", n_1940)
print("times as many in 1990:", round(n_45 / n_1940, 1))
''')

code('''
lats = column("latitude", "1940-01-01", "1941-01-01", 4.5)
lons = column("longitude", "1940-01-01", "1941-01-01", 4.5)

plt.plot(coast.lon, coast.lat, color="0.6", lw=0.6)
plt.scatter(lons, lats, s=3)
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title(f"Earthquakes M4.5+ in 1940 (n = {len(lats)})")
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.gca().set_aspect("equal")
plt.show()
''')

md("""
One earthquake on the day Walter Alvarez was born, and 236 in his whole birth year, against 4430
in 1990. Nearly nineteen times as many, on the same planet, fifty years apart.

The planet did not get nineteen times busier. Put the two maps side by side. The 1990 map has a
line of dots running the length of the Atlantic and out across the southern oceans; on the 1940
map that Atlantic line is all but gone, while the Mediterranean, the Himalaya and the west coast
of the Americas are still populated. Those are the places that had seismometers near them in 1940.
The middle of the Atlantic did not, so its earthquakes are not in the file.

**Catalogue completeness.** A catalogue lists what somebody's instruments recorded, not what happened. Where there are no seismometers there are no earthquakes in the file.

So the answer to the question this section asked is no. The same three lines of code would have
told Walter Alvarez almost nothing about the day he was born — not because his day was quiet, but
because in 1940 almost nobody was listening.
""")

# ---------------------------------------------------------------- the question, answered
md("""
## The question, answered

For the one day the whole room loaded together, the birthquake was the magnitude 5.6 in the
Vanuatu region at 22:11 UTC, 10.2 kilometres down — the last earthquake of 31 December 1990, and
so the last of that year. Yours is the first thing the homework asks for, and it will be a
different earthquake on every screen in the room.
""")

md(weekkit.week_cheatsheet(1))

# ---------------------------------------------------------------- homework
md("""
## Homework

Three parts, all on your own birthday. Everything you need is above: the same `column` and
`count`, the same `max` and `.index`, the same map recipe. Part 1 is the one everybody finishes.

Your birthday has no copy stored with the course, so `column` and `count` go to the catalogue live
and stop with a message naming the fix if they cannot get there. If the network is down when you
sit down to this, nothing here will run; come back to it when you are online.

Each part ends in a question as well as a calculation. The calculation is the evidence; the answer
goes in the markdown cell underneath the self-check, in two or three sentences, about the numbers
your own screen printed.
""")

md("""
✏️ **Part 1 — your own birthquake, and how many earthquakes there really were.**

Class ran on one shared day so that forty-six screens printed the same numbers. This one is yours,
and nobody else in the room will get your answer. The start date is your birthday and the end date
is the **next** day:

```python
my_mags = column("mag", "2007-09-15", "2007-09-16", 4.5)
```

**First**, load `my_mags`, `my_places` and `my_depths` for your own birthday at magnitude 4.5 and
above, and print your birthquake — its magnitude, where it was and how deep — with the same `max`
and `.index` move class used in section 1.

**Then** commit to a guess and put it in `my_guess` before you run anything else: with no magnitude
floor at all, how many earthquakes do you think the catalogue lists for that same one day? Now load
the magnitudes of your day again with the floor removed — pass `-10` as the floor; no catalogue on
Earth holds anything that small, so nothing gets cut — into `all_mags`, and print how many there
are and the smallest magnitude among them.

**Then answer, in the markdown cell under the check:** your birthquake was the largest earthquake
of your day — the largest of *how many*? Quote both counts, and say whether the M4.5-and-above
list you started with is most of your day or a small slice of it.

**Use these names**, because the self-check looks for them: `my_mags`, `my_places`, `my_depths`,
`my_guess` and `all_mags`.
""")

answer_code('''
my_mags = column("mag", "2008-03-12", "2008-03-13", 4.5)
my_places = column("place", "2008-03-12", "2008-03-13", 4.5)
my_depths = column("depth", "2008-03-12", "2008-03-13", 4.5)
my_biggest = max(my_mags)
mine = my_mags.index(my_biggest)
print(f"My birthquake: M{my_biggest}, {my_places[mine]}, {my_depths[mine]} km deep.")

my_guess = 40
all_mags = column("mag", "2008-03-12", "2008-03-13", -10)
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

answer_md("""
My birthquake was the M6.4 fifty-two kilometres south of Lakatoro, Vanuatu, thirteen kilometres
down. With no magnitude floor at all the catalogue lists 301 earthquakes for that day, so my
birthquake was the largest of 301 — not the largest of the 15 I started with. The M4.5-and-above
list is a small slice of the day, about one earthquake in twenty; almost everything that happened
on my birthday is smaller than anything class looked at.
""")

md("""
✏️ **Part 2 — run the ten-to-one rule downwards.**

Class measured about eight and a half earthquakes at M4.5 and above for every one at M5.5 and
above, and the rule of thumb rounds that to ten per step. That was the rule read upwards. Turn it
round, and it predicts how many *small* earthquakes your one day should have had.

Here is the decision, and it is yours to make. **How far down do you run it?**

- **Two steps**, from M4.5 down to M2.5: multiply your M4.5+ count by 10, and then by 10 again.
- **Four steps**, from M4.5 down to M0.5: multiply it by 10 four times.

Then compare like with like, which is the part that is easy to get wrong. A prediction for M2.5 and
above has to be checked against a **count at M2.5 and above** — not against part 1's `all_mags`,
which has no floor at all, reaches further down, and is a different quantity altogether. So
`count` your day once more at whichever floor you chose, and print three things: the prediction,
the count, and how many times one is bigger than the other.

Say which fork you took in a comment. Your M4.5+ count is `len(my_mags)`, from part 1.

**Then answer, in the markdown cell under the check:** by what factor does the prediction
overshoot your count? Then say which of these two readings your numbers support, and why — that
the ten-to-one rule is simply wrong about the Earth at small magnitudes, or that the catalogue is
missing most of the small earthquakes. One of them is settled by a pair of numbers class computed
in the section that asked *would any birthday have worked*; name them.

**Use these names**: `predicted` and `actual`.
""")

answer_code('''
# two steps down, from M4.5 to M2.5
predicted = len(my_mags) * 10 * 10
actual = count("2008-03-12", "2008-03-13", 2.5)

print("the rule predicts, at M2.5 and above:", predicted)
print("the catalogue lists, at M2.5 and above:", actual)
print("times more predicted than listed:", round(predicted / actual, 1))
''')

code(f'''
assert actual < len(all_mags), "actual is the count at YOUR floor, not part 1's no-floor total"
assert predicted > len(my_mags) * 10, "the rule runs down at least two steps — multiply twice"
{check_print("Homework 2",
             "rule predicts {predicted}, catalogue lists {actual}",
             "at the same floor: {round(predicted / actual, 1)}x over")}
''')

answer_md("""
Two steps down, the rule predicts 1500 earthquakes at M2.5 and above on my birthday and the
catalogue lists 88 — it overshoots by a factor of 17. I do not think that means the rule is wrong
about the Earth. The section above loaded the same catalogue for 1940 and for 1990 and found 236
earthquakes against 4430, on a planet that did not change: what changed was how many seismometers
were listening. An M2.5 in the middle of an ocean has nothing near enough to record it, so most of
the small earthquakes the rule predicts were never written down. The rule is a claim about the
Earth; my count is a measurement of the instruments.
""")

md("""
✏️ **Part 3 — how many days of data do you need?**

One day of earthquakes showed you nothing. One year drew the plate boundaries. Somewhere between
the two is the smallest amount of data that would have convinced you, and nobody has told you
where — this part is you finding out, on your own birthday.

Draw three maps, all at magnitude 4.5 and above, all with the coastline, all the same recipe as
section 2 with the dates and the title changed:

1. your birthday alone;
2. your birthday and the six days after it — a week;
3. your birthday and the thirty days after it — a month.

Remember that the end date is the day you stop *before*, so a seven-day week ends on your birthday
plus 7, and a thirty-one-day month ends on your birthday plus 31. Put the number of earthquakes in
each title, the way class did. **Use these names**: `n_day`, `n_week` and `n_month`.

**Then answer, in the markdown cell under the check:** which of the three is the first map where
you would say you can see *lines* rather than scattered dots, how many earthquakes that took, and
what you would have concluded about the Earth if the one-day map had been all you ever saw. There
is no single right answer here. There is a defensible one with a number attached.
""")

answer_code('''
my_lons = column("longitude", "2008-03-12", "2008-03-13", 4.5)
my_lats = column("latitude", "2008-03-12", "2008-03-13", 4.5)
n_day = count("2008-03-12", "2008-03-13", 4.5)

plt.plot(coast.lon, coast.lat, color="0.6", lw=0.6)
plt.scatter(my_lons, my_lats, s=8)
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title(f"My birthday, M4.5+ (n = {n_day})")
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.gca().set_aspect("equal")
plt.show()

my_lons = column("longitude", "2008-03-12", "2008-03-19", 4.5)
my_lats = column("latitude", "2008-03-12", "2008-03-19", 4.5)
n_week = count("2008-03-12", "2008-03-19", 4.5)

plt.plot(coast.lon, coast.lat, color="0.6", lw=0.6)
plt.scatter(my_lons, my_lats, s=8)
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title(f"My birth week, M4.5+ (n = {n_week})")
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.gca().set_aspect("equal")
plt.show()

my_lons = column("longitude", "2008-03-12", "2008-04-12", 4.5)
my_lats = column("latitude", "2008-03-12", "2008-04-12", 4.5)
n_month = count("2008-03-12", "2008-04-12", 4.5)

plt.plot(coast.lon, coast.lat, color="0.6", lw=0.6)
plt.scatter(my_lons, my_lats, s=8)
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title(f"My birth month, M4.5+ (n = {n_month})")
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.gca().set_aspect("equal")
plt.show()
''')

code(f'''
assert n_day < n_week, "the week contains the day, so it cannot hold fewer — check your end dates"
assert n_week < n_month, "the month contains the week, so it cannot hold fewer — check the dates"
assert n_month > 100, "a month at M4.5+ holds a few hundred worldwide — your month is too short"
{check_print("Homework 3", "day {n_day}, week {n_week}, month {n_month} earthquakes")}
''')

answer_md("""
The month is the first map I would call lines rather than dots. My day put 15 earthquakes on the
globe: a loose knot near Vanuatu and half a dozen single dots elsewhere, nothing I could have
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
