#!/usr/bin/env python
"""Build the Labor-Day practice notebook — "Week 1.5: lists, loops, choices and your own function".

Written for students with NO programming background working alone: every new idea gets a toy
example small enough to hold in the head before it meets 2192 earthquakes, every multi-step
question is broken into lettered steps, and section 3 spends a cell on reading an error message.

NOT a week. There is no class on Monday 7 September, and this fills the gap between week 1 and
week 2 with Python practice: no grade, no Gradescope, no submission, and the answers ship beside
it. It covers the whole of week 2's Python — P3 (for, range, append, if/elif/else) and P4 (def,
return, docstring) — so that Monday 14 September can spend its 100 minutes on the exoplanet
science rather than on syntax.

TEACH WEEK 2 AS IF NOBODY DID THIS. It is ungraded, so some students will not, and a Monday
planned on the assumption they did would split the room.

Everything it reads is a CSV already cached in data/ and already read by week 1 — static files
off raw.githubusercontent.com, never the live USGS API, and never load_yours(). In a week with no
class nobody is on hand to debug a failed fetch, so the notebook has no failure mode a student
cannot see through.

Depth is used for counting only — no map, no lat/lon, no geography. Where the deep earthquakes
SIT is week 4's finding and is not touched here.

    .venv/bin/python tools/build_week1b.py

Writes docs/notebooks/01b_practice_solution.ipynb (executed) and docs/notebooks/01b_practice.ipynb
(clean). The solution is public from day one, which is why `01b_practice` is in course.yml's
`policy.solutions_released` — that list is what un-gitignores it and puts it in the site nav.
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
SLUG = "01b_practice"

DATAHUB = (f"{PLATFORM['datahub']}/hub/user-redirect/git-pull"
           f"?repo={PLATFORM['repo'].replace(':', '%3A')}"
           f"&branch={PLATFORM['branch']}"
           f"&urlpath=lab/tree/EPS88_PyEarth/{PLATFORM['notebook_dir']}/{SLUG}.ipynb")

CODE_STUB = "# ← your answer here\n"

cells = []


def md(text):
    cells.append(("markdown", text.strip("\n"), False, None))


def code(text, tags=None):
    cells.append(("code", text.strip("\n"), False, tags))


def answer_code(text):
    cells.append(("code", text.strip("\n"), True, None))


def check(label, summary):
    return f'print(f"{weekkit.CHECK_LINE.format(label=label, summary=summary)}")'


# ---------------------------------------------------------------- opening
md(f"""
# Week 1.5 — lists, loops, choices, and a function of your own

**EPS 88 · PyEarth.** Open your own copy on DataHub: [click here]({DATAHUB}).

Monday 7 September is Labor Day, so there is no class, and the next one is Monday 14 September.
This notebook fills the gap.

It is **practice, not homework.** There is no grade and nothing to submit. The worked answers are
in `01b_practice_solution.ipynb`, in the same folder. Open them whenever you are stuck.

There is no need to do it in one sitting. Everything here uses the earthquake data you already
loaded in week 1, so there is no new science to learn. The point is the code.

**How to use this notebook.** Read a bit of writing, then run the cell under it, then read the
next bit. Places where *you* write something are marked with a pencil ✏️ and the words
*Your turn*, and are followed by an empty cell.

**Two things to remember.** A cell runs when you press **Shift+Enter**. And if something breaks
and you cannot see why, use **Kernel → Restart Kernel and Run All Cells** from the menu at the
top. That forgets everything and runs the notebook again from the beginning. It is never the
wrong thing to do.
""")

md("""
## What you'll practise

Week 1 gave you lists, and how to get things out of them. Sections 1 and 2 below are practice at
that, so you can find out whether it stuck.

Sections 3 to 6 add four new ideas. They are the ones that turn a line of code into a program,
and you will use them every week for the rest of the course:

3. **A loop** — do the same thing to every item, without typing it out each time.
4. **A growing list** — keep the answers a loop works out.
5. **`if`** — treat some items differently from others.
6. **A function** — write a block of code once, then use it as often as you like.

Take them in order. Each one uses the one before.
""")

# ---------------------------------------------------------------- setup
md("""
## Setup

Run this cell first. It loads the earthquake data and gets the plotting ready. You do not need to
understand every line in it — it is the same setup cell you ran in week 1, and everything it makes
is explained where you first use it.
""")

code('''
import pandas as pd
import matplotlib.pyplot as plt

# house style, set once, so every plot cell below holds only what matters
plt.rcParams.update({"figure.figsize": (7, 4), "figure.dpi": 110,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})

CACHE = "https://raw.githubusercontent.com/AI4EPS/EPS88_PyEarth/main/data"


def columns(quakes):
    """the four columns we use, handed back as ordinary lists in the same order"""
    return (list(quakes["time"]), list(quakes["depth"]),
            list(quakes["mag"]), list(quakes["place"]))


def years_of(quakes):
    """the year each earthquake happened in, as a list of whole numbers"""
    return list(quakes["time"].str[:4].astype(int))


# one day of earthquakes — the same 2 December 1983 we loaded in class
day = pd.read_csv(CACHE + "/week01_1983-12-02_1983-12-03_M4.5.csv")
times, depths, mags, places = columns(day)

# fifty years of the larger ones, used from section 3 onwards
big = pd.read_csv(CACHE + "/week01_1976-01-01_2026-01-01_M6.5.csv")
big_years = years_of(big)
big_depths = list(big["depth"])

huge = pd.read_csv(CACHE + "/week01_1976-01-01_2026-01-01_M7.5.csv")
huge_years = years_of(huge)

print(len(mags), "earthquakes on 2 December 1983, at M4.5 and above")
print(len(big_years), "at M6.5+ and", len(huge_years), "at M7.5+ between 1976 and 2025")
''')

# ---------------------------------------------------------------- 1
md("""
## 1. Names, numbers, and printing a sentence

A **name** holds a value. You make one with `=`:

```
n = 14
```

From then on, writing `n` means 14. You can do arithmetic with names, and give the answer a name
of its own.

An **f-string** is a way to print a sentence with values dropped into it. Put an `f` in front of
the quotes, then put a name inside curly brackets:

```
print(f"there were {n} earthquakes")
```

Four functions you already know: `len(list)` is how many items, `max(list)` and `min(list)` are the
largest and smallest, and `round(x, 1)` trims a long decimal to one place. One that is new:
**`sum(list)` adds up every number in a list.**
""")

code('''
n = len(mags)
biggest = max(mags)
smallest = min(mags)

print("earthquakes that day:", n)
print("largest magnitude:", biggest)
print("smallest magnitude:", smallest)

# the same three facts, written as one sentence
print(f"{n} earthquakes, from M{smallest} up to M{biggest}")
''')

md("""
✏️ **Your turn 1.** Two small ones, to get your hands moving.

**(a)** Make a name `mag_range` holding the largest magnitude minus the smallest.

**(b)** Make a name `mag_average` holding the average magnitude. An average is the total divided by
how many there are, so that is `sum(mags) / len(mags)`.

Then print both in one sentence with an f-string, each rounded to one decimal place using
`round(mag_range, 1)` and `round(mag_average, 1)`.
""")

answer_code('''
mag_range = max(mags) - min(mags)
mag_average = sum(mags) / len(mags)

print(f"the day spans {round(mag_range, 1)} magnitude units, "
      f"averaging M{round(mag_average, 1)}")
''')

code(f'''
assert mag_range > 0, "a range is the larger minus the smaller — check the order of the subtraction"
assert min(mags) < mag_average < max(mags), \\
    "an average has to sit between the smallest and the largest — divide by len(mags), not by n"
{check("Your turn 1", "range {round(mag_range, 1)}, average M{round(mag_average, 1)}")}
''')

# ---------------------------------------------------------------- 2
md("""
## 2. Four lists that line up

The setup made four lists — `times`, `depths`, `mags` and `places`. They describe **the same 14
earthquakes, in the same order.** So the item at position 0 in each of them belongs to one
earthquake, the item at position 1 to the next, and so on:

| position | 0 | 1 | 2 | … |
|---|---|---|---|---|
| `mags` | 7.0 | 5.0 | 4.5 | … |
| `depths` | 67.1 | 10.0 | 33.0 | … |
| `places` | Champerico, Guatemala | Svalbard | Champerico, Guatemala | … |

Reading down a column gives you everything known about one earthquake.

That is why `list.index(v)` matters. It does not hand you a value — it hands you a **position**.
And once you have a position, you can look it up in *any* of the four lists.

So finding out about the largest earthquake of the day takes two steps:

1. `biggest = max(mags)` — the largest magnitude, 7.0.
2. `where = mags.index(biggest)` — the position that 7.0 sits at.

Then `places[where]` and `depths[where]` tell you where it was and how deep.
""")

code('''
print("how many:", len(mags))
print("the first one:", mags[0], places[0])
print("the last one:", mags[-1], places[-1])
print("the first three magnitudes:", mags[:3])

# the biggest earthquake of the day, and everything else we know about it
biggest = max(mags)
where = mags.index(biggest)

print(f"the largest was M{biggest} at {places[where]}, {depths[where]} km down")
''')

md("""
✏️ **Your turn 2.** Now do the same for the **deepest** earthquake of the day, in the same two
steps:

**(a)** Make `deepest` holding the largest depth. The depths are in `depths`.

**(b)** Make `where_deep` holding the position that value sits at.

Then print that earthquake's magnitude, its place and its depth in one sentence — reading all
three out of the lists at position `where_deep`.
""")

answer_code('''
deepest = max(depths)
where_deep = depths.index(deepest)
print(f"the deepest was M{mags[where_deep]} at {places[where_deep]}, {deepest} km down")
''')

code(f'''
assert depths[where_deep] == deepest, \\
    "where_deep should be the POSITION of deepest — use depths.index(deepest)"
{check("Your turn 2", "deepest = {deepest} km, at position {where_deep}")}
''')

md("""
### Seeing all four lists at once

`plt.plot` joins points in the order they come, which is right for something measured over time.
For 14 separate earthquakes there is no order to join, so the right picture is `plt.scatter` — one
dot per earthquake. Two lists in, one dot per position:
""")

code('''
plt.scatter(mags, depths)
plt.xlabel("magnitude")
plt.ylabel("depth (km)")
plt.title(f"2 December 1983, M4.5+ (n = {len(mags)})")
plt.show()
''')

md("""
Two dots stand out. The one at the far right is the M7.0 you found first. The one at the top is
the Gorontalo earthquake from Your turn 2 — 211.9 km down, while almost everything else sits in
the top 70 km. (Depth runs *upward* here, so deeper is higher on the page. It is worth reading an
axis before believing a picture.)

A number in a print statement is easy to skim past. The same number in a plot is hard to miss.
That is most of why we plot.
""")

# ---------------------------------------------------------------- 3
md("""
## 3. The `for` loop — doing the same thing many times

Here is a small one. Read it before you run it, then run it.
""")

code('''
for city in ["Berkeley", "Oakland", "Richmond"]:
    print(city)
''')

md("""
Three items in the list, so `print` ran three times. Each time round, the name `city` held the
next item.

That is all a loop is: **the indented code below `for` runs once for every item in the list.**

Three things to notice, because all three are how a loop goes wrong.

**The colon.** The `for` line ends in `:`. Python needs it.

**The indent.** The four spaces before `print` are what put it *inside* the loop. Code that is not
indented is outside the loop and runs once, after the loop has finished. Jupyter indents for you
when you press Enter after a colon — let it.

**The name is yours to choose.** `city` is not a special word. `for x in [...]` or
`for place in [...]` would behave exactly the same. Choosing a name that says what it holds is what
makes the line readable.
""")

md("""
### Now on real data

One list method first: **`list.count(v)` says how many times a value appears in a list.**

`big_years` holds the year of every M6.5+ earthquake between 1976 and 2025 — one entry per
earthquake, 2192 of them. So `big_years.count(1980)` is how many happened in 1980.

Suppose you want that for four different years. Typed out, it looks like this:
""")

code('''
print(1980, big_years.count(1980))
print(1990, big_years.count(1990))
print(2000, big_years.count(2000))
print(2010, big_years.count(2010))
''')

md("""
Four lines that differ in one number. That is exactly what a loop is for:
""")

code('''
for year in [1980, 1990, 2000, 2010]:
    print(year, big_years.count(year))
''')

md("""
Same output, one copy of the line. Read it as a sentence: *for each year in this list, print the
year and its count.*
""")

md("""
### When it goes wrong

You will hit errors this week with nobody sitting next to you, so here is one on purpose.

**The next cell is meant to fail.** Run it anyway, and look at what comes back. Nothing is broken
and you do not need to fix anything — just read it.
""")

code('''
for year in [1980, 1990]:
print(year, big_years.count(year))
''', tags=["raises-exception"])

md("""
**Read an error from the bottom.** The last line is the message:

```
IndentationError: expected an indented block after 'for' statement on line 1
```

Everything above it is Python showing you where it was looking. Here the message is exact — the
`print` was not indented, so it was never inside the loop.

Two habits worth having from today:

- **Read the last line first.** It says what went wrong. The rest is detail.
- **The line Python points at is where it *noticed*, not always where you slipped.** A missing
  bracket or a missing colon is usually reported on the line *after* the real mistake, so look
  one line up too.

Here is the same cell, fixed. The only change is four spaces.
""")

code('''
for year in [1980, 1990]:
    print(year, big_years.count(year))
''')

md("""
### `range` — a list of numbers, without typing them

Writing out `[1990, 1991, 1992, ...]` gets old fast. `range(a, b)` makes those numbers for you: it
starts at `a` and stops **just before** `b`.

So `range(1990, 2000)` gives 1990, 1991, … 1999. Ten numbers — 2000 is *not* included. It is the
same "up to but not including" rule as a slice, `mags[0:3]`.

✏️ **Your turn 3.** Print the M6.5+ count for every year of the 1990s, one line per year, exactly
like the loop two cells above — but loop over `range(1990, 2000)` instead of a typed-out list.

Call the loop name `year`, as the example does. The name is yours to choose, as you just read —
but the check below has to look at something, and it looks at that one.

You should get **ten lines**, starting at 1990 and ending at 1999. If you get nine or eleven, look
again at which end `range` stops.
""")

answer_code('''
for year in range(1990, 2000):
    print(year, big_years.count(year))
''')

code(f'''
assert year == 1999, \\
    "after a loop ends, the loop name still holds the LAST value it took — 1999 if you went 1990 to 1999"
{check("Your turn 3", "the loop finished on {year}")}
''')

# ---------------------------------------------------------------- 4
md("""
## 4. Keeping what a loop works out

The loops so far printed their answers and threw them away. The numbers scrolled past, and there
is nothing left to plot.

To keep them, you need somewhere to put them. The move is always the same three steps:

1. **Before the loop**, make an empty list: `counts = []`
2. **Inside the loop**, add one item to it: `counts.append(...)`
3. **After the loop**, the list holds every answer, in order.

`list.append(x)` puts `x` on the end of the list. The list starts empty and grows by one each time
round. Here it is on the small example first:
""")

code('''
lengths = []                      # empty, before the loop
for city in ["Berkeley", "Oakland", "Richmond"]:
    lengths.append(len(city))     # one more item each time round

print(lengths)
''')

md("""
Three cities in, three numbers out — the number of letters in each name. Now the same three steps
on fifty years of earthquakes.
""")

code('''
years = list(range(1976, 2026))

counts = []                               # empty, before the loop starts
for year in years:
    counts.append(big_years.count(year))  # one more entry each time round

print("one count per year:", len(counts), "of them")
print("years holds", len(years), "years; big_years holds", len(big_years), "earthquakes")
print("the first five:", counts[:5])

# max() and .index() again — this time on the list the loop just built
most = max(counts)
print(f"the busiest year was {years[counts.index(most)]}, with {most}")
''')

md("""
**Two names one letter apart, holding very different things** — worth pausing on, because it is
the easiest thing here to trip over. `big_years` has one entry per **earthquake**: 2192 of them,
the same year repeated many times. `years` has one entry per **year**: 50 of them, each appearing
once. The loop reads the first and builds the second.

`years` and `counts` are now two lists that line up, exactly like the four in section 2 — and two
lists that line up are what `plt.plot` wants.
""")

code('''
plt.plot(years, counts)
plt.xlabel("year")
plt.ylabel("earthquakes at M6.5 and above")
plt.title(f"M6.5+ per year, 1976-2025 (n = {len(big_years)})")
plt.show()
''')

md("""
✏️ **Your turn 4.** Do the same for the largest earthquakes, the M7.5+ ones, whose years are in
`huge_years`.

**(a)** Build a list called `huge_counts` using the same three steps: empty list, loop over `years`,
`append` the count for each year.

**(b)** Plot `huge_counts` against `years`, with an x label, a y label and a title.

There are only 218 of these against 2192, so expect a much lower and much jumpier line.
""")

answer_code('''
huge_counts = []
for year in years:
    huge_counts.append(huge_years.count(year))

plt.plot(years, huge_counts)
plt.xlabel("year")
plt.ylabel("earthquakes at M7.5 and above")
plt.title(f"M7.5+ per year, 1976-2025 (n = {len(huge_years)})")
plt.show()

print("most in one year:", max(huge_counts), "— fewest:", min(huge_counts))
''')

code(f'''
assert len(huge_counts) == len(years), \\
    "one count per year — there should be as many entries as there are years"
assert sum(huge_counts) == len(huge_years), \\
    "the counts should add up to every M7.5+ earthquake in the file — check the years you looped over"
{check("Your turn 4", "{len(huge_counts)} yearly counts, adding to {sum(huge_counts)}")}
''')

# ---------------------------------------------------------------- 5
md("""
## 5. `if` — treating some items differently

Before any syntax, look at what you are about to sort. `big_depths` holds the depth of all 2192
M6.5+ earthquakes. `plt.hist` — the third plot from week 1 — shows how one list of numbers is
spread out, by chopping the range into bins and drawing how many land in each.
""")

code('''
plt.hist(big_depths, bins=60)
plt.xlabel("depth (km)")
plt.ylabel("number of earthquakes")
plt.title(f"M6.5+ depths, 1976-2025 (n = {len(big_depths)})")
plt.show()
''')

md("""
Almost everything is crowded into the first few bins, near the surface. The count falls away fast,
runs close to nothing from about 300 to 500 km — and then picks up again in a small bump around
550 to 650 km before stopping dead. Earthquakes are not spread evenly through the depth of the
Earth, and the shape is stranger than "fewer as you go down". Why it looks like that is a question
for week 4; for now it is the reason anyone bothers sorting them by depth at all.

Seismologists sort them into three classes: **shallow** down to 70 km, **intermediate** from 70 to
300 km, and **deep** beyond 300 km. To count them in code, you need a way to treat some items
differently from others — and that is `if`.

`if` runs the indented code below it **only when the comparison is true**:
""")

code('''
depth = 45

if depth < 70:
    print("shallow")
''')

md("""
Change the 45 to 400 and run it again — nothing prints, because the comparison is false.

`else` catches the times it was false, and `elif` (short for "else if") offers another comparison
to try first:
""")

code('''
depth = 400

if depth < 70:
    print("shallow")
elif depth < 300:
    print("intermediate")
else:
    print("deep")
''')

md("""
Python tries the tests **top to bottom and stops at the first true one.**

That is why `elif depth < 300` does not need to say "between 70 and 300". A depth only reaches that
line if `depth < 70` was already false — so anything arriving there is 70 or more, and the second
test only has to rule out the rest. (Three separate `if` statements would test all three every
time, and count some depths twice. That is the mistake to watch for.)
""")

md("""
### The tests you can write

`if` needs something that is either true or false. These are the six comparisons, the two words
that join them, and one more that asks whether a value is in a list:

| Test | True when |
|---|---|
| `a < b` &nbsp; `a > b` | a is less than / greater than b |
| `a <= b` &nbsp; `a >= b` | …or equal to it |
| `a == b` | a equals b — **two** equals signs |
| `a != b` | a does not equal b |
| `a and b` | both tests are true |
| `a or b` | at least one is true |
| `x in things` | that value is somewhere in the list |

**`=` and `==` are different, and mixing them up is the most common beginner error in Python.**
One equals sign *assigns*: `depth = 45` puts 45 into `depth`. Two equals signs *ask*:
`depth == 45` is a question, and the answer is `True` or `False`.

Run this to see each one answer:
""")

code('''
depth = 211.9

print("deeper than 70?      ", depth > 70)
print("exactly 45?          ", depth == 45)
print("not 45?              ", depth != 45)
print("between 70 and 300?  ", depth >= 70 and depth < 300)
print("shallow or very deep?", depth < 70 or depth > 600)
print("1995 in deep years?  ", 1995 in big_years)
print("distance from 300 km:", abs(depth - 300))
''')

md("""
That fourth line is worth a second look. `depth >= 70 and depth < 300` is the *other* way to write
the intermediate class — spelling out both edges instead of leaning on `elif`. Both are correct.
`elif` is shorter because the earlier test has already ruled out everything below 70; `and` is
clearer when the two tests have nothing to do with each other.

`abs(x)` on the last line throws away a minus sign, so `abs(-88.1)` is `88.1`. It is how you ask
*how far apart* two numbers are without caring which is bigger.
""")

md("""
### Counting with `if`

To count things you do not need a list. A name holding a number is enough:

```
n = 0
n = n + 1
```

That second line looks strange the first time. It is not algebra — it is an instruction. Python
works out the right-hand side first (`n` plus one), then puts that answer back into `n`. So `n`
goes up by one.
""")

md("""
### Predict before you run

`big_depths` holds the depth of all 2192 M6.5+ earthquakes. Deep ones — more than 300 km down —
feel rare and exotic.

Out of 2192, how many do you think there are? Write your guess into the next cell before you run
the count. Getting it wrong is the point: a guess you had to commit to is what makes the real
number stick.
""")

answer_code('''
my_guess = 60
print("I think about", my_guess, "of the 2192 are deeper than 300 km")
''')

code('''
shallow = 0
intermediate = 0
deep = 0

for d in big_depths:
    if d < 70:
        shallow = shallow + 1
    elif d < 300:
        intermediate = intermediate + 1
    else:
        deep = deep + 1

print("shallow, under 70 km:      ", shallow)
print("intermediate, 70 to 300 km:", intermediate)
print("deep, 300 km and deeper:   ", deep)
print("adding up to", shallow + intermediate + deep, "of", len(big_depths))
''')

code(f'''
print("you guessed", my_guess, "— there are", deep,
      "— you were out by", abs(deep - my_guess))
{check("Predict", "guessed {my_guess}, actual {deep}")}
''')

md("""
Most people guess low. "Rare and exotic" is about how *strange* a 600-km earthquake is, not how
many there are: 197 out of 2192 is about one in eleven.
""")

md("""
### Looping over positions instead of values

`for d in big_depths` hands you the depths one at a time. That is fine when the depth is all you
need. But what if you want the **year** of each deep earthquake? The years are in a different list.

Section 2 is the answer: `big_depths` and `big_years` line up, so position `i` in one matches
position `i` in the other. You just need a loop that counts positions rather than handing you
values — and `range(len(...))` does exactly that.

`len(big_depths)` is 2192, so `range(len(big_depths))` is 0, 1, 2, … 2191: every position in the
list. Here it is on the small day lists, stopped at the first three:
""")

code('''
for i in range(3):
    print("position", i, "→ M", mags[i], "at", depths[i], "km,", places[i])
''')

md("""
✏️ **Your turn 5.** The hardest one here. It puts sections 2, 4 and 5 together, so take it in
pieces.

Build a list called `deep_years` holding the **year** of every earthquake deeper than 300 km.

**(a)** Start with an empty list, `deep_years = []`.

**(b)** Loop over the positions: `for i in range(len(big_depths)):`

**(c)** Inside the loop, use `if big_depths[i] >= 300:` to test that earthquake's depth.

**(d)** Inside the `if`, append **`big_years[i]`** — the year at the same position — to
`deep_years`. Note that (c) and (d) mean two levels of indent: the `if` sits inside the loop, and
the `append` sits inside the `if`.

Then print how many years are in the list, and how many of them are 1995 (`.count(1995)`).

If it worked, the length should match the `deep` number the cell above printed.
""")

answer_code('''
deep_years = []
for i in range(len(big_depths)):
    if big_depths[i] >= 300:
        deep_years.append(big_years[i])

print(len(deep_years), "earthquakes deeper than 300 km")
print(deep_years.count(1995), "of them in 1995")
''')

code(f'''
assert len(deep_years) == deep, \\
    "deep_years should hold one year per deep earthquake — the same number the cell above counted"
assert max(deep_years) <= 2025 and min(deep_years) >= 1976, \\
    "deep_years should hold YEARS — it looks like you appended the depth instead of big_years[i]"
{check("Your turn 5", "{len(deep_years)} deep earthquakes, {deep_years.count(1995)} of them in 1995")}
''')

# ---------------------------------------------------------------- 6
md("""
## 6. Writing a block of code once — a function

`def` gives a block of code a name, so you can use it again without writing it again. Here is
about the smallest one there is:
""")

code('''
def double(x):
    """twice whatever number you give it"""
    return x * 2


print(double(5))
print(double(21))
''')

md("""
Four parts. The first and third are required; the other two are not, strictly — but a function
with no `return` hands back nothing, and one with no docstring cannot tell anybody what it is
for, so in this course you write all four.

- **`def double(x):`** — the name you are giving it, and in brackets the name it will use for
  whatever it is handed. Ends in a colon, like `for` and `if`.
- **The docstring** — the triple-quoted line just inside, saying what the function is for.
- **The indented block** — what it actually does.
- **`return`** — the answer it hands back. `print` puts something on the screen; `return` gives a
  value back to whoever called the function, so it can be stored in a name or used in a sum.

Calling it is the easy half: `double(5)` runs the block with `x` set to 5, and comes back as 10.
""")

md("""
### Handing back two things

`return` is not limited to one value. Separate them with a comma, and catch them with two names:
""")

code('''
def high_and_low(values):
    """the largest and the smallest of a list, in that order"""
    return max(values), min(values)


hi, lo = high_and_low(mags)
print("the day ran from M", lo, "up to M", hi)
''')

md("""
You have already used this without knowing it. The setup cell at the top has this line in it:

```
times, depths, mags, places = columns(day)
```

`columns` is a function that returns four things at once, and that line catches them in four
names. Now you know how to write one.
""")

md("""
### Why you want one here

Look back at section 4. You wrote this:

```
counts = []
for year in years:
    counts.append(big_years.count(year))
```

and then, in Your turn 4, you wrote it again with `huge_counts` and `huge_years` in place of
`counts` and `big_years`. Three lines, copied, two names changed.

That is section 3's complaint again, one size up. A **loop** saves you from repeating a *line*; a
**function** saves you from repeating a *block*. Written once as a function, both of your loops
become one line each:
""")

code(f'''{weekkit.CHECKPOINT.format(body="""
# Your turn 4 built huge_counts. If you skipped it, or restarted the kernel, this puts it back
# so the cells below have something to compare against.
try:
    huge_counts
except NameError:
    huge_counts = []
    for year in years:
        huge_counts.append(huge_years.count(year))
""".strip())}''')

code('''
def counts_by_year(quake_years, year_list):
    """how many of these earthquakes fall in each year, one number per year"""
    result = []
    for year in year_list:
        result.append(quake_years.count(year))
    return result


# the two loops from section 4, now one line each
big_again = counts_by_year(big_years, years)
huge_again = counts_by_year(huge_years, years)

print("same answers as the loops you wrote?", big_again == counts, "and", huge_again == huge_counts)
''')

md("""
Both `True`, because nothing new happened — the function does exactly what your loops did. What
changed is that the block now has a name, and a third dataset would cost one line instead of three.

And this is what the docstring was for. `help()` prints it, so anyone can find out what a function
does without reading its insides:
""")

code('''
help(counts_by_year)
''')

md("""
✏️ **Your turn 6.** Write a function called `count_deeper_than` that takes **two** things — a list
of depths, and a depth limit — and hands back how many of those depths are at or deeper than the
limit.

**(a)** `def count_deeper_than(depth_list, limit):`

**(b)** A docstring saying what it does.

**(c)** Inside, the counting pattern from section 5: start `n = 0`, loop over `depth_list`, and use
`if d >= limit:` to decide whether to do `n = n + 1`.

**(d)** `return n` — outside the loop, so it hands back the final total rather than the first one.

Then call it twice and print both answers: once with `big_depths` and a limit of 300, which should
match the `deep` count from section 5, and once with a limit of 70.
""")

answer_code('''
def count_deeper_than(depth_list, limit):
    """how many of these earthquakes were at least `limit` kilometres deep"""
    n = 0
    for d in depth_list:
        if d >= limit:
            n = n + 1
    return n


print("deeper than 300 km:", count_deeper_than(big_depths, 300))
print("deeper than 70 km: ", count_deeper_than(big_depths, 70))
''')

code(f'''
assert count_deeper_than(big_depths, 300) == deep, \\
    "at a limit of 300 it should agree with the `deep` count from section 5"
assert count_deeper_than(big_depths, 70) == intermediate + deep, \\
    "at a limit of 70 it should be the intermediate and deep classes added together"
assert count_deeper_than([], 300) == 0, \\
    "an empty list has nothing in it — make sure n starts at 0, and that return is outside the loop"
{check("Your turn 6", "{count_deeper_than(big_depths, 300)} deeper than 300 km, "
       "{count_deeper_than(big_depths, 70)} deeper than 70 km")}
''')

# ---------------------------------------------------------------- closing
md("""
## The whole of it, on one page

Everything week 1 and this notebook have given you, grouped by what it is for. The examples are
real lines from this file, so you can try any of them in a cell.

### Values, and printing them

| Code | Example | What it does |
|---|---|---|
| `print(x)` | `print("largest:", biggest)` | show a value |
| `f"..."` | `f"M{biggest} at {places[where]}"` | build a sentence with values dropped into it |
| `round(x, 1)` | `round(mag_average, 1)` | trim a long decimal to one place |
| `sum(list)` | `sum(mags) / len(mags)` | add up every number — with `len()`, an average |
| `abs(x)` | `abs(-88.1)` → `88.1` | drop the minus sign: how far apart, not which way |

### Lists

| Code | Example | What it does |
|---|---|---|
| `len(list)` | `len(mags)` → `14` | how many items |
| `list[i]` | `mags[0]` → `7.0`, `mags[-1]` | one item, counting from 0; `-1` is the last |
| `list[a:b]` | `mags[0:3]` → `[7.0, 5.0, 4.5]` | a slice — a up to but not including b |
| `max()` `min()` | `max(mags)` → `7.0` | largest, smallest |
| `list.index(v)` | `mags.index(7.0)` → `0` | the **position** of a value, to read out of another list |
| `list.count(v)` | `big_years.count(1995)` → `66` | how many times a value appears |
| `list.append(x)` | `counts.append(53)` | add one item to the end |

### Doing it again — loops

| Code | Example | What it does |
|---|---|---|
| `for x in things:` | `for year in years:` | run the indented block once per item |
| `range(a, b)` | `range(1990, 2000)` | the whole numbers from a, stopping just before b |
| `range(len(things))` | `range(len(big_depths))` | every **position**, for when you need two lists at once |

### Choosing — `if`

| Code | Example | What it does |
|---|---|---|
| `if` / `elif` / `else` | `if d < 70:` … `elif d < 300:` | the first test that is true wins |
| `<` `>` `<=` `>=` | `d >= 300` | less than, greater than, or equal |
| `==` `!=` | `year == 1995` | equal, not equal — **two** equals signs |
| `and` `or` | `d >= 70 and d < 300` | both tests true / at least one true |
| `x in things` | `1995 in big_years` | is that value somewhere in the list |
| `n = n + 1` | `shallow = shallow + 1` | count, without building a list |

### Pictures

| Code | Example | What it does |
|---|---|---|
| `plt.plot(x, y)` | `plt.plot(years, counts)` | join points with a line — for something measured in order |
| `plt.scatter(x, y)` | `plt.scatter(mags, depths)` | one dot per pair, for points with no order |
| `plt.hist(x, bins=n)` | `plt.hist(big_depths, bins=60)` | how one list of numbers is spread out |
| `plt.xlabel` `ylabel` `title` | `plt.xlabel("depth (km)")` | say what the axes are — always |
| `plt.show()` | `plt.show()` | draw it |

### Your own — functions

| Code | Example | What it does |
|---|---|---|
| `def name(a, b):` | `def count_deeper_than(depth_list, limit):` | write the block once, give it a name |
| `return value` | `return n` | the answer it hands back to whoever called it |
| `return a, b` | `return max(values), min(values)` | two at once — catch with `hi, lo = f(v)` |
| `\"\"\"docstring\"\"\"` | `\"\"\"how many were deeper than limit\"\"\"` | what it is for; `help()` prints it |

### The four that catch everyone

1. **`=` assigns, `==` asks.** `depth = 45` puts 45 in. `depth == 45` is a question.
2. **The colon and the indent** are what make a block. No colon is a `SyntaxError`; no indent is an
   `IndentationError`. Both messages say so — read the last line.
3. **`range(a, b)` and `mags[a:b]` both stop just *before* b.** Same rule, two places.
4. **Read an error from the bottom up,** and look one line above where it points.

## Keep this notebook

When you cannot remember how a loop goes in week 9, or what `append` does in week 12, this is the
page to come back to — everything above is in one place, with the worked answers beside it in
`01b_practice_solution.ipynb`.

On Monday you point these pieces at other worlds, to work out which of them could hold liquid
water.
""")


# ---------------------------------------------------------------- emit
def build():
    answers = nbformat.v4.new_notebook()
    student = nbformat.v4.new_notebook()
    for kind, text, is_answer, tags in cells:
        if kind == "markdown":
            answers.cells.append(nbformat.v4.new_markdown_cell(text))
            student.cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            a = nbformat.v4.new_code_cell(text)
            s = nbformat.v4.new_code_cell(CODE_STUB if is_answer else text)
            if tags:
                a.metadata["tags"] = list(tags)
                s.metadata["tags"] = list(tags)
            answers.cells.append(a)
            student.cells.append(s)

    out = ROOT / PLATFORM["notebook_dir"]
    out.mkdir(parents=True, exist_ok=True)

    print(f"executing {len(answers.cells)} cells ...")
    with weekkit.pinned_kernel():
        NotebookClient(answers, timeout=600, kernel_name="python3",
                       resources={"metadata": {"path": str(out)}}).execute()

    for nb in (answers, student):
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                     "name": "python3"}
    nbformat.write(answers, out / f"{SLUG}_solution.ipynb")
    nbformat.write(student, out / f"{SLUG}.ipynb")
    print(f"wrote {SLUG}_solution.ipynb and {SLUG}.ipynb "
          f"({sum(1 for _, _, a, _ in cells if a)} questions, {len(cells)} cells)")


if __name__ == "__main__":
    build()
