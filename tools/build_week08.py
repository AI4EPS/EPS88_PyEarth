#!/usr/bin/env python
"""Build week 8 — "Was our earthquake forecast wrong — or were we just unlucky?"

Emits both notebooks from one source so they cannot drift:

    docs/notebooks/08_wrong_or_unlucky_solution.ipynb   executed, every output saved
    docs/notebooks/08_wrong_or_unlucky.ipynb            the same file with the answers deleted

It also writes the three cached fallbacks the week reads, into data/.

Every number that appears in prose or in a model answer is computed HERE, by running the same
code the notebook runs, with the same seed. Nothing is typed from memory or copied from the plan.

    python tools/build_week08.py
"""
import json
import pathlib
import subprocess
import sys
import urllib.request

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/notebooks"
SLUG = "08_wrong_or_unlucky"

course = yaml.safe_load((ROOT / "course.yml").read_text())
WEEK = next(s for s in course["schedule"] if s["n"] == 8)
PLATFORM = course["platform"]
CACHE_BASE = PLATFORM["cache_base"]

# ---------------------------------------------------------------------------
# the three live reads, pinned here so cache, notebook and prose cannot drift
# ---------------------------------------------------------------------------
SEA_URL = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=monthly_mean"
           "&station=9414290&datum=STND&units=metric&time_zone=GMT&format=csv"
           "&begin_date=19000101&end_date=20251231")
SEA_CACHE = "week08_sf_sea_level_1900_2025.csv"

FDSN = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&orderby=time-asc"
CA_BOX = "&minlatitude=32&maxlatitude=42&minlongitude=-125&maxlongitude=-114"
EQ_URL = (FDSN + "&starttime=1990-01-01&endtime=2026-01-01"
                 "&minmagnitude=3.5&maxmagnitude=6.9" + CA_BOX)
EQ_CACHE = "week08_ca_1990_2026_M3.5-6.9.csv"
BIG_URL = FDSN + "&starttime=1918-01-01&endtime=2026-01-01&minmagnitude=7.0" + CA_BOX
BIG_CACHE = "week08_ca_1918_2026_M7.csv"

B = 2000            # bootstrap resamples, everywhere in the week
SEED = 88
CHUNK_STARTS = [1900, 1925, 1950, 1975, 2000]
WINDOW_STARTS = [1918, 1954, 1990]
WINDOW_YEARS = 36


def cache(url, name):
    """Fetch a live source once and store it byte-for-byte as the week's fallback."""
    out = ROOT / "data" / name
    if not out.exists():
        out.write_bytes(urllib.request.urlopen(url, timeout=180).read())
    return out


# ---------------------------------------------------------------------------
# 1. measure everything the notebook will say, with the notebook's own code
# ---------------------------------------------------------------------------
sea = pd.read_csv(cache(SEA_URL, SEA_CACHE))
sea.columns = sea.columns.str.strip()
sea["year"] = sea["Year"] + (sea["Month"] - 0.5) / 12
sea["sea_mm"] = sea["MSL"] * 1000

quakes = pd.read_csv(cache(EQ_URL, EQ_CACHE))
big = pd.read_csv(cache(BIG_URL, BIG_CACHE))

M = {}
M["n_months"] = len(sea)
M["first_year"], M["last_year"] = int(sea["Year"].min()), int(sea["Year"].max())
M["n_years_sea"] = M["last_year"] - M["first_year"] + 1
M["n_missing"] = M["n_years_sea"] * 12 - M["n_months"]

fit = LinearRegression().fit(sea[["year"]], sea["sea_mm"])
M["slope"] = float(fit.coef_[0])
M["r2"] = float(fit.score(sea[["year"]], sea["sea_mm"]))

# --- the five 25-year chunks (the attempt that fails) ----------------------
M["chunk_slopes"] = []
for start in CHUNK_STARTS:
    chunk = sea[(sea["year"] >= start) & (sea["year"] < start + 25)]
    M["chunk_slopes"].append(
        float(LinearRegression().fit(chunk[["year"]], chunk["sea_mm"]).coef_[0]))
M["chunk_low"] = min(M["chunk_slopes"])
M["chunk_high"] = max(M["chunk_slopes"])
M["chunk_span"] = M["chunk_high"] - M["chunk_low"]
M["n_chunk"] = int(len(sea[(sea["year"] >= 1900) & (sea["year"] < 1925)]))


def slope_bootstrap(table, seed=SEED, n=B):
    """B bootstrap slopes of sea_mm against year, resampling whole months."""
    np.random.seed(seed)
    out = []
    for _ in range(n):
        boot = table.sample(len(table), replace=True)
        out.append(LinearRegression().fit(boot[["year"]], boot["sea_mm"]).coef_[0])
    return np.array(out)


slopes = slope_bootstrap(sea)
M["ci_low"], M["ci_high"] = (float(x) for x in np.percentile(slopes, [2.5, 97.5]))
M["n_below_zero"] = int((slopes <= 0).sum())
M["boot_min"] = float(slopes.min())
M["boot_max"] = float(slopes.max())

M["cm_century"] = M["slope"] * 100 / 10
M["cm_century_low"] = M["ci_low"] * 100 / 10
M["cm_century_high"] = M["ci_high"] * 100 / 10
M["years_to_2100"] = 2100 - 2025
M["cm_2100"] = M["slope"] * M["years_to_2100"] / 10
M["cm_2100_low"] = M["ci_low"] * M["years_to_2100"] / 10
M["cm_2100_high"] = M["ci_high"] * M["years_to_2100"] / 10

# --- the block bootstrap (the last section) -------------------------------
np.random.seed(SEED)
by_year = {y: sub for y, sub in sea.groupby("Year")}
year_list = sorted(by_year)
block_slopes = []
for _ in range(B):
    picked = np.random.choice(year_list, size=len(year_list), replace=True)
    boot = pd.concat([by_year[y] for y in picked])
    block_slopes.append(LinearRegression().fit(boot[["year"]], boot["sea_mm"]).coef_[0])
block_slopes = np.array(block_slopes)
M["block_low"], M["block_high"] = (float(x) for x in np.percentile(block_slopes, [2.5, 97.5]))
M["iid_width"] = M["ci_high"] - M["ci_low"]
M["block_width"] = M["block_high"] - M["block_low"]
M["width_ratio"] = M["block_width"] / M["iid_width"]
M["n_calendar_years"] = len(year_list)

# --- the earthquake forecast ----------------------------------------------
EDGES = np.arange(4.0, 5.6, 0.1).round(1)
M["n_quakes"] = len(quakes)
M["n_bins"] = len(EDGES)
M["count_at_4"] = int((quakes["mag"] >= 4.0).sum())
M["count_at_55"] = int((quakes["mag"] >= 5.5).sum())


def predicted_m7(mags, edges=EDGES):
    """Fit Gutenberg-Richter on these magnitude bins and read off the expected M7+ count."""
    counts = []
    for edge in edges:
        counts.append((mags >= edge).sum())
    line = LinearRegression().fit(edges.reshape(-1, 1), np.log10(counts))
    return 10 ** (line.intercept_ + line.coef_[0] * 7.0)


M["rate"] = float(predicted_m7(quakes["mag"]))


def rate_bootstrap(mags, edges=EDGES, seed=SEED, n=B):
    """B bootstrap values of the predicted M7+ count, resampling the catalogue."""
    np.random.seed(seed)
    out = []
    for _ in range(n):
        out.append(predicted_m7(mags.sample(len(mags), replace=True), edges))
    return np.array(out)


boot_rates = rate_bootstrap(quakes["mag"])
M["rate_low"], M["rate_high"] = (float(x) for x in np.percentile(boot_rates, [2.5, 97.5]))

np.random.seed(SEED)
boot_counts = np.random.poisson(boot_rates)
M["count_low"], M["count_high"] = (int(x) for x in np.percentile(boot_counts, [2.5, 97.5]))
M["frac_5"] = float((boot_counts >= 5).mean())
M["max_count"] = int(boot_counts.max())

big["when"] = pd.to_datetime(big["time"])
M["window_counts"] = []
for start in WINDOW_STARTS:
    inside = ((big["when"] >= f"{start}-01-01")
              & (big["when"] < f"{start + WINDOW_YEARS}-01-01"))
    M["window_counts"].append(int(inside.sum()))
M["observed"] = M["window_counts"][-1]
M["total_big"] = sum(M["window_counts"])
M["n_windows"] = len(WINDOW_STARTS)
M["span_years"] = WINDOW_YEARS * M["n_windows"]

np.random.seed(SEED)
long_counts = np.random.poisson(boot_rates * M["n_windows"])
M["frac_total"] = float((long_counts >= M["total_big"]).mean())
M["long_low"], M["long_high"] = (int(x) for x in np.percentile(long_counts, [2.5, 97.5]))

# --- homework -------------------------------------------------------------
HALVES = [(1900, 1963), (1963, 2026)]
M["halves"] = []
for lo, hi in HALVES:
    part = sea[(sea["year"] >= lo) & (sea["year"] < hi)]
    s = slope_bootstrap(part)
    M["halves"].append({
        "lo": lo, "hi": hi, "n": len(part),
        "slope": float(LinearRegression().fit(part[["year"]], part["sea_mm"]).coef_[0]),
        "low": float(np.percentile(s, 2.5)), "high": float(np.percentile(s, 97.5))})

FORKS = {"low": np.arange(3.5, 5.1, 0.1).round(1), "high": np.arange(4.5, 6.1, 0.1).round(1)}
M["forks"] = {}
for key, edges in FORKS.items():
    r = rate_bootstrap(quakes["mag"], edges)
    np.random.seed(SEED)
    c = np.random.poisson(r)
    M["forks"][key] = {
        "lo": float(edges[0]), "hi": float(edges[-1]),
        "rate": float(predicted_m7(quakes["mag"], edges)),
        "rate_low": float(np.percentile(r, 2.5)), "rate_high": float(np.percentile(r, 97.5)),
        "count_high": int(np.percentile(c, 97.5)), "frac_5": float((c >= 5).mean())}

if "--measure" in sys.argv:
    print(json.dumps(M, indent=1, default=float))
    sys.exit()

LO, HI = M["forks"]["low"], M["forks"]["high"]


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
    """A code answer cell: the solution carries the model answer, the student a stub."""
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
You have already fitted Gutenberg-Richter to California's small earthquakes and used the line to
predict how many magnitude-7 events the state should expect in thirty-six years. The line said
about one and a half. Five happened. You were asked to write down whether you thought your model
had failed, and the honest answer was that you could not tell — because 1.6 and 5 are two bare
numbers, and two bare numbers cannot be compared.

What is missing is a sense of how far that 1.6 would have moved if the earthquakes had fallen
slightly differently. Today you build it, out of nothing but the data you already have. You will
practise on something slower — a hundred and twenty-six years of sea level in San Francisco Bay,
where the question is whether the water is rising at all — and then take the same six lines back
to the earthquakes and settle the argument.
"""

md(weekkit.OPENING.format(question=WEEK["question"], datahub=datahub, hook=HOOK.strip()))

md("""
## What you'll be able to do

**The science.** Say how fast San Francisco Bay is rising, with a range rather than a number, and
say whether the rise could be zero. Then say whether California's five large earthquakes are
evidence that a Gutenberg-Richter forecast is broken, or the kind of run of luck a working
forecast produces anyway — and know which of those two questions your interval answered.

**The skills.** Resampling: `table.sample(n, replace=True)` draws a new dataset out of the one you
have, `np.percentile` turns a thousand answers into an interval, and a `for` loop puts the two
together. You will put an error bar on a fitted slope, which is something a single call to
`LinearRegression` will never give you.

**Eight places where you write something: five in class, three at home.** Each one is headed
*Your turn*, with an empty cell under it.
""")

code(weekkit.setup_cell(
    imports="import numpy as np\nfrom sklearn.linear_model import LinearRegression\n",
    figsize="(7, 4)",
    cache_base=CACHE_BASE,
    signature="url, cached",
    docstring="Read one live source; fall back to the copy stored with the course.",
    url_expr="url",
    cache_expr="cached",
    unpack=f'''
sea = load("{SEA_URL}",
           "{SEA_CACHE}")
sea.columns = sea.columns.str.strip()      # the NOAA file pads its column names with spaces
sea["year"] = sea["Year"] + (sea["Month"] - 0.5) / 12   # 1903.5 means the middle of 1903
sea["sea_mm"] = sea["MSL"] * 1000                       # the file is in metres, we want mm

FDSN = "{FDSN}"
CA_BOX = "{CA_BOX}"

quakes = load(FDSN + "&starttime=1990-01-01&endtime=2026-01-01"
                     "&minmagnitude=3.5&maxmagnitude=6.9" + CA_BOX,
              "{EQ_CACHE}")
big = load(FDSN + "&starttime=1918-01-01&endtime=2026-01-01&minmagnitude=7.0" + CA_BOX,
           "{BIG_CACHE}")

print("sea level:", sea.shape, " small earthquakes:", quakes.shape, " large ones:", big.shape)
'''.strip("\n")))

# --- section 1 -------------------------------------------------------------
md(f"""
## A slope with nothing after it

A tide gauge is a float in a stilling well, bolted to a pier, writing down where the water sits.
The one in San Francisco — NOAA station 9414290 — has been doing that since before anyone thought
sea level was a question, and NOAA publishes a monthly mean from it. That is what `sea` holds:
{M['n_months']:,} monthly means from {M['first_year']} to {M['last_year']}, in the `MSL` column,
which the setup cell turned into millimetres in `sea_mm`. ({M['n_years_sea']} years is
{M['n_years_sea'] * 12:,} months, so {M['n_missing']} are missing.)

Two warnings about what that column is. The zero is arbitrary: the heights are quoted against
*station datum*, a mark on the pier chosen for convenience, so the height itself means nothing and
only the *change* does. And a tide gauge measures the sea against the land it is bolted to, so
what it records is **relative** sea level: if the pier were sinking, the water would appear to
rise by exactly the same amount.
""")

code("""
plt.scatter(sea["year"], sea["sea_mm"], s=2)
plt.xlabel("year")
plt.ylabel("monthly mean sea level (mm above station datum)")
plt.title(f"San Francisco, {len(sea):,} monthly means")
plt.show()
""")

md("""
There is a rise in there, and a great deal of noise on top of it: winter storms, El Niño years,
the seasons. A straight line through it is the regression you already know, with `year` as the one
input column and `sea_mm` as the thing to predict. Its slope is millimetres per year.
""")

code("""
fit = LinearRegression().fit(sea[["year"]], sea["sea_mm"])
slope = fit.coef_[0]

print("slope:", round(slope, 3), "mm per year")
print("R squared:", round(fit.score(sea[["year"]], sea["sea_mm"]), 3))
""")

md(f"""
{M['slope']:.3f} millimetres a year, and an R squared of {M['r2']:.3f} — the line explains about
half of what the gauge did, and the rest is the month-to-month weather and ocean variability the
line cannot see. Now ask the question that matters: **could
the answer be zero?** The number {M['slope']:.3f} does not answer it. It has nothing after it.

Here is the obvious first move. If the record were telling us something stable, then any big piece
of it should give roughly the same slope. So cut it up and look.
""")

ask(f"""
### ✏️ Your turn 1

Cut the record into five twenty-five-year pieces and fit the same line to each one on its own.

Loop over `{CHUNK_STARTS}`. For each `start`, keep the rows with
`(sea["year"] >= start) & (sea["year"] < start + 25)`, fit `LinearRegression` to that piece exactly
as the cell above did, and append the slope to a list. Print each piece's start year, how many
months it holds, and its slope.

**Use these names**, because the self-check looks for them: `chunk_slopes`.
""")

answer(f"""
chunk_slopes = []

for start in {CHUNK_STARTS}:
    chunk = sea[(sea["year"] >= start) & (sea["year"] < start + 25)]
    chunk_fit = LinearRegression().fit(chunk[["year"]], chunk["sea_mm"])
    chunk_slopes.append(chunk_fit.coef_[0])
    print(start, "-", start + 25, ":", len(chunk), "months, slope",
          round(chunk_fit.coef_[0], 3), "mm/yr")
""", f"""
assert len(chunk_slopes) == {len(CHUNK_STARTS)}, "one slope per twenty-five-year piece"
print("\u2713 the record cut five ways \u2014 slopes from", round(min(chunk_slopes), 2),
      "to", round(max(chunk_slopes), 2), "mm/yr, against", round(slope, 2), "for the whole record")
""")

md(f"""
Five pieces of the same record, and the slopes run from {M['chunk_low']:.2f} to
{M['chunk_high']:.2f} mm/yr. The first quarter of the twentieth century says the Bay was barely
moving; the last twenty-five years say four millimetres a year.

That is honest but useless, for two separate reasons. First, two different things are mixed
together in that spread and staring at five numbers cannot separate them: how much a
twenty-five-year slope wobbles by chance, and whether the rise has genuinely sped up. Second — and
this is the fatal one — we did not ask how much a *twenty-five-year* slope wobbles. We asked how
much the {M['n_years_sea']}-year slope wobbles, and each of those fits threw away four fifths of
the data. Five numbers also cannot tell you what "95% of the time" means.

What we want is {M['n_months']:,} months' worth of answer, many times over. We only have one
record. So we make more of it out of the one we have.
""")

# --- section 2 -------------------------------------------------------------
md("""
## Asking the data the same question a thousand times

The trick has a name and one line of code. **Bootstrap:** *Ask the data the same question a
thousand times, using a different random slice of itself each time.*

The slice is drawn **with replacement**: pick a row at random, write it down, put it back, and
pick again, until you have as many rows as you started with. Some rows get picked twice, some not
at all — and that is the whole point, because that is what a second run of history would have
looked like. Watch it happen on eight numbers first.
""")

code("""
np.random.seed(88)
eight = pd.DataFrame({"height": [1, 2, 3, 4, 5, 6, 7, 8]})

print("the original:", list(eight["height"]))
print("one resample:", list(eight.sample(8, replace=True)["height"]))
print("another:     ", list(eight.sample(8, replace=True)["height"]))
""")

md(f"""
Repeats in one, holes in the other, same length as the original. `np.random.seed` is there for the
same reason as in the probability week: it makes the randomness repeatable, so your numbers and
your neighbour's match.

Now do it to the tide gauge. One resample of `sea` is {M['n_months']:,} months drawn with
replacement from the {M['n_months']:,} months we have; fit the line to that and you get a slope
that is nearly, but not quite, {M['slope']:.3f}. Do it {B:,} times and you have {B:,} slopes.

Notice which way round this is. When you simulated earthquake times you started from a
distribution you had *assumed* — `np.random.poisson(lam)` — and asked what worlds it would make.
Here you assume nothing at all and draw from the data itself.
""")

ask(f"""
### ✏️ Your turn 2

Bootstrap the slope, and turn the answer into a range.

Seed with `np.random.seed(88)`. Then, {B:,} times: draw `sea.sample(len(sea), replace=True)`, fit
`LinearRegression` to that resample as you have twice already, and append `.coef_[0]` to a list.
Make the finished list an array with `np.array(...)` so you can do arithmetic on it.

Then read the middle 95% off it: `np.percentile(slopes, [2.5, 97.5])` returns the two values that
cut off the bottom 2.5% and the top 2.5%. Print the slope, the two ends, and how many of your
{B:,} slopes came out at zero or below — `(slopes <= 0).sum()`.

**Use these names**, because the self-check looks for them: `slopes`, `ci_low`, `ci_high`.
""")

answer(f"""
np.random.seed(88)
slopes = []

for i in range({B}):
    boot = sea.sample(len(sea), replace=True)
    slopes.append(LinearRegression().fit(boot[["year"]], boot["sea_mm"]).coef_[0])

slopes = np.array(slopes)
ci_low, ci_high = np.percentile(slopes, [2.5, 97.5])

print("slope:", round(slope, 3), "mm/yr")
print("95% interval:", round(ci_low, 3), "to", round(ci_high, 3), "mm/yr")
print("resamples at zero or below:", (slopes <= 0).sum(), "out of", len(slopes))
""", f"""
assert len(slopes) == {B}, "one slope per resample — the list should be as long as the loop"
assert ci_low < slope < ci_high, "the interval should straddle the slope you started from"
print("\u2713 the slope, with an interval \u2014", round(slope, 2), "mm/yr, 95% interval",
      round(ci_low, 2), "to", round(ci_high, 2))
""")

md(f"""
That range is a **confidence interval**: *Not one number but the range your number would have
wandered over, had the world rolled differently.* Written the way a paper would write it:
{M['slope']:.2f} mm/yr, 95% CI [{M['ci_low']:.2f}, {M['ci_high']:.2f}].

Every one of those {B:,} slopes is worth drawing, because the interval is just two cuts through a
shape.
""")

code(f"""
plt.hist(slopes, bins=40)
plt.axvline(ci_low, color="firebrick")
plt.axvline(ci_high, color="firebrick")
plt.xlabel("bootstrap slope (mm per year)")
plt.ylabel("number of resamples")
plt.title(f"{{len(slopes):,}} bootstrap slopes, 95% interval marked")
plt.show()
""")

md(f"""
A hump, and the red lines cut {int(round(0.025 * B))} resamples off each side. Two things to read
off it. Nothing in it comes anywhere near zero — the count you printed of resamples at zero or
below was {M['n_below_zero']} out of {B:,} — so the answer to *could the Bay be flat?* is no, and
now we can say so rather than assert it.

Second, look at how narrow it is: {M['iid_width']:.2f} mm/yr wide, against the
{M['chunk_span']:.2f} mm/yr that separated the highest of your five chunks from the lowest. Cutting
the record up made the answer look far less certain than it is, because each piece had a
quarter-century of noise to fit through and a quarter of the data to do it with.
""")

# --- section 3 -------------------------------------------------------------
md("""
## From millimetres a year to centimetres of water

Millimetres per year is not a quantity anyone plans a seawall around. Two conversions get asked
for, they are not the same number, and they are confused constantly:

- **per century** — how much the water rises in a hundred years at this rate.
- **from the end of this record to 2100** — which is a shorter stretch, so it is a smaller number.

Whichever you quote, the interval comes with it: convert `ci_low` and `ci_high` exactly as you
convert the slope, and the answer stays a range.
""")

code(weekkit.CHECKPOINT.format(body="""fit = LinearRegression().fit(sea[["year"]], sea["sea_mm"])
slope = fit.coef_[0]"""))

ask(f"""
### ✏️ Your turn 3

Report the rise both ways, each as a point estimate and a 95% interval, in **centimetres**.

There are 10 millimetres in a centimetre. The record ends at the close of {M['last_year']}, so
2100 is {M['years_to_2100']} years away — not 100.

Print two lines: the rise per century, and the rise from the end of the record to 2100, each with
its interval. Do the conversion on `ci_low` and `ci_high` as well as on `slope`.

**Use these names**, because the self-check looks for them: `cm_century`, `cm_2100`.
""")

answer(f"""
cm_century = slope * 100 / 10
cm_2100 = slope * {M['years_to_2100']} / 10

print("per century:      ", round(cm_century, 1), "cm  95% interval",
      round(ci_low * 100 / 10, 1), "to", round(ci_high * 100 / 10, 1))
print("end of record to 2100:", round(cm_2100, 1), "cm  95% interval",
      round(ci_low * {M['years_to_2100']} / 10, 1), "to",
      round(ci_high * {M['years_to_2100']} / 10, 1))
""", """
assert cm_2100 < cm_century, "2100 is closer than a century away, so it must be the smaller number"
print("\u2713 the same slope, two questions \u2014", round(cm_century, 1),
      "cm per century, but", round(cm_2100, 1), "cm between the end of the record and 2100")
""")

md(f"""
{M['cm_century']:.1f} cm per century and {M['cm_2100']:.1f} cm by 2100 are the same slope answering
two different questions, and quoting one for the other is a {M['cm_century'] - M['cm_2100']:.0f}-centimetre
mistake.

Be careful about what the second number is. It is what the water does **if the rise continues at
the average rate of the last {M['n_years_sea']} years**, and the chunk slopes you computed give a
plain reason to doubt that: the most recent twenty-five years came out at {M['chunk_slopes'][-1]:.1f}
mm/yr, more than twice the long-run figure. The interval [{M['cm_2100_low']:.1f},
{M['cm_2100_high']:.1f}] cm is the uncertainty in *this straight line*, not the uncertainty in what
the ocean will do. It is a floor, not a forecast.
""")

# --- section 4 -------------------------------------------------------------
md(f"""
## Wrong, or unlucky?

Back to the earthquakes, with the same six lines.

`quakes` is every California earthquake between magnitude 3.5 and 6.9 recorded in thirty-six
years — {M['n_quakes']:,} of them, in the same latitude-longitude box you used before. The
Gutenberg-Richter recipe is the one you already know: count how many events reached each
magnitude, plot those counts on a log axis against magnitude, fit a straight line, and read the
line off at a magnitude you have not observed.

Written as a function it is short enough to call {B:,} times.
""")

code(weekkit.CHECKPOINT.format(body=f'''quakes = load(FDSN + "&starttime=1990-01-01&endtime=2026-01-01"
                     "&minmagnitude=3.5&maxmagnitude=6.9" + CA_BOX,
              "{EQ_CACHE}")'''))

code(f"""
edges = np.arange(4.0, 5.6, 0.1).round(1)      # fit between magnitude 4.0 and 5.5


def predicted_m7(mags):
    \"\"\"Fit Gutenberg-Richter to these magnitudes and read off the expected number of M7+.\"\"\"
    counts = []
    for edge in edges:
        counts.append((mags >= edge).sum())
    line = LinearRegression().fit(edges.reshape(-1, 1), np.log10(counts))
    return 10 ** (line.intercept_ + line.coef_[0] * 7.0)


print("events at magnitude 4.0 and above:", (quakes["mag"] >= 4.0).sum())
print("events at magnitude 5.5 and above:", (quakes["mag"] >= 5.5).sum())
print("expected number of M7+ in the same span:", round(predicted_m7(quakes["mag"]), 2))
""")

md(f"""
{M['rate']:.2f} expected, against five that actually happened. That is where the argument stopped
when you first fitted the line.
""")

md(f"""
### Predict before you run

Bootstrap that {M['rate']:.2f} the way you just bootstrapped the slope and you get a 95% interval
around it. **How high do you think the top of that interval reaches?** Commit to a number — change
`my_guess` and run it — and then find out.
""")

code("""
my_guess = 3.0

print("you guessed the interval reaches:", my_guess, "M7+ earthquakes")
print("five actually happened")
""")

ask(f"""
### ✏️ Your turn 4

Bootstrap the forecast. This is the loop you wrote for the slope, with one line changed.

Seed with `np.random.seed(88)`. Then, {B:,} times: draw
`quakes["mag"].sample(len(quakes), replace=True)`, pass it to `predicted_m7`, and append the
answer. Make the list an array, then take `np.percentile(boot_rates, [2.5, 97.5])`.

Print the forecast and its interval.

**Use these names**, because the self-check looks for them: `boot_rates`, `rate_low`, `rate_high`.
""")

answer(f"""
np.random.seed(88)
boot_rates = []

for i in range({B}):
    boot_mags = quakes["mag"].sample(len(quakes), replace=True)
    boot_rates.append(predicted_m7(boot_mags))

boot_rates = np.array(boot_rates)
rate_low, rate_high = np.percentile(boot_rates, [2.5, 97.5])

print("forecast:", round(predicted_m7(quakes["mag"]), 2), "M7+ earthquakes")
print("95% interval:", round(rate_low, 2), "to", round(rate_high, 2))
""", f"""
assert len(boot_rates) == {B}, "one forecast per resample"
assert rate_low < rate_high, "percentile returns the low end first"
print("\u2713 the forecast, with an interval \u2014", round(rate_low, 2), "to",
      round(rate_high, 2), "M7+ earthquakes, against 5 observed")
""")

code(f"""
plt.hist(boot_rates, bins=40)
plt.axvline(rate_high, color="firebrick")
plt.axvline(5, color="black")
plt.xlabel("bootstrap forecast (M7+ earthquakes in {M['span_years'] // M['n_windows']} years)")
plt.ylabel("number of resamples")
plt.title(f"{{len(boot_rates):,}} bootstrap forecasts; interval top red, 5 observed black")
plt.show()
""")

md(f"""
The interval runs [{M['rate_low']:.2f}, {M['rate_high']:.2f}], and the black line at 5 is off past
the end of everything. Not one of the {B:,} resampled catalogues produced a forecast anywhere near
five. Case closed?

No — and this is the most important paragraph in the week. That interval answers the question
*how well do we know the average rate?* Five is not an average rate. Five is a **count**, one
draw of a thing that scatters even when the rate is exactly right, and you have already simulated
exactly that scatter: `np.random.poisson(lam)` turns a rate into the number of events an
individual thirty-six years actually delivers. Comparing a count to an interval on a rate is
comparing two different quantities.

So put the two sources of wobble together. Take each of your {B:,} bootstrapped rates, and let the
world roll once at that rate.
""")

ask(f"""
### ✏️ Your turn 5

Turn the interval on the rate into an interval on the count.

`np.random.poisson(boot_rates)` draws one Poisson count for every rate in the array at once, so
this needs no loop: seed with `np.random.seed(88)`, make `boot_counts`, and take
`np.percentile(boot_counts, [2.5, 97.5])`.

Then print the fraction of those simulated worlds that delivered five or more —
`(boot_counts >= 5).mean()`. That fraction is the answer to *were we unlucky?*

**Use these names**, because the self-check looks for them: `boot_counts`.
""")

answer("""
np.random.seed(88)
boot_counts = np.random.poisson(boot_rates)

count_low, count_high = np.percentile(boot_counts, [2.5, 97.5])

print("95% interval on the COUNT:", count_low, "to", count_high, "M7+ earthquakes")
print("fraction of simulated worlds with 5 or more:", round((boot_counts >= 5).mean(), 4))
print("the busiest simulated world had:", boot_counts.max())
""", f"""
assert len(boot_counts) == len(boot_rates), "one simulated world per bootstrapped rate"
assert boot_counts.max() > boot_rates.max(), "a count scatters above the rate that made it — check you drew from Poisson"
print("\u2713 an interval on the count \u2014 5 or more happened in",
      round(100 * (boot_counts >= 5).mean(), 1), "% of simulated worlds")
""")

code("""
plt.hist(boot_counts, bins=np.arange(-0.5, boot_counts.max() + 1.5, 1))
plt.axvline(5, color="black")
plt.xlabel("M7+ earthquakes in a simulated 36 years")
plt.ylabel("number of simulated worlds")
plt.title(f"{len(boot_counts):,} simulated worlds, 5 observed marked in black")
plt.show()
""")

md(f"""
The picture is completely different. Once the counting noise is in, worlds with five large
earthquakes do turn up: {100 * M['frac_5']:.1f}% of them, and the busiest reached
{M['max_count']}. The 95% interval on the count is [{M['count_low']}, {M['count_high']}], and five
is sitting on its upper edge rather than outside it.

So the honest verdict on this window is *unlikely, not impossible* — a one-in-{round(1 / M['frac_5']):.0f}
outcome. That is the kind of result that should send you to look at more data rather than to a
conclusion. And there is more data: thirty-six years is not the only thirty-six years California
has had.
""")

code(f"""
big["when"] = pd.to_datetime(big["time"])
print(big[["mag", "place"]])

window_counts = []
for start in {WINDOW_STARTS}:
    inside = (big["when"] >= f"{{start}}-01-01") & (big["when"] < f"{{start + {WINDOW_YEARS}}}-01-01")
    window_counts.append(inside.sum())
    print(start, "-", start + {WINDOW_YEARS}, ":", inside.sum(), "M7+ earthquakes")
""")

md(f"""
{M['window_counts'][0]}, {M['window_counts'][1]}, {M['observed']}. The forecast of {M['rate']:.2f}
per {WINDOW_YEARS} years is an excellent description of the two earlier windows and a poor one of
the most recent. So the last cell: if the rate really is what Gutenberg-Richter says, and it held
for all {M['span_years']} years, how often do you get {M['total_big']} or more in total?
""")

code(f"""
np.random.seed(88)
long_counts = np.random.poisson(boot_rates * {M['n_windows']})

print("expected over", {M['span_years']}, "years:", round({M['n_windows']} * predicted_m7(quakes["mag"]), 2))
print("observed:", sum(window_counts))
print("fraction of simulated worlds reaching that:",
      round((long_counts >= sum(window_counts)).mean(), 3))
""")

md(f"""
{M['total_big']} against an expected {M['n_windows'] * M['rate']:.1f}, and {100 * M['frac_total']:.0f}%
of simulated {M['span_years']}-year worlds do at least that well. On the long record the forecast
is not in trouble at all.

Two caveats you should carry, because neither is settled by anything above. The box is a
rectangle of latitude and longitude, not the state, so it collects earthquakes in Nevada and Baja
California as well: of the {M['total_big']} the cell above listed, Fairview Peak is in Nevada and
Sierra El Mayor is in Baja California. And the earlier windows
depend on a catalogue that was thinner: *A catalogue lists what somebody's instruments recorded,
not what happened.* If those windows are undercounted, the true long-run rate is higher than
{M['total_big']} in {M['span_years']} years, and that pushes the same way — it makes five look less
exceptional, not more.
""")

# --- section 5 -------------------------------------------------------------
md(f"""
## One assumption we did not say out loud

Back to the Bay for the assumption the bootstrap slipped past you. Resampling *months* treats each
month as an independent draw — as if the sea level in March told you nothing about April. It
plainly does. A wet winter, an El Niño, a warm year: those last longer than a month, so
neighbouring months carry much of the same information, and {M['n_months']:,} months is worth
rather less than {M['n_months']:,} independent measurements.

The fix is to resample bigger pieces. Draw whole **calendar years** with replacement — all twelve
months of 1931 or none of them — so that whatever is shared inside a year travels with it. There
are {M['n_calendar_years']} years to draw from. `pd.concat` is the function that stacks the chosen
years back into one table.
""")

code(f"""
np.random.seed(88)
by_year = {{}}
for y, one_year in sea.groupby("Year"):
    by_year[y] = one_year

block_slopes = []
for i in range({B}):
    picked = np.random.choice(sorted(by_year), size=len(by_year), replace=True)
    boot = pd.concat([by_year[y] for y in picked])
    block_slopes.append(LinearRegression().fit(boot[["year"]], boot["sea_mm"]).coef_[0])

block_slopes = np.array(block_slopes)
block_low, block_high = np.percentile(block_slopes, [2.5, 97.5])

print("resampling months:", round(ci_low, 2), "to", round(ci_high, 2), "mm/yr")
print("resampling years: ", round(block_low, 2), "to", round(block_high, 2), "mm/yr")
""")

code("""
plt.hist(slopes, bins=40, label="months resampled")
plt.hist(block_slopes, bins=40, alpha=0.6, label="whole years resampled")
plt.xlabel("bootstrap slope (mm per year)")
plt.ylabel("number of resamples")
plt.title(f"{len(slopes):,} resamples each, the same record")
plt.legend()
plt.show()
""")

md(f"""
The interval got **wider** — [{M['block_low']:.2f}, {M['block_high']:.2f}] against
[{M['ci_low']:.2f}, {M['ci_high']:.2f}], about {M['width_ratio']:.1f} times the width — and that is
the direction nobody guesses. Respecting the structure of your data does not sharpen the answer;
it stops you from claiming a sharpness you never had. The month-by-month interval was too narrow,
because it counted {M['n_months']:,} pieces of information where there were closer to
{M['n_calendar_years']}.

Zero is still nowhere near either interval, so the conclusion about the Bay survives. That will
not always be true, and when it is not, the wider interval is the one to believe.
""")

# --- the closing -----------------------------------------------------------
md(f"""
{weekkit.CLOSING_HEADING}

**We cannot show it was wrong — and the reason we can now say that, rather than shrug, is that
the forecast finally has an interval attached.** Bootstrapping the catalogue puts the
Gutenberg-Richter rate at {M['rate']:.2f} M7+ earthquakes per {WINDOW_YEARS} years, 95% CI
[{M['rate_low']:.2f}, {M['rate_high']:.2f}], and five is far outside that. But a rate is not a
count: fold in the Poisson scatter that any real thirty-six years is subject to and five happens
in {100 * M['frac_5']:.1f}% of simulated worlds, sitting on the top edge of the interval on the
count rather than beyond it. Widen the window and the case for a broken model gets weaker still — {M['total_big']} large
earthquakes in {M['span_years']} years against {M['n_windows'] * M['rate']:.1f} expected, which
{100 * M['frac_total']:.0f}% of simulated worlds match or beat. The forecast was not convicted; it
was not acquitted either, and the honest report is the interval rather than the verdict.
""")

md(weekkit.week_cheatsheet(8))

# --- homework --------------------------------------------------------------
md("""
## Homework

Three parts on the two datasets you already have loaded. Part 1 goes back to the Bay and asks
something class did not; parts 2 and 3 make you re-run the earthquake argument on a fitting range
you choose yourself. If you have restarted since class, run the setup cell at the top first, then
the checkpoint cells in the sections you need.
""")

ask(f"""
### ✏️ Your turn 6

Class cut the record into five pieces and found slopes from {M['chunk_low']:.2f} to
{M['chunk_high']:.2f} mm/yr, then never went back to ask whether that spread was real. Settle it
with two intervals rather than five point estimates.

Split the record in half — {HALVES[0][0]} up to {HALVES[0][1]}, and {HALVES[1][0]} up to
{HALVES[1][1]} — and bootstrap each half separately, exactly as you bootstrapped the whole record
in your turn 2. Print each half's slope and 95% interval.

Then print whether the two intervals overlap. Two intervals overlap when the lower one's top end is
above the higher one's bottom end, so `first_high > second_low` is the test.

**Use these names**, because the self-check looks for them: `first`, `second`, `first_low`,
`first_high`, `second_low`, `second_high`.
""")

answer(f"""
first = sea[(sea["year"] >= {HALVES[0][0]}) & (sea["year"] < {HALVES[0][1]})]
second = sea[(sea["year"] >= {HALVES[1][0]}) & (sea["year"] < {HALVES[1][1]})]

np.random.seed(88)
first_slopes = []
for i in range({B}):
    boot = first.sample(len(first), replace=True)
    first_slopes.append(LinearRegression().fit(boot[["year"]], boot["sea_mm"]).coef_[0])
first_low, first_high = np.percentile(first_slopes, [2.5, 97.5])

np.random.seed(88)
second_slopes = []
for i in range({B}):
    boot = second.sample(len(second), replace=True)
    second_slopes.append(LinearRegression().fit(boot[["year"]], boot["sea_mm"]).coef_[0])
second_low, second_high = np.percentile(second_slopes, [2.5, 97.5])

print("{HALVES[0][0]}-{HALVES[0][1]}:", len(first), "months, slope",
      round(LinearRegression().fit(first[["year"]], first["sea_mm"]).coef_[0], 2),
      "95% interval", round(first_low, 2), "to", round(first_high, 2))
print("{HALVES[1][0]}-{HALVES[1][1]}:", len(second), "months, slope",
      round(LinearRegression().fit(second[["year"]], second["sea_mm"]).coef_[0], 2),
      "95% interval", round(second_low, 2), "to", round(second_high, 2))
print("do the intervals overlap?", first_high > second_low)
""", """
assert len(first) + len(second) == len(sea), "every month belongs to exactly one half"
print("\u2713 the two halves \u2014 first", round(first_low, 2), "to", round(first_high, 2),
      ", second", round(second_low, 2), "to", round(second_high, 2),
      "; overlapping:", first_high > second_low)
""")

ask(f"""
### ✏️ Your turn 7

Class fit Gutenberg-Richter between magnitude 4.0 and 5.5. That was a choice, and it was not the
only defensible one — too low and the catalogue is missing small events, too high and there are
barely any events left to fit.

**Pick one:** fit between {LO['lo']} and {LO['hi']}, or between {HI['lo']} and {HI['hi']}. Change
one line — `edges = np.arange(...).round(1)` — and re-run the whole argument on your choice:
`predicted_m7`, the {B:,} bootstrap rates, the Poisson draw, and the fraction of simulated worlds
reaching five.

Print your forecast, your 95% interval on the count, and your fraction. Notice that redefining
`edges` is enough: `predicted_m7` reads it, so nothing else needs editing.

**Use these names**, because the self-check looks for them: `edges`, `my_counts`, `my_fraction`.
""")

answer(f"""
edges = np.arange({LO['lo']}, {LO['hi'] + 0.1:.1f}, 0.1).round(1)   # the low choice

np.random.seed(88)
my_rates = []
for i in range({B}):
    my_rates.append(predicted_m7(quakes["mag"].sample(len(quakes), replace=True)))
my_rates = np.array(my_rates)

np.random.seed(88)
my_counts = np.random.poisson(my_rates)
my_fraction = (my_counts >= 5).mean()

print("fitting magnitude", edges[0], "to", edges[-1])
print("forecast:", round(predicted_m7(quakes["mag"]), 2), "M7+ earthquakes")
print("95% interval on the count:", np.percentile(my_counts, [2.5, 97.5]))
print("fraction of simulated worlds with 5 or more:", round(my_fraction, 4))
""", f"""
assert edges[0] != 4.0 or edges[-1] != 5.5, "change the range — this is class's fit, not a choice"
assert len(my_counts) == {B}, "one simulated world per resample — re-run the whole loop on your range"
print("\u2713 your own fitting range \u2014 magnitude", edges[0], "to", edges[-1],
      ", 5 or more reached in", round(100 * my_fraction, 1), "% of simulated worlds")
""")

ask("""
### ✏️ Your turn 8

Two or three sentences, quoting your own numbers from part 2.

Class's fitting range put five large earthquakes at the top edge of the interval. Say what your
range put it at, and whether your choice changes the verdict — whether a reader of your notebook
would come away thinking the forecast is broken, or thinking the last thirty-six years were busy.
Then say what would have to be true for a single thirty-six-year count to settle the question
either way.
""")

answer_prose(f"""
Fitting between {LO['lo']} and {LO['hi']} gives a forecast of {LO['rate']:.2f} M7+ earthquakes, a
95% interval on the count topping out at {LO['count_high']}, and five or more in only
{100 * LO['frac_5']:.1f}% of simulated worlds — so on my choice five is *outside* the interval and
the forecast does look broken. Class's range, 4.0 to 5.5, gave {100 * M['frac_5']:.1f}%, with five
sitting on the edge rather than beyond it, and the third choice, {HI['lo']} to {HI['hi']}, has only
{M['count_at_55']} events above magnitude 5.5 left to fit, so its interval is wider still. The
verdict therefore turns on a choice nobody can defend as the only right one, and the answer moves
more when I change my fitting range than the observation moves it. For a single thirty-six-year
count to settle the question, it would have to fall outside the interval under *every* defensible
fitting range — five does not — or the window would have to be long enough that the Poisson
scatter, which is roughly the square root of the expected count, became small beside the gap being
argued about.
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
    print(f"cache: data/{SEA_CACHE}, data/{EQ_CACHE}, data/{BIG_CACHE}")


if __name__ == "__main__":
    main()
    weekkit.gate(8)
