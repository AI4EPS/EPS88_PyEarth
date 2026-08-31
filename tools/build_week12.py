#!/usr/bin/env python
"""Build week 12 — "Can you find a fault that nobody mapped?"

Emits both notebooks from one source so they cannot drift:

    docs/notebooks/12_hidden_fault_solution.ipynb   executed, every output saved
    docs/notebooks/12_hidden_fault.ipynb            the same file with the answers deleted

It also writes the week's cached fallbacks: one CSV per catalogue window the notebook reads.

Every number that appears in prose or in a model answer is computed HERE, from the same files
the notebook reads, and formatted in. Nothing is typed from memory or copied from the plan.

The three parameters course.yml pins are not re-chosen here and must not be: StandardScaler on
longitude, latitude AND depth; eps=0.15; min_samples=12. sklearn's default min_samples=5 gives
22 clusters, and unscaled coordinates give 4.

    python tools/build_week12.py
"""
import json
import math
import pathlib
import subprocess
import sys

import pandas as pd
import yaml
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/notebooks"
SLUG = "12_hidden_fault"

course = yaml.safe_load((ROOT / "course.yml").read_text())
WEEK = next(s for s in course["schedule"] if s["n"] == 12)
PLATFORM = course["platform"]
CACHE_BASE = PLATFORM["cache_base"]

# --- the pinned slice, verbatim from course.yml ------------------------------------------
BOX = ("&minlatitude=35.0&maxlatitude=36.5"
       "&minlongitude=-118.5&maxlongitude=-117.0")
MINMAG = 2
MAIN = ("2019-07-01", "2019-12-31")
# The neighbouring windows: three before the sequence and one after it. A conclusion drawn from
# one six-month window of a catalogue is worth nothing until the windows either side are counted.
WINDOWS = [("2018-01-01", "2018-07-01"), ("2018-07-01", "2019-01-01"),
           ("2019-01-01", "2019-07-01"), MAIN, ("2020-01-01", "2020-07-01")]

# --- the pinned constants ----------------------------------------------------------------
EPS, MIN_SAMPLES = 0.15, 12
FEATURES = ["longitude", "latitude", "depth"]
K = 3
K_VALUES = list(range(1, 9))
EPS_SWEEP = [0.075, 0.15, 0.30]

# Earth's mean radius, 6,371 km (IUGG mean radius; read from the NASA Earth fact sheet
# 2026-08-31, which gives a volumetric mean radius of 6,371 km). Everything else is arithmetic.
EARTH_RADIUS_KM = 6371.0
REF_LAT = 35.8                       # the middle of this box, for the east-west scale


def fetch(start, end):
    """Run one live query once, cache it beside the notebooks, and return it."""
    out = ROOT / "data" / f"week12_{start}_{end}.csv"
    if not out.exists():
        url = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&orderby=time-asc"
               f"&starttime={start}&endtime={end}&minmagnitude={MINMAG}" + BOX)
        pd.read_csv(url).to_csv(out, index=False)
    return pd.read_csv(out)


# ---------------------------------------------------------------------------
# 1. measure everything the notebook will say
# ---------------------------------------------------------------------------
M = {}

counts = {}
for start, end in WINDOWS:
    counts[start] = len(fetch(start, end))
M["n_before"] = counts["2019-01-01"]
M["n_before2"] = counts["2018-07-01"]
M["n_before3"] = counts["2018-01-01"]
M["n_after"] = counts["2020-01-01"]

quakes = fetch(*MAIN)[["time", "latitude", "longitude", "depth", "mag", "place"]]
M["n"] = len(quakes)
M["ratio_before"] = round(M["n"] / M["n_before"], 1)

big = quakes[quakes["mag"] >= 5.5].sort_values("mag", ascending=False)
main_row = big.iloc[0]
fore_row = quakes[quakes["mag"] == 6.4].iloc[0]
M["main_mag"] = float(main_row["mag"])
M["main_lat"] = round(float(main_row["latitude"]), 4)
M["main_lon"] = round(float(main_row["longitude"]), 4)
M["main_time"] = str(main_row["time"])[:16].replace("T", " ")
M["fore_mag"] = float(fore_row["mag"])
M["fore_lat"] = round(float(fore_row["latitude"]), 4)
M["fore_lon"] = round(float(fore_row["longitude"]), 4)
M["fore_time"] = str(fore_row["time"])[:16].replace("T", " ")
M["gap_hours"] = round((pd.to_datetime(main_row["time"])
                        - pd.to_datetime(fore_row["time"])).total_seconds() / 3600, 1)
M["n_big"] = len(big)

RADIUS = 0.3           # the trivial method: a circle this many degrees round the mainshock

# scaling
scaler = StandardScaler()
scaled = scaler.fit_transform(quakes[FEATURES])
# rounded to 3 places because that is how the notebook prints scaler.scale_
M["sd_lon"] = round(float(scaler.scale_[0]), 3)
M["sd_lat"] = round(float(scaler.scale_[1]), 3)
M["sd_depth"] = round(float(scaler.scale_[2]), 3)
M["sd_ratio"] = round(M["sd_depth"] / M["sd_lat"], 0)

# k-means
kmeans_labels = KMeans(n_clusters=K, random_state=0).fit_predict(scaled)
quakes["kmeans"] = kmeans_labels
km_sizes = quakes["kmeans"].value_counts()
M["km_sizes"] = [int(v) for v in km_sizes.sort_values(ascending=False)]
km_depth = quakes.groupby("kmeans")["depth"].median().round(2)
M["km_deep_median"] = float(km_depth.max())
M["km_shallow_median"] = float(km_depth.min())
M["inertias"] = [round(KMeans(n_clusters=k, random_state=0).fit(scaled).inertia_, 1)
                 for k in K_VALUES]
M["inertia_drop_2"] = round(100 * (M["inertias"][0] - M["inertias"][1]) / M["inertias"][0], 1)
M["inertia_drop_8"] = round(100 * (M["inertias"][6] - M["inertias"][7]) / M["inertias"][6], 1)

# DBSCAN
labels = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(scaled)
quakes["cluster"] = labels
M["n_clusters"] = int(len(set(labels)) - 1)
M["n_noise"] = int((labels == -1).sum())
M["frac_noise"] = round(M["n_noise"] / M["n"], 3)
sizes = quakes["cluster"].value_counts()
M["sizes"] = {int(k): int(v) for k, v in sizes.items()}
M["largest"] = int(sizes.loc[0])
M["coso_n"] = int(sizes.loc[6])
M["n_small"] = M["n_clusters"] - 2
M["smallest"] = int(sizes.min())

depths = quakes.groupby("cluster")["depth"].median()
starts = quakes.groupby("cluster")["time"].min()
M["depth_0"] = round(float(depths.loc[0]), 2)
M["depth_6"] = round(float(depths.loc[6]), 2)
M["start_6"] = str(starts.loc[6])[:16].replace("T", " ")
M["coso_after_hours"] = round((pd.to_datetime(starts.loc[6])
                               - pd.to_datetime(main_row["time"])).total_seconds() / 3600, 1)
M["n_before_main"] = int(sum(1 for c in sorted(set(labels))
                             if pd.to_datetime(starts.loc[c]) < pd.to_datetime(main_row["time"])
                             and c != -1))

# kilometres
KM_NORTH = round(2 * 3.141592653589793 * EARTH_RADIUS_KM / 360, 2)
KM_EAST = round(KM_NORTH * math.cos(math.radians(REF_LAT)), 2)
M["km_north"] = KM_NORTH
M["km_east"] = KM_EAST
quakes["east_km"] = (quakes["longitude"] - M["main_lon"]) * KM_EAST
quakes["north_km"] = (quakes["latitude"] - M["main_lat"]) * KM_NORTH

rupture = quakes[quakes["cluster"] == 0]
pca = PCA().fit(rupture[["east_km", "north_km", "depth"]])
M["evr_0"] = [float(x) for x in pca.explained_variance_ratio_.round(3)]
M["sd_0"] = [float(x) for x in (pca.explained_variance_ ** 0.5).round(2)]
M["axis1_0"] = [float(x) for x in pca.components_[0].round(2)]
M["axis2_0"] = [float(x) for x in pca.components_[1].round(2)]
M["aspect_0"] = round(M["sd_0"][0] / M["sd_0"][2], 1)

coso = quakes[quakes["cluster"] == 6]
coso_pca = PCA().fit(coso[["east_km", "north_km", "depth"]])
M["evr_6"] = [float(x) for x in coso_pca.explained_variance_ratio_.round(3)]
M["sd_6"] = [float(x) for x in (coso_pca.explained_variance_ ** 0.5).round(2)]
M["aspect_6"] = round(M["sd_6"][0] / M["sd_6"][2], 1)
M["coso_east_km"] = round(float(coso["east_km"].median()), 1)
M["coso_north_km"] = round(float(coso["north_km"].median()), 1)

# what each eps does to two named clusters, for the model answers
M["fate"] = {}
for eps in EPS_SWEEP:
    lab = pd.Series(DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit_predict(scaled))
    for c in (6, 7):
        landed = lab[quakes["cluster"].values == c].value_counts()
        biggest = landed.drop(index=-1, errors="ignore")
        M["fate"][(eps, c)] = {
            "noise": int(landed.get(-1, 0)),
            "groups": int(len(biggest)),
            "host_size": int((lab == biggest.idxmax()).sum()) if len(biggest) else 0,
        }

# how far the mainshock is from the foreshock
M["fore_to_main_km"] = round(
    ((((M["main_lon"] - M["fore_lon"]) * KM_EAST) ** 2
      + ((M["main_lat"] - M["fore_lat"]) * KM_NORTH) ** 2) ** 0.5), 1)


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

HOOK = f"""
On 4 July 2019 the ground under the Mojave Desert broke, and {M['gap_hours']:.0f} hours later it
broke again, harder: magnitude {M['fore_mag']} and then magnitude {M['main_mag']}, the largest
earthquakes in California in twenty years. Afterwards, geologists walked the desert with the state
fault map in hand and found that most of the rupture was not on it. The rock had failed along
structures nobody had drawn (Ross et al., 2019, *Science* 366, 346–351).

That is the ordinary situation, not a scandal. A fault gets onto a map when somebody finds its
scar at the surface, and most faults never reach the surface at all. What does reach us is the
earthquakes. For six months afterwards, thousands of small ones lit up whatever was still moving
down there.

Today you get {M['n']:,} of them, and nothing else: where each one was and how deep. No fault
names, no map. The question is whether a computer can pull the structures out of that cloud —
and whether you should believe it when it says it has.
"""

md(weekkit.OPENING.format(question=WEEK["question"], datahub=datahub, hook=HOOK.strip()))

md("""
## What you'll be able to do

**The science.** Take a cloud of earthquake locations with no labels on it, split it into candidate
structures, measure how long, how wide and how deep each one is in kilometres, and say — with the
evidence, and with what is still missing — which of them you would be willing to draw on a fault
map.

**The skills.** Three new pieces of scikit-learn, all with the same shape you already know from
regression and classification. `StandardScaler` puts columns measured in different units on the
same footing. `KMeans` splits data into a number of groups you choose. `DBSCAN` splits it into
groups you did not choose, and is allowed to refuse. `PCA` measures the shape of a group.

**Eight places where you write something: five in class, three at home.** Each one is headed
*Your turn*, with an empty cell under it.
""")

setup = weekkit.setup_cell(
    imports=("from sklearn.cluster import DBSCAN, KMeans\n"
             "from sklearn.decomposition import PCA\n"
             "from sklearn.preprocessing import StandardScaler\n"),
    figsize="(7, 6)",
    cache_base=CACHE_BASE,
    signature="start, end",
    docstring=("Fetch one window of the USGS catalogue round Ridgecrest; "
               "fall back to the cached copy."),
    url_expr=('f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"\n'
              f'                           f"&orderby=time-asc&starttime={{start}}&endtime={{end}}"\n'
              f'                           f"&minmagnitude={MINMAG}"\n'
              f'                           "&minlatitude=35.0&maxlatitude=36.5"\n'
              f'                           "&minlongitude=-118.5&maxlongitude=-117.0"'),
    cache_expr='f"week12_{start}_{end}.csv"',
    unpack=f'''
quakes = load("{MAIN[0]}", "{MAIN[1]}")
quakes = quakes[["time", "latitude", "longitude", "depth", "mag", "place"]]

# The coastline ships with the course, so there is no live server to try first.
coast = pd.read_csv(CACHE + "/coastlines.csv")
print("earthquakes:", quakes.shape)
'''.strip("\n"))
code(setup)

# --- section 1 -------------------------------------------------------------
md(f"""
## Six months in one box of desert

Every row is one earthquake of magnitude {MINMAG} or above, inside a box
{-118.5} to {-117.0} degrees east and {35.0} to {36.5} degrees north, between
{MAIN[0]} and {MAIN[1]}. Remember what such a file is: *A catalogue lists what somebody's
instruments recorded, not what happened. Where there are no seismometers there are no earthquakes
in the file.* Southern California is densely instrumented, so this is a good catalogue — with one
caveat worth carrying: in the minutes and hours right after a large earthquake, so many small ones
arrive at once that their records overlap and the smallest are missed. The first day of this file
is thinner than the ground actually was.

Start with the biggest ones.
""")

code(f"""
print(quakes.head())
print(quakes[quakes["mag"] >= 5.5][["time", "latitude", "longitude", "depth", "mag"]])
""")

md(f"""
The catalogue holds {M['n_big']} earthquakes at magnitude 5.5 or above, and the two that matter
are the largest: **magnitude {M['fore_mag']} on {M['fore_time']} UTC**, then **magnitude {M['main_mag']} on
{M['main_time']} UTC**, {M['gap_hours']:.0f} hours later and {M['fore_to_main_km']} km to the
northwest.

Before anything else, one sanity check. {M['n']:,} earthquakes in six months sounds like a lot,
but that means nothing until you know what six months in this box normally holds.

### Predict before you run

How many magnitude {MINMAG}+ earthquakes do you think this same box recorded in the **six months
before** the sequence — 1 January to 1 July 2019? Change `my_guess` to your number, then run the
cell under it.
""")

code(f"""
my_guess = 2000

windows = [("2018-01-01", "2018-07-01"), ("2018-07-01", "2019-01-01"),
           ("2019-01-01", "2019-07-01"), ("2019-07-01", "2019-12-31"),
           ("2020-01-01", "2020-07-01")]

for start, end in windows:
    print(start, "to", end, ":", len(load(start, end)), "earthquakes")

print("you guessed:", my_guess, "for 2019-01-01 to 2019-07-01")
""")

md(f"""
The answer is {M['n_before']}, and the two six-month windows before that hold {M['n_before3']}
and {M['n_before2']}. So this box normally produces something like forty to eighty small earthquakes
in six months, and the window we are about to work in produced {M['n']:,} — about
{M['ratio_before']:.0f} times as many. The six months *after* still hold {M['n_after']},
several times the background and falling.

That is what tells us these {M['n']:,} events are one connected episode rather than the ordinary
grumbling of the desert. Everything below is about that episode, and would not be true of a
quiet window.

Now look at where it happened. The first map is wide, so that you can see where in California
this box sits; the grey line is the coastline, from the same file every map in this course uses.
""")

code(f"""
plt.plot(coast.lon, coast.lat, color="0.6", lw=0.6)
plt.scatter(quakes["longitude"], quakes["latitude"], s=2, color="firebrick")
plt.xlim(-122, -114)
plt.ylim(32, 38)
plt.gca().set_aspect("equal")
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title("{M['n']:,} earthquakes, southern California")
plt.show()
""")

md(f"""
That is the Mojave Desert, well inland of the coast, in the Eastern California Shear Zone — a
belt of desert faults that carries part of the Pacific–North America plate motion inland of the
San Andreas. The coast never enters the working box, so from here on the maps have no coastline
to draw.

Zoom in, and mark the two large earthquakes with stars.
""")

code(f"""
plt.scatter(quakes["longitude"], quakes["latitude"], s=2, color="0.3")
plt.scatter([{M['fore_lon']}, {M['main_lon']}], [{M['fore_lat']}, {M['main_lat']}],
            s=120, marker="*", color="firebrick")
plt.locator_params(axis="x", nbins=6)      # or the degree labels run into each other
plt.gca().set_aspect("equal")
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title("{M['n']:,} earthquakes, magnitude {MINMAG}+, six months")
plt.show()
""")

md(f"""
The cloud is not shapeless. A long limb runs northwest–southeast through the two stars, a
shorter one crosses it at a steep angle near the southern star, there is a separate patch up in
the northwest corner, and a scatter of dots that belong to neither. Your eye has already done
some clustering. The rest of the notebook is about making that judgement explicit enough to
argue with.

The first thing anyone tries is a circle. The magnitude {M['main_mag']} epicentre is at
{M['main_lat']} degrees north, {M['main_lon']} degrees east, from the table you printed
above — so measure how far each earthquake is from it and keep the near ones.
""")

ask(f"""
### ✏️ Your turn 1

Make three new columns and one count.

Subtract the mainshock's longitude, {M['main_lon']}, from `quakes["longitude"]` and call the
result `east`. Subtract its latitude, {M['main_lat']}, from `quakes["latitude"]` and call that
`north`. Then `distance` is `(east ** 2 + north ** 2) ** 0.5` — Pythagoras, in degrees.
Then filter `quakes` to the rows where `distance` is less than {RADIUS}, call the result
`near`, and print how many there are out of {M['n']:,}.

**Use these names**, because the self-check looks for them: `east`, `north`, `distance`, `near`.
""")

answer(f"""
east = quakes["longitude"] - ({M['main_lon']})
north = quakes["latitude"] - ({M['main_lat']})
distance = (east ** 2 + north ** 2) ** 0.5

near = quakes[distance < {RADIUS}]

print(len(near), "of", len(quakes), "earthquakes are within {RADIUS} degrees")
""", f"""
assert distance.min() < 0.01, \\
    "the mainshock's own row should be almost zero away — check the order of the subtraction"
print("✓ a circle round the mainshock —", len(near), "of", len(quakes),
      "events,", round(len(near) / len(quakes), 3), "of the catalogue")
""")

md(f"""
That is not an answer to anything, for three reasons, and they are worth being precise about.

You had to know where the mainshock was before you could draw the circle, so the method cannot
find a structure nobody has told you about. The circle is round and the cloud is not, so it takes
in empty desert on one side and cuts the long limb off on the other. And it returns **one** group:
the patch in the northwest corner and the crossing limb are either inside the circle or outside
it, and either way they are not distinguished from anything else.

What we want is a method that is handed the coordinates and nothing else.

## Putting pins on the map

Here is the oldest idea in clustering. *Put k pins on the map, give each point to its nearest pin,
move the pins, repeat.* The pins settle where the data is densest, and each point ends up with
whichever pin it is closest to. That is **k-means**, and `k` is how many pins — you choose it.

One thing has to happen first. "Nearest" means measuring a distance, and our three columns are
not in the same units: longitude and latitude are degrees, depth is kilometres. Their spreads in
this catalogue are about {M['sd_lon']} degrees, {M['sd_lat']} degrees and {M['sd_depth']}
kilometres, which you are about to print. Left alone, depth would count for roughly
{M['sd_ratio']:.0f} times more than latitude simply because its numbers are bigger. `StandardScaler` fixes that the way you scaled features before modelling
earlier in the course: subtract each column's mean, divide by its standard deviation, so every
column has a spread of 1 and none of them shouts.
""")

code(f"""
scaler = StandardScaler()
scaled = scaler.fit_transform(quakes[{FEATURES}])

print("spread of each column before scaling:", scaler.scale_.round(3))
print("shape of the scaled array:", scaled.shape)
""")

ask(f"""
### ✏️ Your turn 2

Put {K} pins on the map and see where they settle.

Make `model = KMeans(n_clusters={K}, random_state=0)` and then
`quakes["kmeans"] = model.fit_predict(scaled)`. `fit_predict` is the same verb you have used all
along, and it hands back one group number per earthquake. Print
`quakes["kmeans"].value_counts()`.

`random_state=0` matters here: k-means starts by dropping its pins at random, so without a fixed
seed the group sizes move by a few tens every time you run it.

**Use these names**, because the self-check looks for them: `model`.
""")

answer(f"""
model = KMeans(n_clusters={K}, random_state=0)
quakes["kmeans"] = model.fit_predict(scaled)

print(quakes["kmeans"].value_counts())
""", f"""
assert len(quakes["kmeans"].value_counts()) == {K}, "there should be exactly {K} groups"
print("✓ k-means with {K} pins — groups of",
      list(quakes["kmeans"].value_counts()))
""")

md(f"""
{M['km_sizes'][0]:,}, {M['km_sizes'][1]:,} and {M['km_sizes'][2]:,}: three groups of almost
exactly the same size. Draw them.
""")

code(f"""
for group in [0, 1, 2]:
    part = quakes[quakes["kmeans"] == group]
    plt.scatter(part["longitude"], part["latitude"], s=2, label=f"group {{group}}")

plt.legend()
plt.locator_params(axis="x", nbins=6)
plt.gca().set_aspect("equal")
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title("k-means, k={K}, {M['n']:,} earthquakes")
plt.show()
""")

md(f"""
Two things went wrong, and neither is a bug.

The long limb has been **cut across the middle**. k-means groups by distance to a pin, so its
groups come out as roughly round blobs; a structure many times longer than it is wide is not a
blob, and no arrangement of {K} pins can make it one. Look at the depths as well —
""")

code(f"""
print(quakes.groupby("kmeans")["depth"].median())
""")

md(f"""
— one group has a median depth of {M['km_deep_median']} km and another {M['km_shallow_median']}
km. The pins have partly split the data into a deep half and a shallow half, which is a real
feature of the data and is not what we asked for.

Second, and worse: **every earthquake got a group.** The isolated dots out in the corners of the
map, tens of kilometres from anything, are all coloured, because k-means has to give every point
to its nearest pin. It has no way to say that a point is not part of anything.

Before fixing that, deal with the obvious complaint: we picked {K} out of the air. The usual way
to choose is **inertia** — the total squared distance from each point to its own pin. It always
falls as you add pins (with one pin per point it would be zero), so you look for the `k` where it
stops falling *fast*.
""")

ask(f"""
### ✏️ Your turn 3

Build the curve.

Make an empty list `inertias`. Loop over `k_values = {K_VALUES}`; inside the loop fit
`KMeans(n_clusters=k, random_state=0)` on `scaled` with `.fit(scaled)`, and append that model's
`.inertia_` to your list. Then plot `k_values` against `inertias` with
`plt.plot(k_values, inertias, marker="o")`, label both axes, and give it a title.

**Use these names**, because the self-check looks for them: `k_values`, `inertias`.
""")

answer(f"""
k_values = {K_VALUES}
inertias = []

for k in k_values:
    fit = KMeans(n_clusters=k, random_state=0).fit(scaled)
    inertias.append(fit.inertia_)

plt.plot(k_values, inertias, marker="o")
plt.xlabel("number of pins k")
plt.ylabel("inertia (total squared distance to the nearest pin)")
plt.title("k-means inertia, {M['n']:,} earthquakes")
plt.show()
""", """
assert len(inertias) == len(k_values), \\
    "one inertia per k — was the append inside the loop?"
first_drop = 100 * (inertias[0] - inertias[1]) / inertias[0]
last_drop = 100 * (inertias[-2] - inertias[-1]) / inertias[-2]
print("✓ the inertia curve — the first extra pin removes", round(first_drop, 1),
      "% of the inertia and the last one still removes", round(last_drop, 1), "%")
""")

md(f"""
The curve bends, but it does not have a corner. The first extra pin takes
{M['inertia_drop_2']}% off the inertia and the eighth still takes {M['inertia_drop_8']}% off what
is left, as the self-check line says. There is no `k` at which the curve says *stop here*, because the data
is not made of `k` round blobs, so no value of `k` is right. That is the honest reading of a
smooth elbow: it is telling you that the question "how many blobs?" does not fit this data.

## Groups by crowding, instead of by pins

*The same idea, but it is allowed to say: this one belongs to nothing.* That is **DBSCAN**, and
it works from crowding rather than from pins. It has two settings and no `k`:

- **`eps`** — how close two points have to be to count as neighbours. We are working on the
  scaled array, so `eps` is measured in standard deviations, not degrees.
- **`min_samples`** — how many neighbours within `eps` a point needs before it counts as being in
  a crowd.

A point with enough neighbours seeds a cluster, and the cluster grows through its neighbours'
neighbours for as far as the crowd goes — which is why a DBSCAN cluster can be any shape at all,
including long and thin. A point with too few neighbours joins nothing and is labelled
**-1**, the noise label. That label is the whole difference from k-means.

We use `eps={EPS}` and `min_samples={MIN_SAMPLES}` on the scaled longitude, latitude **and**
depth. Those three choices are not defaults and they are not innocent — scikit-learn's own
`min_samples` is 5, and each of the three moves the number of clusters you get. The homework is
where you push on `eps`.

### Predict before you run

Of the {M['n']:,} earthquakes, how many do you think DBSCAN will refuse to put in any cluster at
all? Change `my_noise_guess` to your number before you run the cell after it.
""")

code(weekkit.CHECKPOINT.format(body=f'''quakes = load("{MAIN[0]}", "{MAIN[1]}")
quakes = quakes[["time", "latitude", "longitude", "depth", "mag", "place"]]
scaled = StandardScaler().fit_transform(quakes[{FEATURES}])'''))

code(f"""
my_noise_guess = 50

quakes["cluster"] = DBSCAN(eps={EPS}, min_samples={MIN_SAMPLES}).fit_predict(scaled)

print("clusters found:", len(set(quakes["cluster"])) - 1)
print("events in no cluster:", (quakes["cluster"] == -1).sum(), " you guessed:", my_noise_guess)
""")

md(f"""
It found {M['n_clusters']} clusters and put **{M['n_noise']:,} earthquakes —
{M['frac_noise'] * 100:.1f}% of the catalogue — in none of them.** That is not a failure. Roughly one event in five is not in
a crowd by this definition of crowd, and DBSCAN has said so instead of quietly attaching them to
whatever was nearest. Draw it with the noise in grey.
""")

code(f"""
unplaced = quakes[quakes["cluster"] < 0]
placed = quakes[quakes["cluster"] >= 0]

plt.scatter(unplaced["longitude"], unplaced["latitude"], s=2, color="0.8")
plt.scatter(placed["longitude"], placed["latitude"], s=2, c=placed["cluster"], cmap="tab20")
plt.colorbar(label="cluster number")
plt.locator_params(axis="x", nbins=6)
plt.gca().set_aspect("equal")
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title("DBSCAN on {M['n']:,} events: {M['n_clusters']} clusters, {M['n_noise']:,} in grey")
plt.show()
""")

md(f"""
The long limb is one cluster now, end to end, and it did not need to be round to survive. The
patch in the northwest is its own cluster. The grey points are the ones the algorithm declined to
place, and they are exactly where you would decline too — the outlying dots, and the thin fringes
round the edges of the dense limb.

The cluster numbers on that colour bar are just the order the clusters were found in; they mean
nothing on their own. To make them mean something, put the sizes, the depths and the times side
by side.
""")

ask(f"""
### ✏️ Your turn 4

Build the table.

Make three Series with the tools you already have: `sizes = quakes["cluster"].value_counts()`,
`depths = quakes.groupby("cluster")["depth"].median()`, and `starts` the same way with
`["time"].min()`. Then loop over `sizes.index` and print one line per cluster, giving its number,
its size, its median depth rounded to 2 decimal places, and the first 16 characters of its
earliest time. Use `.loc[c]` to read one value out of each Series.

`value_counts()` already sorts, so the biggest cluster comes first and -1 will appear wherever its
size puts it. Finally, print the three most common `place` values in the **second-largest
cluster** — second-largest among the clusters, so not the -1 row — with
`value_counts().head(3)` — the catalogue names where each earthquake was, and we
have not looked at that column yet.

**Use these names**, because the self-check looks for them: `sizes`, `depths`, `starts`.
""")

answer("""
sizes = quakes["cluster"].value_counts()
depths = quakes.groupby("cluster")["depth"].median()
starts = quakes.groupby("cluster")["time"].min()

for c in sizes.index:
    print("cluster", c, "|", sizes.loc[c], "events | median depth",
          round(depths.loc[c], 2), "km | first", starts.loc[c][:16])

print(quakes[quakes["cluster"] == 6]["place"].value_counts().head(3))
""", f"""
assert len(sizes) == {M['n_clusters']} + 1, \\
    "one row per cluster plus one for the -1 noise group"
print("✓ the clusters — largest holds", sizes.max(), "events, smallest holds",
      sizes.min(), ", and", sizes.loc[-1], "events belong to nothing")
""")

md(f"""
Two rows stand out. Cluster 0 holds {M['largest']:,} events — most of the catalogue — with a
median depth of {M['depth_0']} km, and it starts with the very first event in the file. Cluster 6
holds {M['coso_n']}, sits at a median depth of {M['depth_6']} km, and its first event is
{M['start_6']} UTC. The other {M['n_small']} clusters hold between {M['smallest']} and 39 events
each.

Now the point of the week. **DBSCAN was given longitude, latitude and depth. It was never given
the time, the magnitude or the place name.** So those three columns are free evidence — we can
ask them whether the groups it drew are real. Cluster 6's place names say **Coso Junction**.

That patch in the northwest corner of your map sits at and just north of the
**Coso Geothermal Field**, a young volcanic area with a geothermal power station on it. Coso runs
a background of very shallow small earthquakes of its own, and it is known to be set off by large
regional earthquakes (Kaven, 2020, *Bulletin of the Seismological Society of America* 110,
1728–1735).

And the timing agrees: cluster 6's first event is {M['coso_after_hours']:.1f} hours **after** the
mainshock, while {M['n_before_main']} of the {M['n_clusters']} clusters were already running
before it. A clustering that knew nothing about time has separated a group that turns out to have
switched on afterwards, well away from the rupture, under a geothermal field. That is what
"finds structure with the labels hidden" buys you: the labels you held back become a test.

Be careful about which evidence counts. The depth is *not* independent — depth was one of the
three columns DBSCAN clustered on, so of course the groups differ in depth. The time, the
magnitudes and the place names are independent, because the algorithm never saw them.
""")

# --- section 4 -------------------------------------------------------------
md(f"""
## The shape of a cluster

A cluster is not yet a fault. A fault is a surface in the rock, so a cluster that is a fault
should be *long in one direction, less so in a second, and thin in the third*. The map cannot tell
you that, because a map has flattened the depth away.

Measuring it is what **PCA** is for. *If two measurements say nearly the same thing, replace them
with one.* For a cloud of earthquakes strung out along a line, east and north say nearly the same
thing — tell me how far along the line a point is and you have told me both. PCA finds the
direction the cloud is most stretched along, calls it axis 1, then the most stretched direction
left over, and so on, and reports how much of the spread sits on each.

For that to mean anything the three columns have to be in the same real units, so we swap degrees
for kilometres. One degree of latitude is {M['km_north']} km everywhere (that is Earth's
circumference divided by 360, using the mean radius of {EARTH_RADIUS_KM:,.0f} km on the NASA
Earth fact sheet, read 2026-08-31). One degree of
longitude is shorter, and at {REF_LAT} degrees north it is {M['km_east']} km. Depth is already in
kilometres. We measure from the mainshock, so the numbers read as "kilometres east and north of
the magnitude {M['main_mag']}".
""")

code(weekkit.CHECKPOINT.format(body=f'''quakes = load("{MAIN[0]}", "{MAIN[1]}")
quakes = quakes[["time", "latitude", "longitude", "depth", "mag", "place"]]
scaled = StandardScaler().fit_transform(quakes[{FEATURES}])
quakes["cluster"] = DBSCAN(eps={EPS}, min_samples={MIN_SAMPLES}).fit_predict(scaled)'''))

code(f"""
KM_PER_DEGREE_NORTH = {M['km_north']}   # Earth's circumference over 360
KM_PER_DEGREE_EAST = {M['km_east']}    # the same, shrunk to {REF_LAT} degrees north

quakes["east_km"] = (quakes["longitude"] - ({M['main_lon']})) * KM_PER_DEGREE_EAST
quakes["north_km"] = (quakes["latitude"] - ({M['main_lat']})) * KM_PER_DEGREE_NORTH

rupture = quakes[quakes["cluster"] == 0]
print(len(rupture), "events in cluster 0, largest magnitude", rupture["mag"].max())
""")

code(f"""
pca = PCA()
pca.fit(rupture[["east_km", "north_km", "depth"]])

print("share of the spread on each axis:", pca.explained_variance_ratio_.round(3))
print("size along each axis (km):       ", (pca.explained_variance_ ** 0.5).round(2))
print("axis 1, as (east, north, down):  ", pca.components_[0].round(2))
print("axis 2, as (east, north, down):  ", pca.components_[1].round(2))
""")

md(f"""
Read those four lines carefully, because they are the answer to the week's question.

**{M['evr_0'][0] * 100:.1f}% of the spread lies on axis 1 alone.** Three numbers per earthquake,
and one of them carries almost everything — which is exactly the situation PCA exists to find.
In kilometres the cluster measures {M['sd_0'][0]} by {M['sd_0'][1]} by {M['sd_0'][2]} (those are
standard deviations, so the full extent is several times each). Long, much less wide, thinner
still.

Axis 1 came out as {M['axis1_0']} in (east, north, down): {M['axis1_0'][1]} north for every
{abs(M['axis1_0'][0])} west, and only {abs(M['axis1_0'][2])} up or down. So it is a **horizontal
line running northwest–southeast**. Axis 2 is {M['axis2_0']} — almost straight down. A surface
whose long direction is horizontal and whose second direction is vertical is a **near-vertical
plane striking northwest–southeast**, which is what a strike-slip fault in this desert looks
like, and it matches the direction of the limb you saw by eye on the map.

Turn the cloud so you are looking along that plane edge-on. `pca.transform` gives every event its
position on the three new axes, as a grid with one row per event; `along[:, 0]` is the first
column of that grid — position on axis 1 — and `along[:, 2]` is the third.
""")

code(f"""
along = pca.transform(rupture[["east_km", "north_km", "depth"]])

plt.scatter(along[:, 0], along[:, 2], s=2, color="0.3")
plt.gca().set_aspect("equal")
plt.xlabel("position along axis 1 (km)")
plt.ylabel("position along axis 3 (km)")
plt.title("cluster 0 seen edge-on, {M['largest']:,} events")
plt.show()
""")

md(f"""
A sheet, seen from the side. The numbers you printed say the same thing: {M['sd_0'][0]} km of
spread along axis 1 against {M['sd_0'][2]} km across the plane, so it is about
{M['aspect_0']} times longer than it is thick. Some of that thickness is real — faults are zones,
not razor cuts — and some of it is only how accurately these events could be located.
""")

ask(f"""
### ✏️ Your turn 5

Do the same for cluster 6, the {M['coso_n']}-event group under Coso, and see whether it has the
same shape.

Filter `quakes` to `cluster == 6` and call it `coso`. Make `coso_pca = PCA()`, fit it on
`coso[["east_km", "north_km", "depth"]]`, and print the same two lines as above: the share of the
spread on each axis, and the size along each axis in kilometres. Then print the median of
`coso["depth"]`, and the medians of `coso["east_km"]` and `coso["north_km"]` — which say where
this cluster sits relative to the mainshock.

**Use these names**, because the self-check looks for them: `coso`, `coso_pca`.
""")

answer("""
coso = quakes[quakes["cluster"] == 6]

coso_pca = PCA()
coso_pca.fit(coso[["east_km", "north_km", "depth"]])

print("share of the spread on each axis:", coso_pca.explained_variance_ratio_.round(3))
print("size along each axis (km):       ", (coso_pca.explained_variance_ ** 0.5).round(2))
print("median depth:", round(coso["depth"].median(), 2), "km")
print("median position:", round(coso["east_km"].median(), 1), "km east,",
      round(coso["north_km"].median(), 1), "km north of the mainshock")
""", f"""
assert len(coso) > 500, \\
    "cluster 6 should be the {M['coso_n']}-event group from your table — check the cluster number"
print("✓ the shape of cluster 6 —", len(coso), "events,",
      coso_pca.explained_variance_ratio_[0].round(3), "of the spread on axis 1, against",
      pca.explained_variance_ratio_[0].round(3), "for cluster 0")
""")

md(f"""
{M['evr_6'][0] * 100:.1f}% on axis 1 rather than {M['evr_0'][0] * 100:.1f}%, and
{M['sd_6'][0]} by {M['sd_6'][1]} by {M['sd_6'][2]} km rather than {M['sd_0'][0]} by
{M['sd_0'][1]} by {M['sd_0'][2]}. Cluster 6 is stretched, but only {M['aspect_6']} times longer
than it is thick where cluster 0 is {M['aspect_0']}, and it is shallow: a median of
{M['depth_6']} km against {M['depth_0']} km for cluster 0, both from the table you built.

Those are two different kinds of object. One is a long thin near-vertical sheet that started
moving with the first event in the file. The other is a shallow, blobbier patch
{abs(M['coso_east_km']):.0f} km west and {M['coso_north_km']:.0f} km north of the mainshock, which
switched on {M['coso_after_hours']:.1f} hours after it, under a geothermal field. Neither measurement *proves* anything on its own — but you
would defend the first as a fault plane and you would not describe the second that way, and now
you can say why in numbers.
""")

# --- closing ---------------------------------------------------------------
md(f"""
## The question, answered

**Yes — and cluster 0 is one.** Handed nothing but position and depth, DBSCAN pulled a
{M['sd_0'][0]:.0f}-by-{M['sd_0'][2]:.1f} km near-vertical sheet out of {M['n']:,} dots, running
northwest–southeast; most of the fault it traces was not on the California fault map before July
2019. It also separated a shallow swarm under Coso that the mainshock switched on, found
{M['n_small']} smaller groups, and declined to place {M['n_noise']:,} events at all — which is the
useful part, because it is the algorithm telling you where it has no opinion. What it cannot tell
you is which of those {M['n_clusters']} groups a geologist would accept, and the three settings
you fed it — the scaler, `eps` and `min_samples` — decided how many there were. So the answer
always travels with its parameters.
""")

md(weekkit.week_cheatsheet(12))

md(f"""
## Homework

Three parts, all on `quakes` and `scaled`, which you already have. If you have restarted since
class, run the setup cell at the top and then the two checkpoint cells, and you will be back where
you were.

Class ran DBSCAN once, at `eps={EPS}`, and told you the number was not innocent. Now find out how
much of the answer it was carrying.
""")

ask(f"""
### ✏️ Your turn 6

Sweep it.

Make `eps_values = {EPS_SWEEP}` and an empty list `cluster_counts`. Loop over `eps_values`, and
inside the loop run `DBSCAN(eps=eps, min_samples={MIN_SAMPLES}).fit_predict(scaled)`, store the
labels, and print three things: how many clusters it found (`len(set(labels)) - 1`, since -1 is
not a cluster), how many events it left unassigned (`(labels == -1).sum()`), and that count as a
percentage of {M['n']:,}. Append the cluster count to `cluster_counts` as you go.

**Use these names**, because the self-check looks for them: `eps_values`, `cluster_counts`.
""")

answer(f"""
eps_values = {EPS_SWEEP}
cluster_counts = []

for eps in eps_values:
    labels = DBSCAN(eps=eps, min_samples={MIN_SAMPLES}).fit_predict(scaled)
    unassigned = (labels == -1).sum()
    cluster_counts.append(len(set(labels)) - 1)
    print("eps", eps, "->", len(set(labels)) - 1, "clusters,", unassigned, "unassigned (",
          round(100 * unassigned / len(labels), 1), "% )")
""", """
assert len(cluster_counts) == len(eps_values), \\
    "one cluster count per eps — was the append inside the loop?"
print("✓ the eps sweep — cluster counts", cluster_counts, "for eps", eps_values)
""")

ask(f"""
### ✏️ Your turn 7

Now make the call, and show what it costs.

Set `my_eps` to the value from your sweep that you would be willing to defend in a paper, run
DBSCAN once more with it, and put the labels in a new column `quakes["my_cluster"]`. Then print
`quakes["my_cluster"].value_counts().head(6)`, and — this is the part that matters — print
`coso_now = quakes.loc[quakes["cluster"] == 6, "my_cluster"]`'s `value_counts()`, which says what
your choice did to the {M['coso_n']} Coso events class found.

There is no right answer here and the self-check will not judge you; two of the three values are
defensible and they give different pictures.

**Use these names**, because the self-check looks for them: `my_eps`, `coso_now`.
""")

answer(f"""
my_eps = 0.30

quakes["my_cluster"] = DBSCAN(eps=my_eps, min_samples={MIN_SAMPLES}).fit_predict(scaled)

print(quakes["my_cluster"].value_counts().head(6))

coso_now = quakes.loc[quakes["cluster"] == 6, "my_cluster"]
print(coso_now.value_counts())
""", f"""
assert len(coso_now) == {M['coso_n']}, \\
    "coso_now should be the {M['coso_n']} events class put in cluster 6"
print("✓ your choice — eps", my_eps, "gives",
      len(set(quakes["my_cluster"])) - 1, "clusters and leaves",
      (quakes["my_cluster"] == -1).sum(), "events unassigned; the {M['coso_n']} Coso events",
      "landed in", len(coso_now.value_counts()), "group(s)")
""")

ask(f"""
### ✏️ Your turn 8

Name two clusters from the class run: one you would be willing to draw on a fault map, and one you
think the algorithm invented. For each, quote two numbers from your own output as your reason —
size along the axes, median depth, first event, size, distance, whichever you actually used. Then
name one measurement that is **not** in this notebook and that would settle which of you is right.

Four or five sentences.
""")

answer_prose(f"""
I would draw cluster 0. Its {M['largest']:,} events measure {M['sd_0'][0]} by {M['sd_0'][1]} by
{M['sd_0'][2]} km, with {M['evr_0'][0] * 100:.1f}% of the spread on a single horizontal axis
running northwest–southeast, which is the shape a fault plane has and not a shape a blob has; it
holds the magnitude {M['main_mag']} itself and starts with the very first event in the
catalogue, so it is the thing that actually broke. I think the algorithm invented cluster 7: it holds
{M['sizes'][7]} events, exactly `min_samples`, and it is its own cluster at no other setting I
tried — at `eps={EPS_SWEEP[0]}` all {M['fate'][(EPS_SWEEP[0], 7)]['noise']} of them become noise,
and at `eps={EPS_SWEEP[2]}` they are absorbed into a group of
{M['fate'][(EPS_SWEEP[2], 7)]['host_size']}. Its existence is a property of the settings rather
than of the desert. Cluster
6 is a harder case — it is a real place, since {M['coso_n']} events at a median depth of
{M['depth_6']} km under Coso are not an accident, but "real" there means a shallow geothermal
swarm, not a fault I would draw. What would settle it is data this notebook does not have: the
**focal mechanisms** of the events in each cluster, which give the orientation of the plane that
slipped in each earthquake. If a cluster's earthquakes all slipped on planes with the same
orientation, and that orientation matches the plane PCA fitted through their locations, it is one
structure; if they point every which way, the cluster is a crowd of unrelated events that happened
to be close together.
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
    for start, end in WINDOWS:
        print(f"cache: data/week12_{start}_{end}.csv")


if __name__ == "__main__":
    main()
    weekkit.gate(12)
