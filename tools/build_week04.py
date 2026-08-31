#!/usr/bin/env python
"""Build week 4 — "Where do earthquakes and volcanoes happen — and why there?"

Emits both notebooks from one source so they cannot drift:

    docs/notebooks/04_where_and_why_solution.ipynb   executed, every output saved
    docs/notebooks/04_where_and_why.ipynb            the same file with the answers deleted

It also writes the week's three cached fallbacks (one USGS query, two Smithsonian GVP layers)
and, once, data/plate_boundaries.csv — a repo asset like data/coastlines.csv, built from the
archived shapefiles by tools/make_plate_boundaries.py.

Every number that appears in prose or in a model answer is computed HERE, from the same files
the notebook reads, and formatted in. Nothing is typed from memory or copied from the plan.

    python tools/build_week04.py
"""
import json
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/notebooks"
SLUG = "04_where_and_why"

course = yaml.safe_load((ROOT / "course.yml").read_text())
WEEK = next(s for s in course["schedule"] if s["n"] == 4)
PLATFORM = course["platform"]
CACHE_BASE = PLATFORM["cache_base"]

# The three live queries, pinned here so the cached CSVs, the notebook and the prose cannot
# drift apart. The USGS window is the same one week 3 used, so the catalogue is familiar.
START, END, MINMAG = "2000-01-01", "2026-01-01", 5.5
QUAKE_CACHE = f"week04_{START}_{END}_M{MINMAG}.csv"
USGS = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&orderby=time-asc"
        f"&starttime={START}&endtime={END}&minmagnitude={MINMAG}")
GVP = ("https://webservices.volcano.si.edu/geoserver/GVP-VOTW/ows?service=WFS&version=1.0.0"
       "&request=GetFeature&typeName=GVP-VOTW:Smithsonian_VOTW_Holocene_")
VOLCANO_CACHE, ERUPTION_CACHE = "week04_gvp_volcanoes.csv", "week04_gvp_eruptions.csv"

SHAPEFILES = ROOT.parent / "offerings/2026-spring_gleeson/data"

# The class box (whole South American arc) and the two the homework offers.
SOUTH = (-45, 5, -85, -60)
CHILE, JAPAN = (-32, -14, -80, -58), (30, 45, 128, 148)

MAG_EDGES = np.arange(5.5, 9.6, 0.5)
EARTH_RADIUS_KM = 6371          # IUGG mean radius; see the citation in the notebook

# The confining pressure 600 km down — the number that rules out ordinary brittle fracture at the
# depths of the deep earthquakes, cold slab or not. Integrated from the PREM density profile
# (ds.iris.edu/files/products/emc/data/PREM/PREM_1s.csv, read 2026-08-31): the same integration
# returns 364 GPa at the centre of the Earth against PREM's published 363.85, so the profile is
# being read correctly. 13.8 GPa at 410 km and 23.5 GPa at 660 km come out of the same run.
PRESSURE_600KM_GPA = 21
STANDARD_ATMOSPHERE_PA = 101_325   # the SI definition of one atmosphere


# ---------------------------------------------------------------------------
# 1. fetch once, cache, and measure everything the notebook will say
# ---------------------------------------------------------------------------
def cached(url, name):
    """Run one live query the first time, keep the CSV in data/, and read it back."""
    out = ROOT / "data" / name
    if not out.exists():
        pd.read_csv(url).to_csv(out, index=False)
    return pd.read_csv(out)


def make_boundaries():
    """Build data/plate_boundaries.csv from the archived shapefiles, once."""
    out = ROOT / "data/plate_boundaries.csv"
    if not out.exists():
        subprocess.run([sys.executable, str(ROOT / "tools/make_plate_boundaries.py"),
                        str(SHAPEFILES), str(out)], check=True)
    return pd.read_csv(out)


def arc_box(table, box):
    """The rows of a catalogue inside one latitude/longitude box."""
    south, north, west, east = box
    return table[(table["latitude"] >= south) & (table["latitude"] <= north)
                 & (table["longitude"] >= west) & (table["longitude"] <= east)]


quakes = cached(USGS, QUAKE_CACHE)
quakes = quakes[quakes["type"] == "earthquake"]
volcanoes = cached(GVP + "Volcanoes&outputFormat=csv", VOLCANO_CACHE)
eruptions = cached(GVP + "Eruptions&outputFormat=csv", ERUPTION_CACHE)
boundaries = make_boundaries()

M = {}
M["n_quakes"] = len(quakes)
M["n_volcanoes"] = len(volcanoes)
M["n_eruptions"] = len(eruptions)
M["n_boundary_points"] = int(boundaries["lon"].notna().sum())
M["n_segments"] = int(boundaries["segment"].nunique())
for kind in ("ridge", "transform", "trench"):
    M[f"n_{kind}"] = int(boundaries[boundaries["kind"] == kind]["segment"].nunique())

M["n_shallow"] = int((quakes["depth"] <= 70).sum())
M["frac_shallow"] = round(float((quakes["depth"] <= 70).mean()), 3)
M["n_middle"] = int(((quakes["depth"] > 70) & (quakes["depth"] <= 300)).sum())
M["n_deep"] = int((quakes["depth"] > 300).sum())
M["deepest"] = float(quakes["depth"].max())

setting = volcanoes["Tectonic_Setting"]
M["n_subduction"] = int(setting.str.startswith("Subduction", na=False).sum())
M["n_rift"] = int(setting.str.startswith("Rift", na=False).sum())
M["n_intraplate"] = int(setting.str.startswith("Intraplate", na=False).sum())
M["n_rift_ocean"] = int(setting.str.startswith("Rift zone / Oceanic", na=False).sum())
M["pct_subduction"] = round(100 * M["n_subduction"] / M["n_volcanoes"])

# the class flagship: how far behind the trench the deep events sit
south_arc = arc_box(quakes, SOUTH)
M["n_south"] = len(south_arc)
M["south_shallow_lon"] = float(south_arc[south_arc["depth"] <= 70]["longitude"].median())
south_deep = south_arc[south_arc["depth"] > 300]
M["n_south_deep"] = len(south_deep)
M["south_deep_lon"] = float(south_deep["longitude"].median())
M["south_deep_lat"] = float(south_deep["latitude"].median())
M["south_gap_deg"] = round(M["south_deep_lon"] - M["south_shallow_lon"], 1)
M["south_deepest"] = round(float(south_arc["depth"].max()))
M["gpa_600km"] = PRESSURE_600KM_GPA
M["atm_600km"] = int(round(PRESSURE_600KM_GPA * 1e9 / STANDARD_ATMOSPHERE_PA, -4))
M["south_gap_km"] = round(abs(M["south_deep_lon"] - M["south_shallow_lon"])
                          * 2 * np.pi * EARTH_RADIUS_KM / 360
                          * float(np.cos(np.deg2rad(M["south_deep_lat"]))))

# the eruption record
vei = eruptions["ExplosivityIndexMax"].value_counts().sort_index()
M["n_with_vei"] = int(vei.sum())
for k in range(8):
    M[f"vei{k}"] = int(vei.loc[k])
M["n_no_vei"] = M["n_eruptions"] - M["n_with_vei"]
M["pct_no_vei"] = round(100 * M["n_no_vei"] / M["n_eruptions"])
M["vei_windows"] = []
for first_year in [-60000, 1800, 1950]:
    window = eruptions[eruptions["StartDateYear"] >= first_year]
    counts = window["ExplosivityIndexMax"].value_counts()
    M["vei_windows"].append((first_year, len(window),
                             round(float(counts.loc[1] / counts.loc[2]), 2),
                             round(100 * (len(window) - int(counts.sum())) / len(window))))
M["pct_no_vei_1800"] = M["vei_windows"][1][3]
M["pct_no_vei_1950"] = M["vei_windows"][2][3]
recent = eruptions[eruptions["StartDateYear"] >= 1950]
recent_vei = recent["ExplosivityIndexMax"].value_counts().sort_index()
M["n_recent"] = len(recent)
M["n_recent_with_vei"] = int(recent_vei.sum())
M["n_recent_no_vei"] = M["n_recent"] - M["n_recent_with_vei"]
for k in range(3):
    M[f"recent_vei{k}"] = int(recent_vei.loc[k])
M["ratio_all"] = M["vei_windows"][0][2]
M["ratio_recent"] = M["vei_windows"][2][2]
M["n_vei7_all"] = M["vei7"]
tambora = eruptions[(eruptions["Volcano_Name"] == "Tambora")
                    & (eruptions["ExplosivityIndexMax"] == 7)]
M["tambora_year"] = int(tambora["StartDateYear"].iloc[0])

# the two logarithmic scales
M["vol_tambora"] = 10 ** (7 - 5)
M["vol_helens"] = 10 ** (5 - 5)
M["vol_ratio"] = round(M["vol_tambora"] / M["vol_helens"])
energy = 10 ** (1.5 * quakes["mag"] + 4.8)
M["energy_ratio"] = round(float(10 ** (1.5 * 9 + 4.8) / 10 ** (1.5 * 6 + 4.8)))
M["biggest_mag"] = float(quakes["mag"].max())
M["biggest_place"] = str(quakes.loc[quakes["mag"].idxmax(), "place"])
M["biggest_share"] = round(float(energy.max() / energy.sum()), 3)

mag_counts, _ = np.histogram(quakes["mag"], bins=MAG_EDGES)
M["mag_counts"] = [int(c) for c in mag_counts]
M["mag_lowest"], M["mag_second"] = M["mag_counts"][0], M["mag_counts"][1]

# the two homework boxes, so the report can show both forks work
M["hw"] = {}
for name, box in (("Chile", CHILE), ("Japan", JAPAN)):
    arc = arc_box(quakes, box)
    shallow_lon = float(arc[arc["depth"] <= 70]["longitude"].median())
    deep_lon = float(arc[arc["depth"] > 300]["longitude"].median())
    M["hw"][name] = {"n": len(arc), "shallow_lon": round(shallow_lon, 1),
                     "deep_lon": round(deep_lon, 1),
                     "gap": round(deep_lon - shallow_lon, 1),
                     "side": "east" if deep_lon > shallow_lon else "west"}

# How much this equirectangular map stretches high latitudes. 25 and 72 are in the ladder because
# the prose compares Australia with Greenland: the mainland of one straddles latitude 25 and the
# other straddles 72, and the claim is only as good as the two factors the cell prints.
STRETCH_LATS = (0, 25, 60, 72)
M["stretch"] = {lat: round(1 / float(np.cos(np.deg2rad(lat))), 1) for lat in STRETCH_LATS}
M["stretch_ratio"] = round(M["stretch"][72] / M["stretch"][25])

# Land areas, for the one comparison the map's distortion is worth making. Read from
# en.wikipedia.org/wiki/Australia and en.wikipedia.org/wiki/Greenland on 2026-08-31; the notebook
# cites both with that date, the same way it cites Earth's radius and the VEI thresholds.
AREA_AUSTRALIA_KM2, AREA_GREENLAND_KM2 = 7_688_287, 2_166_086
M["australia_over_greenland"] = round(AREA_AUSTRALIA_KM2 / AREA_GREENLAND_KM2, 1)

# The ink each landmass actually costs on this map, by the shoelace area of its coastline polygon
# in square degrees — the quantity "looks the size of" is about. Segment 132 is Greenland and 51 is
# Australia; Africa is not separable, because the shipped coastline joins it to Eurasia at Suez.
# Measured on data/coastlines.csv: Greenland 678, Australia 688. Equal ink, and the two stretch
# factors above say why. Nothing in the prose quotes these, so they stay a comment rather than a
# formatted-in number: the claim a student can check is the one their own figure shows.

# The Alpine-Himalayan belt, for the caution about what "trench" means in the boundary file.
BELT = (25, 45, 40, 100)
belt = arc_box(quakes, BELT)
M["n_belt"] = len(belt)
M["n_belt_deep"] = int((belt["depth"] > 300).sum())
hindu_kush = belt[belt["depth"] >= 200]
M["n_hindu_kush"] = len(hindu_kush)
M["hindu_kush_deepest"] = round(float(hindu_kush["depth"].max()))
M["hindu_kush_lon"] = round(float(hindu_kush["longitude"].median()), 1)
M["hindu_kush_lat"] = round(float(hindu_kush["latitude"].median()), 1)


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
Two files today. One holds every earthquake of magnitude 5.5 and above that the USGS has located
since the start of 2000. The other holds every volcano known to have erupted since the last ice
age. Put either one on a picture and something odd happens: the dots are not scattered. They fall
on lines — down the middle of oceans where there is no land at all, and round the rim of the
Pacific where there is almost nothing else.

You will draw those lines yourself, from nothing but longitude and latitude, and put the plate
boundaries on top of them. Then you will go one step further and colour each earthquake by how
deep it was. The deep ones are not where you would guess. Where they are is one of the clearest
pieces of evidence anybody has that the floor of the Pacific is sinking back into the mantle.
"""

md(weekkit.OPENING.format(question=WEEK["question"], datahub=datahub, hook=HOOK.strip()))

md("""
## What you'll be able to do

**The science.** Say where earthquakes and volcanoes happen, which kind of plate boundary each
one prefers, and where the deep earthquakes sit relative to a trench — and why that geometry is
what a sinking slab looks like. Read a count of eruptions off a scale where each step means ten
times more rock.

**The skills.** Draw a map with no map library: `plt.scatter(longitude, latitude)`, the coastline
and the plate boundaries from a CSV with `plt.plot`. Colour a scatter by a third column with `c=`
and `cmap=`, and label the colours with `plt.colorbar`. Put two panels side by side with
`plt.subplot`. And when the counts run from thousands down to single figures, `plt.yscale("log")`,
without which the figure hides most of its own data.

**Eight places where you write something: five in class, three at home.** Each one is headed
*Your turn*, with an empty cell under it. The first two homework parts ask for numbers and then
for a sentence about them, so those have a second cell as well.

1. Where on the planet are they?
2. What draws those lines?
3. Why are the deep earthquakes not where you'd guess?
4. Why does a count of eruptions need a log axis — and what is missing from the bottom of it?
""")

setup = weekkit.setup_cell(
    imports="import numpy as np\n",
    figsize="(9, 4.5)",
    cache_base=CACHE_BASE,
    signature="url, cached",
    docstring="Read the live source; fall back to the copy stored with the course.",
    url_expr="url",
    cache_expr="cached",
    unpack=f'''
USGS = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&orderby=time-asc"
        "&starttime={START}&endtime={END}&minmagnitude={MINMAG}")
GVP = ("https://webservices.volcano.si.edu/geoserver/GVP-VOTW/ows?service=WFS&version=1.0.0"
       "&request=GetFeature&typeName=GVP-VOTW:Smithsonian_VOTW_Holocene_")

quakes = load(USGS, "{QUAKE_CACHE}")
volcanoes = load(GVP + "Volcanoes&outputFormat=csv", "{VOLCANO_CACHE}")
eruptions = load(GVP + "Eruptions&outputFormat=csv", "{ERUPTION_CACHE}")

# These two files live in this repository, so CACHE is their home rather than their fallback:
# there is no live server to try first.
coast = pd.read_csv(CACHE + "/coastlines.csv")
boundaries = pd.read_csv(CACHE + "/plate_boundaries.csv")

print("earthquakes:", quakes.shape, " volcanoes:", volcanoes.shape,
      " eruptions:", eruptions.shape, " boundary points:", boundaries.shape)
'''.strip("\n"))
code(setup)

# --- section 1 -------------------------------------------------------------
md("""
## 1. Where on the planet are they?

The earthquake catalogue has a `longitude` column and a `latitude` column. Longitude runs from
-180 to 180 across the world and latitude from -90 to 90 up it, so putting one on the bottom axis
and the other on the side is already a map. Nothing else is needed — no map package, no
projection, no install.

First, one line of housekeeping from the tables week: not every row in an earthquake catalogue is
an earthquake, so keep the rows where `type` says it is.

Then the map. Three things beyond the scatter earn their place, and they will be on every map you
draw this term:

- `plt.xlim` and `plt.ylim` fix the edges at the whole world, so one map is comparable with the
  next one;
- `plt.gca().set_aspect("equal")` makes one degree across the same length as one degree up, so
  nothing is stretched;
- the coastline goes on from `data/coastlines.csv` with a single `plt.plot`, exactly as in week
  one — the file has a blank row between coastline segments, which is what makes matplotlib lift
  the pen instead of joining Africa to Australia.
""")

code("""
quakes = quakes[quakes["type"] == "earthquake"]
print(len(quakes), "earthquakes")
""")

code(f"""
plt.scatter(quakes["longitude"], quakes["latitude"], s=1, color="crimson")
plt.plot(coast.lon, coast.lat, color="0.6", lw=0.6)
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.gca().set_aspect("equal")
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title(str(len(quakes)) + " earthquakes, M5.5 and above")
plt.savefig("earthquake_map.png", dpi=150)   # a copy you can drop into a report; before show()
plt.show()
""")

md("""
The dots are not spread over the planet. They are lines, and two of them are worth staring at.

One runs round the edge of the Pacific, through Alaska, Japan, Indonesia, New Zealand and the
whole west coast of the Americas — the Ring of Fire, and if you had guessed anywhere before
running the cell, it was probably there. The other is the surprise: a line straight down the
middle of the Atlantic, from Iceland to the far south, thousands of kilometres from any coast. It
is drawn only by earthquakes. No land marks it, and the coastline you plotted underneath goes
nowhere near it.

Before the next map, one honest caveat about this one. Every degree of longitude is drawn the same
width, but on the ground a degree of longitude at latitude *L* is only `cos(L)` as wide as one at
the equator — the same `cos(latitude)` that came up when you weighted grid cells by area. So the
map stretches everything away from the equator sideways, by a factor of `1 / cos(L)`.
""")

code(f"""
for lat in {list(STRETCH_LATS)}:
    stretch = 1 / np.cos(np.deg2rad(lat))
    print("at latitude", lat, "this map stretches east-west by a factor of", round(stretch, 1))
""")

md(f"""
Only the width is stretched — a degree of latitude is drawn the same length everywhere on this
map — so whatever factor a place gets, its *area* on the page is inflated by that same factor.

Look back at your map with that in mind, at Greenland and at Australia. They come out much the same
size on it — and they are not. Australia covers {AREA_AUSTRALIA_KM2 / 1e6:.1f} million square
kilometres and Greenland {AREA_GREENLAND_KM2 / 1e6:.1f} million, so Australia is
{M['australia_over_greenland']} times the larger (both read from
`en.wikipedia.org/wiki/Australia` and `en.wikipedia.org/wiki/Greenland` on 2026-08-31). Australia
straddles latitude 25, where this map inflates area by {M['stretch'][25]}; Greenland straddles
{STRETCH_LATS[3]}, where it inflates area by {M['stretch'][72]}, about {M['stretch_ratio']} times as
much. The resemblance is the projection, and nothing else. (Greenland coming out the size of
*Africa* is a different projection's story, not this one's: put those two side by side on your own
map and Africa is still several times the larger, even with Greenland flattered by
{M['stretch'][72]}.)

Every flat map has to distort something; this one keeps latitude and longitude honest as
*coordinates* and pays for it in shape and area. For finding out where things are, that is a fine
trade, and it costs no extra library.
""")

ask("""
### ✏️ Your turn 1

The volcano table is loaded as `volcanoes`. Its columns are named differently from the earthquake
catalogue — `Longitude` and `Latitude`, with capitals — because it comes from a different archive.

Draw the same map for the volcanoes: a scatter of longitude against latitude, the coastline on
top, the same limits and aspect, labelled axes, and a title carrying how many there are. Use
`marker="^"` so the volcanoes are triangles, and `s=8` so they are big enough to see.

`plt.scatter` hands back the thing it drew. Catch it in a name — `triangles = plt.scatter(...)` —
so that the self-check can look at the points you actually plotted.

**Use these names**, because the self-check looks for them: `triangles`.
""")

answer(f"""
triangles = plt.scatter(volcanoes["Longitude"], volcanoes["Latitude"],
                        s=8, color="darkgreen", marker="^")
plt.plot(coast.lon, coast.lat, color="0.6", lw=0.6)
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.gca().set_aspect("equal")
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title(str(len(volcanoes)) + " volcanoes that have erupted since the last ice age")
plt.show()
""", """
assert len(triangles.get_offsets()) == len(volcanoes), \\
    "the scatter drew something else — pass the volcano columns, not the earthquake ones"
assert round(triangles.get_offsets()[:, 1].max()) == round(volcanoes["Latitude"].max()), \\
    "longitude goes across and latitude up — check which column you gave scatter first"
print("✓ the volcano map —", len(triangles.get_offsets()), "volcanoes, between latitude",
      round(triangles.get_offsets()[:, 1].min()), "and", round(triangles.get_offsets()[:, 1].max()))
""")

# --- section 2 -------------------------------------------------------------
md("""
## 2. What draws those lines?

The volcanoes fall on lines too, and mostly the same ones: the Andes, the Cascades, the Aleutians,
Japan, Indonesia. So both maps are drawing something neither file contains. That something is the
edges of the tectonic plates, and we can put them on the map from a third file.

`data/plate_boundaries.csv` was converted once from the USGS plate-boundary map into a plain table
of longitudes and latitudes. It has the blank-row trick that `coastlines.csv` has, so one
`plt.plot` draws a whole layer without joining the end of one line to the start of the next. It
also has a `kind` column, because there are three ways two plates can meet:

- a **ridge**, where they pull apart and new ocean floor is made;
- a **transform**, where they slide past each other;
- a **trench**, where one plate bends and goes down underneath the other.
""")

code("""
print(boundaries.head())
print(boundaries["kind"].value_counts())
""")

code(f"""
ridges = boundaries[boundaries["kind"] == "ridge"]
transforms = boundaries[boundaries["kind"] == "transform"]
trenches = boundaries[boundaries["kind"] == "trench"]

plt.scatter(quakes["longitude"], quakes["latitude"], s=1, color="0.75")
plt.plot(coast.lon, coast.lat, color="0.6", lw=0.5)
plt.plot(ridges.lon, ridges.lat, color="tab:blue", lw=1, label="ridge")
plt.plot(trenches.lon, trenches.lat, color="black", lw=1, label="trench")
plt.plot(transforms.lon, transforms.lat, color="tab:green", lw=1, label="transform")
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.gca().set_aspect("equal")
plt.legend(loc="lower left")
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title(str(len(quakes)) + " earthquakes and the plate boundaries")
plt.show()
""")

md("""
The lines land on the dots. The blue ridge down the middle of the Atlantic is the line the
earthquakes drew on their own, and the black trenches trace the Ring of Fire. That is the answer
to the first half of today's question, and it is worth noticing how little work it took: two
scatter plots and three `plt.plot` calls.

One caution about the black line before we use it. `trench` in this file is a label somebody
applied to a boundary, not a measurement, and it has been stretched to cover a handful of features
that are nothing of the sort: the run of black across Asia through the Zagros and the Himalaya is
India and Arabia driving into Eurasia, and a few short pieces around Taiwan and the Philippines are
plates sliding past each other. The file cannot tell you which is which — the names of the features
were dropped when it was converted to a table of longitudes and latitudes. Keep it in mind for the
next section, where the difference will be visible.

The volcanoes have their own opinion about which boundary they like, and the table will say so
directly. `Tectonic_Setting` is a text column, and `.str.startswith("Subduction")` asks the same
question of every row at once — a mask, like the ones you built on arrays, with *does this text
begin like that?* as the question.
""")

code("""
setting = volcanoes["Tectonic_Setting"]
n_subduction = setting.str.startswith("Subduction", na=False).sum()

print("subduction zone:", n_subduction)
print("rift zone:      ", setting.str.startswith("Rift", na=False).sum())
print("  of those, on oceanic crust:",
      setting.str.startswith("Rift zone / Oceanic", na=False).sum())
print("intraplate:     ", setting.str.startswith("Intraplate", na=False).sum())
print("subduction share:", round(100 * n_subduction / len(volcanoes)), "percent of",
      len(volcanoes), "volcanoes")
""")

md(f"""
{M['pct_subduction']}% of them sit at a subduction zone. That is a mechanism, not a coincidence:
the plate going down carries wet ocean-floor minerals with it, and once it is deep enough those
minerals give their water up into the hot mantle above. Water lowers the melting temperature of
rock, so mantle that would otherwise stay solid melts, and the melt rises and builds a line of
volcanoes a hundred kilometres or so behind the trench. Ridges melt rock a different way — by
letting it rise and decompress — and they are the longest boundary system on the planet, but of the
{M['n_rift']} volcanoes the table puts at a rift zone, only {M['n_rift_ocean']} are on oceanic
crust, which is where the mid-ocean ridges are. Most ridge volcanism happens two
kilometres under water, where an eruption leaves nothing for anybody to write down. *A catalogue
lists what somebody's instruments recorded, not what happened. Where there are no seismometers
there are no earthquakes in the file.* The same is true of eruptions, and it will matter again
before the end of the notebook.
""")

# --- section 3 -------------------------------------------------------------
md("""
## 3. Why are the deep earthquakes not where you'd guess?

Every earthquake in the catalogue has a `depth` in kilometres as well as a position. Deep
earthquakes are strange: at a few hundred kilometres down the rock is hot enough and squeezed hard
enough that it should flow rather than snap. Count them first, in three classes seismologists
actually use.
""")

code(weekkit.CHECKPOINT.format(body='quakes = quakes[quakes["type"] == "earthquake"]'))

code("""
shallow = quakes[quakes["depth"] <= 70]
middle = quakes[(quakes["depth"] > 70) & (quakes["depth"] <= 300)]
deep = quakes[quakes["depth"] > 300]

print("shallow, 0-70 km:      ", len(shallow))
print("intermediate, 70-300:  ", len(middle))
print("deep, more than 300 km:", len(deep))
print("deepest:", quakes["depth"].max(), "km")
""")

md(f"""
{M['n_shallow']:,} of the {M['n_quakes']:,} are in the top 70 km, and the deepest is
{M['deepest']} km down. At that depth rock under that much pressure and heat deforms by flowing
rather than by breaking, so a deep earthquake is something that needs explaining. Where the
{M['n_deep']} of them are is the question, and a map can answer it if the colour of each dot
carries the depth.

`plt.scatter` takes `c=` for the values to colour by and `cmap=` for the colour scheme;
`vmin` and `vmax` fix what the ends of the scale mean, so that the colours mean the same thing on
every map you draw. The thing `plt.scatter` hands back — the one you caught as `triangles` on the
volcano map — is what `plt.colorbar(label=...)` needs, to know which colours it is the key to.
""")

ask(f"""
### ✏️ Your turn 2

Redraw the world map of earthquakes, but colour each dot by its depth.

Pass `c=quakes["depth"]` and `cmap="plasma_r"` to `plt.scatter`, with `vmin=0` and `vmax=600` so
that the scale is fixed and pale means shallow. Catch what `plt.scatter` hands back, as you did on
the volcano map, and give that to `plt.colorbar`:

```
dots = plt.scatter(...)
plt.colorbar(dots, label="depth (km)")
```

Keep everything else from the first map — the coastline, the limits, the equal aspect, the labels,
and a title with the sample size in it.

**Use these names**, because the self-check looks for them: `dots`.
""")

answer(f"""
dots = plt.scatter(quakes["longitude"], quakes["latitude"], s=2,
                   c=quakes["depth"], cmap="plasma_r", vmin=0, vmax=600)
plt.colorbar(dots, label="depth (km)")
plt.plot(coast.lon, coast.lat, color="0.6", lw=0.6)
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.gca().set_aspect("equal")
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title(str(len(quakes)) + " earthquakes, coloured by depth")
plt.show()
""", """
assert dots.get_array() is not None, "pass c=quakes[\\"depth\\"], or the dots carry no colour"
assert dots.get_array().max() > 300, "the colours should carry depth in km, not magnitude"
print("✓ the depth map — the colours run from", round(quakes["depth"].min()),
      "to", round(quakes["depth"].max()), "km")
""")

md(f"""
The ridges are uniformly pale: everything that happens at a spreading centre happens in the top
few tens of kilometres. The dark dots — the deep ones — appear in only a handful of places, and
every one of them is a place where the previous figure drew a trench: South America, Japan,
Indonesia, Tonga.

Notice also what is *not* dark. Nowhere along the black line across Asia is there a dot at the dark
end of the scale: of the {M['n_belt']} earthquakes this catalogue puts in that belt, not one is
deeper than 300 km. Through the Zagros and the Himalaya that is the caution from the last section
made visible — two continents colliding, with nothing going down to break. But the belt is not all
one thing, and the map does not claim it is: over northern Afghanistan, near
{M['hindu_kush_lon']}°E, sits a knot of {M['n_hindu_kush']} mid-scale events between 200 km and
{M['hindu_kush_deepest']} km down. That is the Hindu Kush, and something *is* descending under it.
So a line labelled "trench" can be a collision, or a slab, or neither, and the colours can tell
them apart where the label cannot.

And in the places that do go dark, the dark dots are not on the trench line; they are set back
from it. That offset is the whole point, and it is easier to measure on one arc than on the whole
world.
""")

# --- section 3b: one arc, close up ------------------------------------------
md("""
Zooming a map means changing `plt.xlim` and `plt.ylim` and nothing else — the data is the same
data. To keep only the earthquakes inside the box, filter on latitude and longitude the way you
filtered on magnitude in the tables week, joining the four conditions with `&`, one bracketed
condition at a time.
""")

code(weekkit.CHECKPOINT.format(body="""quakes = quakes[quakes["type"] == "earthquake"]
trenches = boundaries[boundaries["kind"] == "trench"]"""))

ask(f"""
### ✏️ Your turn 3

Take South America: latitude {SOUTH[0]} to {SOUTH[1]}, longitude {SOUTH[2]} to {SOUTH[3]}.

1. Filter `quakes` to that box and call it `south`.
2. Draw it exactly like your depth map — `c=south["depth"]`, `cmap="plasma_r"`, `vmin=0`,
   `vmax=600`, a colorbar — but with the limits set to the box, and with the trenches drawn on top
   in black (`plt.plot(trenches.lon, trenches.lat, color="black", lw=1.2)`).
3. Then print two numbers: the median longitude of the shallow events (`depth` at most 70) and the
   median longitude of the deep ones (`depth` over 300).

**Use these names**, because the self-check looks for them: `south`, `shallow_lon`, `deep_lon`.
""")

answer(f"""
south = quakes[(quakes["latitude"] >= {SOUTH[0]}) & (quakes["latitude"] <= {SOUTH[1]})
               & (quakes["longitude"] >= {SOUTH[2]}) & (quakes["longitude"] <= {SOUTH[3]})]

dots = plt.scatter(south["longitude"], south["latitude"], s=8,
                   c=south["depth"], cmap="plasma_r", vmin=0, vmax=600)
plt.colorbar(dots, label="depth (km)")
plt.plot(coast.lon, coast.lat, color="0.6", lw=0.6)
plt.plot(trenches.lon, trenches.lat, color="black", lw=1.2)
plt.xlim({SOUTH[2]}, {SOUTH[3]})
plt.ylim({SOUTH[0]}, {SOUTH[1]})
plt.gca().set_aspect("equal")
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
plt.title("South America: " + str(len(south)) + " earthquakes")
plt.show()

shallow_lon = south[south["depth"] <= 70]["longitude"].median()
deep_lon = south[south["depth"] > 300]["longitude"].median()
print("shallow events, median longitude:", round(shallow_lon, 2))
print("deep events, median longitude:   ", round(deep_lon, 2))
""", f"""
assert len(south) > 0, "no earthquakes in the box — check the four limits"
assert south["latitude"].max() <= {SOUTH[1]} and south["longitude"].min() >= {SOUTH[2]}, \\
    "south still reaches outside the box — the four conditions join with &, not |"
assert deep_lon > shallow_lon, \\
    "here the deep events lie east of the shallow ones, so this difference should be positive; " \\
    "a negative one means the two depth classes have been taken the wrong way round"
print("✓ South America —", len(south), "earthquakes; median longitude", round(shallow_lon, 1),
      "shallow against", round(deep_lon, 1), "deep, a gap of",
      round(deep_lon - shallow_lon, 1), "degrees")
""")

md(f"""
The black trench line runs down the coast; the pale shallow events sit on it, and the dark ones
sit in two tight clusters several degrees inland. Degrees of longitude are hard to feel, so turn
them into kilometres. One degree of longitude is the equator's circumference divided by 360, times
`cos(latitude)`. Earth's mean radius is {EARTH_RADIUS_KM} km — the value the International Union
of Geodesy and Geophysics publishes, read from `en.wikipedia.org/wiki/Earth_radius` on 2026-08-31.
""")

# This checkpoint deliberately rebuilds LESS than the cell below it reads. `south`, `shallow_lon`
# and `deep_lon` are Your turn 3's graded answer, three cells above; writing them out here would
# hand the answer to anybody who scrolls. So it rebuilds the class cell they were built from and
# names the three, telling the student to re-run their own cell — which is what
# check_checkpoints_rebuild's re-run exemption is for.
code(weekkit.CHECKPOINT.format(body="""quakes = quakes[quakes["type"] == "earthquake"]

# The next cell works from your own Your turn 3 answer. Re-run that cell to rebuild `south`,
# `shallow_lon` and `deep_lon`; they are not repeated here, because they are the answer."""))

code(f"""
deep_lat = south[south["depth"] > 300]["latitude"].median()
km_per_degree = 2 * np.pi * {EARTH_RADIUS_KM} / 360 * np.cos(np.deg2rad(deep_lat))

print("one degree of longitude here:", round(km_per_degree), "km")
print("the deep events sit", round((deep_lon - shallow_lon) * km_per_degree), "km inland")
print("the deepest event in this box:", round(south["depth"].max()), "km")
""")

md(f"""
About {M['south_gap_km']} kilometres. Put that beside the depths and the shape is forced: the
earthquakes get deeper the further inland you go, from a few tens of kilometres at the coast to
{M['south_deepest']} km under the interior of the continent. They are not scattered through the
mantle at random depths; they lie on a surface that starts at the trench and dips away under the
continent. That surface is the plate: cold ocean floor bending down at the trench and sliding into
the mantle, with the deep earthquakes happening inside it.

That is *where*, and it is only *where*. Being cold is why the slab is the only place down there
that makes earthquakes at all — it is the one cold thing in a mantle that is otherwise hot and
flowing quietly — but being cold does not make deep rock breakable. Six hundred kilometres down
the weight of the rock above is about {M['gpa_600km']} gigapascals, {M['atm_600km']:,} times the
pressure of the air in this room (worked out from PREM, the standard profile of density inside the
Earth, `ds.iris.edu/files/products/emc/data/PREM/PREM_1s.csv`, read 2026-08-31), and under that
squeeze rock does not crack open however cold it is. So the strangeness this section opened with
is real, and it is still unsolved. Seismologists have candidates: water, given up as minerals in
the slab break down, which can let a fault slip somewhere between 70 and 300 km; and below roughly
400 km, a mineral that has survived too long in the wrong crystal form flipping suddenly into a
denser one, or a thin sliding layer heating itself faster than the heat can escape. Which of them
does the work, and at what depth, is still argued about (Green and Houston, *Annual Review of Earth
and Planetary Sciences*, 1995; Houston, *Deep Earthquakes*, in the *Treatise on Geophysics*, 2015).
Your figure cannot tell you the mechanism. It tells you the geometry, which is the part that is
settled: whatever is breaking down there, it is breaking inside a sinking plate.
""")

# --- section 4 -------------------------------------------------------------
md(f"""
## 4. Why does a count of eruptions need a log axis — and what is missing from the bottom of it?

That is where; the rest of the notebook is how big. Volcanologists score an eruption on the
**Volcanic Explosivity Index**, a whole number from 0 to 8, and the eruption table carries it in
`ExplosivityIndexMax`. There are {M['n_eruptions']:,} eruptions in the file — every one known
since the last ice age.
""")

md(f"""
### Predict before you run

VEI 0 is the smallest kind of eruption and VEI 2 is two steps up. The record holds
{M['vei2']:,} eruptions at VEI 2. How many do you think it holds at VEI 0? Write your guess into
`my_guess` before you run the cell.
""")

code(f"""
my_guess = 20000

vei_counts = eruptions["ExplosivityIndexMax"].value_counts().sort_index()
print(vei_counts)
print("you guessed", my_guess, "at VEI 0; the record holds", int(vei_counts.loc[0]))
""")

md(f"""
Fewer, not more — {M['vei0']:,} against {M['vei2']:,}. Hold that; it is the second surprise of the
day, and we come back to it before the section is out.

One thing to notice about that table on the way past: it adds up to {M['n_with_vei']:,}, not the
{M['n_eruptions']:,} eruptions in the file. `.value_counts()` counts the values it finds and says
nothing about the rows where there is no value to count — and {M['n_no_vei']:,} of these eruptions,
{M['pct_no_vei']}% of the record, carry no VEI at all, because nobody could say how big they were.
That is why the charts below are titled with the number that carries a VEI rather than the length
of the file. Keep the number; it comes back with something to say.

First, what the index means. VEI is not a volume, it is an *index*: each whole step stands for
roughly ten times more erupted rock. From VEI 2 upwards the convention is that VEI *n* means at
least 10 to the power (*n* − 5) cubic kilometres — so VEI 5 is 1 km³ and VEI 7 is 100 km³.
(Newhall and Self defined the index in 1982; the thresholds here were read from
`en.wikipedia.org/wiki/Volcanic_explosivity_index` on 2026-08-31. The rule breaks below VEI 2,
where the steps are not tenfold, which is one reason not to trust the bottom of this scale.)
""")

ask("""
### ✏️ Your turn 4

Write `vei_volume(vei)`: one argument, a docstring saying what it does, and it returns the
smallest erupted volume in cubic kilometres that the index stands for — 10 to the power
(`vei` − 5).

Then print the volume for Tambora, which is VEI 7, the volume for Mount St Helens in 1980, which
is VEI 5, and how many times bigger the first is than the second.

**Use these names**, because the self-check looks for them: `vei_volume`.
""")

answer("""
def vei_volume(vei):
    \"\"\"The smallest erupted volume, in cubic kilometres, that a VEI number stands for.\"\"\"
    return 10 ** (vei - 5)


print("Tambora, VEI 7:  ", vei_volume(7), "cubic km")
print("St Helens, VEI 5:", vei_volume(5), "cubic km")
print("ratio:", vei_volume(7) / vei_volume(5))
""", """
assert vei_volume(6) / vei_volume(5) == 10, "one step of VEI should be a factor of ten"
print("✓ VEI is an index, not a volume — two steps up is a factor of",
      round(vei_volume(7) / vei_volume(5)))
""")

md(f"""
A factor of {M['vol_ratio']} between two eruptions that are only two apart on the scale. A count
of eruptions per VEI has the same problem: the classes at the top are rare by exactly as much as
they are big. Draw it on an ordinary axis and see what happens.

`plt.bar(positions, heights)` draws one bar per category — the right chart when the thing on the
bottom axis is a label rather than a measurement. `plt.subplot(1, 2, 1)` means *one row of two
panels, and I am drawing in the first*, so two charts can sit side by side and be compared.
""")

code(f"""
plt.subplot(1, 2, 1)
plt.bar(vei_counts.index, vei_counts.values, color="darkorange")
plt.xlabel("VEI")
plt.ylabel("number of eruptions")
plt.title("ordinary axis, n = " + str(int(vei_counts.sum())))

plt.subplot(1, 2, 2)
plt.bar(vei_counts.index, vei_counts.values, color="darkorange")
plt.yscale("log")
plt.xlabel("VEI")
plt.ylabel("number of eruptions")
plt.title("log axis, n = " + str(int(vei_counts.sum())))
plt.show()
""")

md(f"""
On the left, the top of the scale is unreadable. The axis has to climb to {M['vei2']:,} to fit the
tallest bar, so VEI 4 and VEI 5 are there but impossible to compare, VEI 6 is a hairline, and the
{M['vei7']} eruptions at VEI 7 do not show at all. On the right, the same numbers on a log axis:
*when the values span factors of a thousand, plot the exponents instead and a curve becomes a
line.* Every class is now readable, and the tops of the bars from
VEI 2 to VEI 6 come down in near-equal steps — which on a log axis means each class is a roughly
constant factor rarer than the one below it.

And now the low end is impossible to miss. VEI 1 and VEI 0 sit *below* VEI 2, which is the wrong
way round. The steps from VEI 2 upwards say that going one step *down* the scale should multiply
the count, so the two smallest classes ought to be the two tallest bars on the chart, and instead
they are shorter than the class above them. Either the world really does make fewer small
eruptions than middling ones, or the record is missing them. There is a way to tell the two apart:
look at a window of the catalogue where the recording is better.
""")

ask(f"""
### ✏️ Your turn 5

Modern volcano monitoring is nothing like the record of the last ten thousand years as a whole. So
cut the table to eruptions that started in 1950 or later, using `StartDateYear`, and draw the same
log-axis bar chart for that window alone.

Then, to compare the windows as numbers rather than pictures, loop over
`first_years = [-60000, 1800, 1950]`. For each one, take the eruptions from that year onwards,
count them by VEI with `.value_counts()`, and print three things: how many eruptions the window
holds, how many of them carry a VEI at all (`counts.sum()`), and how many VEI 1 eruptions there are
for each VEI 2 — `counts.loc[1] / counts.loc[2]`, rounded to two decimals.

**Use these names**, because the self-check looks for them: `recent`, `recent_counts`.
""")

answer(f"""
recent = eruptions[eruptions["StartDateYear"] >= 1950]
recent_counts = recent["ExplosivityIndexMax"].value_counts().sort_index()

plt.bar(recent_counts.index, recent_counts.values, color="darkorange")
plt.yscale("log")
plt.xlabel("VEI")
plt.ylabel("number of eruptions")
plt.title("eruptions since 1950, n = " + str(int(recent_counts.sum())))
plt.show()

first_years = [-60000, 1800, 1950]
for first_year in first_years:
    window = eruptions[eruptions["StartDateYear"] >= first_year]
    counts = window["ExplosivityIndexMax"].value_counts()
    print("from", first_year, "onwards:", len(window), "eruptions,", int(counts.sum()),
          "with a VEI, VEI 1 per VEI 2 =", round(counts.loc[1] / counts.loc[2], 2))
""", """
assert recent["StartDateYear"].min() >= 1950, \\
    "recent holds eruptions from before 1950 — check which way round the comparison points"
assert recent_counts.sum() <= len(recent), \\
    "recent_counts should count the VEI column of `recent`, not of the whole table"
print("✓ the modern window —", len(recent), "eruptions since 1950, of which",
      len(recent) - int(recent_counts.sum()), "carry no VEI at all")
""")

md(f"""
Two numbers move the same way. The ratio of VEI 1 to VEI 2 climbs from {M['ratio_all']} over the
whole record to {M['ratio_recent']} since 1950; and the share of eruptions with no VEI at all —
where the record does not even say how big — falls from {M['pct_no_vei']}% over the whole record to
{M['pct_no_vei_1800']}% since 1800 and {M['pct_no_vei_1950']}% since 1950. Improve the recording and
the deficit shrinks. That is the argument: what is missing from the bottom of the chart is a
property of the archive, not of the planet. A VEI 2 eruption a thousand years ago left a layer of
ash somebody can still dig up. A VEI 0 eruption a thousand years ago left nothing, unless somebody
was standing there.

Shrinks, though, is not *gone*. Your own 1950 chart still has {M['recent_vei0']} eruptions at VEI 0
against {M['recent_vei1']} at VEI 1 and {M['recent_vei2']:,} at VEI 2 — both of the smallest classes
still below the class above them, in the best-recorded window the file has. Seventy years of
instruments have made the record better, not complete — and the honest reading of the low end of
that chart is still *we do not know*, rather than a count.

Earthquake magnitude works the same way, and its constant is worth knowing. The energy an
earthquake radiates as seismic waves goes as log₁₀ *E* = 1.5 *M* + 4.8, with *E* in joules — the
Gutenberg–Richter convention; the factor of about 31.6 in energy per whole magnitude step was read
from `en.wikipedia.org/wiki/Richter_scale` on 2026-08-31.
""")

code(f"""
def quake_energy(mag):
    \"\"\"The energy an earthquake of this magnitude radiates as seismic waves, in joules.\"\"\"
    return 10 ** (1.5 * mag + 4.8)


print("an M9 against an M6:", round(quake_energy(9) / quake_energy(6)), "times the energy")

energy = quake_energy(quakes["mag"])
print(quakes.sort_values("mag", ascending=False).head(1)[["mag", "place"]])
print("that one event's share of the catalogue's energy:", round(energy.max() / energy.sum(), 3))
""")

md(f"""
One earthquake out of {M['n_quakes']:,} released {M['biggest_share'] * 100:.0f}% of the seismic
energy of twenty-six years. Both scales — VEI and magnitude — are built that way on purpose: they
are indices with a factor hiding in every step, so a count of them belongs on a log axis and a
difference of two on the scale is never a difference of two in the world.
""")

# --- closing ---------------------------------------------------------------
md(f"""
{weekkit.CLOSING_HEADING}

**On the plate boundaries, and the deep ones behind the trenches.** Every earthquake and almost
every volcano you plotted sits on the edge of a tectonic plate: the earthquakes drew the boundary
lines before you loaded them, including the mid-Atlantic ridge, where no land marks it at all.
{M['pct_subduction']}% of the volcanoes sit at subduction zones, where water carried down with the
sinking plate melts the mantle above it. And the deep earthquakes, absent from the ridges and
clustered near the trenches, sit {M['south_gap_km']} km inland of the shallow ones in South America,
because they are happening inside a cold plate that is still sinking, hundreds of kilometres past
the point where it went under. How rock manages to break at all under that much pressure is a
question seismologists have not closed; *where* it breaks, your own figure settled. One arc is one
arc; the homework asks you to check a second one.
""")

md(weekkit.week_cheatsheet(4))

# --- homework --------------------------------------------------------------
md("""
## Homework

Three parts, on the same two catalogues you already have loaded. If you have restarted since
class, run the setup cell at the top first and then the `type` filter in the first section.
""")

ask(f"""
### ✏️ Your turn 6

Class found that the eruption record is missing its smallest eruptions. Is the earthquake
catalogue missing its smallest earthquakes in the same way? The same picture answers it.

VEI came ready-made in whole numbers, so `.value_counts()` was enough. Magnitude does not, so bin
it first, exactly as you binned elevations in the grids week:

```
edges = np.arange(5.5, 9.6, 0.5)
counts, edges = np.histogram(quakes["mag"], bins=edges)
centres = (edges[:-1] + edges[1:]) / 2
```

Then draw `counts` against `centres` as a bar chart with `width=0.45`, put the count axis on a log
scale, label both axes, title it with the sample size, and print `counts` so you can read the
numbers off.

Then answer the question in one sentence in the cell after: do your two smallest magnitude bins
fall below the bin above them, the way VEI 0 and VEI 1 fell below VEI 2 in class? Quote
`counts[0]` and `counts[1]` from your own printout.

**Use these names**, because the self-check looks for them: `counts`, `centres`.
""")

answer("""
edges = np.arange(5.5, 9.6, 0.5)
counts, edges = np.histogram(quakes["mag"], bins=edges)
centres = (edges[:-1] + edges[1:]) / 2

plt.bar(centres, counts, width=0.45, color="crimson")
plt.yscale("log")
plt.xlabel("magnitude")
plt.ylabel("number of earthquakes")
plt.title("magnitudes in half-unit bins, n = " + str(counts.sum()))
plt.show()

print(counts)
""", """
assert counts.sum() == len(quakes), "every earthquake should land in a bin — check your edges"
print("✓ magnitudes on a log axis — the smallest bin holds", counts[0],
      "earthquakes and the next one", counts[1])
""")

answer_prose(f"""
No — mine do the opposite. `counts[0]` is {M['mag_lowest']:,} earthquakes in the 5.5–6.0 bin and
`counts[1]` is {M['mag_second']:,} in the 6.0–6.5 bin, so the smallest bin is the tallest bar on
the chart and the counts keep climbing all the way down to the left edge. The eruption record does
the reverse — VEI 0 and VEI 1 both sit below VEI 2 — so whatever is eating the small eruptions is
not eating the small earthquakes.
""")

ask(f"""
### ✏️ Your turn 7

Class drew South America from above. Seen from the side, the same scatter becomes a cross-section:
put longitude on the bottom axis and depth up the side, and the sinking plate draws itself.

**Choose one arc**, and say in a comment which you chose:

- **Chile** — latitude {CHILE[0]} to {CHILE[1]}, longitude {CHILE[2]} to {CHILE[3]}
- **Japan** — latitude {JAPAN[0]} to {JAPAN[1]}, longitude {JAPAN[2]} to {JAPAN[3]}

Filter `quakes` to your box and call it `arc`. Plot `arc["longitude"]` against `-arc["depth"]` —
the minus sign turns a depth into a height, so the picture is the right way up and the axis label
should say so. Catch what `plt.scatter` hands back as `section`, as you did on the maps, so that
the self-check can look at the points you drew. Label both axes and title the figure with the arc's
name and how many earthquakes are in it. Then work out `shallow_lon` and
`deep_lon`, the median longitudes of the events no deeper than 70 km and of those deeper than
300 km — the same two numbers you printed for South America.

The two boxes do not give the same answer, and both are right.

Then, in one sentence in the cell after, worked out from your own two medians rather than from the
figure: which side of the shallow events do the deep ones sit on, and which way is the plate going
down under the arc you chose? The self-check under your answer prints the gap between the two
medians but not its direction, so this one is yours to read off the sign.

**Use these names**, because the self-check looks for them: `arc`, `section`, `shallow_lon`,
`deep_lon`.
""")

# The payload of the fork, written into the solution as a comment so that a grader marking a Japan
# submission has the key for it. Wrapped here rather than by hand: the counts and the offsets are
# formatted in, so hand-laid line breaks would come out ragged whenever a number changes width.
WHY_BOTH_ARE_RIGHT = textwrap.fill(
    f"Why the two arcs disagree, and why both answers are right. Chile has "
    f"{M['hw']['Chile']['n']} earthquakes in the box and its deep ones sit "
    f"{abs(M['hw']['Chile']['gap'])} degrees EAST of the shallow ones; Japan has "
    f"{M['hw']['Japan']['n']} and its deep ones sit {abs(M['hw']['Japan']['gap'])} degrees WEST "
    f"of them. Off South America the Nazca plate is sinking eastwards under the continent, so the "
    f"slab gets deeper towards the east and carries its earthquakes inland that way. Off Japan "
    f"the Pacific plate is sinking westwards, so the same geometry is mirrored and the deep "
    f"events land on the other side of the shallow ones. The sign of the offset is the direction "
    f"the plate is going down, and the size, a few degrees either way, is the same slab dipping "
    f"at much the same angle. So a negative number for Japan and a positive one for Chile are "
    f"both right, and neither is arbitrary: the sign follows the plate.",
    width=96, initial_indent="# ", subsequent_indent="# ")

answer(f"""
# Chile. Either box is a right answer, so both are worked here.
arc = quakes[(quakes["latitude"] >= {CHILE[0]}) & (quakes["latitude"] <= {CHILE[1]})
             & (quakes["longitude"] >= {CHILE[2]}) & (quakes["longitude"] <= {CHILE[3]})]

section = plt.scatter(arc["longitude"], -arc["depth"], s=8, color="crimson")
plt.xlabel("longitude (degrees east)")
plt.ylabel("height relative to the surface (km)")
plt.title("Chile: " + str(len(arc)) + " earthquakes")
plt.show()

shallow_lon = arc[arc["depth"] <= 70]["longitude"].median()
deep_lon = arc[arc["depth"] > 300]["longitude"].median()
print("Chile, shallow events, median longitude:", round(shallow_lon, 2))
print("Chile, deep events, median longitude:   ", round(deep_lon, 2))

# The other choice, Japan, worked exactly the same way.
japan = quakes[(quakes["latitude"] >= {JAPAN[0]}) & (quakes["latitude"] <= {JAPAN[1]})
               & (quakes["longitude"] >= {JAPAN[2]}) & (quakes["longitude"] <= {JAPAN[3]})]

plt.scatter(japan["longitude"], -japan["depth"], s=8, color="crimson")
plt.xlabel("longitude (degrees east)")
plt.ylabel("height relative to the surface (km)")
plt.title("Japan: " + str(len(japan)) + " earthquakes")
plt.show()

japan_shallow_lon = japan[japan["depth"] <= 70]["longitude"].median()
japan_deep_lon = japan[japan["depth"] > 300]["longitude"].median()
print("Japan, shallow events, median longitude:", round(japan_shallow_lon, 2))
print("Japan, deep events, median longitude:   ", round(japan_deep_lon, 2))

{WHY_BOTH_ARE_RIGHT}
""", """
assert len(arc) > 0, "no earthquakes in that box — check the four numbers"
assert section.get_offsets()[:, 1].max() <= 0, \\
    "the deepest events are drawn at the TOP — plot -arc[\\"depth\\"], so that down the page is down"
assert 1 < abs(deep_lon - shallow_lon) < 15, \\
    "the two medians should be a few degrees apart; a gap of nearly nothing usually means both " \\
    "were taken over the same depth class"
print("✓ the slab —", len(arc), "earthquakes; deep median longitude minus shallow is",
      round(deep_lon - shallow_lon, 1), "degrees")
""")

answer_prose(f"""
I took Chile. My deep events have a median longitude of {M['hw']['Chile']['deep_lon']} and my
shallow ones {M['hw']['Chile']['shallow_lon']}, so deep minus shallow is
{M['hw']['Chile']['gap']} degrees — a positive number, which means the deep earthquakes sit
{M['hw']['Chile']['gap']} degrees **{M['hw']['Chile']['side']}** of the shallow ones, inland of the
trench. The slab therefore dips {M['hw']['Chile']['side']}wards: the Nazca plate goes down under
South America, getting deeper the further inland you follow it. (Japan gives
{M['hw']['Japan']['gap']} degrees, a negative number, because there the Pacific plate is sinking
{M['hw']['Japan']['side']}wards under the arc — the mirror image, and just as right.)
""")

ask("""
### ✏️ Your turn 8

Two of the pictures in this notebook are counts on a log axis: the eruptions by VEI, and your own
magnitudes from part 6. One of them falls off at its low end and the other does not.

In three or four sentences, and using your own printed numbers — the three shortest VEI bars from
class, and the first two numbers in your `counts` array — say which chart has the broken low end,
and explain what is different about how the two catalogues were made. Your answer should say what
would have to be true for the *other* chart's low end to break as well.
""")

answer_prose(f"""
The eruption chart is the broken one. Its two smallest classes hold {M['vei0']:,} eruptions at VEI
0 and {M['vei1']:,} at VEI 1, both below the {M['vei2']:,} at VEI 2, even though smaller eruptions
must be commoner than larger ones. The magnitude chart does the opposite: {M['mag_lowest']:,}
earthquakes in the 5.5-6.0 bin against {M['mag_second']:,} in the next one up, so the counts keep
rising all the way down to the edge of the plot. The difference is how the two records are made.
An earthquake of magnitude 5.5 is recorded by instruments on the far side of the planet, and the
network has been dense enough for that throughout the window this query asks for, so the
catalogue holds essentially every one wherever it happened. Eruptions are recorded by whoever was
nearby and by whatever the eruption left behind, so a small eruption in an empty place a thousand
years ago is simply absent — and the deficit gets worse the further back the window reaches: the
number of VEI 1 eruptions per VEI 2 rises from {M['ratio_all']} over the whole record to
{M['ratio_recent']} since 1950. For the magnitude chart to break the same way, the catalogue would
have to be missing earthquakes in whole regions, which is what catalogue completeness warns about
and what would start to happen if the query's magnitude floor were pushed low enough: a small
earthquake is only recorded where somebody has already put an instrument close by.
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

    # the notebook's savefig cell writes a PNG beside itself; that is the student's copy to keep,
    # not something the repository should carry
    for stray in (OUT / "earthquake_map.png", ROOT / "earthquake_map.png"):
        stray.unlink(missing_ok=True)

    (OUT / f"{SLUG}.ipynb").write_text(json.dumps(stu, indent=1) + "\n")
    print(f"wrote {SLUG}.ipynb ({len(CELLS)} cells) and the executed solution")
    print(f"cache: data/{QUAKE_CACHE}, data/{VOLCANO_CACHE}, data/{ERUPTION_CACHE}")
    print("repo asset: data/plate_boundaries.csv")


if __name__ == "__main__":
    main()
    weekkit.gate(4)
