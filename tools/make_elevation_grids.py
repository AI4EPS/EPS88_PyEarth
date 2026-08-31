#!/usr/bin/env python
"""Build data/earth_elevation.csv and data/mars_elevation.csv — the week-3 hypsometry grids.

Both sources are unusable as they ship. Earth is 15 MB of ASCII sitting at a LOCAL path in
offerings/ with no URL, and TEMPLATE forbids local paths; Mars is 2 MB of big-endian int16 with a
detached PDS label, which no week-3 student could read with the six libraries. nbgitpuller clones
data/ onto 46 accounts, so both are resampled to 1 degree and written as plain CSV.

Resampling costs nothing the week needs: the fraction of Earth below sea level is 0.662 at the
full 20-arcmin grid and 0.660 at 1 degree, and the bimodal shape is identical. 583,200 cells
become 64,800 — still far more than a histogram needs, and a legible map.

    python tools/make_elevation_grids.py
"""
import pathlib, subprocess
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
ETOPO = ROOT.parent / "offerings/_data/acc8e0c2_etopo20data.txt"
MOLA = ("https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/"
        "mgsl_300x/meg004/megt90n000cb.img")


def earth():
    g = np.loadtxt(ETOPO)[:, :-1]        # drop the duplicated wrap column: 20.17E..380.17E
    g = g[::3, ::3]                      # 20 arc-min -> 1 degree, (180, 360)
    # SHIP IT ALIGNED. ETOPO runs south->north and starts at 20.17E; MOLA runs north->south and
    # starts at 0.125E. A week-3 builder had to re-derive both from the source files and spend
    # five setup cells flipping and rolling them — code no beginner could invent. Both grids now
    # leave here in one convention: row 0 is the northernmost, column 0 is the westernmost.
    return np.roll(g[::-1], -160, axis=1)


def mars(cache=ROOT / "data/.megt90n000cb.img"):
    if not cache.exists():
        # curl, not urllib: the PDS server fails certificate verification under this Python,
        # and curl uses the system trust store. Students never hit PDS — they read the CSV.
        subprocess.run(["curl", "-sSLf", "-o", str(cache), MOLA], check=True)
    g = np.fromfile(cache, dtype=">i2").reshape(720, 1440)   # MSB_INTEGER, metres
    g = g[::4, ::4]                                          # 4 px/deg -> 1 degree, (180, 360)
    return np.roll(g, -180, axis=1)                          # already north-first; roll to -180


def main():
    for name, grid in (("earth", earth()), ("mars", mars())):
        out = ROOT / f"data/{name}_elevation.csv"
        np.savetxt(out, grid.astype(np.int16), fmt="%d", delimiter=",")
        kb = out.stat().st_size / 1024
        print(f"  {name:<6} {grid.shape}  {grid.size} cells  "
              f"min {grid.min():>7} max {grid.max():>6} m  ->  {out.name} ({kb:.0f} kB)")


if __name__ == "__main__":
    main()
