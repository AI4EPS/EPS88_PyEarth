#!/usr/bin/env python
"""Build data/coastlines.csv — geographic context for every map in the course.

Cartopy is not in the DataHub image and was dropped, which left every map in this course as
dots in a blank rectangle: a beginner cannot tell the Pacific from the page. This converts
Natural Earth's 110 m coastline (public domain) into a plain lon/lat CSV that students draw
with one plt.plot per segment — the same trick as tools/make_plate_boundaries.py.

Run once, instructor-side. Students read the CSV, never this script.
"""
import csv, json, pathlib, urllib.request

URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
       "master/geojson/ne_110m_coastline.geojson")
OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "coastlines.csv"


def main():
    g = json.loads(urllib.request.urlopen(URL, timeout=120).read())
    rows, seg = [], 0
    for f in g["features"]:
        geom = f["geometry"]
        parts = ([geom["coordinates"]] if geom["type"] == "LineString"
                 else geom["coordinates"])
        for part in parts:
            prev = None
            for lon, lat in part:
                # split any segment that jumps the dateline, or plt.plot draws a line
                # straight across the map
                if prev is not None and abs(lon - prev) > 180:
                    seg += 1
                rows.append((seg, round(lon, 3), round(lat, 3)))
                prev = lon
            seg += 1

    # A blank row between segments makes matplotlib lift the pen, so the whole coastline
    # draws with ONE plt.plot and no loop. Week 1 teaches neither loops nor groupby, and a
    # rule that every map carries coastlines has to be followable in week 1.
    out, last = [], None
    for s, lon, lat in rows:
        if last is not None and s != last:
            out.append(("", "", ""))
        out.append((s, lon, lat))
        last = s

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["segment", "lon", "lat"])
        w.writerows(out)
    print(f"{OUT.name}: {seg} segments, {len(rows)} vertices, "
          f"{OUT.stat().st_size/1e3:.0f} kB — plot with a single plt.plot(c.lon, c.lat)")


if __name__ == "__main__":
    main()
