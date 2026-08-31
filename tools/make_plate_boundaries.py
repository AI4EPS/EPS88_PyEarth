#!/usr/bin/env python
"""Convert USGS plate-boundary shapefiles to a plain lon/lat CSV.

Run once by the instructor; the CSV ships as a GitHub release asset. Students then
draw the boundaries with plt.plot() and never install cartopy or geopandas.

Uses only the Python standard library — no GIS stack, so it runs anywhere.

    python tools/make_plate_boundaries.py <dir-with-shapefiles> plate_boundaries.csv
"""
import csv, pathlib, struct, sys

SHAPE_POLYLINE = 3


def polylines(path: pathlib.Path):
    """Yield each polyline in an ESRI shapefile as a list of (lon, lat)."""
    b = path.read_bytes()
    off = 100                                    # skip the file header
    while off < len(b):
        _rec_no, content_len = struct.unpack(">ii", b[off:off + 8])
        off += 8
        rec = b[off:off + content_len * 2]
        off += content_len * 2
        if struct.unpack("<i", rec[:4])[0] != SHAPE_POLYLINE:
            continue                             # ignore points/polygons
        n_parts, n_points = struct.unpack("<ii", rec[36:44])
        starts = list(struct.unpack(f"<{n_parts}i", rec[44:44 + 4 * n_parts])) + [n_points]
        base = 44 + 4 * n_parts
        pts = struct.unpack(f"<{2 * n_points}d", rec[base:base + 16 * n_points])
        for i in range(n_parts):
            yield [(pts[2 * k], pts[2 * k + 1]) for k in range(starts[i], starts[i + 1])]


def main(src: str, out: str) -> None:
    kinds = {"ridges": "ridge", "transform": "transform", "trenches": "trench"}
    rows, seg_id = [], 0
    for suffix, label in kinds.items():
        for shp in sorted(pathlib.Path(src).glob(f"*{suffix}*.shp")):
            for seg in polylines(shp):
                for lon, lat in seg:
                    rows.append((round(lon, 4), round(lat, 4), seg_id, label))
                seg_id += 1
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lon", "lat", "segment", "kind"])
        w.writerows(rows)
    kb = pathlib.Path(out).stat().st_size / 1024
    print(f"{seg_id} segments, {len(rows)} vertices -> {out} ({kb:.0f} KB)")
    print("plot with:  for _, g in df.groupby('segment'): plt.plot(g.lon, g.lat)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
