"""Derive one grading spec per week from the week's own solution notebook.

Hand-writing thirteen specs invites drift, and a marker that does not match its notebook
fails silently at grading time. Everything here is read out of the notebook and checked.
Week 1's spec is hand-tuned and is left alone.
"""

import json
import glob
import os
import re

MARK = re.compile(r'✓[^{"\']*')
NB_DIR = "docs/notebooks"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTOGRADER_TOTAL = 85
HYGIENE = 5
FIGURE = 5


def cells_of(path):
    return json.load(open(path, encoding="utf-8"))["cells"]


def source(cell):
    s = cell.get("source", "")
    return s if isinstance(s, str) else "".join(s)


def homework_index(cells):
    for i, c in enumerate(cells):
        if c["cell_type"] == "markdown" and "## Homework" in source(c):
            return i
    raise SystemExit("no '## Homework' heading")


def released_map(stem):
    """Source of every code cell as the student receives it, keyed by its stable cell id.

    did_work() compares against this, so a checkpoint block that arrives complete is not
    mistaken for the student's own writing.
    """
    path = f"{NB_DIR}/{stem}.ipynb"
    out = {}
    for c in cells_of(path):
        if c["cell_type"] == "code" and c.get("id"):
            out[c["id"]] = source(c)
    if not out:
        raise SystemExit(f"{stem}: released notebook has no cell ids to key on")
    return out


def label_for(marker):
    body = marker[1:].strip()
    return body.split("—")[0].strip() or body[:40]


def split_points(total, n):
    """code is a flat 5 a part; whatever is left is the self-check, remainder to the front."""
    code = [5] * n
    pool = total - 5 * n
    base, extra = divmod(pool, n)
    return [base + (1 if i < extra else 0) for i in range(n)], code


def build(path):
    stem = os.path.basename(path).replace("_solution.ipynb", "")
    cells = cells_of(path)
    hw = homework_index(cells)

    markers, figures = [], 0
    for c in cells[hw:]:
        if c["cell_type"] != "code":
            continue
        m = MARK.search(source(c))
        if m:
            markers.append(m.group(0).strip())
        figures += sum(1 for o in c.get("outputs", [])
                       if any(k.startswith("image/") for k in o.get("data", {})))

    if not markers:
        raise SystemExit(f"{stem}: no self-check markers after the homework heading")

    # every marker must name exactly one code cell, and that cell must be in the homework
    for mk in markers:
        hits = [i for i, c in enumerate(cells)
                if c["cell_type"] == "code" and mk in source(c)]
        if len(hits) != 1:
            raise SystemExit(f"{stem}: marker {mk!r} matches {len(hits)} cells, need exactly 1")
        if hits[0] < hw:
            raise SystemExit(f"{stem}: marker {mk!r} is in the class section, not the homework")

    wants_figures = figures > 0
    pool = AUTOGRADER_TOTAL - HYGIENE - (FIGURE if wants_figures else 0)
    checks, codes = split_points(pool, len(markers))

    parts = []
    for i, mk in enumerate(markers):
        part = {"name": f"Part {i + 1} — {label_for(mk)}",
                "marker": mk,
                "check_points": checks[i],
                "code_points": codes[i]}
        if wants_figures and i == len(markers) - 1:
            part["figures"] = 1
            part["figure_points"] = FIGURE
        parts.append(part)

    spec = {"week": int(stem[:2]), "notebook_stem": stem,
            "homework_heading": "## Homework", "parts": parts,
            "hygiene": {"points": HYGIENE}, "released": released_map(stem)}

    total = sum(p["check_points"] + p["code_points"] + p.get("figure_points", 0)
                for p in parts) + HYGIENE
    assert total == AUTOGRADER_TOTAL, f"{stem}: points total {total}"
    return stem, spec


if __name__ == "__main__":
    for path in sorted(glob.glob(f"{NB_DIR}/*_solution.ipynb")):
        stem, spec = build(path)
        if spec["week"] == 1:
            out = os.path.join(OUT_DIR, "spec_week01.json")
            existing = json.load(open(out, encoding="utf-8"))
            existing["released"] = released_map(stem)
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(existing, fh, indent=2, ensure_ascii=False)
            print(f"{stem:<28} hand-tuned points kept · released map refreshed")
            continue
        out = os.path.join(OUT_DIR, f"spec_week{spec['week']:02d}.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(spec, fh, indent=2, ensure_ascii=False)
        figs = "yes" if any("figures" in p for p in spec["parts"]) else "no"
        print(f"{stem:<28} {len(spec['parts'])} parts · figures {figs} · {os.path.basename(out)}")
