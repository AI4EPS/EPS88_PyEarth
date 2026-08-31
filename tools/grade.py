#!/usr/bin/env python
"""Grade a submitted notebook against the released one, without running anything.

The submission already carries the evidence: the student ran their notebook, so every self-check
either produced an error or did not. Re-executing would need network, would take 46 x 13 runs, and
would fail for reasons that are not the student's — a slow endpoint, a rate limit. Reading is
enough, and it is deterministic.

Cells are matched by their nbformat id, so inserted or deleted cells do not shift the mapping.

    python tools/grade.py 1 submission.ipynb
    python tools/grade.py 1 submissions/*.ipynb --json results.json
"""
import argparse, json, pathlib, re, sys, yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
course = yaml.safe_load((ROOT / "course.yml").read_text())
STUB = re.compile(r"your answer here", re.I)
PROSE_STUB = "Double-click"


class Unreadable(Exception):
    """A submission we cannot grade at all — say why, in words a student can act on."""


def cells_of(path):
    """Read a submission defensively. Students upload the wrong file, export from other tools,
    and occasionally upload something that is not a notebook at all."""
    raw = pathlib.Path(path).read_bytes()
    if not raw.strip():
        raise Unreadable("the file is empty")
    if raw[:4] == b"%PDF":
        raise Unreadable("this is a PDF — upload the .ipynb notebook file itself")
    try:
        nb = json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError as e:
        raise Unreadable(f"this is not a readable notebook file ({e.msg})")
    if not isinstance(nb, dict) or "cells" not in nb:
        raise Unreadable("this file has no notebook cells in it")
    return [c for c in nb["cells"] if isinstance(c, dict) and "cell_type" in c]


def src(c):
    return "".join(c.get("source", []))


def released(week):
    slug = next(s["slug"] for s in course["schedule"] if s["n"] == week)
    return cells_of(ROOT / "docs/notebooks" / f"{slug}.ipynb")


def grade(week, submission):
    key_cells = released(week)
    sub_cells = cells_of(submission)
    key = {c["id"]: c for c in key_cells if c.get("id")}
    sub = {}
    for c in sub_cells:                       # first id wins: a duplicated cell must not shadow
        if c.get("id") and c["id"] not in sub:
            sub[c["id"]] = c
    order = [c["id"] for c in key_cells if c.get("id")]

    graded = [i for i in order if i.endswith(("answer", "check", "prose"))]
    # A notebook whose ids all name a DIFFERENT week is the wrong file, not a bad attempt.
    other = {i.split("-")[0] for c in sub_cells if (i := c.get("id", "")).startswith("w")}
    if other and f"w{week:02d}" not in other:
        raise Unreadable(f"this looks like {'/'.join(sorted(other))}, not week {week:02d} — "
                         f"check which notebook you uploaded")
    matched = sum(1 for i in graded if i in sub)
    if graded and matched < len(graded) * 0.5:
        # Some export paths rewrite or drop cell ids. Fall back to position among cells of the
        # same role, which is right as long as the student did not reorder the notebook.
        by_role = {}
        for c in sub_cells:
            s = "".join(c.get("source", []))
            role = ("check" if c["cell_type"] == "code" and "assert " in s else
                    "answer" if c["cell_type"] == "code" else
                    "prose" if c["cell_type"] == "markdown" else None)
            by_role.setdefault(role, []).append(c)
        seen = {}
        for cid in graded:
            role = cid.rsplit("-", 1)[1]
            n = seen.get(role, 0); seen[role] = n + 1
            if n < len(by_role.get(role, [])):
                sub[cid] = by_role[role][n]
    results, label = [], None

    for cid in order:
        k, s = key[cid], sub.get(cid)
        text = src(k)
        if k["cell_type"] == "markdown" and re.search(r"(?m)^\s*(#{1,4}\s*)?✏️", text):
            label = re.sub(r"[#*✏️\s]+", " ", text.split("\n")[0]).strip()[:48]
            continue

        if k["cell_type"] == "code" and STUB.search(text):
            if s is None:
                results.append((label, 0, "that cell is missing from your notebook"))
            elif not [ln for ln in src(s).split("\n")
                      if ln.strip() and not ln.strip().startswith("#")]:
                results.append((label, 0, "nothing was written here"))
            elif s.get("execution_count") is None:
                results.append((label, 0, "you wrote code but never ran the cell"))
            else:
                results.append((label, 1, "answered"))

        elif k["cell_type"] == "markdown" and PROSE_STUB in text:
            wrote = s is not None and PROSE_STUB not in src(s) and len(src(s).strip()) > 40
            results.append((label, 1 if wrote else 0,
                            "answered — read it against the rubric" if wrote else "no paragraph written"))

        elif k["cell_type"] == "code" and "assert " in text:
            if s is None:
                results.append((f"{label} · self-check", 0, "the self-check cell was deleted"))
            elif src(s).strip() != text.strip():
                results.append((f"{label} · self-check", 0, "the self-check cell was edited"))
            elif s.get("execution_count") is None:
                results.append((f"{label} · self-check", 0, "never run"))
            else:
                err = next((o for o in s.get("outputs", []) if o.get("output_type") == "error"), None)
                msg = str(err.get("evalue", ""))[:70] if err else "passed"
                results.append((f"{label} · self-check", 0 if err else 1, msg))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("week", type=int)
    ap.add_argument("submissions", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    out = {}
    for sub in a.submissions:
        try:
            r = grade(a.week, sub)
        except Unreadable as e:
            print(f"\n{pathlib.Path(sub).name}  —  0, not gradeable: {e}")
            out[pathlib.Path(sub).name] = {"score": 0, "max": 0, "items": [
                {"name": "submission", "score": 0, "output": str(e)}]}
            continue
        got, total = sum(s for _, s, _ in r), len(r)
        print(f"\n{pathlib.Path(sub).name}  —  {got}/{total}")
        for label, s, why in r:
            print(f"  {'PASS' if s else 'FAIL'}  {str(label):<34} {why}")
        out[pathlib.Path(sub).name] = {"score": got, "max": total,
                                       "items": [{"name": l, "score": s, "output": w} for l, s, w in r]}
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
