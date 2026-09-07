#!/usr/bin/env python
"""Re-emit the generated week-summary cell in place, without rebuilding the notebook.

`week_cheatsheet()` reads course.yml and modules.yml, so editing either — a takeaway, a `does:`,
the grouping — makes every built summary stale, and check_notebook says so for all thirteen weeks
at once. The honest fix is a rebuild, but a rebuild re-executes every cell, refetches live data,
churns cell ids and invalidates the Gradescope specs keyed to them. That is a large blast radius
for a markdown cell that is generated from the plan and contains no student work.

So this swaps that ONE cell, in both the student and solution copies, and touches nothing else.

    python tools/refresh_summaries.py           # show which weeks are stale
    python tools/refresh_summaries.py --apply

It REFUSES a frozen week (course.yml `frozen_until`). Students hold that notebook and the deployed
Gradescope bundle stores its cell sources; the repair goes on the week's rebuild branch instead.
"""
import argparse, datetime, json, pathlib, re, sys, yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
COURSE = yaml.safe_load((ROOT / "course.yml").read_text())
NB = ROOT / COURSE["platform"]["notebook_dir"]


def summary_index(cells, n):
    return next((k for k, c in enumerate(cells) if c["cell_type"] == "markdown"
                 and re.search(rf"(?im)^##\s*Week {n} summary", "".join(c["source"]))), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    today = datetime.date.today()
    changed = skipped = 0

    for s in COURSE["schedule"]:
        n, slug = s["n"], s.get("slug")
        if not slug:
            continue
        frozen = s.get("frozen_until")
        want = weekkit.week_cheatsheet(n)
        files = [p for p in (NB / f"{slug}.ipynb", NB / f"{slug}_solution.ipynb") if p.exists()]
        if not files:
            continue

        stale = []
        for f in files:
            nb = json.loads(f.read_text())
            i = summary_index(nb["cells"], n)
            if i is not None and "".join(nb["cells"][i]["source"]) != want:
                stale.append((f, nb, i))
        if not stale:
            continue

        if frozen and today <= frozen:
            print(f"  SKIP  week {n} — FROZEN until {frozen}; students hold it and the deployed "
                  f"bundle stores its cell sources. Repair on the rebuild branch.")
            skipped += 1
            continue

        print(f"  {'week ' + str(n):<8} {len(stale)} file(s) stale")
        changed += 1
        if args.apply:
            for f, nb, i in stale:
                # Preserve the file's OWN serialisation. A three-line change must produce a
                # three-line diff: writing with different options — `—` for `\u2014`, a trailing
                # newline, nbformat's current defaults — rewrites every line of a 900-line file
                # and buries the edit. The committed notebooks escape non-ASCII; the ones this
                # .venv's nbformat writes do not, so detect it per file rather than assuming.
                text = f.read_text()
                ascii_only = not any(ord(ch) > 127 for ch in text)
                nb["cells"][i]["source"] = want.splitlines(keepends=True)
                out = json.dumps(nb, indent=1, ensure_ascii=ascii_only)
                f.write_text(out + ("\n" if text.endswith("\n") else ""))

    if not (changed or skipped):
        print("every week summary is current")
    elif not args.apply:
        print(f"\n{changed} week(s) would be rewritten — pass --apply")
    else:
        print(f"\nrewrote {changed} week(s); {skipped} skipped as frozen")


if __name__ == "__main__":
    main()
