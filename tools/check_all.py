#!/usr/bin/env python
"""Run every check, in the order that fails cheapest first.

There was no single entry point: validating the repo meant remembering four commands and which
of them take a week number. That is the shape of a check nobody runs.

    python tools/check_all.py
"""
import pathlib, subprocess, sys, yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable
course = yaml.safe_load((ROOT / "course.yml").read_text())


def run(label, *args):
    r = subprocess.run([PY, str(ROOT / "tools" / args[0]), *args[1:]],
                       capture_output=True, text=True, cwd=ROOT)
    tail = [l for l in r.stdout.strip().split("\n") if l.strip()][-1:] or [""]
    print(f"  {'ok  ' if r.returncode == 0 else 'FAIL'}  {label:<34} {tail[0][:64]}")
    if r.returncode:
        print("\n".join("        " + l for l in (r.stdout + r.stderr).strip().split("\n")[-12:]))
    return r.returncode


def generated_is_current():
    """Every generated file matches what its generator would emit right now.

    docs/README.md sat on main with an EMPTY week table on the morning of the first class: the
    notebooks were moved aside for a test, make_docs.py correctly emptied the table, and nobody
    re-ran it when they came back. A generated file that is not regenerated is stale, and only a
    comparison catches it.
    """
    stale = []
    for gen, out in (("make_docs.py", "docs/README.md"),
                     ("make_mkdocs.py", "mkdocs.yml"),
                     ("make_schedule.py", "SCHEDULE.md")):
        before = (ROOT / out).read_text()
        subprocess.run([PY, str(ROOT / "tools" / gen)], capture_output=True, cwd=ROOT)
        if (ROOT / out).read_text() != before:
            stale.append(out)
            # PUT IT BACK. This check regenerates in place to compare, so on a file someone has
            # edited by hand it silently destroys the edit — which is exactly what it did to
            # Weiqiang's changes to docs/README.md, before anyone could read what they were.
            # Reporting "stale" is this function's whole job; overwriting is a side effect of
            # how it measures, and the measurement must not cost the user their work.
            (ROOT / out).write_text(before)
    # Generators with MANY outputs. The Gradescope specs and the marking sheets are both read
    # out of the notebooks, so a week that is rebuilt silently invalidates them — and a stale
    # spec grades against self-checks the notebook no longer has. Same failure as the empty
    # README, one directory wider, so it is checked the same way.
    for gen, tree, pattern in (("make_gradescope.py", "tools/gradescope", "week*/spec.json"),
                               ("make_rubrics.py", "tools/gradescope", "week*/rubric.md")):
        d = ROOT / tree
        before = {p: p.read_bytes() for p in sorted(d.glob(pattern))}
        subprocess.run([PY, str(ROOT / "tools" / gen)], capture_output=True, cwd=ROOT)
        after = {p: p.read_bytes() for p in sorted(d.glob(pattern))}
        if before != after:
            stale.append(f"{tree}/{pattern}")
            for p, b in before.items():          # put the hand edits back, as above
                p.write_bytes(b)

    print(f"  {'ok  ' if not stale else 'FAIL'}  {'generated files are current':<34} "
          f"{'' if not stale else 'stale: ' + ', '.join(stale)}")
    return 1 if stale else 0


bad = 0
bad += run("the plan", "check_course.py")
bad += generated_is_current()
bad += run("the checkers themselves", "selftest_checks.py")
bad += run("the autograder", "selftest_gradescope.py")
bad += run("the track checker", "selftest_track.py")

built = [s for s in course["schedule"] if s["modules"]
         and (ROOT / "docs/notebooks" / f"{s['slug']}.ipynb").exists()]
if not built:
    print("  --    no notebooks built yet")
for s in built:
    bad += run(f"week {s['n']} notebook", "check_notebook.py", str(s["n"]))
    bad += run(f"week {s['n']} prior knowledge", "check_prior_knowledge.py", str(s["n"]))

# Project tracks. They are notebooks like a week, but keyed by track id rather than week
# number, so they need their own checker — see tools/check_track.py for what carries over
# unchanged, what needs a track reading, and what is meaningless for a notebook that has no
# homework boundary and deliberately stops asserting after the load.
import glob
built_tracks = sorted({pathlib.Path(p).name.split("_")[0]
                       for p in glob.glob(str(ROOT / "docs/notebooks/T*_*.ipynb"))
                       if "_solution" not in p})
for tid in built_tracks:
    bad += run(f"track {tid}", "check_track.py", tid)
if not built_tracks:
    print("  --    no project tracks built yet")

print("OK" if not bad else f"{bad} check(s) failed")
sys.exit(1 if bad else 0)
