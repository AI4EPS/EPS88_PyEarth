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
    import shutil, tempfile
    stale = []
    for gen, out in (("make_docs.py", "docs/README.md"),
                     ("make_mkdocs.py", "mkdocs.yml"),
                     ("make_schedule.py", "SCHEDULE.md")):
        before = (ROOT / out).read_text()
        subprocess.run([PY, str(ROOT / "tools" / gen)], capture_output=True, cwd=ROOT)
        if (ROOT / out).read_text() != before:
            stale.append(out)
    print(f"  {'ok  ' if not stale else 'FAIL'}  {'generated files are current':<34} "
          f"{'' if not stale else 'stale: ' + ', '.join(stale)}")
    return 1 if stale else 0


bad = 0
bad += run("the plan", "check_course.py")
bad += generated_is_current()
bad += run("the checkers themselves", "selftest_checks.py")

built = [s for s in course["schedule"] if s["modules"]
         and (ROOT / "docs/notebooks" / f"{s['slug']}.ipynb").exists()]
if not built:
    print("  --    no notebooks built yet")
for s in built:
    bad += run(f"week {s['n']} notebook", "check_notebook.py", str(s["n"]))
    bad += run(f"week {s['n']} prior knowledge", "check_prior_knowledge.py", str(s["n"]))

print("OK" if not bad else f"{bad} check(s) failed")
sys.exit(1 if bad else 0)
