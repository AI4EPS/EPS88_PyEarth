#!/usr/bin/env python
"""Prove the Gradescope autograder scores what it claims to, on every week.

Written after the grader was found to be unrunnable and, once running, to give 95/100 to a
submission that deleted every assert and pasted a fabricated tick. A grader nobody has attacked
is a formality; these are the attacks.

    python tools/selftest_gradescope.py        # every built week, plus the corner cases
    python tools/selftest_gradescope.py 3      # just week 3's calibration
"""
import json, os, pathlib, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
GS = ROOT / "tools" / "gradescope"
fails = []
# One week, for weekkit.gate() to run at the end of a build. A FRESHLY GENERATED spec can still
# be ungradeable -- that is the failure that scored a model answer 40/100 -- so regenerating is
# not the same as checking. With a week number the corner cases and container tests are skipped:
# they are about the bundle machinery, not about this week, and a build must stay quick.
ONLY = int(sys.argv[1]) if len(sys.argv) > 1 else None


def run(bundle, files):
    """Grade a submission made of {filename: bytes} and return (score, max, payload)."""
    with tempfile.TemporaryDirectory() as tmp:
        sub = pathlib.Path(tmp) / "sub"; sub.mkdir()
        for name, data in files.items():
            (sub / name).write_bytes(data)
        res = pathlib.Path(tmp) / "r" / "results.json"
        subprocess.run([sys.executable, str(bundle / "grade.py")], capture_output=True,
                       env={**os.environ, "GS_SUBMISSION": str(sub), "GS_RESULTS": str(res),
                            "GS_SPEC": str(bundle / "spec.json")})
        r = json.loads(res.read_text())
        if "tests" not in r:
            return r.get("score", 0), None, r
        return (sum(t["score"] for t in r["tests"]),
                sum(t["max_score"] for t in r["tests"]), r)


def check(label, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}{'' if cond else '   ' + detail}")
    if not cond:
        fails.append(label)


def tamper(nb_bytes):
    """A submission that strips every assert and fakes the tick output."""
    nb = json.loads(nb_bytes)
    k = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if "✓" in s and "assert" in s:
            c["source"] = [l + "\n" for l in s.splitlines()
                           if not l.strip().startswith("assert")]
            c["execution_count"] = 1
            k += 1
            c["outputs"] = [{"output_type": "stream", "name": "stdout",
                             "text": [f"✓ faked {k}\n"]}]
    return json.dumps(nb).encode()


print("per-week calibration")
for d in sorted(GS.glob("week*/")):
    spec = json.loads((d / "spec.json").read_text())
    stem, n = spec["notebook_stem"], spec["week"]
    if ONLY is not None and n != ONLY:
        continue
    sol = ROOT / "docs/notebooks" / f"{stem}_solution.ipynb"
    stu = ROOT / "docs/notebooks" / f"{stem}.ipynb"
    if not sol.exists():
        print(f"  --    week {n}: no solution on disk, skipped")
        continue
    got, mx, _ = run(d, {f"{stem}.ipynb": sol.read_bytes()})
    check(f"week {n:>2}: the model answer scores full marks", got == mx, f"{got}/{mx}")
    got, mx, _ = run(d, {f"{stem}.ipynb": stu.read_bytes()})
    check(f"week {n:>2}: an untouched notebook scores zero", got == 0, f"scored {got}")

w1 = GS / "week01"
stem = json.loads((w1 / "spec.json").read_text())["notebook_stem"]
_sol_path = ROOT / "docs/notebooks" / f"{stem}_solution.ipynb"
if not _sol_path.exists():
    # Solutions are gitignored until release, so a public checkout — CI's — has none. Every
    # assertion below is about how a WORKED notebook scores, and there is no worked notebook
    # here to score. Skipping is honest; failing would be a checker complaining that the
    # release policy is in force.
    print("\ncorner cases skipped: no solution notebooks in this checkout")
    print(f"\n{'all checks pass' if not fails else str(len(fails)) + ' FAILED'}")
    sys.exit(1 if fails else 0)

if ONLY is not None:
    # Scoped to one week for a build gate: everything below is about the BUNDLE machinery
    # (attacks, submission shapes, container isolation), which no single week's rebuild
    # can change. The full run still covers them.
    print(f"\n{'all checks pass' if not fails else str(len(fails)) + ' FAILED'} (week {ONLY} only)")
    sys.exit(1 if fails else 0)

print("\ncorner cases, on week 1")
sol = _sol_path.read_bytes()

got, mx, r = run(w1, {f"{stem}.ipynb": tamper(sol)})
check("a stripped self-check with a faked tick scores almost nothing", got <= 10, f"scored {got}")

got, _, r = run(w1, {"homework.pdf": b"%PDF-1.4\n"})
check("a PDF is refused with an instruction, not a crash",
      got == 0 and ".ipynb" in r.get("output", ""))

got, _, r = run(w1, {})
check("an empty submission is refused", got == 0 and ".ipynb" in r.get("output", ""))

got, _, r = run(w1, {f"{stem}.ipynb": b"{not json"})
check("a corrupt notebook is refused by name",
      got == 0 and "could not be read" in r.get("output", ""))

other = ROOT / "docs/notebooks" / "05_clustered_or_random_solution.ipynb"
if other.exists():
    got, _, r = run(w1, {"05_clustered_or_random.ipynb": other.read_bytes()})
    check("the wrong week is named as the wrong week",
          got == 0 and "wrong notebook" in r.get("output", "").lower())

print("\nhow students actually submit")

def submit(files):
    """files: {relative path: bytes}. Returns (points, payload)."""
    with tempfile.TemporaryDirectory() as tmp:
        sub = pathlib.Path(tmp) / "sub"
        for rel, data in files.items():
            p = sub / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        sub.mkdir(exist_ok=True)
        res = pathlib.Path(tmp) / "r" / "results.json"
        subprocess.run([sys.executable, str(w1 / "grade.py")], capture_output=True,
                       env={**os.environ, "GS_SUBMISSION": str(sub), "GS_RESULTS": str(res),
                            "GS_SPEC": str(w1 / "spec.json")})
        r = json.loads(res.read_text())
        return (sum(x["score"] for x in r["tests"]) if "tests" in r else 0), r

FULL = 100
pts, _ = submit({f"EPS88_PyEarth/docs/notebooks/{stem}.ipynb": sol})
check("a zipped folder, notebook nested inside it", pts == FULL, f"scored {pts}")

pts, _ = submit({f"{stem}.ipynb": sol,
                 f".ipynb_checkpoints/{stem}-checkpoint.ipynb": sol})
check("a DataHub .ipynb_checkpoints copy sitting beside it", pts == FULL, f"scored {pts}")

pts, _ = submit({"EPS88 homework 1 FINAL.ipynb": sol})
check("a notebook the student renamed is still graded", pts == FULL, f"scored {pts}")

other = ROOT / "docs/notebooks" / "05_clustered_or_random.ipynb"
if other.exists():
    pts, _ = submit({f"{stem}.ipynb": sol, "05_clustered_or_random.ipynb": other.read_bytes()})
    check("this week plus another week: the named one wins", pts == FULL, f"scored {pts}")
    pts, r = submit({"05_clustered_or_random.ipynb": other.read_bytes()})
    check("...but a different week ALONE is refused by name",
          pts == 0 and "wrong notebook" in r.get("output", "").lower())

print("\nthe bundle is self-contained (Gradescope runs it with nothing else present)")
with tempfile.TemporaryDirectory() as tmp:
    box = pathlib.Path(tmp) / "source"; box.mkdir()
    for name in ("grade.py", "run_autograder", "setup.sh", "spec.json"):
        (box / name).write_bytes((w1 / name).read_bytes())
    sub = pathlib.Path(tmp) / "submission"; sub.mkdir()
    (sub / f"{stem}.ipynb").write_bytes(sol)
    res = pathlib.Path(tmp) / "results" / "results.json"
    # cwd inside the copied bundle, and nothing from this repo importable
    p = subprocess.run([sys.executable, "grade.py"], capture_output=True, cwd=box,
                       env={"PATH": os.environ["PATH"], "GS_SUBMISSION": str(sub),
                            "GS_RESULTS": str(res)})
    ok = res.exists() and sum(x["score"] for x in json.loads(res.read_text())["tests"]) == FULL
    check("the bundle alone grades a full-marks submission", ok,
          p.stderr.decode()[:160])

with tempfile.TemporaryDirectory() as tmp:
    sub = pathlib.Path(tmp) / "sub"; sub.mkdir()
    (sub / f"{stem}.ipynb").write_bytes(sol)
    res = pathlib.Path(tmp) / "r" / "results.json"
    subprocess.run([sys.executable, str(w1 / "grade.py")], capture_output=True,
                   env={**os.environ, "GS_SUBMISSION": str(sub), "GS_RESULTS": str(res),
                        "GS_SPEC": "/nonexistent-spec.json"})
    r = json.loads(res.read_text())
    check("a misconfigured autograder writes a result and blames itself, not the student",
          r.get("score") == 0 and "not your fault" in r.get("output", ""))

print(f"\n{'all checks pass' if not fails else str(len(fails)) + ' FAILED'}")
sys.exit(1 if fails else 0)
