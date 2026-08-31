#!/usr/bin/env python
"""Prove the Gradescope autograder scores what it claims to, on every week.

Written after the grader was found to be unrunnable and, once running, to give 95/100 to a
submission that deleted every assert and pasted a fabricated tick. A grader nobody has attacked
is a formality; these are the attacks.

    python tools/selftest_gradescope.py
"""
import json, os, pathlib, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
GS = ROOT / "tools" / "gradescope"
fails = []


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
