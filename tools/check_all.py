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

    THE MEASUREMENT MUST NOT COST ANYONE THEIR WORK. There is no way to ask a generator what it
    WOULD emit, so this function regenerates in place and puts back what it found — and a plain
    restore is only safe in a quiet tree. Under concurrent writes (five track fixers were running
    when a peer session pointed this out) the "before" snapshot goes stale the moment someone
    else writes, and restoring it reverts their work: the check would destroy exactly what it
    exists to protect, which is what it already did once to Weiqiang's hand edits to
    docs/README.md, arriving from the other direction.

    So the restore is CONDITIONAL: put back the snapshot only if what is on disk right now is
    byte-for-byte what this function's own regeneration produced. If it is anything else,
    somebody wrote between the two reads and their bytes stay — reported, never overwritten.
    """
    stale, kept = [], []

    def restore(path, before, mine):
        """Undo our regeneration, unless someone else has written since."""
        if path.read_bytes() != mine:
            kept.append(str(path.relative_to(ROOT)))   # not ours to put back
            return
        path.write_bytes(before)

    for gen, out in (("make_docs.py", "docs/README.md"),
                     ("make_mkdocs.py", "mkdocs.yml"),
                     ("make_schedule.py", "SCHEDULE.md")):
        f = ROOT / out
        before = f.read_bytes()
        subprocess.run([PY, str(ROOT / "tools" / gen)], capture_output=True, cwd=ROOT)
        mine = f.read_bytes()
        if mine != before:
            stale.append(out)
            restore(f, before, mine)
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
            for f, b in before.items():
                restore(f, b, after.get(f, b))

    note = "" if not stale else "stale: " + ", ".join(stale)
    print(f"  {'ok  ' if not stale else 'FAIL'}  {'generated files are current':<34} {note}")
    if kept:
        print(f"        NOT restored — changed by something else while this check ran, so the "
              f"newer bytes were left alone: {', '.join(kept)}")
    return 1 if stale else 0



def main():
    """The whole suite. Behind a guard so that a caller — the Gradescope deploy tool wants
    `generated_is_current()` as a preflight — can import one function without running all of it
    as an import side effect, and without paying for a subprocess to avoid that."""
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
    import datetime
    for s in built:
        # A week frozen between its release and its deadline cannot be repaired without breaking
        # grading for students who already hold it. Its defects are still PRINTED — hiding them
        # would be worse than a red build — but they do not fail the run, so a failure still means
        # something new broke rather than something known and dated.
        fu = s.get("frozen_until")
        if isinstance(fu, str):
            fu = datetime.date.fromisoformat(fu)
        # INCLUSIVE of the date itself. `<` lifted the freeze at 00:00 on 2026-09-09 while
        # students can still submit until 23:59 that day -- and on that last day everyone
        # still submitting is a LATE submitter, holding the old notebook. A peer session
        # comparing its own deploy gate against this one found it.
        frozen = fu and datetime.date.today() <= fu
        a = run(f"week {s['n']} notebook", "check_notebook.py", str(s["n"]))
        b = run(f"week {s['n']} prior knowledge", "check_prior_knowledge.py", str(s["n"]))
        if frozen:
            if a or b:
                print(f"        ^ week {s['n']} is FROZEN until {fu} and these are not counted — "
                      f"students hold it; the repair is on branch week01-v2")
        else:
            bad += a + b

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
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
