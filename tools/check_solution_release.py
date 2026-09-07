#!/usr/bin/env python
"""Guard the answer keys: what is tracked must match what course.yml says is released.

Solutions are gitignored by default (.gitignore line 2), so the only way one reaches main is a
deliberate `git add -f` — which is what tools/release_solution.py does on release day. This is
what makes that deliberate rather than possible: a tracked solution nobody recorded is an
accidental commit, and by the time anyone notices it is in the history and on the site.

Replaces the inline shell in .github/workflows/check.yml, which could only ever say "none", and
so had to be hand-edited on the first release of the term and every one after it.

    python tools/check_solution_release.py

Exits non-zero on a mismatch. Run from anywhere; paths are resolved against the repo.
"""
import pathlib, subprocess, sys, yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB = "docs/notebooks"


def tracked_solutions():
    r = subprocess.run(["git", "ls-files", f"{NB}/*_solution.ipynb"],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"git ls-files failed: {r.stderr.strip()}")
    return sorted(pathlib.Path(p).name[: -len("_solution.ipynb")]
                  for p in r.stdout.split() if p)


def main():
    course = yaml.safe_load((ROOT / "course.yml").read_text())
    released = list(course["policy"].get("solutions_released", []))
    tracked = tracked_solutions()

    errors, warnings = [], []

    # The one that matters: an answer key on main that nobody recorded as released.
    for slug in tracked:
        if slug not in released:
            errors.append(
                f"{NB}/{slug}_solution.ipynb is TRACKED but not in policy.solutions_released.\n"
                f"        If this is release day, run: python tools/release_solution.py {slug} --apply\n"
                f"        If not, it is an accidental commit: git rm --cached {NB}/{slug}_solution.ipynb")

    # The other direction. Listed but missing entirely is a broken record; listed and built but
    # not yet committed is just work in progress, and saying so every run would train people to
    # ignore this check.
    for slug in released:
        if slug in tracked:
            continue
        if (ROOT / NB / f"{slug}_solution.ipynb").exists():
            warnings.append(f"{slug} is listed as released and built, but not committed yet — "
                            f"students have nothing until it is on main")
        else:
            errors.append(
                f"policy.solutions_released lists {slug}, but {NB}/{slug}_solution.ipynb does not "
                f"exist.\n        Build it, or take the slug off the list.")

    for w in warnings:
        print(f"  warn  {w}")
    for e in errors:
        print(f"  FAIL  {e}")
    if errors:
        sys.exit(1)
    print(f"solution release OK — {len(tracked)} tracked, "
          f"{len(released)} listed as released ({', '.join(released) or 'none'})")


if __name__ == "__main__":
    main()
