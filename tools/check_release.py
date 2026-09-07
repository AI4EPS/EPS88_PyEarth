#!/usr/bin/env python
"""Guard what is public: what is tracked must match what course.yml says is released.

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


def tracked(suffix):
    r = subprocess.run(["git", "ls-files", f"{NB}/*{suffix}"],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"git ls-files failed: {r.stderr.strip()}")
    names = [pathlib.Path(p).name for p in r.stdout.split() if p]
    if suffix == ".ipynb":                       # the notebook glob also matches the keys
        names = [n for n in names if not n.endswith("_solution.ipynb")]
    return sorted(n[: -len(suffix)] for n in names)


ARTIFACTS = [("notebook", "notebooks_released", ".ipynb"),
             ("solution", "solutions_released", "_solution.ipynb")]


def main():
    course = yaml.safe_load((ROOT / "course.yml").read_text())
    errors, warnings, counts = [], [], []

    for what, key, suffix in ARTIFACTS:
        released = list(course["policy"].get(key, []))
        on_main = tracked(suffix)
        counts.append(f"{len(on_main)} {what}(s) tracked, {len(released)} listed")

        # The one that matters: something on main that nobody recorded as released.
        for slug in on_main:
            if slug not in released:
                errors.append(
                    f"{NB}/{slug}{suffix} is TRACKED but not in policy.{key}.\n"
                    f"        If this is release day:  python tools/release.py {slug}"
                    f"{' --solution' if what == 'solution' else ''} --apply\n"
                    f"        If not, it is an accidental commit: "
                    f"git rm --cached {NB}/{slug}{suffix}")

        # The other direction. Listed but missing entirely is a broken record; listed and built
        # but not committed is work in progress, and saying so every run trains people to ignore
        # this check.
        for slug in released:
            if slug in on_main:
                continue
            if (ROOT / NB / f"{slug}{suffix}").exists():
                warnings.append(f"{slug} is listed as a released {what} and built, but not "
                                f"committed — students have nothing until it is on main")
            else:
                errors.append(f"policy.{key} lists {slug}, but {NB}/{slug}{suffix} does not "
                              f"exist.\n        Build it, or take the slug off the list.")

    for w in warnings:
        print(f"  warn  {w}")
    for e in errors:
        print(f"  FAIL  {e}")
    if errors:
        sys.exit(1)
    print("release state OK — " + "; ".join(counts))


if __name__ == "__main__":
    main()
