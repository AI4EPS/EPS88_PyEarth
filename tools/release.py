#!/usr/bin/env python
"""Publish one notebook, or one answer key, on the day it is due to go out.

    python tools/release.py 02_liquid_water              # the notebook: show what would change
    python tools/release.py 02_liquid_water --apply      # do it, then commit by hand
    python tools/release.py 01_birthquake --solution     # the answer key, the Wednesday after

Everything under docs/notebooks/ is gitignored so that the default is hidden and releasing is
deliberate — a week is built weeks ahead and edited in the days before it is taught. This is the
deliberate act, in one command instead of four half-remembered ones:

  1. records the slug in course.yml's policy.notebooks_released (or solutions_released with
     --solution) — the single record, which
  2. tools/make_mkdocs.py reads to un-exclude that file and put it in the site nav, and
  3. tools/check_release.py reads to stop calling it an accidental commit, then
  4. `git add -f` stages it past .gitignore.

It stages and stops. It does NOT commit and does NOT push: what reaches 46 students is Weiqiang's
click, and a script that pushed would be one typo away from releasing the wrong week.

TIMING. For a SOLUTION: policy.late runs to Wednesday 23:59 and says nothing is accepted once
solutions post, so run it at the END of Wednesday. A morning release hands the answers to students
who can still submit for credit that evening.

GRADESCOPE. Releasing a week's notebook is also when its autograder should go up — the two are
one act from a student's point of view. This prints the command; it does not run it, because
deploying puts marking in front of the class.
"""
import argparse, pathlib, re, subprocess, sys, yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB = "docs/notebooks"


def run(*a, **kw):
    return subprocess.run(a, cwd=ROOT, capture_output=True, text=True, **kw)


def known_slugs(course):
    slugs = {s["slug"]: f"week {s['n']}" for s in course["schedule"]}
    slugs.update({p["slug"]: "practice" for p in course.get("practice", [])})
    for p in sorted((ROOT / NB).glob("T*_*.ipynb")):
        if not p.stem.endswith("_solution"):
            slugs[p.stem] = "project track"
    return slugs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug", help="notebook slug, e.g. 02_liquid_water")
    ap.add_argument("--solution", action="store_true",
                    help="release the answer key instead of the notebook")
    ap.add_argument("--apply", action="store_true", help="make the change (default: show it)")
    args = ap.parse_args()
    what = "solution" if args.solution else "notebook"
    key = "solutions_released" if args.solution else "notebooks_released"
    suffix = "_solution.ipynb" if args.solution else ".ipynb"

    text = (ROOT / "course.yml").read_text()
    course = yaml.safe_load(text)
    slugs = known_slugs(course)

    if args.slug not in slugs:
        sys.exit(f"unknown slug {args.slug!r}. Known: " + ", ".join(sorted(slugs)))

    target = ROOT / NB / f"{args.slug}{suffix}"
    if not target.exists():
        sys.exit(f"{target.relative_to(ROOT)} does not exist — build the week before releasing it")
    if args.solution and args.slug not in course["policy"].get("notebooks_released", []):
        sys.exit(f"{args.slug}'s notebook is not released yet — an answer key with no question "
                 f"in front of it is not a release. Run without --solution first.")

    released = list(course["policy"].get(key, []))
    if args.slug in released:
        print(f"{args.slug} is already in policy.{key} — nothing to record")
    else:
        released.append(args.slug)

    # Targeted edit. course.yml is mostly comments, and a yaml round-trip would drop every one
    # of them; this touches the single line and leaves the file otherwise byte-identical.
    pat = re.compile(rf"^(\s*{key}:\s*)\[[^\]]*\]\s*$", re.M | re.S)
    if not pat.search(text):
        sys.exit(f"could not find a `{key}: [...]` block in course.yml")
    # Re-wrap rather than emitting one very long line: by December this list is twenty slugs,
    # and course.yml is a file a person reads.
    def rewrap(m):
        head = m.group(1)
        indent = " " * (len(head) - len(head.lstrip("\n")) + len(head.split(":")[0]) + 2)
        lines, cur = [], head + "["
        for i, slug in enumerate(released):
            piece = slug + ("," if i < len(released) - 1 else "]")
            if len(cur) + len(piece) + 1 > 96 and not cur.endswith("["):
                lines.append(cur.rstrip())
                cur = indent + piece + " "
            else:
                cur += piece + " "
        lines.append(cur.rstrip())
        return "\n".join(lines)

    new_text = pat.sub(rewrap, text, count=1)

    print(f"release the {what} for {args.slug} ({slugs[args.slug]})")
    print(f"  course.yml   {key}: [{', '.join(released)}]")
    print(f"  git add -f   {NB}/{args.slug}{suffix}")
    print(f"  regenerate   mkdocs.yml, docs/README.md")

    if not args.apply:
        print("\n(dry run — pass --apply to make these changes)")
        return

    (ROOT / "course.yml").write_text(new_text)
    for gen in ("make_mkdocs.py", "make_docs.py"):
        r = run(sys.executable, f"tools/{gen}")
        if r.returncode:
            sys.exit(f"tools/{gen} failed:\n{r.stderr}")
        print("  " + r.stdout.strip())

    r = run("git", "add", "-f", f"{NB}/{args.slug}{suffix}")
    if r.returncode:
        sys.exit(f"git add -f failed: {r.stderr.strip()}")
    run("git", "add", "course.yml", "mkdocs.yml", "docs/README.md")

    r = run("git", "status", "--short")
    print("\nstaged:\n" + "\n".join("  " + l for l in r.stdout.splitlines() if l.strip()))
    print(f'\nNothing is committed or pushed. When you are ready:\n'
          f'  git commit -m "Release the {slugs[args.slug]} {what}"\n'
          f'  git push')
    if not args.solution and args.slug in {s["slug"] for s in course["schedule"]}:
        week = next(s["n"] for s in course["schedule"] if s["slug"] == args.slug)
        print(f'\nThe autograder goes up with it — students meet the notebook and its marking\n'
              f'together. Check it, then deploy:\n'
              f'  python tools/gradescope_deploy.py --mark {week}')


if __name__ == "__main__":
    main()
