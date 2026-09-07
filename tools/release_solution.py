#!/usr/bin/env python
"""Publish one answer key, on the Wednesday its week is due for release.

    python tools/release_solution.py 01_birthquake            # show what would change
    python tools/release_solution.py 01_birthquake --apply    # do it, then commit by hand

Solutions are gitignored so that the default is hidden and releasing is deliberate. This is the
deliberate act, in one command instead of four half-remembered ones:

  1. records the slug in course.yml's policy.solutions_released — the single record, which
  2. tools/make_mkdocs.py reads to un-exclude that file and put it in the site nav, and
  3. tools/check_solution_release.py reads to stop calling it an accidental commit, then
  4. `git add -f` stages it past .gitignore.

It stages and stops. It does NOT commit and does NOT push: what reaches 46 students is Weiqiang's
click, and a script that pushed would be one typo away from releasing the wrong week.

TIMING. policy.late runs to Wednesday 23:59 and policy.late says nothing is accepted once
solutions post, so run this at the END of Wednesday. A morning release hands the answers to
students who can still submit for credit that evening.
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
    ap.add_argument("slug", help="notebook slug, e.g. 01_birthquake")
    ap.add_argument("--apply", action="store_true", help="make the change (default: show it)")
    args = ap.parse_args()

    text = (ROOT / "course.yml").read_text()
    course = yaml.safe_load(text)
    slugs = known_slugs(course)

    if args.slug not in slugs:
        sys.exit(f"unknown slug {args.slug!r}. Known: " + ", ".join(sorted(slugs)))

    sol = ROOT / NB / f"{args.slug}_solution.ipynb"
    if not sol.exists():
        sys.exit(f"{sol.relative_to(ROOT)} does not exist — build the week before releasing it")

    released = list(course["policy"].get("solutions_released", []))
    if args.slug in released:
        print(f"{args.slug} is already in policy.solutions_released — nothing to record")
    else:
        released.append(args.slug)

    # Targeted edit. course.yml is mostly comments, and a yaml round-trip would drop every one
    # of them; this touches the single line and leaves the file otherwise byte-identical.
    pat = re.compile(r"^(\s*solutions_released:\s*)\[[^\]]*\]\s*$", re.M)
    if not pat.search(text):
        sys.exit("could not find a `solutions_released: [...]` line in course.yml")
    new_text = pat.sub(lambda m: f"{m.group(1)}[{', '.join(released)}]", text, count=1)

    print(f"release {args.slug} ({slugs[args.slug]})")
    print(f"  course.yml   solutions_released: [{', '.join(released)}]")
    print(f"  git add -f   {NB}/{args.slug}_solution.ipynb")
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

    r = run("git", "add", "-f", f"{NB}/{args.slug}_solution.ipynb")
    if r.returncode:
        sys.exit(f"git add -f failed: {r.stderr.strip()}")
    run("git", "add", "course.yml", "mkdocs.yml", "docs/README.md")

    r = run("git", "status", "--short")
    print("\nstaged:\n" + "\n".join("  " + l for l in r.stdout.splitlines() if l.strip()))
    print(f'\nNothing is committed or pushed. When you are ready:\n'
          f'  git commit -m "Release the {slugs[args.slug]} solution"\n'
          f'  git push')


if __name__ == "__main__":
    main()
