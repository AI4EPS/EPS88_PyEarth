#!/usr/bin/env python
"""Track which Gradescope autograders are out of date, and package the ones that are.

The notebooks are still moving, and every rebuild can change a self-check marker, the released
scaffolding or the points. A bundle uploaded last week is then quietly grading against a notebook
that no longer exists. So nothing here trusts memory: each week's bundle is hashed, compared with
what was last uploaded, and only the weeks that actually differ are reported.

    python tools/gradescope_deploy.py              # what is stale, and where to upload it
    python tools/gradescope_deploy.py --zip        # also write the zips
    python tools/gradescope_deploy.py --mark 1 3   # record weeks 1 and 3 as uploaded

Gradescope has no public upload API and the account is behind CalNet SSO, so the upload itself
is a browser step. This makes it a short one: it names the exact weeks and their exact URLs.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import zipfile

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUNDLES = ROOT / "tools" / "gradescope"
STATE = BUNDLES / "deployed.json"
ZIPS = ROOT / "build" / "gradescope"
FILES = ("spec.json", "grade.py", "run_autograder", "setup.sh")
COURSE_URL = "https://www.gradescope.com/courses/1379264"


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"course_url": COURSE_URL, "weeks": {}}


def digest(week_dir):
    """One hash over everything that reaches the container."""
    h = hashlib.sha256()
    for name in FILES:
        p = week_dir / name
        if not p.exists():
            return None
        h.update(name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def contract_digest(week_dir):
    """Hash of only what must match the notebook a student is already holding.

    The grader refuses to credit a self-check whose assert lines differ from the released
    ones, so those lines are a contract with everybody who has already pulled the notebook.
    Changing them mid-week fails honest students from one side or the other, whichever way
    we jump. Points, names and grade.py carry no such promise and can go up at any time,
    so they are hashed separately and a grader-only change stays deployable.
    """
    spec_path = week_dir / "spec.json"
    if not spec_path.exists():
        return None
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    bound = {
        "notebook_stem": spec.get("notebook_stem"),
        "homework_heading": spec.get("homework_heading"),
        "released": spec.get("released"),
        "parts": [{"marker": p.get("marker"), "checks": p.get("checks")}
                  for p in spec.get("parts", [])],
    }
    blob = json.dumps(bound, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def frozen_until(week):
    """When this week's autograder may next change, read from course.yml.

    Deliberately NOT mirrored here. A freeze date is one fact, and the moment it lives in two
    files they drift — mine said the due date and course.yml said the late deadline, three days
    apart, which is three days of late submitters graded against asserts they do not have.
    course.yml is the record; this only reads it.
    """
    try:
        course = yaml.safe_load((ROOT / "course.yml").read_text())
    except Exception:
        return None
    for w in course.get("schedule", []):
        if w.get("n") == week:
            return w.get("frozen_until")
    return None


def preflight():
    """Run the repo's own checks before packaging anything for upload.

    Two failures this cannot see on its own: a bundle that matches HEAD but no longer matches
    the notebook it was generated from, and a notebook that no longer passes its own checks.
    check_all.py answers both, and its generated-files check restores what it regenerates, so
    running it costs nobody their hand edits. Deploy-time only — it is slow, and a status
    query should stay cheap.
    """
    print("  preflight: tools/check_all.py ...", flush=True)
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "check_all.py")],
                       capture_output=True, text=True, cwd=ROOT)
    lines = [l for l in r.stdout.strip().split("\n") if l.strip()]
    for l in lines:
        if "FAIL" in l or l.startswith("OK") or "check(s) failed" in l:
            print(f"    {l}")
    if r.returncode:
        print("    preflight FAILED — not packaging anything.")
    return r.returncode


def committed_state(week_dir):
    """Whether this bundle matches what is committed.

    The notebooks are rebuilt in the working tree and only committed once Weiqiang has
    approved them, so the tree can be ahead of main by work nobody has signed off. Uploading
    from it would put unapproved material in front of students, which is not a call this
    tool gets to make silently.
    """
    rel = week_dir.relative_to(ROOT)
    for name in FILES:
        path = f"{rel}/{name}"
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", path],
                                 cwd=ROOT, capture_output=True)
        if tracked.returncode != 0:
            return "untracked"
        diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", path],
                              cwd=ROOT, capture_output=True)
        if diff.returncode != 0:
            return "uncommitted"
    return "clean"


def week_dirs():
    return sorted(d for d in BUNDLES.glob("week*") if d.is_dir())


def make_zip(week_dir):
    ZIPS.mkdir(parents=True, exist_ok=True)
    out = ZIPS / f"autograder_{week_dir.name}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            z.write(week_dir / name, name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="store_true", help="write the zip for every stale week")
    ap.add_argument("--mark", nargs="*", type=int, metavar="WEEK",
                    help="record these weeks as uploaded at their current hash")
    ap.add_argument("--all", action="store_true", help="zip every week, stale or not")
    ap.add_argument("--no-rebuild", action="store_true",
                    help="skip make_gradescope.py and hash the bundles as they are")
    ap.add_argument("--allow-uncommitted", action="store_true",
                    help="zip a week whose bundle is not yet committed (unapproved work)")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="package without running check_all.py first (not advised)")
    args = ap.parse_args()

    if not args.no_rebuild:
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "make_gradescope.py")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("make_gradescope.py failed, so the bundles were not refreshed:\n" + r.stderr)
            return 1

    state = load_state()
    weeks = state.setdefault("weeks", {})

    if args.mark:
        for n in args.mark:
            d = BUNDLES / f"week{n:02d}"
            h = digest(d)
            if h is None:
                print(f"week {n}: no bundle to mark")
                continue
            rec = weeks.setdefault(str(n), {})
            rec["sha"] = h
            rec["contract"] = contract_digest(d)
            print(f"week {n}: recorded as uploaded at {h} (contract {rec['contract']})")
        STATE.write_text(json.dumps(state, indent=2) + "\n")
        return 0

    stale, unknown, current = [], [], []
    for d in week_dirs():
        n = int(d.name[4:])
        h = digest(d)
        if h is None:
            continue
        rec = weeks.get(str(n))
        if rec is None or not rec.get("assignment_id"):
            unknown.append((n, h, rec))
        elif rec.get("sha") != h:
            stale.append((n, h, rec, contract_digest(d)))
        else:
            current.append((n, h, rec))

    print(f"{len(current)} up to date · {len(stale)} stale · {len(unknown)} not yet in Gradescope\n")

    for n, h, rec, contract in stale:
        url = f"{COURSE_URL}/assignments/{rec['assignment_id']}/configure_autograder"
        breaking = rec.get("contract") is not None and rec["contract"] != contract
        frozen = frozen_until(n)
        kind = "BREAKING" if breaking else "SAFE    "
        print(f"  STALE   week {n:>2}  {rec.get('sha','?')} -> {h}   [{kind}]")
        if breaking:
            print(f"          the self-checks changed, so a student holding the old notebook "
                  f"would be marked against asserts they do not have.")
            print(f"          Deploy only once nobody is still working from the old copy.")
        else:
            print(f"          grader-only change; the student contract is unchanged, safe to deploy now.")
        if frozen:
            print(f"          FROZEN until {frozen} — do not upload before then.")
        print(f"          {url}")
    for n, h, rec in unknown:
        print(f"  MISSING week {n:>2}  {h}  (create the assignment, then --mark {n})")

    dirty = [d for d in week_dirs() if committed_state(d) != "clean"]
    if dirty:
        print("\n  NOT COMMITTED — rebuilt in the working tree and not yet approved:")
        for d in dirty:
            print(f"    {d.name}  ({committed_state(d)})")
        print("    Deploying these would put unapproved work in front of students.")

    if (args.zip or args.all) and not args.skip_preflight:
        if preflight():
            return 1

    if args.zip or args.all:
        todo = week_dirs() if args.all else [BUNDLES / f"week{n:02d}" for n, _, _, _ in stale] \
                                            + [BUNDLES / f"week{n:02d}" for n, _, _ in unknown]
        held = []
        for d in todo:
            if committed_state(d) != "clean" and not args.allow_uncommitted:
                held.append(d.name)
                continue
            print(f"  wrote {make_zip(d).relative_to(ROOT)}")
        if held:
            print(f"  HELD BACK (not committed): {', '.join(held)}")
            print("  Re-run with --allow-uncommitted only if the change is approved.")

    if not stale and not unknown:
        print("  every deployed autograder matches its bundle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
