#!/usr/bin/env python
"""What teaching a week taught us, and when it is safe to act on it.

A week cannot be edited between its release and its deadline: students hold the notebook,
nbgitpuller gives their copy priority on a merge, and the autograder stores the released
self-checks, so changing one fails whoever is on the other side of the change. But that is
exactly the window in which you learn what is wrong with it — standing in the room.

So observations go here at the moment they happen, each with the date it becomes safe to apply,
and this tool tells you what is due. Nothing is lost to "I will remember", and nothing is applied
while it would break a student's copy.

    python tools/teaching_log.py            # what is due now, and what is waiting
    python tools/teaching_log.py --all      # everything, including what has been applied

Entries live in ../notes/teaching-log.yml. Add them by hand — it is a text file on purpose.
"""
import argparse, datetime, pathlib, sys, yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG = ROOT.parent / "notes" / "teaching-log.yml"
course = yaml.safe_load((ROOT / "course.yml").read_text())
TODAY = datetime.date.today()

# A week is frozen from release until its homework deadline plus the accommodation window; the
# course releases solutions the Wednesday after, which is the first day nothing is in flight.
WEEK = {s["n"]: s for s in course["schedule"] if s.get("slug")}


def safe_from(n):
    """The Wednesday after week n's Sunday deadline — the day its notebook can move again."""
    taught = WEEK[n]["date"]
    if isinstance(taught, str):
        taught = datetime.date.fromisoformat(taught)
    sunday = taught + datetime.timedelta(days=(6 - taught.weekday()) % 7)
    return sunday + datetime.timedelta(days=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if not LOG.exists():
        sys.exit(f"no log yet — create {LOG}")
    entries = yaml.safe_load(LOG.read_text()) or []

    due, waiting, later, done = [], [], [], []
    for e in entries:
        if e.get("status") == "applied":
            done.append(e); continue
        when = e.get("lands", "after-deadline")
        if when == "next-offering":
            later.append(e)
        elif when == "now":
            due.append((e, None))
        else:
            d = safe_from(e["week"])
            (due if TODAY >= d else waiting).append((e, d))

    def show(rows, dated=True):
        for e, d in rows:
            when = f"  (was frozen until {d})" if dated and d else ""
            print(f"  week {e['week']:>2} · {e.get('taught', '?')}{when}")
            print(f"      seen : {e['observation']}")
            print(f"      do   : {e['change']}")

    print(f"DUE NOW — safe to apply ({len(due)})")
    show(due) if due else print("  nothing")
    print(f"\nWAITING — still frozen, students hold these ({len(waiting)})")
    for e, d in sorted(waiting, key=lambda x: x[1]):
        print(f"  week {e['week']:>2} · unblocks {d} ({(d - TODAY).days} days)")
        print(f"      {e['observation'][:96]}")
    if not waiting:
        print("  nothing")
    print(f"\nNEXT OFFERING — too big for this term ({len(later)})")
    for e in later:
        print(f"  week {e['week']:>2} · {e['observation'][:96]}")
    if not later:
        print("  nothing")
    if args.all and done:
        print(f"\nAPPLIED ({len(done)})")
        for e in done:
            print(f"  week {e['week']:>2} · {e['observation'][:96]}")
    print(f"\n{len(entries)} entr(ies). Add by editing {LOG.relative_to(ROOT.parent)}")


main()
