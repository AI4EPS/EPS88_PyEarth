#!/usr/bin/env python
"""Prove every rule check_track.py adds still fires on the defect that motivated it.

Same contract as selftest_checks.py, and for the same reason: two rules in this project were DEAD
when written. Each rule below carries the smallest input that must trip it, and one that must not.

check_track's other rules are check_notebook's, unchanged, and selftest_checks.py already covers
them. Only the track-specific ones are here.

    python tools/selftest_track.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import check_notebook as C
import check_track as T

MD = lambda s: {"cell_type": "markdown", "source": [s]}
CODE = lambda s: {"cell_type": "code", "source": [s]}
OUT = lambda s, t: {"cell_type": "code", "source": [s],
                    "outputs": [{"output_type": "stream", "text": [t]}]}

TRACK = {"id": "TX", "open_question": "Numbers the plan recalled. Is the thing knowable at all?"}
STUB = CODE("# ← your answer here\n")
LOAD = CODE("assert len(d) == 10\nprint('✓ the data — 10 rows')")
Q = MD("### ✏️ Your turn 1\n\nFind something.")
NET = MD("### And that is the last self-check in this notebook\n\nThe safety net stops here.")
FIVE = MD("one-sentence answer baseline split by structure what I got wrong AI disclosure")


def run(fn, *a):
    C.errs.clear(); C.warns.clear()
    fn(*a)
    return list(C.errs)


def leak(cells, sol_cells):
    """check_closing_does_not_answer is the one rule here that WARNS rather than errors, and the
    CASES loop reads `errs`. Promote its warnings so "did it fire?" is asked the same way for
    every rule -- the alternative is a second loop that would itself need proving."""
    C.warns.clear()
    T.check_closing_does_not_answer(cells, sol_cells)
    C.errs.extend(C.warns)


def reveal(cells, sol_cells):
    """Warns, like the closing rule, so promote the warnings for the shared CASES loop."""
    C.warns.clear()
    T.check_reveal_does_not_answer(cells, sol_cells)
    C.errs.extend(C.warns)


def spine(cells):
    return run(C.check_spine, cells, T.TRACK_SPINE_SKIP)


def open_q(cells):
    return run(T.check_open_question, cells, TRACK)


PAD = [MD("filler")] * 6

CASES = [
    # --- the safety net stops at the loading step ---------------------------------------------
    ("an assert after the first question", T.check_scaffolding_stops,
     ([NET, LOAD, Q, CODE("assert chi > 19.7\nprint('✓')")], [2]), True),
    ("...but not the one assert on the load, before it", T.check_scaffolding_stops,
     ([NET, LOAD, Q, STUB], [2]), False),
    ("no self-check anywhere", T.check_scaffolding_stops, ([NET, Q, STUB], [1]), True),
    ("a notebook that never says the safety net stops", T.check_scaffolding_stops,
     ([LOAD, Q, STUB], [1]), True),

    # --- a track starts at solo ---------------------------------------------------------------
    ("an answer cell shipped with the model answer still in it", T.check_no_worked_answer,
     ([Q, CODE("# ← your answer here\nper_day = x.value_counts()")],
      [Q, CODE("# ← your answer here\nper_day = x.value_counts()")]), True),
    ("...but not one the build actually stubbed out", T.check_no_worked_answer,
     ([Q, STUB], [Q, CODE("# ← your answer here\nper_day = x.value_counts()")]), False),

    # --- the five required project sections ---------------------------------------------------
    ("a track missing the required project sections", T.check_required_sections,
     ([MD("## The open question")],), True),
    ("...but not one that heads all five", T.check_required_sections, ([FIVE],), False),

    # --- it does not close --------------------------------------------------------------------
    ("a track with no open question at all", open_q, ([MD("## The question, answered")],), True),
    ("an open question buried in the middle", open_q,
     ([MD("## The open question\n\nIs the thing knowable at all?")] + PAD + PAD,), True),
    ("an open question that is not the one the plan records", open_q,
     ([MD("## The open question\n\nWhy is July high?")],), True),
    ("...but not the plan's own question, last in the file", open_q,
     ([MD("## The open question\n\nIs the thing knowable at all?")],), False),

    # --- binding wording, checked the other way round -----------------------------------------
    ("a track that paraphrases a recorded plain-words sentence", T.check_bound_wording,
     ([MD("**Bootstrap:** resample the data over and over.")],), True),
    ("...but not one that quotes it verbatim", T.check_bound_wording,
     ([MD("**Bootstrap:** Ask the data the same question a thousand times, using a different "
          "random slice of itself each time.")],), False),
    ("...and not one that never introduces the idea", T.check_bound_wording,
     ([MD("We bootstrap by volcano.")],), False),

    # --- the spine, with a track's frame skipped rather than a week's -------------------------
    # check_spine is check_notebook's, and selftest_checks already proves the rule. What is
    # track-specific and therefore proven HERE is the skip list: a track's four questions sit
    # among seven frame headings a week does not have, and if TRACK_SPINE_SKIP misses one of
    # them the count comparison fails on every track at once. That is a silent-pass risk in the
    # other direction too — a skip list that swallowed a real section would hide a missing one.
    ("a track whose spine promises a section it does not ship", spine,
     ([MD("## What you'll be able to do\n\n1. A?\n2. B?\n3. C?"),
       MD("## A?"), MD("## B?"), MD("## The open question")],), True),
    # --- the closing must not answer the prompts ----------------------------------------------
    # A WARN rule, so `run` (which returns errs) cannot see it -- `leak` reads warns instead.
    ("a closing that prints a number from a stubbed cell's output", leak,
     ([Q, STUB, MD("## The question, answered\n\nThe gap is +0.222.")],
      [Q, OUT("print(gap)", "+0.222\n"), MD("x")]), True),
    ("...but not one that names the quantity instead of the value", leak,
     ([Q, STUB, MD("## The question, answered\n\nThe gap is the number you found.")],
      [Q, OUT("print(gap)", "+0.222\n"), MD("x")]), False),
    ("...and not a number the notebook GAVE, which is in no stub's output", leak,
     ([Q, CODE("print(1)"), MD("## The open question\n\n2348 traces carry a label.")],
      [Q, OUT("print(1)", "1\n"), MD("x")]), False),

    # --- a mid-notebook reveal must not answer the empty cell above it -----------------------
    ("a markdown reveal that states the answer to the stub above it", reveal,
     ([Q, STUB, MD("Nothing about volcanoes puts 390 eruptions on the 16th.")],
      [Q, OUT("print(peak)", "390\n"), MD("x")]), True),
    ("...but not one that names the finding without the number", reveal,
     ([Q, STUB, MD("Nothing about volcanoes distinguishes the 16th from the 15th.")],
      [Q, OUT("print(peak)", "390\n"), MD("x")]), False),

    ("...but not one where the frame headings are the only extras", spine,
     ([MD("## How this notebook is different"),
       MD("## What you'll be able to do\n\n1. A?\n2. B?\n3. C?"),
       MD("## Setup"), MD("## A?"), MD("## B?"), MD("## C?"),
       MD("## The question, answered"), MD("## What track TX leans on"),
       MD("## What your project must contain"), MD("## The open question")],), False),
]

bad = 0
for name, fn, args, should_fire in CASES:
    fired = bool(run(fn, *args))
    ok = fired == should_fire
    bad += not ok
    print(f"  {'ok  ' if ok else 'DEAD'}  {name}")
    if not ok:
        print(f"        expected {'a failure' if should_fire else 'silence'}, got the opposite")
print(f"{len(CASES) - bad}/{len(CASES)} track rules behave as documented")
sys.exit(1 if bad else 0)
