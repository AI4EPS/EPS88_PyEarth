#!/usr/bin/env python
"""Prove every rule in check_notebook.py still fires on the defect that motivated it.

Two of these rules were DEAD when written — they looked right and matched nothing — and three
had false positives. A checker nobody has seen fail is not evidence of anything, so each rule
below carries the smallest input that must trip it, and one that must not.

    python tools/selftest_checks.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import check_notebook as C

MD = lambda s: {"cell_type": "markdown", "source": [s]}
REPEAT = 'quakes = load("2019-07-01", "2019-12-31")\nscaled = StandardScaler().fit_transform(quakes[COLS])\nquakes["cluster"] = DBSCAN(eps=0.15).fit_predict(scaled)'
CODE = lambda s: {"cell_type": "code", "source": [s]}


def run(fn, cells, *a):
    C.errs.clear(); C.warns.clear()
    fn(cells, *a)
    return list(C.errs)


def run_warn(fn, cells, *a):
    """Some rules warn rather than error, and a warning nobody can test is a rule nobody can
    trust: the repeated-block rule fired on mandatory checkpoints for days, and it took two
    independent reviewers to notice, because this file could only see errs."""
    C.errs.clear(); C.warns.clear()
    fn(cells, *a)
    return list(C.warns)


# Week 2's modules list SYNTAX rather than functions, so the near-miss has to WRITE that syntax:
# `if / elif / else` and `None` can never be an ast.Name, and the call test rejected them all.
SYNTAX_WEEK = '''
def check(temperatures, albedo):
    """what it is for"""
    kept = []
    for i in range(len(temperatures)):
        if temperatures[i] is None or not albedo:
            kept.append(abs(albedo))
        elif temperatures[i] < 273 and albedo > 0:
            kept.append(0)
        else:
            kept.append(1)
    return kept, albedo


help(check)
plt.axvline(273)
plt.text(1, 2, "Venus")
'''

# A function the STUDENT writes cannot be rebuilt by a checkpoint without publishing the answer,
# so the checkpoint has to name it and send them back to their own cell. The stub is what marks
# the cell as theirs, which is why these four lists come in student/solution pairs.
_CHECKPOINT = "# ── Checkpoint ──\nvei = np.array([2, 3])"
_TOLD = "# ── Checkpoint ──\n# Re-run your own count_at_least cell too.\nvei = np.array([2, 3])"
_DEF = 'def count_at_least(values, level):\n    """How many are at or above the level."""\n    return (values >= level).sum()'
THEIR_STU = [CODE("quakes = load(URL)"), CODE("# ← your answer here"),
             CODE(_CHECKPOINT), CODE("print(count_at_least(vei, 2))")]
THEIR_SOL = [CODE("quakes = load(URL)"), CODE(_DEF),
             CODE(_CHECKPOINT), CODE("print(count_at_least(vei, 2))")]
TOLD_STU = [CODE("quakes = load(URL)"), CODE("# ← your answer here"),
            CODE(_TOLD), CODE("print(count_at_least(vei, 2))")]
TOLD_SOL = [CODE("quakes = load(URL)"), CODE(_DEF),
            CODE(_TOLD), CODE("print(count_at_least(vei, 2))")]

CASES = [
    ("assert names nothing declares", C.check_asserts,
     [MD("✏️ use `my_mags`"), CODE("assert len(other) > 0")], True),
    ("...and stays quiet when the prompt names it", C.check_asserts,
     [MD("✏️ use `my_mags`"), CODE("assert len(my_mags) > 0")], False),
    ("assert that cannot fail", C.check_asserts,
     [MD("✏️ q"), CODE("m = load('x&minmagnitude=4.5')"), CODE("assert max(m) >= 4.5")], True),
    ("...but not the one checking the floor was removed", C.check_asserts,
     [MD("✏️ q"), CODE("m = load('x&minmagnitude=4.5')"), CODE("assert min(m) < 4.5")], False),
    ("commented-out code", C.check_code_quality,
     [CODE("# old = pd.read_csv(u)\nx = 1")], True),
    ("...but not a prose comment", C.check_code_quality,
     [CODE("# house style, set once\nx = 1")], False),
    ("an import nothing uses", C.check_code_quality, [CODE("import json\nx = 1")], True),
    ("the same import twice", C.check_code_quality,
     [CODE("import numpy as np\nimport numpy as np\nnp.array([1])")], True),
    ("...but not one clean import", C.check_code_quality,
     [CODE("import numpy as np\nnp.array([1])")], False),
    ("a banned buffer sentence", C.check_banned, [MD("We may not get to this in class")], True),
    ("an AI-disclosure cell", C.check_banned, [MD("### AI disclosure")], True),
    ("a plot with no axis labels", C.check_figures, [CODE("plt.scatter(a, b)\nplt.show()")], True),
    ("a map without coastlines", C.check_figures,
     [CODE("plt.scatter(lons, lats)\nplt.xlabel('x')\nplt.ylabel('y')")], True),
    ("an import outside the six", C.check_imports, [CODE("import scipy")], True),
    ("a paraphrased plain-words sentence", lambda c: C.check_plain_words(c, 1),
     [MD("A catalogue only lists what instruments happened to record.")], True),
    ("...but not the recorded wording, verbatim", lambda c: C.check_plain_words(c, 1),
     [MD("A catalogue lists what somebody's instruments recorded, not what happened. Where there are no seismometers there are no earthquakes in the file.")], False),
    ("a summary listing a function the week never calls",
     lambda c: C.check_summary_is_this_week(c, 1), [CODE("print(1)")], True),
    ("a summary listing syntax the week never writes",
     lambda c: C.check_summary_is_this_week(c, 2), [CODE("print(1)")], True),
    ("...but not a week whose code writes all of it",
     lambda c: C.check_summary_is_this_week(c, 2), [CODE(SYNTAX_WEEK)], False),
    ("a self-check with no conventional result line", C.check_conventions,
     [MD("## The question, answered"), CODE("assert x > 0\nprint('nice work')")], True),
    ("...but not one that uses weekkit.CHECK_LINE", C.check_conventions,
     [MD("## The question, answered"), CODE("assert x > 0\nprint(f'✓ Q3 — you found 14')")], False),
    # The checkpoint rule, and its one exemption. A checkpoint may not rebuild a function the
    # STUDENT wrote — that publishes the answer — so naming it in a re-run instruction is the
    # only discharge, and it has to actually name it.
    ("a checkpoint that leaves the section reaching for a class variable",
     lambda c: C.check_checkpoints_rebuild(c, c),
     [CODE("quakes = load(URL)"), CODE("# ── Checkpoint ──\nvei = np.arange(3)"),
      CODE("print(vei, mag_levels)")], True),
    ("...but not one that rebuilds it", lambda c: C.check_checkpoints_rebuild(c, c),
     [CODE("quakes = load(URL)"),
      CODE("# ── Checkpoint ──\nvei = np.arange(3)\nmag_levels = np.arange(3)"),
      CODE("print(vei, mag_levels)")], False),
    ("a student's own function, with no instruction to re-run it",
     lambda c: C.check_checkpoints_rebuild(c, THEIR_SOL), THEIR_STU, True),
    ("...but not when the checkpoint names it and says to re-run it",
     lambda c: C.check_checkpoints_rebuild(c, TOLD_SOL), TOLD_STU, False),
    ("a spine whose questions outnumber the headings", C.check_spine,
     [MD(SP), MD("## Where do the earthquakes go?"), MD("## Homework")], True),
    ("...but not one where every question is a heading", C.check_spine,
     [MD(SP), MD("## Where do the earthquakes go?"), MD("## Why there?"),
      MD("## Homework")], False),
    ("a section heading that is a topic rather than a question", C.check_spine,
     [MD(SP), MD("## Where do the earthquakes go?"), MD("## Plate boundaries"),
      MD("## Homework")], True),
    ("a notebook with no spine at all", C.check_spine,
     [MD("## What you'll be able to do\n\nprose only."), MD("## Homework")], True),

    # WARNING rules, via run_warn.
    ("a self-check that is membership in a list the prompt dictated", C.check_weak_asserts,
     [CODE("bins = [3, 4, 5]\nassert answer in bins, 'pick one you tried'")], True, "warn"),
    ("a self-check that only catches doing nothing", C.check_weak_asserts,
     [CODE("assert weighted != plain, 'the weights were not used'")], True, "warn"),
    ("a self-check on a length the prompt fixed", C.check_weak_asserts,
     [CODE("assert len(slopes) == 5, 'five windows'")], True, "warn"),
    ("...but not a length compared against another object",
     C.check_weak_asserts,
     [CODE("assert len(drawn.get_offsets()) == len(volcanoes), 'wrong table'")], False, "warn"),
    ("...and not a check against a measured value", C.check_weak_asserts,
     [CODE("assert 0.5 < rate < 2.0, 'that is not a rate per year'")], False, "warn"),
    ("the same three lines in three ordinary cells", C.check_code_quality,
     [CODE(REPEAT), CODE(REPEAT), CODE(REPEAT)], True, "warn"),
    ("...but not when those cells are checkpoints, which MUST repeat",
     C.check_code_quality,
     [CODE("# ── Checkpoint ── run this if you restarted\n" + REPEAT)] * 3, False, "warn"),
    ("a notebook with no Predict cell", C.check_predict, [MD("## 1. Something")], True),
    ("...but not one that has the conventional heading", C.check_predict,
     [MD("### Predict before you run\n\nHow many do you expect?")], False),
]

bad = 0
for case in CASES:
    name, fn, cells, should_fire = case[:4]
    kind = case[4] if len(case) > 4 else "err"
    fired = bool((run_warn if kind == "warn" else run)(fn, cells))
    ok = fired == should_fire
    bad += not ok
    print(f"  {'ok  ' if ok else 'DEAD'}  {name}")
    if not ok:
        print(f"        expected {'a failure' if should_fire else 'silence'}, got the opposite")
print(f"{len(CASES) - bad}/{len(CASES)} rules behave as documented")
sys.exit(1 if bad else 0)
