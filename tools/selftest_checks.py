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
CODE = lambda s: {"cell_type": "code", "source": [s]}


def run(fn, cells, *a):
    C.errs.clear(); C.warns.clear()
    fn(cells, *a)
    return list(C.errs)


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
    ("a self-check with no conventional result line", C.check_conventions,
     [MD("## The question, answered"), CODE("assert x > 0\nprint('nice work')")], True),
    ("...but not one that uses weekkit.CHECK_LINE", C.check_conventions,
     [MD("## The question, answered"), CODE("assert x > 0\nprint(f'✓ Q3 — you found 14')")], False),
    ("a notebook with no Predict cell", C.check_predict, [MD("## 1. Something")], True),
    ("...but not one that has the conventional heading", C.check_predict,
     [MD("### Predict before you run\n\nHow many do you expect?")], False),
]

bad = 0
for name, fn, cells, should_fire in CASES:
    fired = bool(run(fn, cells))
    ok = fired == should_fire
    bad += not ok
    print(f"  {'ok  ' if ok else 'DEAD'}  {name}")
    if not ok:
        print(f"        expected {'a failure' if should_fire else 'silence'}, got the opposite")
print(f"{len(CASES) - bad}/{len(CASES)} rules behave as documented")
sys.exit(1 if bad else 0)
