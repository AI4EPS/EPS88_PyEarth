#!/usr/bin/env python
"""Shared week-building kit: the standards list, and the closing summary generator.

STANDARDS is the single source for both. `agent_brief.py` renders it flat for a builder and as
three gated tiers for a reviewer, so a notebook cannot satisfy one and fail the other. Keep it
here; do not restate it in a prompt or in TEMPLATE.md.

`week_cheatsheet()` builds a week's closing summary from `course.yml` and `modules.yml`, so a
takeaway or a definition has one wording across the whole course. Import it; do not reimplement
it per week.
"""
import pathlib, yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent

# tier 1 = is it TRUE, and is it a valid notebook — gates the rest
# tier 2 = is it well made (figures and code)
# tier 3 = does it teach (judgement)
STANDARDS = [
    # The three truth checks lead tier 1 deliberately: a notebook that teaches something false
    # must not ship however well made it is, and a reviewer that stops early must hit these first.
    (1, "EVERY EARTH-SCIENCE CLAIM IS ONE A SPECIALIST IN THAT FIELD WOULD ACCEPT — the mechanism "
        "and the interpretation, not only the arithmetic. A number can be computed correctly and "
        "the physics around it still be wrong"),
    (1, "no claim outruns its evidence: a correlation is not called a cause, and a result from one "
        "region or one year is not stated as general"),
    (1, "any constant taken from the literature rather than computed is cited and named as a "
        "convention, not presented as derived"),
    (1, "structure is present and in order, and the week summary sits before the homework"),
    (1, "every question is a markdown block asking it, then its own answer cell"),
    (1, "every answer cell looks identical wherever it appears, so a student scrolling finds them all"),
    (1, "the notebook states how many places the student writes, split between class and home"),
    (1, "every question has a complete model answer in the solution, prose questions included"),
    (1, "every self-check runs for a student who used only the names the prompt gave, and can fail"),
    (1, "the solution executes clean on a fresh kernel"),
    (1, "only the six libraries; the standard library is not a loophole"),
    (1, "nothing arrives before its week except in a setup cell flagged 'Coming later'"),
    (1, "no AI-disclosure cell, no EXTENSION or Buffer banner, no marker comments, no offering dates"),
    (1, "the cached CSV this week needs is written into data/"),
    (2, "every claim in prose is supported by the output or figure beside it"),
    (2, "nothing is asserted that was not computed in this session, except a cited published constant"),
    (2, "every map draws data/coastlines.csv; no map is dots in a blank rectangle"),
    (2, "every figure has axis labels with units and its sample size in the title"),
    (2, "no plot cell carries lines that do not change what the reader learns"),
    (2, "the code is clean enough to be imitated: plainest form that works, names that read "
        "as English, one job per cell and per function, nothing dead, nothing repeated "
        "that a function could say once"),
    (2, "45-60 cells and 7-9 questions (5-6 in class, 2-3 homework); at most two "
         "answered in prose \u2014 this is a data-science course"),
    (3, "every question serves a stated takeaway or a teaches: item, and you can say which"),
    (3, "the week contains a genuine surprise, and the predict-before-you-run cell catches a wrong "
        "guess a real student would make"),
    (3, "each method appears only after something simpler has visibly failed at the same task"),
    (3, "the first question is a win and the last is hard"),
    (3, "the homework asks a question class did not answer — its first part cannot be done by "
        "copying a class cell and changing one string"),
    (3, "the science is a real question, not a toy with a known answer"),
    (3, "a student who finished could answer the week's question to a friend, and say why they "
        "believe it"),
]

TIER_NAMES = {1: "is it true, and is it a valid notebook?",
              2: "is it well made?",
              3: "does it teach?"}

# Every week's setup cell does the same job: plot defaults, and data from the live source with a
# fallback to the cache. Four trial builds wrote four versions of these twenty lines. Format this
# with the week's own url/cache/unpack and the shape stays constant across the course.
# A week may run one query or eleven. The nullary load() an earlier version of this template
# carried could express only the first, so week 1 had to hand-write a setup cell that matched
# its shape — which is exactly the drift this template exists to prevent. {signature} and
# {url_expr} keep the shape identical while the parameters vary: pass "" and a literal URL for
# a one-query week, or a parameter list and an f-string for a week that loads several.
SETUP_CELL = """import pandas as pd
import matplotlib.pyplot as plt

# house style, set once, so every plot cell below holds only what matters
plt.rcParams.update({{"figure.figsize": {figsize}, "figure.dpi": 110,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True}})

CACHE = "{cache_base}"

def load({signature}):
    \"\"\"{docstring}\"\"\"
    try:
        return pd.read_csv({url_expr})
    except Exception as e:
        print("live source unreachable, using the cached copy:", type(e).__name__)
        return pd.read_csv(CACHE + "/" + {cache_expr})

{unpack}
"""


def stop_list():
    """The builder's view: the same standards, marked with the tier that grades them."""
    mark = {1: "[must]", 2: "[craft]", 3: "[teaching]"}
    return "\n".join(f"- {mark[tier]} {text}" for tier, text in STANDARDS)


def tiers():
    """The reviewer's view: the same standards, grouped and gated."""
    out = []
    for t in (1, 2, 3):
        out.append(f"**TIER {t} — {TIER_NAMES[t]}**")
        out += [f"- {text}" for tier, text in STANDARDS if tier == t]
        out.append("")
    return "\n".join(out)


def week_cheatsheet(week_n, module_ids=None):
    """The notebook's closing summary, generated from course.yml and modules.yml.

    Built from the plan rather than typed into the notebook, so a takeaway or a definition has
    one wording that cannot drift between weeks.

    module_ids defaults to the week's own modules. Pass it only to narrow that set, and it is
    checked: an id the catalogue does not have raises rather than silently producing an empty
    section, and an id that is not part of this week raises too.
    """
    course = yaml.safe_load((ROOT / "course.yml").read_text())
    mods = yaml.safe_load((ROOT / "modules.yml").read_text())
    wk = next(s for s in course["schedule"] if s["n"] == week_n)
    by_id = {m["id"]: m for m in mods["modules"]}

    if module_ids is None:
        module_ids = wk["modules"]
    unknown = [m for m in module_ids if m not in by_id]
    if unknown:
        raise ValueError(f"week {week_n}: no such module(s) in modules.yml: {unknown}")
    stray = [m for m in module_ids if m not in wk["modules"]]
    if stray:
        raise ValueError(f"week {week_n} teaches {wk['modules']}, not {stray}")

    out = [f"## Week {week_n} summary", "",
           f"**The question.** {wk['question']}", "", "### What to remember", "",
           "| | |", "|---|---|"]
    for i, tk in enumerate(wk.get("takeaways", []), 1):
        out.append(f"| **{i}** | {tk} |")

    ideas = [d for d in mods.get("plain_words", []) if d["module"] in module_ids]
    if ideas:
        out += ["", "### The ideas, in plain words", "", "| Idea | Means |", "|---|---|"]
        out += [f"| **{d['idea']}** | {d['words']} |" for d in ideas]

    fns = [f for mid in module_ids for f in by_id.get(mid, {}).get("functions", [])]
    if fns:
        out += ["", "### Code you met this week", "", "| Function | What it does |", "|---|---|"]
        out += [f"| `{f['name']}` | {f['does']} |" for f in fns]
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    print(tiers() if len(sys.argv) > 1 and sys.argv[1] == "tiers" else stop_list())


def gate(week_n, variant=""):
    """The two gates a build must pass. Called at the END of every build_weekNN.py.

    A build that has not passed these is not a build, so this is not a step anyone can forget
    to run: it lives in the build script and exits non-zero. Judging whether the notebook is
    GOOD is a separate job, done by an agent that did not write it.
    """
    import json, pathlib, subprocess, sys
    root = pathlib.Path(__file__).resolve().parent.parent
    slug = next(s["slug"] for s in yaml.safe_load((root / "course.yml").read_text())["schedule"]
                if s["n"] == week_n)
    sol = root / f"docs/notebooks{variant}" / f"{slug}_solution.ipynb"
    bad = []

    cells = json.loads(sol.read_text())["cells"]
    counts = [c["execution_count"] for c in cells
              if c["cell_type"] == "code" and c.get("execution_count")]
    if not counts:
        bad.append("the solution has no execution counts — it was never executed")
    elif counts[0] != 1:
        bad.append(f"execution counts start at {counts[0]}, not 1 — a cell was run and deleted, "
                   f"so these outputs are not what the shipped code produces")
    elif counts != list(range(1, len(counts) + 1)):
        bad.append("execution counts are not contiguous — the solution was executed piecemeal")
    if any(o.get("output_type") == "error" for c in cells for o in c.get("outputs", [])):
        bad.append("the solution contains an error output — it does not execute clean")

    cmd = [sys.executable, str(root / "tools/check_notebook.py"), str(week_n)]
    if variant:
        cmd += ["--variant", variant]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    print(r.stdout.rstrip())
    if r.returncode:
        bad.append("check_notebook reported errors (above)")

    # Nothing else can catch a week reaching forward: a single-week reviewer does not know what
    # week 9 is supposed to introduce, and the weeks are built in parallel.
    if not variant:
        r2 = subprocess.run([sys.executable, str(root / "tools/check_prior_knowledge.py"),
                             str(week_n)], capture_output=True, text=True, cwd=root)
        print(r2.stdout.rstrip())
        if r2.returncode:
            bad.append("the week uses something the course has not taught yet (above)")

    if bad:
        print("\nBUILD REJECTED:")
        for b in bad:
            print(f"  - {b}")
        sys.exit(1)
    print("\ngates passed: executes clean, checker OK, nothing used before its week")
