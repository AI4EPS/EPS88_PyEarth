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
    # (tier, one testable claim, enforced_by_a_checker)
    #
    # One assertion each, no rationale and no history — TEMPLATE.md carries the reasoning and a
    # builder reads it once, while this list gets read on every build and every review. An [auto]
    # item is settled by check_notebook.py or check_prior_knowledge.py; a reader reports what they
    # said and spends their attention on the rest. Items without a checker came from
    # notes/defects.yml, which is where the ones that keep happening are recorded.

    # --- tier 1: is it TRUE, and is it a valid notebook? -------------------------------------
    (1, "every Earth-science claim is one a specialist would accept — the mechanism, not only the "
        "arithmetic", False),
    (1, "no claim outruns its evidence: no cause from a correlation, no general result from one "
        "region or one year", False),
    (1, "before concluding anything from one window of a catalogue, compute the neighbouring "
        "windows", False),
    (1, "every literature constant is cited with its value and the date it was read, and named as "
        "a convention", False),
    (1, "every network read goes through the same cached fallback", False),
    (1, "every cached fallback URL this week reads actually resolves", False),
    (1, "every question has a complete model answer in the solution, prose questions included",
        False),
    (1, "the sections run in TEMPLATE 1's order, each opening with the state it needs", False),
    (1, "the answer-cell convention holds throughout: a markdown ask, its own answer cell, "
        "identical stubs", True),
    (1, "the opening cell is weekkit.OPENING with only the question and the hook changed", True),
    (1, "the week summary sits before the homework, and the opening cells carry the DataHub link",
        True),
    (1, "every self-check uses only names the prompt gave, and can fail", True),
    (1, "the solution executes clean on a fresh kernel", True),
    (1, "only the six libraries", True),
    (1, "nothing arrives before its week except in a setup cell flagged 'Coming later'", True),
    (1, "each plain_words idea this week introduces appears in its recorded wording, verbatim",
        True),
    (1, "no AI-disclosure cell, buffer banner, marker comment or offering date", True),

    # --- tier 2: is it WELL MADE? -------------------------------------------------------------
    (2, "every prose claim is supported by the output beside it, and nothing is asserted that was "
        "not computed", False),
    (2, "every proper noun in a model answer is a value some cell printed", False),
    (2, "no plot cell carries a line that does not change what the reader learns", False),
    (2, "the code is plain enough to imitate: simplest form that works, English names, one job per "
        "cell", False),
    (2, "the axis labels carry units, and every figure title carries its sample size", False),
    (2, "built to the target of 50 cells and 8 questions, not to the 60/9 ceiling", False),
    (2, "every map draws data/coastlines.csv", True),
    (2, "every figure has axis labels", True),
    (2, "nothing dead: no commented-out code, no unused import, no block said three times", True),

    # --- tier 3: does it TEACH? ---------------------------------------------------------------
    (3, "every question serves a named takeaway or teaches: item", False),
    (3, "the summary lists what a student must remember, not everything the notebook touched",
        True),
    (3, "every takeaway is exercised by a question, and the homework touches all of them", False),
    (3, "the week's question is answered FOR THE STUDENT, on their own data or their own choice",
        False),
    (3, "no worked cell prints the answer to the question below it", False),
    (3, "no answer is stated in the prose under its own question", False),
    (3, "a section heading names the territory, never the finding", False),
    (3, "the week contains a genuine surprise", False),
    (3, "each method arrives only after something simpler has visibly failed — N/A if the week "
        "teaches no method", False),
    (3, "the first class question is a win; the hardest thing in the week is its last homework "
        "part", False),
    (3, "the homework asks what class did not; part 1 is not a class cell with one string changed",
        False),
    (3, "an 'explain' part makes them produce evidence, not restate the summary", False),
    (3, "the science is a real question, even where the answer is known and checkable", False),
    (3, "at least one '### Predict before you run' cell, inviting a guess a real student gets "
        "wrong", True),
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
OPENING = """# {question}

**EPS 88 · PyEarth.** Open your own copy on DataHub: [click here]({datahub}).

{hook}

Every place you write something opens with a pencil icon and the words *Your turn*, and is
followed by an empty cell. Fill them all in, then export the notebook as a PDF and upload that.

Two habits from the first minute. A cell runs when you press **Shift+Enter**, and the notebook
remembers everything it has already run — so when something breaks and you cannot see why,
**Kernel → Restart Kernel and Run All Cells** throws the memory away and rebuilds it from the
top. That is never the wrong thing to do.
"""
# Only {question} and {hook} change between weeks. Everything else a student reads on the way in
# — where to click, how to submit, what a pencil means, how to recover — is the same in week 13
# as in week 1, so it is written once here rather than re-invented thirteen times.

# A self-check ends with this line and nothing else. The shape is the point: submissions are
# PDFs read in SpeedGrader, and a stable "✓ <label> — " prefix is what lets a grader find the
# same fact in 46 documents. Every week uses it, so what is learned reading week 3 transfers.
CHECK_LINE = '✓ {label} — {summary}'

# Rebuilding the state a section needs, for anyone who arrived late or restarted the kernel.
# Silent by design: it is scaffolding, not a result.
CHECKPOINT = """# ── Checkpoint ── run this if you restarted the kernel or fell behind ──
{body}"""

CLOSING_HEADING = "## The question, answered"

SETUP_CELL = """{imports}import pandas as pd
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
    return "\n".join(f"- {mark[tier]}{' [auto]' if auto else ''} {text}"
                     for tier, text, auto in STANDARDS)


def tiers():
    """The reviewer's view: the same standards, grouped and gated."""
    out = []
    for t in (1, 2, 3):
        out.append(f"**TIER {t} — {TIER_NAMES[t]}**")
        out += [f"- {'[auto] ' if auto else ''}{text}"
                for tier, text, auto in STANDARDS if tier == t]
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

    # `remember: false` marks a call the notebook legitimately uses but a student never needs
    # again — set_aspect on a map, integer tick locators. check_prior_knowledge still counts them
    # as taught; the summary does not, because a summary listing everything teaches nothing.
    fns = [f for mid in module_ids for f in by_id.get(mid, {}).get("functions", [])
           if f.get("remember", True)]
    if fns:
        out += ["", "### Code you met this week", "", "| Function | What it does |", "|---|---|"]
        out += [f"| `{f['name']}` | {f['does']} |" for f in fns]
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    print(tiers() if len(sys.argv) > 1 and sys.argv[1] == "tiers" else stop_list())


def _course():
    return yaml.safe_load((pathlib.Path(__file__).resolve().parent.parent / "course.yml").read_text())


def modules_upto(week_n, inclusive):
    """Module ids taught up to week_n. `inclusive` is the whole difference between the two
    callers: the prior-knowledge CONTRACT is what a student knows BEFORE this week (exclusive),
    the prior-knowledge CHECK is what the week may legally use (inclusive)."""
    return [m for s in _course()["schedule"]
            if (s["n"] <= week_n if inclusive else s["n"] < week_n)
            for m in s["modules"]]


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
