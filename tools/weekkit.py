#!/usr/bin/env python
"""Shared week-building kit: the standards list, and the closing summary generator.

STANDARDS is the single source for both. `agent_brief.py` renders it flat for a builder and as
three gated tiers for a reviewer, so a notebook cannot satisfy one and fail the other. Keep it
here; do not restate it in a prompt or in TEMPLATE.md.

`week_cheatsheet()` builds a week's closing summary from `course.yml` and `modules.yml`, so a
takeaway or a definition has one wording across the whole course. Import it; do not reimplement
it per week.
"""
import contextlib, json, os, pathlib, re, shutil, subprocess, sys, tempfile, yaml

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
    (1, "every LIVE read goes through the same cached fallback — an asset that lives only in the "
        "repo has nothing to fall back from and is read directly", False),
    (1, "every cached fallback URL this week reads actually resolves", False),
    (1, "every question has a complete model answer in the solution, prose questions included",
        False),
    (1, "the sections run in TEMPLATE 1's order, each opening with the state it needs", False),
    (1, "the answer-cell convention holds throughout: a markdown ask, its own answer cell, "
        "identical stubs", True),
    (1, "the opening cell is weekkit.OPENING with only the question and the hook changed", True),
    (1, "the week summary sits before the homework, and the opening cells carry the DataHub link",
        True),
    # NOT enforced_by_checker, though check_asserts covers part of it. Reviewers of weeks 4, 7,
    # 8, 9 and 12 each found a self-check that passes whatever the student writes, and five of
    # them said the same thing about this line: marked [auto], it told them not to hand-check
    # what the machine cannot see. check_asserts tests name availability plus ONE tautology
    # (asserting above a minmagnitude floor); an assert that is true by construction for any
    # other reason is invisible to it. The claim a reviewer must actually make is the last
    # clause, and no checker makes it.
    # From teaching week 1. All three are things no reviewer reported and the instructor hit in
    # the first hour; they are graded now so a reviewer looks for them.
    (2, "the 3-4 spine questions are present as a list, and every section heading is the "
        "question its section answers rather than the topic it covers", False),
    (2, "one operation has ONE call shape everywhere; no helper is defined in setup and first "
        "called in the homework; no return value forces an unpacking at every call site", False),
    (1, "every homework part ends in a question answered in words, not a list of names to "
        "produce — a hint pointing at an earlier section does not count", False),
    (1, "every self-check uses only names the prompt gave, and fails for the wrong answer a "
        "student will really produce — not for one the code makes impossible", False),
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
    (2, "a function of ten or more lines names its steps in comments — a docstring says what it "
        "is FOR, not what the middle of it is doing", True),
    (2, "fewer names is a PROXY and not the goal: where a smaller count and a clearer notebook "
        "disagree, clarity wins", False),
    (2, "the axis labels carry units, and every figure title carries its sample size", False),
    (2, "built to the target of 50 cells and 8 questions, not to the 60/9 ceiling", False),
    (2, "every map OF EARTH draws data/coastlines.csv", True),
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
    (3, "no answer is stated in the prose under its own question — a later section may restate "
        "a number the argument needs, but not before the question that produces it", False),
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
OPENING = (ROOT / "tools/opening.md").read_text()
# Student-facing prose lives in tools/opening.md, not in this file. It is the first thing 46
# people read and the instructor reviews it directly; prose nobody can open as prose goes stale
# invisibly, which is why the agent briefs moved out of agent_brief.py for the same reason.
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

def {fname}({signature}):
    \"\"\"{docstring}\"\"\"{guard}
    # Ask the live archive first. If it is down, or you are offline, read the copy stored with
    # the course instead, so the notebook still runs.
    try:
        {assign} pd.read_csv({url_expr})
    except Exception as e:
        print("live source unreachable, using the cached copy:", type(e).__name__)
        {assign} pd.read_csv(CACHE + "/" + {cache_expr}){result}

{unpack}
"""


def setup_cell(**kw):
    """Fill SETUP_CELL, defaulting any slot the caller does not pass.

    Adding `{imports}` to the template broke every already-built week's build script with
    KeyError until it was rerun — a shared template that grows a slot must not invalidate the
    scripts that predate it. Defaults here mean a new slot is opt-in.
    """
    # fname/guard/result exist for the ONE week whose public data function is not `load`. Week 1
    # reached the catalogue through `load` and then a second function `column` that called it, so
    # a student who never calls `load` still met it, ten lines deep, as the first `def` in the
    # course. Folding the two into one is a real reduction for a beginner -- and doing it with
    # slots keeps the shape in this template rather than sending week 1 back to hand-writing its
    # setup cell, which is the drift this template exists to prevent. Every default reproduces
    # the previous text exactly, so no other week's output moves.
    defaults = {"imports": "", "figsize": "(7, 4)", "signature": "", "cache_expr": '""',
                "docstring": "Read the live source; fall back to the copy stored with the course.",
                "url_expr": '""', "unpack": "", "cache_base": "",
                "fname": "load", "guard": "", "assign": "return", "result": ""}
    return SETUP_CELL.format(**{**defaults, **kw})


# A dataset that SHIPS WITH THE COURSE has no upstream, so it cannot use the template above. Week
# 11 shipped with a try/except whose two branches read the same file, byte for byte, from the same
# repository: the "live source" was a pinned commit of this repo and the "cached copy" was main of
# this repo. The except branch could only fire in conditions that would kill the fallback too, and
# the docstring — "fall back to the copy stored with the course" — was false, because the try
# branch already WAS that copy. That is what the tier-1 standard means by "an asset that lives
# only in the repo has nothing to fall back from and is read directly". Same imports, same plot
# defaults, same CACHE; only the four lines that pretended to be a live read are gone.
ASSET_SETUP_CELL = """{imports}import pandas as pd
import matplotlib.pyplot as plt

# house style, set once, so every plot cell below holds only what matters
plt.rcParams.update({{"figure.figsize": {figsize}, "figure.dpi": 110,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True}})

CACHE = "{cache_base}"

{unpack}
"""


def asset_setup_cell(**kw):
    """Fill ASSET_SETUP_CELL: the setup cell for a week whose data ships with the course.

    Use this instead of `setup_cell` when the week reads no live archive. Hand-writing the cell
    instead is the drift `setup_cell` exists to prevent, which is why the direct-read case is a
    variant here rather than a note telling a builder to improvise one.
    """
    defaults = {"imports": "", "figsize": "(7, 4)", "unpack": "", "cache_base": ""}
    return ASSET_SETUP_CELL.format(**{**defaults, **kw})


# The third shape: an asset too big to keep beside the notebook, fetched once from a release of
# the course repository and then kept. It is not `setup_cell` — there is no live archive and no
# cache to fall back to, only a file that is either already on disk or is not yet. It is not
# `asset_setup_cell` either, because that one reads a file the repository ships.
#
# Week 13 and track T8 each hand-wrote this cell, and the two came out byte-identical down to
# the docstring — which is not reassuring, it is the definition of drift waiting to happen: the
# next one to be edited would have been the only one edited. `pd` is deliberately absent; both
# users read arrays, and an unused pandas import is exactly the kind of thing the code-quality
# check exists to catch.
DOWNLOAD_SETUP_CELL = """{imports}import matplotlib.pyplot as plt

# house style, set once, so every plot cell below holds only what matters
plt.rcParams.update({{"figure.figsize": {figsize}, "figure.dpi": 110,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True}})

{const} = ("{url}")


def load():
    \"\"\"{docstring}\"\"\"
    try:
        return {reader}("{filename}")
    except FileNotFoundError:
        torch.hub.download_url_to_file({const}, "{filename}", progress=False)
        return {reader}("{filename}")


{unpack}
"""


def download_setup_cell(**kw):
    """Fill DOWNLOAD_SETUP_CELL: the setup cell for a week whose data arrives from a release.

    `torch.hub.download_url_to_file` is the downloader because torch is already one of the six
    libraries and every user of this cell is a week that trains something; adding `requests` to
    the student environment to save one import would be the wrong trade.
    """
    defaults = {"imports": "", "figsize": "(7, 4)", "unpack": "", "const": "DATA",
                "reader": "np.load", "url": "", "filename": "",
                "docstring": "Read the data file, downloading it from the course release the "
                             "first time."}
    return DOWNLOAD_SETUP_CELL.format(**{**defaults, **kw})


def dedupe_ids(cells):
    """Make sure no cell id was issued twice. Returns `cells`, so it can wrap a return.

    Every id scheme in this course keys a cell by its ROLE, which assumes each role occurs once
    per question window. Week 7 broke that the day a second assert-bearing cell appeared inside
    one, shipping `w07-q03-check` twice; track T2 broke it the same day when its Predict cell
    became two cells and both matched the same branch. A duplicate id is invalid nbformat, and
    because Gradescope keys a submission off cell ids it is exactly how a cell grades as
    "missing from your notebook" — a false zero, on a student who did the work.

    `stable_ids` has this built in. The seven tracks each carry their OWN `track_ids`, so they
    did not; rather than seven copies of the same four lines, they end `return
    weekkit.dedupe_ids(cells)`. The FIRST cell keeps the plain id, so anything already released
    stays stable.
    """
    seen = {}
    for c in cells:
        seen[c["id"]] = seen.get(c["id"], 0) + 1
        if seen[c["id"]] > 1:
            c["id"] = f'{c["id"]}-{seen[c["id"]]}'
    return cells


def stable_ids(cells, week_n):
    """Give every cell an id that survives a rebuild — and give the GRADED cells one that also
    survives reordering.

    nbformat assigns random ids, so every rebuild reassigned all of them: a submission made
    against an earlier release would grade as 'that cell is missing from your notebook', for
    every cell, silently, mid-term. Graded cells (an answer stub, a self-check) are keyed to the
    question they belong to, so inserting a paragraph above them changes nothing.
    """
    q, p = 0, 0
    for i, c in enumerate(cells):
        s = "".join(c.get("source", []))
        if c["cell_type"] == "markdown" and re.search(r"(?m)^\s*(#{1,4}\s*)?✏️", s):
            q += 1
            c["id"] = f"w{week_n:02d}-q{q:02d}-ask"
        elif c["cell_type"] == "code" and re.search(r"your answer here", s, re.I):
            c["id"] = f"w{week_n:02d}-q{q:02d}-answer"
        elif c["cell_type"] == "markdown" and "Double-click" in s:
            c["id"] = f"w{week_n:02d}-q{q:02d}-prose"
        # The Predict pair carries an assert but is not a question's self-check: it sits BEFORE
        # the first ✏️, so the generic branch below would key it to `q00` and collide with the
        # loading check, which is the real q00. Both cells get their own ids, numbered because
        # week 10 has two pairs. The guess cell matches in both copies -- solution `= 0.70`,
        # student `= None` -- which is what keeps a submission graded against the release.
        elif c["cell_type"] == "code" and re.search(r"(?m)^my_guess\w*\s*=", s):
            p += 1
            c["id"] = f"w{week_n:02d}-predict{p:02d}"
        elif c["cell_type"] == "code" and re.search(r"assert my_guess\w* is not None", s):
            c["id"] = f"w{week_n:02d}-predict{p:02d}-check"
        elif c["cell_type"] == "code" and "assert " in s:
            c["id"] = f"w{week_n:02d}-q{q:02d}-check"
        else:
            c["id"] = f"w{week_n:02d}-c{i:03d}"
    # BACKSTOP. Every branch above keys a cell by its ROLE, which assumes each role occurs once
    # per question window -- and week 7 broke that assumption the day a second assert-bearing
    # cell appeared inside one, shipping `w07-q03-check` twice. A duplicate id is invalid
    # nbformat, and because Gradescope keys a submission off cell ids it is exactly how a cell
    # grades as "missing from your notebook". The dedicated predict branch above fixes the case
    # that caused it; this catches the next one, whatever it turns out to be. The FIRST cell
    # keeps the plain id, so anything already released stays stable.
    return dedupe_ids(cells)


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
    course, mods = _course(), _modules()
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
    """One loader. course.yml was being read three times in this file, by three different
    expressions, two of which re-derived ROOT instead of using it."""
    return yaml.safe_load((ROOT / "course.yml").read_text())


def _modules():
    return yaml.safe_load((ROOT / "modules.yml").read_text())


def modules_upto(week_n, inclusive):
    """Module ids taught up to week_n. `inclusive` is the whole difference between the two
    callers: the prior-knowledge CONTRACT is what a student knows BEFORE this week (exclusive),
    the prior-knowledge CHECK is what the week may legally use (inclusive)."""
    return [m for s in _course()["schedule"]
            if (s["n"] <= week_n if inclusive else s["n"] < week_n)
            for m in s["modules"]]


def predict_cell(guess, summary, name="my_guess", label="committed"):
    """A "Predict before you run" cell, as TWO (solution, student) pairs — assign, then check.

    The device only works if the student commits to a number BEFORE seeing the answer. Sixteen of
    the twenty built notebooks shipped the guess already filled in, byte-identical in both copies,
    so most of 46 freshmen pressed shift-enter and committed to nothing -- and in three of them
    the pre-filled value was effectively the answer (week 3's 0.70 against a true 0.711, T5's 0.31
    against +0.309, week 6's intercept of 0, which is the whole discovery).

    WHY TWO CELLS, since one looks tidier and the first version of this helper emitted one. A
    single cell that assigns the name and then asserts on it fails two existing rules, and neither
    can be satisfied without splitting: `check_asserts` registers a cell's assignments only AFTER
    reading its asserts, so a name bound in the same cell is invisible to it -- and a predict cell
    sits before the first ✏️, so the prompt-names-it escape hatch is empty too; and
    `check_conventions` requires any cell containing `assert` to print the course's ✓ line. Both
    rules are written for a self-check. Two cells also match what a student physically does:
    change the number, run on.

    Returns [(sol, stu), (sol, stu)]. The second pair is identical in both copies.
    """
    check = (f'assert {name} is not None, \\\n'
             f'    "write a number into {name} in the cell above — the commitment is the point, "\\\n'
             f'    "and a guess you made before you saw the answer is the only one that can '
             f'teach you anything"\n'
             f'print("✓ {label} — I think", {name}, "{summary}")')
    return [(f"{name} = {guess}", f"{name} = None    # ← your number, written down before you look"),
            (check, check)]


@contextlib.contextmanager
def pinned_kernel():
    """Make the notebook kernel BE the interpreter running this build, and yield the env for it.

    `ipykernel` writes its kernelspec `argv` as a bare `"python"`, so whatever starts a kernel —
    nbconvert in a subprocess, or `NotebookClient` in process — resolves the interpreter from
    PATH rather than from `sys.executable`. On this machine PATH is the shared base env, which
    has no torch, so a build launched correctly as `.venv/bin/python tools/build_week13.py`
    still died on `import torch` INSIDE the notebook, with nothing in the traceback pointing at
    the kernelspec.

    Both drivers resolve kernelspecs through JUPYTER_PATH, so one temporary spec fixes both.
    Pinning by argv beats prefixing PATH (which `build_track_T1.py` had grown independently)
    because it names the interpreter instead of hoping the search finds it, and it beats editing
    the venv's own kernel.json because that file is inside a gitignored directory: it protects
    nobody who clones this repository and does not survive rebuilding the venv.
    """
    spec_dir = pathlib.Path(tempfile.mkdtemp(prefix="weekkit-kernel-"))
    kernel = spec_dir / "kernels" / "python3"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(json.dumps(
        {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
         "display_name": "Python 3", "language": "python"}))
    old = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = str(spec_dir)
    try:
        yield {**os.environ, "JUPYTER_PATH": str(spec_dir)}
    finally:
        if old is None:
            os.environ.pop("JUPYTER_PATH", None)
        else:
            os.environ["JUPYTER_PATH"] = old
        shutil.rmtree(spec_dir, ignore_errors=True)


def execute(path, timeout=600, cwd=None, env=None):
    """Execute a notebook in place. The ONE way eighteen of the twenty builds run their solution.

    There were two shapes and several spellings: `-m jupyter nbconvert` in most, `-m nbconvert`
    in the two that needed torch, a PATH prefix in one, a hand-written kernelspec in two. That is
    one job done several ways inside the tooling that enforces "one operation, one call shape" on
    the notebooks themselves. `-m nbconvert` rather than `-m jupyter nbconvert` because the
    latter dispatches through the `jupyter` launcher, which finds `jupyter-nbconvert` on PATH —
    one more chance to run the wrong interpreter. The kernel is pinned by `pinned_kernel`.

    Weeks 1 and 2 drive `NotebookClient` in process instead, because they hold the solution as an
    nbformat object rather than a file; they use `pinned_kernel` directly, so the interpreter is
    settled the same way in all twenty.
    """
    with pinned_kernel() as kernel_env:
        return subprocess.run(
            [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
             f"--ExecutePreprocessor.timeout={timeout}", str(path)],
            capture_output=True, text=True, cwd=cwd or ROOT,
            # JUPYTER_PATH last so a caller-supplied full environ (T7 passes
            # dict(os.environ, IPYTHONDIR=...)) cannot shadow the spec just written.
            env={**kernel_env, **(env or {}), "JUPYTER_PATH": kernel_env["JUPYTER_PATH"]})


def gate(week_n, variant=""):
    """The two gates a build must pass. Called at the END of every build_weekNN.py.

    A build that has not passed these is not a build, so this is not a step anyone can forget
    to run: it lives in the build script and exits non-zero. Judging whether the notebook is
    GOOD is a separate job, done by an agent that did not write it.
    """
    root = ROOT
    slug = next(s["slug"] for s in _course()["schedule"] if s["n"] == week_n)
    sol = root / f"docs/notebooks{variant}" / f"{slug}_solution.ipynb"
    bad = []

    # Normalise ids here: it is the single point every build passes through, and a builder
    # cannot forget it. Both notebooks, so a submission matches the released student copy.
    for f in (sol, sol.with_name(sol.name.replace("_solution", ""))):
        nb = json.loads(f.read_text())
        stable_ids(nb["cells"], week_n)
        f.write_text(json.dumps(nb, indent=1))

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

    # THE GRADED CONTRACT. A rebuild that renames a self-check silently invalidates the week's
    # Gradescope spec, and the autograder then zeroes every part whose assert it no longer
    # recognises -- "The self-check in this cell is not the one the assignment ships." A model
    # answer scored 40/100 that way on the week-1 branch, through a build that printed
    # "gates passed", because only check_all.py ever regenerated the specs and a builder does
    # not run check_all.py. So the build regenerates ITS OWN week and says whether that changed
    # anything. It does not restore: the new spec IS the correct one for the notebook just
    # built, and leaving the stale one behind is the defect. Reported, never silent.
    if not variant:
        spec = root / "tools/gradescope" / f"week{week_n:02d}" / "spec.json"
        before = spec.read_bytes() if spec.exists() else None
        subprocess.run([sys.executable, str(root / "tools/make_gradescope.py"), str(week_n)],
                       capture_output=True, cwd=root)
        after = spec.read_bytes() if spec.exists() else None
        if after != before:
            print(f"  regenerated {spec.relative_to(root)} — this build changed the graded "
                  f"contract. If week {week_n} is already released, that is BREAKING and the "
                  f"spec must be re-uploaded to Gradescope before the next submission is marked.")
        # Regenerating is not the same as CHECKING. A freshly generated spec can still be
        # ungradeable — that is exactly how a model answer scored 40/100 — so grade the model
        # answer against the bundle and fail the build if it does not come out full marks.
        # Scoped to this week: the corner cases are about the bundle machinery, not the week.
        r3 = subprocess.run([sys.executable, str(root / "tools/selftest_gradescope.py"),
                             str(week_n)], capture_output=True, text=True, cwd=root)
        if r3.returncode:
            print(r3.stdout.rstrip())
            bad.append("the autograder does not score this week's own model answer correctly")

    if bad:
        print("\nBUILD REJECTED:")
        for b in bad:
            print(f"  - {b}")
        sys.exit(1)
    print("\ngates passed: executes clean, checker OK, nothing used before its week")
