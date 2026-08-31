#!/usr/bin/env python
"""Emit the brief for a week's BUILDER or REVIEWER agent.

Generate briefs; never hand-write one. Every week is then asked for the same thing, and the
paths and names come from course.yml rather than from whoever is orchestrating.

    python tools/agent_brief.py build 2
    python tools/agent_brief.py review 2
"""
import os, re, sys, pathlib, tempfile, yaml
import prior_knowledge
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = str(ROOT)
# A session-specific scratch path was hardcoded here, so every brief this generated told its
# agent to write into a directory that vanishes with the session that wrote it. Derive it, and
# let the caller override.
SCRATCH = os.environ.get("EPS88_SCRATCH", str(pathlib.Path(tempfile.gettempdir()) / "eps88-build"))
# NOT a hardcoded venv path: whoever runs this generator is the interpreter the agent should
# use, and a literal path breaks on any other machine or if the venv moves.
PY = sys.executable

course = yaml.safe_load((ROOT / "course.yml").read_text())


def _render(template, ns):
    """Fill a brief template from the caller's locals.

    The brief text used to live inside f-strings here, 220 lines of prose in a 350-line Python
    file. Prose nobody can read as prose goes stale invisibly: a line telling every builder that
    "TEMPLATE section 7 is a loop you run" survived for hours after section 7 stopped being a
    loop, because it was buried in a string literal. As .md it diffs, and a stale sentence is
    visible in review.
    """
    text = (pathlib.Path(__file__).parent / template).read_text()
    return text.format(**{**globals(), **ns})   # module constants, then the caller's


def week(n):
    return next(s for s in course["schedule"] if s["n"] == n)


def names(n, variant=""):
    """variant is a suffix used only when A/B-testing two builds of the same week."""
    s = week(n)["slug"]
    d = f"docs/notebooks{variant}"
    return f"{d}/{s}.ipynb", f"{d}/{s}_solution.ipynb", f"tools/build_week{n:02d}{variant}.py"



def raw_block(path, start_pat, next_pat):
    """Copy a YAML block out of a file VERBATIM, comments included.

    yaml.safe_dump would drop every comment, and the comments on `pinned:` are where the
    reasoning lives — why this date and not another. A slice that silently discards them
    hands the builder a decision with no argument attached.
    """
    lines = path.read_text().split("\n")
    i = next(k for k, l in enumerate(lines) if re.match(start_pat, l))
    j = next((k for k in range(i + 1, len(lines)) if re.match(next_pat, lines[k])), len(lines))
    return "\n".join(lines[i:j]).rstrip()


def write_slice(n):
    """Write the week's OWN specification to one file, so a builder reads ~150 lines
    instead of the 1,773 in course.yml + modules.yml + CLAUDE.md + the audit.

    Reading whole files was the single biggest cost in the first timed build: the spec is
    long because it covers fourteen weeks, and a builder needs one of them.
    """
    w = week(n)
    mraw = yaml.safe_load((ROOT / "modules.yml").read_text())
    mods = {m["id"]: m for m in mraw["modules"]}
    import datetime
    out = [f"# Week {n} — the whole specification for THIS week",
           "",
           f"**Generated {datetime.datetime.now():%Y-%m-%d %H:%M}. This is a SNAPSHOT.** If you "
           f"are reading it without having just run `agent_brief.py`, re-run it — a slice left "
           f"on disk from an earlier session carries whatever course.yml said then, and week 2's "
           f"stale copy survived a correction to Mars's surface temperature that changed the "
           f"week's headline number.",
           "",
           "This is course.yml's entry for week "
           f"{n} and modules.yml's entries for its modules, copied verbatim. It replaces "
           "reading those two files; TEMPLATE.md you still read in full.",
           "", "## course.yml — week %d" % n, "```yaml",
           raw_block(ROOT / "course.yml", rf"^  - n: {n}$", "^  - n: "), "```", ""]
    for mid in w["modules"]:
        out += [f"## modules.yml — {mid}", "```yaml",
                raw_block(ROOT / "modules.yml", rf"^  - id: {mid}$", "^  - id: "), "```", ""]
    pw = [d for d in mraw.get("plain_words", []) if d["module"] in w["modules"]]
    if pw:
        out += ["## plain_words for this week's modules", "",
                "Binding wording: use these words for these ideas.", "```yaml",
                yaml.safe_dump(pw, sort_keys=False, allow_unicode=True, width=100).rstrip(),
                "```", ""]
    out += ["## Libraries", "",
            "Students may import ONLY: "
            + ", ".join(f"`{l}`" for l in course["platform"]["libraries"])
            + ". The standard library is not a loophole.", ""]
    f = ROOT / "tools" / f"_week{n:02d}_spec.md"
    f.write_text("\n".join(out))
    return f

def past_defects():
    """Judgement defects from previous builds — the ones no checker can catch.

    Mechanised defects are deliberately left out: check_notebook.py already catches those, and
    a brief that repeats what a checker enforces is longer for no gain.
    """
    f = ROOT.parent / "notes" / "defects.yml"
    if not f.exists():
        return ""
    # Newest first, capped: a mechanised defect drops out on its own (a checker catches it),
    # but the judgement ones only accumulate, and a brief nobody finishes teaches nothing.
    raw = yaml.safe_load(f.read_text())["defects"]
    d = [(i, x) for i, x in enumerate(raw)
         if x.get("scope") == "build" and not x.get("mechanised")
         and not x.get("promoted") and not x.get("superseded")]
    # Date, THEN position in the file. Nine build defects were added on 2026-08-31 and twelve
    # were competing for eight slots, so sorting on date alone left it to Python which of that
    # day's lessons reached a builder — the mechanism for carrying lessons forward was dropping
    # the newest ones at random. Later in the file is newer; that is deterministic.
    d = [x for _, x in sorted(d, key=lambda p: (str(p[1].get("date", "")), p[0]),
                              reverse=True)[:8]]
    if not d:
        return ""
    body = "\n".join(f"- **{x['id']}** ({x['caught_by']}) — {x['lesson'].strip()}" for x in d)
    return f"""
## Defects from previous builds — do not repeat these

Each of these shipped once and was rejected. None is machine-checkable, which is why they are
here rather than in a checker. Read them as things a reviewer will look for.

{body}
"""


def build_brief(n, variant=""):
    variant_flag = f" --variant {variant}" if variant else ""
    write_slice(n)
    audits = ", ".join(f"`{ROOT.parent / e}`" for e in week(n).get("evidence", [])) \
             or "(none cited — say so in your report)"
    w = week(n)
    student, solution, script = names(n, variant)
    question, EPS88 = w["question"], ROOT.parent
    defects, standards_flat = past_defects(), weekkit.stop_list()
    return _render("brief_build.md", locals())


def review_brief(n, variant=""):
    variant_flag = f" --variant {variant}" if variant else ""
    write_slice(n)          # the reviewer reads it too; do not depend on a build having run
    audits = ", ".join(f"`{ROOT.parent / e}`" for e in week(n).get("evidence", [])) \
             or "(none cited — say so)"
    w = week(n)
    student, solution, script = names(n, variant)
    question, EPS88 = week(n)["question"], ROOT.parent
    defects, standards_tiered = past_defects(), weekkit.tiers()
    return _render("brief_review.md", locals())


if __name__ == "__main__":
    mode, n = sys.argv[1], int(sys.argv[2])
    variant = sys.argv[3] if len(sys.argv) > 3 else ""
    print(build_brief(n, variant) if mode == "build" else review_brief(n, variant))
