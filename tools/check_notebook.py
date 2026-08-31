#!/usr/bin/env python
"""Validate a built week's notebooks against TEMPLATE.md.

check_course.py validates the PLAN; this validates the ARTIFACT. Every rule here is a defect
that actually reached the instructor at least once, mechanised so it cannot reach them twice.

    python tools/check_notebook.py 1
    python tools/check_notebook.py 1 --variant _b
"""
import ast, builtins, json, pathlib, re, sys, yaml
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
course = yaml.safe_load((ROOT / "course.yml").read_text())
LIBS = {"numpy", "np", "pandas", "pd", "matplotlib", "sklearn", "torch"}
errs, warns = [], []


def src(c):
    return "".join(c.get("source", []))


def load(n, variant):
    w = next(s for s in course["schedule"] if s["n"] == n)
    d = ROOT / f"docs/notebooks{variant}"
    st, so = d / f"{w['slug']}.ipynb", d / f"{w['slug']}_solution.ipynb"
    for f in (st, so):
        if not f.exists():
            sys.exit(f"missing: {f}")
    return w, json.loads(st.read_text()), json.loads(so.read_text())


# --- the two copies must be the same notebook ---------------------------------
def check_pair(student, solution):
    a, b = student["cells"], solution["cells"]
    if len(a) != len(b):
        errs.append(f"student has {len(a)} cells, solution {len(b)} — they have drifted apart")
    if any(c.get("outputs") for c in a):
        errs.append("the student notebook carries outputs — it must ship clean")
    if not any(c.get("outputs") for c in b):
        errs.append("the solution notebook has no outputs — it was never executed")
    figs = sum(1 for c in b for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
    if figs == 0:
        errs.append("the solution contains no figures")

    # "it has outputs" is not "it ran". A solution can carry figures and still raise.
    for i, c in enumerate(b):
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                errs.append(f"solution cell {i}: raised {o.get('ename')} — it does not execute")

    # Execution counts that do not start at 1 mean a cell was run and deleted before saving,
    # so the shipped outputs were produced by code that is not in the file.
    counts = [c["execution_count"] for c in b
              if c["cell_type"] == "code" and c.get("execution_count")]
    if counts and counts[0] != 1:
        errs.append(f"solution execution counts start at {counts[0]}, not 1 — a cell was executed "
                    f"and removed, so these outputs are not what the shipped code produces")
    return figs


def check_summary_is_generated(cells, n):
    """TEMPLATE 1.7: the week summary is generated, never typed. A hand-edited one drifts from
    modules.yml and can advertise syntax the week never taught."""
    i = next((k for k, c in enumerate(cells) if c["cell_type"] == "markdown"
              and re.search(rf"(?im)^##\s*Week {n} summary", src(c))), None)
    if i is None:
        return
    want = re.sub(r"\s+", " ", weekkit.week_cheatsheet(n)).strip()
    got = re.sub(r"\s+", " ", src(cells[i])).strip()
    if want != got:
        errs.append(f"cell {i}: the week summary does not match weekkit.week_cheatsheet({n}) — "
                    f"it was hand-edited, so a rebuild would silently change it")


# --- banned content: each of these shipped to the instructor once --------------
BANNED = [
    (r"(?i)\bAI disclosure\b",              "an AI-disclosure cell (weekly notebooks carry none)"),
    (r"(?im)^#{1,4}\s*(EXTENSION|Buffer|If we have time)", "a labelled buffer/EXTENSION section"),
    (r"▶\s*WATCH|✎\s*TOGETHER|✔\s*YOUR TURN", "planning markers left in the file"),
    (r"(?i)may not get to this|if we have time",
     "a sentence flagging the buffer — TEMPLATE 1.5: the buffer is invisible to students"),
    (r"(?i)\babout \d+ ?(min|minutes)\b",   "a minute estimate shown to students"),
    (r"(?i)\b(Fall|Spring) 20\d\d\b",       "an offering term"),
    (r"(?i)\bdue (on |by )?(Sunday|Monday|Friday)\b", "a due date"),
]


def check_banned(cells):
    for i, c in enumerate(cells):
        s = src(c)
        for pat, why in BANNED:
            if re.search(pat, s):
                errs.append(f"cell {i}: {why}")


# --- questions ----------------------------------------------------------------
def check_questions(cells):
    # Anchored to a heading: the front matter that NAMES the convention ("every one is marked
    # with a pencil") was counted as a question, and its neighbour failed for missing an answer
    # stub. A question is a HEADING that carries the marker, not any mention of it.
    qs = [i for i, c in enumerate(cells)
          if c["cell_type"] == "markdown" and re.search(r"(?m)^\s*(#{1,4}\s*)?✏️", src(c))]
    prose = 0
    for i in qs:
        nxt = cells[i + 1] if i + 1 < len(cells) else None
        if nxt is None:
            errs.append(f"question at cell {i} has no answer cell after it")
            continue
        s = src(nxt)
        if nxt["cell_type"] == "markdown":
            prose += 1
            if "Double-click" not in s:
                errs.append(f"cell {i+1}: prose answer cell is missing the standard stub")
        elif "your answer here" not in s.lower():
            errs.append(f"cell {i+1}: code answer cell is missing the standard stub")
    # Position, not wording: a class prompt that merely SAYS "you will need this in the
    # homework" was silently counted as homework, and this file already had a second, correct
    # definition twenty lines down.
    hw_at = next((k for k, c in enumerate(cells) if c["cell_type"] == "markdown"
                  and re.search(r"(?im)^##\s*Homework\s*$", src(c))), len(cells))
    hw = sum(1 for i in qs if i > hw_at)
    cls = len(qs) - hw
    if not 7 <= len(qs) <= 9:
        errs.append(f"{len(qs)} questions — TEMPLATE 1 asks for 7-9")
    if not 5 <= cls <= 6:
        warns.append(f"{cls} class questions — TEMPLATE 1 asks for 5-6")
    if not 2 <= hw <= 3:
        warns.append(f"{hw} homework questions — TEMPLATE 1 asks for 2-3")
    if prose > 2:
        errs.append(f"{prose} questions answered in prose — at most two; this is a DS course")
    if not 45 <= len(cells) <= 60:
        warns.append(f"{len(cells)} cells — TEMPLATE 1 asks for 45-60")
    return qs


# --- the summary sits before the homework -------------------------------------
def check_plain_words(cells, n):
    """TEMPLATE 5 calls the plain_words table BINDING — the sentence a student meets must be the
    sentence in the table — and nothing checked it.

    This is the drift that scales: thirteen weeks written by thirteen agents, one calling it
    "the accumulator pattern" and another "a running total", and a student meets two names for
    one idea. Week 2's builder had already paraphrased three of four before being told.
    """
    mraw = yaml.safe_load((ROOT / "modules.yml").read_text())
    wk = next(s for s in course["schedule"] if s["n"] == n)
    # Strip blockquote and list markers before comparing: putting the binding sentence in a
    # "> " blockquote is the obvious formatting choice and used to fail.
    raw = " ".join(src(c) for c in cells)
    text = re.sub(r"\s+", " ", re.sub(r"(?m)^\s*[>*-]\s?", "", raw))
    for d in mraw.get("plain_words", []):
        if d["module"] not in wk["modules"]:
            continue
        want = re.sub(r"\s+", " ", d["words"]).strip()
        if want not in text:
            errs.append(f"the recorded wording for '{d['idea']}' does not appear verbatim — "
                        f"TEMPLATE 5 makes it binding, so paraphrasing it gives the course two "
                        f"names for one idea")


def check_opening(cells):
    """The two invariant paragraphs of weekkit.OPENING must appear verbatim.

    Only the question and the hook change between weeks. Everything else a student reads on the
    way in is the same in week 13 as in week 1, and thirteen agents left to write it themselves
    will write thirteen versions of it.
    """
    fixed = [p for p in weekkit.OPENING.split("\n\n") if "{" not in p]
    head = re.sub(r"\s+", " ", " ".join(src(c) for c in cells[:4]))
    for para in fixed:
        want = re.sub(r"\s+", " ", para).strip()
        if want and want not in head:
            errs.append(f"the opening does not match weekkit.OPENING — missing: "
                        f"{want[:64]}...")


def check_summary_is_this_week(cells, n):
    """Every function the summary lists is one this week's notebook actually calls.

    The table is built from what the MODULE declares, and nothing checked it against what the
    week does. A module can outlive the week that introduced it, or a week can drop a section —
    either way the summary starts advertising functions the student never met, which is the
    opposite of a summary.
    """
    mods_ = {m["id"]: m for m in yaml.safe_load((ROOT / "modules.yml").read_text())["modules"]}
    wk = next(s for s in course["schedule"] if s["n"] == n)
    called = set()
    for c in cells:
        if c["cell_type"] != "code":
            continue
        try:
            tree = ast.parse(src(c))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            # Attributes and subscripts too, not just calls: grid.shape, grid.size and
            # table.loc[...] are things a week teaches and no Call node ever names.
            if isinstance(node, ast.Attribute):
                base = node.value.id if isinstance(node.value, ast.Name) else None
                called.add(f"{base}.{node.attr}" if base in ("plt", "pd", "np") else node.attr)
            elif isinstance(node, ast.Name):
                called.add(node.id)
    for mid in wk["modules"]:
        for f in mods_.get(mid, {}).get("functions", []) or []:
            if not f.get("remember", True):
                continue
            names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", f["name"]))
            if not (names & called) and not ({x.split(".")[-1] for x in names} & called):
                errs.append(f"the summary lists `{f['name']}`, which this week's notebook never "
                            f"calls — a summary is what the student met, not what the module owns")


def check_conventions(cells):
    """The shapes that must be identical in every week: the self-check line and the closing.

    Grading is 46 PDFs per week read on screen. A self-check that states its result in a
    different shape each week makes the grader re-learn where to look thirteen times.
    """
    text = " ".join(src(c) for c in cells)
    if weekkit.CLOSING_HEADING not in text:
        errs.append(f"no '{weekkit.CLOSING_HEADING}' section")
    for i, c in enumerate(cells):
        s = src(c)
        if c["cell_type"] != "code" or "assert " not in s:
            continue
        if "✓" not in s:
            errs.append(f"cell {i}: a self-check with no '✓ <label> — ...' line; a grader reading "
                        f"46 PDFs needs the same shape every week (weekkit.CHECK_LINE)")


def check_predict(cells):
    """TEMPLATE 1: at least one cell headed exactly '### Predict before you run'."""
    if not any(c["cell_type"] == "markdown"
               and re.search(r"(?im)^#{2,4}\s*Predict before you run", src(c)) for c in cells):
        errs.append("no '### Predict before you run' cell — TEMPLATE 1 requires one, and the "
                    "heading is the convention that lets a reviewer find it")


def check_order(cells):
    def find(pat):
        return next((i for i, c in enumerate(cells)
                     if c["cell_type"] == "markdown" and re.search(pat, src(c), re.M)), None)
    summ, hw = find(r"(?im)^##\s*Week \d+ summary"), find(r"(?im)^##\s*Homework\s*$")
    if summ is None:
        errs.append("no '## Week N summary' section")
    if hw is None:
        errs.append("no '## Homework' section")
    if summ is not None and hw is not None and summ > hw:
        errs.append("the week summary comes AFTER the homework; it must come before")
    if not any("datahub.berkeley.edu" in src(c) for c in cells[:3]):
        errs.append("no DataHub link in the opening cells")


# --- self-checks --------------------------------------------------------------
def check_asserts(cells):
    """Every name an assert uses must be one the student was told to create, or one the
    notebook already defined. A NameError from the reassurance cell is the worst failure
    mode there is."""
    defined, last_prompt = set(), ""
    floors = {m for c in cells for m in re.findall(r"minmagnitude=([\d.]+)", src(c))}
    for i, c in enumerate(cells):
        s = src(c)
        if c["cell_type"] == "markdown":
            if "✏️" in s:
                last_prompt = s
            continue
        try:
            tree = ast.parse(s)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                for nm in {x.id for x in ast.walk(node) if isinstance(x, ast.Name)}:
                    if nm in defined or hasattr(builtins, nm) or nm in LIBS:
                        continue
                    if nm not in last_prompt:
                        errs.append(f"cell {i}: assert uses `{nm}`, which no prompt names and "
                                    f"nothing defines — this raises NameError for every student")
                # Tautology, precisely: asserting a value is ABOVE the floor the query
                # already imposed. `min(m) < 4.5` is the opposite — it checks the student
                # removed the floor — so only >= and > count.
                for cmp in [x for x in ast.walk(node) if isinstance(x, ast.Compare)]:
                    for op, comp in zip(cmp.ops, cmp.comparators):
                        if not isinstance(op, (ast.GtE, ast.Gt)):
                            continue
                        if isinstance(comp, ast.Constant) and str(comp.value) in floors:
                            errs.append(f"cell {i}: assert requires >= {comp.value}, the floor the "
                                        f"query already imposes — true by construction, cannot fail")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.For)):
                tgt = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in tgt:
                    defined |= {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
            elif isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom)):
                defined |= {getattr(node, "name", None) or ""} | {
                    (a.asname or a.name).split(".")[0] for a in getattr(node, "names", [])}


# --- figures and imports ------------------------------------------------------
def check_figures(cells):
    for i, c in enumerate(cells):
        s = src(c)
        if c["cell_type"] != "code" or not re.search(r"plt\.(scatter|plot|hist|bar)\(", s):
            continue
        if "plt.xlabel" not in s or "plt.ylabel" not in s:
            errs.append(f"cell {i}: a plot with no axis labels")
        # The lon/lat names must be PLOTTED, not merely mentioned: a checkpoint line that
        # unpacks six lists into a histogram cell made the histogram fail as a map.
        if re.search(r"plt\.(scatter|plot)\(\s*lons?\b", s) and "coast" not in s:
            errs.append(f"cell {i}: a map that does not draw data/coastlines.csv")


def check_imports(cells):
    code = [(i, src(c)) for i, c in enumerate(cells) if c["cell_type"] == "code"]
    for i, s in code:
        for m in re.findall(r"(?m)^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", s):
            top = m.split(".")[0]
            if top not in LIBS:
                errs.append(f"cell {i}: imports `{top}`, which is not one of the six libraries")
            if code and i != code[0][0]:
                warns.append(f"cell {i}: import outside the setup cell")


def check_code_quality(cells):
    """The mechanical half of TEMPLATE 8 'the code is teaching material'."""
    code = [(i, src(c)) for i, c in enumerate(cells) if c["cell_type"] == "code"]
    all_code = "\n".join(s for _, s in code)

    # commented-out code: a comment that parses as a statement, not as prose
    for i, s in code:
        for ln in s.split("\n"):
            body = ln.strip()
            if not body.startswith("#") or body.startswith("# \u2190"):
                continue
            body = body.lstrip("#").strip()
            if not body or "=" not in body and "(" not in body:
                continue
            try:
                tree = ast.parse(body)
            except SyntaxError:
                continue
            if any(isinstance(x, (ast.Assign, ast.Call, ast.Import, ast.For))
                   for x in ast.walk(tree)):
                errs.append(f"cell {i}: commented-out code — `{body[:50]}`; delete it")

    # the same import twice — it runs, and every other rule passed it
    seen = {}
    for i, src_ in code:
        for m in re.finditer(r"(?m)^\s*import\s+([\w.]+)(?:\s+as\s+(\w+))?", src_):
            name = m.group(2) or m.group(1)
            if name in seen:
                errs.append(f"cell {i}: imports `{name}` again — already imported in cell {seen[name]}")
            seen[name] = i

    # imports nothing uses
    for i, s in code:
        for m in re.finditer(r"(?m)^\s*import\s+([\w.]+)(?:\s+as\s+(\w+))?", s):
            name = m.group(2) or m.group(1).split(".")[0]
            uses = len(re.findall(rf"\b{re.escape(name)}\b", all_code))
            if uses <= 1:
                errs.append(f"cell {i}: imports `{name}` and never uses it")

    # a block said three times is a function waiting to be written
    blocks = {}
    for i, s in code:
        lines = [l for l in s.split("\n") if l.strip() and not l.strip().startswith("#")]
        for k in range(len(lines) - 2):
            blocks.setdefault("\n".join(lines[k:k + 3]), []).append(i)
    for blk, where in blocks.items():
        if len(set(where)) >= 3:
            warns.append(f"cells {sorted(set(where))}: the same three lines appear in all of "
                         f"them — say it once (`{blk.split(chr(10))[0].strip()[:44]}`)")


def main():
    n = int(sys.argv[1])
    variant = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--variant" else ""
    w, student, solution = load(n, variant)
    cells = student["cells"]
    figs = check_pair(student, solution)
    check_banned(cells); qs = check_questions(cells); check_order(cells)
    check_opening(cells)
    # both notebooks: a function taught only in the homework is stubbed out of the
    # student copy, so scanning that alone made it unlistable
    check_summary_is_this_week(cells + solution['cells'], n); check_conventions(cells); check_predict(cells); check_plain_words(cells, n)
    check_asserts(cells); check_imports(cells)
    # Figures live in the SOLUTION too: a model answer that draws a map was never
    # checked for labels or coastlines, because only the student copy was passed in.
    check_figures(cells); check_figures(solution['cells'])
    check_code_quality(cells); check_summary_is_generated(cells, n)
    print(f"week {n} · {len(cells)} cells · {len(qs)} questions · {figs} figures")
    for x in warns: print(f"  warn  {x}")
    for e in errs:  print(f"  ERROR {e}")
    print("OK" if not errs else f"{len(errs)} error(s)")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
