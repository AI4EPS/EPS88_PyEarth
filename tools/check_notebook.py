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
    qs = [i for i, c in enumerate(cells)
          if c["cell_type"] == "markdown" and "✏️" in src(c)]
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
        if re.search(r"lons?\b|longitude", s) and "coast" not in s:
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


def check_write_count(cells):
    """The notebook states how many places the student writes. Nothing checked it, so the
    sentence could drift from the file silently. Week 2's builder wrote this check for itself;
    it belongs to every week."""
    hw = next((k for k, c in enumerate(cells) if c["cell_type"] == "markdown"
               and re.search(r"(?im)^##\s*Homework\s*$", src(c))), len(cells))
    # Count the ANSWER STUBS themselves. Two earlier versions counted the phrase anywhere
    # (which matched the front matter that NAMES the convention) and then counted prompt
    # successors (which misses a part with two answer cells, e.g. code then a prose paragraph
    # after the self-check). Two independent reviewers proved the notebooks right and this
    # check wrong, both times. A code stub is a code cell carrying the marker; a prose stub is
    # the short italic markdown line and nothing else.
    def is_stub(c):
        s = src(c)
        if c["cell_type"] == "code":
            return "your answer here" in s.lower()
        return "Double-click" in s and len(s.strip()) < 200
    places = [i for i, c in enumerate(cells) if is_stub(c)]
    n_class = sum(1 for i in places if i < hw)
    n_home = len(places) - n_class
    front = " ".join(src(c) for c in cells[:6]).lower()
    m = re.search(r"(\w+)\s+places where you write something:\s*(\w+)\s+in class,?\s*"
                  r"(?:and\s*)?(\w+)\s+at home", front)
    if not m:
        warns.append("the front matter does not state how many places the student writes")
        return
    words = {w: i for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen".split())}
    num = lambda s: words.get(s, int(s) if s.isdigit() else -1)
    said = (num(m.group(1)), num(m.group(2)), num(m.group(3)))
    real = (n_class + n_home, n_class, n_home)
    if said != real:
        errs.append(f"front matter says {said[0]} write-places ({said[1]} class, {said[2]} home); "
                    f"the file has {real[0]} ({real[1]} class, {real[2]} home)")


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
    check_predict(cells)
    check_asserts(cells); check_figures(cells); check_imports(cells)
    check_code_quality(cells); check_summary_is_generated(cells, n)
    check_write_count(cells)
    print(f"week {n} · {len(cells)} cells · {len(qs)} questions · {figs} figures")
    for x in warns: print(f"  warn  {x}")
    for e in errs:  print(f"  ERROR {e}")
    print("OK" if not errs else f"{len(errs)} error(s)")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
