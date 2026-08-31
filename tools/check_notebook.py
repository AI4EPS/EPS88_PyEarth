#!/usr/bin/env python
"""Validate a built week's notebooks against TEMPLATE.md.

check_course.py validates the PLAN; this validates the ARTIFACT. Every rule here is a defect
that actually reached the instructor at least once, mechanised so it cannot reach them twice.

    python tools/check_notebook.py 1
    python tools/check_notebook.py 1 --variant _b
"""
import ast, builtins, io, json, keyword, pathlib, re, sys, tokenize, yaml
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
course = yaml.safe_load((ROOT / "course.yml").read_text())
# Derived from course.yml, not restated here: the list existed in three places (this file,
# TEMPLATE.md and platform: libraries:) and a fourth library would have had to be added to all
# three. The aliases are what a notebook actually writes.
_ALIAS = {"numpy": "np", "pandas": "pd", "scikit-learn": "sklearn", "pytorch": "torch"}
LIBS = {l for lib in course["platform"]["libraries"] if lib != "python"
        for l in (lib, _ALIAS.get(lib, lib))}
errs, warns = [], []


def src(c):
    return "".join(c.get("source", []))


def load(n, variant):
    """The solution may legitimately be absent.

    Solutions are gitignored until the Wednesday after each due date, so a checkout of the
    public repo — CI's, or a student's — has the student notebook and nothing else. Treating
    that as `missing:` made every CI run fail from the day the repo was created: the checker
    was demanding a file the release policy forbids. Absent solution means the solution-side
    checks do not run; it does not mean the week is broken.
    """
    w = next(s for s in course["schedule"] if s["n"] == n)
    d = ROOT / f"docs/notebooks{variant}"
    st, so = d / f"{w['slug']}.ipynb", d / f"{w['slug']}_solution.ipynb"
    if not st.exists():
        sys.exit(f"missing: {st}")
    sol = json.loads(so.read_text()) if so.exists() else None
    return w, json.loads(st.read_text()), sol


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


def keywords_written(cells):
    """Every Python keyword the notebook's code actually writes, from a real tokenizer.

    Not a regex over the source: `not` inside a comment or a docstring is not the student
    meeting `not`, and a rule that counted it would pass a week that never wrote one.
    """
    out = set()
    for c in cells:
        if c["cell_type"] != "code":
            continue
        try:
            for tok in tokenize.generate_tokens(io.StringIO(src(c) + "\n").readline):
                if tok.type == tokenize.NAME and keyword.iskeyword(tok.string):
                    out.add(tok.string)
        except (tokenize.TokenError, IndentationError, SyntaxError):
            continue
    return out


def check_summary_is_this_week(cells, n):
    """Every function the summary lists is one this week's notebook actually calls.

    The table is built from what the MODULE declares, and nothing checked it against what the
    week does. A module can outlive the week that introduced it, or a week can drop a section —
    either way the summary starts advertising functions the student never met, which is the
    opposite of a summary.

    Not every entry names a function, though. The loops-and-functions modules list SYNTAX —
    `if / elif / else`, `and / or / not`, `None`, `return a, b` — and syntax is never an
    ast.Name, so the call test rejected eight of P3 and P4's fifteen entries no matter what the
    notebook contained. Those are checked on the keyword the student actually has to write; the
    x and things of `for x in things:` are notation, not names to go looking for.
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
    written = keywords_written(cells)
    for mid in wk["modules"]:
        for f in mods_.get(mid, {}).get("functions", []) or []:
            if not f.get("remember", True):
                continue
            names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", f["name"]))
            kws = {x for x in names if keyword.iskeyword(x)}
            if kws:
                missing = sorted(kws - written)
                if missing:
                    errs.append(f"the summary lists `{f['name']}`, and this week's notebook never "
                                f"writes `{missing[0]}` — a summary is what the student met, not "
                                f"what the module owns")
                continue
            names -= kws
            # An entry that writes neither a call nor an attribute names notation, not a
            # function: `a docstring`, `list[i]`. There is nothing to look for.
            if not re.search(r"[A-Za-z_][A-Za-z0-9_.]*\s*\(", f["name"]) \
                    and not any("." in x for x in names):
                continue
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
def check_weak_asserts(cells, solution_cells=()):
    """Flag the assert SHAPES that cannot fail for the answer a student would really write.

    Reviewers found one of these in weeks 3, 4, 6, 7, 8, 9 and 12 — the single most common
    defect in the course, and always the same handful of forms. It stays a warning because
    only a human can say whether a given assert catches the realistic mistake; what a checker
    can do is name the shapes that never do, so the builder has to look.

      assert x in [3, 4, 5]     — membership in a list the prompt itself dictated
      assert a != b             — detects only "did nothing at all"
      assert len(x) == 3        — a length fixed by a constructor the prompt supplied
      assert len(a) == len(b)   — two lengths the worked cell above set

    Not flagged: comparisons against a measured value (`assert 0.5 < rate < 2`), which is what
    these should become.
    """
    # Names bound to a list/tuple/set literal anywhere in the notebook. The real shape is
    # `assert earth_fewest_bins in bin_counts`, where bin_counts is the list of candidates the
    # prompt handed over two cells earlier — so testing only for a literal on the right missed
    # the case the rule was written for. My own selftest caught that.
    # Scan the SOLUTION too: the candidate list often lives in the answer the student writes,
    # which the student copy stubs out, so scanning their copy alone cannot see it.
    literal_lists = set()
    for c in list(cells) + list(solution_cells or ()):
        if c["cell_type"] != "code":
            continue
        try:
            tr = ast.parse(src(c))
        except SyntaxError:
            continue
        for nd in ast.walk(tr):
            if (isinstance(nd, ast.Assign) and len(nd.targets) == 1
                    and isinstance(nd.targets[0], ast.Name)
                    and isinstance(nd.value, (ast.List, ast.Tuple, ast.Set))):
                literal_lists.add(nd.targets[0].id)

    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        try:
            tree = ast.parse(src(c))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assert):
                continue
            test = node.test
            why = None
            if (isinstance(test, ast.Compare) and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.In)
                    and (isinstance(test.comparators[0], (ast.List, ast.Tuple, ast.Set))
                         or getattr(test.comparators[0], "id", None) in literal_lists)):
                why = "membership in a list of candidates the prompt offered — all of them pass"
            elif (isinstance(test, ast.Compare) and len(test.ops) == 1
                  and isinstance(test.ops[0], ast.NotEq)):
                why = "`!=` only catches doing nothing at all, not doing it wrongly"
            elif (isinstance(test, ast.Compare) and len(test.ops) == 1
                  and isinstance(test.ops[0], ast.Eq)
                  and isinstance(test.left, ast.Call)
                  and getattr(test.left.func, "id", "") == "len"):
                rhs = test.comparators[0]
                # A CONSTANT only. `len(a) == len(b)` against an independent object is a real
                # check — week 4's `len(triangles.get_offsets()) == len(volcanoes)` catches
                # plotting the earthquakes instead of the volcanoes — and flagging it made the
                # rule noisier than the defect it was written for.
                if isinstance(rhs, ast.Constant):
                    why = ("a length the prompt already fixed — it passes whatever the student "
                           "put in the container")
            if why:
                line = src(c).splitlines()[node.lineno - 1].strip()[:56]
                warns.append(f"cell {i}: this self-check probably cannot fail — {why} "
                             f"(`{line}`). Assert a measured value instead.")


# --- the spine has to match the headings ---------------------------------------
SPINE_SKIP = ("What you'll", "The question", "Week ", "Homework", "Setup")


def check_spine(cells):
    """The 3-4 spine questions are the section headings, one for one.

    TEMPLATE 1 asks for a bare numbered list after "What you'll be able to do" AND for every
    section heading to be the question its section answers. Shipped without a check, three
    agents wrote the list three different ways and three weeks listed four questions while
    shipping two headings — so the instructor reads a spine promising four moves and finds
    nothing to scroll to for half of them. A spine that does not match the headings is worse
    than no spine: it is a map of a building that was not built.
    """
    spine = []
    for c in cells:
        s = src(c)
        if c["cell_type"] == "markdown" and s.startswith("## What you'll be able to do"):
            spine = [l.strip() for l in s.split("\n") if re.match(r"^\d+\.\s", l.strip())]
            break
    heads = [src(c).split("\n")[0][3:].strip() for c in cells
             if c["cell_type"] == "markdown" and src(c).startswith("## ")]
    body = [h for h in heads if not any(h.startswith(k) for k in SPINE_SKIP)]

    if not spine:
        errs.append("no spine: TEMPLATE 1 asks for the 3-4 questions that lead the class as a "
                    "numbered list at the end of \"What you'll be able to do\"")
        return
    if not 3 <= len(spine) <= 4:
        errs.append(f"the spine lists {len(spine)} questions; TEMPLATE 1 asks for 3 or 4")
    if len(spine) != len(body):
        errs.append(f"the spine lists {len(spine)} questions but the notebook has {len(body)} "
                    f"section headings — every spine question must BE a heading, or the "
                    f"instructor has nothing to scroll to. Headings: {body}")
        return
    for i, (q, h) in enumerate(zip(spine, body), start=1):
        # Strip the number from BOTH sides: weeks 2-5 number their headings ("## 1. What ...")
        # and weeks 10-13 do not, and both are fine — the rule is that the TEXT matches.
        strip = lambda s: re.sub(r"^\d+\.\s*", "", s).strip().rstrip("?").lower()
        qt, ht = strip(q), strip(h)
        if not (ht.startswith(qt[:28]) or qt.startswith(ht[:28])):
            errs.append(f"spine question {i} and section heading {i} do not match:\n"
                        f"        spine:   {q[:76]}\n        heading: {h[:76]}")
    for h in body:
        if not h.rstrip().endswith("?"):
            errs.append(f"section heading is a topic, not a question: \"{h[:70]}\"")


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
        # Only an unbounded map claims to show the world; a zoomed box may hold no coast at all
        # (week 12's Ridgecrest box contains zero coastline points).
        if (re.search(r"plt\.(scatter|plot)\(\s*lons?\b", s) and "coast" not in s
                and "plt.xlim" not in s):
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


def check_code_quality(cells, solution_cells=()):
    """The mechanical half of TEMPLATE 8 'the code is teaching material'."""
    code = [(i, src(c)) for i, c in enumerate(cells) if c["cell_type"] == "code"]
    all_code = "\n".join(s for _, s in code)
    # The unused-import test must see the SOLUTION too. In a student copy every answer is a
    # stub, so an import used only inside answers reads as dead — and in a project track, where
    # the whole notebook after the load is stubs, that is a false positive on every build. An
    # import the model answer uses is used.
    if solution_cells:
        all_code += "\n" + "\n".join(src(c) for c in solution_cells
                                      if c.get("cell_type") == "code")

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
        # A CHECKPOINT is exempt for the same reason plot furniture is: TEMPLATE §1.4 REQUIRES
        # one at the head of every section that needs earlier state, and its whole job is to
        # repeat the setup verbatim. Saying it once is the one thing a checkpoint may not do.
        # Weeks 10 and 12 each carry three, and two reviewers independently reported the warning
        # as a false positive — a rule fighting a rule.
        if s.lstrip().startswith("# ── Checkpoint"):
            continue
        # Plot furniture is exempt: check_figures requires plt.xlabel/ylabel literally inside
        # every plot cell, so a week with four maps MUST repeat them. Requiring the repetition
        # and then flagging it is a rule fighting itself.
        lines = [l for l in s.split("\n")
                 if l.strip() and not l.strip().startswith("#")
                 and not re.match(r"\s*plt\.(xlabel|ylabel|title|show|xlim|ylim|gca|"
                                  r"locator_params|colorbar|legend|figure)\b", l)]
        for k in range(len(lines) - 2):
            blocks.setdefault("\n".join(lines[k:k + 3]), []).append(i)
    for blk, where in blocks.items():
        if len(set(where)) >= 3:
            warns.append(f"cells {sorted(set(where))}: the same three lines appear in all of "
                         f"them — say it once (`{blk.split(chr(10))[0].strip()[:44]}`)")


# --- the homework must run on a cold kernel ------------------------------------
def check_checkpoints_rebuild(cells, solution_cells):
    """Every checkpoint rebuilds everything the section after it reads.

    Weeks 2, 3 and 5 all shipped a homework that raised NameError on a fresh kernel, each in the
    same way: the checkpoint rebuilt the constants and the function but not the DATA, and the
    homework's first loop reached for a list built during class. Nobody notices while building,
    because the builder's kernel has run the class cells; the student meets it days later, alone,
    at the one moment they cannot ask. TEMPLATE §1 already requires the rebuild "including the
    scalars" — this is what makes the requirement true rather than merely stated.

    Static, not executed: collect what the homework READS, subtract what setup + the homework's
    own checkpoint + the homework itself BIND. Anything left is a name that only class defined.
    """
    # Needs the SOLUTION. Every name a student writes is bound only there, so on a checkout
    # without solutions this rule reports the student's own answers as missing state — which is
    # exactly what it did on CI, failing weeks 1, 2 and 3 for defects that do not exist. The
    # rule is real where the solution is; where it is not, there is nothing to reason from.
    if not solution_cells:
        return
    cs = solution_cells
    setup = [c for c in cs if c["cell_type"] == "code"][:1]

    # Every checkpoint, plus the homework heading — which is a checkpoint in intent whether or
    # not one was written, because it is the boundary a student crosses on a cold kernel.
    starts = [i for i, c in enumerate(cs)
              if c["cell_type"] == "code" and src(c).lstrip().startswith("# ── Checkpoint")]
    hw = next((i for i, c in enumerate(cs)
               if c["cell_type"] == "markdown" and src(c).startswith("## Homework")), None)
    if hw is not None and not any(hw <= s <= hw + 3 for s in starts):
        starts.append(hw)
    if not starts:
        return

    theirs = _student_functions(cells, solution_cells)
    for start in sorted(starts):
        _one_checkpoint(cs, setup, start, theirs)


def _student_functions(cells, solution_cells):
    """Functions the STUDENT writes, in their own answer cell.

    The one thing a checkpoint may not rebuild. Pasting the model answer into a cell every
    student reads publishes the answer to the question that asked for it — week 7 asks them to
    write `count_at_least` and `predict_count`, and the whole volcano half then calls both. The
    honest instruction is the one the homework intro already gives: name the cells and tell them
    to re-run their own. So these names, and only these, may be discharged that way, and only
    where the boundary says it in as many words.

    Everything an ANSWER STUB binds, not just its functions. The line that matters is whether
    the cell is the student's own answer, not whether the name happens to be a `def`: week 3's
    `earth_deep` and `earth_high` are assignments in the same stub as `peak_position`, and
    pasting them into a checkpoint publishes Your turn 2 exactly as pasting the function would.

    This does NOT reopen what the rule was built to catch. The variables that failed in weeks 2,
    3 and 5 — `usable_names`, `below`, `fraction_below` — are bound in WORKED class cells, which
    carry no stub marker and are therefore not exempt here; a checkpoint must still rebuild
    those, and it does. Only a name whose sole definition is the student's own work can be
    discharged this way, and only where the boundary names it in a re-run instruction.
    """
    out = set()
    for stu, sol in zip(cells, solution_cells or []):
        if stu["cell_type"] != "code" or "your answer here" not in src(stu).lower():
            continue
        try:
            tree = ast.parse(src(sol))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                out.add(node.name)
            elif isinstance(node, ast.Assign):
                for tg in node.targets:
                    out |= {x.id for x in ast.walk(tg) if isinstance(x, ast.Name)}
    return out


RERUN = re.compile(r"re-?run\b", re.I)


def _told_to_rerun(text, name):
    """The boundary tells the student, by name, to re-run their own cell for this one."""
    return any(re.search(rf"\b{re.escape(name)}\b", text[m.start():m.start() + 200])
               for m in RERUN.finditer(text))


def _one_checkpoint(cs, setup, start, student_functions=frozenset()):
    """A student who restarts, runs setup, then runs from `start`, must not get NameError."""
    after = [c for c in cs[start:] if c["cell_type"] == "code"]
    if not after:
        return
    bound, read = set(), set()
    for c in setup + after:
        try:
            tree = ast.parse(src(c))
        except SyntaxError:
            return                      # a stub with a blank body; nothing to conclude
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                bound |= {(a.asname or a.name).split(".")[0] for a in node.names}
            elif isinstance(node, ast.FunctionDef):
                bound.add(node.name)
                bound |= {a.arg for a in node.args.args}
            elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For,
                                   ast.comprehension, ast.withitem)):
                tgts = (node.targets if isinstance(node, ast.Assign) else
                        [getattr(node, "target", None) or getattr(node, "optional_vars", None)])
                for tg in tgts:
                    if tg is not None:
                        bound |= {x.id for x in ast.walk(tg) if isinstance(x, ast.Name)}
    for c in after:
        try:
            tree = ast.parse(src(c))
        except SyntaxError:
            return
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                read.add(node.id)

    safe = bound | set(dir(builtins)) | set(keyword.kwlist) | LIBS
    boundary = src(cs[start])
    missing = sorted(n for n in read - safe
                     if not n.startswith("_")
                     and not (n in student_functions and _told_to_rerun(boundary, n)))
    if missing:
        errs.append(
            f"the section from cell {start} reads {', '.join('`' + m + '`' for m in missing)}, "
            f"which neither setup nor that checkpoint builds — a student who restarts and does "
            f"what the checkpoint says gets NameError. Rebuild them in it.")


def main():
    n = int(sys.argv[1])
    variant = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--variant" else ""
    w, student, solution = load(n, variant)
    cells = student["cells"]
    sol_cells = solution["cells"] if solution else []
    figs = check_pair(student, solution) if solution else 0
    check_banned(cells); qs = check_questions(cells); check_order(cells)
    check_spine(cells)
    check_opening(cells)
    # both notebooks: a function taught only in the homework is stubbed out of the
    # student copy, so scanning that alone made it unlistable
    if solution:
        # Needs BOTH: the summary legitimately lists functions a model answer calls but the
        # student stub does not, so scanning the student copy alone flags every one of them.
        check_summary_is_this_week(cells + sol_cells, n)
    check_conventions(cells); check_predict(cells); check_plain_words(cells, n)
    check_asserts(cells); check_weak_asserts(cells, sol_cells); check_imports(cells)
    # Figures live in the SOLUTION too: a model answer that draws a map was never
    # checked for labels or coastlines, because only the student copy was passed in.
    check_figures(cells); check_figures(sol_cells)
    check_code_quality(cells, sol_cells); check_summary_is_generated(cells, n)
    check_checkpoints_rebuild(cells, sol_cells)
    scope = "" if solution else " · student only (no solution in this checkout)"
    print(f"week {n} · {len(cells)} cells · {len(qs)} questions · {figs} figures{scope}")
    for x in warns: print(f"  warn  {x}")
    for e in errs:  print(f"  ERROR {e}")
    print("OK" if not errs else f"{len(errs)} error(s)")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
