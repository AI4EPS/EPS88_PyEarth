#!/usr/bin/env python
"""Does week N use anything the course has not taught by week N?

`prior_knowledge.py` TELLS a builder what students already know. Nothing verified it afterwards —
and thirteen weeks will be built in parallel by different agents, so a week quietly reaching for
`groupby` six weeks before D2 teaches it would surface in December, in a classroom, and no
single-week reviewer could catch it. This is the only check that can.

    python tools/check_prior_knowledge.py 5      # one week
    python tools/check_prior_knowledge.py all    # every week that exists
"""
import ast, builtins, json, pathlib, re, sys, yaml
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
course = yaml.safe_load((ROOT / "course.yml").read_text())
mods = {m["id"]: m for m in yaml.safe_load((ROOT / "modules.yml").read_text())["modules"]}

# Syntax and names that belong to no module because they are Python itself, plus the plotting
# aliases. Anything here is allowed in every week.
ALWAYS = set(dir(builtins)) | {
    "plt", "pd", "np", "torch", "sklearn", "f", "self",
    "rcParams", "update", "show", "format", "strip", "split", "join", "lower", "upper",
    "read_csv", "figure", "subplots", "tight_layout",
    # matplotlib internals a self-check reaches for; not course content and never
    # something a student is taught to write
    "get_array", "get_offsets", "get_children",
}


def taught_by(week_n):
    """Every function name the course has introduced up to and including week n."""
    names = set()
    for mid in weekkit.modules_upto(week_n, inclusive=True):
        for f in mods.get(mid, {}).get("functions", []) or []:
            # Every identifier in the entry, not just the head: an entry may chain
            # (plt.gca().set_aspect) or list alternatives (max(list) / min(list)), and reading
            # only the head silently lost set_aspect.
            for ident in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", f["name"]):
                names.add(ident)
                names.add(ident.split(".")[-1])
    return names


def used_in(path, skip_setup=True):
    """Every function called in the notebook, plus the ones it defines itself.

    The setup cell is skipped: TEMPLATE 1.3 exempts it explicitly, flagging what arrives early
    as "Coming later", so its pandas calls are sanctioned rather than a violation. Setup is
    everything before the first question prompt.
    """
    nb = json.loads(path.read_text())
    cells = nb["cells"]
    if skip_setup:
        # Anchored like check_questions: the front matter that NAMES the pencil convention was
        # ending the setup exemption, so the setup cell's own flagged imports read as violations.
        first_q = next((i for i, c in enumerate(cells)
                        if c["cell_type"] == "markdown"
                        and re.search(r"(?m)^\s*(#{1,4}\s*)?\u270f\ufe0f", "".join(c["source"]))),
                       0)
        cells = cells[first_q:]
    # Collect DEFINITIONS from every cell but CALLS only from post-setup cells: the setup may
    # legitimately define the week's helpers while also using syntax that arrives later.
    calls, defined = set(), set()
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            try:
                for node in ast.walk(ast.parse("".join(c["source"]))):
                    if isinstance(node, ast.FunctionDef):
                        defined.add(node.name)
                    # A callable held in a VARIABLE — model(x), loss_function(a, b) — is what
                    # PyTorch requires, and reads as an untaught function unless assignment
                    # targets count as defined too.
                    elif isinstance(node, ast.Assign):
                        for tgt in node.targets:
                            defined |= {x.id for x in ast.walk(tgt) if isinstance(x, ast.Name)}
            except SyntaxError:
                pass
    for c in cells:
        if c["cell_type"] != "code":
            continue
        try:
            tree = ast.parse("".join(c["source"]))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined.add(node.name)
            elif isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Name):
                    calls.add(f.id)
                elif isinstance(f, ast.Attribute):
                    base = f.value.id if isinstance(f.value, ast.Name) else None
                    calls.add(f"{base}.{f.attr}" if base in ("plt", "pd", "np") else f.attr)
    return calls, defined


def check(week_n):
    s = next(x for x in course["schedule"] if x["n"] == week_n)
    nb = ROOT / "docs/notebooks" / f"{s['slug']}.ipynb"
    if not nb.exists():
        return None
    calls, defined = used_in(nb)
    sol = nb.with_name(nb.stem + "_solution.ipynb")
    have_sol = sol.exists()
    if have_sol:
        # BOTH halves from the solution: a function the student must write is defined only there,
        # and so is any variable a model answer binds — model = make_picker() reads as an untaught
        # call otherwise, which is exactly what PyTorch requires a week to do.
        defined |= used_in(sol)[1]
        # NOT the solution's calls. Adding them surfaces every function a model answer uses that
        # no module declares — real gaps in weeks 3, 4, 7 and 8, and worth closing, but it is a
        # separate job from checking that a week does not reach forward. Deferred deliberately.
    allowed = taught_by(week_n) | ALWAYS | defined
    stray = sorted(c for c in calls if c not in allowed and c.split(".")[-1] not in allowed)
    # A module with NO functions: declared cannot be checked against — that is an unpopulated
    # catalogue, not a notebook using something early. Ten of thirteen weeks were in that state,
    # and failing them would have blocked every build on a gap only the orchestrator can fill.
    empty = [mid for mid in s["modules"] if not (mods.get(mid, {}).get("functions") or [])]
    print(f"week {week_n:>2} · {len(calls)} calls · {len(stray)} not taught by week {week_n}")
    for x in stray:
        print(f"    {x}")
    if empty and stray:
        print(f"    NOT A FAILURE: {', '.join(empty)} declare no functions: yet, so nothing can be"
              f"\n    checked. Report the list above for modules.yml and the orchestrator applies it.")
        return []
    if stray and not have_sol:
        # A function the student is told to write is defined only in the solution, and the
        # solution is gitignored until release. Without it the `defined` set is incomplete by
        # construction, so a stray name here is unresolved, not untaught — week 9's
        # `crossing_year` is exactly this, and failing on it made every CI run red.
        print("    NOT A FAILURE: no solution in this checkout, so functions the student is asked"
              "\n    to write cannot be resolved. Re-run where the solution exists to check this.")
        return []
    return stray


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    weeks = [s["n"] for s in course["schedule"] if s["modules"]] if arg == "all" else [int(arg)]
    bad = [n for n in weeks if check(n)]
    print("OK" if not bad else f"weeks with untaught calls: {bad}")
    sys.exit(1 if bad else 0)
