#!/usr/bin/env python
"""Generate one Gradescope autograder bundle per week, from the built notebook.

Twelve of the thirteen weeks had no spec at all, and the one that existed was hand-written —
which is the arrangement that guarantees drift: the notebook is rebuilt by a script, the spec
is edited by a person, and nothing compares them. Everything here is read out of the solution
notebook that students are actually graded against, so a rebuild that renames a self-check
regenerates the spec that looks for it.

    python tools/make_gradescope.py          # every built week
    python tools/make_gradescope.py 1        # just week 1
"""
import json, pathlib, re, sys, yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "tools" / "gradescope"
course = yaml.safe_load((ROOT / "course.yml").read_text())

HYGIENE_POINTS = 5
OWN_DATA_POINTS = 5
# What one hand-marked homework part is worth. Gradescope adds it as a manual rubric item.
MANUAL_POINTS_EACH = 25
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def src(c):
    return "".join(c.get("source", []))


def images(c):
    return sum(1 for o in c.get("outputs", []) if "image/png" in o.get("data", {}))


def split_points(pool, n_parts, figures):
    """Share the pool equally between parts, then inside a part by what it asks for.

    Week 1's hand-written spec summed to 85, not 100, so its Gradescope total silently
    disagreed with the 100 the syllabus promises. Here the arithmetic is forced to close:
    whatever rounding is left over lands on the first part.
    """
    each = pool // n_parts
    out = []
    for k in range(n_parts):
        fig = 5 if figures[k] else 0
        code = 5
        out.append({"check_points": each - code - fig, "code_points": code,
                    "figure_points": fig})
    short = pool - sum(p["check_points"] + p["code_points"] + p["figure_points"] for p in out)
    out[0]["check_points"] += short
    return out


def build(n):
    w = next(s for s in course["schedule"] if s["n"] == n)
    sol = ROOT / "docs/notebooks" / f"{w['slug']}_solution.ipynb"
    if not sol.exists():
        return None
    cells = json.loads(sol.read_text())["cells"]
    # The RELEASED student notebook, so the grader can tell scaffolding it was given from work
    # the student did. Several parts hand out starter lines; counting those as "code written"
    # gave an untouched notebook marks for opening it.
    stu_path = ROOT / "docs/notebooks" / f"{w['slug']}.ipynb"
    stu_cells = json.loads(stu_path.read_text())["cells"] if stu_path.exists() else []

    hw = next((i for i, c in enumerate(cells)
               if c["cell_type"] == "markdown" and src(c).startswith("## Homework")), None)
    if hw is None:
        return None

    # A part is delimited by its self-check. weekkit.CHECK_LINE fixes the shape as
    # '✓ {label} — {summary}', so the marker is everything up to the em dash.
    # Keyed by CELL, not by tick line. Week 5's homework prints two ✓ lines from one cell
    # ("the box" and "the window"), and a spec that made those two parts sent the grader
    # looking for the second marker in the cells AFTER the cell it lives in — so part 3 read
    # as missing and lost 31 marks on the model answer itself.
    parts, marks = [], []
    for i in range(hw, len(cells)):
        ticks = [line.split("—")[0].strip()
                 for o in cells[i].get("outputs", [])
                 for line in "".join(o.get("text", "")).splitlines()
                 if line.strip().startswith("✓")]
        if ticks:
            parts.append(i)
            marks.append(ticks[0])
    if not parts:
        return None

    # Name each part what the notebook calls it. A Gradescope row reading "Part 2" tells a
    # student nothing about which question they lost marks on; "Part 2 — run the ten-to-one
    # rule downwards" tells them where to look. Week 1 titles its parts in the prompt, the
    # other weeks head them "Your turn N", so fall back to the self-check's own label.
    titles = []
    for c in cells[hw:]:
        s = src(c)
        if "✏️" in s:
            line = s.splitlines()[0]
            line = re.sub(r"^#+\s*", "", line).replace("✏️", "").strip().strip("*").strip()
            titles.append(line)

    # Figures belong to the part whose self-check follows them.
    figs, cursor = [], hw
    for check_i in parts:
        figs.append(sum(images(cells[j]) for j in range(cursor, check_i)))
        cursor = check_i + 1

    # Every week's homework has three parts and only two self-checks: part 3 is the prose
    # synthesis, and a paragraph cannot be asserted on. That is the right design, but it means
    # the autograder can score two parts, not three. Giving its marks to the other two would
    # inflate them and score the essay at zero — so the essay's share is RESERVED here and
    # graded by hand in Gradescope, and the bundle says so out loud.
    n_questions = sum(1 for x in cells[hw:] if "✏️" in src(x))
    manual = max(0, n_questions - len(parts))
    manual_points = MANUAL_POINTS_EACH * manual

    # Only a week whose homework runs on data the student chooses can be checked for using
    # their own. Week 1 is the birthday week; nothing else asks for a personal date.
    own = bool(w.get("pinned", {}).get("per_student")) or n == 1
    pool = 100 - manual_points - HYGIENE_POINTS - (OWN_DATA_POINTS if own else 0)
    alloc = split_points(pool, len(parts), figs)

    spec = {"week": n, "notebook_stem": w["slug"], "homework_heading": "## Homework",
            "parts": []}
    for k, (marker, a) in enumerate(zip(marks, alloc), start=1):
        title = titles[k - 1] if k - 1 < len(titles) else ""
        generic = (not title) or re.fullmatch(r"Your turn \d+", title)
        name = f"Part {k}" if generic else title
        if generic and marker.startswith("✓"):
            name = f"Part {k} — {marker[1:].strip()}"
        # The exact self-check the student is given, so tampering is detectable. Without this
        # a submission that deletes every assert and pastes a fabricated "✓ ..." output scores
        # full marks having written no code at all — measured, not hypothesised.
        checks = [ln.strip() for ln in src(cells[parts[k - 1]]).splitlines()
                  if ln.strip().startswith("assert ")]
        p = {"name": name, "marker": marker, "checks": checks,
             "check_points": a["check_points"], "code_points": a["code_points"]}
        if figs[k - 1]:
            p["figures"] = figs[k - 1]
            p["figure_points"] = a["figure_points"]
        spec["parts"].append(p)

    # Every homework code cell exactly as released, keyed by its stable id. Comparing a part
    # against one stub cell was not enough: a part owns the cells between the previous check and
    # its own, and some of those are given to the student (a checkpoint, a scaffolding cell).
    # Measured against a single stub they read as work, and an untouched notebook collected
    # marks for cells nobody had touched. The ids survive a rebuild, so this is the honest key.
    if stu_cells:
        hw_stu = next((i for i, cc in enumerate(stu_cells)
                       if cc["cell_type"] == "markdown" and src(cc).startswith("## Homework")), 0)
        spec["released"] = {cc["id"]: src(cc).strip()
                            for cc in stu_cells[hw_stu:]
                            if cc["cell_type"] == "code" and cc.get("id")}

    if own:
        # Every date the CLASS half uses is a date the student did not choose. Derived, not
        # typed: a hand-written list goes stale the moment a pinned date moves.
        seen = set()
        for c in cells[:hw]:
            seen.update(DATE_RE.findall(src(c)))
            for o in c.get("outputs", []):
                seen.update(DATE_RE.findall("".join(o.get("text", ""))))
        spec["own_dates"] = {"points": OWN_DATA_POINTS, "forbidden": sorted(seen)}
    spec["hygiene"] = {"points": HYGIENE_POINTS}

    auto = sum(p["check_points"] + p["code_points"] + p.get("figure_points", 0)
               for p in spec["parts"]) + HYGIENE_POINTS + (OWN_DATA_POINTS if own else 0)
    spec["manual"] = {"parts": manual, "points_each": MANUAL_POINTS_EACH,
                      "points": manual_points,
                      "note": f"Part {len(parts) + 1} of {n_questions} is answered in prose and "
                              f"is marked by hand. Add a {manual_points}-point manual rubric "
                              f"item in Gradescope; the autograder is out of {auto}."}
    assert auto + manual_points == 100, f"week {n} totals {auto + manual_points}, not 100"
    spec["autograder_max"] = auto
    return spec


wanted = [int(sys.argv[1])] if len(sys.argv) > 1 else [s["n"] for s in course["schedule"]]
made = 0
for n in wanted:
    spec = build(n)
    if spec is None:
        continue
    d = OUT / f"week{n:02d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
    for f in ("grade.py", "run_autograder", "setup.sh"):
        (d / f).write_bytes((OUT / f).read_bytes())
    (d / "run_autograder").chmod(0o755)
    made += 1
    print(f"  week {n:>2}  {len(spec['parts'])} auto-graded + "
          f"{spec['manual']['parts']} by hand · "
          f"{'own-data check · ' if spec.get('own_dates') else ''}"
          f"autograder {spec['autograder_max']}/100")
print(f"{made} autograder bundle(s) in tools/gradescope/weekNN/")
