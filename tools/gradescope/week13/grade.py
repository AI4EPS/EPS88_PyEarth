"""Grade a submitted PyEarth notebook from what it recorded, without re-running it.

Every student works on their own birthday, so there is no fixed answer to compare against.
What CAN be checked is the notebook's own evidence: each homework part ends in a self-check
cell whose asserts reference the required names, so a printed "check line" is proof that
those names existed and that the asserts passed on the student's machine.

The week is described by spec.json, so the same grader serves all thirteen weeks.
"""

import glob
import json
import os
import re
import sys

SUBMISSION_DIR = os.environ.get("GS_SUBMISSION", "/autograder/submission")
RESULTS_PATH = os.environ.get("GS_RESULTS", "/autograder/results/results.json")
SPEC_PATH = os.environ.get("GS_SPEC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec.json"))

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


# ── reading the notebook ────────────────────────────────────────────────────

def find_notebook(directory, stem):
    """The submitted notebook: the one whose name matches the week, else the only one there."""
    found = [p for p in glob.glob(os.path.join(directory, "**", "*.ipynb"), recursive=True)
             if ".ipynb_checkpoints" not in p]
    if not found:
        return None, "No .ipynb file was found in your submission."
    exact = [p for p in found if os.path.basename(p).startswith(stem)]
    if exact:
        return sorted(exact, key=len)[0], None
    if len(found) == 1:
        return found[0], None
    named = ", ".join(sorted(os.path.basename(p) for p in found))
    return None, f"Several notebooks were submitted and none is named {stem}.ipynb: {named}"


def load_cells(path):
    with open(path, encoding="utf-8") as fh:
        nb = json.load(fh)
    return nb.get("cells", [])


def source_of(cell):
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def text_output(cell):
    """Everything the cell printed or returned, as one string."""
    chunks = []
    for out in cell.get("outputs", []):
        kind = out.get("output_type")
        if kind == "stream":
            text = out.get("text", "")
            chunks.append(text if isinstance(text, str) else "".join(text))
        elif kind in ("execute_result", "display_data"):
            plain = out.get("data", {}).get("text/plain", "")
            chunks.append(plain if isinstance(plain, str) else "".join(plain))
        elif kind == "error":
            chunks.append(f"{out.get('ename', 'Error')}: {out.get('evalue', '')}")
    return "".join(chunks)


def image_count(cell):
    return sum(1 for out in cell.get("outputs", [])
               if any(k.startswith("image/") for k in out.get("data", {})))


def errored(cell):
    return any(out.get("output_type") == "error" for out in cell.get("outputs", []))


def did_work(cell, spec):
    """True when this cell differs from the cell we released under the same id.

    Not "is it non-empty": several homework cells ship starter lines, and a checkpoint cell is
    given complete. Judging those by emptiness paid out marks to a notebook nobody had opened.
    The comparison is against the released source for this cell's own stable id, so scaffolding
    counts as scaffolding however much of it there is.
    """
    released = (spec.get("released") or {}).get(cell.get("id"))
    mine = student_part(source_of(cell)).strip()
    if is_stub(mine):
        return False
    if released is None:
        return True                     # a cell they added themselves is work
    return mine != student_part(released).strip()


def student_part(src):
    """The part of a cell the student wrote, i.e. everything above the supplied self-check.

    A self-check is handed to them complete; counting its lines as their work would give full
    marks for an empty answer, since the assert and the print are always present.
    """
    out = []
    for line in src.splitlines():
        st = line.strip()
        if st.startswith("assert ") or st.startswith("print(\"\u2713") or st.startswith("print('\u2713"):
            break
        out.append(line)
    return "\n".join(out)


def is_stub(src):
    """True when the cell holds nothing but comments and blank lines."""
    return not [ln for ln in src.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def normalise(src):
    return "\n".join(ln.rstrip() for ln in src.splitlines()).strip()


def provided_sources(path):
    """Every cell of the notebook as released. Checkpoint and scaffold cells arrive already
    filled in, and crediting those as the student's own work hands out marks for nothing."""
    if not path or not os.path.exists(path):
        return set()
    try:
        with open(path, encoding="utf-8") as fh:
            return {normalise(source_of(c)) for c in json.load(fh).get("cells", [])
                    if c.get("cell_type") == "code"}
    except Exception:
        return set()


# ── locating the parts ──────────────────────────────────────────────────────

def locate(cells, marker, start=0):
    """Index of the self-check cell carrying this marker, searching from `start`."""
    for i in range(start, len(cells)):
        if cells[i].get("cell_type") == "code" and marker in source_of(cells[i]):
            return i
    return None


def homework_start(cells, heading):
    """Where the homework begins, so class-section cells are never mistaken for answers."""
    if heading:
        for i, cell in enumerate(cells):
            if cell.get("cell_type") == "markdown" and heading in source_of(cell):
                return i
    return 0


# ── the checks ──────────────────────────────────────────────────────────────

def grade(cells, spec, provided=frozenset()):
    """Each part owns every code cell between the part before it and its own self-check,
    so a student may add scratch cells or split a figure across cells without losing marks."""
    tests = []

    def add(name, score, maximum, output):
        tests.append({"name": name, "score": round(score, 2), "max_score": maximum,
                      "output": output, "status": "passed" if score >= maximum else "failed"})

    cursor = homework_start(cells, spec.get("homework_heading"))
    owned = []          # every code cell belonging to any part

    for part in spec["parts"]:
        marker = part["marker"]
        check_i = locate(cells, marker, cursor)

        if check_i is None:
            add(part["name"], 0, part["check_points"],
                "The self-check cell for this part is not in the notebook.\n"
                f'It is the cell that prints "{marker}". Do not delete it — it is how this part is checked.')
            add(f"{part['name']} — code written", 0, part["code_points"],
                "Not checked: the self-check cell for this part is missing.")
            if part.get("figures"):
                add(f"{part['name']} — {part['figures']} figures", 0, part["figure_points"],
                    "Not checked: the self-check cell for this part is missing.")
            continue

        # THROUGH check_i, not up to it. In this course the student writes their answer in the
        # SAME cell as the self-check — the stub is `# ← your answer here` with the assert and
        # the print below it — so a part that owned only the cells strictly before its check
        # owned nothing at all, and "code written" scored zero on every model answer.
        mine = [i for i in range(cursor, check_i + 1) if cells[i].get("cell_type") == "code"]
        owned.extend(mine)
        cursor = check_i + 1

        # 1. did the self-check run and pass?
        # First: is it still the self-check we gave them? A printed ✓ is only evidence that the
        # asserts passed if the asserts are still there. Deleting them and pasting the expected
        # output otherwise scores full marks for an empty notebook.
        printed = text_output(cells[check_i])
        want = part.get("checks") or []
        have = [ln.strip() for ln in source_of(cells[check_i]).splitlines()
                if ln.strip().startswith("assert ")]
        if want and have != want:
            missing = [w for w in want if w not in have]
            add(part["name"], 0, part["check_points"],
                "The self-check in this cell is not the one the assignment ships.\n"
                + (f"{len(missing)} of its {len(want)} checks are missing or altered.\n"
                   if missing else "Its checks have been altered.\n")
                + "The ✓ line is only meaningful when the checks above it are the ones given to "
                  "you. Restore the cell from the released notebook, run it, and submit again.")
            add(f"{part['name']} — code written", 0, part["code_points"],
                "Not checked: this part's self-check was altered.")
            if part.get("figures"):
                add(f"{part['name']} — {part['figures']} figures", 0, part["figure_points"],
                    "Not checked: this part's self-check was altered.")
            continue
        if marker in printed and not errored(cells[check_i]):
            line = next((ln for ln in printed.splitlines() if marker in ln), "").strip()
            add(part["name"], part["check_points"], part["check_points"],
                f"Self-check passed.\n{line}")
        elif errored(cells[check_i]):
            add(part["name"], 0, part["check_points"],
                "The self-check cell ended in an error, so this part is not finished yet:\n"
                f"  {printed.strip()[:400]}")
        else:
            add(part["name"], 0, part["check_points"],
                "The self-check cell has no recorded output, so it was never run.\n"
                "Run every cell (Kernel ▸ Restart & Run All) and submit the saved notebook.")

        # 2. was any code actually written for this part?
        # The self-check's own assert and print are given to the student, so they are not
        # evidence of work: judge the check cell on what sits ABOVE the first assert.
        written = [i for i in mine if did_work(cells[i], spec)]
        if written:
            lines = sum(len([l for l in student_part(source_of(cells[i])).splitlines()
                             if l.strip()]) for i in written)
            add(f"{part['name']} — code written", part["code_points"], part["code_points"],
                f"{lines} lines of code across {len(written)} cell(s).")
        else:
            add(f"{part['name']} — code written", 0, part["code_points"],
                'No code was written for this part — the answer cell still holds only '
                'the "your answer here" comment.')

        # 3. figures the part is required to draw, counted across all of its cells
        if part.get("figures"):
            want = part["figures"]
            drawn = sum(image_count(cells[i]) for i in mine)
            if drawn >= want:
                add(f"{part['name']} — {want} figures", part["figure_points"], part["figure_points"],
                    f"{drawn} figures saved in the notebook.")
            else:
                add(f"{part['name']} — {want} figures", 0, part["figure_points"],
                    f"{drawn} figures are saved in this part; it asks for {want}. Run each "
                    "plotting cell and check the picture is showing before you save.")

    # student_part, not source_of: a part now owns its self-check cell, and that cell always
    # holds the assert and print we gave them. Measuring the whole cell made an untouched
    # notebook read as attempted, and it collected the "runs clean" marks for running nothing.
    attempted = any(did_work(cells[i], spec) for i in owned)

    # own data, not the day class worked through together
    if spec.get("own_dates"):
        points = spec["own_dates"]["points"]
        forbidden = set(spec["own_dates"]["forbidden"])
        used = set()
        for i in owned:
            used.update(DATE_RE.findall(source_of(cells[i])))
        fresh = used - forbidden
        if not attempted:
            add("Your own day", 0, points, "No homework code has been written yet.")
        elif not used:
            add("Your own day", points, points, "No dates read from the code; nothing to flag.")
        elif fresh:
            add("Your own day", points, points, f"Dates used: {', '.join(sorted(fresh))}.")
        else:
            add("Your own day", 0, points,
                "The only dates in your homework are the ones class used together "
                f"({', '.join(sorted(used))}). The homework asks for your OWN birthday.")

    # the notebook was actually run, top to bottom, without errors
    if spec.get("hygiene"):
        points = spec["hygiene"]["points"]
        broken = [i for i in owned if errored(cells[i])]
        unrun = [i for i in owned if cells[i].get("execution_count") is None
                 and not is_stub(source_of(cells[i]))]
        if not attempted:
            add("Notebook runs clean", 0, points, "No homework code has been written yet.")
        elif broken:
            add("Notebook runs clean", 0, points,
                f"{len(broken)} of your homework cells ended in an error. "
                "Fix them, re-run the notebook, and submit it again.")
        elif unrun:
            add("Notebook runs clean", 0, points,
                f"{len(unrun)} homework cells hold code but were never run. "
                "Use Kernel ▸ Restart & Run All before you save.")
        else:
            add("Notebook runs clean", points, points, "No errors in the homework cells.")

    return tests


def main():
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    def write(payload):
        with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    # Loading the spec used to sit ABOVE this, outside any handler, so a missing or malformed
    # spec.json raised before results.json existed — and Gradescope shows a student an
    # autograder failure, not a score, when there is no results file. An instructor-side
    # mistake must never look like the student's problem.
    try:
        with open(SPEC_PATH, encoding="utf-8") as fh:
            spec = json.load(fh)
    except Exception as exc:
        write({"score": 0, "output":
               f"The autograder is misconfigured and this is not your fault: {exc}\n"
               "Your submission was received. Tell your instructor; it will be re-graded.",
               "visibility": "visible"})
        return

    path, problem = find_notebook(SUBMISSION_DIR, spec["notebook_stem"])
    if path and spec["notebook_stem"] not in os.path.basename(path):
        # Otherwise this reads as "every self-check is missing", which sounds like the student
        # deleted them rather than uploaded the wrong week — the likeliest mistake of all, on a
        # course where thirteen notebooks sit in one folder with near-identical names.
        write({"score": 0, "visibility": "visible", "output":
               f"This looks like the wrong notebook. Gradescope received "
               f"{os.path.basename(path)}, but this assignment grades "
               f"{spec['notebook_stem']}.ipynb.\n\nRe-submit that file and you will be graded "
               f"normally — there is no penalty for uploading the wrong one."})
        return
    if problem:
        write({"score": 0, "output": problem + "\n\nSubmit the .ipynb file itself, saved after "
               "you have run every cell.", "visibility": "visible"})
        return

    try:
        cells = load_cells(path)
    except Exception as exc:
        write({"score": 0, "visibility": "visible",
               "output": f"{os.path.basename(path)} could not be read as a notebook ({exc})."})
        return

    here = os.path.dirname(os.path.abspath(__file__))
    provided = provided_sources(os.path.join(here, "student_notebook.ipynb"))
    tests = grade(cells, spec, provided)
    total = sum(t["score"] for t in tests)
    out_of = sum(t["max_score"] for t in tests)
    write({
        "output": (f"Graded {os.path.basename(path)} — {total:g} of {out_of:g} "
                   "from the autograder. The written reflection is read by hand and "
                   "carries the remaining marks.\n\n"
                   "Nothing here re-runs your code: it reads what your notebook recorded "
                   "when you ran it, so always submit a notebook you have just run."),
        "tests": tests,
        # Results stay hidden until Weiqiang publishes grades for the assignment. Students are
        # never handed a mark by the autograder itself; he reviews first and releases when ready.
        "visibility": "after_published",
        "stdout_visibility": "hidden",
    })


if __name__ == "__main__":
    main()
