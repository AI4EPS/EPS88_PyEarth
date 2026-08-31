#!/usr/bin/env python
"""Generate docs/README.md from course.yml and modules.yml.

Everything students read about the shape of the course comes from the plan, so the site cannot
say one thing while course.yml says another. ONE page: it is both the GitHub landing page and the
mkdocs home, and a separate syllabus.md only duplicated it.
"""
import pathlib, urllib.parse as up, yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
c = yaml.safe_load((ROOT / "course.yml").read_text())
mods = {m["id"]: m for m in yaml.safe_load((ROOT / "modules.yml").read_text())["modules"]}
p = c["platform"]


def link(slug):
    repo_name = p["repo"].rstrip("/").split("/")[-1]
    q = up.urlencode({"repo": p["repo"],
                      "urlpath": f"lab/tree/{repo_name}/{p['notebook_dir']}/{slug}.ipynb",
                      "branch": p["branch"]})
    return f"{p['datahub']}/hub/user-redirect/git-pull?{q}"


# Only weeks whose notebook exists. make_mkdocs.py has guarded this since the site was set up;
# this generator did not, and published DataHub links for eleven weeks nobody has built.
weeks = [s for s in c["schedule"] if s["modules"]
         and (ROOT / "docs" / "notebooks" / f'{s["slug"]}.ipynb').exists()]
_d = c["policy"]["drop_lowest"]
drop_line = ("No weekly notebook is dropped." if _d == 0 else
             "The lowest weekly notebook is dropped." if _d == 1 else
             f"The {_d} lowest weekly notebooks are dropped.")

rows = "\n".join(
    f"| {s['n']} | [{s['question']}]({link(s['slug'])}) | "
    f"{', '.join(mods[m]['topic'] for m in s['modules'])} | {s['field']} |"
    for s in weeks)

(ROOT / "docs" / "README.md").write_text(f"""# EPS 88 — PyEarth

*A Python Introduction to Earth Science* · {c['meeting']} · {c['units']} units
**Instructor** {c['instructor']}

{c['catalog_line'].strip()}

## Prerequisites

{c['prerequisites'].strip()}

## The weeks

Each link opens that week's notebook in your own DataHub account.

| | Earth-science question | Python | Field |
|---|---|---|---|
{rows}

## How the course works

One notebook a week. You work in it during class and continue in the same file at home — the
class questions are ones you have just been shown how to do, the homework asks something the
class deliberately did not answer. **All of it is your work and all of it is graded.** Submit the
whole notebook once, {c['policy']['exercise_due']}; solutions go up {c['policy']['solution_release']}.

## Marks

{" · ".join(f"{k.replace('_', ' ')} **{v}%**" for k, v in c['grading'].items())}

- **Weekly notebook** — one file per week, class work and homework together, submitted once.
  Due {c['policy']['exercise_due']}. {drop_line}
- **Participation** — observed in the room; there is nothing separate to upload.
- **Project** — a track notebook or your own question. Lightning talks {c['project']['talks']};
  notebook due {c['project']['due']}.

**Late work** — {c['policy']['late']}.

## What you will use

{" · ".join(f"`{lib}`" for lib in p['libraries'])}
""")
print("docs/README.md generated")
