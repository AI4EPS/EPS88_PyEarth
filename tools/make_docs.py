#!/usr/bin/env python
"""Generate docs/README.md from course.yml and modules.yml.

Everything students read about the shape of the course comes from the plan, so the site cannot
say one thing while course.yml says another. ONE page: it is both the GitHub landing page and the
mkdocs home, and a separate syllabus.md only duplicated it.
"""
import pathlib, re, urllib.parse as up, yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
c = yaml.safe_load((ROOT / "course.yml").read_text())
mods = {m["id"]: m for m in yaml.safe_load((ROOT / "modules.yml").read_text())["modules"]}
p = c["platform"]

# "Weiqiang Zhu <zhuwq@berkeley.edu>" is a mail header, and <...> is an autolink only on GitHub;
# mkdocs parses it as an HTML tag and drops it, so the site showed the name with no address.
_m = re.match(r"(.*?)\s*<(.+?)>\s*$", c["instructor"])
instructor_line = (f"{_m.group(1)}](mailto:{_m.group(2)})".join(["Instructor ", ""])
                   if _m else f"Instructor {c['instructor']}")
instructor_line = (f"Instructor [{_m.group(1)}](mailto:{_m.group(2)})" if _m
                   else f"Instructor {c['instructor']}")


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

# Reference-style links. Inline, every row carried a ~250-character DataHub URL, so the table
# source was one unreadable column of percent-encoding and a row could not be checked by eye.
# The rendering is identical; the definitions collect at the foot of the file.
rows = "\n".join(
    f"| {s['n']} | [{s['question']}][w{s['n']}] | "
    f"{', '.join(mods[m]['topic'] for m in s['modules'])} | {s['field'].capitalize()} |"
    for s in weeks)
link_defs = "\n".join(f"[w{s['n']}]: {link(s['slug'])}" for s in weeks)


def longdate(v):
    """2026-12-07 -> Monday 7 December. An ISO date is a machine's format; course.yml keeps it
    so the plan stays sortable and unambiguous, and the page a student reads spells it out.

    yaml resolves a bare date to datetime.date and a date-with-time to datetime.datetime, so
    what arrives here is one of three types, not always the string it looks like in the file.
    """
    import datetime as _dt
    if isinstance(v, _dt.datetime):
        return v.strftime("%A %-d %B, %H:%M")
    if isinstance(v, _dt.date):
        return v.strftime("%A %-d %B")
    d, _, clock = str(v).partition(" ")
    y, m, dd = (int(x) for x in d.split("-"))
    out = _dt.date(y, m, dd).strftime("%A %-d %B")
    return f"{out}, {clock}" if clock else out


# Heaviest component first. The dict is in course.yml's own order, which put participation --
# 10% -- at the head of the line, so the first number a student read was the smallest one.
grading_line = " · ".join(f"{k.replace('_', ' ')} **{v}%**"
                          for k, v in sorted(c['grading'].items(), key=lambda kv: -kv[1]))

(ROOT / "docs" / "README.md").write_text(f"""# EPS 88 — PyEarth

*A Python Introduction to Earth Science*

{c['meeting']} · {c['units']} units · **{instructor_line}**

{c['catalog_line'].strip()}

## Prerequisites

{c['prerequisites'].strip()}

## The weeks

Each link opens that week's notebook in your own DataHub account.

| Week | Earth-science question | Python | Field |
|---:|---|---|---|
{rows}

## How the course works

One notebook a week. You work in it during class and continue in the same file at home. The
class questions apply a method you have just been shown; the homework asks the question the
class deliberately left open. Both halves are your own work, and both are graded. Submit the
notebook once, {c['policy']['exercise_due']}; solutions are published {c['policy']['solution_release']}.

## Grading

{grading_line}

- **Weekly notebook** — one file per week, class work and homework together, submitted once.
  Due {c['policy']['exercise_due']}. {drop_line}
- **Participation** — assessed from your contribution in class; nothing is submitted separately.
- **Project** — a track notebook or a question of your own. Lightning talks
  {longdate(c['project']['talks'])}; notebook due {longdate(c['project']['due'])}.

**Late work** — {c['policy']['late']}.

## What you will use

{" · ".join(f"`{lib}`" for lib in p['libraries'])}

{link_defs}
""")
print("docs/README.md generated")
