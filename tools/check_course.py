#!/usr/bin/env python
"""Validate course.yml against modules.yml. Part of the CI checker."""
import pathlib, re, sys, yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
_mraw = yaml.safe_load(open('modules.yml'))
mods = {m['id']: m for m in _mraw['modules']}
mods_all = set(mods)
c = yaml.safe_load(open('course.yml'))
taught = [m for s in c['schedule'] for m in s['modules']]
order = {m: i for i, m in enumerate(taught)}
errs, warns = [], []

# The dataset audits live in ../notes/, OUTSIDE this repo — they are the authoring evidence base,
# not student material. A checkout that has only the repo (CI's, or anyone else's) cannot see
# them, and demanding them there failed 25 sessions on a condition no CI run can ever satisfy.
# The check is real where the notes are; where they are not, there is nothing to check.
HAVE_NOTES = (ROOT.parent / 'notes' / 'dataset-audit').is_dir()

for m in taught:
    if m not in mods: errs.append(f"{m} scheduled but not in the catalogue")
if len(taught) != len(set(taught)): errs.append("a module is scheduled twice")

for m in [x for x in taught if x in mods]:
    for r in mods[m].get('requires', []):
        if r not in mods:            errs.append(f"{m} requires {r}, which does not exist")
        elif r not in order:         errs.append(f"{m} requires {r}, which is never taught")
        elif order[r] > order[m]:    errs.append(f"{m} (session {order[m]+1}) requires {r}, taught later (session {order[r]+1})")

ns = c.get('not_scheduled', {})
unavailable = {x['id'] for k in ('flexible', 'reference', 'deferred') for x in (ns.get(k) or [])}
for m in [x for x in taught if x in mods]:
    for r in mods[m].get('requires', []):
        if r in unavailable:
            errs.append(f"RULE: live module {m} depends on {r}, which is not taught live "
                        f"— beginners skip that material first")

for s in c['schedule']:
    tot = sum(mods[m]['minutes'] for m in s['modules'])
    if tot > 100: warns.append(f"session {s['n']} ({s['date']}) is {tot} min of material")

# --- duplicate YAML keys ------------------------------------------------------
# yaml keeps the last and discards the rest, so a duplicated key deletes content silently.
class DupCheck(yaml.SafeLoader):
    pass


def _no_dups(loader, node, deep=False):
    out = {}
    for k, v in node.value:
        key = loader.construct_object(k, deep=deep)
        if key in out:
            errs.append(f"duplicate key '{key}' at line {k.start_mark.line + 1} of "
                        f"{k.start_mark.name} — yaml keeps only the last and discards the rest")
        out[key] = loader.construct_object(v, deep=deep)
    return out


DupCheck.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_dups)
for _f in ('course.yml', 'modules.yml'):
    yaml.load(open(_f), DupCheck)

# --- the stable catalogue must never name a term's week numbers ---------------
# Weeks move whenever a semester is re-planned; module ids do not. Notes that said
# "the bootstrap (week 7)" silently went stale in the Fall-2026 reorder and would have
# been copied into a notebook as fact. Say "S4", not "week 8".
raw_m = open('modules.yml').read()
for i, line in enumerate(raw_m.splitlines(), 1):
    if (re.search(r'\bweek\s+\d', line, re.I) and 'Generation' not in line
            and not re.search(r'offerings/|\.ipynb', line)):
        errs.append(f"modules.yml:{i} names a week number — use the module id instead")
for f in ('modules.yml', 'course.yml'):
    if '\\"' in open(f).read():
        errs.append(f"{f} contains literal backslash-quotes — a value is mis-escaped")

# --- every taught session needs homework -------------------------------------
for s in c['schedule']:
    if s['modules'] and not s.get('exercise'):
        errs.append(f"session {s['n']} ({s['date']}) has no exercise")

# --- a track without a genuine open question is a re-run of a lecture -------
import os
for tr in c['project']['tracks']:
    if not tr.get('open_question'):
        errs.append(f"project track {tr['id']} has no open_question — it is not ready to ship")
    ev = tr.get('evidence')
    if tr['id'] != 'T6':
        if not ev:
            errs.append(f"project track {tr['id']} cites no evidence file")
        elif HAVE_NOTES and not os.path.exists(os.path.join('..', ev)):
            errs.append(f"project track {tr['id']} cites {ev}, which does not exist")

# --- every scheduled week carries the keys the build pipeline reads -----------
for s in c['schedule']:
    for k in ('slug', 'takeaways'):
        if not s.get(k):
            errs.append(f"session {s['n']} has no {k}: — the build script and the week summary "
                        f"both read it")
    if s['modules'] and not s.get('pinned'):
        errs.append(f"session {s['n']} has no pinned: — the data choices would live in prose, "
                    f"where a builder re-decides them and nothing can check them")
    ev = s.get('evidence')
    if s['modules'] and not ev:
        warns.append(f"session {s['n']} cites no dataset audit (evidence:)")
    elif ev is not None and not isinstance(ev, list):
        errs.append(f"session {s['n']} evidence: is a string, not a list — anything iterating "
                    f"it walks characters one at a time")
    else:
        for e in (ev or []):
            if HAVE_NOTES and not (ROOT.parent / e).exists():
                errs.append(f"session {s['n']} cites {e}, which does not exist")

# --- the docs table stays scannable ------------------------------------------
# The Python column is a signpost, not an inventory: at most two items per week, so a week
# with two modules gets one each. Longer and the table stops being readable at a glance.
for s_ in c['schedule']:
    if not s_['modules']:
        continue
    cell = ", ".join(mods[m]['topic'] for m in s_['modules'] if m in mods)
    if len(cell.split(",")) > 2:
        errs.append(f"session {s_['n']} topic column is {len(cell.split(','))} items "
                    f"({cell!r}) — at most two")

# --- ids referenced anywhere must exist in the catalogue ----------------------
for d in _mraw.get('plain_words', []):
    if d['module'] not in mods_all:
        errs.append(f"plain_words '{d['idea']}' names module {d['module']}, which does not exist")
for tr in c['project']['tracks']:
    for mid in tr.get('uses', []):
        if mid not in mods_all:
            errs.append(f"track {tr['id']} uses {mid}, which does not exist")

# --- a file or tool named in the instructions must be on disk ----------------
import re as _re
# ROOT.parent/CLAUDE.md sits OUTSIDE the repo — it is the working-directory instruction file,
# not a tracked one. Reading it unconditionally crashed check_course.py in every checkout that
# is not Weiqiang's laptop, which is every CI run.
for doc in (ROOT / 'TEMPLATE.md', ROOT.parent / 'CLAUDE.md'):
    if not doc.exists():
        continue
    for m in _re.finditer(r'`(tools/[\w./]+\.py|data/[\w./-]+\.csv|docs/[\w./-]+\.md)`',
                          doc.read_text()):
        ref = m.group(1)
        if 'weekNN' in ref:            # a per-week template name, not a real path
            continue
        if not (ROOT / ref).exists():
            errs.append(f"{doc.name} names {ref}, which does not exist")

# --- the grading comment must agree with the grading number ------------------
# Only the grading block: elsewhere a "key: 1983  # 4124 events" is a pinned value with a
# note, not a total with parts, and reading it as arithmetic produced two false errors.
for line in open('course.yml'):
    m = _re.match(r'\s*(\w+):\s*(\d+)\s*#\s*(.+)', line)
    if not m or m.group(1) not in c['grading']:
        continue
    parts = [int(x) for x in _re.findall(r'(\d+)', m.group(3))]
    if parts and len(parts) > 1 and sum(parts) != int(m.group(2)):
        errs.append(f"grading '{m.group(1)}' is {m.group(2)} but its comment sums to "
                    f"{sum(parts)}: {m.group(3).strip()}")

if sum(c['grading'].values()) != 100:            errs.append("grading does not sum to 100")
if sum(c['project']['rubric'].values()) != 100:  errs.append("rubric does not sum to 100")

print(f"{len(c['schedule'])} sessions · {len(taught)} modules taught · "
      f"{len(mods)-len(taught)} not scheduled"
      f"{'' if HAVE_NOTES else ' · evidence files not in this checkout, not checked'}")
for w in warns: print(f"  warn  {w}")
for e in errs:  print(f"  ERROR {e}")
print("OK" if not errs else f"{len(errs)} error(s)")
sys.exit(1 if errs else 0)
