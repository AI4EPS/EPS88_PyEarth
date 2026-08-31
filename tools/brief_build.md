You are building **week {n}** of EPS 88 "PyEarth", a Fall 2026 UC Berkeley course for
freshmen with no programming experience.

## Read the specification. It is binding, and it is the source of truth — not this brief.

This brief tells you WHAT to produce and WHERE things are. It deliberately does not restate the
rules, because a summary of them would compete with the real thing. Read these in order:

1. `{EPS88}/CLAUDE.md` — section "To build a week"
2. `{REPO}/TEMPLATE.md` — **read all of it.** §7 is the gates your build must pass.
3. `{REPO}/tools/_week{n:02d}_spec.md` — **week {n}'s entire specification**: its `course.yml`
   entry and its modules' `modules.yml` entries, copied verbatim, with the `plain_words` that
   apply. Four keys carry the week's design and you read all four: **`pinned:`** is settled data —
   the slice, the constants, the results that must come out — and you neither re-choose it nor go
   measuring for a better one; **`note:`** and **`note_critical:`** record decisions already made
   rather than ones to re-take; **`flagship:`** designs the week's central problem. You do NOT
   need to open course.yml or modules.yml; this file is those two files' week-{n} content.
4. The dataset audit the spec's `evidence:` names, in `{EPS88}/notes/dataset-audit/`.
   **Its measured numbers and traps outrank your expectations.**

## Know what students already know

Run this first and obey it:

    {PY} tools/prior_knowledge.py {n}

You cannot read the other weeks' notebooks, and they may be built in any order. This contract is
how you avoid using something a later week is meant to introduce, and how you reuse the exact
wording of an idea already named.

## Environment

- Work from `{REPO}`
- Python: `{PY}` (pandas, numpy, matplotlib, sklearn, torch, nbformat, nbclient, yaml)
- Put every temporary file in `{SCRATCH}/wk{n}{variant}/` — never in the repo
- Students may import ONLY the libraries the spec file lists. The standard library is not a
  loophole for dodging one of them.

## Produce

- `{script}` — emits BOTH notebooks from one source so they cannot drift apart
- `{solution}` — **executed**, with every output and figure saved in it
- `{student}` — the student version, no outputs
- `data/week{n:02d}_*.csv` — a cached fallback for every live cell, read from `platform:
  cache_base:`. The one thing you write into the repo rather than scratch. `main` is pushed, so
  these URLs resolve once you commit; until then only the files already there will load.
  **Exempt: any query built from data the student supplies** — it cannot be cached, and it should
  fail loudly with a message naming the fix rather than quietly returning someone else's data.

## Do not edit anything else

Not `course.yml`, not `modules.yml`, not `TEMPLATE.md`, not `CLAUDE.md`. They are shared by all
fourteen weeks: a `functions:` entry you add for your week changes every other week's summary, and
a rule you relax is relaxed everywhere. If one is wrong — and on every run so far at least one has
been — **put it in your report**. The orchestrator applies it once, where it propagates correctly.

**Deliver this week at the scope asked.** Make routine judgement calls yourself; where the spec is
silent, choose and say what you chose. Do not widen the week, do not add a section nobody asked
for, and do not re-decide anything the `pinned:` block settles.

## When you are done

**Build it once.** You are not reviewing your own work — a separate reviewer does that, and does
it better, because it did not write this. Your job ends at three gates, all objective, all run by `weekkit.gate({n})` at the end of your
build script — which must refuse to finish if any fails:

1. **The solution executes clean on a fresh kernel**, with no scaffolding cell, no redirect, and
   execution counts contiguous from 1.
2. **`{PY} tools/check_notebook.py {n}{variant_flag}` reports OK.**
3. **`{PY} tools/check_prior_knowledge.py {n}` reports OK** — nothing used before its week.

**Read your prose against your own outputs before you stop** — open every figure, check every
sentence beside a number. That is writing, not reviewing, and it catches real errors.

What you do NOT do is grade yourself against the standards below. That was tried: it produced a
notebook self-graded PASS on 27 of 27 standards which the reviewer returned with two blocking
defects, and it was where most of the build time went.

The standards below are what the reviewer will grade you against. Build to them — but the check
that they hold is not yours to make.

{standards_flat}

If a gate will not pass, stop and report what and why. A reported failure is useful; a quiet one
is not, and a fabricated PASS is worse than either.
{defects}
**A separate reviewer will read your student notebook without seeing any of your reasoning.**
Write for that reader.

## Report

**1. Every number the notebook prints.** As a table. Each one must have come from code you ran in
this session, not from the plan, not from an audit, not from memory. If a number in your prose has
no printed source, that is a defect — fix it or report it.

**2. Every FAIL, and nothing else.** Do not walk the standards list emitting PASS — you are not
grading your own work, and a builder that did exactly that reported PASS on 27 of 27 standards for
a notebook the reviewer returned with two blocking defects. List only what you know is wrong, weak
or unfinished, with the evidence. An empty list is a fine answer if it is true.

**3. Each question against the standard it serves.** Number, class or homework, and which takeaway
or `teaches:` item it serves. A question serving neither is a drill and should not have shipped.

**4. Rounds** — how many, and what each one fixed.

**5. `functions:` and `plain_words:` entries a module is missing.** Give the name and a one-line
description. For an idea `modules.yml` already records, the recorded sentence is binding and your
notebook must use it verbatim — propose wording only for what is genuinely absent.

**6. Anything in the specification that was ambiguous, missing, contradictory, or that you had to
guess at.** This is worth as much as the notebook. Every previous round of these reports found
real defects in the specification and all of them were fixed — assume more remain. Where you made
a judgement call the spec does not cover, say what you chose and what it cost.

**Length: cover the substance and stop.** No padding, no restating the brief back, no summary of
your summary. Six sections; a table wherever a table is clearer than prose.
