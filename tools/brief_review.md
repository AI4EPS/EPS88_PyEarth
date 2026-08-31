You are reviewing **week {n}** of EPS 88 "PyEarth", a course for freshmen with no
programming experience. Someone else built it. You have not seen their reasoning, and you should
not go looking for it — that blindness is the point of asking you.

**The week's question:** {question}

## What to read

1. `{REPO}/tools/_week{n:02d}_spec.md` — week {n}'s own specification: its `question`,
   `exercise`, `takeaways`, `pinned:` values and any `note:`, copied verbatim from the plan.
2. `{REPO}/TEMPLATE.md` — the rules the notebook was built to. The graded standards are
   reproduced below and they are what you grade against; read TEMPLATE for the reasoning behind
   them, not for extra criteria.
3. This week's dataset audit: {audits} — measured, and it outranks the notebook
4. `{REPO}/{student}` — **read this one first, straight through, as a student would.**
5. `{REPO}/{solution}` — the executed version, for checking outputs and figures

Do NOT read `{script}` until after you have formed a judgement from the notebooks. Reading the
generator first tells you what the author intended, which is exactly what a student will not have.

## First, let the machines do their half

    {PY} tools/check_notebook.py {n}
    {PY} tools/check_prior_knowledge.py {n}

Report their output verbatim. They enforce the mechanical standards — counts, banned strings,
labels, asserts that cannot fail, anything used before its week — so you can spend your attention
on what they cannot see. If either reports something, say so; if you think either is WRONG, say
that too and show your hand count. Twice now a reviewer has proved a checker wrong, and both times
the checker was fixed.

## Then do what only a reader can

- **Open every figure as an image and look at it.** Then check the sentence printed beside it. A
  claim not visible in its figure is the single most common defect in this course, and it has
  survived clean execution more than once.
- **Recompute every number quoted in prose.** Ratios, counts, factors. Run the code yourself.
- **Check the science, not just the arithmetic.** Read every Earth-science sentence and ask whether
  a specialist in that field would sign it. This is a separate job from checking the numbers, and
  it is the one that matters most: a correct computation can sit under a wrong mechanism. Real
  examples from this course, all of which executed cleanly — "the catalogue is in time order,
  earliest first" (the endpoint returns newest first), "Earth fails its own test while Venus
  passes" (only true if you use two different albedos), "the July excess is seasonal" (it is a
  placeholder date). Check interpretations against the week's `evidence:` audit, and say plainly
  when a claim is beyond what the data can support.
- **Run each self-check as a student who followed the prompt exactly would**, including using only
  the variable names the prompt actually gave them. An `assert` on a variable the prompt never
  named fails for everyone; an assert that is true by construction tests nothing.
- **Check every question against the week's `takeaways:`.** A question serving none is a syntax
  drill. Say which takeaway each one serves, or that it serves none.
- **Confirm every question has a complete model answer in the solution**, prose questions included.
- **Check nothing arrives before its week.** Compare against `{PY} tools/prior_knowledge.py {n}`.

## Judge it in three tiers

The tiers say how much a defect matters, not when to stop looking. Review all three every time:
a notebook can be true and still not teach, and the Tier 3 findings are the ones nobody else in
this pipeline can produce.

**This list is the complete set of graded criteria.** If you find something you think should be
graded and it is not here, that is a defect in the specification, not in the notebook — put it in
the last section rather than marking the notebook down for it.

{standards_tiered}
**Items marked `[auto]` are already settled by the two checkers you ran.** Report what they said
and move on; do not re-verify them by hand. Your attention belongs on the rest.
{defects}
*Report EVERYTHING you find, then classify. Do not stop early, do not decide something is too
minor to mention, and do not filter for severity as you go — filtering is a separate pass and it
is not yours. A finding you withhold cannot be overruled; a finding you report costs one line.

Then tag each: Tier 1 blocks release. The VERDICT follows the size of the fix, not the tier that
blocked — the first reviewer to run this found two Tier 1 failures that were each a two-cell edit
and said so, correctly, that "REBUILD" overstated them. FIX when the work is sentence-level or
mechanical. REBUILD only when the week does not teach what it claims.*

## Report

A numbered list of findings, worst first, each tagged with its tier. For each: what is wrong,
where, and the evidence — the number you computed, or what the figure actually shows.

**Every Tier 1 finding carries the command that reproduces it.** One line someone else can paste.
Your findings will be checked before they are acted on, and a finding that can be confirmed in
thirty seconds gets fixed; one that has to be re-derived gets argued about.

**Length: as long as the findings need and no longer.** No preamble, no restating the brief, no
summary of your summary. End with a verdict:

- **SHIP** — meets every standard above
- **FIX** — specific defects listed, all of them mechanical
- **REBUILD** — a structural problem: the week does not teach what it claims, or a question set
  does not serve the takeaways

Then one more section: **anything in the specification that is ambiguous, missing or
contradictory** — judged from trying to grade against it. You are the second reader of these
rules; where you could not tell whether something passed, the rule is the problem, not the
notebook.

Say SHIP only if you would put it in front of 46 students on Monday. Do not soften a finding
because the notebook is otherwise good, and do not pad the list to look thorough — a short honest
review is worth more than a long polite one. Report no findings at all if there are none.

Do not edit any file. You are reading, computing and reporting.
