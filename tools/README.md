# tools/

Everything here runs instructor-side. Students never see any of it.

| Tool | What it does |
|---|---|
| `check_course.py` | Validates `course.yml` against `modules.yml`. **Run after every edit.** Catches duplicate YAML keys (they silently discard content), missing prerequisites, ids that don't exist, files named in the instructions that aren't on disk, and grading comments that don't sum. It does **not** open a notebook. |
| `agent_brief.py build\|review N` | Generates the brief for a week's builder or reviewer. **Generate them; never hand-write one** — two hand-written briefs for the same job drift apart immediately. |
| `prior_knowledge.py N` | What students know entering week N, and what arrives later. This is what makes it safe to build weeks in parallel. |
| `weekkit.py` | The standards list (one source, rendered flat for the builder and as gated tiers for the reviewer) and `week_cheatsheet()`, the generated closing summary. |
| `make_schedule.py` | Regenerates `SCHEDULE.md`. |
| `make_mkdocs.py` | Regenerates `mkdocs.yml`. The nav comes from `course.yml`, so the site cannot drift from the plan — the previous repo's hand-written nav pointed at eleven notebooks that no longer existed. |
| `make_docs.py` | Regenerates `docs/README.md`, including every week's DataHub link. |
| `make_coastlines.py` | Builds `data/coastlines.csv` — every map draws it. One `plt.plot`, no loop. |
| `make_plate_boundaries.py` | Builds the plate-boundary CSV from shapefiles, stdlib only, so cartopy isn't needed. |
| `make_phasenet_subset.py` | Builds `data/phasenet_ncedc.npz` for week 13 from `gs://quakeflow_dataset/NCEDC`. |

A week's own build script is `build_weekNN.py`, written by whoever builds that week. It must emit
both notebooks from one source so the student version cannot drift from the solution.
| `check_notebook.py` | Validates a BUILT week: counts, banned strings, asserts that cannot fail, figures, imports, summary drift. Run by the build gate. |
| `check_prior_knowledge.py` | Does week N use anything the course has not taught by week N? The only check that sees across weeks. |
| `selftest_checks.py` | Proves every check_notebook rule fires on its motivating defect and stays quiet on the near-miss beside it. |

## Validating

    python tools/check_all.py     # the plan, the checkers themselves, then every built week

## Running a build or a review

Give the agent the COMMAND and nothing else:

    python tools/agent_brief.py build 3     # then follow what it prints
    python tools/agent_brief.py review 3

Writing the brief to a file first and pointing the agent at the file makes a snapshot: edit the
generator afterwards and the agent follows a stale brief. That happened once — a build was
running against a brief whose loop had already been deleted, and had to be stopped. Having the
agent run the generator itself means it cannot read a version that no longer exists.

**Never hand-write context into the agent prompt.** Not for the builder, not for the reviewer.
Everything an agent needs about a week belongs in `course.yml` — `pinned:`, `note:`,
`note_critical:` — where every future build reads it, the checkers can see it, and it survives
the session. Context typed into a prompt is invisible to all three.

The two failures this rule comes from. A reviewer was handed five bullets of design context
including "do not report this as a defect", which destroys the signal twice over: if the reviewer
is right you never hear it, and if the spec is unclear you never learn that either. And a builder
was handed four points of week-specific setup, of which four were already in the spec and the
fifth — the grids' orientation — was a real gap that should have been fixed in the plan rather
than papered over in a prompt.
