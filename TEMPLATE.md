# How to build a week

**ONE notebook per week.** Students open one file, work in it during class, continue in it at
home, submit it once.

```
docs/notebooks/07_transformations_solution.ipynb   written FIRST; gitignored until release
docs/notebooks/07_transformations.ipynb            derived: the solution with the answers deleted
```

`tools/build_weekNN.py` emits both from one source so they cannot drift. Links and the closing
summary are generated from `course.yml` and `modules.yml`, never typed.

---

## 1. The shape of a notebook

**100 teaching minutes**, of a 120-minute Monday. Budget ~70 to the core and ~30 to the buffer —
but read YOUR week's figures in `modules.yml`, which are smaller and vary because several Mondays
run two modules.

**Everything in the notebook is the student's work, and all of it is graded.** Class cells are not
a demonstration they watch: they fill those in during class, having just been shown how. Coming to
class is what makes them doable.

So the file must make one thing unmissable — **where they have to write something.** Every answer
cell looks identical wherever it appears, and a student scrolling can find all of them without
reading a word of prose.

The class/homework split is about **when**, not whose work it is:

| | When | What makes it doable |
|---|---|---|
| **Class questions** | in the room, with you | you have just shown them the move |
| **Homework questions** | alone, afterwards | class built the machinery; the question is new |

### Sections, in order

0. **The opening** — `weekkit.OPENING`, with only `{question}` and `{hook}` filled in. The
   DataHub link, how to submit, what a pencil means and how to recover from a broken kernel are
   identical in week 13 and week 1; do not rewrite them.
1. **The hook** — the week's Earth-science question, ~150 words, into `{hook}`.
2. **What you'll be able to do** — the science *and* the technical skill, naming the functions.
3. **Setup** — one imports cell, one data cell, both out of `weekkit`, so every week's setup is
   identical. `weekkit.setup_cell()` carries the plot defaults and the
   live-source-with-cached-fallback pattern, and is what a week reads a live archive with.
   **A dataset that ships with the course is the exception, and takes
   `weekkit.asset_setup_cell()` instead** — the same cell with the pretend live read removed, so
   the data arrives as one `pd.read_csv(CACHE + "/name.csv")`. It has no upstream to be live from:
   wrapped in the pattern, both branches of the try/except read the same file from the same
   repository, the except branch can only fire in conditions that would kill the fallback too, and
   the docstring promising to "fall back to the copy stored with the course" is false, because the
   try branch already is that copy. Week 11 shipped exactly that before anyone noticed.
4. **Sections** — prose → a worked example typed together → a question. Roughly every 8–10
   minutes. A section that needs state from an earlier one opens with `weekkit.CHECKPOINT`.
5. **The buffer.** `modules.yml` calls it `extension`; it is a budget for you, not a section for
   students, and it is **invisible in the notebook** — no label, no banner, and no sentence about
   possibly not reaching it. "EXTENSION", "Buffer", "If we have time" and "we may not get to this
   in class" all read as a third category and make a student wonder whether that part is really
   theirs. It is. The last section looks exactly like the others and is graded like the others.
   **Nothing later in the course may depend on it.**
6. **The question, answered** — `weekkit.CLOSING_HEADING`, then one sentence.
7. **The week summary** — `weekkit.week_cheatsheet(n, [module ids])`, placed **before** the
   homework. Because it sits there, its `takeaways:` must cover only what class taught: a takeaway
   that states the homework's discovery gives it away.

   **Ask of every function in it: will a student need to remember this?** If not, mark it
   `remember: false` in `modules.yml` — the notebook may still use it and `check_prior_knowledge`
   still counts it as taught, but it stays out of the table. `set_aspect("equal")` and
   `locator_params(integer=True)` are formatting; a summary that lists them alongside `max()`
   tells a student the two matter equally, and a summary that lists everything teaches nothing.

### Two devices that must appear

**Predict before you run.** Before any surprising result, a cell asking them to commit to a number
first — committing to a wrong answer beats being shown the right one. The heading is exactly
`### Predict before you run`, a convention like the ✏️ on a question: without one, two weeks used
two forms and no two reviewers graded the same cell.

**Checkpoints.** A section that needs state from an earlier one opens with `weekkit.CHECKPOINT`,
rebuilding it — including the scalars, not just the dataframes. The difference between a student
losing ten minutes and losing the day.

They collide: a checkpoint that **prints** a count gives away a prediction later in the same
section. Rebuild state silently, or put the prediction first.

### Three markers — for your planning, never written in the file

▶ WATCH (hands off the keyboard) · ✎ TOGETHER (type along) · ✔ YOUR TURN (alone, then compare).
Say them out loud. Not as code comments: students read a comment as code, and the distinction is
invisible once class is over. WATCH cells matter most — without them students transcribe
everything and hear none of the reasoning.

---

### How big a week is

**Aim at 50 cells and 8 questions**; 60 and 9 are ceilings, not goals. Every build so far has
landed exactly on whatever number this line names as the maximum, so it names the target first:
five or six questions in class, two or three the homework.
Trial builds ranged 44 to 88 cells and 7 to 19 questions, a difference a student would feel from
one week to the next. The room is 100 minutes and a beginner needs about ten of them per question,
counting reading it, typing it and getting it wrong once. If a week wants more, it is too big:
move something into the buffer.

**At most two questions a week are answered in prose.** This is a data-science course: they are
here to write code, and a week where they type more sentences than statements has drifted. Prose
earns its place where a number cannot carry the meaning — which of two defensible readings the
data supports, what a result rules out — never as a way to check they were paying attention.

---

## 2. Questions and answers

**Write the week's `takeaways:` before any question**, then table the questions against them. A
question must serve a takeaway **or** an item in its module's `teaches:`, and you must be able to
say which — takeaways are often pure science while `teaches:` is pure programming, so a good
syntax question can serve only the latter. A question serving neither is a drill: cut it or rebuild
it. One trivial win on day one is the single exception; a beginner needs one cell that simply
works.

**Every question is two cells: a markdown block that asks, then an empty cell that answers.**

```
markdown   ┌ ✏️ YOUR TURN
           │ Print the depth of the second earthquake, and the place where the third
           └ one happened. Counting starts at 0, so the second item is [1].
code       # ← your answer here
```

Never a question as a comment inside the cell the student types in: they cannot see where the
prompt ends and their work begins, and neither can you when marking 46 of them. Prose answers get
a markdown answer cell — *"(Double-click this cell and replace this line with your answer.)"*

**If a self-check names a variable, the prompt must give that name.** `assert len(my_mags) > 0`
raises `NameError` for every student who chose their own — from the cell whose whole job is
reassurance. Write ``**Use these names**, because the self-check looks for them.``

**Every question gets a complete model answer in the solution, prose included.** Write it in the
voice of a good student, using the numbers the notebook actually produced. A written question with
no model answer cannot be graded or handed to a reader.

**Never print a time estimate.** Budget the week in minutes if it helps you plan, but a student
slower than the number on screen learns only that they are behind.

---

## 3. Figures

**Execute the solution, then open every figure and look at it.** Not "check it ran" — look. A
scatter of twelve dots under the sentence *"they fall on lines, those lines are the plate
boundaries"* executes perfectly and is false.

- **Write the claim only after seeing the figure.** If the figure does not show it: change the
  data until it does, or drop the claim. Never keep the sentence and hope.
- **A thin figure is fine if you say it is thin.** *"Twelve earthquakes cannot show you a shape —
  hold that thought"* sets up a reveal. Hand-waving over a bad plot teaches that hand-waving is
  normal practice.
- **Informative and short.** Hoist styling into `plt.rcParams` in the setup cell so each plot cell
  holds only lines that change what the reader learns. Four to six: the plot, axis labels with
  units, a title with the sample size, `show()`. Anything else earns its place or goes, and if it
  stays, a trailing comment says why. A beginner reading eight lines of matplotlib cannot tell
  which one is the idea.
- **Every map draws the coastline.** One line, no loop — the file carries blank rows between
  segments so matplotlib lifts the pen, which is why this works in week 1:

  ```python
  coast = pd.read_csv(CACHE + "/coastlines.csv")
  plt.plot(coast.lon, coast.lat, color="0.6", lw=0.6)
  ```

  Plate boundaries (`tools/make_plate_boundaries.py`) go on the same way when a week needs them.
- **Always worth it**: axis labels with units · sample size in the title · `set_aspect("equal")` on
  maps · `plt.locator_params(axis="y", integer=True)` on count axes, since a y-axis reading 0.5,
  1.5, 2.5 says "half an earthquake" · explicit limits on a world map.
- **Usually not**: single-entry legends · `tight_layout()` · decorative colour · a `figsize` in
  every cell · gridlines that are only decoration. Grid earns its place where the lines carry
  meaning — reading a bar height, or meridians on a map.

---

## 4. The homework

**The homework is the question the class set up and deliberately did not answer.** Same data and
same tools — a fresh dataset would make a beginner fight loading and cleaning instead of
practising. A new question, because re-running the class code on your own birthday teaches typing.
**If part 1 can be completed by copying a class cell and changing one string, it is the wrong
part 1.**

1. **The new question** — something the week's tools answer that class did not ask. They may look
   back for the mechanics; the answer is not in the notebook above.
2. **The fork** — the week's one real decision, made by them, with the consequence reported. Two
   defensible choices that give different answers, and the solution works **both**, so neither
   reads as the one that was secretly wrong.
3. **Explain** — one paragraph about *their own numbers*, checked against the week's takeaways.
   Make them quote their own output back (*"your two counts differ by a factor of ___"*).

**Design the homework from the week's takeaways, not from the class cells.** Name the takeaway each
part serves before writing the part, and check the set: a takeaway no part touches was taught and
never used. Week 1's third takeaway — how much data you need is itself a scientific question — is
the kind that is easiest to leave out and worst to lose, because it is the week's actual argument.

**An "explain" part must make them reason from evidence they produced, not restate a definition the
summary already gave them.** If the answer is a phrase from the plain-words table, the question is
a copying exercise. Ask them to *find* something — at what point does the pattern appear, which of
two choices survives, what would have to be true for this number to make sense — and to report it
in two or three sentences about their own output. One task per part, not four.

The rules above are about the question. These are about whether a beginner alone at home can
finish it — which is what decides whether the homework grades understanding or stamina.

**Count the work; do not time it.** Your own wall clock is meaningless here — one builder honestly
reported 31 seconds. Report instead, per part: how many lines the student writes, how many ideas
are new, and whether the structure was rehearsed in class. About thirty lines across three parts,
with no more than three new ideas, all named in the prompt that needs them, is the size that fits
a freshman's two hours. More than that, cut a part.

**No syntax the week did not teach.** Every function a part needs was used in a class cell above,
or is in the week summary table. They have nobody to ask.

**A wrong part 1 must not cost them parts 2 and 3.** Where a later part needs a number from an
earlier one, print that number in the prompt or let the part stand alone. Compounding failure
grades persistence, not understanding.

**Part 1 is the one everybody finishes; part 3 is the one that separates them.** Nobody submits
blank, and nobody coasts.

**If the data is theirs and cannot be cached** — 46 unknown birthdays — the load must fail loudly
with a message naming the fix. It must never fall back to someone else's data and report it as
theirs, and the prompt must say what to do when the network is down.

**A self-check after every part where one is possible, catching the mistake a student will actually
make** — did you replace the placeholder, did you change the parameter, did you commit before
looking. End with a `print` echoing their numbers, so a pass says something.

Some parts cannot be checked — "did you draw a figure" is invisible from a variable. Ship no assert
rather than a decorative one, and say in the prompt what a right answer looks like.

**No reflection essays.** *"Your model predicted 1.6 and reality gave 5 — do you believe the model
or the data?"* is worth reading. *"What surprised you?"* is not.

The final project carries an AI-disclosure section; a weekly notebook does not.

---

## 5. Introducing an idea — plain words for hard ideas

**Every advanced idea enters through one plain sentence, one line of code different from something
they know, and one real consequence. Never through its formal name or its mathematics.** The name
may follow, once the idea is understood.

The sentences are data, in `plain_words:` at the top of `modules.yml`, and they are **binding: the
sentence a student meets must be the sentence in the table.** One idea, one wording, all term. Not
only for machine-learning ideas — equilibrium temperature and the greenhouse effect need fixed
wording as much as DBSCAN does, because later weeks refer back to them.

A module may have no entry, and several do not; the generated summary omits that section. If you
invent a sentence for a hard idea while building, **report it** so it can be added — otherwise the
next week to mention it invents a second one.

Three fields feed the summary; fill them as you build:

| Field | Where | What it is |
|---|---|---|
| `takeaways:` | the week, `course.yml` | two or three sentences a student should still have a year later — written *before* the questions |
| `functions:` | the module, `modules.yml` | each new function and what it does, in the notebook's own words |
| `plain_words:` | top of `modules.yml` | each hard idea in one sentence |

---

## 6. Borrowing from a later week

A week may need something it has not taught. Fine, in a setup cell, **as long as you say so**:

> **Coming later:** the second uses **pandas**, which we meet properly in the tables week, to fetch the catalogue and hand us six
> lists. You are not expected to follow it yet.

**Never reach outside the six libraries to dodge this** — `csv`, `urllib` and `io` are three new
ideas to avoid one honest forward reference. And **name new syntax the first time it appears**,
even trivial syntax: the dot in `mags.index(...)` is a new idea in someone's first hour.

---

## 7. Build once, then two gates

**Write the solution first and derive the student version by deleting code — never the reverse.**
Fall 2024's exercises were 41–86 % byte-identical to their lectures, with no marker showing which
cells were the student's. Deriving from the solution makes that impossible.

Building and reviewing are different jobs done by different agents. Yours is to build it once and
prove it is not broken; judging whether it is *good* belongs to someone who did not write it. An
earlier version of this section asked the builder to re-read its own work several times: it was
where most of the build time went, and the notebook that came out was self-graded PASS on every
standard and then returned by the reviewer with two blocking defects.

So the build ends at two gates, and both are objective:

1. **The solution executes clean on a fresh kernel** — no scaffolding cell, no redirect, no
   `allow_errors`, execution counts contiguous from 1. Outputs existing is not the same as it
   having run; a deleted setup cell leaves counts starting at 2, and that has happened.
2. **`python tools/check_notebook.py N` reports OK**, and the build script runs it and refuses to
   finish if it fails. A build that does not pass the checker is not a build.

If you add a rule to the checker, add to `tools/selftest_checks.py` both the case that must trip
it and the near-miss that must not. Two rules there were dead when written and three had false
positives; a check nobody has watched fail is decoration that reads as coverage.

**The standards list is what the reviewer grades against:**

    python tools/weekkit.py

Build to it. `agent_brief.py` renders it flat for the builder and as three gated tiers for the
reviewer — one list, so a notebook cannot pass one and fail the other.

**The loop is outside the builder**: build → check → review → fix → check → review. Two review
cycles at most. A third means the specification is wrong rather than the notebook, and it goes to
Weiqiang instead.

---

## 8. Conventions

### The code is teaching material

Every line a student reads is a line they will imitate, so the code carries as much of the lesson
as the prose. Write it to be **copied**, not admired.

- **The plainest form that works.** Clarity to a beginner outranks brevity and outranks idiom: a
  `for` loop they can trace beats a comprehension they cannot, and neither belongs in a week
  before its module. Cleverness in a notebook is a cost paid by 46 people.
- **Every line earns its place.** If deleting a line changes nothing a student sees, delete it.
  Names read as English — `mags`, `year_mags`, never `x`, `m1`, `tmp` — and the same thing keeps
  the same name across the notebook and across weeks.
- **One idea per cell**, and one job per function, with a single-line docstring saying what it is
  for. A helper that needs a paragraph to explain is two helpers.
- **Say a thing once.** The same six lines appearing three times is a function — once functions
  exist. Before that week, it is a checkpoint cell, and that is the only reason to repeat.
- **The same shapes every week.** Setup, loading, plotting and checking look identical in week 12
  and week 1, so by December the only new thing on the screen is the science.

- **A large binary asset has exactly one door.** A `.npz` or similar too big for `data/` goes to
  a GitHub release, and `torch.hub.download_url_to_file(url, name)` is the ONLY route in the six
  libraries that fetches a URL to disk — `np.load` takes no URL and `urllib` is standard library,
  which is closed. Wrap it in the same try/except shape as `SETUP_CELL`, write no `data/` copy,
  and add the filename to `.gitignore`: the notebook downloads it into the working directory, and
  nothing else stops 42 MB being committed into a repo nbgitpuller clones onto 46 accounts.
- **It is graded as a PDF.** Students export the notebook and upload it to Canvas, and it is read
  on screen, not run. So everything that earns marks has to be legible in print: no output wider
  than the page (a printed line clipped at the right margin is ungradeable, and a twenty-column
  DataFrame is exactly that), figures readable at page width and never relying on colour alone,
  and every self-check ending in a `print` that states the student's own numbers — that line is
  what the grader actually reads. The ✏️ headings and the identical answer cells are what make
  46 PDFs navigable; they are not decoration.
- **Only the libraries in `course.yml`'s `platform: libraries:`** — that list is the data, and
  `check_notebook.py` reads it from there. A ceiling on the whole course; the standard library is
  not a loophole.
- **A live query that feeds a number written into prose must be reproducible.** Pin what can be
  pinned in `course.yml` — a date range, a magnitude floor, a version. Where the query has nothing
  to pin — some archives return whatever they hold that day — say in `course.yml`
  that **the cached CSV is authoritative for the prose**, and write the prose numbers from the
  cache. A week whose markdown hardcodes eleven numbers off a query that can silently change is a
  week that goes wrong quietly, months later, in front of students.
- **A constant read off a web page is cited with the value AND the date you read it.** Earth's
  Bond albedo on the NASA fact sheet changed from 0.306 to 0.294 during this course's own
  lifetime, and a citation with no read-date cannot be checked or defended.
- **No OFFERING dates in a notebook** — no term, no meeting date, no due date; those live in
  `course.yml` and the notebooks outlive the offering. **Name a topic, never a week number** —
  "we meet pandas properly in the tables week", not "(week 3)": weeks move when a semester is
  re-planned, which is why `check_course.py` already forbids week numbers in the catalogue, and
  the argument is stronger for the artifact than for the plan. Dates that are *data* are fine and often
  necessary: a pinned `starttime`/`endtime`, a demo birthday, the year a catalogue covers. Pin
  them, and record them in `course.yml` so the numbers reproduce.
- **No local file paths** — always a URL. Every live cell reads its cached fallback from
  `platform: cache_base:`, and the CSV must be pushed before release. Exempt: a query built from
  data the student supplies, which cannot be cached and should fail loudly instead.
- **One submission per week**, Sunday 23:59, the whole notebook. Participation is observed in the
  room, not uploaded.
- **Never edit a released notebook in place.** nbgitpuller keeps the student's copy; fixes go in a
  new file, and the announcement is "delete your copy and click the link again."

---

## 9. Record what changed from the harvest

Most weeks start from an archived notebook named in `modules.yml` (`harvest:`). When done, record
what you changed in that module's `harvest_changes:` — each item a change and its reason. Not a
diff; the reasons are the point, and they are the fastest way to review a week, because the
harvest is familiar and the changes are the only new thing.

---

## 10. What the checkers enforce

`tools/check_all.py` runs them all; `tools/README.md` says what each one is for. Anything they
catch is not yours to check by hand — that is the point of their existing.
