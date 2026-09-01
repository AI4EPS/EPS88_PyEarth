#!/usr/bin/env python
"""Build project track T8 — "Which way did the ground first move, and can a machine tell?"

Emits both notebooks from one source so they cannot drift:

    docs/notebooks/T8_which_way_first_solution.ipynb   executed, every output saved
    docs/notebooks/T8_which_way_first.ipynb            the same file with the answers deleted

A TRACK is not a week (course.yml `project: track_notebooks:`). Two things differ, and both are
deliberate:

  * LESS HELP. No worked example before a question. The notebook loads the data and reproduces
    the ONE result the title names — that one line of arithmetic reads the first motion — so a
    student can trust the pipeline, and then stops helping. Everything after is a prompt in
    words and an empty cell.
  * IT DOES NOT CLOSE. There is exactly one self-check, on the load, and the notebook ends on an
    open question this course cannot answer.

The data is the same 44 MB .npz of NCEDC waveforms week 13 uses. It ships as a GitHub RELEASE
asset, not in data/, because nbgitpuller clones data/ onto every student account; there is
therefore no cached CSV and this script writes none. `torch.hub.download_url_to_file` is the only
route to a URL in the six libraries, and `phasenet_ncedc.npz` is already in .gitignore.

Every number that appears in prose or in a model answer is computed HERE, by the same code the
notebook runs, and formatted in. Nothing is typed from memory or copied from the plan — the
plan's own figures for this track came from an audit of an EARLIER build of the file and several
of them do not reproduce, which is why `PLAN DRIFT` lines are printed rather than patched.

    python tools/build_track_T8.py

Needs torch, which the shared base environment does not carry; run it with an interpreter that
has torch, numpy, matplotlib, sklearn, pyyaml and nbconvert.
"""
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/notebooks"
SLUG = "T8_which_way_first"

course = yaml.safe_load((ROOT / "course.yml").read_text())
modules = yaml.safe_load((ROOT / "modules.yml").read_text())
TRACK = next(t for t in course["project"]["tracks"] if t["id"] == "T8")
PLATFORM = course["platform"]

DATA_URL = ("https://github.com/AI4EPS/EPS88_PyEarth/releases/download/"
            "data-v1/phasenet_ncedc.npz")
# Kept outside the repository: 44 MB of waveforms must never end up in a clone that nbgitpuller
# copies onto 46 student accounts.
LOCAL = pathlib.Path(tempfile.gettempdir()) / "eps88_wk13_phasenet_ncedc.npz"

SAMPLE_RATE = 100
LEAD_IN = 10                 # samples of background kept before every pick, in every window
OFFSET = 2                   # the sample the one-line rule reads, counting from the pick
SEED = 88                    # the course number, fixed before anything was run
EPOCHS = 30                  # measured below; the sweep runs seven of these
WINDOWS = [0.1, 0.25, 0.5, 1, 2, 5, 10]

DEFAULT_THREADS = torch.get_num_threads()
torch.set_num_threads(4)


# ---------------------------------------------------------------------------
# 1. measure everything the notebook will say, with the notebook's own code
# ---------------------------------------------------------------------------
if not LOCAL.exists():
    torch.hub.download_url_to_file(DATA_URL, str(LOCAL), progress=False)

data = np.load(LOCAL)
waveform = data["waveform"]
polarity = data["polarity"]
labelled = (polarity == "U") | (polarity == "D")

vertical = waveform[labelled][:, 2, :]
p_index = data["p_index"][labelled].astype(int)
up = polarity[labelled] == "U"
event_id = data["event_id"][labelled]
snr = data["snr"][labelled]

M = {}
M["file_bytes"] = LOCAL.stat().st_size
M["n_all"] = int(len(polarity))
M["n_labelled"] = int(labelled.sum())
M["n_up"] = int(up.sum())
M["n_down"] = int((~up).sum())
M["n_unlabelled"] = M["n_all"] - M["n_labelled"]
# The audit reports the unlabelled entries as the literal four-character string "nan". They are
# not, in the file that ships: the column's dtype is <U1 and the 152 unlabelled entries are "".
M["unlabelled_values"] = sorted({str(x) for x in polarity[~labelled]})
M["n_events"] = int(len(np.unique(event_id)))
M["n_stations"] = int(len(np.unique(data["station"][labelled])))
M["n_samples"] = int(vertical.shape[1])
M["seconds"] = round(vertical.shape[1] / SAMPLE_RATE, 2)
M["p_min_s"] = round(float(p_index.min() / SAMPLE_RATE), 2)
M["p_max_s"] = round(float(p_index.max() / SAMPLE_RATE), 2)
M["max_abs"] = round(float(np.abs(waveform).max()), 4)
M["clip_frac"] = round(float((np.abs(waveform) >= 10).mean()), 3)
M["snr_med"] = round(float(np.median(snr)), 1)


def cut(seconds):
    """Every recording's vertical trace, from just before the pick to `seconds` after it."""
    n = int(seconds * SAMPLE_RATE)
    out = np.zeros((len(p_index), LEAD_IN + n), dtype="float32")
    for i in range(len(p_index)):
        out[i] = vertical[i, p_index[i] - LEAD_IN:p_index[i] + n]
    return out


# --- the split. By EARTHQUAKE: many recordings share one, and splitting rows at random puts the
# same earthquake on both sides of the wall.
rng = np.random.default_rng(SEED)
earthquakes = np.unique(event_id)
rng.shuffle(earthquakes)
is_train = np.isin(event_id, earthquakes[:int(0.7 * len(earthquakes))])
is_test = ~is_train
up_train, up_test = up[is_train], up[is_test]
M["n_train"] = int(is_train.sum())
M["n_test"] = int(is_test.sum())
M["n_train_events"] = int(len(earthquakes[:int(0.7 * len(earthquakes))]))

always_up = np.full(len(up_test), True)
M["baseline"] = round(float(accuracy_score(up_test, always_up)), 3)

# --- the one line of arithmetic
piece = cut(0.1)
background = piece[:, :LEAD_IN].mean(axis=1)
first_swing = piece[:, LEAD_IN + OFFSET] - background
rule_says_up = first_swing[is_test] > 0
M["rule"] = round(float(accuracy_score(up_test, rule_says_up)), 3)

# What the plan's own wording — "the sign of the FIRST sample after the pick" — actually scores,
# and the sweep behind the choice of OFFSET. Reported, because the two differ by five points.
M["rule_by_offset"] = {}
for off in range(0, 6):
    guess = (piece[:, LEAD_IN + off] - background)[is_test] > 0
    M["rule_by_offset"][off] = round(float(accuracy_score(up_test, guess)), 3)
M["rule_by_mean"] = {}
for k in range(1, 7):
    guess = (piece[:, LEAD_IN:LEAD_IN + k].mean(axis=1) - background)[is_test] > 0
    M["rule_by_mean"][k] = round(float(accuracy_score(up_test, guess)), 3)
BEST_K = max(M["rule_by_mean"], key=lambda k: M["rule_by_mean"][k])
M["best_k"] = BEST_K
M["rule_best_mean"] = M["rule_by_mean"][BEST_K]
# The third of Your turn 1's three choices, measured so the model answer's claim about which of
# them matters is checked rather than asserted: the lead-in barely moves the score, the offset
# moves it by fifteen points.
M["rule_by_lead"] = {}
for lead in [2, 3, 5, LEAD_IN]:
    quieter = piece[:, LEAD_IN - lead:LEAD_IN].mean(axis=1)
    guess = (piece[:, LEAD_IN + OFFSET] - quieter)[is_test] > 0
    M["rule_by_lead"][lead] = round(float(accuracy_score(up_test, guess)), 3)
M["spread_offset"] = round(max(M["rule_by_offset"].values()) - min(M["rule_by_offset"].values()), 3)
M["spread_lead"] = round(max(M["rule_by_lead"].values()) - min(M["rule_by_lead"].values()), 3)
best_line_test = (piece[:, LEAD_IN:LEAD_IN + BEST_K].mean(axis=1) - background)[is_test] > 0

# --- the two traces the opening figure draws. Chosen for LEGIBILITY only: a big clean first
# swing on a quiet background, one of each polarity, never for the answer they give.
clarity = np.abs(first_swing) / np.maximum(piece[:, :LEAD_IN].std(axis=1), 1e-6)
legible = (snr > 20) & (clarity < 200)
UP_TRACE = int(np.argsort(np.where(legible & up, clarity, -1))[-1])
DOWN_TRACE = int(np.argsort(np.where(legible & ~up, clarity, -1))[-1])
M["up_trace"] = UP_TRACE
M["down_trace"] = DOWN_TRACE


# --- the network
def network_says_up(seconds, epochs=EPOCHS, seed=0):
    """Train a network on `seconds` of waveform; one True/False per held-out recording."""
    # 1. Cut the same stretch of trace out of every recording. A 1-D convolution reads a stack
    #    of channels, so `unsqueeze(1)` inserts the length-1 channel dimension it expects.
    window = cut(seconds)
    x = torch.tensor(window).unsqueeze(1)
    x_train, x_test = x[is_train], x[is_test]
    # 2. What the network has to learn is a sign, so the answer it is shown is a sign: +1 where
    #    the first motion is up, -1 where it is down.
    y_train = torch.tensor(np.where(up_train, 1.0, -1.0).astype("float32")).reshape(-1, 1)

    # 3. Fix the random numbers before the layers are built, so the network starts from the same
    #    weights every time and a second run gives the same answer as the first.
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Conv1d(1, 8, 7, stride=2, padding=3), nn.ReLU(),
        nn.Conv1d(8, 16, 7, stride=2, padding=3), nn.ReLU(),
        nn.Flatten(),
        # Each convolution steps two samples at a time, so the trace comes out a quarter as
        # long as it went in; `Flatten` lays those 16 shortened channels end to end, which is
        # where 16 x a quarter of the window comes from.
        nn.Linear(16 * ((window.shape[1] + 3) // 4), 1))
    optimiser = torch.optim.Adam(net.parameters(), lr=0.005)
    loss_function = nn.MSELoss()
    # 4. Train for a fixed number of passes rather than stopping when the loss looks good
    #    enough: the sweep trains one network per window length, and comparing their scores is
    #    only fair if every network got the same amount of training.
    for epoch in range(epochs):
        order = torch.randperm(len(x_train))
        for start in range(0, len(x_train), 32):
            batch = order[start:start + 32]
            loss = loss_function(net(x_train[batch]), y_train[batch])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    # 5. Hand back one True/False per held-out recording, not the fraction it got right. A
    #    fraction can only be compared with another fraction; the answers themselves can be
    #    asked WHICH recordings the network missed.
    return net(x_test).detach().numpy().ravel() > 0


M["sweep"] = {}
M["sweep_seconds"] = {}
said_at = {}
for seconds in WINDOWS:
    t0 = time.time()
    said_at[seconds] = network_says_up(seconds)
    M["sweep_seconds"][seconds] = round(time.time() - t0, 1)
    M["sweep"][seconds] = round(float(accuracy_score(up_test, said_at[seconds])), 3)
BEST_WINDOW = max(M["sweep"], key=lambda s: M["sweep"][s])
M["best_window"] = BEST_WINDOW
M["network_best"] = M["sweep"][BEST_WINDOW]
M["network_worst"] = M["sweep"][WINDOWS[-1]]
M["sweep_total_s"] = round(sum(M["sweep_seconds"].values()), 1)
network_best = said_at[BEST_WINDOW]

# The clip did not only shave the tops off. 186 vertical traces are constant at exactly ±10 for
# all 2048 samples — the ones whose per-component normalisation had blown up before the file was
# cut, which the ±10 bound then flattened into a straight line. They carry no first motion at
# all, they are 7.9% of the labelled set, and they are where BOTH methods fall to chance. Nothing
# in the audit or the plan records them; Your turn 5 is where a student meets them.
flat = vertical.std(axis=1) == 0
M["n_flat"] = int(flat.sum())
M["flat_values"] = sorted({float(v) for v in vertical[flat][:, 0]})
M["n_flat_test"] = int(flat[is_test].sum())
M["rule_on_flat"] = round(float(accuracy_score(up_test[flat[is_test]],
                                               best_line_test[flat[is_test]])), 3)
M["rule_off_flat"] = round(float(accuracy_score(up_test[~flat[is_test]],
                                                best_line_test[~flat[is_test]])), 3)

# --- where the two disagree, at the network's best window against the best one line
M["net_on_flat"] = round(float(accuracy_score(up_test[flat[is_test]],
                                              network_best[flat[is_test]])), 3)
M["net_off_flat"] = round(float(accuracy_score(up_test[~flat[is_test]],
                                               network_best[~flat[is_test]])), 3)
M["n_disagree"] = int((best_line_test != network_best).sum())
M["disagree_flat"] = int(((best_line_test != network_best) & flat[is_test]).sum())
M["both_right"] = int(((best_line_test == up_test) & (network_best == up_test)).sum())
M["both_wrong"] = int(((best_line_test != up_test) & (network_best != up_test)).sum())
M["rule_only"] = int(((best_line_test == up_test) & (network_best != up_test)).sum())
M["net_only"] = int(((best_line_test != up_test) & (network_best == up_test)).sum())
M["agree"] = round(float((best_line_test == network_best).mean()), 3)

# --- the ceiling: how often the waveform agrees with the analyst, by how clear the swing is
agree_label = (first_swing > 0) == up
M["agree_all"] = round(float(agree_label.mean()), 3)
edges = np.quantile(clarity, [0, 0.25, 0.5, 0.75, 1.0])
M["agree_quartile"] = []
for q in range(4):
    inside = (clarity >= edges[q]) & (clarity <= edges[q + 1] if q == 3 else clarity < edges[q + 1])
    M["agree_quartile"].append(round(float(agree_label[inside].mean()), 3))
M["agree_top_decile"] = round(float(agree_label[clarity >= np.quantile(clarity, 0.9)].mean()), 3)

torch.set_num_threads(DEFAULT_THREADS)

for k in sorted(M):
    print(f"  measured  {k:>18} = {M[k]}")

# Does every number the plan quotes still reproduce? This USED to be a hand-transcribed table of
# course.yml's figures, which is two records of one fact and drifted the moment the plan was
# corrected: the transcription kept reporting DRIFT against numbers course.yml no longer claimed.
# So read the plan itself. Every 0.NNN in T8's open_question must be within 0.02 of something
# this build measured; one that matches nothing is a claim the notebook cannot support.
MEASURED = {"rule read literally (one sample after the pick)": M["rule_by_offset"][1],
            "rule at its best": M["rule_best_mean"],
            "network, best window": M["network_best"],
            "network at 0.1 s": M["sweep"][0.1],
            "network at 5 s": M["sweep"][5],
            "network at 10 s": M["sweep"][10],
            "label agreement, all": M["agree_all"],
            "label agreement, clearest quarter": M["agree_quartile"][3],
            "label agreement, clearest tenth": M["agree_top_decile"],
            "rule with the flat traces removed": M["rule_off_flat"],
            "network with the flat traces removed": M["net_off_flat"]}
print(f"  DATA DEFECT  {M['n_flat']} of the {M['n_labelled']} labelled vertical traces "
      f"({M['n_flat'] / M['n_labelled']:.1%}) are CONSTANT at exactly {M['flat_values']} for all "
      f"{M['n_samples']} samples — no arrival, no first motion, nothing to read. Both methods "
      f"score {M['rule_on_flat']} / {M['net_on_flat']} on them (chance) and "
      f"{M['rule_off_flat']} / {M['net_off_flat']} with them left out; "
      f"{M['disagree_flat']} of the {M['n_disagree']} disagreements between the two methods ARE "
      f"them. Recorded 2026-08-31 in notes/dataset-audit/ncedc-phasenet.md and in course.yml's "
      f"T8 open_question — the audit's own fix for the spikes (clipping at ±10) is what "
      f"flattened them.")
_plan_numbers = sorted({float(x) for x in re.findall(r"\b0\.\d{3}\b",
                                                    TRACK["open_question"])})
for planned in _plan_numbers:
    near = [(k, v) for k, v in MEASURED.items() if abs(planned - v) <= 0.02]
    if not near:
        print(f"  PLAN DRIFT  course.yml T8 quotes {planned}, which matches nothing this build "
              f"measured: {', '.join(f'{k} {v}' for k, v in MEASURED.items())}")
if TRACK["data"].count("1,233") == 0 or TRACK["data"].count("1,115") == 0:
    print(f"  PLAN DRIFT  course.yml T8 `data:` no longer names the label counts")
elif (M["n_up"], M["n_down"]) != (1233, 1115):
    print(f"  PLAN DRIFT  course.yml T8 `data:` says 1,233 U / 1,115 D; measured "
          f"{M['n_up']} U / {M['n_down']} D")


# ---------------------------------------------------------------------------
# 2. the summary, generated from modules.yml so the wording cannot drift
# ---------------------------------------------------------------------------
def idea(module_id, name):
    """One plain_words sentence, verbatim from modules.yml."""
    return next(d for d in modules["plain_words"]
                if d["module"] == module_id and d["idea"] == name)


def fn(module_id, name):
    """One function entry, verbatim from modules.yml."""
    return next(f for f in next(m for m in modules["modules"] if m["id"] == module_id)["functions"]
                if f["name"] == name)


# The ideas and calls this track leans on, named here and worded there. A track teaches nothing
# new, so the full module tables would list sixty functions it never uses; these are the ones the
# notebook and its model answers actually write.
TRACK_IDEAS = [("ML3", "Baseline"), ("ML2", "Train/test split"), ("ML2", "Overfitting"),
               ("ML6", "1-D CNN"), ("ML6", "Loss"), ("ML6", "Epoch")]
TRACK_FNS = [("ML3", "accuracy_score(y, pred)"),
             ("ML6", "np.unique(a)"), ("ML6", "np.isin(a, values)"),
             ("ML6", "torch.tensor(array)"), ("ML6", "torch.manual_seed(n)"),
             ("ML6", "nn.Sequential(layers)"), ("ML6", "nn.Conv1d(in, out, width)"),
             ("ML6", "nn.ReLU()"), ("ML6", "nn.Linear(in, out)"), ("ML6", "nn.MSELoss()"),
             ("ML6", "torch.optim.Adam(model.parameters(), lr=)"),
             ("ML6", "loss.backward() / optimiser.step() / optimiser.zero_grad()"),
             ("ML6", "torch.randperm(n)"), ("ML6", "tensor.detach().numpy()")]


def track_summary():
    out = [f"## What track {TRACK['id']} leans on", "",
           f"**The question.** {TITLE}", "",
           "Nothing here is new. These are the weeks to look back at while you work, and the "
           "wording is the course's own.", "",
           "Two calls this track needs are not in any of those tables, because no week has "
           "wanted them yet, and both are named where you first need them: "
           "`tensor.unsqueeze(1)`, which adds the single-channel dimension a `nn.Conv1d` "
           "expects, and `nn.Flatten()`, which lays a convolution's output out flat so an "
           "`nn.Linear` can read it.", "",
           "### The ideas, in plain words", "", "| Idea | Means |", "|---|---|"]
    out += [f"| **{d['idea']}** | {d['words']} |" for d in (idea(m, i) for m, i in TRACK_IDEAS)]
    out += ["", "### Code you will reach back for", "", "| Function | What it does |", "|---|---|"]
    out += [f"| `{f['name']}` | {f['does']} |" for f in (fn(m, n) for m, n in TRACK_FNS)]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 3. the cells
# ---------------------------------------------------------------------------
CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n"), None))


def code(text):
    CELLS.append(("code", text.strip("\n"), None))


def ask(text):
    """A question: the markdown that asks. The answer cell follows."""
    md(text)


def answer(model, check=""):
    """A code answer cell. The solution carries the model answer; the student gets the stub."""
    solution = model.strip("\n") + (("\n\n" + check.strip("\n")) if check else "")
    student = "# ← your answer here\n\n" + (("\n" + check.strip("\n")) if check else "")
    CELLS.append(("code", solution, student))


def answer_prose(model):
    CELLS.append(("markdown", model.strip("\n"),
                  "*(Double-click this cell and replace this line with your answer.)*"))


def blank_prose():
    """A section of the student's own project. It is empty in the solution too, because there is
    no model answer to a section whose content is the student's own work."""
    stub = "*(Double-click this cell and replace this line with your answer.)*"
    CELLS.append(("markdown", stub, stub))


datahub = (f"{PLATFORM['datahub']}/hub/user-redirect/git-pull"
           f"?repo={PLATFORM['repo'].replace(':', '%3A').replace('/', '%2F')}"
           f"&branch={PLATFORM['branch']}"
           f"&urlpath=lab%2Ftree%2FEPS88_PyEarth%2F{PLATFORM['notebook_dir']}%2F{SLUG}.ipynb")

# The title is the plan's, checked rather than copied: it carries no number, so nothing in it can
# have drifted, and the em dash is rewritten as a comma so the heading reads as one question.
TITLE = "Which way did the ground first move, and can a machine tell?"
if TRACK["title"] != TITLE:
    print(f"  note  course.yml T8 title: {TRACK['title']!r}\n"
          f"        on the page        : {TITLE!r} — the em dash is a comma so the heading reads "
          f"as one question; no fact in it has changed")

HOOK = f"""
When a fault slips, the very first thing it does to the ground at any one station is either push
it up or pull it down — and which of the two happens depends on where that station sits relative
to the fault. Read the first motion at enough stations and you can work backwards to the
orientation of the fault that broke and the direction it slipped, which is how almost every
earthquake mechanism in the record was determined before computers were involved. It is one
letter of information per recording: **U** or **D**.

An analyst reads that letter by eye, in a fraction of a second, from the first wiggle after the P
arrives. This notebook has {M['n_labelled']:,} of their readings — {M['n_up']:,} up and
{M['n_down']:,} down — on recordings of {M['n_events']} Northern California earthquakes, and the
question is whether a machine can do the same job. The interesting part is not whether it can. It
is how little of the seismogram it needs, and what it is doing with the rest.
"""

md(weekkit.OPENING.format(question=TITLE, datahub=datahub, hook=HOOK.strip()))

md("""
## How this notebook is different

This is a **project track**. It is not a weekly notebook and it does not behave like one.

A weekly notebook shows you a move, walks you through it, and then asks you to make it once
yourself. This one loads the data and reproduces the single result its title rests on — that one
line of arithmetic can read a first motion — and then stops helping. From there on every section
is a sentence describing what to find out and an empty cell to find it out in. There is no worked
example above to pattern-match against, because on a real question there never is one.

**There is exactly one self-check in this notebook, and it is on the data loading.** After that,
nothing tells you whether you are right. That is not an oversight and it is not laziness: past
the loading step there is no single right answer here, so a cell that said `assert` would be
lying to you about how research works. What replaces it is the thing researchers actually use — a
number you can get two ways, a result you can predict before you compute it, and a claim you can
try to break.

**And it does not close.** The last section is a question this course does not know the answer
to. Everything above it is scaffolding; that question is the project.
""")

md(f"""
## What you'll be able to do

**The science.** Say how much of a seismogram the polarity of the first motion actually lives in,
and defend the answer with a number rather than an adjective. Then say whether a trained network
is worth its cost on this problem, and what is stopping every method from doing better.

**The skills.** Split data by the structure that is really in it rather than by row. Write down
the dumbest possible method first and make everything else beat it. Train a small
one-dimensional convolutional network on a signal, and change one thing about its input at a time
until the change tells you something.

**The four questions, in order:**

1. Which way did the ground first move, and can one line of arithmetic tell?
2. How much waveform does the network need?
3. Did the network earn its place?
4. Is what is left the model, or the labels?

The open question at the end is not on that list. It is the project; the four above are what you
build to reach it.
""")

md(f"""
## Setup

The waveforms are {M['file_bytes'] / 1e6:.0f} MB — far too big to keep beside the notebook, so
they arrive from a release of the course repository the first time you run the cell below and are
kept on disk after that.

Three things about the file are worth reading before you start:

- `polarity` holds one character per recording: `"U"`, `"D"`, or nothing at all.
  {M['n_unlabelled']} recordings carry no polarity, and the only safe filter is the positive one
  — keep the rows that say `U` or `D`, rather than dropping the rows that look empty.
- Each waveform row has been divided by its own typical size, so a `1` means *one typical wiggle
  for this instrument on this recording*, not a fixed number of nanometres. The file is also
  already cut off at ±{M['max_abs']:.0f}: {M['clip_frac']:.1%} of all its samples sit exactly on
  that bound, so the tops of the largest arrivals have been shaved off. Any *size* you measure
  therefore has our ceiling on it — and it is worth holding on to the thought that a cut-off can
  do more to a recording than shave the top off a peak.
- Polarity is read on the **up-down** component, which is row 2 of the three. Rows 0 and 1 are
  the two horizontal directions, and the first motion is not defined on them.

The file also carries `magnitude`, `distance_km`, `depth_km`, `station` and `snr`, one per
recording. Any of them comes out the same way as the arrays below, filtered with the same mask.
""")

code(weekkit.download_setup_cell(
    imports="import numpy as np\nimport torch\nfrom torch import nn\n"
            "from sklearn.metrics import accuracy_score\n",
    const="WAVEFORMS", url=DATA_URL, filename="phasenet_ncedc.npz",
    docstring="Read the waveform file, downloading it from the course release the first "
              "time.",
    unpack=f'''
data = load()
labelled = (data["polarity"] == "U") | (data["polarity"] == "D")

vertical = data["waveform"][labelled][:, 2, :]   # the up-down component, one row per recording
p_index = data["p_index"][labelled].astype(int)  # sample number of the analyst's P pick
up = data["polarity"][labelled] == "U"           # True where the analyst wrote U
event_id = data["event_id"][labelled]            # which earthquake each recording is of
snr = data["snr"][labelled]                      # how much louder the quake is than the background
SAMPLE_RATE = {SAMPLE_RATE}                             # samples per second
LEAD_IN = {LEAD_IN}                                   # samples of background kept before every pick

print("recordings with a polarity:", len(vertical), "of", len(data["polarity"]))
print("first motion up:", up.sum(), " down:", (~up).sum())
print("earthquakes:", len(np.unique(event_id)), " stations:", len(np.unique(data["station"][labelled])))
print("each recording:", vertical.shape[1], "samples =",
      round(vertical.shape[1] / SAMPLE_RATE, 2), "seconds")
print("the P arrives between", round(p_index.min() / SAMPLE_RATE, 2), "and",
      round(p_index.max() / SAMPLE_RATE, 2), "seconds in")
print("largest value anywhere in the file:", round(float(np.abs(data["waveform"]).max()), 2))
'''.strip("\n")))

code(f"""
assert vertical.shape == (len(up), {M['n_samples']}), \\
    "the waveforms are not the shape this notebook expects — the file was read wrong"
assert 1000 < up.sum() < 1400 and 1000 < (~up).sum() < 1400, \\
    "the two polarities should be close to balanced; they are not, so the filter is wrong"
assert (p_index > LEAD_IN).all() and (p_index + 1000 < vertical.shape[1]).all(), \\
    "every pick must leave room for the windows below — some do not"
print(f"✓ the data — {{len(vertical)}} recordings carrying a polarity, {{up.sum()}} up and "
      f"{{(~up).sum()}} down, from {{len(np.unique(event_id))}} earthquakes")
""")

md("""
### And that is the last self-check in this notebook

The pipeline is now trustworthy: the file is the file, the filter is the filter, the numbers
below are the numbers. Everything from here is yours, and nothing will tell you when you have it
right.
""")

# --- section 1: the verified half -------------------------------------------
md(f"""
## Which way did the ground first move, and can one line of arithmetic tell?

The P wave is the compression that arrives first. If the fault's motion pushed the rock towards
this station, the ground's first move is **away from the source** — upwards at the surface — and
the analyst writes `U`. If it pulled, the first move is downwards, and they write `D`. Everything
after that first swing is the rest of the earthquake: more P energy, the S wave, reflections, the
ground ringing. None of it is the first motion.

So the whole of the label ought to live in the handful of samples immediately after the pick. The
figure below is two recordings, one of each letter, and it is worth looking at before anything is
computed.
""")

code(f"""
def cut(seconds):
    \"\"\"Every recording's vertical trace, from just before the pick to `seconds` after it.\"\"\"
    n = int(seconds * SAMPLE_RATE)
    out = np.zeros((len(p_index), LEAD_IN + n), dtype="float32")
    for i in range(len(p_index)):
        out[i] = vertical[i, p_index[i] - LEAD_IN:p_index[i] + n]
    return out
""")

code(f"""
half_second = cut(0.5)
time = (np.arange(half_second.shape[1]) - LEAD_IN) / SAMPLE_RATE

plt.plot(time, half_second[{UP_TRACE}], color="0.2", label="the analyst wrote U")
plt.plot(time, half_second[{DOWN_TRACE}], color="firebrick", label="the analyst wrote D")
plt.axvline(0, color="steelblue", lw=1.2)
plt.xlabel("seconds from the analyst's P pick")
plt.ylabel("ground motion (in units of this trace's own background)")
plt.title("Two first motions of the {M['n_labelled']:,}; the line is the pick")
plt.legend()
plt.show()
""")

md(f"""
Both traces sit on nothing until the blue line and then leave it, one upwards and one downwards,
within a few hundredths of a second. That is the entire signal this project is about.

Before measuring anything, the data has to be cut in two, and **how** it is cut is the first
place this problem can be got wrong. The {M['n_labelled']:,} recordings are of only
{M['n_events']} earthquakes, so the same earthquake is recorded at many stations at once. Split
the rows at random and most earthquakes end up on both sides of the wall — the model sees a
recording of an earthquake, then is scored on another recording of the same one. Splitting by
earthquake is the honest cut here.

**Train/test split:** {idea('ML2', 'Train/test split')['words']}
""")

code(f"""
rng = np.random.default_rng({SEED})
earthquakes = np.unique(event_id)
rng.shuffle(earthquakes)

is_train = np.isin(event_id, earthquakes[:int(0.7 * len(earthquakes))])
is_test = ~is_train
up_train = up[is_train]
up_test = up[is_test]

always_up = np.full(len(up_test), True)

print("training on", is_train.sum(), "recordings from",
      int(0.7 * len(earthquakes)), "earthquakes")
print("held out:  ", len(up_test), "recordings from the other", len(earthquakes) - int(0.7 * len(earthquakes)))
print("always say up:", round(accuracy_score(up_test, always_up), 3))
""")

md(f"""
That last number is the one every method in this notebook has to beat. `up` is slightly the more
common letter, so a rule that ignores the waveform completely and answers *up* every time is
already right about half the time.

**Baseline:** {idea('ML3', 'Baseline')['words']}

Now the rule the physics suggests, in one line: take the average of the {LEAD_IN} samples of
background just before the pick, and ask whether the trace has gone above it or below it a moment
later. `cut` puts those {LEAD_IN} lead-in samples at the front of every window, so
`piece[:, :LEAD_IN]` is the background and `piece[:, LEAD_IN]` is the sample at the pick itself.
The line below reads {OFFSET} samples further on, because an analyst's pick usually lands a
sample or two before the ground has actually started moving.
""")

code(f"""
piece = cut(0.1)
background = piece[:, :LEAD_IN].mean(axis=1)
first_swing = piece[:, LEAD_IN + {OFFSET}] - background
rule_says_up = first_swing[is_test] > 0

print("one line of arithmetic:", round(accuracy_score(up_test, rule_says_up), 3),
      "on", len(up_test), "held-out recordings")
print("always say up:        ", round(accuracy_score(up_test, always_up), 3))
""")

md(f"""
One subtraction and a comparison, no training, no fitting, nothing learned from the
{M['n_train']:,} training recordings at all — and it is right about four times in five. That is
the number the rest of this notebook is measured against, and it is deliberately the *first*
thing here rather than an afterthought.

`rule_says_up` is one True/False per held-out recording, not a score. Keep every method in that
shape: a method that hands back its accuracy can only ever be compared, while one that hands back
its answers can also be asked *which* ones it got wrong, which is where this project ends up.
""")

ask(f"""
### ✏️ Your turn 1

Three things about that line are choices rather than physics, and none of them was justified
above:

- **which sample** it reads — `LEAD_IN + {OFFSET}` was asserted, not argued;
- **how much background** it averages — `LEAD_IN` is {LEAD_IN} samples, and could be 3 or 100;
- **one sample or several** — `piece[:, LEAD_IN + {OFFSET}]` reads one number, but the mean of
  the first few samples after the pick is just as much "one line".

Try each. Score every version with `accuracy_score(up_test, ...)` so the numbers sit on the same
held-out recordings and can be compared, and print each one as you go.

Then print one more line answering it in a sentence, on your own numbers: **what is the best
score one line of arithmetic reaches on this data, and which of the three choices moved it
most?**
""")

answer(f"""
by_offset = []
for offset in range(0, 6):
    guess = (piece[:, LEAD_IN + offset] - background)[is_test] > 0
    by_offset.append(accuracy_score(up_test, guess))
    print("one sample,", offset, "after the pick —", round(by_offset[-1], 3))

by_lead = []
for lead in [2, 3, 5, LEAD_IN]:
    quieter = piece[:, LEAD_IN - lead:LEAD_IN].mean(axis=1)
    guess = (piece[:, LEAD_IN + {OFFSET}] - quieter)[is_test] > 0
    by_lead.append(accuracy_score(up_test, guess))
    print("background of", lead, "samples —", round(by_lead[-1], 3))

by_mean = []
for k in range(1, 7):
    guess = (piece[:, LEAD_IN:LEAD_IN + k].mean(axis=1) - background)[is_test] > 0
    by_mean.append(accuracy_score(up_test, guess))
    print("mean of the first", k, "samples —", round(by_mean[-1], 3))

best_line = (piece[:, LEAD_IN:LEAD_IN + {BEST_K}].mean(axis=1) - background)[is_test] > 0

print("The best one line I found scores", round(accuracy_score(up_test, best_line), 3),
      "— the mean of the first {BEST_K} samples after the pick, against the background before it.")
print("Which sample it reads matters far more than how much background it averages: moving the",
      "offset spreads the score by", round(max(by_offset) - min(by_offset), 3),
      "while changing the lead-in spreads it by only", round(max(by_lead) - min(by_lead), 3), ".")
print("That is because the pick is uncertain by a sample or two and the swing itself lasts only",
      "a few, so reading too early lands in the background and too late lands past the turn —",
      "while the background is flat, and averaging more or less of something flat changes",
      "nothing.")
""")

# --- section 2: the fork ----------------------------------------------------
md(f"""
## How much waveform does the network need?

A network is not restricted to one sample. Hand it the whole window and it can use anything in
there: the shape of the swing, how fast it decays, how the coda rings, how far away the S wave
is. All of that is real information about the earthquake.

The question this track is built on is whether any of it is information about the *polarity*.
`cut(seconds)` is the one knob — it is the only thing that changes between the runs below, and
every window starts at the same place and differs only in how far past the pick it reaches.
""")

md(f"""
### Predict before you run

You are about to train the same network on windows from a tenth of a second to ten seconds long.
Which one do you think will score highest on the held-out recordings? Change `my_guess_seconds`
and run the cell — you will find out two questions from now, and a wrong guess you committed to
is worth more than a right answer you were shown.
""")

CELLS.extend(("code", s, a) for s, a in
             weekkit.predict_cell("2", "seconds of waveform after the pick will score highest",
                                  name="my_guess_seconds"))

ask(f"""
### ✏️ Your turn 2

Write **one** function, and give it exactly this shape:

```python
def network_says_up(seconds):
    \"\"\"Train a network on `seconds` of waveform after the pick; one True/False per held-out
    recording.\"\"\"
```

The recipe, in words:

1. `window = cut(seconds)` gives one row per recording. PyTorch wants a channel dimension on a
   1-D convolution, so hand it `torch.tensor(window).unsqueeze(1)` — **`unsqueeze(1)`** inserts a
   length-1 dimension in the middle, turning *(recordings, samples)* into *(recordings, 1,
   samples)*. Split that with `is_train` and `is_test`, the same masks the labels use.
2. The thing to learn is a sign, so make the target a sign: `+1.0` where the polarity is up and
   `-1.0` where it is down. `np.where(up_train, 1.0, -1.0)` builds it, and
   `.reshape(-1, 1)` gives it the shape the network's single output has.
3. The network: two `nn.Conv1d` layers with `nn.ReLU` after each, then **`nn.Flatten()`** — which
   lays the convolution's output out as one long row per recording — then one `nn.Linear` down to
   a single number. Put `torch.manual_seed(0)` before you build it so a re-run repeats.
4. Train it the way you trained the picker: `torch.optim.Adam`, `nn.MSELoss`, batches of 32, a
   fixed number of epochs, `torch.randperm` to shuffle the order each time.
5. Hand back `output > 0` — one True/False per held-out recording, **not** the fraction it got
   right. A function that returns the fraction can only be compared with another fraction; one
   that returns the answers can be asked which recordings it got wrong, which is what the last
   two sections of this notebook do.

Then run it on the shortest window there is — a tenth of a second — and score it against
`up_test`. This is a slow cell; nothing more prints until it has finished.

Print one line answering it in a sentence, on your own two numbers: **did the network beat the
one line you sharpened in Your turn 1, and by how much?**
""")

answer(f"""
def network_says_up(seconds):
    \"\"\"Train a network on `seconds` of waveform after the pick; one True/False per held-out
    recording.\"\"\"
    # 1. Cut the same stretch of trace out of every recording. A 1-D convolution reads a stack
    #    of channels, so `unsqueeze(1)` inserts the length-1 channel dimension it expects.
    window = cut(seconds)
    x = torch.tensor(window).unsqueeze(1)
    x_train = x[is_train]
    x_test = x[is_test]
    # 2. What the network has to learn is a sign, so the answer it is shown is a sign: +1 where
    #    the first motion is up, -1 where it is down.
    y_train = torch.tensor(np.where(up_train, 1.0, -1.0).astype("float32")).reshape(-1, 1)

    # 3. Fix the random numbers before the layers are built, so the network starts from the same
    #    weights every time and a second run of this cell gives the same answer as the first.
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Conv1d(1, 8, 7, stride=2, padding=3), nn.ReLU(),
        nn.Conv1d(8, 16, 7, stride=2, padding=3), nn.ReLU(),
        nn.Flatten(),
        # Each convolution steps two samples at a time, so the trace comes out a quarter as
        # long as it went in; `Flatten` lays those 16 shortened channels end to end, which is
        # where 16 x a quarter of the window comes from.
        nn.Linear(16 * ((window.shape[1] + 3) // 4), 1))
    optimiser = torch.optim.Adam(net.parameters(), lr=0.005)
    loss_function = nn.MSELoss()

    # 4. Train for a fixed {EPOCHS} passes rather than stopping when the loss looks good enough.
    #    The sweep further down trains one of these networks per window length and compares the
    #    scores, and that comparison is only fair if every network got the same amount of training.
    for epoch in range({EPOCHS}):
        order = torch.randperm(len(x_train))
        for start in range(0, len(x_train), 32):
            batch = order[start:start + 32]
            loss = loss_function(net(x_train[batch]), y_train[batch])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    # 5. Hand back one True/False per held-out recording, not the fraction it got right. A
    #    fraction can only be compared with another fraction; the answers themselves can be
    #    asked WHICH recordings the network missed, which the last two sections do.
    return net(x_test).detach().numpy().ravel() > 0


shortest = network_says_up(0.1)

print("network, 0.1 s of waveform:", round(accuracy_score(up_test, shortest), 3))
print("my best one line:          ", round(accuracy_score(up_test, best_line), 3))
print("always say up:             ", round(accuracy_score(up_test, always_up), 3))
gap = accuracy_score(up_test, shortest) - accuracy_score(up_test, best_line)
print("The network came out", "ahead of" if gap > 0 else "behind", "the one line, by",
      round(abs(gap), 3), "— which on", len(up_test), "held-out recordings is",
      round(abs(gap) * len(up_test)), "recordings, so it is not obviously a real gap at all.",
      "Both are far above always-say-up.")
""")

md(f"""
One window is one point. The fork this project turns on is the whole curve.
""")

ask(f"""
### ✏️ Your turn 3

Run `network_says_up` at every window length in `{WINDOWS}` seconds and score each one on
`up_test`. This is the slow cell in the notebook — it trains the same network seven times over —
so print each score as it arrives rather than at the end.

Draw the result: held-out accuracy against window length, with your best one-line rule and the
always-say-up baseline marked as horizontal lines so all three are readable together. A log
x-axis (`plt.xscale("log")`) spaces the windows evenly.

Then print one more line answering it in a sentence, on your own curve: **which window wins, and
what does the shape of the curve say about what the network is doing with the extra seconds?**
""")

answer(f"""
lengths = {WINDOWS}
scores = []
for seconds in lengths:
    scores.append(accuracy_score(up_test, network_says_up(seconds)))
    print(seconds, "seconds —", round(scores[-1], 3))

plt.plot(lengths, scores, marker="o", color="0.2", label="the network")
plt.axhline(accuracy_score(up_test, best_line), color="firebrick", lw=1.2,
            label="one line of arithmetic")
plt.axhline(accuracy_score(up_test, always_up), color="steelblue", lw=1.2, ls="--",
            label="always say up")
plt.xscale("log")
plt.xlabel("seconds of waveform after the pick")
plt.ylabel("accuracy on the held-out recordings")
plt.title(f"How much waveform helps (n = {{len(up_test)}} held out)")
plt.legend()
plt.show()

print("The shortest window wins:", round(scores[0], 3), "at", lengths[0], "seconds against",
      round(scores[-1], 3), "at", lengths[-1], "seconds, and", round(min(scores), 3),
      "at the worst window of all — a spread of", round(max(scores) - min(scores), 3),
      "with nothing in between beating the shortest.")
print("So the extra seconds carry nothing about the polarity. They are the rest of the",
      "earthquake — more P energy, the S wave, the coda — and none of it says which way the",
      "ground moved first. What they do give the network is more numbers to fit, so a longer",
      "window does not add information, it adds room to memorise the training set.")
""")

# --- section 3: did it earn its place ---------------------------------------
md(f"""
## Did the network earn its place?

You now have three numbers on the same {M['n_test']} held-out recordings: a rule that ignores the
data, a rule that reads one number out of it, and a network that chose its own weights from
{M['n_train']:,} training recordings and took far longer than either to produce.

**Overfitting:** {idea('ML2', 'Overfitting')['words']}
""")

ask(f"""
### ✏️ Your turn 4

Two or three paragraphs, quoting **your own three numbers** — the baseline, your best one line,
and the network at its best window.

1. Did the network earn its place here? Say what it would have to score before you would use it
   instead of the one line, and why that threshold and not a smaller one. Remember that the held
   out set is {M['n_test']} recordings, so one percentage point is a countable number of them —
   work out how many, and say it.
2. Whichever of the two is ahead: how big would the gap have to be before you believed it was
   real rather than an accident of which earthquakes landed in the held-out set? Name the thing
   you would run to find out, and what its answer would look like either way.
""")

answer_prose(f"""
The three numbers on my held-out set are {M['baseline']} for always-say-up, {M['rule_best_mean']}
for the best single line of arithmetic, and {M['network_best']} for the network on its best
window. The network did not earn its place. It is separated from the one line by
{abs(M['network_best'] - M['rule_best_mean']):.3f}, and on {M['n_test']} held-out recordings one
percentage point is about {M['n_test'] / 100:.0f} recordings, so the whole difference between the
two methods is a handful of traces either way. Against that, the network costs a training loop,
several thousand weights, a random seed, and a wait every time it is run, while the one line costs
a subtraction. For me to prefer the network it would have to be ahead by enough that the gap
could not be produced by swapping a dozen recordings between the two sides of the split — call it
three or four points — because below that I am choosing between two methods that are doing the
same job equally well, and the cheap one is the one I can explain to somebody in a sentence.

Both are far above the baseline, and that is the part that matters scientifically: always-say-up
gets {M['baseline']}, so the first motion really is readable, and both methods are reading it.

To decide whether the gap between the two is real I would rerun the whole comparison on several
different splits — reshuffle which earthquakes are held out, five or ten times, and look at the
spread of each method's score rather than at one number each. If the spread of either method
across splits is wider than the gap between them, which I expect, then the gap is an accident of
this particular split and there is nothing to choose. If instead the network were ahead on every
split by a similar margin, that would be evidence of a real difference, and I would then want to
know what the network is reading that the one line is not — which is the next question in this
notebook.
""")

md(f"""
Both methods hand back one True/False per held-out recording, which means they can be compared
recording by recording rather than only in aggregate. Two methods can reach the same score by
being right about the same recordings, or by being right about different ones — and those are
completely different findings.
""")

ask(f"""
### ✏️ Your turn 5

Take your best one line from *Your turn 1* and the network's answers at its best window, and count
the four cases: both right, both wrong, only the line right, only the network right. Print all
four, and print how often the two methods simply agree with each other regardless of who is
right.

Then draw two or three of the recordings where they disagree — the same figure as the one at the
top of this notebook, one trace at a time — and look at them. If the first ones you draw all look
like each other, that is a finding rather than a bug: count how many of the disagreements look
that way, and keep drawing until you have seen one that does not.

Print one more line answering it in a sentence, on your own four counts: **is the network doing
something different from the one line, or the same thing slightly better?**
""")

answer(f"""
network_best = network_says_up(lengths[scores.index(max(scores))])
line_right = best_line == up_test
net_right = network_best == up_test

print("both right:       ", (line_right & net_right).sum())
print("both wrong:       ", (~line_right & ~net_right).sum())
print("only the line:    ", (line_right & ~net_right).sum())
print("only the network: ", (net_right & ~line_right).sum())
print("they agree with each other on", round((best_line == network_best).mean(), 3),
      "of the held-out recordings")

disagree = np.nonzero(is_test)[0][best_line != network_best]
dead = vertical.std(axis=1) == 0

for i in list(disagree[:2]) + [i for i in disagree if not dead[i]][:1]:
    plt.plot(time, half_second[i], color="0.2")
    plt.axvline(0, color="steelblue", lw=1.2)
    plt.xlabel("seconds from the analyst's P pick")
    plt.ylabel("ground motion (in units of this trace's own background)")
    plt.title(f"Recording {{i}} — the analyst wrote {{'U' if up[i] else 'D'}}")
    plt.show()

print("recordings that are one flat line for all", vertical.shape[1], "samples:", dead.sum(),
      "of", len(dead))
print("of the", len(disagree), "disagreements,", dead[disagree].sum(), "are one of those")
print("both methods, with the flat ones left out — line",
      round(accuracy_score(up_test[~dead[is_test]], best_line[~dead[is_test]]), 3), " network",
      round(accuracy_score(up_test[~dead[is_test]], network_best[~dead[is_test]]), 3))

print("The same thing slightly better, and most of what looked like a difference is not one.",
      "The first recordings I drew were not ambiguous first motions at all — they are flat",
      "lines sitting on the file's own ±10 cut-off, with no arrival in them, and they account",
      "for most of the disagreements. On those two coin tosses disagree; everywhere else the",
      "two methods agree with each other more often than either agrees with the analyst, and",
      "leaving the flat ones out moves both scores up together and leaves the gap between them",
      "no bigger. The network has not found a second, independent cue — it has learnt a",
      "smoothed version of the same subtraction.")
""")

# --- section 4: the ceiling -------------------------------------------------
md(f"""
## Is what is left the model, or the labels?

None of the numbers you have collected is 1.0, and what the missing part is made of is a different
kind of question from any asked so far. Either every method here is too weak and a better one
would keep climbing — or the recordings being got wrong do not have a readable answer in them,
and the analyst who wrote the letter was guessing too.

Those can be told apart, and the tool is the size of the first swing against the background noise
before it. A swing ten times the background is unambiguous; a swing the size of the background is
a coin toss whoever is reading it.
""")

ask(f"""
### ✏️ Your turn 6

Measure how clear each first motion is: the size of `first_swing` against the typical size of the
{LEAD_IN} background samples before the pick. `piece[:, :LEAD_IN].std(axis=1)` gives you that
background size, one number per recording.

Then, over **all** {M['n_labelled']:,} recordings rather than only the held-out ones — this is a
question about the labels, not about a model, so nothing is being fitted and nothing is being
scored — split them into four groups by that clarity and, in each group, work out how often the
sign of the waveform agrees with the letter the analyst wrote. Plot it.

Print one more line answering it in a sentence, on your own four numbers: **on the clearest
arrivals, how often does the waveform agree with the analyst — and does what is left over look
like the model's problem or the labels'?**
""")

answer(f"""
clarity = np.abs(first_swing) / np.maximum(piece[:, :LEAD_IN].std(axis=1), 1e-6)
agrees = (first_swing > 0) == up

edges = np.quantile(clarity, [0, 0.25, 0.5, 0.75, 1.0])
groups = []
for q in range(4):
    if q == 3:
        inside = (clarity >= edges[q]) & (clarity <= edges[q + 1])
    else:
        inside = (clarity >= edges[q]) & (clarity < edges[q + 1])
    groups.append(agrees[inside].mean())
    print("clarity quarter", q + 1, "—", inside.sum(), "recordings, agreement",
          round(groups[-1], 3))

print("over all", len(agrees), "recordings:", round(agrees.mean(), 3))
print("on the clearest tenth:", round(agrees[clarity >= np.quantile(clarity, 0.9)].mean(), 3))

plt.bar([1, 2, 3, 4], groups, color="0.4")
plt.axhline(agrees.mean(), color="firebrick", lw=1.2)
plt.xlabel("quarter of the recordings, by how clear the first swing is")
plt.ylabel("waveform agrees with the analyst")
plt.title(f"Agreement with the analyst, by clarity (n = {{len(agrees)}}); the line is the average")
plt.locator_params(axis="x", integer=True)
plt.ylim(0, 1)
plt.show()

print("of the murkiest quarter,", dead[clarity < edges[1]].sum(), "are the flat lines from the",
      "last question — recordings with no arrival in them at all")

print("On the clearest tenth the waveform and the analyst agree",
      round(agrees[clarity >= np.quantile(clarity, 0.9)].mean(), 3), "of the time, against",
      round(groups[0], 3), "in the murkiest quarter. That says the labels are not the limit",
      "where the signal is clear — one subtraction reproduces the analyst almost exactly there.",
      "What is left over is concentrated in the quarter where the first swing is no bigger than",
      "the background, and a large part of that quarter is the flat recordings, which contain",
      "nothing to read. So the honest answer is neither of the two the section offered: most of",
      "the gap is not the model and not the labels, it is recordings that carry no first motion.")
""")

# --- closing ----------------------------------------------------------------
md(f"""
{weekkit.CLOSING_HEADING}

The ground's first move is written in the three or four samples immediately after the P pick, and
one line of arithmetic reads it correctly on {M['rule']:.1%} of the {M['n_test']} held-out
recordings against {M['baseline']:.1%} for a rule that never looks at the waveform — so yes, a
machine can tell, and it does not need to be much of a machine. Do not quote that {M['rule']:.1%}
on its own, though: {M['n_flat'] / M['n_labelled']:.1%} of this file is a flat line at the
clipping bound with no arrival in it at all, nothing can beat a coin toss on those, and *Your turn
5* measured what the sharpened rule reads once they are set aside. Report both, or neither is the
answer. What this notebook has deliberately not told you is where your own network landed on that
scale, and that comparison, not the network, is the project.
""")

md(track_summary())

# --- the project ------------------------------------------------------------
md(f"""
## What your project must contain

Five sections, empty below, required of **every** EPS 88 project regardless of track. They are
headed here so the shape of a good answer is visible while you work. Fill them in as you go; they
are not a write-up you do at the end.
""")

# course.yml's `required_of_every_project:` values are DESIGN notes, not student prose: they name
# a week number ("the week-10 data") and credit the source of the idea (MLGeo). Both are for
# whoever plans the course. What goes in the notebook is the same requirement said to a student.
# The five keys are read from the plan, so a sixth requirement cannot be added there and silently
# skipped here; only the wording is local.
REQUIRED = [list(item)[0] for item in course["project"]["required_of_every_project"]]
STUDENT_WORDING = {
    "one_sentence_answer": ("1 · A one-sentence answer", """
Your claim and its uncertainty, in one sentence, at the top of your report. If you cannot put a
number and a range in it, you do not have a result yet.
"""),
    "baseline_first": ("2 · The trivial baseline", """
Before any model, state the dumbest answer to your question and what it gives. Every later number
is reported against it.

This track hands you two, on purpose and in that order: a rule that never reads the waveform, and
a rule that reads one number out of it. Say what each of them gives on your split, and say what
your network bought you over the better of the two — in accuracy, and in what it cost to run.
"""),
    "split_by_structure": ("3 · Split by structure", """
Earth data are correlated in space and in time, so a train/test split has to follow the structure
that is really there — never a random cut across rows.

The split here is by earthquake, and the setup cell made that choice for you. Say why it is the
right one on this data, what a random split would have leaked, and — if you have the patience —
what happens to your numbers when you make the split by *station* instead.
"""),
    "what_i_got_wrong": ("4 · What I got wrong", """
What failed, and what you believed before it failed. Honest failure is graded; a faked success is
not. Your *Predict before you run* guess belongs here if it was wrong.
"""),
    "ai_disclosure": ("5 · AI disclosure", """
Which tool, what you asked it, what you changed in what it gave you, and how you checked that the
result was true.
"""),
}

# Reading order, not course.yml's order: the one-sentence answer goes at the top of a report by
# definition, and course.yml lists the baseline first because that is the order it was designed in.
ORDER = ["one_sentence_answer", "baseline_first", "split_by_structure",
         "what_i_got_wrong", "ai_disclosure"]
missing = set(REQUIRED) - set(ORDER)
if missing:
    sys.exit(f"course.yml requires {sorted(missing)} of every project and this notebook has no "
             f"section for it")

for key in ORDER:
    heading, guidance = STUDENT_WORDING[key]
    ask(f"### ✏️ {heading}\n{guidance.rstrip()}")
    blank_prose()

# --- the open question ------------------------------------------------------
OPEN = re.findall(r"[^.?]*\?", " ".join(TRACK["open_question"].split()))[-1].strip()

md(f"""
## The open question

> **{OPEN}**

Nobody grading this knows the answer, and neither does the literature. Everything above is the
scaffolding; this is the project.

Here is what is actually established, and it is less than it looks. That the polarity lives in
the first few samples is settled — the sweep shows it, and one subtraction reads it. What is
**not** settled is why every method here stops in the same place, and the two candidate
explanations make different predictions that this dataset can be made to distinguish.

The question as it is written above offers two answers, and this notebook has already turned up
evidence for a third: {M['n_flat']} of the {M['n_labelled']:,} recordings are a single flat line
at the file's own ±{M['max_abs']:.0f} cut-off for their whole length, with no arrival in them at
all. That is {M['n_flat'] / M['n_labelled']:.1%} of the data on which no method can be better
than a coin toss, and it is neither the model's fault nor the analyst's. How much of the ceiling
is that, how those recordings got into the file, and whether an honest project should drop them —
saying so, and reporting both numbers — is the first thing to settle, and nothing above settles
it.

Four more directions, none of them worked out here:

1. **Score the labels, not the models.** Section 4 measured agreement between the waveform and
   the analyst as a function of clarity. Turn that around: for the recordings where the two
   disagree *at high clarity*, one of the two is simply wrong, and it can be decided by eye. Look
   at twenty of them yourself and count. If most are analyst errors, the ceiling is the labels
   and no model will pass it; if most are the rule's, the rule is worse than it looks.
2. **Ask the earthquake, not the recording.** Every earthquake here is recorded at many stations,
   and the true first motions across those stations are not independent — they are set by one
   fault orientation and the station's position on it. A method that reads all the recordings of
   one earthquake together has information no single-trace method has. Nothing in this notebook
   uses it.
3. **Watch the network memorise.** The sweep scores only the held-out set. Score the *training*
   set at every window as well, and the two curves together say whether the long-window collapse
   is a failure to learn or a success at memorising. That is one extra line inside
   `network_says_up`, and it turns a result into a mechanism.
4. **Give it something other than raw samples.** Everything above hands the model the waveform as
   it is. A filtered version, the derivative, or the trace divided by its own first-swing size are
   all the same information rearranged — and if any of them moves the score, what moved it was the
   representation and not the model.

And one that is bigger than a semester: polarity is not wanted for its own sake, it is wanted
because enough polarities determine a fault plane. A method that is right {M['rule']:.0%} of the
time on single recordings feeds errors into that inversion at a rate nobody in this notebook has
measured. How accurate does a per-recording polarity have to be before the mechanism it produces
is usable — and is {M['rule']:.0%} already enough, given how many stations a real earthquake is
recorded at? If the answer is that {M['rule']:.0%} is plenty, then the whole question of beating
the one line was the wrong one to ask, and saying so is a result.
""")

ask(f"""
### ✏️ Your turn 7 — the first move

Before you close this notebook: in a few sentences, name the **one** measurement you would make
first, say what it would show if the ceiling is the labels, what it would show if it is the
models, and name the number that would change your mind. Then make it, in the cell below the
prose.
""")

answer_prose(f"""
The flat recordings are already dealt with: *Your turn 5* counted them, and setting them aside
raises both methods together without changing the gap between them, so they explain part of the
ceiling and none of the choice between the two methods. With them out of the way I would do the
first direction — hand-check the high-clarity disagreements — because it is the only measurement
here where the two remaining explanations predict opposite things rather than different sizes. If the ceiling is the labels, then among the recordings whose first swing is large and
clean but whose sign contradicts the letter the analyst wrote, most should look unambiguous to me
on the screen: a clear upward swing under a `D`. If the ceiling is the models, those same
recordings should look genuinely hard — a small swing, a noisy background, an arrival that starts
before the pick — and my own reading should agree with the analyst rather than with the rule. The
number that would change my mind is the fraction of that set where I side with the waveform
against the analyst: above about half and I would say the labels are the limit; below a quarter
and I would say the rule is being lazy and a better method has room to grow. The count below is
the catch — the pile is small enough that either verdict would rest on single figures, so the
same pass has to run down through the clearest quarter before it means anything.

What makes me expect the first answer is the clarity curve. Agreement rises from
{M['agree_quartile'][0]} in the murkiest quarter to {M['agree_quartile'][3]} in the clearest, and
on the clearest tenth it reaches {M['agree_top_decile']} — so where the seismogram is legible the
one-line rule and the analyst almost never disagree, and there is very little headroom left for a
model to find. The recordings that are being lost are concentrated where the swing is no bigger
than the noise, and on those the honest description is not that the model failed but that the
question has no answer in the data. The measurement below counts how many high-clarity
disagreements there even are, which is the first thing to know before spending an afternoon
looking at them.
""")

answer(f"""
clear = clarity >= np.quantile(clarity, 0.9)
wrong = (first_swing > 0) != up

quarter = clarity >= np.quantile(clarity, 0.75)

print("recordings in the clearest tenth:", clear.sum())
print("of those, the waveform contradicts the analyst on:", (clear & wrong).sum())
print("in the clearest quarter:", (quarter & wrong).sum(), "contradictions out of", quarter.sum())

for i in np.nonzero(clear & wrong)[0][:3]:
    plt.plot(time, half_second[i], color="0.2")
    plt.axvline(0, color="steelblue", lw=1.2)
    plt.xlabel("seconds from the analyst's P pick")
    plt.ylabel("ground motion (in units of this trace's own background)")
    plt.title(f"Recording {{i}} — clear swing, but the analyst wrote {{'U' if up[i] else 'D'}}")
    plt.show()

print("That is a pile small enough to look at every one of by hand, which is what this project",
      "should do next — and small enough that the answer will be a count out of tens rather",
      "than a percentage with an error bar on it, so the clearest tenth alone will not settle",
      "it and the same pass has to be repeated down through the clearest quarter.")
""")


# ---------------------------------------------------------------------------
# 4. emit, execute, gate
# ---------------------------------------------------------------------------
def notebook(cells):
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4, "nbformat_minor": 5}


def cell(kind, source):
    c = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def track_ids(cells):
    """weekkit.stable_ids keys cells to a week number; a track keys them to its id instead.

    Same contract and same reason: a submission graded against an earlier release must not report
    every cell as missing because a paragraph was inserted above it.
    """
    q = 0
    for i, c in enumerate(cells):
        s = "".join(c.get("source", []))
        if c["cell_type"] == "markdown" and re.search(r"(?m)^\s*(#{1,4}\s*)?✏️", s):
            q += 1
            c["id"] = f"{TRACK['id']}-q{q:02d}-ask"
        elif c["cell_type"] == "code" and re.search(r"your answer here", s, re.I):
            c["id"] = f"{TRACK['id']}-q{q:02d}-answer"
        elif c["cell_type"] == "markdown" and "Double-click" in s:
            c["id"] = f"{TRACK['id']}-q{q:02d}-prose"
        # The Predict pair carries an assert but is not a question's self-check: it sits BEFORE
        # the first ✏️, so the generic branch below would key it to `q00` and collide with the
        # loading check, which is the real q00. Its two cells get their own ids, and the guess
        # cell matches in both copies -- solution `= 2`, student `= None`.
        elif c["cell_type"] == "code" and re.search(r"(?m)^my_guess\w*\s*=", s):
            c["id"] = f"{TRACK['id']}-predict"
        elif c["cell_type"] == "code" and re.search(r"assert my_guess\w* is not None", s):
            c["id"] = f"{TRACK['id']}-predict-check"
        elif c["cell_type"] == "code" and "assert " in s:
            c["id"] = f"{TRACK['id']}-q{q:02d}-check"
        else:
            c["id"] = f"{TRACK['id']}-c{i:03d}"
    return weekkit.dedupe_ids(cells)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    sol = notebook([cell(k, s) for k, s, _ in CELLS])
    stu = notebook([cell(k, alt if alt is not None else s) for k, s, alt in CELLS])

    sol_path = OUT / f"{SLUG}_solution.ipynb"
    sol_path.write_text(json.dumps(sol, indent=1) + "\n")

    # Execute somewhere disposable. The notebook keeps the 44 MB download in its working
    # directory, which on DataHub is the right thing and in this repository is not.
    run_dir = pathlib.Path(tempfile.mkdtemp(prefix="trackT8-run-"))
    shutil.copy(LOCAL, run_dir / "phasenet_ncedc.npz")

    # weekkit.execute pins the kernel to THIS interpreter — this track needs torch, which the
    # shared base environment does not carry. run_dir is the cwd because the notebook keeps its
    # 44 MB download beside itself.
    print(f"executing {sol_path.name} in {run_dir} ...")
    started = time.time()
    r = weekkit.execute(sol_path, timeout=3600, cwd=run_dir)
    shutil.rmtree(run_dir, ignore_errors=True)
    if r.returncode:
        print(r.stderr[-6000:])
        sys.exit("the solution did not execute")
    print(f"the solution executed in {time.time() - started:.0f} s wall clock")

    (OUT / f"{SLUG}.ipynb").write_text(json.dumps(stu, indent=1) + "\n")

    for f in (sol_path, OUT / f"{SLUG}.ipynb"):
        nb = json.loads(f.read_text())
        track_ids(nb["cells"])
        f.write_text(json.dumps(nb, indent=1))

    print(f"wrote {SLUG}.ipynb ({len(CELLS)} cells) and the executed solution")
    gate(sol_path)


def gate(sol_path):
    """The half of weekkit.gate that does not need a week number.

    weekkit.gate looks the notebook up by `slug` in course.yml's `schedule:` and then runs
    check_notebook.py and check_prior_knowledge.py, both of which take a week number. A track has
    none. What transfers unchanged is the gate that matters most and needs nothing from the plan:
    the solution executed clean on a fresh kernel, with contiguous execution counts from 1.
    """
    bad = []
    cells = json.loads(sol_path.read_text())["cells"]
    counts = [c["execution_count"] for c in cells
              if c["cell_type"] == "code" and c.get("execution_count")]
    if not counts:
        bad.append("the solution has no execution counts — it was never executed")
    elif counts[0] != 1:
        bad.append(f"execution counts start at {counts[0]}, not 1")
    elif counts != list(range(1, len(counts) + 1)):
        bad.append("execution counts are not contiguous — the solution was executed piecemeal")
    if any(o.get("output_type") == "error" for c in cells for o in c.get("outputs", [])):
        bad.append("the solution contains an error output — it does not execute clean")

    stu = json.loads((sol_path.parent / sol_path.name.replace("_solution", "")).read_text())
    if len(stu["cells"]) != len(cells):
        bad.append("student and solution have drifted apart")
    if any(c.get("outputs") for c in stu["cells"]):
        bad.append("the student notebook carries outputs — it must ship clean")
    figs = sum(1 for c in cells for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
    if figs == 0:
        bad.append("the solution contains no figures")

    r = subprocess.run([sys.executable, str(ROOT / "tools/check_track.py"), TRACK["id"]],
                       capture_output=True, text=True, cwd=ROOT)
    print(r.stdout.rstrip())
    if r.returncode:
        bad.append("check_track reported errors (above)")

    if bad:
        print("\nBUILD REJECTED:")
        for b in bad:
            print(f"  - {b}")
        sys.exit(1)
    print(f"\ngates passed: executes clean ({len(counts)} cells, {figs} figures), check_track OK")


if __name__ == "__main__":
    main()
