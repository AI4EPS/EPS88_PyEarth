#!/usr/bin/env python
"""Build week 13 — "Can a machine hear an earthquake?"

Emits both notebooks from one source so they cannot drift:

    docs/notebooks/13_machine_hears_solution.ipynb   executed, every output saved
    docs/notebooks/13_machine_hears.ipynb            the same file with the answers deleted

The week's data is a 42 MB .npz of NCEDC waveforms that ships as a GitHub RELEASE asset, not in
data/, because nbgitpuller clones data/ onto every student account. There is therefore no cached
CSV for this week and this script writes none: the release asset is the only source, and the
notebook reads it directly, downloading it once with torch.hub.

Every number that appears in prose or in a model answer is computed HERE, by the same code the
notebook runs, and formatted in. Nothing is typed from memory or copied from the plan. The
training is seeded, so the numbers below and the notebook's own outputs are the same numbers.

    python tools/build_week13.py

Needs torch, which the shared base environment does not carry; run it with an interpreter that
has torch, numpy, matplotlib, sklearn, pyyaml and nbconvert.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import yaml
from sklearn.linear_model import LinearRegression
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/notebooks"
SLUG = "13_machine_hears"

# TEMPLATE 5 makes these sentences binding, so they are READ rather than retyped: a hand-copied
# one drifts from the catalogue the first time either is edited, and four of them already had.
WORDS = {d["idea"]: d["words"] for d in weekkit._modules().get("plain_words", [])
         if d["module"] == "ML6"}

course = yaml.safe_load((ROOT / "course.yml").read_text())
WEEK = next(s for s in course["schedule"] if s["n"] == 13)
PLATFORM = course["platform"]

DATA_URL = ("https://github.com/AI4EPS/EPS88_PyEarth/releases/download/"
            "data-v1/phasenet_ncedc.npz")
# Kept outside the repository: 42 MB of waveforms must never end up in a clone that nbgitpuller
# copies onto 46 student accounts.
LOCAL = pathlib.Path(tempfile.gettempdir()) / "eps88_wk13_phasenet_ncedc.npz"

DEFAULT_THREADS = torch.get_num_threads()   # what a student's DataHub kernel actually uses
torch.set_num_threads(4)


# ---------------------------------------------------------------------------
# 1. measure everything the notebook will say, with the notebook's own code
# ---------------------------------------------------------------------------
if not LOCAL.exists():
    torch.hub.download_url_to_file(DATA_URL, str(LOCAL), progress=False)

data = np.load(LOCAL)
waveform = np.clip(data["waveform"], -10, 10)
p_index = data["p_index"].astype(int)
s_index = data["s_index"].astype(int)
distance_km = data["distance_km"]
depth_km = data["depth_km"]
event_id = data["event_id"]
SAMPLE_RATE = 100

M = {}
M["n_traces"] = len(waveform)
M["n_events"] = int(len(np.unique(event_id)))
M["n_stations"] = int(len(np.unique(data["station"])))
M["n_samples"] = int(waveform.shape[2])
M["seconds"] = round(waveform.shape[2] / SAMPLE_RATE, 2)
M["p_min_s"] = round(float(p_index.min() / SAMPLE_RATE), 2)
M["p_max_s"] = round(float(p_index.max() / SAMPLE_RATE), 2)
M["mag_min"] = round(float(data["magnitude"].min()), 2)
M["mag_max"] = round(float(data["magnitude"].max()), 2)
M["mag_med"] = round(float(np.median(data["magnitude"])), 2)
M["dist_med"] = round(float(np.median(distance_km)), 1)
M["dist_max"] = round(float(distance_km.max()), 1)
M["depth_min"] = round(float(depth_km.min()), 1)
M["depth_max"] = round(float(depth_km.max()), 1)

# The release asset is ALREADY clipped at write time, so the np.clip in the setup cell is a
# guard and not a cleaning step. These two numbers are what stops the notebook calling it one:
# the bound is saturated, which censors any median taken above it and rails a few dead channels.
M["clip_frac"] = round(float((np.abs(waveform) >= 10).mean()), 3)
flat = (waveform.std(axis=2) == 0).all(axis=1)
M["n_flat"] = int(flat.sum())

rng = np.random.default_rng(0)
events = np.unique(event_id)
rng.shuffle(events)
train_events = events[:int(0.7 * len(events))]
is_train = np.isin(event_id, train_events)
M["n_train"] = int(is_train.sum())
M["n_test"] = int((~is_train).sum())
M["n_train_events"] = int(len(train_events))
p_test = p_index[~is_train]

# --- section 1: one trace, and what the S-P gap is worth
# The trace every single-trace figure draws. Chosen for LEGIBILITY and nothing else — the first
# recording whose P and S are 2.5 to 4 s apart, whose SNR is over 20, and whose P is late enough
# in the window to leave some background before it — never for the answer it happens to give.
sp_samples = s_index - p_index
TRACE = int(np.nonzero((sp_samples > 250) & (sp_samples < 400)
                       & (data["snr"] > 20) & (p_index > 400))[0][0])
M["trace"] = TRACE
M["trace_station"] = str(data["station"][TRACE])
M["trace_mag"] = round(float(data["magnitude"][TRACE]), 2)
M["trace_p_s"] = round(float(p_index[TRACE] / SAMPLE_RATE), 2)
M["trace_s_s"] = round(float(s_index[TRACE] / SAMPLE_RATE), 2)
M["trace_sp_s"] = round(M["trace_s_s"] - M["trace_p_s"], 2)
M["trace_dist"] = round(float(distance_km[TRACE]), 1)

sp_time = (s_index - p_index) / SAMPLE_RATE
M["sp_min"] = round(float(sp_time.min()), 2)
M["sp_max"] = round(float(sp_time.max()), 2)
M["sp_med"] = round(float(np.median(sp_time)), 2)
# NOT through the origin. distance_km is EPICENTRAL — measured along the surface to the point
# above the earthquake — while the earthquake is kilometres down, so a station standing on the
# epicentre still has the whole depth between it and the source and still sees a gap. Measured
# below: nothing in this file has a gap anywhere near zero, and forcing the line through a point
# no recording occupies costs R squared as well as being false.
line = LinearRegression()
line.fit(sp_time.reshape(-1, 1), distance_km)
M["sp_slope"] = round(float(line.coef_[0]), 2)
M["sp_intercept"] = round(float(line.intercept_), 1)
M["sp_r2"] = round(float(line.score(sp_time.reshape(-1, 1), distance_km)), 3)
origin = LinearRegression(fit_intercept=False).fit(sp_time.reshape(-1, 1), distance_km)
M["sp_r2_origin"] = round(float(origin.score(sp_time.reshape(-1, 1), distance_km)), 3)
near_epicentre = distance_km < 2
M["n_near2"] = int(near_epicentre.sum())
M["sp_near2_med"] = round(float(np.median(sp_time[near_epicentre])), 2)

# --- section 2: was there an earthquake at all?
WINDOW = 256
quiet = np.zeros((len(waveform), 3, WINDOW), dtype="float32")
shaken = np.zeros((len(waveform), 3, WINDOW), dtype="float32")
for i in range(len(waveform)):
    quiet[i] = waveform[i][:, p_index[i] - WINDOW - 6:p_index[i] - 6]
    shaken[i] = waveform[i][:, p_index[i] + 6:p_index[i] + 6 + WINDOW]
loud_quiet = np.abs(quiet).max(axis=(1, 2))
loud_shaken = np.abs(shaken).max(axis=(1, 2))
M["window_s"] = round(WINDOW / SAMPLE_RATE, 2)
M["quiet_med"] = round(float(np.median(loud_quiet[is_train])), 2)
M["shaken_med"] = round(float(np.median(loud_shaken[is_train])), 2)
# The after-P median sits ON the clip bound, so it is censored: more than half of these windows
# were cut off at 10 when the file was written and the median only reports where we cut.
M["shaken_railed"] = round(float((loud_shaken[is_train] == 10).mean()), 3)
M["quiet_railed"] = round(float((loud_quiet[is_train] == 10).mean()), 3)
THRESHOLD = 2.5
M["threshold"] = THRESHOLD
M["detect_right"] = int((loud_quiet[~is_train] < THRESHOLD).sum()
                        + (loud_shaken[~is_train] >= THRESHOLD).sum())
M["detect_acc"] = round(M["detect_right"] / (2 * M["n_test"]), 3)

loudest = np.abs(waveform).max(axis=1).argmax(axis=1)
M["loudest_near_p"] = round(float((np.abs(loudest - p_index) <= 50).mean()), 3)
M["loudest_near_s"] = round(float((np.abs(loudest - s_index) <= 50).mean()), 3)

# --- section 3: the classical picker
strength = np.abs(waveform).max(axis=1)


def sta_lta(trace, short, long):
    """How much louder the last `short` samples are than the last `long` samples."""
    power = trace ** 2
    ratio = np.zeros(len(power))
    for i in range(long, len(power)):
        ratio[i] = power[i - short:i].mean() / power[i - long:i].mean()
    return ratio


def first_trigger(ratio, threshold):
    """The first sample where the ratio crosses the threshold, or None if it never does."""
    above = np.nonzero(ratio > threshold)[0]
    if len(above) == 0:
        return None
    return int(above[0])


# One helper, returning ONE True/False per held-out trace. It used to return the FRACTION, which
# is a scalar, and the notebook needs the vector three separate times — so "run STA/LTA on every
# held-out trace and keep its first trigger" was written out by hand twice more, once in the
# class cell that splits the test set by arrival time and once by the STUDENT in Your turn 6,
# whose prompt had to say "the way sta_lta_score does, except keeping one True/False per
# recording instead of one total". A helper the week hands out and then cannot use is week 1's
# `columns()` again. Take `.mean()` wherever the fraction is what is wanted.
def sta_lta_hits(short, long, threshold):
    """True/False per held-out trace: did STA/LTA land within half a second of the pick?"""
    hits = []
    for i in np.nonzero(~is_train)[0]:
        pick = first_trigger(sta_lta(strength[i], short, long), threshold)
        hits.append(pick is not None and abs(pick - p_index[i]) <= 50)
    return np.array(hits)


textbook_ok = sta_lta_hits(50, 500, 3)
M["stalta_textbook"] = round(float(textbook_ok.mean()), 3)
SWEEP = [(30, 300, 3), (20, 200, 3), (50, 500, 5)]
sweep_hits = {f"{a}, {b}, {c}": sta_lta_hits(a, b, c) for a, b, c in SWEEP}
M["stalta_sweep"] = {k: round(float(v.mean()), 3) for k, v in sweep_hits.items()}
M["stalta_best_setting"], M["stalta_best"] = max(M["stalta_sweep"].items(), key=lambda kv: kv[1])
best_ok = sweep_hits[M["stalta_best_setting"]]

# WHERE the textbook setting's misses are, which is the whole lesson of this section: with
# long=500 the ratio is pinned at zero for the first five seconds, so a P arriving before 4.5 s
# cannot be found however strong it is. Split the held-out set on that and score each half.
#
# BOTH settings are scored on BOTH halves. The old version scored only the textbook setting here
# and then set its 0.602 on the late traces beside the sweep's 0.602 over all 741 — two different
# denominators, and two rounded numbers agreeing is a coincidence, not a demonstration. The claim
# "off the dead zone they are the same picker" is the load-bearing one in this section, so the
# notebook runs the comparison it rests on — with `sta_lta_hits`, which is exactly that scoring
# pass. The loop below survives only for the two quantities that are NOT it: whether the textbook
# setting triggered on the S instead, and whether it triggered at all.
BEST = tuple(int(x) for x in M["stalta_best_setting"].split(", "))
on_the_s, never = [], []
for i in np.nonzero(~is_train)[0]:
    pick = first_trigger(sta_lta(strength[i], 50, 500), 3)
    never.append(pick is None)
    on_the_s.append(pick is not None and abs(pick - s_index[i]) <= 50)
on_the_s = np.array(on_the_s)
never = np.array(never)
early = p_test < 450                    # the long window has not filled until sample 500
dead_test = (waveform[~is_train].std(axis=2) == 0).all(axis=1)
M["stalta_never"] = round(float(never.mean()), 3)
M["stalta_near_s"] = round(float(on_the_s.mean()), 3)
M["n_early"] = int(early.sum())
M["n_late"] = int((~early).sum())
M["early_frac"] = round(float(early.mean()), 3)
M["textbook_early"] = round(float(textbook_ok[early].mean()), 3)
M["textbook_late"] = round(float(textbook_ok[~early].mean()), 3)
M["best_early"] = round(float(best_ok[early].mean()), 3)
M["best_late"] = round(float(best_ok[~early].mean()), 3)
M["s_in_dead_zone"] = round(float(early[on_the_s].mean()), 3)
# The dead zone was being credited with the never-triggered as well, on no evidence at all — the
# only number offered for it was the share of the S TRIGGERS that are in the dead zone. Measure
# the never-triggered separately, and count how many of them are the flat dead channels the
# notebook meets at the end and never used to connect to anything.
M["n_never"] = int(never.sum())
M["never_in_dead"] = int((never & early).sum())
M["never_in_dead_frac"] = round(float(early[never].mean()), 3)
M["never_dead_channel"] = int((never & dead_test).sum())

# --- section 4: a pattern-detector made by hand


def onset_filter(width):
    """Energy in the last `width` samples minus energy in the `width` samples before those."""
    return np.concatenate([np.ones(width) / width, -np.ones(width) / width])


# Scored TWO ways on the same responses, because the difference between them is the point.
# First crossing is the rule STA/LTA was scored by; argmax is the rule the network will be
# scored by. Twenty fixed weights read one way are a fair rival; read the other way they are
# a detector for the S. Scoring the two methods under different rules and calling the gap
# evidence about the weights is the mistake this section used to make.
hand_hits = hand_argmax_hits = hand_argmax_near_s = 0
for i in np.nonzero(~is_train)[0]:
    response = np.convolve(strength[i] ** 2, onset_filter(10), mode="same")
    pick = first_trigger(response, 3)
    hand_hits += pick is not None and abs(pick - p_index[i]) <= 50
    hand_argmax_hits += abs(response.argmax() - p_index[i]) <= 50
    hand_argmax_near_s += abs(response.argmax() - s_index[i]) <= 50
M["hand_acc"] = round(float(hand_hits / M["n_test"]), 3)
M["hand_argmax_acc"] = round(float(hand_argmax_hits / M["n_test"]), 3)
M["hand_argmax_near_s"] = round(float(hand_argmax_near_s / M["n_test"]), 3)

# --- section 5: the network


def make_target(pick_index, sigma):
    """A bump centred on each trace's P arrival: what we want the network to output."""
    sample = np.arange(2048)
    target = np.zeros((len(pick_index), 2048), dtype="float32")
    for i in range(len(pick_index)):
        target[i] = np.exp(-(sample - pick_index[i]) ** 2 / (2 * sigma ** 2))
    return target


def make_picker():
    """Five convolution layers: squeeze the trace down to a summary, then stretch it back out."""
    return nn.Sequential(
        nn.Conv1d(3, 8, 7, stride=4, padding=3), nn.ReLU(),
        nn.Conv1d(8, 16, 7, stride=4, padding=3), nn.ReLU(),
        nn.Conv1d(16, 16, 7, padding=3), nn.ReLU(),
        nn.Upsample(scale_factor=4), nn.Conv1d(16, 8, 7, padding=3), nn.ReLU(),
        nn.Upsample(scale_factor=4), nn.Conv1d(8, 1, 7, padding=3))


x_train = torch.tensor(waveform[is_train])
x_test = torch.tensor(waveform[~is_train])


def picks_from(model, x):
    """Where the network says the P is: the sample with the highest output."""
    return model(x).squeeze(1).detach().numpy().argmax(axis=1)


def within_half_second(picks, truth):
    """Fraction of picks landing within half a second of the analyst's pick."""
    return (np.abs(picks - truth) <= 50).mean()


def train_picker(sigma=20, epochs=25):
    """Train the picker; hand back the model, and the loss and test score after every epoch."""
    torch.manual_seed(0)
    model = make_picker()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_function = nn.MSELoss()
    y_train = torch.tensor(make_target(p_index[is_train], sigma))
    losses, scores = [], []
    for epoch in range(epochs):
        order = torch.randperm(len(x_train))
        total = 0.0
        for start in range(0, len(x_train), 32):
            batch = order[start:start + 32]
            loss = loss_function(model(x_train[batch]).squeeze(1), y_train[batch])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total = total + loss.item() * len(batch)
        losses.append(total / len(x_train))
        scores.append(within_half_second(picks_from(model, x_test), p_test))
    return model, losses, scores


M["n_weights"] = int(sum(w.numel() for w in make_picker().parameters()))

t0 = time.time()
picker, losses, scores = train_picker()
M["train_seconds"] = round(time.time() - t0, 1)
M["threads"] = 4
M["seconds_per_epoch"] = round(M["train_seconds"] / len(losses), 2)
net_picks = picks_from(picker, x_test)
M["net_acc"] = round(float(within_half_second(net_picks, p_test)), 3)
M["net_med_err"] = round(float(np.median(np.abs(net_picks - p_test)) / SAMPLE_RATE), 3)
M["net_acc_first"] = round(float(scores[0]), 3)
M["net_best_epoch"] = int(np.argmax(scores) + 1)
M["net_best_acc"] = round(float(max(scores)), 3)
M["loss_first"] = round(float(losses[0]), 5)
M["loss_last"] = round(float(losses[-1]), 5)
M["acc_ep10"] = round(float(scores[9]), 3)
M["acc_late_low"] = round(float(min(scores[10:])), 3)
M["acc_late_high"] = round(float(max(scores[10:])), 3)
M["acc_wobble"] = round(float(max(scores[10:]) - min(scores[10:])), 3)

# The lower panel of the closest/furthest figure. The prose says the recording is DEAD, so the
# claim is measured rather than assumed: three zero standard deviations, and every sample of all
# three rows sitting on the clip bound our own write step put there.
worst = int(np.abs(net_picks - p_test).argmax())
M["worst_std"] = [round(float(v), 3) for v in x_test[worst].numpy().std(axis=1)]
M["worst_values"] = sorted(round(float(v), 1) for v in np.unique(x_test[worst].numpy()))
M["worst_is_flat"] = bool(max(M["worst_std"]) == 0)

# --- homework: does the network fail where the classical picker fails?
# best_ok IS this loop, run once above at the same best setting on the same held-out traces.
classic_ok = best_ok
network_ok = np.abs(net_picks - p_test) <= 50
M["both_right"] = int((classic_ok & network_ok).sum())
M["network_only"] = int((~classic_ok & network_ok).sum())
M["classic_only"] = int((classic_ok & ~network_ok).sum())
M["both_wrong"] = int((~classic_ok & ~network_ok).sum())

snr_test = data["snr"][~is_train]
quiet_half = snr_test < np.median(snr_test)
M["snr_median"] = round(float(np.median(snr_test)), 1)
M["n_quiet"] = int(quiet_half.sum())
M["n_loud"] = int((~quiet_half).sum())
M["net_quiet"] = round(float(network_ok[quiet_half].mean()), 3)
M["net_loud"] = round(float(network_ok[~quiet_half].mean()), 3)
M["classic_quiet"] = round(float(classic_ok[quiet_half].mean()), 3)
M["classic_loud"] = round(float(classic_ok[~quiet_half].mean()), 3)

# --- homework: how wide should the label be?
sigma_gaps = []
for sigma in (5, 40):
    hw_model, hw_losses, hw_scores = train_picker(sigma=sigma)
    hw_picks = picks_from(hw_model, x_test)
    M[f"sigma{sigma}_acc"] = round(float(hw_scores[-1]), 3)
    M[f"sigma{sigma}_loss"] = round(float(hw_losses[-1]), 5)
    M[f"sigma{sigma}_med"] = round(float(np.median(np.abs(hw_picks - p_test)) / SAMPLE_RATE), 3)
    sigma_gaps.append(abs(float(hw_scores[-1]) - float(scores[-1])))
# The floor Q7's self-check compares against, MEASURED rather than chosen: whichever of the two
# offered widths lands closest to the sigma=20 run still misses it by this much, so a student who
# forgot to pass sigma= (and so re-ran the class model) trips the check and nobody else does.
M["sigma_gap"] = round(min(sigma_gaps), 3)
M["sigma_floor"] = round(min(sigma_gaps) / 2, 3)

# --- how long this actually takes on a CPU, at the thread count a student's kernel uses.
# modules.yml carried 1.2 s/epoch, which was measured on a different and much larger network.
torch.set_num_threads(DEFAULT_THREADS)
t0 = time.time()
train_picker(epochs=2)
M["default_threads"] = int(DEFAULT_THREADS)
M["seconds_per_epoch_default"] = round((time.time() - t0) / 2, 1)
torch.set_num_threads(4)

# --- homework: does more training help?
_, long_losses, long_scores = train_picker(epochs=60)
M["long_acc25"] = round(float(long_scores[24]), 3)
M["long_acc60"] = round(float(long_scores[-1]), 3)
M["long_loss25"] = round(float(long_losses[24]), 5)
M["long_loss60"] = round(float(long_losses[-1]), 5)
M["long_best_epoch"] = int(np.argmax(long_scores) + 1)
M["long_best_acc"] = round(float(max(long_scores)), 3)


# ---------------------------------------------------------------------------
# 2. the cells
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


# Code that appears BOTH in a class cell and in the homework checkpoint, written once here so the
# two copies cannot drift. check_notebook's checkpoint rule wants the homework to run on a cold
# kernel; this week's homework needs the split, the classical picker and the whole training
# apparatus, and hand-copying nine definitions into a checkpoint is how they go stale.
SRC_SPLIT = """rng = np.random.default_rng(0)
events = np.unique(event_id)
rng.shuffle(events)
train_events = events[:int(0.7 * len(events))]
is_train = np.isin(event_id, train_events)      # True for a recording of a training earthquake"""

SRC_STRENGTH = ("strength = np.abs(waveform).max(axis=1)"
                "     # one number per sample: the biggest of the three rows")

SRC_STALTA = '''def sta_lta(trace, short, long):
    """How much louder the last `short` samples are than the last `long` samples."""
    power = trace ** 2
    ratio = np.zeros(len(power))
    for i in range(long, len(power)):
        ratio[i] = power[i - short:i].mean() / power[i - long:i].mean()
    return ratio


def first_trigger(ratio, threshold):
    """The first sample where the ratio crosses the threshold, or None if it never does."""
    above = np.nonzero(ratio > threshold)[0]
    if len(above) == 0:
        return None
    return int(above[0])'''

# Written once here because it is used in three class cells AND named by Your turn 6, which a
# student may reach on a cold kernel: a checkpoint that rebuilds the picker but not the function
# that scores it is a NameError from the prompt's own words.
#
# It hands back ONE True/False per held-out trace rather than a fraction, and `.mean()` turns that
# into the fraction wherever the fraction is what is wanted. The scalar version forced the same
# loop to be written out again in the cell that splits the test set by arrival time, and a third
# time by the student in Your turn 6.
SRC_STALTA_HITS = '''def sta_lta_hits(short, long, threshold):
    """True/False per held-out trace: did STA/LTA land within half a second of the pick?"""
    hits = []
    for i in np.nonzero(~is_train)[0]:
        pick = first_trigger(sta_lta(strength[i], short, long), threshold)
        hits.append(pick is not None and abs(pick - p_index[i]) <= 50)
    return np.array(hits)'''

SRC_TARGET = f'''def make_target(pick_index, sigma):
    """A bump centred on each trace's P arrival: what we want the network to output."""
    sample = np.arange({M['n_samples']})
    target = np.zeros((len(pick_index), {M['n_samples']}), dtype="float32")
    for i in range(len(pick_index)):
        target[i] = np.exp(-(sample - pick_index[i]) ** 2 / (2 * sigma ** 2))
    return target'''

SRC_MODEL = '''def make_picker():
    """Five convolution layers: squeeze the trace down to a summary, then stretch it back out."""
    return nn.Sequential(
        nn.Conv1d(3, 8, 7, stride=4, padding=3), nn.ReLU(),
        nn.Conv1d(8, 16, 7, stride=4, padding=3), nn.ReLU(),
        nn.Conv1d(16, 16, 7, padding=3), nn.ReLU(),
        nn.Upsample(scale_factor=4), nn.Conv1d(16, 8, 7, padding=3), nn.ReLU(),
        nn.Upsample(scale_factor=4), nn.Conv1d(8, 1, 7, padding=3))


x_train = torch.tensor(waveform[is_train])
x_test = torch.tensor(waveform[~is_train])
p_test = p_index[~is_train]'''

SRC_TRAIN = '''def picks_from(model, x):
    """Where the network says the P is: the sample with the highest output."""
    return model(x).squeeze(1).detach().numpy().argmax(axis=1)


def within_half_second(picks, truth):
    """Fraction of picks landing within half a second of the analyst's pick."""
    return (np.abs(picks - truth) <= 50).mean()


def train_picker(sigma=20, epochs=25):
    """Train the picker; hand back the model, and the loss and test score after every epoch."""
    # 1. Set up. The seed fixes the random weights the network starts from, so the cell gives
    #    the same answer every time it runs and your numbers match the ones written below.
    torch.manual_seed(0)
    model = make_picker()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_function = nn.MSELoss()
    y_train = torch.tensor(make_target(p_index[is_train], sigma))
    losses, scores = [], []
    for epoch in range(epochs):
        # 2. One epoch. Shuffle the traces, then hand them over 32 at a time rather than all
        #    at once: every small batch gets its own step, so one pass improves the weights
        #    many times over instead of once.
        order = torch.randperm(len(x_train))
        total = 0.0
        for start in range(0, len(x_train), 32):
            batch = order[start:start + 32]
            # 3. The step itself: measure the miss, work out which way each weight would have
            #    to move, then move them. `zero_grad` clears those directions before
            #    `backward` works out new ones, because PyTorch adds them up — leave it out
            #    and every batch is still carrying the ones before it.
            loss = loss_function(model(x_train[batch]).squeeze(1), y_train[batch])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total = total + loss.item() * len(batch)
        # 4. Record both at the end of each epoch: the loss on what it trained on, and the
        #    score on traces it has never seen. Plotted against the epoch, those two are the
        #    learning curve, and they are the only view you get of what training is doing.
        losses.append(total / len(x_train))
        scores.append(within_half_second(picks_from(model, x_test), p_test))
    return model, losses, scores'''

# The checkpoint rebuilds everything the homework touches, silently: the split, the classical
# picker and the function that scores it, the network apparatus and the trained model itself.
SRC_CHECKPOINT = "\n\n".join([SRC_SPLIT, SRC_STRENGTH, SRC_STALTA, SRC_STALTA_HITS, SRC_TARGET,
                              SRC_MODEL, SRC_TRAIN,
                              "picker, losses, scores = train_picker()\n"
                              "net_picks = picks_from(picker, x_test)"])


datahub = (f"{PLATFORM['datahub']}/hub/user-redirect/git-pull"
           f"?repo={PLATFORM['repo'].replace(':', '%3A').replace('/', '%2F')}"
           f"&branch={PLATFORM['branch']}"
           f"&urlpath=lab%2Ftree%2FEPS88_PyEarth%2F{PLATFORM['notebook_dir']}%2F{SLUG}.ipynb")

HOOK = """
A seismometer never stops. It writes down how the ground is moving a hundred times a second, day
and night, and nearly all of what it writes is traffic, wind and surf. Buried in that are the
earthquakes, almost all of them far too small to feel.

Finding them is one job. Timing them is a harder one, and it is the one that matters. What a
seismologist needs from a recording is the *instant* the first wave arrived, because the gap
between the fast P wave and the slower S wave gives the distance to the earthquake, and distances
from three stations give its location. For most of the history of the subject that instant was
marked by a person, by eye.

Today you get 2,500 real recordings from northern California, each already marked by an analyst.
You will watch the standard automatic picker miss, and then train a small network to find the
instant for you.
"""

md(weekkit.OPENING.format(question=WEEK["question"], datahub=datahub, hook=HOOK.strip()))

md("""
## What you'll be able to do

**The science.** Say what a seismogram contains and why the P and S arrival times are the
quantity a seismologist actually wants. Show that loudness answers *whether* an earthquake
happened but not *when* it started, and say how well the classical automatic picker does on real
data before anything is learned from it.

**The skills.** Slide a pattern-detector along a signal with `np.convolve`. Build a neural
network in PyTorch out of `nn.Conv1d`, `nn.ReLU` and `nn.Upsample`, give it a loss to make small,
train it with gradient descent for a fixed number of epochs, and read its learning curve to see
when more training stops buying anything.

**Eight places where you write something: five in class, three at home.** Each one is headed
*Your turn*, with an empty cell under it.

**The four questions, in order:**

1. What is in a seismogram, and why is the arrival time the number a seismologist wants?
2. Was there an earthquake at all — and does loudness say when it started?
3. How well can a picker do with numbers we choose by hand?
4. Can a machine that chooses its own numbers beat it?
""")

code(weekkit.download_setup_cell(
    imports="import numpy as np\nimport torch\nfrom torch import nn\n"
            "from sklearn.linear_model import LinearRegression\n",
    const="WAVEFORMS", url=DATA_URL, filename="phasenet_ncedc.npz",
    docstring="Read the waveform file, downloading it from the course release the first "
              "time.",
    unpack='''
# 42 MB of waveforms — too big to keep beside the notebook, so it arrives from a release of the
# course repository rather than from the repository itself, and is kept once it has arrived.
data = load()
waveform = np.clip(data["waveform"], -10, 10)   # a guard: the file is already cut off here
p_index = data["p_index"].astype(int)           # sample number of the analyst's P pick
s_index = data["s_index"].astype(int)           # sample number of the analyst's S pick
distance_km = data["distance_km"]               # station to epicentre, along the surface
depth_km = data["depth_km"]                     # how far below the epicentre the quake was
event_id = data["event_id"]
station = data["station"]
magnitude = data["magnitude"]
snr = data["snr"]                               # how much louder the quake is than the background
SAMPLE_RATE = 100                               # samples per second

print("recordings:      ", waveform.shape)
print("earthquakes:     ", len(np.unique(event_id)), " stations:", len(np.unique(station)))
print("magnitudes:      ", round(magnitude.min(), 2), "to", round(magnitude.max(), 2),
      " median", round(np.median(magnitude), 2))
print("distance (km):   ", round(np.median(distance_km), 1), "median,",
      round(distance_km.max(), 1), "furthest")
print("depth (km):      ", round(depth_km.min(), 1), "to", round(depth_km.max(), 1))
print("P arrives between", round(p_index.min() / SAMPLE_RATE, 2), "and",
      round(p_index.max() / SAMPLE_RATE, 2), "seconds in")
print("samples sitting exactly on the ±10 cut-off:",
      round((np.abs(waveform) >= 10).mean(), 3))
'''.strip("\n")))

# --- section 1 -------------------------------------------------------------
md(f"""
## What is in a seismogram, and why is the arrival time the number a seismologist wants?

`waveform` holds {M['n_traces']:,} recordings. Each one is a small array of its own: **3 rows**,
because a seismometer measures the ground moving east, north and up-down at the same time, and
**{M['n_samples']:,} columns**, one per sample. At {SAMPLE_RATE} samples a second that is
{M['seconds']} seconds of shaking per recording.

These come from the Northern California Earthquake Data Center, whose analysts marked the P and
the S on every one by hand. `p_index` and `s_index` are those marks, stored as sample numbers, so
sample {M['n_samples'] // 2} means {M['n_samples'] // 2 / SAMPLE_RATE:.2f} seconds into the
window. Each row has been divided by its own typical size, so a "1" in `waveform` means *one
typical wiggle for this instrument on this recording*, not a fixed number of nanometres —
loudness here is always relative to the same trace's own background.

It was then cut off at ±10, and that bound is worth remembering, because you will meet it twice
more. **The file already carries the cut**, so the `np.clip` in the setup cell changes nothing —
it is a guard, not a cleaning step. What the cut does do is throw away the tops of the big
arrivals, and the last line the setup cell printed is how much of the file that touches:
{M['clip_frac']:.1%} of all the samples sit exactly on ±10. Any "biggest swing" you measure is
therefore a number with a ceiling on it, and the ceiling is ours.

One thing was done deliberately when the file was built: the window was cut at a **random** offset
before each P, so the arrival lands anywhere from {M['p_min_s']} to {M['p_max_s']} seconds in.
Nothing here can score well by always answering "sample {M['n_samples'] // 2}".

Look at one. The three rows are drawn on the same axes, and the two vertical lines are the
analyst's marks. Before the first line there is nothing but background. The P is the first
arrival; the S, which follows it, shakes harder and is the wave that does the damage.
""")

code(f"""
seconds = np.arange({M['n_samples']}) / SAMPLE_RATE
example = waveform[{TRACE}]

for row, name in zip(example, ["east", "north", "up-down"]):
    plt.plot(seconds, row, lw=0.6, label=name)
plt.axvline(p_index[{TRACE}] / SAMPLE_RATE, color="k")
plt.axvline(s_index[{TRACE}] / SAMPLE_RATE, color="k")
plt.xlabel("time (s)")
plt.ylabel("ground motion (in units of this trace's own size)")
plt.title("station {M['trace_station']}, magnitude {M['trace_mag']} — 1 of "
          "{M['n_traces']:,} recordings; the lines are P and S")
plt.legend()
plt.show()
""")

ask(f"""
### ✏️ Your turn 1

Draw a different one. Choose any trace number you like between 0 and {M['n_traces'] - 1}, plot
just its **up-down** row (that is `waveform[my_trace][2]`) against `seconds`, mark the P and the S
with `plt.axvline`, and label both axes.

Then print four things in seconds: the P time, the S time, the gap between them, and the moment
of the **largest swing** on that row. `np.abs(row).argmax()` gives the sample number of the
largest value, and dividing a sample number by `SAMPLE_RATE` turns it into seconds.

**Use these names**, because the self-check looks for them: `my_trace` for the trace number,
`sp_seconds` for the gap, and `loudest_second` for the moment of the largest swing.
""")

answer(f"""
my_trace = 100

plt.plot(seconds, waveform[my_trace][2], lw=0.6)
plt.axvline(p_index[my_trace] / SAMPLE_RATE, color="k")
plt.axvline(s_index[my_trace] / SAMPLE_RATE, color="k")
plt.xlabel("time (s)")
plt.ylabel("ground motion (up-down)")
plt.title("trace " + str(my_trace) + " of {M['n_traces']:,}, station " + str(station[my_trace]))
plt.show()

sp_seconds = (s_index[my_trace] - p_index[my_trace]) / SAMPLE_RATE
loudest_second = np.abs(waveform[my_trace][2]).argmax() / SAMPLE_RATE

print("P at            ", p_index[my_trace] / SAMPLE_RATE, "s")
print("S at            ", s_index[my_trace] / SAMPLE_RATE, "s")
print("gap:            ", round(sp_seconds, 2), "s")
print("biggest swing at", loudest_second, "s")
""", """
assert sp_seconds > 0, "the S always arrives after the P, so the gap must be positive"
print("✓ one trace — trace", my_trace, "has its S", round(sp_seconds, 2),
      "s after its P, and its biggest swing at", loudest_second, "s")
""")

md(f"""
That gap is the whole reason anyone cares about the exact arrival time. The P and the S leave the
earthquake together and travel at different speeds, so the further you are from it, the further
apart they arrive — the gap is a distance measurement, taken at a single station.

Which means we can check it. Fit a straight line and the slope is how many kilometres each second
of gap is worth.

It is tempting to force that line through the origin — a station standing on top of the
earthquake ought to see no gap at all. Do not. `distance_km` is measured **along the surface**,
from the station to the point above the earthquake, and the earthquake itself is kilometres
down: the depths in this file run from {M['depth_min']} to {M['depth_max']} km. A station
directly above an event 8 km deep still has 8 km of rock between it and the source, and still
sees a gap.

The cell below checks that, and prints both fits so you can see what forcing the line costs.
Nothing in this file has a gap near zero: the smallest anywhere is {M['sp_min']} s, and even the
{M['n_near2']} recordings taken within 2 km of the epicentre have a median gap of
{M['sp_near2_med']} s. The origin is not a point this data set knows anything about, and the line
that is allowed its own intercept fits better for it — R squared {M['sp_r2']} against
{M['sp_r2_origin']}. Notice where the orange line stops, too: at the smallest and largest gaps
actually measured, because a fitted line says nothing outside the range you fitted it on.
""")

code(f"""
sp_all = (s_index - p_index) / SAMPLE_RATE
near_epicentre = distance_km < 2            # stations practically on top of the epicentre

line = LinearRegression()                   # with an intercept: the line need not reach (0, 0)
line.fit(sp_all.reshape(-1, 1), distance_km)
origin = LinearRegression(fit_intercept=False)
origin.fit(sp_all.reshape(-1, 1), distance_km)

ends = np.array([sp_all.min(), sp_all.max()]).reshape(-1, 1)   # only where there is data

plt.scatter(sp_all, distance_km, s=3, alpha=0.3)
plt.plot(ends, line.predict(ends), color="C1")
plt.xlabel("S minus P (s)")
plt.ylabel("distance to the epicentre (km)")
plt.title("{M['n_traces']:,} recordings: the gap is a distance measurement")
plt.show()

print("kilometres per second of gap:", round(line.coef_[0], 2))
print("R squared, with an intercept: ", round(line.score(sp_all.reshape(-1, 1), distance_km), 3))
print("R squared, forced through 0:  ",
      round(origin.score(sp_all.reshape(-1, 1), distance_km), 3))
print("smallest gap in the file:", round(sp_all.min(), 2), "s;",
      near_epicentre.sum(), "recordings within 2 km, median gap",
      round(np.median(sp_all[near_epicentre]), 2), "s")
""")


# --- section 2 -------------------------------------------------------------
md(f"""
## Was there an earthquake at all?

Before *when*, the easier question: **did anything happen?** Cut two windows out of every
recording, each {M['window_s']} seconds long — one ending just before the analyst's P, one
starting just after it. The first contains only background noise. The second contains an
earthquake. A machine that can tell those apart is an earthquake detector.

We are going to be scoring things from here to the end of the notebook, so build the held-out set
first — and build it by **earthquake**, not by recording. Several stations recorded the same
event; if some of those recordings go into training and the rest into testing, the test is not
held out at all. That is leakage, and grouping by `event_id` is the fix.
""")

code(f"""
{SRC_SPLIT}

print("training on", is_train.sum(), "recordings from", len(train_events), "earthquakes")
print("testing on ", (~is_train).sum(), "recordings from",
      len(events) - len(train_events), "earthquakes")
""")

code(f"""
WINDOW = {WINDOW}
quiet = np.zeros((len(waveform), 3, WINDOW), dtype="float32")
shaken = np.zeros((len(waveform), 3, WINDOW), dtype="float32")
for i in range(len(waveform)):
    quiet[i] = waveform[i][:, p_index[i] - WINDOW - 6:p_index[i] - 6]
    shaken[i] = waveform[i][:, p_index[i] + 6:p_index[i] + 6 + WINDOW]

loud_quiet = np.abs(quiet).max(axis=(1, 2))     # biggest swing in each before-window
loud_shaken = np.abs(shaken).max(axis=(1, 2))   # biggest swing in each after-window

print("typical biggest swing before the P:", round(np.median(loud_quiet[is_train]), 2))
print("typical biggest swing after the P: ", round(np.median(loud_shaken[is_train]), 2))
print("after-windows sitting exactly on the cut-off: ",
      round((loud_shaken[is_train] == 10).mean(), 3))
print("before-windows sitting exactly on the cut-off:",
      round((loud_quiet[is_train] == 10).mean(), 3))
""")

md(f"""
Those two medians came from the training recordings only, so we are allowed to look at them — but
read the second one before you use it. **{M['shaken_med']} is the cut-off**, not a measurement:
{M['shaken_railed']:.1%} of the after-windows are pinned exactly on ±10 by the clip the file was
written with, so their median can only ever come back as the bound itself. All that number
honestly says is *most of these earthquakes reach the ceiling*. The before-windows are barely
censored — {M['quiet_railed']:.1%} of them touch the bound — so {M['quiet_med']} is a real
number, and it is the one to build on.

Now the rule. **Write the dumbest rule you can, first. Any model that cannot beat it is
decoration.** The dumbest rule here is one number: call it an earthquake when the biggest swing
in the window is above {M['threshold']}, which is more than twice a typical quiet window and far
below where the shaken ones pile up.
""")

ask(f"""
### ✏️ Your turn 2

Score the dumb rule on the held-out recordings only.

A before-window is correct when its biggest swing is **below** {M['threshold']}; an after-window
is correct when its biggest swing is **at or above** {M['threshold']}. Count both, add them up,
and divide by the number of windows you scored — remember every held-out recording gives you two
windows, one of each kind.

**Use these names**, because the self-check looks for them: `detector_accuracy`.
""")

answer(f"""
threshold = {M['threshold']}
right = ((loud_quiet[~is_train] < threshold).sum()
         + (loud_shaken[~is_train] >= threshold).sum())

detector_accuracy = right / (2 * (~is_train).sum())
print("windows scored:", 2 * (~is_train).sum())
print("accuracy:", round(detector_accuracy, 3))
""", """
assert detector_accuracy <= 1, "an accuracy is a fraction of the windows, so it cannot exceed 1"
print("✓ the one-number rule — it is right on",
      round(100 * detector_accuracy, 1), "% of held-out windows")
""")

md(f"""
So *detection* barely needs us. One `if` statement, one number, and it is right on about
{M['detect_acc']:.0%} of held-out windows. Keep that number, because it is the **bar**: the
week's second rule is that a new method has to beat the simple one, measured, on the same
recordings — and a network arriving now would be starting from {M['detect_acc']:.0%}, not from
zero. Notice what we have not done: we have not built a network for detection, so nothing on
this page entitles us to say one would be better *or* worse. (When this dataset was audited
before the course, a small convolutional network trained on windows like these did not beat the
one-line rule. That is what the rule is for.)

The interesting question was never whether the ground shook. It is **when it started**, and your
own trace gave you one data point on that: compare the moment of its largest swing with the moment
of its P. Here is the same comparison, over all {M['n_traces']:,} recordings at once.
""")

code(f"""
{SRC_STRENGTH}
loudest = strength.argmax(axis=1)           # sample number of the biggest swing in each recording
near_p = np.abs(loudest - p_index) <= 50    # 50 samples is half a second
near_s = np.abs(loudest - s_index) <= 50

print("loudest sample is within 0.5 s of the P:", round(near_p.mean(), 3))
print("loudest sample is within 0.5 s of the S:", round(near_s.mean(), 3))

plt.hist((loudest - p_index) / SAMPLE_RATE, bins=60)
plt.axvline(0, color="k")
plt.xlabel("loudest sample minus P arrival (s)")
plt.ylabel("number of recordings")
plt.title("where the biggest swing sits, {M['n_traces']:,} recordings; 0 is the P")
plt.show()
""")

md(f"""
The tallest bar does sit against the line — but it holds only about a sixth of the recordings,
and the rest are strung out for seconds afterwards, with a scatter to the left as well, on
recordings where the loudest thing in the window happened before the earthquake got there at all.
{M['loudest_near_p']:.1%} of the loudest samples are within half a second of the P;
{M['loudest_near_s']:.1%} are within half a second of the **S**. Loudness is a fine answer to *did
something happen*. It is close to useless as an answer to *when did it start*, because the loudest
moment is usually a different wave arriving.

Whatever finds the P has to work on the **shape** of the signal — the moment a quiet trace stops
being quiet — and not on how big it gets.
""")

# --- section 3 -------------------------------------------------------------
md("""
## How well can a picker do with numbers we choose by hand?

Seismology has had an automatic answer to this for decades, and it is two averages and a
division. Take the average power over a **short** window ending at the current sample, take it
again over a **long** window ending at the same sample, and divide. In background noise the two
averages are the same and the ratio sits near 1. The instant a wave arrives, the short window
fills with the new energy while the long window is still mostly old quiet — so the ratio jumps.
Trigger the first time it crosses some level, and that is your pick.

It is called STA/LTA, for short-term average over long-term average. Three numbers to choose: how
short, how long, and how big a jump counts. The ratio below is left at zero until there is a full
long window behind it to average over — five seconds, at this setting.
""")

code(f"""
{SRC_STALTA}


ratio = sta_lta(strength[{TRACE}], 50, 500)
plt.plot(seconds, ratio, lw=0.8)
plt.axhline(3, color="C1")
plt.axvline(p_index[{TRACE}] / SAMPLE_RATE, color="k")
plt.xlabel("time (s)")
plt.ylabel("short average / long average")
plt.title("STA/LTA on 1 trace — black is the analyst's P, orange is the trigger level")
plt.show()
""")

md(f"""
### Predict before you run

That is the standard picker, on a clear recording, landing on the arrival. Now the whole held-out
set: on what fraction of {M['n_test']} recordings do you think it puts the pick within half a
second of the analyst's? Commit to a number before you run the next cell — change `my_guess` to
whatever you think, then run it.
""")

CELLS.extend(("code", s, a) for s, a in
             weekkit.predict_cell("0.85", "of the held-out recordings get a pick within half a "
                                          "second of the analyst's"))

code(f"""
{SRC_STALTA_HITS}


print("you guessed:", my_guess)
print("textbook setting (0.5 s, 5 s, threshold 3):",
      round(sta_lta_hits(50, 500, 3).mean(), 3))
""")

ask(f"""
### ✏️ Your turn 3

One setting is not a result. Run `sta_lta_hits` on three more settings and print the fraction each
one gets, so that we know whether {M['stalta_textbook']:.3f} is what STA/LTA can do or merely what
this particular setting does. `sta_lta_hits` hands back one True/False per held-out trace, so
`.mean()` on it is the fraction. Try these:

| short | long | threshold |
|---|---|---|
| 30 | 300 | 3 |
| 20 | 200 | 3 |
| 50 | 500 | 5 |

Then print the best of the four scores.

**Use these names**, because the self-check looks for them: `stalta_scores` for the list you
collect, and `best_stalta` for the best of the four.

(Each call is a Python loop over every sample of every held-out trace, so none of them is
instant.)
""")

answer(f"""
stalta_scores = []
for short, long, threshold in [(30, 300, 3), (20, 200, 3), (50, 500, 5)]:
    score = sta_lta_hits(short, long, threshold).mean()
    stalta_scores.append(score)
    print(short, long, threshold, "->", round(score, 3))

best_stalta = max(stalta_scores + [{M['stalta_textbook']}])
print("best of the four:", round(best_stalta, 3))
""", f"""
assert max(stalta_scores) > {M['stalta_textbook']}, \\
    "at least one of these three should beat the textbook setting's {M['stalta_textbook']}; " \\
    "if none does, check the argument order — sta_lta_hits(short, long, threshold)"
print("✓ the sweep — STA/LTA's best of four settings is",
      round(100 * best_stalta, 1), "%")
""")

md(f"""
Tuned as well as four tries can tune it, the classical picker lands within half a second on
{M['stalta_best']:.1%} of held-out traces — at short {M['stalta_best_setting'].split(', ')[0]},
long {M['stalta_best_setting'].split(', ')[1]}, threshold
{M['stalta_best_setting'].split(', ')[2]}. Both scores are now on your screen —
{M['stalta_textbook']:.3f} at the textbook setting, {M['stalta_best']:.3f} at the best — so before
reading on, look at what changed between them. It is worth
{100 * (M['stalta_best'] - M['stalta_textbook']):.1f} percentage points, which is a large amount
of tuning for two numbers.

It is not tuning. Go back to the sentence above the first STA/LTA cell: *the ratio is left at
zero until there is a full long window behind it to average over.* At the textbook setting the
long window is 500 samples, so for the first **five seconds** of every recording the ratio is
identically zero and cannot cross anything. And the setup cell told you the P lands anywhere from
{M['p_min_s']} to {M['p_max_s']} seconds in. Split the held-out set on that, and score **both**
settings on **both** halves — the comparison only means something if the two numbers you set side
by side were taken over the same recordings.
""")

code(f"""
textbook_ok = sta_lta_hits(50, 500, 3)
best_ok = sta_lta_hits({BEST[0]}, {BEST[1]}, {BEST[2]})        # your winner from Your turn 3

# two things `sta_lta_hits` does not answer: did the textbook setting land on the S instead, and
# did it trigger at all? Both need the pick itself rather than whether it was right.
on_the_s = []
never = []
for i in np.nonzero(~is_train)[0]:
    pick = first_trigger(sta_lta(strength[i], 50, 500), 3)
    never.append(pick is None)
    on_the_s.append(pick is not None and abs(pick - s_index[i]) <= 50)

on_the_s = np.array(on_the_s)
never = np.array(never)
early = p_index[~is_train] < 450        # a P this early is gone before the ratio starts moving
dead = (waveform[~is_train].std(axis=2) == 0).all(axis=1)   # nothing recorded on any row

print("P earlier than 4.5 s:", early.sum(), " later:", (~early).sum())
print("  textbook setting, early:", round(textbook_ok[early].mean(), 3),
      " late:", round(textbook_ok[~early].mean(), 3))
print("  best setting,     early:", round(best_ok[early].mean(), 3),
      " late:", round(best_ok[~early].mean(), 3))
print("triggered on the S:", round(on_the_s.mean(), 3),
      " of those, share whose P is in the dead zone:", round(early[on_the_s].mean(), 3))
print("never triggered at all:", never.sum(),
      " of those, P in the dead zone:", (never & early).sum(),
      " nothing recorded at all:", (never & dead).sum())
""")

md(f"""
{M['n_early']} of the {M['n_test']} held-out recordings — {M['early_frac']:.1%} of them — have
their P before 4.5 s, and on those the textbook setting scores **{M['textbook_early']:.3f}**. Not
"badly": zero. However clear the arrival, it happened inside a window where the ratio was still
zero, and no threshold can be crossed there. That is a **dead zone**, and we built it ourselves
when we chose a five-second long window for recordings whose P can arrive as early as
{M['p_min_s']} seconds.

Now read the two settings against each other, on the same recordings both times. On the
{M['n_late']} whose P arrives **after** the dead zone the textbook setting scores
{M['textbook_late']:.3f} and your best setting scores {M['best_late']:.3f} — the same picker, to
three decimal places. On the {M['n_early']} **inside** the dead zone they score
{M['textbook_early']:.3f} and {M['best_early']:.3f}. So the {M['stalta_textbook']:.3f} →
{M['stalta_best']:.3f} jump you measured over the whole held-out set is not the shorter windows
being better at finding P arrivals. It is the shorter windows having a shorter dead zone. Take the
dead zone away and the two settings really are the same picker.

That last comparison is the one worth copying. It would have been easy to stop a step earlier and
say "{M['textbook_late']:.3f} on the late traces, and the sweep gave {M['stalta_best']:.3f} — same
number". But those two are averages over different sets of recordings, {M['n_late']} against
{M['n_test']}, and two rounded numbers agreeing across different denominators is a coincidence,
not a demonstration. Scoring both settings on both halves is what turns it into one.

The S triggers say the same thing from the other side. {M['stalta_near_s']:.1%} of the held-out
recordings triggered on the S rather than the P, and {M['s_in_dead_zone']:.0%} of *those* are
recordings whose P was in the dead zone — so the S was simply the first arrival the ratio was
awake for. The {M['n_never']} that never
triggered at all are a different story, and the cell keeps them apart rather than lumping them in:
only {M['never_in_dead']} of those {M['n_never']} have their P in the dead zone. The rest are
recordings where the ratio was wide awake and the arrival was still too gentle to cross 3 — and
{M['never_dead_channel']} of them are recordings with no ground motion on them at all, which you
will meet again in the last figure of the notebook.

The general lesson is worth more than the seismology. **A parameter sweep will happily hand you a
winner without telling you what it won on.** Ours was not measuring how well the picker finds
arrivals; it was measuring how much of each recording the picker was allowed to look at.

That is still the number to beat. Anything we build now has to beat {M['stalta_best']:.3f}, or we
have built decoration.
""")

# --- section 4: the second half of spine question 3 -------------------------
md(f"""
Look again at what each of STA/LTA's two averages is. Take a small list of weights — one over the
window length, repeated — line it up against the samples ending at the current one, multiply and
add. Then move along one sample and do it again. That sliding-and-summing has a name: it is a
**convolution**. {WORDS['1-D CNN']} STA/LTA does it twice, with two
different window lengths, and divides the answers.

The small list of weights is the detector, and what it detects depends entirely on the numbers in
it. Put a step in the weights and it responds where the signal steps. `np.convolve` slides it for
you — lining the weights up in reverse, which is the difference between a convolution and a
sliding dot product, so the *first* half of the list is the half that lands on the most recent
samples.
""")

code(f"""
def onset_filter(width):
    \"\"\"Energy in the last `width` samples minus energy in the `width` samples before those.\"\"\"
    return np.concatenate([np.ones(width) / width, -np.ones(width) / width])


response = np.convolve(strength[{TRACE}] ** 2, onset_filter(10), mode="same")

plt.plot(seconds, response, lw=0.8)
plt.axhline(3, color="C2")                      # the same trigger level STA/LTA used
plt.axvline(p_index[{TRACE}] / SAMPLE_RATE, color="k")
plt.axvline(s_index[{TRACE}] / SAMPLE_RATE, color="C1")
plt.xlabel("time (s)")
plt.ylabel("filter response (energy jump)")
plt.title("a 20-sample onset detector on 1 trace — black P, orange S, green trigger")
plt.show()

print("first crossing of 3:", first_trigger(response, 3), " biggest response:", response.argmax())
print("the analyst's P:    ", p_index[{TRACE}], "        the analyst's S:", s_index[{TRACE}])
""")

md("""
On this recording the filter answers the question twice, and the two answers are not the same
place. Its response **first crosses 3** at the P, within a sample of the analyst's mark. Its
**largest** value is at the S — which is where the biggest jump in energy in a seismogram usually
sits, though "usually" is doing real work in that sentence and the next cell is where you find out
how much: on a fair number of these recordings the largest response is at neither mark.

Which of those two is "the detector's pick" is a decision we make, not something the twenty
weights decide, so score it both ways.
""")

ask(f"""
### ✏️ Your turn 4

Score that hand-made detector on the held-out traces, both ways, from the same responses.

Loop over the held-out traces and convolve `strength[i] ** 2` with `onset_filter(10)`. From each
response take two picks: `first_trigger(response, 3)` — exactly the rule you scored STA/LTA with,
same threshold — into `hand_picks`, and `response.argmax()` into `peak_picks`.

Then print three fractions: how often `hand_picks` lands within 50 samples of `p_index[i]`, how
often `peak_picks` does, and — because the figure above hints at where the peak actually goes —
how often `peak_picks` lands within 50 samples of `s_index[i]`. Remember `first_trigger` can hand
back `None`, which is never within 50 samples of anything.

**Use these names**, because the self-check looks for them: `hand_picks`, `peak_picks` and
`hand_accuracy` for the first of the three fractions.
""")

answer(f"""
hand_picks = []
peak_picks = []
for i in np.nonzero(~is_train)[0]:
    response = np.convolve(strength[i] ** 2, onset_filter(10), mode="same")
    hand_picks.append(first_trigger(response, 3))
    peak_picks.append(response.argmax())

crossing_hits = 0
peak_hits = 0
peak_near_s = 0
for crossing, peak, i in zip(hand_picks, peak_picks, np.nonzero(~is_train)[0]):
    crossing_hits = crossing_hits + (crossing is not None and abs(crossing - p_index[i]) <= 50)
    peak_hits = peak_hits + (abs(peak - p_index[i]) <= 50)
    peak_near_s = peak_near_s + (abs(peak - s_index[i]) <= 50)

hand_accuracy = crossing_hits / len(hand_picks)
print("first crossing, within 0.5 s of the P:", round(hand_accuracy, 3))
print("biggest response, within 0.5 s of the P:", round(peak_hits / len(peak_picks), 3))
print("biggest response, within 0.5 s of the S:", round(peak_near_s / len(peak_picks), 3))
""", """
assert len(hand_picks) == len(peak_picks) == (~is_train).sum(), \
    "one of each pick per held-out trace, and none of the rest"
print("✓ the hand-made detector — read by first crossing it finds the P on",
      round(100 * hand_accuracy, 1), "% of held-out traces")
""")

md(f"""
Twenty weights, one set of responses, and a factor of {M['hand_acc'] / M['hand_argmax_acc']:.1f}
between the two ways of reading them. Read by first crossing, the hand-made filter finds the P on
{M['hand_acc']:.1%} of held-out traces — a real picker, and only a little behind STA/LTA's
{M['stalta_best']:.1%}. Read by taking its largest value, the same responses find the P on
{M['hand_argmax_acc']:.1%} and land on the **S** on {M['hand_argmax_near_s']:.1%} — which settles
the "usually" above. The biggest jump in energy is at the S more often than anywhere else, but on
well under two-thirds of these recordings, and on the ones left over it is at neither mark. A
tendency, not a rule.

So the weights are not what separates a good picker from a bad one here. The decision rule is,
and the reason is a property of every fixed filter of this kind: **it has no way to be quiet away
from the arrival.** Its response is large wherever the signal is large, so its biggest value
tracks the loudest wave rather than the first one. Only a threshold rescues it — and the
threshold is one more number chosen by hand, for this dataset, at this normalisation.

What we would rather have is something whose output is near zero everywhere except at the P, so
that simply taking the largest value is the right thing to do and no threshold is needed at all.
That is a much stronger requirement than "respond to onsets", and it is not a shape anyone should
try to guess in twenty numbers. So stop guessing: hand the machine the shape we want back, and
let it choose the numbers.
""")

# --- section 5 -------------------------------------------------------------
md(f"""
## Can a machine that chooses its own numbers beat it?

**{WORDS['Neural network']}** That is all a neural network is. Take the piece it stacks first.
**{WORDS['Perceptron']}** That is a **perceptron**. Put several of them side by side, feed their
outputs into another row of them, and you have a stack.

Between the rows sits one more step. **{WORDS['Activation']}** That is the **activation**, and it
is what makes stacking worth anything: without it, a weighted sum of weighted sums is still a
weighted sum, however many rows you use. Here are two stacks of identical shape, one with the bend
and one without.
""")

code("""
grid = torch.linspace(-3, 3, 200).reshape(-1, 1)
torch.manual_seed(1)
flat = nn.Sequential(nn.Linear(1, 8), nn.Linear(8, 1))
torch.manual_seed(1)
bent = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))

plt.plot(grid.numpy(), flat(grid).detach().numpy(), label="2 layers, no activation")
plt.plot(grid.numpy(), bent(grid).detach().numpy(), label="2 layers, with ReLU")
plt.xlabel("input")
plt.ylabel("output")
plt.title("what 8 hidden units can draw, with and without an activation")
plt.legend()
plt.show()
""")

md(f"""
Three more words and we can build one.

**{WORDS['Loss']}** That is the **loss**, and ours is the same one you used to fit a straight
line: the average squared miss between what the network says and what we wanted.

**{WORDS['Gradient descent']}** That is **gradient descent** — the loss depends on every weight in
the network, PyTorch works out which way each weight would have to move to make the loss smaller,
and every weight takes a small step that way. **{WORDS['Epoch']}** That is an **epoch**, and
training is doing it again and again.

The last thing to decide is what "what we wanted" means. We are not asking for a number. We are
asking the network, at each of the {M['n_samples']:,} samples, *how much does this look like the
P arrival* — so the answer we train it towards is a bump centred on the analyst's mark, and the
pick we read back out is wherever the network's answer is highest.

That is not a trick invented for this notebook. It is how the pickers that time modern earthquake
catalogues work: Zhu, W. and Beroza, G.C. (2019), *PhaseNet: a deep-neural-network-based seismic
arrival-time picking method*, Geophysical Journal International 216, 261–273, trains on exactly
this kind of input — three components at 100 Hz, P and S marked by analysts — towards exactly this
kind of bump, with a much larger network and a great many more recordings. What you are about to
build is that idea at classroom scale.
""")

code(f"""
{SRC_TARGET}


plt.plot(seconds, waveform[{TRACE}][2] / np.abs(waveform[{TRACE}][2]).max(), lw=0.6,
         label="up-down, scaled")
plt.plot(seconds, make_target(p_index[{TRACE}:{TRACE} + 1], 20)[0], label="what we want back")
plt.xlabel("time (s)")
plt.ylabel("scaled to 1")
plt.title("1 trace and its training target — a bump of width 20 samples on the P")
plt.legend()
plt.show()
""")

md(f"""
Now the network. `nn.Conv1d(3, 8, 7)` is eight pattern-detectors, each 7 samples long, each
looking at all 3 rows at once — the same sliding operation as before, except that the numbers
inside are what training will choose. `stride=4` slides in steps of 4 instead of 1, which makes
the signal four times shorter and lets the next layer's 7 samples cover four times as much time;
`padding=3` puts three zeros on each end, so a 7-sample detector still has something to line up
against right at the edges. That is what makes the shortening exact rather than approximate: our
two stride-4 layers take {M['n_samples']:,} samples to 512 and then to 128, a clean quarter each
time instead of a few samples short, and a stride-1 layer with the same padding comes out exactly
as long as it went in. `nn.Upsample` then stretches the whole thing back out at the end, so the
answer is one number per original sample.

In PyTorch a model is an `nn.Module` — a box that holds the numbers to be learned and knows how
to run them. `nn.Sequential` is the simplest one there is: hand it layers, and it runs them in
order. Building it is instant; the cell after that is the slow one — it trains the network from
scratch and prints nothing while it works, so start it and let it run.
""")

code(f"""
{SRC_MODEL}

print("weights to learn:", sum(w.numel() for w in make_picker().parameters()))
""")

code(f"""
{SRC_TRAIN}


print("training 25 epochs — the slow cell; nothing more prints until it has finished")
picker, losses, scores = train_picker()
print("trained for", len(losses), "epochs")
""")

# --- section 6: the second half of spine question 4 -------------------------
md("""
Two lines, on two axes because they are in different units. On the left, the loss on the data the
network trained on. On the right, the fraction of **held-out** traces it picks within half a
second — recordings of earthquakes it has never seen.

The left line is what gradient descent is directly pushing down, so it should fall smoothly. The
right line is the one we actually care about, and nothing is pushing on it at all.

**Read the two panels in opposite directions.** The left one is an error, so down is better; the
right one is a score, so **up** is better. This pair is a *learning curve*, the same idea as when
we chose how many bends a curve was allowed — and what it is for is the moment the two panels
stop agreeing. Training error still falling while the held-out score has stopped climbing is the
signal that more training is buying a closer fit to the training set and nothing else.
""")

code(f"""
epochs = np.arange(1, len(losses) + 1)

plt.figure(figsize=(9, 3.5))                    # two panels side by side need a wider figure
plt.subplot(1, 2, 1)
plt.plot(epochs, losses)
plt.xlabel("epoch")
plt.ylabel("training loss")

plt.subplot(1, 2, 2)
plt.plot(epochs, scores)
plt.xlabel("epoch")
plt.ylabel("held-out traces within 0.5 s")
plt.suptitle("learning curve — {M['n_train']:,} training and {M['n_test']} held-out recordings")
plt.show()

print("after epoch 10 the held-out line still wanders by",
      round(max(scores[10:]) - min(scores[10:]), 3), "from epoch to epoch")
""")

ask(f"""
### ✏️ Your turn 5

Report the result properly, so it can be set beside {M['stalta_best']:.3f}.

Get the network's picks on the held-out traces with `picks_from(picker, x_test)`, then print
three things: the fraction within half a second, the **median** error in seconds, and the epoch
at which the held-out score was highest — `scores` is an ordinary list, so `np.argmax(scores)`
gives its position and epochs are counted from 1.

**Use these names**, because the self-check looks for them: `net_picks` and `net_accuracy`.
""")

answer(f"""
net_picks = picks_from(picker, x_test)
net_accuracy = within_half_second(net_picks, p_test)

print("within 0.5 s:", round(net_accuracy, 3))
print("median error:", round(np.median(np.abs(net_picks - p_test)) / SAMPLE_RATE, 3), "s")
print("best epoch:  ", np.argmax(scores) + 1, "of", len(scores))
""", f"""
assert abs(net_accuracy - scores[-1]) < 1e-9, \\
    "the last point of the learning curve IS this number, measured the same way on the same " \\
    "traces — if they differ, check what you scored the picks against"
print("✓ the network —", round(100 * net_accuracy, 1), "% of held-out traces within 0.5 s,",
      "median error", round(np.median(np.abs(net_picks - p_test)) / SAMPLE_RATE, 3), "s")
""")

md(f"""
Two of the held-out recordings, drawn: the one the network places most accurately and the one it
places worst. Behind the network's answer in each panel is the liveliest of that recording's three
rows, and the black line is where the analyst put the P.

The lower panel is not a plotting mistake, and it is worth reading the numbers under it carefully,
because the obvious reading is the wrong one. All three of that recording's rows have a standard
deviation of zero, and every sample on all three sits on ±10. That looks like an instrument driven
off the end of its range by enormous shaking. It is the **opposite**: the instrument recorded
nothing at all.

Remember what ±10 is. It is not the instrument's range — it is the cut-off *we* wrote the file
with. Each row was divided by its own typical size before the cut, and a row that never moves has
a typical size of zero; dividing by that blows up, and the cut catches the result and parks the
whole row on the bound. So ±10 here is the signature of a **dead channel**, not a saturated one,
and there is no ground motion in the file to find. {M['n_flat']} of the {M['n_traces']:,}
recordings are like this, and you have met them before without knowing it: {M['never_dead_channel']}
of the {M['n_never']} held-out recordings where STA/LTA never triggered at all are these same dead
channels. Neither method failed on them. There was nothing there to find. An archive of real
instruments contains records like that, and no picker can be blamed for them.
""")

code(f"""
closest = np.abs(net_picks - p_test).argmin()
furthest = np.abs(net_picks - p_test).argmax()

plt.figure(figsize=(7, 5))                      # two stacked traces need the extra height
for panel, which in [(1, closest), (2, furthest)]:
    liveliest = x_test[which].numpy().std(axis=1).argmax()   # the row with the most in it
    plt.subplot(2, 1, panel)
    plt.plot(seconds, x_test[which][liveliest].numpy() / 10, lw=0.5, label="ground motion, scaled")
    plt.plot(seconds, picker(x_test[[which]]).squeeze().detach().numpy(), lw=1,
             label="the network's answer")
    plt.axvline(p_test[which] / SAMPLE_RATE, color="k")
    plt.ylabel("scaled to 1")
    if panel == 1:
        plt.legend()
plt.xlabel("time (s)")
plt.suptitle("the network on its closest and its furthest of {{}} held-out recordings"
             .format(len(p_test)))
plt.show()

print("standard deviation of the three rows, worst recording:",
      x_test[furthest].numpy().std(axis=1).round(3))
print("the only values on those three rows:", np.unique(x_test[furthest].numpy()))
print("recordings this flat, in the whole file:",
      (waveform.std(axis=2) == 0).all(axis=1).sum(), "of", len(waveform))
""")

# --- closing ----------------------------------------------------------------
md(f"""
{weekkit.CLOSING_HEADING}

Yes, and better than the method it replaces: on {M['n_test']} held-out recordings from
earthquakes it had never seen, a network of {M['n_weights']:,} weights put the P arrival within
half a second on {M['net_acc']:.1%} of them, against {M['stalta_best']:.1%} for the best of four
STA/LTA settings — and it did it by learning the shape of an onset, on traces where the loudest
moment is a different wave entirely.
""")

md(weekkit.week_cheatsheet(13))

# --- homework ---------------------------------------------------------------
md("""
## Homework

Three parts. The first asks something about the class result that class did not ask; the second
asks you to decide something and defend the decision; the third settles an argument class left
open.

Two of the three train a network from scratch, so start them and let them run. If you restarted
the kernel since class, run the checkpoint cell first: it rebuilds everything class built — the
held-out split, the classical picker and `sta_lta_hits` that scores it, the target, the network
and the trained model itself — and prints nothing. Because it retrains, it is the slow cell rather
than a free one.
""")

code(weekkit.CHECKPOINT.format(body=SRC_CHECKPOINT))

ask(f"""
### ✏️ Your turn 6

The two pickers were scored on the same held-out recordings, and those recordings are not equally
easy: `snr` says how many times louder the earthquake is than the background on each one. An
average over all of them hides that. So where does the network's advantage actually come from —
is it ahead everywhere, or only where the signal is faint?

Split the held-out recordings in half at the median of their `snr` and report four numbers: the
network's accuracy and STA/LTA's on the quiet half, and both again on the loud half. Score
STA/LTA at its best setting from Your turn 3 — short {M['stalta_best_setting'].split(', ')[0]},
long {M['stalta_best_setting'].split(', ')[1]}, threshold
{M['stalta_best_setting'].split(', ')[2]} — with `sta_lta_hits`, which already hands back the one
True/False per held-out recording this part needs.

Print how much accuracy each method loses going from the loud half to the quiet half. Then answer
the question this part opened with, in one more printed line: is the network ahead everywhere, or
only where the signal is faint — and quote those two losses as your reason.

**Use these names**, because the self-check looks for them: `classic_ok` and `network_ok`, each
one True/False per held-out recording, and `quiet` for the mask picking out the quiet half.
""")

answer(f"""
snr_test = snr[~is_train]
quiet = snr_test < np.median(snr_test)

classic_ok = sta_lta_hits({M['stalta_best_setting'].split(', ')[0]},
                          {M['stalta_best_setting'].split(', ')[1]},
                          {M['stalta_best_setting'].split(', ')[2]})
network_ok = np.abs(net_picks - p_test) <= 50

print("quiet half,", quiet.sum(), "recordings: network", round(network_ok[quiet].mean(), 3),
      " STA/LTA", round(classic_ok[quiet].mean(), 3))
print("loud half, ", (~quiet).sum(), "recordings: network", round(network_ok[~quiet].mean(), 3),
      " STA/LTA", round(classic_ok[~quiet].mean(), 3))
print("lost going quiet: network",
      round(network_ok[~quiet].mean() - network_ok[quiet].mean(), 3),
      " STA/LTA", round(classic_ok[~quiet].mean() - classic_ok[quiet].mean(), 3))

print("The network is ahead on both halves, but its lead is made on the quiet one: going from",
      "the loud half to the quiet half it gives up",
      round(network_ok[~quiet].mean() - network_ok[quiet].mean(), 3),
      "while STA/LTA gives up", round(classic_ok[~quiet].mean() - classic_ok[quiet].mean(), 3),
      "so the faint arrivals are where the learned picker earns its keep.")
""", """
assert len(classic_ok) == len(network_ok) == len(quiet), \
    "one True/False per held-out recording, for both methods"
print("✓ by loudness — on the quiet half the network scores",
      round(network_ok[quiet].mean(), 3), "against STA/LTA's",
      round(classic_ok[quiet].mean(), 3))
""")

ask(f"""
### ✏️ Your turn 7

The bump we trained towards was 20 samples wide, and nobody made us choose 20. A narrow bump asks
the network for a precise answer and gives it almost nothing to aim at; a wide one is easy to hit
and blurry when you read the peak back off it.

**Pick one and defend it: `sigma=5` (0.05 s) or `sigma=40` (0.4 s).** Retrain with
`train_picker(sigma=...)`, and print the final held-out fraction within half a second, the median
error in seconds, and the final training loss. Then print the same three numbers for the network
you already trained at `sigma=20`, so the comparison is on the page.

Say in the same cell, as a printed line, which you chose and why.

**Use these names**, because the self-check looks for them: `my_sigma` and `my_scores`.

(Careful with the losses: a narrow bump is mostly zeros, so its loss is a smaller number even when
the network is doing worse. Losses computed against different targets are not comparable.)
""")

answer(f"""
my_sigma = 40
my_picker, my_losses, my_scores = train_picker(sigma=my_sigma)
my_picks = picks_from(my_picker, x_test)

print("sigma", my_sigma, ": within 0.5 s", round(my_scores[-1], 3),
      " median error", round(np.median(np.abs(my_picks - p_test)) / SAMPLE_RATE, 3), "s",
      " final loss", round(my_losses[-1], 5))
print("sigma 20 : within 0.5 s", round(scores[-1], 3),
      " median error", round(np.median(np.abs(net_picks - p_test)) / SAMPLE_RATE, 3), "s",
      " final loss", round(losses[-1], 5))
print("I chose sigma = 40, the wider target, because a bump 0.4 s across is something the "
      "network can find on almost every trace, and the peak of a wide bump is still one sample; "
      "the narrow target buys precision I cannot use, since the test allows half a second.")
""", f"""
assert abs(my_scores[-1] - scores[-1]) > {M['sigma_floor']}, \\
    "this run scored what the class run at sigma=20 scored, so the label width never changed " \\
    "— did you pass sigma=my_sigma to train_picker?"
print("✓ the label width — sigma", my_sigma, "scores",
      round(my_scores[-1], 3), "against", round(scores[-1], 3), "at sigma 20")
""")

ask(f"""
### ✏️ Your turn 8

In class the held-out score was still at its highest on the very last epoch, which leaves an
obvious doubt: perhaps the picker was simply not trained for long enough.

Settle it. Run `train_picker(epochs=60)` — this is the slow one — and pull four numbers out of
what it hands back: the training loss and the held-out score after epoch 25, and both again after
epoch 60. Print all four, and print the best held-out score of the whole run and the epoch it came
at as well. (A list counts from 0, so epoch 25 is at position 24.)

Training is seeded, so the first 25 epochs of this run are the very same run you did in class —
which is what the self-check uses to tell you whether you read the right position.

Then, in the **markdown** cell below the code cell, answer in two or three sentences **using your
own four numbers**: over
those 35 extra epochs, what happened to the loss, what happened to the held-out score, and what
does the pair of answers say about where this picker's remaining error is coming from? Your answer
has to name a number that would have to change before the picker could do better.

**Use these names**, because the self-check looks for them: `long_losses` and `long_scores` for
what `train_picker` hands back, and `loss25`, `score25`, `loss60`, `score60` for the four numbers
you take out of them.
""")

answer(f"""
long_picker, long_losses, long_scores = train_picker(epochs=60)

loss25 = long_losses[24]        # a list counts from 0, so epoch 25 is at position 24
score25 = long_scores[24]
loss60 = long_losses[59]
score60 = long_scores[59]

print("epoch 25: loss", round(loss25, 5), " held-out", round(score25, 3))
print("epoch 60: loss", round(loss60, 5), " held-out", round(score60, 3))
print("best held-out score:", round(max(long_scores), 3), "at epoch", np.argmax(long_scores) + 1)
""", """
assert abs(score25 - scores[-1]) < 1e-9, \\
    "the first 25 epochs of this run are the seeded run you already did in class, so score25 " \\
    "must equal the class score — if it does not, you are reading the wrong position"
print("✓ more training — the held-out score went from", round(score25, 3), "at epoch 25 to",
      round(score60, 3), "at epoch 60, while the loss fell from", round(loss25, 5),
      "to", round(loss60, 5))
""")

answer_prose(f"""
Over the extra 35 epochs the training loss kept falling, from {M['long_loss25']} at epoch 25 to
{M['long_loss60']} at epoch 60, so the network went on learning the whole time. The held-out score
did not follow it: {M['long_acc25']} at epoch 25 and {M['long_acc60']} at epoch 60, and the best
value of the entire run, {M['long_best_acc']} at epoch {M['long_best_epoch']}, is only
{100 * (M['long_best_acc'] - M['long_acc25']):.1f} points above where it already was — smaller
than the {100 * M['acc_wobble']:.1f}-point range the class learning curve wandered over after
epoch 10, so I cannot call it an improvement at all. More training bought a closer fit to the
training labels and no better picker.

The number that would have to change is therefore not the epoch count but the labels. Every target
is a bump centred on where one analyst put the P, and on a faint arrival that mark is itself
uncertain by a few tenths of a second; a picker that already agrees with the analyst to a median
of {M['net_med_err']} s has nothing left to agree with more closely. To go further you would have
to improve the marks — several analysts per trace, and only the ones they agree on — rather than
run more epochs.
""")


# ---------------------------------------------------------------------------
# 3. emit, execute, gate
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    sol = notebook([cell(k, s) for k, s, _ in CELLS])
    stu = notebook([cell(k, alt if alt is not None else s) for k, s, alt in CELLS])

    sol_path = OUT / f"{SLUG}_solution.ipynb"
    sol_path.write_text(json.dumps(sol, indent=1) + "\n")

    # Execute somewhere disposable. The notebook keeps the 42 MB download in its working
    # directory, which on DataHub is the right thing and in this repository is not.
    run_dir = pathlib.Path(tempfile.mkdtemp(prefix="wk13-run-"))
    shutil.copy(LOCAL, run_dir / "phasenet_ncedc.npz")

    # weekkit.execute pins the kernel to THIS interpreter — this week needs torch, which
    # the shared base environment does not carry. run_dir is the cwd because the notebook
    # keeps its 42 MB download beside itself.
    print(f"executing {sol_path.name} in {run_dir} ...")
    r = weekkit.execute(sol_path, timeout=2400, cwd=run_dir)
    shutil.rmtree(run_dir, ignore_errors=True)
    if r.returncode:
        print(r.stderr[-6000:])
        sys.exit("the solution did not execute")

    (OUT / f"{SLUG}.ipynb").write_text(json.dumps(stu, indent=1) + "\n")
    print(f"wrote {SLUG}.ipynb ({len(CELLS)} cells) and the executed solution")
    print("measured:", json.dumps(M, indent=1))


if __name__ == "__main__":
    main()
    weekkit.gate(13)
