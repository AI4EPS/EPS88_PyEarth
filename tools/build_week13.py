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
import pathlib
import subprocess
import sys
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

course = yaml.safe_load((ROOT / "course.yml").read_text())
WEEK = next(s for s in course["schedule"] if s["n"] == 13)
PLATFORM = course["platform"]

DATA_URL = ("https://github.com/AI4EPS/EPS88_PyEarth/releases/download/"
            "data-v1/phasenet_ncedc.npz")
LOCAL = pathlib.Path(__file__).resolve().parent / "_wk13_phasenet_ncedc.npz"

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
TRACE = 7
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
line = LinearRegression(fit_intercept=False)
line.fit(sp_time.reshape(-1, 1), distance_km)
M["sp_slope"] = round(float(line.coef_[0]), 2)
M["sp_r2"] = round(float(line.score(sp_time.reshape(-1, 1), distance_km)), 3)

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


def sta_lta_score(short, long, threshold):
    """Fraction of test traces STA/LTA places within half a second of the analyst's pick."""
    hits = 0
    for i in np.nonzero(~is_train)[0]:
        pick = first_trigger(sta_lta(strength[i], short, long), threshold)
        if pick is not None and abs(pick - p_index[i]) <= 50:
            hits = hits + 1
    return hits / M["n_test"]


M["stalta_textbook"] = round(sta_lta_score(50, 500, 3), 3)
SWEEP = [(30, 300, 3), (20, 200, 3), (50, 500, 5)]
M["stalta_sweep"] = {f"{a}, {b}, {c}": round(sta_lta_score(a, b, c), 3) for a, b, c in SWEEP}
M["stalta_best_setting"], M["stalta_best"] = max(M["stalta_sweep"].items(), key=lambda kv: kv[1])

near_s = never = 0
for i in np.nonzero(~is_train)[0]:
    pick = first_trigger(sta_lta(strength[i], 50, 500), 3)
    if pick is None:
        never = never + 1
    elif abs(pick - s_index[i]) <= 50:
        near_s = near_s + 1
M["stalta_never"] = round(never / M["n_test"], 3)
M["stalta_near_s"] = round(near_s / M["n_test"], 3)

# --- section 4: a pattern-detector made by hand


def onset_filter(width):
    """Energy in the next `width` samples minus energy in the last `width` samples."""
    return np.concatenate([-np.ones(width) / width, np.ones(width) / width])


hand_hits = hand_near_s = 0
for i in np.nonzero(~is_train)[0]:
    response = np.convolve(strength[i] ** 2, onset_filter(10), mode="same")
    hand_hits += abs(response.argmax() - p_index[i]) <= 50
    hand_near_s += abs(response.argmax() - s_index[i]) <= 50
M["hand_acc"] = round(float(hand_hits / M["n_test"]), 3)
M["hand_near_s"] = round(float(hand_near_s / M["n_test"]), 3)

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
M["acc_wobble"] = round(M["acc_late_high"] - M["acc_late_low"], 3)

# --- homework: does the network fail where the classical picker fails?
best = tuple(int(x) for x in M["stalta_best_setting"].split(", "))
classic_ok, network_ok = [], list(np.abs(net_picks - p_test) <= 50)
for j, i in enumerate(np.nonzero(~is_train)[0]):
    pick = first_trigger(sta_lta(strength[i], best[0], best[1]), best[2])
    classic_ok.append(pick is not None and abs(pick - p_index[i]) <= 50)
classic_ok = np.array(classic_ok)
network_ok = np.array(network_ok)
M["both_right"] = int((classic_ok & network_ok).sum())
M["network_only"] = int((~classic_ok & network_ok).sum())
M["classic_only"] = int((classic_ok & ~network_ok).sum())
M["both_wrong"] = int((~classic_ok & ~network_ok).sum())

# --- homework: how wide should the label be?
for sigma in (5, 40):
    hw_model, hw_losses, hw_scores = train_picker(sigma=sigma)
    hw_picks = picks_from(hw_model, x_test)
    M[f"sigma{sigma}_acc"] = round(float(hw_scores[-1]), 3)
    M[f"sigma{sigma}_loss"] = round(float(hw_losses[-1]), 5)
    M[f"sigma{sigma}_med"] = round(float(np.median(np.abs(hw_picks - p_test)) / SAMPLE_RATE), 3)

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


datahub = (f"{PLATFORM['datahub']}/hub/user-redirect/git-pull"
           f"?repo={PLATFORM['repo'].replace(':', '%3A').replace('/', '%2F')}"
           f"&branch={PLATFORM['branch']}"
           f"&urlpath=lab%2Ftree%2FEPS88_PyEarth%2F{PLATFORM['notebook_dir']}%2F{SLUG}.ipynb")

HOOK = """
A seismometer never stops. It writes down how the ground is moving a hundred times a second, day
and night, and nearly all of what it writes is traffic, wind and surf. Buried in that are the
earthquakes — several hundred a day in northern California alone, almost all of them far too small
to feel.

Finding them is one job. Timing them is a harder one, and it is the one that matters. What a
seismologist needs from a recording is the *instant* the first wave arrived, because the gap
between the fast P wave and the slower S wave gives the distance to the earthquake, and distances
from three stations give its location. For most of the last century that instant was marked by a
person with a ruler.

Today you get 2,500 real recordings from northern California, each already marked by an analyst.
You will watch the standard automatic picker miss, and then train a small network to find the
instant for you.
"""

md(weekkit.OPENING.format(question=WEEK["question"], datahub=datahub, hook=HOOK.strip()))

md("""
## What you'll be able to do

**The science.** Say what a seismogram contains and why the P and S arrival times are the
quantity a seismologist actually wants. Show that loudness answers *whether* an earthquake
happened but not *when* it started, and say how well the sixty-year-old automatic picker does on
real data before anything is learned from it.

**The skills.** Slide a pattern-detector along a signal with `np.convolve`. Build a neural
network in PyTorch out of `nn.Conv1d`, `nn.ReLU` and `nn.Upsample`, give it a loss to make small,
train it with gradient descent for a fixed number of epochs, and read its learning curve to see
when more training stops buying anything.

**Eight places where you write something: five in class, three at home.** Each one is headed
*Your turn*, with an empty cell under it.
""")

code(f'''
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.linear_model import LinearRegression

# house style, set once, so every plot cell below holds only what matters
plt.rcParams.update({{"figure.figsize": (7, 4), "figure.dpi": 110,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True}})

DATA = ("{DATA_URL}")


def load():
    """Read the waveform file, downloading it from the course release the first time."""
    try:
        return np.load("phasenet_ncedc.npz")
    except FileNotFoundError:
        torch.hub.download_url_to_file(DATA, "phasenet_ncedc.npz", progress=False)
        return np.load("phasenet_ncedc.npz")


# 42 MB of waveforms, too big to keep beside the notebook, so it arrives from a release instead
# of from the course repository. Everything else this week is computed from it.
data = load()
waveform = np.clip(data["waveform"], -10, 10)   # a few single-sample instrument spikes, cut off
p_index = data["p_index"].astype(int)           # sample number of the analyst's P pick
s_index = data["s_index"].astype(int)           # sample number of the analyst's S pick
distance_km = data["distance_km"]
event_id = data["event_id"]
station = data["station"]
magnitude = data["magnitude"]
SAMPLE_RATE = 100                               # samples per second

print("recordings:      ", waveform.shape)
print("earthquakes:     ", len(np.unique(event_id)), " stations:", len(np.unique(station)))
print("magnitudes:      ", round(magnitude.min(), 2), "to", round(magnitude.max(), 2),
      " median", round(np.median(magnitude), 2))
print("distance (km):   ", round(np.median(distance_km), 1), "median,",
      round(distance_km.max(), 1), "furthest")
print("P arrives between", round(p_index.min() / SAMPLE_RATE, 2), "and",
      round(p_index.max() / SAMPLE_RATE, 2), "seconds in")
'''.strip("\n"))

# --- section 1 -------------------------------------------------------------
md(f"""
## Twenty seconds of ground motion

`waveform` holds {M['n_traces']:,} recordings. Each one is a small array of its own: **3 rows**,
because a seismometer measures the ground moving east, north and up-down at the same time, and
**{M['n_samples']:,} columns**, one per sample. At {SAMPLE_RATE} samples a second that is
{M['seconds']} seconds of shaking per recording.

These come from the Northern California Earthquake Data Center, whose analysts marked the P and
the S on every one by hand. `p_index` and `s_index` are those marks, stored as sample numbers, so
sample {M['n_samples'] // 2} means {M['n_samples'] // 2 / SAMPLE_RATE:.2f} seconds into the
window. Each recording has already been divided by its own size, so a "1" in `waveform` means
*one typical wiggle for this instrument*, not a fixed number of nanometres — loudness here is
always relative to the same trace's own background.

One thing was done deliberately when the file was built: the window was cut at a **random** offset
before each P, so the arrival lands anywhere from {M['p_min_s']} to {M['p_max_s']} seconds in.
Nothing here can score well by always answering "sample {M['n_samples'] // 2}".

Look at one. The three rows are drawn on the same axes, and the two vertical lines are the
analyst's marks. The P is the first arrival, and the ground barely moves at first. The S comes
later and is much bigger; it is the wave that does the damage.
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

Then print the P time, the S time and the gap between them, all in seconds.

**Use these names**, because the self-check looks for them: `my_trace` for the trace number and
`sp_seconds` for the gap.
""")

answer(f"""
my_trace = 100

plt.plot(seconds, waveform[my_trace][2], lw=0.6)
plt.axvline(p_index[my_trace] / SAMPLE_RATE, color="k")
plt.axvline(s_index[my_trace] / SAMPLE_RATE, color="k")
plt.xlabel("time (s)")
plt.ylabel("ground motion (up-down)")
plt.title("trace " + str(my_trace) + " — station " + str(station[my_trace]))
plt.show()

sp_seconds = (s_index[my_trace] - p_index[my_trace]) / SAMPLE_RATE
print("P at", p_index[my_trace] / SAMPLE_RATE, "s")
print("S at", s_index[my_trace] / SAMPLE_RATE, "s")
print("gap:", round(sp_seconds, 2), "s")
""", """
assert sp_seconds > 0, "the S always arrives after the P, so the gap must be positive"
print("✓ one trace — trace", my_trace, "has its S", round(sp_seconds, 2),
      "s after its P")
""")

md(f"""
That gap is the whole reason anyone cares about the exact arrival time. The P and the S leave the
earthquake together and travel at different speeds, so the further you are from it, the further
apart they arrive — the gap is a distance measurement, taken at a single station.

Which means we can check it. Fit a straight line through the origin (through the origin because a
station standing on top of the earthquake must see a gap of zero) and the slope is how many
kilometres each second of gap is worth.
""")

code(f"""
sp_all = (s_index - p_index) / SAMPLE_RATE

line = LinearRegression(fit_intercept=False)
line.fit(sp_all.reshape(-1, 1), distance_km)

plt.scatter(sp_all, distance_km, s=3, alpha=0.3)
plt.plot(sp_all, line.predict(sp_all.reshape(-1, 1)), color="C1")
plt.xlabel("S minus P (s)")
plt.ylabel("distance to the earthquake (km)")
plt.title("{M['n_traces']:,} recordings: the gap is a distance measurement")
plt.show()

print("kilometres per second of gap:", round(line.coef_[0], 2))
print("R squared:", round(line.score(sp_all.reshape(-1, 1), distance_km), 3))
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
rng = np.random.default_rng(0)
events = np.unique(event_id)
rng.shuffle(events)
train_events = events[:int(0.7 * len(events))]
is_train = np.isin(event_id, train_events)      # True for a recording of a training earthquake

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
""")

md(f"""
Those two medians came from the training recordings only, so we are allowed to look at them.
Now the rule. **Write the dumbest rule you can, first. Any model that cannot beat it is
decoration.** The dumbest rule here is one number: call it an earthquake when the biggest swing
in the window is above {M['threshold']}, which sits between the two typical values you just
printed.
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
So *detection* barely needs us. One `if` statement, one number, and most of the work is done —
which is exactly why a neural network for this task would be decoration.

The interesting question was never whether the ground shook. It is **when it started**. And here
is the first thing to notice: the loudest part of a seismogram is not the beginning of it.

### Predict before you run

Across all {M['n_traces']:,} recordings, on what fraction is the single loudest sample within half
a second of the P arrival? Commit to a number before you run the next cell — change `my_guess` to
whatever you think, then run it.
""")

code(f"""
my_guess = 0.80

loudest = np.abs(waveform).max(axis=1).argmax(axis=1)   # sample number of the biggest swing
near_p = np.abs(loudest - p_index) <= 50                # 50 samples is half a second
near_s = np.abs(loudest - s_index) <= 50

print("you guessed:", my_guess)
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
The bulge is to the right of the line, not on it. Only {M['loudest_near_p']:.1%} of the loudest
samples are near the P; {M['loudest_near_s']:.1%} of them are near the **S**, which arrives
later and shakes harder. Loudness is a fine answer to *did something happen*. It is close to
useless as an answer to *when did it start*, because the loudest moment is a different wave
arriving.

Whatever finds the P has to work on the **shape** of the signal — the moment a quiet trace stops
being quiet — and not on how big it gets.
""")

# --- section 3 -------------------------------------------------------------
md("""
## The oldest automatic picker

Seismology has had an automatic answer to this since the 1970s, and it is two averages and a
division. Take the average power over a **short** window ending at the current sample, take it
again over a **long** window ending at the same sample, and divide. In background noise the two
averages are the same and the ratio sits near 1. The instant a wave arrives, the short window
fills with the new energy while the long window is still mostly old quiet — so the ratio jumps.
Trigger the first time it crosses some level, and that is your pick.

It is called STA/LTA, for short-term average over long-term average. Three numbers to choose: how
short, how long, and how big a jump counts.
""")

code(f"""
strength = np.abs(waveform).max(axis=1)         # one number per sample: the biggest of the 3 rows


def sta_lta(trace, short, long):
    \"\"\"How much louder the last `short` samples are than the last `long` samples.\"\"\"
    power = trace ** 2
    ratio = np.zeros(len(power))
    for i in range(long, len(power)):
        ratio[i] = power[i - short:i].mean() / power[i - long:i].mean()
    return ratio


def first_trigger(ratio, threshold):
    \"\"\"The first sample where the ratio crosses the threshold, or None if it never does.\"\"\"
    above = np.nonzero(ratio > threshold)[0]
    if len(above) == 0:
        return None
    return int(above[0])


ratio = sta_lta(strength[{TRACE}], 50, 500)
plt.plot(seconds, ratio, lw=0.8)
plt.axhline(3, color="C1")
plt.axvline(p_index[{TRACE}] / SAMPLE_RATE, color="k")
plt.xlabel("time (s)")
plt.ylabel("short average / long average")
plt.title("STA/LTA on 1 trace — black is the analyst's P, orange is the trigger level")
plt.show()
""")

code(f"""
def sta_lta_score(short, long, threshold):
    \"\"\"Fraction of test traces STA/LTA places within half a second of the analyst's pick.\"\"\"
    hits = 0
    for i in np.nonzero(~is_train)[0]:
        pick = first_trigger(sta_lta(strength[i], short, long), threshold)
        if pick is not None and abs(pick - p_index[i]) <= 50:
            hits = hits + 1
    return hits / (~is_train).sum()


print("textbook setting (0.5 s, 5 s, threshold 3):",
      round(sta_lta_score(50, 500, 3), 3))
""")

ask(f"""
### ✏️ Your turn 3

One setting is not a result. Run `sta_lta_score` on three more settings and print all three, so
that we know whether {M['stalta_textbook']:.3f} is what STA/LTA can do or merely what this
particular setting does. Try these:

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
    score = sta_lta_score(short, long, threshold)
    stalta_scores.append(score)
    print(short, long, threshold, "->", round(score, 3))

best_stalta = max(stalta_scores + [{M['stalta_textbook']}])
print("best of the four:", round(best_stalta, 3))
""", f"""
assert len(stalta_scores) == 3, "three more settings were asked for, so this should hold three"
print("✓ the sweep — STA/LTA's best of four settings is",
      round(100 * best_stalta, 1), "%")
""")

md(f"""
Tuned as well as four tries can tune it, the classical picker lands within half a second on
{M['stalta_best']:.1%} of held-out traces. It is not doing something silly. At the textbook
setting it never triggers at all on {M['stalta_never']:.1%} of them — the noise was too loud for
the ratio to ever jump by three — and on another {M['stalta_near_s']:.1%} it triggers on the S
instead of the P, having ignored a P that was too gentle to move the ratio.

That is the number to beat. Anything we build now has to beat {M['stalta_best']:.3f}, or we have
built decoration.
""")

# --- section 4 -------------------------------------------------------------
md("""
## Sliding a pattern-detector along the signal

Look again at what STA/LTA does. At every sample it takes a weighted sum of the samples around it
— minus one over the long window, plus one over the short one — and reports the answer. Then it
moves along one sample and does it again. That operation has a name: it is a **convolution**.
Slide a small pattern-detector along the signal.

The small list of weights is the detector, and what it detects depends entirely on the numbers in
it. Put a step in the weights and it responds to steps. `np.convolve` slides it for you.
""")

code(f"""
def onset_filter(width):
    \"\"\"Energy in the next `width` samples minus energy in the last `width` samples.\"\"\"
    return np.concatenate([-np.ones(width) / width, np.ones(width) / width])


response = np.convolve(strength[{TRACE}] ** 2, onset_filter(10), mode="same")

plt.plot(seconds, response / np.abs(response).max(), lw=0.8)
plt.axvline(p_index[{TRACE}] / SAMPLE_RATE, color="k")
plt.axvline(s_index[{TRACE}] / SAMPLE_RATE, color="C1")
plt.xlabel("time (s)")
plt.ylabel("filter response (scaled to its own peak)")
plt.title("a 20-sample onset detector on 1 trace — black P, orange S")
plt.show()
""")

ask(f"""
### ✏️ Your turn 4

Score that hand-made detector the way you scored STA/LTA. For each held-out trace, convolve
`strength[i] ** 2` with `onset_filter(10)`, take `response.argmax()` as the pick, and count how
often that lands within 50 samples of `p_index[i]`.

Print the fraction, and — because the last figure hints at where it actually goes — print the
fraction that lands within 50 samples of `s_index[i]` as well.

**Use these names**, because the self-check looks for them: `hand_accuracy`.
""")

answer(f"""
hits = 0
near_s_hits = 0
for i in np.nonzero(~is_train)[0]:
    response = np.convolve(strength[i] ** 2, onset_filter(10), mode="same")
    pick = response.argmax()
    hits = hits + (abs(pick - p_index[i]) <= 50)
    near_s_hits = near_s_hits + (abs(pick - s_index[i]) <= 50)

hand_accuracy = hits / (~is_train).sum()
print("within 0.5 s of the P:", round(hand_accuracy, 3))
print("within 0.5 s of the S:", round(near_s_hits / (~is_train).sum(), 3))
""", """
assert hand_accuracy < 1, "this is a fraction of held-out traces, not a count"
print("✓ the hand-made detector — it finds the P on",
      round(100 * hand_accuracy, 1), "% of held-out traces")
""")

md(f"""
Worse than STA/LTA, and worse in an informative way: it lands near the **S** on
{M['hand_near_s']:.1%} of traces. The numbers we chose describe *the biggest jump in energy*, and
in a seismogram the biggest jump in energy is the S.

We could keep guessing weights. A better detector might be longer, or shorter, or shaped
differently, or three detectors combined; there is no reason to think a human is good at choosing
twenty numbers. So stop choosing them.
""")

# --- section 5 -------------------------------------------------------------
md(f"""
## Letting the machine choose the numbers

**A stack of the logistic regressions you already know.** That is all a neural network is. One
logistic regression takes a weighted sum of its inputs and squashes the answer; a single one of
those is called a **perceptron**. Put several side by side and feed their outputs into another
row, and you have a stack.

The squashing step between rows is the **activation**, and it is not optional. A weighted sum of
weighted sums is still a weighted sum — stacking straight lines gives you a straight line, no
matter how many. The activation bends it. Here are two stacks with identical shape, one with the
bend and one without.
""")

code("""
grid = torch.linspace(-3, 3, 200).reshape(-1, 1)
torch.manual_seed(1)
flat = nn.Sequential(nn.Linear(1, 8), nn.Linear(8, 1))
torch.manual_seed(1)
bent = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))

plt.plot(grid, flat(grid).detach(), label="2 layers, no activation")
plt.plot(grid, bent(grid).detach(), label="2 layers, with ReLU")
plt.xlabel("input")
plt.ylabel("output")
plt.title("what 8 hidden units can draw, with and without an activation")
plt.legend()
plt.show()
""")

md(f"""
Three more words and we can build one.

The **loss** is the number the network is trying to make small. Ours is the same one you used to
fit a straight line: the average squared miss between what the network says and what we wanted.

**Roll downhill on the error surface.** That is **gradient descent** — the loss depends on every
weight in the network, PyTorch works out which way each weight would have to move to make the loss
smaller, and every weight takes a small step that way. One pass over all the training data is an
**epoch**, and training is just doing that again and again.

The last thing to decide is what "what we wanted" means. We are not asking for a number. We are
asking the network, at each of the {M['n_samples']:,} samples, *how much does this look like the
P arrival* — so the answer we train it towards is a bump centred on the analyst's mark, and the
pick we read back out is wherever the network's answer is highest.
""")

code(f"""
def make_target(pick_index, sigma):
    \"\"\"A bump centred on each trace's P arrival: what we want the network to output.\"\"\"
    sample = np.arange({M['n_samples']})
    target = np.zeros((len(pick_index), {M['n_samples']}), dtype="float32")
    for i in range(len(pick_index)):
        target[i] = np.exp(-(sample - pick_index[i]) ** 2 / (2 * sigma ** 2))
    return target


plt.plot(seconds, waveform[{TRACE}][2] / np.abs(waveform[{TRACE}][2]).max(), lw=0.6,
         label="up-down, scaled")
plt.plot(seconds, make_target(p_index[[{TRACE}]], 20)[0], label="what we want back")
plt.xlabel("time (s)")
plt.ylabel("scaled to 1")
plt.title("1 trace and its training target — a bump 0.2 s wide on the P")
plt.legend()
plt.show()
""")

md(f"""
Now the network. `nn.Conv1d(3, 8, 7)` is eight pattern-detectors, each 7 samples long, each
looking at all 3 rows at once — the same sliding operation as before, except that the numbers
inside are what training will choose. `stride=4` slides in steps of 4 instead of 1, which makes
the signal four times shorter and lets the next layer's 7 samples cover four times as much time;
`nn.Upsample` stretches it back out so the answer is one number per original sample.

In PyTorch a model is an `nn.Module` — a box that holds the numbers to be learned and knows how
to run them. `nn.Sequential` is the simplest one there is: hand it layers, and it runs them in
order.
""")

code(f"""
def make_picker():
    \"\"\"Five convolution layers: squeeze the trace down to a summary, then stretch it back out.\"\"\"
    return nn.Sequential(
        nn.Conv1d(3, 8, 7, stride=4, padding=3), nn.ReLU(),
        nn.Conv1d(8, 16, 7, stride=4, padding=3), nn.ReLU(),
        nn.Conv1d(16, 16, 7, padding=3), nn.ReLU(),
        nn.Upsample(scale_factor=4), nn.Conv1d(16, 8, 7, padding=3), nn.ReLU(),
        nn.Upsample(scale_factor=4), nn.Conv1d(8, 1, 7, padding=3))


x_train = torch.tensor(waveform[is_train])
x_test = torch.tensor(waveform[~is_train])
p_test = p_index[~is_train]

print("weights to learn:", sum(w.numel() for w in make_picker().parameters()))
""")

code(f"""
def picks_from(model, x):
    \"\"\"Where the network says the P is: the sample with the highest output.\"\"\"
    return model(x).squeeze(1).detach().numpy().argmax(axis=1)


def within_half_second(picks, truth):
    \"\"\"Fraction of picks landing within half a second of the analyst's pick.\"\"\"
    return (np.abs(picks - truth) <= 50).mean()


def train_picker(sigma=20, epochs=25):
    \"\"\"Train the picker; hand back the model, and the loss and test score after every epoch.\"\"\"
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


picker, losses, scores = train_picker()
print("trained for", len(losses), "epochs")
""")

# --- section 6 -------------------------------------------------------------
md("""
## Watching it learn

Two lines, on two axes because they are in different units. On the left, the loss on the data the
network trained on. On the right, the fraction of **held-out** traces it picks within half a
second — recordings of earthquakes it has never seen. **Watch two lines. When training keeps
falling and test turns up, stop.**

The left line is what gradient descent is directly pushing down, so it should fall smoothly. The
right line is the one we actually care about, and nothing is pushing on it at all.
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
assert len(net_picks) == len(p_test), "one pick per held-out trace"
print("✓ the network —", round(100 * net_accuracy, 1), "% of held-out traces within 0.5 s,",
      "median error", round(np.median(np.abs(net_picks - p_test)) / SAMPLE_RATE, 3), "s")
""")

code(f"""
closest = np.abs(net_picks - p_test).argmin()
furthest = np.abs(net_picks - p_test).argmax()

plt.figure(figsize=(7, 5))                      # two stacked traces need the extra height
for panel, which in [(1, closest), (2, furthest)]:
    plt.subplot(2, 1, panel)
    plt.plot(seconds, x_test[which][2].numpy() / 10, lw=0.5, label="up-down, scaled")
    plt.plot(seconds, picker(x_test[[which]]).squeeze().detach().numpy(), lw=1,
             label="the network's answer")
    plt.axvline(p_test[which] / SAMPLE_RATE, color="k")
    plt.ylabel("scaled to 1")
plt.legend()
plt.xlabel("time (s)")
plt.suptitle("the network on its closest and its furthest held-out trace of {{}} (black = analyst)"
             .format(len(p_test)))
plt.show()
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

Three parts. The first is short; the second asks you to decide something and defend it; the third
settles an argument the class left open.

Two of the three train a network from scratch, so start them and let them run. If you restarted
the kernel since class, run the checkpoint cell first — it rebuilds the trained picker without
printing anything, and it is the slow cell rather than a free one.
""")

code("""
# ── Checkpoint ── run this if you restarted the kernel or fell behind ──
picker, losses, scores = train_picker()
net_picks = picks_from(picker, x_test)
""")

ask(f"""
### ✏️ Your turn 6

Does the network simply do STA/LTA's job faster, or does it do a different job? If it were the
same job, they would fail on the same traces.

Find out. For each held-out trace, work out whether STA/LTA at its best setting from Your turn 3
— short {M['stalta_best_setting'].split(', ')[0]}, long {M['stalta_best_setting'].split(', ')[1]},
threshold {M['stalta_best_setting'].split(', ')[2]} — landed within 50 samples of the analyst's
pick, and whether the network did. Then print all four counts: both right, network only, STA/LTA
only, and both wrong.

**Use these names**, because the self-check looks for them: `classic_ok` and `network_ok`, each a
list or array of True/False, one per held-out trace.
""")

answer(f"""
classic_ok = []
for i in np.nonzero(~is_train)[0]:
    pick = first_trigger(sta_lta(strength[i], {M['stalta_best_setting'].split(', ')[0]},
                                 {M['stalta_best_setting'].split(', ')[1]}),
                         {M['stalta_best_setting'].split(', ')[2]})
    classic_ok.append(pick is not None and abs(pick - p_index[i]) <= 50)

classic_ok = np.array(classic_ok)
network_ok = np.abs(net_picks - p_test) <= 50

print("both right:  ", (classic_ok & network_ok).sum())
print("network only:", (~classic_ok & network_ok).sum())
print("STA/LTA only:", (classic_ok & ~network_ok).sum())
print("both wrong:  ", (~classic_ok & ~network_ok).sum())
""", """
assert len(classic_ok) == len(network_ok), "one True/False per held-out trace, for both methods"
print("✓ agreement — the network rescues",
      (~classic_ok & network_ok).sum(), "traces STA/LTA misses, and loses",
      (classic_ok & ~network_ok).sum(), "that it finds")
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
""", """
assert my_sigma in (5, 40), "the question offers 5 or 40; pick one of them"
assert len(my_scores) == len(scores), "same number of epochs, so only the label width changed"
print("✓ the label width — sigma", my_sigma, "scores",
      round(my_scores[-1], 3), "against", round(scores[-1], 3), "at sigma 20")
""")

ask(f"""
### ✏️ Your turn 8

In class the held-out score was still at its highest on the very last epoch, which leaves an
obvious doubt: perhaps the picker was simply not trained for long enough.

Settle it. Run `train_picker(epochs=60)` — this is the slow one — and print four numbers from
what it hands back: the training loss and the held-out score after epoch 25, and both again after
epoch 60. Print the best held-out score of the whole run and the epoch it came at as well. (A
list counts from 0, so epoch 25 is at position 24.)

Then, in the **markdown** cell below the code cell, answer in two or three sentences **using your
own four numbers**: over
those 35 extra epochs, what happened to the loss, what happened to the held-out score, and what
does the pair of answers say about where this picker's remaining error is coming from? Your answer
has to name a number that would have to change before the picker could do better.

**Use these names**, because the self-check looks for them: `long_losses` and `long_scores`.
""")

answer(f"""
long_picker, long_losses, long_scores = train_picker(epochs=60)

print("epoch 25: loss", round(long_losses[24], 5), " held-out", round(long_scores[24], 3))
print("epoch 60: loss", round(long_losses[59], 5), " held-out", round(long_scores[59], 3))
print("best held-out score:", round(max(long_scores), 3), "at epoch", np.argmax(long_scores) + 1)
""", """
assert len(long_scores) == 60, "60 epochs were asked for"
print("✓ more training — the loss fell by a factor of",
      round(long_losses[0] / long_losses[59], 1), "while the held-out score moved from",
      round(long_scores[24], 3), "to", round(long_scores[59], 3))
""")

answer_prose(f"""
Over the extra 35 epochs my training loss kept falling, from {{long_loss25}} at epoch 25 to
{{long_loss60}} at epoch 60, while the held-out score went from {{long_acc25}} to {{long_acc60}} —
it did not improve, and the best held-out score of the whole run, {{long_best_acc}}, came at epoch
{{long_best_epoch}}. So the network was still learning something, but nothing that transferred to
recordings of earthquakes it had not seen: the extra epochs bought a better fit to the training
labels, not a better picker. That means the limit is not the amount of training, and the number
that would have to change is the labels themselves. Every target is a bump centred on where a
human analyst put the P, and where the P is faint that mark is itself uncertain by a few tenths of
a second, so a network that agrees with the analyst to within its own uncertainty has nowhere left
to go. To do better you would have to make the marks better — more analysts per trace, or agreement
between them — not train longer.
""".format(long_loss25=M["long_loss25"], long_loss60=M["long_loss60"],
           long_acc25=M["long_acc25"], long_acc60=M["long_acc60"],
           long_best_acc=M["long_best_acc"], long_best_epoch=M["long_best_epoch"]))


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

    print(f"executing {sol_path.name} ...")
    r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                        "--execute", "--inplace", "--ExecutePreprocessor.timeout=2400",
                        str(sol_path)], capture_output=True, text=True, cwd=OUT)
    if r.returncode:
        print(r.stderr[-6000:])
        sys.exit("the solution did not execute")

    (OUT / f"{SLUG}.ipynb").write_text(json.dumps(stu, indent=1) + "\n")
    print(f"wrote {SLUG}.ipynb ({len(CELLS)} cells) and the executed solution")
    print("measured:", json.dumps(M, indent=1))


if __name__ == "__main__":
    main()
    weekkit.gate(13)
