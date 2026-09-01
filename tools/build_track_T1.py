#!/usr/bin/env python
"""Build project track T1 — "How hard will the ground shake at distance R from a magnitude M
earthquake?"

Emits both notebooks from one source so they cannot drift:

    docs/notebooks/T1_how_hard_will_it_shake_solution.ipynb   executed, every output saved
    docs/notebooks/T1_how_hard_will_it_shake.ipynb            the same file with the answers deleted

It also writes the track's cached fallback, data/trackT1_esm_flatfile.csv.gz.

A TRACK is not a week (course.yml `project: track_notebooks:`). Two things differ, and both are
deliberate:

  * LESS HELP. No worked example before a question. The notebook loads the data and reproduces
    the ONE result the field already agrees on — the classic three-parameter ground-motion
    equation, its coefficients and its sigma — so a student can trust the pipeline, and then
    stops helping. Everything after is a prompt in words and an empty cell.
  * IT DOES NOT CLOSE. There is exactly one self-check, on the load, and the notebook ends on an
    open question this course cannot answer.

Every number that appears in prose or in a model answer is computed HERE, by the same expressions
the notebook runs, and formatted in. Nothing is typed from memory or copied from the plan.

    python tools/build_track_T1.py

Needs torch, which the shared base environment does not carry; run it with an interpreter that
has torch, numpy, pandas, matplotlib, sklearn, pyyaml and nbconvert.
"""
import json
import os
import pathlib
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import weekkit

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/notebooks"
SLUG = "T1_how_hard_will_it_shake"

course = yaml.safe_load((ROOT / "course.yml").read_text())
modules = yaml.safe_load((ROOT / "modules.yml").read_text())
TRACK = next(t for t in course["project"]["tracks"] if t["id"] == "T1")
PLATFORM = course["platform"]
CACHE_BASE = PLATFORM["cache_base"]

# The live source. Pinned here so the cached file, the notebook and the prose below cannot drift.
# ONE request, not the fifty-one the dataset audit used: the audit pulled a year at a time because
# each request costs about the same regardless of size, but a student cannot sit through fifty-one
# of them. Raising the magnitude floor from the audit's 4 to 4.5 makes the whole 1970-2026 span
# fit in a single query that returns in under a minute and caches to under 2 MB.
FIELD_LINES = ["esm_event_id,event_time,ev_latitude,ev_longitude,ev_depth_km,ev_nation_code,",
               "fm_type_code,ml,mw,network_code,station_code,st_latitude,st_longitude,",
               "st_nation_code,preferred_vs30_m_s,preferred_ec8_code,epi_dist,jb_dist,",
               "quality_class,late_triggered_event_01,rotd50_pga,rotd50_pgv"]
FIELDS = "".join(FIELD_LINES)
MIN_MAGNITUDE = 4.5
STARTTIME, ENDTIME = "1970-01-01", "2026-01-01"
ESM = (f"https://esm-db.eu/esmws/flatfile/1/query?min-magnitude={MIN_MAGNITUDE}"
       f"&starttime={STARTTIME}&endtime={ENDTIME}&include-fields={FIELDS}")
ESM_CACHE = "trackT1_esm_flatfile.csv.gz"

MAG_FLOOR = 4.0              # the magnitude the fitted model is honest for
R_MAX = 200                  # km; beyond this a European flatfile is mostly triggered noise
PSEUDO_DEPTH = 8             # km, the h in sqrt(R^2 + h^2); the audit's value, refitted below
G_CM_S2 = 981.0              # rotd50_pga arrives in cm/s^2
SEED = 88                    # the course number, fixed before anything was run
CI_Z = 1.96                  # the multiplier that really is "nineteen times in twenty"
N_BOOT = 2000                # station-block bootstrap resamples
EPOCHS = 200                 # passes over the training data for the small network
BATCH = 512
WIDTH = 32
MAP_BOX = (-12, 48, 30, 50)  # lon lo, lon hi, lat lo, lat hi — where 96% of the records are


# ---------------------------------------------------------------------------
# 1. measure everything the notebook will say
# ---------------------------------------------------------------------------
def fetch(url, name):
    """Run the live query once, cache it beside the course, and return the cached copy.

    Written with sep=";" on BOTH sides — the service is semicolon-delimited, so the cache is too,
    and the notebook's one `load()` reads either with the same call.
    """
    out = ROOT / "data" / name
    if not out.exists():
        pd.read_csv(url, sep=";").to_csv(out, index=False, sep=";")
    return pd.read_csv(out, sep=";")


records = fetch(ESM, ESM_CACHE)

# --- the cleaning, exactly as the notebook writes it ---
records["mag"] = records["mw"].fillna(records["ml"])
records["R"] = records["jb_dist"].fillna(records["epi_dist"])
records["pga_g"] = records["rotd50_pga"] / G_CM_S2
records["vs30"] = records["preferred_vs30_m_s"]
records["station"] = records["network_code"] + "." + records["station_code"]

usable = records[(records["pga_g"] > 0)
                 & records["mag"].notna()
                 & (records["quality_class"] != "BAD")
                 & (records["late_triggered_event_01"].fillna(0) == 0)]

shaking = usable[(usable["mag"] >= MAG_FLOOR)
                 & (usable["R"] <= R_MAX)
                 & usable["vs30"].notna()].reset_index(drop=True).copy()
shaking["log_pga"] = np.log10(shaking["pga_g"])
shaking["log_r"] = np.log10(np.sqrt(shaking["R"] ** 2 + PSEUDO_DEPTH ** 2))
shaking["log_vs30"] = np.log10(shaking["vs30"])

FEATURES = ["mag", "log_r", "log_vs30"]
X = shaking[FEATURES]
y = shaking["log_pga"]

M = {}
M["n_raw"] = len(records)
M["n_cols"] = records.shape[1]
M["n_usable"] = len(usable)
M["n_work"] = len(shaking)
M["events"] = int(shaking["esm_event_id"].nunique())
M["stations"] = int(shaking["station"].nunique())
M["nations"] = int(shaking["st_nation_code"].nunique())
M["usable_events"] = int(usable["esm_event_id"].nunique())
M["usable_stations"] = int(usable["station"].nunique())
M["year_lo"] = str(shaking["event_time"].str[:4].min())
M["year_hi"] = str(shaking["event_time"].str[:4].max())
M["jb_missing_raw"] = float(records["jb_dist"].isna().mean())
M["vs30_present"] = float(usable["vs30"].notna().mean())
M["mag_lo"], M["mag_hi"] = float(shaking["mag"].min()), float(shaking["mag"].max())
M["r_max"] = float(shaking["R"].max())
M["pga_max"] = float(shaking["pga_g"].max())
M["vs30_lo"], M["vs30_hi"] = float(shaking["vs30"].min()), float(shaking["vs30"].max())
M["per_event_median"] = float(shaking["esm_event_id"].value_counts().median())
M["per_event_max"] = int(shaking["esm_event_id"].value_counts().max())
M["per_station_median"] = float(shaking["station"].value_counts().median())
M["per_station_max"] = int(shaking["station"].value_counts().max())
top_nations = shaking["st_nation_code"].value_counts()
M["top_nations"] = {str(k): int(v) for k, v in top_nations.head(4).items()}
M["italy_rows"] = int((shaking["st_nation_code"] == "IT").sum())
M["greece_rows"] = int(top_nations.get("GR", 0))
inside = (shaking["st_longitude"].between(MAP_BOX[0], MAP_BOX[1])
          & shaking["st_latitude"].between(MAP_BOX[2], MAP_BOX[3]))
M["in_box"] = int(inside.sum())
M["out_box"] = int((~inside).sum())

# --- the known result: the classic three-parameter form ---
gmpe = LinearRegression().fit(X, y)
residual = y - gmpe.predict(X)
M["b0"] = float(gmpe.intercept_)
M["bM"], M["bR"], M["bV"] = [float(c) for c in gmpe.coef_]
M["r2_in_sample"] = float(gmpe.score(X, y))
M["sigma"] = float(residual.std())
M["spread"] = float(y.std())
M["factor_mean"] = 10 ** M["spread"]
M["factor_model"] = 10 ** M["sigma"]
M["factor_bought"] = M["factor_mean"] / M["factor_model"]

# a concrete prediction, for the closing sentence
_demo_M, _demo_R, _demo_V = 6.0, 20, 500
_lp = (M["b0"] + M["bM"] * _demo_M
       + M["bR"] * np.log10(np.sqrt(_demo_R ** 2 + PSEUDO_DEPTH ** 2))
       + M["bV"] * np.log10(_demo_V))
M["demo_pga"] = float(10 ** _lp)
M["demo_lo"] = float(10 ** (_lp - CI_Z * M["sigma"]))
M["demo_hi"] = float(10 ** (_lp + CI_Z * M["sigma"]))
M["demo_factor"] = float(10 ** (CI_Z * M["sigma"]))
M["demo_span"] = M["demo_hi"] / M["demo_lo"]

# the pseudo-depth sweep, so the h in the formula is a measured choice and not a borrowed constant
M["h_sweep"] = {}
for h in (4, 6, 8, 10, 12):
    _X = pd.DataFrame({"mag": shaking["mag"],
                       "log_r": np.log10(np.sqrt(shaking["R"] ** 2 + h ** 2)),
                       "log_vs30": shaking["log_vs30"]})
    M["h_sweep"][h] = float(LinearRegression().fit(_X, y).score(_X, y))
M["h_spread"] = max(M["h_sweep"].values()) - min(M["h_sweep"].values())


# --- the fork: four ways to split, and the machinery the notebook hands over ---
def hold_out_quarter(labels, seed=SEED):
    """True for the rows to TRAIN on."""
    # 1. Count how many records each label owns, then shuffle that list. Without the shuffle the
    #    biggest earthquakes would be held out every time, which asks a different question.
    counts = pd.Series(labels).value_counts().sample(frac=1, random_state=seed)
    held_out = []
    rows_out = 0
    # 2. Move whole labels into the test set until a quarter of the RECORDS have gone. Counting
    #    records rather than labels keeps the test set the same size whichever column you split on.
    for name, n_rows in counts.items():
        if rows_out >= 0.25 * len(labels):
            break
        held_out.append(name)
        rows_out = rows_out + n_rows
    # 3. Every record carrying a held-out label leaves together. That is the point of splitting
    #    this way: if two records from one earthquake sat on opposite sides of the split, the
    #    model could half-remember the answer instead of having to predict it.
    return ~np.isin(labels, held_out)


def r_squared(predicted, actual):
    """The fraction of the up-and-down variation a prediction accounts for."""
    return 1 - ((actual - predicted) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()


def rmse(predicted, actual):
    """The typical miss, in the units of the thing being predicted."""
    return np.sqrt(((actual - predicted) ** 2).mean())


def held_out_predictions(model, is_train):
    """Fit the model on the training records; hand back what it predicts for the held-out ones.

    `model` is either a scikit-learn estimator — anything with `.fit` and `.predict` — or a plain
    function of `is_train` that does its own fitting and returns the held-out predictions. The
    second form is what a torch model needs, and taking both here is what lets every model in
    this notebook be scored by the same call.
    """
    # A scikit-learn model is fitted and then asked; a torch model brings its own training loop
    # and so arrives as a function. Asking whether the object has a `.fit` tells the two apart.
    if hasattr(model, "fit"):
        # The model is shown the training records only. It never sees `X[~is_train]` while it is
        # learning, which is what makes the score below a test rather than a memory check.
        model.fit(X[is_train], y[is_train])
        return model.predict(X[~is_train])
    return model(is_train)


def held_out_r2(model, is_train):
    """Fit on the training records, score on the held-out ones."""
    return r_squared(held_out_predictions(model, is_train), y[~is_train])


def held_out_rmse(model, is_train):
    """Fit on the training records, and how far off it typically is on the held-out ones."""
    return rmse(held_out_predictions(model, is_train), y[~is_train])


SPLITS = {"at random": hold_out_quarter(np.arange(len(shaking))),
          "by event": hold_out_quarter(shaking["esm_event_id"]),
          "by station": hold_out_quarter(shaking["station"]),
          "across a border": (shaking["st_nation_code"] != "IT").values}


def network_predictions(is_train, epochs=EPOCHS, seed=0):
    """Train the small network on the training records; predict the held-out ones."""
    # 1. Put the three features on the same scale. A network learns by nudging its weights, and
    #    an unscaled column would be nudged far harder than the others for no good reason. The
    #    scaler is fitted on the TRAINING records and only then applied to the held-out ones, so
    #    nothing the model is about to be tested on leaks into how it was trained.
    scaler = StandardScaler()
    x_train = torch.tensor(scaler.fit_transform(X[is_train]), dtype=torch.float32)
    x_test = torch.tensor(scaler.transform(X[~is_train]), dtype=torch.float32)
    y_train = torch.tensor(y[is_train].values, dtype=torch.float32).reshape(-1, 1)

    # 2. Fix the random numbers before the layers are built, so the network starts from the same
    #    weights every run and the numbers repeat.
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(len(FEATURES), WIDTH), nn.ReLU(),
                        nn.Linear(WIDTH, WIDTH), nn.ReLU(),
                        nn.Linear(WIDTH, 1))
    optimiser = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_function = nn.MSELoss()
    # 3. Train: `epochs` passes over the training records, and a fresh random order each pass so
    #    the network cannot pick anything up from the order the records happen to sit in.
    for epoch in range(epochs):
        order = torch.randperm(len(x_train))
        for start in range(0, len(x_train), BATCH):
            batch = order[start:start + BATCH]
            optimiser.zero_grad()
            loss = loss_function(net(x_train[batch]), y_train[batch])
            loss.backward()
            optimiser.step()
    # 4. `detach` drops the bookkeeping torch keeps in order to train, leaving plain numbers.
    return net(x_test).detach().numpy().ravel()


forest = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=0)

TABLE = {}
for name, is_train in SPLITS.items():
    actual = y[~is_train]
    share_event = float(shaking["esm_event_id"][~is_train]
                        .isin(shaking["esm_event_id"][is_train]).mean())
    share_station = float(shaking["station"][~is_train]
                          .isin(shaking["station"][is_train]).mean())
    line = held_out_r2(LinearRegression(), is_train)
    trees = held_out_r2(forest, is_train)
    net = held_out_r2(network_predictions, is_train)
    TABLE[name] = {
        "n_train": int(is_train.sum()), "n_test": int((~is_train).sum()),
        "share_event": share_event, "share_station": share_station,
        "mean": float(r_squared(y[is_train].mean(), actual)),
        "line": float(line), "forest": float(trees), "net": float(net),
        "forest_gap": float(trees - line), "net_gap": float(net - line),
        # R2 is MSE divided by the test set's OWN spread, so four R2 on four different test sets
        # are four different quantities. These three are what make them comparable again: the
        # spread each R2 was divided by, and the miss itself in the units of the thing predicted.
        "sd": float(actual.std()),
        "rmse_line": float(held_out_rmse(LinearRegression(), is_train)),
        "rmse_forest": float(held_out_rmse(forest, is_train)),
        "median_r": float(shaking["R"][~is_train].median()),
    }

M["importances"] = {f: float(v) for f, v in
                    zip(FEATURES, RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                                        random_state=0).fit(X, y).feature_importances_)}

# --- is the border reversal about the border? hold out each big country in turn ---------------
# The mechanism the plan records — the forest memorising site terms — is testable and fails: the
# station split ALREADY has zero station overlap and the forest is still ahead there. So the
# reversal has to be something the border does that the station split does not, and the way to
# find out is to move the border. Four countries, largest first.
COUNTRIES = {}
for _country in ("GR", "IT", "TR", "RO"):
    _is_train = (shaking["st_nation_code"] != _country).values
    _line = held_out_r2(LinearRegression(), _is_train)
    _trees = held_out_r2(forest, _is_train)
    # How far outside the training data's own range the held-out records sit, on the three
    # features the models can see — and the depth, which is NOT one of them and turns out to be
    # what separates the two hold-outs that reverse.
    _lo, _hi = X[_is_train].quantile(0.05), X[_is_train].quantile(0.95)
    _outside = ((X[~_is_train] < _lo) | (X[~_is_train] > _hi)).any(axis=1)
    COUNTRIES[_country] = {
        "n_test": int((~_is_train).sum()), "line": float(_line), "forest": float(_trees),
        "gap": float(_trees - _line), "median_r": float(shaking["R"][~_is_train].median()),
        "rmse_line": float(held_out_rmse(LinearRegression(), _is_train)),
        "rmse_forest": float(held_out_rmse(forest, _is_train)),
        "outside": float(_outside.mean()),
        "median_depth": float(shaking["ev_depth_km"][~_is_train].median()),
        "median_depth_train": float(shaking["ev_depth_km"][_is_train].median()),
    }

# Trained IN Italy and tested everywhere else: the other direction of the same border.
_reverse = (shaking["st_nation_code"] == "IT").values
REVERSE = {"n_train": int(_reverse.sum()), "n_test": int((~_reverse).sum()),
           "line": float(held_out_r2(LinearRegression(), _reverse)),
           "forest": float(held_out_r2(forest, _reverse))}
REVERSE["gap"] = REVERSE["forest"] - REVERSE["line"]

# --- what a piecewise-constant model cannot do: leave the region it was trained in -------------
# A forest predicts the average of training records that fell in the same leaf, so its output is
# bounded by the training targets and flattens at the edge of the training cloud. A line has no
# such bound. Measured on the border split, where the test set sits at the near-distance edge.
_border = SPLITS["across a border"]
_train_r, _test_r = shaking["log_r"][_border], shaking["log_r"][~_border]
_q05 = float(_train_r.quantile(0.05))
EDGE = {
    "median_r_test": float(shaking["R"][~_border].median()),
    "median_r_train": float(shaking["R"][_border].median()),
    "below_q05": float((_test_r < _q05).mean()),
    "actual_hi": float(y[~_border].max()),
    "line_hi": float(held_out_predictions(LinearRegression(), _border).max()),
    "forest_hi": float(held_out_predictions(forest, _border).max()),
    "sd_line": float(np.std(y[~_border].values - held_out_predictions(LinearRegression(), _border))),
    "sd_forest": float(np.std(y[~_border].values - held_out_predictions(forest, _border))),
}

# --- how big is the gap, next to the noise in the test set itself ---
BOOT = {}
for name in ("at random", "by station"):
    is_train = SPLITS[name]
    actual = y[~is_train].values
    predicted_line = held_out_predictions(LinearRegression(), is_train)
    predicted_forest = held_out_predictions(forest, is_train)
    held = shaking[~is_train].reset_index(drop=True)
    positions = []
    for station, rows in held.groupby("station"):
        positions.append(rows.index.values)
    rng = np.random.default_rng(SEED)
    gaps = []
    for i in range(N_BOOT):
        picked = rng.integers(0, len(positions), size=len(positions))
        parts = []
        for p in picked:
            parts.append(positions[p])
        take = np.concatenate(parts)
        gaps.append(r_squared(predicted_forest[take], actual[take])
                    - r_squared(predicted_line[take], actual[take]))
    gaps = np.array(gaps)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    BOOT[name] = {"n_blocks": len(positions), "observed": TABLE[name]["forest_gap"],
                  "lo": float(lo), "hi": float(hi),
                  "below_zero": float((gaps <= 0).mean())}

# --- sigma, and the two halves it is made of ---
shaking["residual"] = residual
event_mean = shaking.groupby("esm_event_id")["residual"].mean()
shaking["event_term"] = shaking["esm_event_id"].map(event_mean)
shaking["within_event"] = shaking["residual"] - shaking["event_term"]
M["tau"] = float(shaking["event_term"].std())
M["phi"] = float(shaking["within_event"].std())
M["tau2_phi2"] = float(M["tau"] ** 2 + M["phi"] ** 2)
M["sigma2"] = float(M["sigma"] ** 2)
M["naive_tau"] = float(event_mean.std())
M["sigma_ln"] = float(M["sigma"] * np.log(10))

# The build log is the record that every number was computed. Print all of it, not a selection.
for k in sorted(M):
    print(f"  measured  {k:>18} = {M[k]}")
for name in TABLE:
    print(f"  measured  {name:>18} : {TABLE[name]}")
for name in BOOT:
    print(f"  bootstrap {name:>18} : {BOOT[name]}")
for name in COUNTRIES:
    print(f"  hold out  {name:>18} : {COUNTRIES[name]}")
print(f"  measured  {'Italy -> elsewhere':>18} = {REVERSE}")
print(f"  measured  {'border edge':>18} = {EDGE}")

# --- what the plan records, against what the data gives -----------------------
# A builder does not edit the plan. It prints the mismatch so an orchestrator can.
PLAN_NOTES = []
if str(M["n_usable"]) not in TRACK["data"]:
    PLAN_NOTES.append(
        f"course.yml T1 `data:` quotes the audit's fifty-one-request pull at min-magnitude=4 — "
        f"'54,179 records, 4,869 events, 3,165 stations, 65 countries'. This notebook runs ONE "
        f"request at min-magnitude={MIN_MAGNITUDE} so a student can wait for it: "
        f"{M['n_raw']:,} rows arrive, {M['n_usable']:,} survive the quality filters "
        f"({M['usable_events']:,} events, {M['usable_stations']:,} stations), and "
        f"{M['n_work']:,} are usable for a ground-motion model.")
if "LOSES by 0.012 split by station" in " ".join(TRACK["open_question"].split()):
    PLAN_NOTES.append(
        f"course.yml T1 `open_question:` says the forest 'LOSES by 0.012 split by station'. "
        f"Measured here on one held-out quarter (the course excludes cross-validation) the "
        f"station split still leaves the forest AHEAD by "
        f"{TABLE['by station']['forest_gap']:+.3f}; the sign flips at the region holdout "
        f"({TABLE['across a border']['forest_gap']:+.3f}), not at the station split. The gap "
        f"under the station split is however inside its own bootstrap interval "
        f"[{BOOT['by station']['lo']:+.3f}, {BOOT['by station']['hi']:+.3f}], so 'the advantage "
        f"does not survive a station split' holds; 'it reverses' does not.")
if "memorising site terms" in " ".join(TRACK["open_question"].split()):
    PLAN_NOTES.append(
        f"course.yml T1 `open_question:` attributes the border reversal to 'the forest memorising "
        f"site terms'. That mechanism is refuted by the notebook's own table: the station split "
        f"has share-a-station {TABLE['by station']['share_station']:.3f} — no station in the test "
        f"set was trained on — and the forest is still AHEAD there by "
        f"{TABLE['by station']['forest_gap']:+.3f}. Nor is it the border as such: holding out "
        + ", ".join(f"{k} gives {v['gap']:+.3f}" for k, v in COUNTRIES.items())
        + f". The measured mechanism for Italy is extrapolation: the Italian records are nearer "
        f"({EDGE['median_r_test']:.0f} km median against {EDGE['median_r_train']:.0f} km outside "
        f"it, {EDGE['below_q05'] * 100:.0f}% of them below the training set's 5th percentile of "
        f"log distance) and a piecewise-constant forest cannot leave the region it was trained "
        f"in — its highest prediction there is {EDGE['forest_hi']:.2f} in log10 PGA against the "
        f"line's {EDGE['line_hi']:.2f} and an actual maximum of {EDGE['actual_hi']:.2f}. Romania "
        f"reverses by the same amount for a different reason and should not be folded in: its "
        f"earthquakes are intermediate-depth ({COUNTRIES['RO']['median_depth']:.0f} km median "
        f"against {COUNTRIES['RO']['median_depth_train']:.0f} km), depth is not a feature, and "
        f"the EQUATION's own R2 there is {COUNTRIES['RO']['line']:.3f} — both models fail and the "
        f"forest fails harder. The conclusion — that ML learns no transferable physics here — "
        f"survives; the mechanism is extrapolation, not memorisation.")
for note in PLAN_NOTES:
    print(f"  PLAN DRIFT  {note}")


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
# new, so the full module tables would list a hundred functions; these are the ones the notebook
# and its model answers actually write.
TRACK_IDEAS = [("ML1", "Linear regression"), ("D4", "Log axes"), ("ML3", "Baseline"),
               ("ML2", "Train/test split"), ("ML2", "Leakage"), ("ML4", "Random forest"),
               ("ML6", "Neural network"), ("S4", "Bootstrap"), ("S4", "Confidence interval")]
TRACK_FNS = [("S3", "np.log10(values)"), ("ML2", "np.sqrt(x) / np.mean(x)"),
             ("D2", "table.groupby(column)"), ("D2", "column.value_counts()"),
             ("ML1", "LinearRegression().fit(x, y)"), ("ML1", "model.coef_[0]"),
             ("ML1", "model.intercept_"), ("ML1", "model.predict(x)"),
             ("ML1", "model.score(x, y)"),
             ("ML2", "table.sample(frac=1, random_state=n)"),
             ("ML4", "StandardScaler()"),
             ("ML4", "filler.fit_transform(X_train) / filler.transform(X_test)"),
             ("ML4", "forest.feature_importances_"),
             ("ML4", "plt.bar(x, heights) / plt.barh(labels, values)"),
             ("ML6", "torch.tensor(array)"), ("ML6", "torch.manual_seed(n)"),
             ("ML6", "nn.Sequential(layers)"), ("ML6", "nn.Linear(in, out)"),
             ("ML6", "nn.ReLU()"), ("ML6", "nn.MSELoss()"),
             ("ML6", "torch.optim.Adam(model.parameters(), lr=)"),
             ("ML6", "loss.backward() / optimiser.step() / optimiser.zero_grad()"),
             ("ML6", "torch.randperm(n)"), ("ML6", "tensor.detach().numpy()"),
             ("S4", "np.percentile(values, [2.5, 97.5])")]

TITLE = TRACK["title"]


def track_summary():
    out = [f"## What track {TRACK['id']} leans on", "",
           f"**The question.** {TITLE}", "",
           "Nothing here is new. These are the weeks to look back at while you work, and the "
           "wording is the course's own. It is a long table because this track reaches across "
           "six of them — from a straight line to a neural network — which is also why it is the "
           "one that will take you longest.", "",
           "### The ideas, in plain words", "", "| Idea | Means |", "|---|---|"]
    out += [f"| **{d['idea']}** | {d['words']} |" for d in (idea(m, i) for m, i in TRACK_IDEAS)]
    out += ["", "### Code you will reach back for", "", "| Function | What it does |", "|---|---|"]
    out += [f"| `{f['name']}` | {f['does']} |" for f in (fn(m, n) for m, n in TRACK_FNS)]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 2b. orderings, derived rather than typed
# ---------------------------------------------------------------------------
# The model answers below make claims about RANK — "the leaky split is third of four", "the border
# split is best on RMSE". A rank is a number like any other and must not be typed from a run: the
# service adds earthquakes, and a hand-written ordering would go stale silently while every figure
# beside it stayed right. These derive it.
ORDINAL = {1: "first", 2: "second", 3: "third", 4: "fourth"}


def ranked(key, best="high"):
    """The four splits listed best-first on one score, as 'name value, name value, ...'."""
    order = sorted(TABLE.items(), key=lambda kv: kv[1][key], reverse=(best == "high"))
    return ", ".join(f"{name} {row[key]:.3f}" for name, row in order)


def rank_of(name, key, best="high"):
    """Where one split comes on one score, as a word: 'first', 'second', ..."""
    order = sorted(TABLE, key=lambda n: TABLE[n][key], reverse=(best == "high"))
    return ORDINAL[order.index(name) + 1]


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

HOOK = f"""
Somebody designing a hospital has to answer this with a number before the concrete is poured. Every
seismic building code in the world rests on a **ground-motion prediction equation**: a formula that
takes a magnitude, a distance and a description of the ground underfoot, and returns the peak
acceleration to build for. It is fitted to recordings of real earthquakes, and the fit is never
close — the same magnitude at the same distance can shake one site several times harder than
another.
The width of that miss, not the middle of it, is what a building code actually spends money on.

There are now enough recordings to fit something far more flexible than a formula, and papers
reporting that machine learning beats the classical equation appear every year. This project is
about whether that is true. The answer turns out to depend almost entirely on one decision that has
nothing to do with the model: **which records you hide from yourself before you score it.**
"""

md(weekkit.OPENING.format(question=TITLE, datahub=datahub, hook=HOOK.strip()))

md("""
## How this notebook is different

This is a **project track**. It is not a weekly notebook and it does not behave like one.

A weekly notebook shows you a move, walks you through it, and then asks you to make it once
yourself. This one loads the data and reproduces the one result the field already agrees on — the
classic ground-motion equation, its coefficients and its scatter — and then stops helping. From
there on every section is a sentence describing what to find out and an empty cell to find it out
in. There is no worked example above to pattern-match against, because on a real question there
never is one.

**There is exactly one self-check in this notebook, and it is on the data loading.** After that,
nothing tells you whether you are right. That is not an oversight and it is not laziness: past the
loading step there is no single right answer here, so a cell that said `assert` would be lying to
you about how research works. What replaces it is the thing researchers actually use — a result you
can get two ways, a number you can predict before you compute it, and a claim you can try to break.

**And it does not close.** The last section is a question this course does not know the answer to.
Everything above it is scaffolding; that question is the project.
""")

md(f"""
## What you'll be able to do

**The science.** Fit the equation that seismic building codes are built on, say how wrong it
typically is and in what units that matters, and then decide — from your own measurements, not from
a paper's claim — whether a machine-learning model genuinely predicts shaking better or only appears
to under a careless test.

**The skills.** Turn raw columns into the features a physical model needs. Split a dataset four
different ways and see the score change without the model changing at all. Put an interval on the
*difference* between two models, so that "better" is a claim with a number attached.

**The four questions, in order:**

1. How much of the shaking can one straight line explain?
2. How do you split train from test, and does the choice change the answer?
3. Does a flexible model beat the physics, or only look like it?
4. What is the leftover scatter made of?

The open question at the end is not on that list. It is the project; the four above are what you
build to reach it.
""")

md(f"""
## Setup

The Engineering Strong Motion database publishes a *flatfile*: one row per recording of one
earthquake at one instrument, with the earthquake, the instrument, the ground beneath it and the
processed shaking all on the same line. There is no key and no login, and the whole thing comes
back from one URL.

**Read this before you go on.** Four things about the file decide what you can honestly do with it,
and all four are measurable rather than assumed:

- The file is **semicolon-delimited**, so `pd.read_csv` needs `sep=";"`. Read it the usual way and
  you get a single column with the whole row inside it.
- **`rotd50_pga` is in cm/s²**, and everything in earthquake engineering is quoted in *g*. Dividing
  by {G_CM_S2:.0f} is not a formatting choice; forget it and every number you report is out by a
  factor of a thousand.
- **Distance is not one column.** `jb_dist` is the distance to the rupture surface, which is what a
  ground-motion equation really wants, and it is missing on {M['jb_missing_raw'] * 100:.0f}% of the
  rows because it requires a fault model somebody had to build. `epi_dist`, the distance to the
  epicentre, is always there. Filling one from the other is a compromise, and it is yours to defend.
- **`preferred_vs30_m_s`** is how fast a shear wave travels through the top thirty metres of ground
  — the standard one-number description of a site, low for soft mud and high for hard rock. It is
  present on {M['vs30_present'] * 100:.0f}% of the usable rows and blank on the rest.

The query is pinned to a magnitude floor of {MIN_MAGNITUDE} and the window {STARTTIME} to
{ENDTIME}, and it takes the better part of a minute. The database grows as new earthquakes are
processed, so your counts may differ from the ones printed in this notebook by a few.
""")

code(weekkit.setup_cell(
    imports=("import numpy as np\n"
             "import torch\n"
             "from sklearn.ensemble import RandomForestRegressor\n"
             "from sklearn.linear_model import LinearRegression\n"
             "from sklearn.preprocessing import StandardScaler\n"
             "from torch import nn\n"),
    figsize="(7, 4)",
    cache_base=CACHE_BASE,
    signature="url, cache_name",
    docstring="Read the live flatfile; fall back to the copy stored with the course.",
    url_expr='url, sep=";"',
    cache_expr='cache_name, sep=";"',
    unpack=f'''
FIELDS = ({chr(10).join('          "' + line + '"' for line in FIELD_LINES).lstrip()})
ESM = (f"https://esm-db.eu/esmws/flatfile/1/query?min-magnitude={MIN_MAGNITUDE}"
       f"&starttime={STARTTIME}&endtime={ENDTIME}&include-fields={{FIELDS}}")

records = load(ESM, "{ESM_CACHE}")
print("the flatfile as it arrives:", records.shape)
print(records[["event_time", "mw", "epi_dist", "preferred_vs30_m_s", "rotd50_pga"]].head())
'''.strip("\n")))

md(f"""
Five columns are built from the raw ones, and then two filters are applied. The first throws away
rows the database itself flags as unreliable. The second cuts to the records a ground-motion
equation is honest about: magnitude {MAG_FLOOR} and up, within {R_MAX} km, and with the site
actually measured.
""")

code(f"""
records["mag"] = records["mw"].fillna(records["ml"])
records["R"] = records["jb_dist"].fillna(records["epi_dist"])
records["pga_g"] = records["rotd50_pga"] / {G_CM_S2}
records["vs30"] = records["preferred_vs30_m_s"]
records["station"] = records["network_code"] + "." + records["station_code"]

usable = records[(records["pga_g"] > 0)
                 & records["mag"].notna()
                 & (records["quality_class"] != "BAD")
                 & (records["late_triggered_event_01"].fillna(0) == 0)]

shaking = usable[(usable["mag"] >= {MAG_FLOOR})
                 & (usable["R"] <= {R_MAX})
                 & usable["vs30"].notna()].reset_index(drop=True).copy()

shaking["log_pga"] = np.log10(shaking["pga_g"])
shaking["log_r"] = np.log10(np.sqrt(shaking["R"] ** 2 + {PSEUDO_DEPTH} ** 2))
shaking["log_vs30"] = np.log10(shaking["vs30"])

print("rows from the service: ", len(records))
print("passing the quality cut:", len(usable))
print("usable for a model:    ", len(shaking))
""")

code(f"""
assert "rotd50_pga" in records.columns and "preferred_vs30_m_s" in records.columns, \\
    "a column this project needs is missing — the query or the service's schema changed"
assert 14000 < len(shaking) < 20000, \\
    "expected about {M['n_work']} usable records; a very different number means a filter missed"
assert shaking["pga_g"].max() < 3, \\
    "a peak acceleration above 3 g means the cm/s^2 to g conversion did not happen"
print(f"✓ the data — {{len(records)}} rows from the service, {{len(shaking)}} usable for a "
      f"ground-motion model, from {{shaking['esm_event_id'].nunique()}} earthquakes recorded at "
      f"{{shaking['station'].nunique()}} stations")
""")

md("""
### And that is the last self-check in this notebook

The pipeline is now trustworthy: the file is the file, the filters are the filters, the units are
in g. Everything from here is yours, and nothing will tell you when you have it right.
""")

# --- SECTION 1 ---------------------------------------------------------------
md(f"""
## How much of the shaking can one straight line explain?

Before any model, look at what you have. Two given figures: where the recordings are, and how the
shaking falls off with distance.
""")

code(f"""
coast = pd.read_csv(CACHE + "/coastlines.csv")

plt.plot(coast.lon, coast.lat, color="0.6", lw=0.6)
plt.scatter(shaking["st_longitude"], shaking["st_latitude"], s=2, color="0.35",
            label="recording stations")
plt.scatter(shaking["ev_longitude"], shaking["ev_latitude"], s=6, color="firebrick",
            label="earthquakes")
plt.xlim({MAP_BOX[0]}, {MAP_BOX[1]})
plt.ylim({MAP_BOX[2]}, {MAP_BOX[3]})
plt.gca().set_aspect("equal")
plt.xlabel("longitude (degrees east)")
plt.ylabel("latitude (degrees north)")
inside = (shaking["st_longitude"].between({MAP_BOX[0]}, {MAP_BOX[1]})
          & shaking["st_latitude"].between({MAP_BOX[2]}, {MAP_BOX[3]}))
plt.title(f"{{len(shaking)}} recordings; {{inside.sum()}} of them inside this box")
plt.legend(loc="lower left", fontsize=7)
plt.show()

print(shaking["st_nation_code"].value_counts().head(6).to_dict())
""")

md(f"""
Now the physics. Shaking grows with magnitude and dies away with distance, and both effects span
factors of thousands, so neither is visible on ordinary axes.

**Log axes:** {idea('D4', 'Log axes')['words']}
""")

code(f"""
bands = [(4.0, 5.0), (5.0, 6.0), (6.0, 9.0)]
shades = ["0.75", "0.45", "firebrick"]

for band, shade in zip(bands, shades):
    low, high = band
    rows = shaking[(shaking["mag"] >= low) & (shaking["mag"] < high)]
    plt.scatter(rows["R"], rows["pga_g"], s=2, color=shade,
                label=f"M {{low}}-{{high}}  (n = {{len(rows)}})")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("distance to the earthquake (km)")
plt.ylabel("peak ground acceleration (g)")
plt.title(f"How shaking falls off with distance (n = {{len(shaking)}})")
plt.legend(fontsize=7)
plt.show()
""")

md(f"""
Three clouds, one above the other, each falling away in a straight line. That is what a ground-motion
equation is: a straight line in the right coordinates. Notice what a log axis cannot draw —
{int((shaking['R'] == 0).sum())} of these records sit at exactly R = 0 km and are silently missing
from the figure, which is the same problem the formula below has to solve.

```
log10 PGA  =  b0  +  bM * M  +  bR * log10(sqrt(R^2 + h^2))  +  bV * log10(Vs30)
```

Three of those terms are just columns you already have, put on log axes. The fourth is the one piece
of engineering in the formula: `sqrt(R^2 + h^2)` instead of `R`. An earthquake is not a point, so at
zero distance `log10(R)` would be minus infinity and the predicted shaking infinite; `h` is a
made-up depth of a few kilometres that stops that happening. Sweeping it from 4 to 12 km moves the
fit by {M['h_spread']:.4f} in R² — smaller than the spread the four splits below will produce
without touching the model at all — so {PSEUDO_DEPTH} km is used here, and you may refit it.

**Linear regression:** {idea('ML1', 'Linear regression')['words']}
""")

code(f"""
FEATURES = ["mag", "log_r", "log_vs30"]
X = shaking[FEATURES]
y = shaking["log_pga"]

gmpe = LinearRegression().fit(X, y)
residual = y - gmpe.predict(X)

print("log10 PGA[g] = %.3f %+.3f*M %+.3f*log10(sqrt(R^2 + %d^2)) %+.3f*log10(Vs30)"
      % (gmpe.intercept_, gmpe.coef_[0], gmpe.coef_[1], {PSEUDO_DEPTH}, gmpe.coef_[2]))
print("R2 on the records it was fitted to:", round(gmpe.score(X, y), 3))
print("sigma, the typical miss, in log10 units:", round(residual.std(), 4))
""")

md(f"""
Those four coefficients are the result this notebook hands you, and they are the shape a
seismologist expects: shaking rises with magnitude, falls with distance faster than one over R, and
falls as the ground gets stiffer. Everything after this point is yours.

**Baseline:** {idea('ML3', 'Baseline')['words']}
""")

ask(f"""
### ✏️ Your turn 1

Put a number on what the physics bought, in units an engineer would recognise.

The dumbest possible model ignores magnitude, distance and site and always guesses the average
log shaking. Its typical miss is just the spread of `y` itself. The fitted model's typical miss is
the spread of `residual`. Both are in log10 units, and a miss of 1.0 in log10 means "wrong by a
factor of 10", so `10 ** miss` turns each into a factor.

Print both misses and both factors, and the ratio between the two factors.

Then, in one printed sentence: every one of the {M['n_work']:,} records was used to choose those
four coefficients, so what has that R² *not* told you, and what would you have to do to find out?
""")

answer(f"""
always_the_mean = y.std()
the_fitted_line = residual.std()

print("guess the average every time — typical miss", round(always_the_mean, 3),
      "log10 units, a factor of", round(10 ** always_the_mean, 2), "in shaking")
print("the three-term equation   — typical miss", round(the_fitted_line, 3),
      "log10 units, a factor of", round(10 ** the_fitted_line, 2), "in shaking")
print("so magnitude, distance and site together buy a factor of",
      round(10 ** always_the_mean / 10 ** the_fitted_line, 2))

print("It has not told me whether the equation predicts a record it has never seen.",
      "Every one of the", len(shaking), "records was used to choose the four coefficients,",
      "so the line has been fitted to its own test. To find out I would have to hide some",
      "records from the fit, score the equation only on those, and report that number instead.")
""")

# --- SECTION 2 ---------------------------------------------------------------
md(f"""
## How do you split train from test, and does the choice change the answer?

**Train/test split:** {idea('ML2', 'Train/test split')['words']}

That sentence is one line of code and it hides the entire difficulty of this project. Hide *which*
data? The obvious answer — a quarter of the rows, picked at random — is the one every introductory
tutorial gives, and on this dataset the rows are not independent things. The median earthquake here
contributes {M['per_event_median']:.0f} rows and the largest contributes {M['per_event_max']}; the
median station contributes {M['per_station_median']:.0f} and the busiest {M['per_station_max']}.
What that does to a split made at random is the first thing to measure.

There are at least four defensible answers, and they are the one real decision in this track:

- **at random** — every record is its own independent thing;
- **by event** — hide whole earthquakes, so the model has never seen this rupture;
- **by station** — hide whole recording sites, so the model has never seen this patch of ground;
- **across a border** — train everywhere except Italy and test in Italy, which is what you are
  really doing whenever you apply a European equation in California.

This is the idea the model-selection week called *leakage*: any route by which information about the
data you are scoring on reaches the model before you score it. Below are the helpers you will need;
the four splits are yours to build.

Two scores, not one, and the reason matters. **R² is a ratio**: the miss divided by the spread of
the very test set it was measured on. Change the test set and you change the denominator, so four
R² from four different splits are four different quantities and cannot be laid side by side.
**RMSE is the miss itself**, in log10 units of PGA, and those units mean the same thing in every
test set. Report both, and where they disagree, believe the one whose units you can name.
""")

code(f"""
def hold_out_quarter(labels, seed={SEED}):
    \"\"\"True for the rows to TRAIN on, False for the rows held out.

    Whole labels go into the test set, in random order, until a quarter of the records are gone —
    so `labels` is where you say what one independent thing is.
    \"\"\"
    # 1. Count how many records each label owns, then shuffle that list. Without the shuffle the
    #    biggest earthquakes would be held out every time, which asks a different question.
    counts = pd.Series(labels).value_counts().sample(frac=1, random_state=seed)
    held_out = []
    rows_out = 0
    # 2. Move whole labels into the test set until a quarter of the RECORDS have gone. Counting
    #    records rather than labels keeps the test set the same size whichever column you split on.
    for name, n_rows in counts.items():
        if rows_out >= 0.25 * len(labels):
            break
        held_out.append(name)
        rows_out = rows_out + n_rows
    # 3. Every record carrying a held-out label leaves together. That is the point of splitting
    #    this way: if two records from one earthquake sat on opposite sides of the split, the
    #    model could half-remember the answer instead of having to predict it.
    return ~np.isin(labels, held_out)


def r_squared(predicted, actual):
    \"\"\"The fraction of the up-and-down variation a prediction accounts for.\"\"\"
    return 1 - ((actual - predicted) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()


def rmse(predicted, actual):
    \"\"\"The typical miss, in the units of the thing being predicted.\"\"\"
    return np.sqrt(((actual - predicted) ** 2).mean())


def held_out_predictions(model, is_train):
    \"\"\"Fit the model on the training records; hand back what it predicts for the held-out ones.

    `model` is either a scikit-learn estimator — anything with `.fit` and `.predict` — or a plain
    function of `is_train` that does its own fitting and returns the held-out predictions. The
    second form is the one a torch model needs, and accepting both is what lets every model in
    this notebook be scored by the same call instead of two.
    \"\"\"
    # A scikit-learn model is fitted and then asked; a torch model brings its own training loop
    # and so arrives as a function. Asking whether the object has a `.fit` tells the two apart.
    if hasattr(model, "fit"):
        # The model is shown the training records only. It never sees `X[~is_train]` while it is
        # learning, which is what makes the score below a test rather than a memory check.
        model.fit(X[is_train], y[is_train])
        return model.predict(X[~is_train])
    return model(is_train)


def held_out_r2(model, is_train):
    \"\"\"Fit on the training records, score on the held-out ones.\"\"\"
    return r_squared(held_out_predictions(model, is_train), y[~is_train])


def held_out_rmse(model, is_train):
    \"\"\"Fit on the training records, and how far off it typically is on the held-out ones.\"\"\"
    return rmse(held_out_predictions(model, is_train), y[~is_train])
""")

ask(f"""
### ✏️ Your turn 2

Build the four splits, as a dictionary of name → mask, so that every later section can loop over
the same four. Three of them come from `hold_out_quarter` with a different `labels` argument; the
fourth is a comparison you write yourself.

    "at random"       hold_out_quarter(np.arange(len(shaking)))
    "by event"        hold_out_quarter(shaking["esm_event_id"])
    "by station"      hold_out_quarter(shaking["station"])
    "across a border" (shaking["st_nation_code"] != "IT").values

For each split print six things: how many records are in the training and the test set; what
**fraction of the held-out records share their earthquake with a record in the training set**;
what fraction **share their station**; the held-out R²; and the held-out RMSE. Print the standard
deviation of `y` in each test set too — that is the denominator each R² was divided by, and it is
the reason the two scores will not rank the four splits the same way.

Draw the four R² values and the four RMSE values as two bar charts side by side.

Then print two sentences: which one of these four splits would you put in a paper as "the
equation's predictive accuracy", and what are the other three measuring instead?
""")

answer(f"""
splits = {{"at random": hold_out_quarter(np.arange(len(shaking))),
          "by event": hold_out_quarter(shaking["esm_event_id"]),
          "by station": hold_out_quarter(shaking["station"]),
          "across a border": (shaking["st_nation_code"] != "IT").values}}

scores = []
misses = []
for name, is_train in splits.items():
    same_event = shaking["esm_event_id"][~is_train].isin(shaking["esm_event_id"][is_train])
    same_station = shaking["station"][~is_train].isin(shaking["station"][is_train])
    scores.append(held_out_r2(LinearRegression(), is_train))
    misses.append(held_out_rmse(LinearRegression(), is_train))
    print(f"{{name:<16}} train {{is_train.sum():>6}}  test {{(~is_train).sum():>5}}"
          f"   share an earthquake {{same_event.mean():.3f}}"
          f"   share a station {{same_station.mean():.3f}}"
          f"   sd(y) {{y[~is_train].std():.3f}}"
          f"   R2 {{scores[-1]:.3f}}   RMSE {{misses[-1]:.3f}}")

plt.bar(list(splits), scores, color="0.4")
plt.axhline(gmpe.score(X, y), color="firebrick", lw=1.2)
plt.xlabel("how the held-out quarter was chosen")
plt.ylabel("R2 on the held-out records")
plt.title(f"The same equation, scored four ways (n = {{len(shaking)}})")
plt.show()

plt.bar(list(splits), misses, color="0.4")
plt.xlabel("how the held-out quarter was chosen")
plt.ylabel("RMSE, log10 units of PGA")
plt.title("The same four splits, scored in the units of the thing predicted")
plt.show()

print("I would report the 'across a border' number, because that is the situation an equation is",
      "actually used in: somebody applies it where it was not fitted. The random split is not a",
      "test at all — almost every held-out record shares its earthquake AND its station with a",
      "training record, so the equation is being asked about data it has effectively seen.")
print("'By event' measures whether the equation transports to a NEW earthquake at familiar",
      "stations; 'by station' whether it transports to a new SITE in familiar earthquakes.",
      "Each answers a real question, and none of them answers the same one.")
""")

ask(f"""
### ✏️ Your turn 3

Three paragraphs, quoting **your own numbers** — the four R², the four RMSE, the four test-set
standard deviations and the two 'share' fractions.

1. Only one of the four splits gives a number you could honestly call this equation's predictive
   accuracy. Say which, and say what each of the other three has let in, using the share fractions
   as your evidence rather than as an assertion.
2. Now check what those splits actually did to the score. Rank the four splits by R², then rank
   them by RMSE, and account for the difference between the two rankings using the test-set
   standard deviations — one of these scores is a ratio and one is not. Say plainly whether the
   split you called leaky came out highest, and if it did not, explain why not rather than
   explaining it away.
3. A split that lets information through does not automatically inflate a score; it inflates the
   score of a model that can use the information. Say what that implies about this particular
   equation, and predict what should happen to the spread of these four numbers when you give a
   model more freedom in the next section.
""")

answer_prose(f"""
Only the border split is a real test of the thing an equation is for. My four share-an-earthquake
fractions are {TABLE['at random']['share_event']:.3f}, {TABLE['by event']['share_event']:.3f},
{TABLE['by station']['share_event']:.3f} and {TABLE['across a border']['share_event']:.3f}, and my
four share-a-station fractions are {TABLE['at random']['share_station']:.3f},
{TABLE['by event']['share_station']:.3f}, {TABLE['by station']['share_station']:.3f} and
{TABLE['across a border']['share_station']:.3f}. Under the random split
{TABLE['at random']['share_event'] * 100:.0f}% of the records I am scoring on come from an
earthquake that is already in the training set and
{TABLE['at random']['share_station'] * 100:.0f}% come from a station that is, so whatever it
returns is not a prediction — it is a measurement of how well the model interpolates between
records it has almost already seen. The event split fixes one of those and leaves the other
({TABLE['by event']['share_station'] * 100:.0f}% still share a station), and the station split
fixes the other and leaves the first
({TABLE['by station']['share_event'] * 100:.0f}% still share an earthquake). Only the border split
takes both away at once — {TABLE['across a border']['share_event'] * 100:.0f}% share an earthquake,
and those are the border-region events recorded on both sides, which is honest rather than a bug.

I expected that to show up as the random split scoring highest, and it does not. Ranked by R² my
four splits go {ranked('line')} — the split with both kinds of leakage comes
{rank_of('at random', 'line')} of four, and the border split I called the only honest one scores
{TABLE['across a border']['line'] - TABLE['at random']['line']:+.3f} *above* it. Ranked by RMSE the
order is not the same: {ranked('rmse_line', best='low')}, and now the random split is
{rank_of('at random', 'rmse_line', best='low')} — the worst of the four — while the border split is
{rank_of('across a border', 'rmse_line', best='low')}. Two rankings of the same four fits cannot
both be a ranking of accuracy, and the one to distrust is the R². R² is a ratio: the miss divided
by the spread of whichever records it was scored on, and my four test sets have spreads of
{TABLE['at random']['sd']:.3f}, {TABLE['by event']['sd']:.3f}, {TABLE['by station']['sd']:.3f} and
{TABLE['across a border']['sd']:.3f}. Four different denominators make four different quantities,
and a spread of {max(t['line'] for t in TABLE.values()) - min(t['line'] for t in TABLE.values()):.3f}
across them is not one thing varying. RMSE is the miss itself, in log10 units of PGA in all four,
so it is the one I can lay side by side.

On RMSE the border result stops being a paradox and becomes a fact about the test set: Italy is
easier. Its median source distance is {TABLE['across a border']['median_r']:.0f} km against
{TABLE['at random']['median_r']:.0f} km for a random quarter of the file, and near-field records are
where a distance-attenuation term is best determined, so the absolute miss there is the smallest of
the four ({TABLE['across a border']['rmse_line']:.3f}). Italy also has the smallest spread to
explain ({TABLE['across a border']['sd']:.3f}), which pushes its R² *down* — the two effects work
against each other and the smaller miss wins.

So the lesson is not that a random split always overstates. It can, and on this equation it does
not, and the reason is the equation rather than the split. Leakage inflates a score only for a
model that has somewhere to put what
leaks through. This one has four numbers and three of them are physics — bigger earthquakes shake
harder, distance attenuates, soft ground amplifies — so being handed a held-out record whose
earthquake and whose station are both already in the training set buys it nothing: it has no
per-event and no per-station parameter to bend. That is the useful result here, and it is a stronger
statement than "the random split lies", because it says *when* the random split lies. My trivial
baseline is the other end of the same argument: guessing the average scores
{TABLE['across a border']['mean']:.3f} across the border, and the difference between that and
{TABLE['across a border']['line']:.3f} is what the three physical terms are worth. The prediction I
would carry into the next section is that a model with more freedom will spread these four numbers
much further apart, because freedom is exactly what lets a model record which station a row came
from instead of learning why that station shakes.
""")

# --- SECTION 3 ---------------------------------------------------------------
md(f"""
## Does a flexible model beat the physics, or only look like it?

Now give a model the same three columns and much more freedom.

**Random forest:** {idea('ML4', 'Random forest')['words']} The rock week used it to choose between
labels; here the same forest predicts a number instead, which in scikit-learn means
`RandomForestRegressor` in place of `RandomForestClassifier` and changes nothing else about how it
is called.

**Neural network:** {idea('ML6', 'Neural network')['words']}

Neither knows any seismology. Neither has been told that shaking falls off with distance. Both have
enough freedom to notice things about the {M['n_work']:,} particular records they are shown.
""")

md(f"""
### Predict before you run

You are about to score a random forest against the straight line, on the same three columns, under
each of your four splits. Commit to two numbers first: how much R² the forest gains over the
equation when the held-out quarter is random, and how much it gains when the held-out quarter is
whole stations. A wrong guess you committed to is worth more than a right answer you were shown.
""")

code(f"""
my_random_gain = 0.10
my_station_gain = 0.10

print("I think the forest gains", my_random_gain, "R2 under a random split and",
      my_station_gain, "under a station split")
""")

ask(f"""
### ✏️ Your turn 4

Score `RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=0)` against
`LinearRegression()` on every one of your four splits, using `held_out_r2` for both so that the
only thing changing between rows is the split.

Print, per split: the equation's R², the forest's R², the difference, and both RMSE values so the
row is also readable in the units of the thing predicted. Draw the four differences as a bar chart
with zero marked, so the sign is visible.

Then print `forest.feature_importances_` beside `FEATURES` for a forest fitted on everything.

Finish with three printed sentences on your own numbers. Say what the forest is doing that the
straight line cannot. Then check the obvious explanation against your own share fractions: if the
forest's advantage came from records that share an earthquake or a station with the training set,
what should the advantage be on the split where the share-a-station fraction is zero, and what is
it? Last, say which of your four numbers you would have quoted if you had only ever run the first
one.
""")

answer(f"""
forest = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=0)

gaps = []
for name, is_train in splits.items():
    line = held_out_r2(LinearRegression(), is_train)
    trees = held_out_r2(forest, is_train)
    gaps.append(trees - line)
    print(f"{{name:<16}} equation {{line:.3f}}   forest {{trees:.3f}}"
          f"   forest - equation {{trees - line:+.3f}}"
          f"   RMSE {{held_out_rmse(LinearRegression(), is_train):.3f}}"
          f" -> {{held_out_rmse(forest, is_train):.3f}}")

plt.bar(list(splits), gaps, color="0.4")
plt.axhline(0, color="firebrick", lw=1.2)
plt.xlabel("how the held-out quarter was chosen")
plt.ylabel("forest R2 minus equation R2")
plt.title(f"What the extra freedom is worth, four ways (n = {{len(shaking)}})")
plt.show()

forest.fit(X, y)
print("what the forest leaned on:")
print(pd.Series(forest.feature_importances_.round(3), index=FEATURES).to_string())

print("The forest fits the surface in pieces, so it can bend where the straight line cannot —",
      "near-field saturation, and magnitude and distance interacting instead of adding. The line",
      "has four numbers and they apply everywhere.")
print("The obvious explanation is that the gain is leakage: the forest recognising a station it",
      "has already been trained on. If that were the whole story the gain should be zero on the",
      "station split, where the share-a-station fraction is 0.000. It is not zero, it is",
      round(gaps[2], 3), "- so part of the advantage is genuine curvature and survives having",
      "every test station withheld. That also means leakage cannot be what makes the border",
      "column negative, because the station split has none either and is still positive. Whatever",
      "reverses the sign is something the border does and the station split does not, and I have",
      "not measured it yet.")
print("If I had only run the random split I would have quoted", round(gaps[0], 3),
      "as the improvement from machine learning, and it would have been the wrong number by",
      "about", round(abs(gaps[0] - gaps[-1]), 3), "R2.")
""")

ask(f"""
### ✏️ Your turn 5

Now the model the open question is actually about: a small neural network on the same three columns.

**The contract, which is all you are given.** Write a function `network_predictions(is_train)` that
trains a network on the training records and hands back its predictions for the held-out ones as a
plain numpy array — so that `held_out_r2(network_predictions, is_train)` scores it by exactly the
call that scored the equation and the forest. The name and that one argument are what the rest of
the notebook depends on; everything inside is yours.

The design, in words. The calls are the waveform week's, and every one of them is in the summary
table at the foot of this notebook.

- Standardise the three features first. Fit the scaler on the training rows **only** and apply it
  to the held-out rows — fitting it on everything is leakage, of a small and famous kind.
- Two hidden layers of {WIDTH} units with a rectifier between them, and one output. Torch wants
  float32 tensors, and the targets as a column rather than a row.
- Mean-squared-error loss, Adam at a learning rate of 0.01, {EPOCHS} passes over the training data
  in reshuffled minibatches of {BATCH}.
- Seed the network before you build it, so that rerunning the cell gives you the same answer twice.
- Bring the predictions back out of torch as a flat numpy array.

Score it on all four splits beside the equation and the forest, and print the three R² values per
split. It trains in a second or two per split on a laptop; if the third decimal moves when you rerun
it, that is the random start, not a mistake.

Then print two or three sentences answering the question this track exists for, on your own
numbers: does the network beat the equation? Say under which splits it does and which it does not,
and say what your answer would have been if the only split you had run were the random one.
""")

answer(f"""
def network_predictions(is_train, epochs={EPOCHS}, seed=0):
    \"\"\"Train the small network on the training records; predict the held-out ones.\"\"\"
    # 1. Put the three features on the same scale. A network learns by nudging its weights, and
    #    an unscaled column would be nudged far harder than the others for no good reason. The
    #    scaler is fitted on the TRAINING records and only then applied to the held-out ones, so
    #    nothing the model is about to be tested on leaks into how it was trained.
    scaler = StandardScaler()
    x_train = torch.tensor(scaler.fit_transform(X[is_train]), dtype=torch.float32)
    x_test = torch.tensor(scaler.transform(X[~is_train]), dtype=torch.float32)
    y_train = torch.tensor(y[is_train].values, dtype=torch.float32).reshape(-1, 1)

    # 2. Fix the random numbers before the layers are built, so the network starts from the same
    #    weights every run and the numbers this cell prints repeat.
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(len(FEATURES), {WIDTH}), nn.ReLU(),
                        nn.Linear({WIDTH}, {WIDTH}), nn.ReLU(),
                        nn.Linear({WIDTH}, 1))
    optimiser = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_function = nn.MSELoss()

    # 3. Train: `epochs` passes over the training records, and a fresh random order each pass so
    #    the network cannot pick anything up from the order the records happen to sit in.
    for epoch in range(epochs):
        order = torch.randperm(len(x_train))
        for start in range(0, len(x_train), {BATCH}):
            batch = order[start:start + {BATCH}]
            optimiser.zero_grad()
            loss = loss_function(net(x_train[batch]), y_train[batch])
            loss.backward()
            optimiser.step()

    # 4. `detach` drops the bookkeeping torch keeps in order to train, leaving plain numbers.
    return net(x_test).detach().numpy().ravel()


network_gaps = []
for name, is_train in splits.items():
    line = held_out_r2(LinearRegression(), is_train)
    trees = held_out_r2(forest, is_train)
    brain = held_out_r2(network_predictions, is_train)
    network_gaps.append(brain - line)
    print(f"{{name:<16}} equation {{line:.3f}}   forest {{trees:.3f}}   network {{brain:.3f}}"
          f"   network - equation {{brain - line:+.3f}}")

print("The network beats the equation under the three splits that leave shared earthquakes or",
      "shared stations in the test set, by between", round(min(network_gaps[:3]), 3), "and",
      round(max(network_gaps[:3]), 3), "R2, and it loses by", round(-network_gaps[-1], 3),
      "once the test records are across a border. It is also no better than the forest, which is",
      "a hint that the extra flexibility is being spent on the same thing.")
print("Had I only run the random split I would have written that a neural network improves",
      "ground-motion prediction, and I would have had a real number to put in the sentence.",
      "The number would have been measuring the wrong thing.")
""")

md(f"""
### Is the difference bigger than the noise in the test set it was measured on?

One thing is still missing before any of this is a claim. A difference in R² is itself a
measurement, made on one particular held-out set, and it has a spread like anything else. The
station-split gap you just printed is a small number; quoting it with no width beside it is
reporting a coin flip as a tendency.

**Bootstrap:** {idea('S4', 'Bootstrap')['words']}

**Confidence interval:** {idea('S4', 'Confidence interval')['words']}
""")

ask(f"""
### ✏️ Your turn 6

Put an interval on the forest's advantage, for the **station** split, by resampling the held-out set
itself.

**The contract.** Produce `gaps`, an array of {N_BOOT} values, each one (forest R² − equation R²)
recomputed on a resampled version of the same held-out set — with both models fitted once, on the
real training set, and then left alone. You are resampling what you *scored on*, not what you
trained on.

Two things decide whether it is right, and both are yours to get right:

- **Resample stations, not rows.** The held-out records are no more independent than the rest of
  the file; two records from one station carry nearly one station's worth of information. So one
  draw is a whole station's block of positions, drawn with replacement until you have as many
  blocks as there are held-out stations. Resampling rows would give you an interval several times
  too narrow, which is the whole reason this section exists.
- **Refit nothing.** The predictions are computed once, before the loop. A bootstrap that refits
  inside the loop is measuring something else, and takes an hour.

Report the 2.5th and 97.5th percentiles of `gaps`, and the fraction of the {N_BOOT} resamples in
which the forest's advantage is zero or negative. Draw the {N_BOOT} differences as a histogram with
zero and your observed difference marked. Then do the whole thing again for the random split.

Finish with two printed sentences on your own two intervals. Is the advantage you measured in *Your
turn 4* bigger than the noise in the test set it was measured on, and does your answer depend on
which split you ask it about? Be careful what you claim: with the fits frozen, this interval covers
the luck of **which held-out stations you happened to score on**. It says nothing about which
stations went into the training set — changing that would refit both models, and is a different
experiment.
""")

answer(f"""
def gap_interval(split_name):
    \"\"\"The 95% interval on (forest - equation) for one split, resampling held-out stations.\"\"\"
    # 1. Both models are fitted on the same training records and asked about the same held-out
    #    ones, so the only thing that differs between the two sets of predictions is the model.
    is_train = splits[split_name]
    actual = y[~is_train].values
    predicted_line = held_out_predictions(LinearRegression(), is_train)
    predicted_forest = held_out_predictions(forest, is_train)

    # 2. Collect the row numbers belonging to each held-out station. Two records from one station
    #    are not two independent pieces of evidence, so the resampling below moves whole stations
    #    rather than single records — otherwise the interval comes out far too narrow.
    held = shaking[~is_train].reset_index(drop=True)
    positions = []
    for station, rows in held.groupby("station"):
        positions.append(rows.index.values)

    rng = np.random.default_rng({SEED})
    gaps = []
    # 3. Build {N_BOOT} alternative test sets by drawing stations at random, with replacement, and
    #    score both models on each one. The spread of the resulting gaps is how much of the gap
    #    could be luck in which stations happened to be held out.
    for i in range({N_BOOT}):
        picked = rng.integers(0, len(positions), size=len(positions))
        parts = []
        for p in picked:
            parts.append(positions[p])
        take = np.concatenate(parts)
        gaps.append(r_squared(predicted_forest[take], actual[take])
                    - r_squared(predicted_line[take], actual[take]))
    return np.array(gaps), len(positions)


for split_name in ["at random", "by station"]:
    gaps, n_blocks = gap_interval(split_name)
    observed = (held_out_r2(forest, splits[split_name])
                - held_out_r2(LinearRegression(), splits[split_name]))
    low, high = np.percentile(gaps, [2.5, 97.5])
    print(f"{{split_name:<12}} observed {{observed:+.4f}}"
          f"   95% interval {{low:+.4f}} to {{high:+.4f}}"
          f"   resamples at or below zero {{(gaps <= 0).mean():.3f}}"
          f"   ({{n_blocks}} held-out stations)")

    plt.hist(gaps, bins=40, color="0.4")
    plt.axvline(0, color="firebrick", lw=1.5)
    plt.axvline(observed, color="steelblue", lw=1.5, ls="--")
    plt.xlabel("forest R2 minus equation R2, on a resampled test set")
    plt.ylabel("resamples")
    plt.title(f"{{split_name}}: {N_BOOT} station-block resamples (red = no difference)")
    plt.show()

print("Under the random split the whole interval sits above zero, so that advantage is bigger",
      "than the noise in its own test set — it is a real difference, about a difference that does",
      "not matter. Under the station split the interval straddles zero: rescoring the same two",
      "frozen models on resampled draws of the same held-out stations moves the gap from one side",
      "of zero to the other, so I cannot tell the advantage I reported there from which of those",
      "stations I happened to score on.")
print("That is the only thing this interval covers. Both models were fitted once and never",
      "refitted, so it says nothing about how the gap would move if a different quarter of the",
      "stations had been held out — that would change the training set as well, and I have not",
      "measured it.")
print("So yes, the answer depends entirely on which split I ask about, and the split I would",
      "report is the one where the answer is 'no measurable advantage'.")
""")

# --- SECTION 4 ---------------------------------------------------------------
md(f"""
## What is the leftover scatter made of?

A ground-motion equation's sigma is not a nuisance — it is the number seismic hazard analysis
actually consumes, because a building code asks for the shaking that is exceeded once in five
hundred years, and that lives in the tail. Nothing you have fitted today moved it. Before asking
what could, it is worth knowing what it is made of, because sigma splits into two pieces with
completely different meanings: how much whole *earthquakes* come out above or below the equation,
and how much *individual recordings* scatter within one earthquake.
""")

ask(f"""
### ✏️ Your turn 7

Split the equation's own scatter into the part that belongs to whole earthquakes and the part that
belongs to individual recordings.

1. `shaking["residual"] = y - gmpe.predict(X)`.
2. The average residual of each earthquake is `shaking.groupby("esm_event_id")["residual"].mean()`.
   Put it back on every row with
   `shaking["event_term"] = shaking["esm_event_id"].map(event_mean)` — `map` looks each row's event
   id up in that table of averages.
3. `shaking["within_event"]` is the residual minus the event term.
4. Print the standard deviation of all three: sigma, tau (the event terms) and phi (what is left).
   Print `tau ** 2 + phi ** 2` next to `sigma ** 2` as well.

Draw the event terms and the within-event residuals as two histograms on the same axes.

Then print two or three sentences: which of the two pieces is bigger on your numbers, what a model
would have to know in order to shrink each one, and which of the three models you have fitted today
had any chance of shrinking either.
""")

answer(f"""
shaking["residual"] = y - gmpe.predict(X)
event_mean = shaking.groupby("esm_event_id")["residual"].mean()
shaking["event_term"] = shaking["esm_event_id"].map(event_mean)
shaking["within_event"] = shaking["residual"] - shaking["event_term"]

sigma = shaking["residual"].std()
tau = shaking["event_term"].std()
phi = shaking["within_event"].std()

print("sigma (all of it)          ", round(sigma, 4), "log10 units, a factor of",
      round(10 ** sigma, 2))
print("tau   (whole earthquakes)  ", round(tau, 4), "over", len(event_mean), "earthquakes")
print("phi   (single recordings)  ", round(phi, 4))
print("tau^2 + phi^2 =", round(tau ** 2 + phi ** 2, 5), " and sigma^2 =", round(sigma ** 2, 5))

plt.hist(shaking["event_term"], bins=60, range=(-1.5, 1.5), histtype="step", lw=1.6,
         color="firebrick", label=f"whole earthquakes, tau = {{round(tau, 3)}}")
plt.hist(shaking["within_event"], bins=60, range=(-1.5, 1.5), histtype="step", lw=1.6, ls="--",
         color="0.3", label=f"single recordings, phi = {{round(phi, 3)}}")
plt.xlabel("residual, log10 units of PGA")
plt.ylabel("records")
plt.title(f"What the {{round(sigma, 3)}} of scatter is made of (n = {{len(shaking)}})")
plt.legend(fontsize=7)
plt.show()

print("The two halves are almost the same size, and they add up exactly:", round(tau, 3),
      "and", round(phi, 3), "give", round(tau ** 2 + phi ** 2, 4), "against a sigma squared of",
      round(sigma ** 2, 4), "because an event term and what is left over cannot overlap.")
print("Shrinking tau needs something about the earthquake the equation does not have — the stress",
      "drop, the rupture direction, the style of faulting. Shrinking phi needs something about the",
      "path and the site: the actual rock between here and there, not one number for the top",
      "thirty metres.")
print("None of the three models I fitted today had either. All of them saw the same three",
      "columns, so the most a flexible model could do was rearrange the same information —",
      "which is why the honest splits show it gaining nothing.")
""")

# --- closing ----------------------------------------------------------------
md(f"""
{weekkit.CLOSING_HEADING}

For a magnitude {_demo_M} earthquake {_demo_R} km away on ground with Vs30 = {_demo_V} m/s, this
fit says **{M['demo_pga']:.3f} g** — and its own scatter says the true value will lie between
{M['demo_lo']:.3f} g and {M['demo_hi']:.3f} g nineteen times in twenty, which is {CI_Z} sigma and
not the 2 sigma it is tempting to round it to. That is a factor of {M['demo_factor']:.1f} either
side of the middle, and about {M['demo_span']:.0f} from one end of the range to the other — from
four coefficients fitted to {M['n_work']:,} recordings of {M['events']:,} earthquakes.

The width is the answer. A building code cannot use the middle of that range; it uses the tail, so
the quantity worth improving is sigma, and sigma is what nothing in this notebook moved.
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
Before any statistic, state the dumbest answer to your question and what it gives. Every later
number is reported against it.

On this track the trivial baseline is "always guess the average", and its held-out R² is zero by
construction — which is exactly what makes it useful, because every R² you report afterwards is a
statement about how far past it you got. Quote it, and quote it again in whatever units your
question is really in.
"""),
    "split_by_structure": ("3 · Split by structure", """
Earth data are correlated in space and in time, so a random split puts tomorrow in the training set
and yesterday in the test set, and the score is a lie. Split by time, or by region, and say which
you chose and why.

This track is *about* that choice, so this section carries more weight here than anywhere else in
the course. Name the unit you treated as independent, show the number that told you the other units
were not independent, and report what the score did when you changed it. A project on this track
that reports one split has not done the project.
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
# T1's open_question carries exactly one question mark and it is the first sentence; the rest of
# the entry is the plan's summary of the audit's evidence, which the notebook measures for itself.
OPEN = re.findall(r"[^.?]*\?", " ".join(TRACK["open_question"].split()))[0].strip()

md(f"""
## The open question

> **{OPEN}**

Nobody grading this knows the answer, and neither does the literature. Everything above is the
scaffolding; this is the project.

Here is what is established by the notebook you have just run, and it is less than it looks — stated
as quantities, because the values are the ones you measured and they are yours to read off your own
output rather than mine. Under a random split both flexible models gain R² over the equation, and
the random split's own bootstrap interval excludes zero, so that gain is real; it is just not a gain
at anything useful, because almost every record being scored shares an earthquake and a station with
the training set. Take the shared earthquakes away and the gain shrinks. Take the shared stations
away and it shrinks again, to a number whose interval straddles zero. Send the model across a border
and the sign flips: the four-coefficient equation wins outright, in a country neither model has
seen. Four splits, one model, and the conclusion changes with the split.

What is **not** established is why, or whether it has to be that way. Four directions, none of them
worked out here:

1. **Give the flexible model something the equation does not have.** Every model in this notebook
   saw the same three columns, so the most a network could do was rearrange information the
   equation already used. The flatfile carries the style of faulting, the depth, the event and
   station coordinates and thirty-six spectral periods. Add a feature that is genuinely new physics,
   and re-run all four splits. Does the advantage survive the border this time?
2. **Fit the ergodic assumption instead of ignoring it.** Modern ground-motion models add an
   explicit term per site and per region — a *non-ergodic* model. Your event terms from *Your turn
   7* are a first draft of exactly that. What happens to tau and phi if you allow a per-station
   term, and does the leftover phi shrink enough to be worth the parameters?
3. **Predict which hold-outs will reverse, before running them.** If the reversal is about a test
   set sitting outside the region the model was trained in, then some measure of how far a
   country's records sit from the rest of the file — in magnitude, distance and Vs30 together —
   should order the country hold-outs by their gap. Build that measure, commit to the ordering,
   and only then run the hold-outs. A rule that predicts in advance is worth ten that explain
   afterwards.
4. **Ask how much of sigma is even reducible.** Two recordings of the same earthquake at two
   stations 500 m apart on the same rock still differ. Find such pairs in this file — same event,
   nearly the same distance, nearly the same Vs30 — and measure how far apart they are. Whatever
   that number is, no model with these columns can do better than it.

And one that is bigger than a semester. Sigma has barely moved in forty years of ground-motion
research, across an enormous increase in data and model complexity. The fit at the top of this
notebook gives {M['sigma']:.3f} in log10 units, a factor of {M['factor_model']:.1f} in shaking, and
*Your turn 7* splits it into a between-earthquake and a within-earthquake half. If that number is
close to irreducible with the observations that exist, then the interesting question is not which
model predicts shaking best, but what would have to be *measured* — and not modelled — to make it
smaller. Answering that with a number, even a rough one, would be a real result.
""")

ask(f"""
### ✏️ Your turn 8 — the first move

Before you close this notebook: in a few sentences, name the **one** measurement you would make
first, say what it would show if a flexible model genuinely improves ground-motion prediction, what
it would show if it does not, and name the number that would change your mind. Then make it, in the
cell below the prose.
""")

answer_prose(f"""
The one measurement I would make first is to **move the border**. My border split is a single
hold-out — train everywhere except Italy, score in Italy — and it is the only one of the four splits
where the sign changed. Before I explain that sign I should find out whether it is a property of
holding out a country at all, or a property of holding out *that* country. Holding out each of the
four largest countries in turn costs one line I have already written. The two explanations predict
different patterns rather than different sizes, which is what makes it worth doing first. If the
flexible models lose whenever a country hold-out takes away something they were leaning on, then
every country should reverse by roughly the same amount. If instead they lose because the held-out
records sit somewhere the model was never trained, only the countries whose records sit away from
the rest of the file should reverse, and the others should come out at nothing. The number that
would change my mind is the gap on the largest hold-out, Greece with
{COUNTRIES['GR']['n_test']:,} records: if it came out near Italy's, the flexible models really would
be losing at every border and I would have to prefer the first explanation.

It comes out at {COUNTRIES['GR']['gap']:+.3f}, and Turkey at {COUNTRIES['TR']['gap']:+.3f} — both
indistinguishable from no difference at all, on hold-outs that withhold an entire country. Only
Italy ({COUNTRIES['IT']['gap']:+.3f}) and Romania ({COUNTRIES['RO']['gap']:+.3f}) reverse. So the
border is not the mechanism, and neither is the forest memorising its stations: my station split
already withheld **every** station in the test set, share-a-station
{TABLE['by station']['share_station']:.3f}, and the forest was still ahead there by
{TABLE['by station']['forest_gap']:+.3f}. A model that was winning by remembering stations would
have lost that split first, before any border was involved.

What Italy has instead is a test set sitting outside the part of the feature space the models were
trained on. Its median source distance is {EDGE['median_r_test']:.0f} km against
{EDGE['median_r_train']:.0f} km for the records outside it, and
{EDGE['below_q05'] * 100:.0f}% of the Italian records are nearer than the 5th percentile of the
training set's distances. A forest is piecewise constant: every prediction is an average of
training records that fell in the same leaf, so it cannot return a value those records never
reached, and at the edge of the cloud every nearby leaf lies on one side. On the Italian test set
its highest prediction is {EDGE['forest_hi']:.2f} in log10 PGA, while the equation reaches
{EDGE['line_hi']:.2f} and the recordings themselves reach {EDGE['actual_hi']:.2f} — the forest
simply stops, a factor of {10 ** (EDGE['line_hi'] - EDGE['forest_hi']):.1f} in shaking short of the
records it is being scored on. Extended past its data a straight line keeps going, and here it goes
in the right direction — not because it learned to, but because somebody built the direction into
its shape. The forest's residual scatter on those records is {EDGE['sd_forest']:.3f} against the
equation's {EDGE['sd_line']:.3f}, and it is not a constant offset; it is that flattening.

I should not fold Romania into the same explanation, and this is where I would stop and say so.
Romania reverses by nearly the same amount, but its records are not near-field — their median
distance is {COUNTRIES['RO']['median_r']:.0f} km. What is unusual about them is a variable the
models cannot see at all. Most of Romania's records come from intermediate-depth earthquakes in the
Vrancea zone: their median depth is {COUNTRIES['RO']['median_depth']:.0f} km against
{COUNTRIES['RO']['median_depth_train']:.0f} km for everything else, and depth is not one of my
three features. There the *equation* scores
{COUNTRIES['RO']['line']:.3f} — worse than guessing the average — so both models have failed and
the forest has merely failed harder. Two hold-outs reversing for two different reasons is exactly
why I would not claim to have found the mechanism, only to have ruled one out.

So the conclusion I will defend is narrower than "the forest memorised the site terms" and better
supported: a model with more freedom and no more information cannot leave the region it was trained
in, while a parametric equation can, because its shape carries physics nobody had to learn from
this dataset. For this particular border the loss holds in both directions — training in Italy and
testing everywhere else gives {REVERSE['gap']:+.3f} — but that is one border, and Greece and Turkey
say I should not generalise from it. What makes me doubt the flexible models in advance is *Your
turn 7*: sigma splits into a between-earthquake half and a within-earthquake half, and neither is
something these three columns can explain — the first needs to know how this particular rupture
went, the second what the ground is like along the actual path. So my report is not "machine
learning does not work here" but "nothing works better here until somebody measures something new",
which is a result about the data rather than about the models.
""")

answer(f"""
for country in ["GR", "IT", "TR", "RO"]:
    is_train = (shaking["st_nation_code"] != country).values
    line = held_out_r2(LinearRegression(), is_train)
    trees = held_out_r2(forest, is_train)
    low, high = X[is_train].quantile(0.05), X[is_train].quantile(0.95)
    outside = ((X[~is_train] < low) | (X[~is_train] > high)).any(axis=1)
    print(f"hold out {{country}}   test {{(~is_train).sum():>5}}"
          f"   median R {{shaking['R'][~is_train].median():>6.1f}} km"
          f"   median depth {{shaking['ev_depth_km'][~is_train].median():>5.1f}} km"
          f"   outside the training 5-95% band {{outside.mean():.3f}}"
          f"   equation {{line:.3f}}   forest {{trees:.3f}}   forest - equation {{trees - line:+.3f}}")

train_in_italy = (shaking["st_nation_code"] == "IT").values
print(f"the same border the other way — train in Italy, test outside:"
      f"   equation {{held_out_r2(LinearRegression(), train_in_italy):.3f}}"
      f"   forest {{held_out_r2(forest, train_in_italy):.3f}}")

is_train = splits["across a border"]
predicted_line = held_out_predictions(LinearRegression(), is_train)
predicted_forest = held_out_predictions(forest, is_train)
edge = shaking["log_r"][is_train].quantile(0.05)

print("share of the Italian records nearer than the training set's 5th percentile of log distance:",
      round((shaking["log_r"][~is_train] < edge).mean(), 3))
print("highest value reached on the Italian records, log10 PGA — the recordings",
      round(y[~is_train].max(), 2), " the equation", round(predicted_line.max(), 2),
      " the forest", round(predicted_forest.max(), 2))
print("The forest cannot predict a value its training records never reached, and the Italian",
      "records are nearer and shake harder than most of what it was trained on, so it flattens",
      "exactly where the equation keeps going.")
print("Romania is not the same story and I will not report it as one: its records are not",
      "near-field, and what is unusual about them is depth, which is not a feature. Both models",
      "score below the trivial baseline there.")
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

    print(f"executing {sol_path.name} ...")
    r = weekkit.execute(sol_path, timeout=1800)
    if r.returncode:
        print(r.stderr[-4000:])
        sys.exit("the solution did not execute")

    (OUT / f"{SLUG}.ipynb").write_text(json.dumps(stu, indent=1) + "\n")

    for f in (sol_path, OUT / f"{SLUG}.ipynb"):
        nb = json.loads(f.read_text())
        track_ids(nb["cells"])
        f.write_text(json.dumps(nb, indent=1))

    print(f"wrote {SLUG}.ipynb ({len(CELLS)} cells) and the executed solution")
    print(f"cache: data/{ESM_CACHE} "
          f"({(ROOT / 'data' / ESM_CACHE).stat().st_size / 1e6:.2f} MB)")

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
