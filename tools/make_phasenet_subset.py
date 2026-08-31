#!/usr/bin/env python
"""Build the week-13 PhaseNet dataset: a notebook-sized subset of gs://quakeflow_dataset/NCEDC.

INSTRUCTOR-SIDE, run once. Students never touch GCS — they load the .npz it writes.

Follows Zhu & Beroza (2019), "PhaseNet: a deep-neural-network-based seismic arrival-time
picking method", GJI 216, 261-273. Same task, same labels, same 100 Hz 3-component input;
only the size is reduced so a CPU notebook can train on it in class.

Selection: 3-component traces carrying BOTH a manual P and a manual S pick, S-P under 12 s
(local events), P placed at a RANDOM offset in the window so the network cannot cheat on
position. Each trace is normalised per component, as PhaseNet does.
"""
import argparse, os, random, subprocess, tempfile
import numpy as np, pyarrow.parquet as pq

BUCKET = "gs://quakeflow_dataset/NCEDC/waveform_parquet"
NPTS, SR = 2048, 100          # 20.48 s at 100 Hz
COLS = ["waveform", "p_phase_index", "s_phase_index", "component", "snr", "event_magnitude",
        "distance_km", "station", "network", "event_id", "p_phase_polarity", "event_depth_km"]


def days(seed=0, n=40):
    """Spread the sample over years and seasons rather than one contiguous block."""
    rng = random.Random(seed)
    out = [(y, f"{rng.randint(1, 365):03d}") for y in range(2005, 2025) for _ in range(n // 10 + 1)]
    rng.shuffle(out)          # interleave years, or the target is met before the recent ones
    return out[:n]


def harvest(y, d, rng, tmp):
    dst = os.path.join(tmp, f"{y}_{d}.parquet")
    if subprocess.run(["gsutil", "-q", "cp", f"{BUCKET}/{y}/{d}.parquet", dst],
                      capture_output=True).returncode:
        return None
    try:
        tab = pq.read_table(dst, columns=COLS)
        # the waveform column is a nested fixed_size_list; convert it in bulk, not per row
        wf = np.array(tab["waveform"].to_pylist(), dtype=np.float32)   # (n, 3, 12288)
        t = tab.drop(["waveform"]).to_pandas()
    finally:
        os.path.exists(dst) and os.remove(dst)

    keep = []
    for i, r in enumerate(t.itertuples()):
        p, s = r.p_phase_index, r.s_phase_index
        if p is None or s is None or np.isnan(p) or np.isnan(s):     continue
        if r.component != "ENZ":                                     continue
        p, s = int(p), int(s)
        if not (0.5 * SR < s - p < 12 * SR):                         continue
        off = rng.randint(int(0.15 * NPTS), int(0.45 * NPTS))        # P lands anywhere in 15-45%
        a = p - off
        if a < 0 or a + NPTS > 12288 or s - a >= NPTS - 50:          continue
        w = wf[i][:, a:a + NPTS]
        if w.shape != (3, NPTS) or not np.isfinite(w).all():         continue
        sd = w.std(axis=1, keepdims=True)
        if (sd == 0).any():                                          continue
        # Clip after normalising. A handful of traces carry instrument glitches reaching 1e7 std;
        # they do not raise, they just make the picking net converge to zero accuracy with a
        # falling loss and no error message. Verified: unclipped 0.0% within 0.5 s, clipped 90.8%.
        w = np.clip(w / sd, -10, 10)
        keep.append((w, p - a, s - a, r.event_magnitude, r.distance_km, r.snr,
                     r.p_phase_polarity, r.event_depth_km, f"{r.network}.{r.station}", r.event_id))
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2000)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--out", default="data/phasenet_ncedc.npz")
    a = ap.parse_args()
    rng, rows = random.Random(0), []
    with tempfile.TemporaryDirectory() as tmp:
        for y, d in days(n=a.days):
            if len(rows) >= a.target:
                break
            got = harvest(y, d, rng, tmp)
            if got is None:
                print(f"  {y}/{d}  unavailable"); continue
            rows += got
            print(f"  {y}/{d}  +{len(got):4d}   total {len(rows)}")
    rng.shuffle(rows); rows = rows[:a.target]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(
        a.out,
        waveform=np.stack([r[0] for r in rows]).astype(np.float32),
        p_index=np.array([r[1] for r in rows], np.int16),
        s_index=np.array([r[2] for r in rows], np.int16),
        magnitude=np.array([r[3] for r in rows], np.float32),
        distance_km=np.array([r[4] for r in rows], np.float32),
        snr=np.array([r[5] for r in rows], np.float32),
        polarity=np.array([r[6] if isinstance(r[6], str) and r[6] in ("U", "D") else ""
                           for r in rows]),
        depth_km=np.array([r[7] for r in rows], np.float32),
        station=np.array([r[8] for r in rows]),
        event_id=np.array([r[9] for r in rows]),
        sampling_rate=np.array(SR),
    )
    print(f"\n{len(rows)} traces -> {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
