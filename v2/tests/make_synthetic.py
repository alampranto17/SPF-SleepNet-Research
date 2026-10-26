"""Synthetic EDF files mimicking the ISRUC and Sleep-EDF folder layouts.

FOR CODE TESTING ONLY -- signals are simulated; any metric computed on them is
meaningless and must never be reported as a result.
"""
import os
import sys

import mne
import numpy as np

rng = np.random.default_rng(7)


def hypnogram(n_epochs, n3_scale=1.0):
    seq = [0] * 20
    cycle_n3 = [int(n3_scale * x) for x in (45, 30, 15, 6, 0)]
    for c, n3 in enumerate(cycle_n3):
        seq += [1] * 6 + [2] * 30 + [3] * n3 + [2] * 15 + [4] * (12 + 6 * c)
        if c % 2:
            seq += [0] * 3
        if len(seq) >= n_epochs:
            break
    seq += [0] * max(0, n_epochs - len(seq))
    return np.array(seq[:n_epochs])


def simulate(stages, fs, amp_scale=1.0, cycle_decay=True):
    n = int(30 * fs)
    t = np.arange(n) / fs
    out = []
    n3_seen = 0
    for i, s in enumerate(stages):
        x = rng.normal(0, 4, n)
        if s == 0:
            x += 18 * np.sin(2 * np.pi * rng.uniform(9, 11) * t) + rng.normal(0, 8, n)
        elif s == 1:
            x += 22 * np.sin(2 * np.pi * rng.uniform(4, 7) * t)
        elif s == 2:
            x += 18 * np.sin(2 * np.pi * rng.uniform(2, 6) * t)
            x += rng.uniform(0, 70) * np.sin(2 * np.pi * rng.uniform(0.7, 1.6) * t)  # overlap with N3
            for _ in range(rng.integers(1, 4)):  # spindles
                c = rng.uniform(2, 28)
                env = np.exp(-((t - c) / 0.4) ** 2)
                x += 25 * env * np.sin(2 * np.pi * 13 * t)
            if rng.random() < 0.5:  # K-complex
                c = rng.uniform(3, 27)
                x += -90 * np.exp(-((t - c) / 0.25) ** 2) + 50 * np.exp(-((t - c - 0.5) / 0.3) ** 2)
        elif s == 3:
            n3_seen += 1
            decay = np.exp(-i / len(stages) * 1.5) if cycle_decay else 1.0
            a = amp_scale * rng.uniform(30, 130) * decay
            f = rng.uniform(0.7, 1.6)
            x += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) + \
                0.4 * a * np.sin(2 * np.pi * 2 * f * t)
        elif s == 4:
            x += 15 * np.sin(2 * np.pi * rng.uniform(5, 7) * t)
        if rng.random() < 0.03:  # occasional movement artifact
            c = int(rng.uniform(0, n - fs))
            x[c:c + int(fs)] += rng.normal(0, 200, int(fs))
        out.append(x)
    return np.concatenate(out) * 1e-6  # volts


def write_isruc(root, sid, chan_style, n_epochs=520, amp=1.0):
    d = os.path.join(root, str(sid))
    os.makedirs(d, exist_ok=True)
    fs = 200
    st = hypnogram(n_epochs, n3_scale=rng.uniform(0.7, 1.2))
    sig = simulate(st, fs, amp)
    if chan_style == "A":
        names = ["LOC-A2", "ROC-A1", "F3-A2", "C3-A2", "O1-A2", "F4-A1", "C4-A1", "O2-A1"]
    else:
        names = ["E1-M2", "E2-M1", "F3-M2", "C3-M2", "O1-M2", "F4-M1", "C4-M1", "O2-M1"]
    data = np.vstack([rng.normal(0, 20e-6, len(sig)) if "C4" not in nm else sig for nm in names])
    raw = mne.io.RawArray(data, mne.create_info(names, fs, "eeg"), verbose="error")
    tmp = os.path.join(d, f"{sid}.edf")
    mne.export.export_raw(tmp, raw, fmt="edf", overwrite=True, verbose="error")
    os.replace(tmp, os.path.join(d, f"{sid}.rec"))
    isruc_code = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5}
    lab1 = [isruc_code[s] for s in st] + [0] * 30   # ISRUC files carry extra epochs
    lab2 = list(lab1)
    for i in range(1, len(st) - 1):                  # scorer 2 disagrees at N2/N3 borders
        if st[i] in (2, 3) and st[i - 1] != st[i] and rng.random() < 0.6:
            lab2[i] = 3 if st[i] == 2 else 2
    for name, lab in (("_1", lab1), ("_2", lab2)):
        with open(os.path.join(d, f"{sid}{name}.txt"), "w") as f:
            f.write("\n".join(map(str, lab)) + "\n")


def _stage_runs(st):
    i = 0
    while i < len(st):
        j = i
        while j < len(st) and st[j] == st[i]:
            j += 1
        yield i, j, st[i]
        i = j


def write_hmc(root, k, age):
    fs = 256
    st = hypnogram(600, rng.uniform(0.7, 1.2))
    sig = simulate(st, fs, 1.0 - 0.005 * (age - 25))
    names = ["EEG F4-M1", "EEG C4-M1", "EEG O2-M1", "EEG C3-M2", "EOG E1-M2"]
    data = np.vstack([sig if nm == "EEG C4-M1" else rng.normal(0, 15e-6, len(sig)) for nm in names])
    raw = mne.io.RawArray(data, mne.create_info(names, fs, "eeg"), verbose="error")
    key = f"SN{k:03d}"
    mne.export.export_raw(os.path.join(root, f"{key}.edf"), raw, fmt="edf", overwrite=True,
                          verbose="error")
    m = {0: "Sleep stage W", 1: "Sleep stage N1", 2: "Sleep stage N2", 3: "Sleep stage N3",
         4: "Sleep stage R"}
    on, du, de = [0.0], [0.0], ["Lights off@@EEG F4-A1"]
    for i, j, s_ in _stage_runs(st):
        for e in range(i, j):                       # HMC scores epoch by epoch
            on.append(e * 30.0); du.append(30.0); de.append(m[s_])
    hraw = mne.io.RawArray(np.zeros((1, len(st) * 30)), mne.create_info(["dummy"], 1, "misc"),
                           verbose="error")
    hraw.set_annotations(mne.Annotations(on, du, de))
    mne.export.export_raw(os.path.join(root, f"{key}_sleepscoring.edf"), hraw, fmt="edf",
                          overwrite=True, verbose="error")


def write_dod(root, scorers_root, ds, rid):
    import h5py, json as _json
    fs = 250
    st = hypnogram(560, rng.uniform(0.7, 1.2))
    sig = simulate(st, fs, 0.9)                     # volts (tests unit guard)
    os.makedirs(root, exist_ok=True)
    with h5py.File(os.path.join(root, f"{rid}.h5"), "w") as h5:
        g = h5.create_group("signals/eeg")
        for nm in ("C3_M2", "F3_F4", "O1_M2"):
            d = g.create_dataset(nm, data=(sig if nm == "C3_M2" else rng.normal(0, 1e-5, len(sig))))
            d.attrs["fs"] = fs
        h5.create_dataset("hypnogram", data=st)
    for k in range(1, 6):                           # 5 scorers disagree near N2/N3 borders
        lab = st.copy()
        for i in range(1, len(st) - 1):
            if st[i] in (2, 3) and (st[i - 1] != st[i] or st[i + 1] != st[i]) and rng.random() < 0.45:
                lab[i] = 3 if st[i] == 2 else 2
        d = os.path.join(scorers_root, ds, f"scorer_{k}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{rid}.json"), "w") as f:
            _json.dump([int(x) for x in lab], f)


def main(root):
    import pandas as pd
    i1, i2, i3, hm = (os.path.join(root, x) for x in ("ISRUC-Sleep-I", "ISRUC-Sleep-II",
                                                     "ISRUC-Sleep-III", "HMC"))
    for d in (i1, i2, i3, hm):
        os.makedirs(d, exist_ok=True)
    for sid in range(1, 9):
        write_isruc(i1, sid, "A" if sid <= 2 else "M")
    pd.DataFrame({"Subject": [f"Subject {i}" for i in range(1, 9)],
                  "Age": rng.integers(20, 80, 8), "Sex": rng.choice(["M", "F"], 8),
                  "Diagnosis": "synthetic"}).to_excel(
        os.path.join(i1, "Details_subgroup_I_Submission.xlsx"), index=False)
    for sid in range(1, 4):                         # ISRUC-II: two sessions per subject
        for ses in (1, 2):
            write_isruc(os.path.join(i2, str(sid)), ses, "M")
    for sid in range(1, 3):
        write_isruc(i3, sid, "M")
    for k in range(1, 9):
        write_hmc(hm, k, int(rng.integers(25, 80)))
    sc = os.path.join(root, "dod_scorers")
    import uuid
    for ds, folder, n in (("dodh", "DOD-H", 4), ("dodo", "DOD-O", 4)):
        for _ in range(n):
            write_dod(os.path.join(root, folder), sc, ds, str(uuid.UUID(int=int(rng.integers(1 << 62)))))
    print("synthetic data written to", root)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sqn2_synth")
