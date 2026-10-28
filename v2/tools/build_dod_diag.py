"""Build a small standalone Kaggle notebook that diagnoses DOD-H / DOD-O label-signal alignment."""
import nbformat as nbf, sys
cells = []
md = nbf.v4.new_markdown_cell; code = nbf.v4.new_code_cell
cells.append(md("""# DOD alignment diagnostic (≈5 min, CPU is fine, Internet ON)
Checks, for a few DOD-H and DOD-O records: the h5 structure and sampling rate, signal length vs hypnogram length,
agreement between the 5-scorer consensus and the hypnogram stored inside the h5 file at different time lags,
and the correlation between slow-wave (0.5–2 Hz) power and the consensus N3 labels at different lags.
Nothing is trained; the output is only printed. Send the printed output back."""))
cells.append(code('''import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "remotezip", "h5py"], check=False)
import io, json, re, zipfile, urllib.request, numpy as np, h5py
from remotezip import RemoteZip
from scipy.signal import butter, sosfiltfilt, welch
ZIP = {"dodh": "https://zenodo.org/records/15900394/files/dodh.zip?download=1",
       "dodo": "https://zenodo.org/records/15900394/files/dodo.zip?download=1"}
data = None
for br in ("master", "main"):
    try:
        data = urllib.request.urlopen(f"https://codeload.github.com/Dreem-Organization/dreem-learning-evaluation/zip/refs/heads/{br}", timeout=120).read(); break
    except Exception as e:
        print("scorers", br, e)
SC = {"dodh": {}, "dodo": {}}
with zipfile.ZipFile(io.BytesIO(data)) as zf:
    for n in zf.namelist():
        m = re.search(r"scorers/(dod[ho])/([^/]+)/([^/]+)\\.json$", n)
        if m:
            SC[m.group(1)].setdefault(m.group(3), {})[m.group(2)] = json.loads(zf.read(n))
print({k: len(v) for k, v in SC.items()}, "records with scorer hypnograms")

def consensus(hyps):
    H = np.array([h for _, h in sorted(hyps.items())])
    out = []
    for col in H.T:
        v = col[col >= 0]
        out.append(np.bincount(v, minlength=5).argmax() if len(v) else -1)
    return np.array(out)

def show(h5):
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"    {name:40s} shape={obj.shape} dtype={obj.dtype} attrs={dict(obj.attrs)}")
    h5.visititems(visit)
    print("    root attrs:", dict(h5.attrs))
'''))
cells.append(code('''def diagnose(ds, n_records=4):
    rz = RemoteZip(ZIP[ds])
    members = sorted(n for n in rz.namelist() if n.endswith(".h5") and "__MACOSX" not in n and not n.split("/")[-1].startswith("._"))
    print(f"\\n=== {ds}: {len(members)} h5 files in zip")
    for m in members[:n_records]:
        rid = m.split("/")[-1][:-3]
        p = rz.extract(m, "/tmp/dod")
        with h5py.File(p, "r") as h5:
            print(f"\\n--- {ds} {rid}")
            show(h5)
            eeg = [k for k in ("signals/eeg/C4_M1", "signals/eeg/C3_M2") if k in h5]
            k = eeg[0]
            fs = float(np.ravel(h5[k].attrs.get("fs", h5["signals/eeg"].attrs.get("fs", 250)))[0])
            x = np.asarray(h5[k][()], float)
            print(f"    channel {k} raw shape {x.shape}; fs={fs}; p1/p50/p99 |x| =", np.percentile(np.abs(x), [1, 50, 99]).round(3))
            x = x.ravel()
            h5hyp = np.asarray(h5["hypnogram"][()]).astype(int).ravel() if "hypnogram" in h5 else None
        cons = consensus(SC[ds][rid]) if rid in SC[ds] else None
        n_sig = len(x) / (30 * fs)
        print(f"    signal epochs={n_sig:.1f}; consensus len={None if cons is None else len(cons)}; h5 hypnogram len={None if h5hyp is None else len(h5hyp)}")
        if cons is not None and h5hyp is not None:
            best = []
            for lag in range(-120, 121):
                a = cons[max(0, lag):len(cons) + min(0, lag)]
                b = h5hyp[max(0, -lag):len(h5hyp) + min(0, -lag)]
                L = min(len(a), len(b)); a, b = a[:L], b[:L]; mm = (a >= 0) & (b >= 0)
                best.append((float((a[mm] == b[mm]).mean()) if mm.any() else 0, lag))
            best.sort(reverse=True)
            agree0 = [v for v, l in best if l == 0][0]
            print(f"    consensus vs h5 hypnogram agreement: lag0={agree0:.3f}; best={best[0][0]:.3f} at lag {best[0][1]} epochs")
        if cons is not None:
            sos = butter(4, [0.5, 2.0], btype="band", fs=fs, output="sos")
            sw = sosfiltfilt(sos, x)
            ne = int(n_sig)
            pw = np.log(np.mean(sw[: ne * int(30 * fs)].reshape(ne, -1) ** 2, 1) + 1e-6)
            y = (cons == 3).astype(float)
            res = []
            for lag in range(-120, 121, 2):
                a = pw[max(0, lag):]; b = y[max(0, -lag):]
                L = min(len(a), len(b))
                if L > 100 and b[:L].std() > 0:
                    res.append((float(np.corrcoef(a[:L], b[:L])[0, 1]), lag))
            res.sort(reverse=True)
            r0 = [v for v, l in res if l == 0]
            print(f"    corr(slow-wave power, consensus N3): lag0={r0[0] if r0 else float('nan'):.3f}; best={res[0][0]:.3f} at lag {res[0][1]} epochs"
                  f" ({res[0][1] * 30 / 60:+.0f} min);  N3 epochs={int(y.sum())}")
diagnose("dodh", 3)
diagnose("dodo", 5)
'''))
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "kaggle": {"isInternetEnabled": True}}
nbf.write(nb, sys.argv[1])
print("wrote", sys.argv[1])
