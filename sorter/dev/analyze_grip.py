"""From the training data: which wrist-image region stays still while a brick is held,
and how large the change is with vs without a brick. Sets grip.roi and grip.hold_max_change."""
import glob, os, sys, numpy as np, pandas as pd, av, cv2
SORTER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SORTER)
import vision
S = glob.glob(os.path.expanduser("~/.local/share/physicalai/snapshots/snapshot_2026-11-14_23-00-13*"))[0]
eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(S + "/meta/episodes/**/*.parquet", recursive=True))])
data = pd.concat([pd.read_parquet(f, columns=["episode_index","frame_index","observation.state"]) for f in glob.glob(S + "/data/**/*.parquet", recursive=True)])
def frames(row, cam, idxs):
    k = f"videos/observation.images.{cam}"
    path = f"{S}/{k}/chunk-{int(row[k+'/chunk_index']):03d}/file-{int(row[k+'/file_index']):03d}.mp4"
    t0 = float(row[k+'/from_timestamp']); want = sorted(set(idxs)); out = {}
    with av.open(path) as c:
        st = c.streams.video[0]; c.seek(int((t0 + want[0]/30) / st.time_base), stream=st, backward=True)
        for fr in c.decode(st):
            i = round((fr.time - t0) * 30)
            if i in want and i not in out: out[i] = fr.to_ndarray(format="rgb24")
            if i > want[-1]: break
    return out
hold_maps, open_maps, hold_vals, open_vals = [], [], [], []
rng = np.random.default_rng(0)
for ep in rng.choice(np.arange(85, 400), 40, replace=False):
    row = eps[eps.episode_index == ep].iloc[0]
    st = np.stack(data[data.episode_index == ep].sort_values("frame_index")["observation.state"].values)
    g = st[:, 5]; closed = np.where(g < 8)[0]
    if len(closed) < 25: continue
    h0 = int(closed[0] + 3); h1 = int(min(closed[0] + 23, closed[-1]))
    o1 = int(closed[0] - 3); o0 = max(0, o1 - 20)   # approach, gripper still open, arm moving
    motion_h = np.abs(st[h1, :5] - st[h0, :5]).sum(); motion_o = np.abs(st[o1, :5] - st[o0, :5]).sum()
    f = frames(row, "wrist", [o0, o1, h0, h1])
    if len(f) < 4: continue
    gray = {i: cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), (9, 9), 0).astype(np.float32) for i, im in f.items()}
    hold_maps.append(np.abs(gray[h1] - gray[h0])); open_maps.append(np.abs(gray[o1] - gray[o0]))
    hold_vals.append((vision.grip_change(f[h0], f[h1]), motion_h)); open_vals.append((vision.grip_change(f[o0], f[o1]), motion_o))
H, O = np.mean(hold_maps, 0), np.mean(open_maps, 0)
print("episodes", len(hold_vals))
# coarse grid: where is hold-change low and open-change high
gh, gw = 12, 16
cell = lambda m: m.reshape(gh, m.shape[0] // gh, gw, m.shape[1] // gw).mean((1, 3))
ratio = cell(O) / (cell(H) + 1)
np.set_printoptions(precision=1, suppress=True, linewidth=200)
print("open/hold change ratio per cell (rows top->bottom):"); print(ratio)
print("hold mean change per cell:"); print(cell(H))
hv = np.array(hold_vals); ov = np.array(open_vals)
print("current ROI  hold: p50 %.1f p90 %.1f max %.1f | open (moving): p10 %.1f p50 %.1f min %.1f" % (
    np.median(hv[:, 0]), np.percentile(hv[:, 0], 90), hv[:, 0].max(), np.percentile(ov[:, 0], 10), np.median(ov[:, 0]), ov[:, 0].min()))
np.savez(os.path.join(SORTER, "samples", "grip_maps.npz"), H=H, O=O, hv=hv, ov=ov)
