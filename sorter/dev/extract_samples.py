"""Pull test frames from the training snapshot: wrist/webcam while holding, realsense at start and mid-episode."""
import glob, os, sys, numpy as np, pandas as pd, av, cv2
S = glob.glob(os.path.expanduser("~/.local/share/physicalai/snapshots/snapshot_2026-11-14_23-00-13*"))[0]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples", "ds")
os.makedirs(OUT, exist_ok=True)
eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(S + "/meta/episodes/**/*.parquet", recursive=True))])
data = pd.concat([pd.read_parquet(f, columns=["episode_index","frame_index","observation.state"]) for f in glob.glob(S + "/data/**/*.parquet", recursive=True)])
def frames(row, cam, idxs):
    k = f"videos/observation.images.{cam}"
    path = f"{S}/{k}/chunk-{int(row[k+'/chunk_index']):03d}/file-{int(row[k+'/file_index']):03d}.mp4"
    t0 = float(row[k+'/from_timestamp']); want = sorted(idxs); out = {}
    with av.open(path) as c:
        st = c.streams.video[0]; c.seek(int((t0 + want[0]/30) / st.time_base), stream=st, backward=True)
        for fr in c.decode(st):
            i = round((fr.time - t0) * 30)
            if i in want and i not in out: out[i] = fr.to_ndarray(format="bgr24")
            if i > want[-1]: break
    return out
for ep in [int(x) for x in sys.argv[1:]]:
    row = eps[eps.episode_index == ep].iloc[0]
    g = np.stack(data[data.episode_index == ep].sort_values("frame_index")["observation.state"].values)[:, 5]
    closed = np.where(g < 8)[0]
    hold = int(closed[0] + 8) if len(closed) else len(g) // 2
    for cam, idxs in (("wrist", [hold]), ("webcam", [0, hold]), ("realsense", [0, hold])):
        for i, img in frames(row, cam, idxs).items():
            cv2.imwrite(f"{OUT}/ep{ep:03d}_{cam}_{'start' if i == 0 else 'hold'}.png", img)
    print(ep, "hold frame", hold, flush=True)
