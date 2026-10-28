import numpy as np
from PIL import Image
from pipeline import build_tracks, build_tracks_fast, DATA
from detect_fast import residual, find_blobs

for seq in (1, 3, 30, 72, 144):
    P = []
    for f in range(1, 6):
        img = np.array(Image.open(DATA / "train" / str(seq) / f"{f}.png"))
        resid, noise = residual(img)
        P.append(np.array([[c["x"], c["y"]] for c in find_blobs(resid, noise, k=4)]).reshape(-1, 2))
    old = build_tracks(P)
    new = build_tracks_fast(P)
    same_tracks = all(np.allclose(a[0], b[0]) and np.allclose(a[1], b[1]) for a, b in zip(old, new))
    same_frames = all([f for f, _ in a[2]] == [f for f, _ in b[2]] for a, b in zip(old, new))
    print(seq, len(old) == len(new), same_tracks, same_frames)