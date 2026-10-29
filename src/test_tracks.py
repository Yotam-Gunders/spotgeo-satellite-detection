"""Check that build_tracks_fast returns the same hypotheses as the reference build_tracks.

Start points and velocities must be exactly equal. Members must lie in the same frames; the candidate
index inside a frame may differ when two candidates are at exactly the same distance (a tie).
"""
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
    assert len(old) == len(new), f"sequence {seq}: {len(old)} vs {len(new)} hypotheses"
    for a, b in zip(old, new):
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), f"sequence {seq}: track differs"
        assert [f for f, _ in a[2]] == [f for f, _ in b[2]], f"sequence {seq}: member frames differ"
    print(seq, len(old), "hypotheses identical")
