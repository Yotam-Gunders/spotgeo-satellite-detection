from PIL import Image
import numpy as np
from pathlib import Path
from collections import Counter
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detect import detect

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "SpotGEOv2"


def build_tracks(P, R=60, tol=1.5, min_hits=3):
    tracks = {}
    for i in range(5):
        for j in range(i + 1, 5):
            for a in P[i]:
                for b in P[j]:
                    v = (b - a) / (j - i)
                    if np.hypot(v[0], v[1]) > R:
                        continue
                    p0 = a - i * v
                    members = []
                    for f in range(5):
                        pred = p0 + f * v
                        d = np.hypot(P[f][:, 0] - pred[0], P[f][:, 1] - pred[1])
                        if d.min() < tol:
                            members.append((f, int(d.argmin())))
                    if len(members) >= min_hits:
                        key = (round(p0[0] / 2), round(p0[1] / 2), round(v[0]), round(v[1]))
                        if key not in tracks or len(members) > len(tracks[key][2]):
                            tracks[key] = (p0, v, members)
    return list(tracks.values())


# step 1: candidates per frame
seq = 1
P = []
for f in range(1, 6):
    img = np.array(Image.open(DATA / "train" / str(seq) / f"{f}.png"))
    cands = detect(img)
    P.append(np.array([[c["x"], c["y"]] for c in cands]))
print("candidates per frame:", [len(p) for p in P])

# step 2: all hypotheses, find the star velocity
tracks = build_tracks(P)
V = np.array([v for p0, v, m in tracks])
bins = Counter((round(v[0] / 4) * 4, round(v[1] / 4) * 4) for v in V)
v_star = np.array(bins.most_common(1)[0][0])
print("tracks:", len(tracks), " star velocity:", v_star)

# step 3: every candidate that belongs to a star track is a star. remove it.
star_members = set()
for p0, v, members in tracks:
    if np.hypot(*(v - v_star)) <= 10:
        star_members.update(members)
P2 = [np.array([p for idx, p in enumerate(P[f]) if (f, idx) not in star_members]) for f in range(5)]
print("after removing stars:", [len(p) for p in P2])

# step 4: hypotheses again, on what is left
sats = build_tracks(P2)
print("remaining tracks:", len(sats))
for p0, v, members in sorted(sats, key=lambda t: -len(t[2])):
    print(p0.round(1), v.round(1), len(members))

# step 5: merge duplicates (close start, close velocity)
merged = []
for p0, v, m in sorted(sats, key=lambda t: -len(t[2])):
    dup = any(np.hypot(*(p0 - q0)) < 4 and np.hypot(*(v - w)) < 3 for q0, w, _ in merged)
    if not dup:
        merged.append((p0, v, m))

# step 6: weak tracks need a partner with the same velocity
final = []
for p0, v, m in merged:
    agree = sum(1 for q0, w, _ in merged if np.hypot(*(v - w)) < 3)
    if len(m) >= 4 or agree >= 2:
        final.append((p0, v, m))

print("final:", len(final))
for p0, v, m in final:
    print(p0.round(1), v.round(1), len(m))