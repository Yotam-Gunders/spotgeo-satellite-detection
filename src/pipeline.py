from PIL import Image
import numpy as np
from pathlib import Path
from collections import Counter
from detect_fast import residual, find_blobs
from scipy.ndimage import label, find_objects, binary_dilation, uniform_filter, maximum_filter
from scipy.spatial import cKDTree


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
                        if len(P[f]) == 0:
                            continue
                        pred = p0 + f * v
                        d = np.hypot(P[f][:, 0] - pred[0], P[f][:, 1] - pred[1])
                        if d.min() < tol:
                            members.append((f, int(d.argmin())))
                    if len(members) >= min_hits:
                        key = (round(p0[0] / 2), round(p0[1] / 2), round(v[0]), round(v[1]))
                        if key not in tracks or len(members) > len(tracks[key][2]):
                            tracks[key] = (p0, v, members)
    return list(tracks.values())


def build_tracks_fast(P, R=60, tol=1.5, min_hits=3):
    trees = [cKDTree(p) if len(p) else None for p in P]
    tracks = {}
    for i in range(5):
        for j in range(i + 1, 5):
            if trees[i] is None or trees[j] is None:
                continue
            A, B = [], []
            for a, near in zip(P[i], trees[j].query_ball_point(P[i], r=R * (j - i))):
                for b in sorted(near):
                    A.append(a)
                    B.append(P[j][b])
            if not A:
                continue
            A, B = np.array(A), np.array(B)
            V = (B - A) / (j - i)
            ok = np.hypot(V[:, 0], V[:, 1]) <= R
            A, V = A[ok], V[ok]
            P0 = A - i * V

            hits = np.zeros(len(P0), int)
            idx = []
            for f in range(5):
                if trees[f] is None:
                    idx.append(np.full(len(P0), -1))
                    continue
                d, k = trees[f].query(P0 + f * V, distance_upper_bound=tol)
                good = d < tol
                hits += good
                idx.append(np.where(good, k, -1))

            for n in np.nonzero(hits >= min_hits)[0]:
                members = [(f, int(idx[f][n])) for f in range(5) if idx[f][n] >= 0]
                p0, v = P0[n], V[n]
                key = (round(p0[0] / 2), round(p0[1] / 2), round(v[0]), round(v[1]))
                if key not in tracks or len(members) > len(tracks[key][2]):
                    tracks[key] = (p0, v, members)
    return list(tracks.values())


def refine_v(P, v0, r=4.0):
    D = []
    for f in range(4):
        for a in P[f]:
            d = P[f + 1] - a
            near = np.hypot(*(d - v0).T) < r
            D.extend(d[near])
    return np.median(D, axis=0) if D else v0


def star_points(P, v_star, tol=2.5):
    star = set()
    for f in range(5):
        for idx, p in enumerate(P[f]):
            for g in range(5):
                if g == f or len(P[g]) == 0:
                    continue
                d = np.hypot(*(P[g] - (p + (g - f) * v_star)).T)
                if d.min() < tol:
                    star.add((f, idx))
                    break
    return star


def signal(resid, noise, p):
    H, W = resid.shape
    x = min(max(int(round(p[0])), 1), W - 2)
    y = min(max(int(round(p[1])), 1), H - 2)
    return resid[y - 1:y + 2, x - 1:x + 2].sum() / (3 * noise)


def star_mask(z):
    lab, n = label(z > 3)
    mask = np.zeros(z.shape, bool)
    for i, (ys, xs) in enumerate(find_objects(lab)):
        if max(ys.stop - ys.start, xs.stop - xs.start) > 6:
            mask[ys, xs] |= lab[ys, xs] == i + 1
    return binary_dilation(mask, iterations=2)


def stack(Z, v):
    H, W = Z[0].shape
    acc = np.zeros((H, W))
    cnt = np.zeros((H, W))
    for f in range(5):
        dx, dy = int(round(f * v[0])), int(round(f * v[1]))
        x0, x1 = max(0, -dx), min(W, W - dx)
        y0, y1 = max(0, -dy), min(H, H - dy)
        if x1 <= x0 or y1 <= y0:
            continue
        acc[y0:y1, x0:x1] += Z[f][y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        cnt[y0:y1, x0:x1] += 1
    return acc, cnt


def search(Z, v, thr=11.0):
    acc, cnt = stack(Z, v)
    s = uniform_filter(acc, 3) * 9 / np.sqrt(45)
    s[cnt < 5] = 0
    peaks = (s == maximum_filter(s, 5)) & (s > thr)
    hits = []
    for y, x in zip(*np.nonzero(peaks)):
        p0 = np.array([x, y], float)
        b = [signal(Z[f], 1.0, p0 + f * v) for f in range(5)]
        if sorted(b)[1] >= 1 and max(b) <= 0.6 * sum(b):
            hits.append(p0)
    return hits


def find_satellites(seq, split="train"):
    P, R, N, Z = [], [], [], []
    for f in range(1, 6):
        img = np.array(Image.open(DATA / split / str(seq) / f"{f}.png"))
        resid, noise = residual(img)
        cands = find_blobs(resid, noise, k=4)
        P.append(np.array([[c["x"], c["y"]] for c in cands]).reshape(-1, 2))
        R.append(resid)
        N.append(noise)
        z = np.clip(resid / noise, -3, 3)
        z[star_mask(resid / noise)] = 0
        Z.append(z)

    tracks = build_tracks_fast(P)
    if len(tracks) == 0:
        return []

    V = np.array([v for p0, v, m in tracks])
    bins = Counter((round(v[0] / 4) * 4, round(v[1] / 4) * 4) for v in V)
    v_star = np.array(bins.most_common(1)[0][0])
    v_star = refine_v(P, v_star)

    star_members = star_points(P, v_star)
    P2 = [np.array([p for idx, p in enumerate(P[f]) if (f, idx) not in star_members]).reshape(-1, 2) for f in range(5)]

    sats = build_tracks_fast(P2, tol=2.0)
    sats = [t for t in sats if 18 <= np.hypot(*(t[1] - v_star)) <= 32]

    def inside(p0, v):
        return all(0 <= (p0 + f * v)[0] <= 639 and 0 <= (p0 + f * v)[1] <= 479 for f in range(5))

    sats = [t for t in sats if inside(t[0], t[1])]

    merged = []
    for p0, v, m in sorted(sats, key=lambda t: -len(t[2])):
        dup = any(np.mean([np.hypot(*((p0 + f * v) - (q0 + f * w))) for f in range(5)]) < 5 for q0, w, m2 in merged)
        if not dup:
            merged.append((p0, v, m))

    final = []
    for p0, v, m in merged:
        voted = {f for f, idx in m}
        sig = [signal(R[f], N[f], p0 + f * v) for f in range(5) if f not in voted]
        if len(m) == 5 or min(sig) >= 2:
            final.append((p0, v, m))
    if final:
        vc = np.median([v for p0, v, m in final], axis=0)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                w = vc + np.array([dx, dy])
                for q in search(Z, w):
                    if not any(np.mean([np.hypot(*((q + f * w) - (p0 + f * v))) for f in range(5)]) < 6 for p0, v, m in
                               final):
                        final.append((q, w, []))
    return final