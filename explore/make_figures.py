import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Annulus
from PIL import Image
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pipeline import (find_satellites, build_tracks_fast, refine_v, star_points, star_mask, stack,
                      DATA, ROOT)
from detect_fast import residual, find_blobs

OUT = ROOT / "docs"
anno = json.load(open(DATA / "train_anno.json"))
GT = {}
for a in anno:
    GT.setdefault(a["sequence_id"], {})[a["frame"]] = a["object_coords"]


def load(seq):
    imgs = [np.array(Image.open(DATA / "train" / str(seq) / f"{f}.png")) for f in range(1, 6)]
    res = [residual(img) for img in imgs]
    P = [np.array([[c["x"], c["y"]] for c in find_blobs(r, n, k=4)]).reshape(-1, 2) for r, n in res]
    return imgs, res, P


def star_velocity(P):
    V = np.array([v for p0, v, m in build_tracks_fast(P)])
    bins = Counter((round(v[0] / 4) * 4, round(v[1] / 4) * 4) for v in V)
    return refine_v(P, np.array(bins.most_common(1)[0][0])), V


def show(ax, img, title, lo=50, hi=99.5):
    ax.imshow(img, cmap="gray", vmin=np.percentile(img, lo), vmax=np.percentile(img, hi))
    ax.set_title(title, fontsize=12)
    ax.axis("off")


# 1. background removal
def fig_background(seq=30):
    imgs, res, P = load(seq)
    resid, noise = res[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    show(axes[0], imgs[0], "raw frame: sky glow, clouds, stars, satellites", 1, 99.5)
    show(axes[1], np.clip(resid / noise, -3, 8), "after subtracting the 31x31 median background", 1, 99.9)
    fig.savefig(OUT / "step1_background.png", dpi=80, bbox_inches="tight", pad_inches=0.15)


# 2. per-point star test
def fig_star_test(seq=30):
    imgs, res, P = load(seq)
    v_star, _ = star_velocity(P)
    star = star_points(P, v_star)
    pts = P[0]
    is_star = np.array([(0, i) in star for i in range(len(pts))])
    gt = np.array(GT[seq][1])
    fig, ax = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
    show(ax, imgs[0], "")
    ax.plot(pts[is_star, 0], pts[is_star, 1], "x", color="tomato", ms=6, mew=1.2,
            label=f"star: has a partner at p + k * v_star ({is_star.sum()})")
    ax.plot(pts[~is_star, 0], pts[~is_star, 1], "o", mfc="none", mec="lime", ms=8, mew=1.4,
            label=f"kept for satellite search ({(~is_star).sum()})")
    ax.plot(gt[:, 0], gt[:, 1], "s", mfc="none", mec="cyan", ms=14, mew=1.2, label="ground truth satellites")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_title(f"Candidates in frame 1 (sequence {seq}) after the per-point star test", fontsize=12)
    fig.savefig(OUT / "step3_star_test.png", dpi=80, bbox_inches="tight", pad_inches=0.15)


# 3. the physics: hypothesis velocities
def fig_velocities(seq=30):
    imgs, res, P = load(seq)
    v_star, V = star_velocity(P)
    final = find_satellites(seq)
    fig, ax = plt.subplots(figsize=(7.5, 7), constrained_layout=True)
    ax.plot(V[:, 0], V[:, 1], ".", color="0.6", ms=2, alpha=0.4, label="all track hypotheses")
    ax.add_patch(Annulus(v_star, 32, 14, color="orange", alpha=0.25,
                         label="search ring: 18-32 px/frame from the stars"))
    ax.plot(*v_star, "*", color="tomato", ms=18, label="star velocity (densest cluster)")
    sv = np.array([v for p0, v, m in final]).reshape(-1, 2)
    ax.plot(sv[:, 0], sv[:, 1], "o", mfc="none", mec="blue", ms=11, mew=1.5, label=f"{len(sv)} confirmed satellites (all at the same velocity)")
    ax.set_xlim(-60, 60)
    ax.set_ylim(60, -60)
    ax.set_aspect("equal")
    ax.set_xlabel("vx [px/frame]")
    ax.set_ylabel("vy [px/frame]")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(f"Velocity of every track hypothesis (sequence {seq})", fontsize=12)
    fig.savefig(OUT / "step2_velocities.png", dpi=80, bbox_inches="tight", pad_inches=0.15)


# 4. shift-and-add
def fig_stacking(seq=59, half=20):
    imgs, res, P = load(seq)
    Z = []
    for r, n in res:
        z = np.clip(r / n, -3, 3)
        z[star_mask(r / n)] = 0
        Z.append(z)
    p0, v = [(p0, v) for p0, v, m in find_satellites(seq) if m == []][0]
    acc, cnt = stack(Z, v)
    x, y = int(round(p0[0])), int(round(p0[1]))
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.9), constrained_layout=True)
    for f in range(5):
        xf, yf = int(round(x + f * v[0])), int(round(y + f * v[1]))
        axes[f].imshow(Z[f][yf - half:yf + half + 1, xf - half:xf + half + 1], cmap="gray", vmin=-3, vmax=5)
        axes[f].set_title(f"frame {f + 1}", fontsize=12)
    stacked = acc[y - half:y + half + 1, x - half:x + half + 1] / np.sqrt(5)
    axes[5].imshow(stacked, cmap="gray", vmin=-3, vmax=5)
    axes[5].set_title("5 frames shifted and summed", fontsize=12, fontweight="bold")
    for ax in axes:
        ax.plot(half, half, "o", mfc="none", mec="cyan", ms=22, mew=1)
        ax.axis("off")
    fig.suptitle(f"Sequence {seq}: a satellite below the detection threshold in most single frames, "
                 f"revealed by shifting along v = ({v[0]:.0f}, {v[1]:.0f}) px/frame and adding "
                 f"(same noise scale in every panel; flat grey = masked star streaks)", fontsize=12)
    fig.savefig(OUT / "step5_stacking.png", dpi=80, bbox_inches="tight", pad_inches=0.15)


if __name__ == "__main__":
    fig_background()
    fig_star_test()
    fig_velocities()
    fig_stacking()
    print("saved to", OUT)
