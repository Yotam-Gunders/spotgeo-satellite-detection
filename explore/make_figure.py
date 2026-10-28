import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pipeline import find_satellites, DATA, ROOT

seq = int(sys.argv[1]) if len(sys.argv) > 1 else 30

anno = json.load(open(DATA / "train_anno.json"))
gt = {a["frame"]: a["object_coords"] for a in anno if a["sequence_id"] == seq}
final = find_satellites(seq)

img = np.array(Image.open(DATA / "train" / str(seq) / "1.png"))
fig, ax = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
ax.imshow(img, cmap="gray", vmin=np.percentile(img, 50), vmax=np.percentile(img, 99.5))
for f in range(1, 6):
    for x, y in gt[f]:
        ax.plot(x, y, "o", mfc="none", mec="red", ms=12, mew=1.2)
for p0, v, m in final:
    path = np.array([p0 + f * v for f in range(5)])
    ax.plot(path[:, 0], path[:, 1], "-", color="cyan", lw=1)
    ax.plot(path[:, 0], path[:, 1], "s", mfc="none", mec="cyan", ms=7)
ax.plot([], [], "o", mfc="none", mec="red", label="ground truth (frames 1-5)")
ax.plot([], [], "s-", mfc="none", color="cyan", label="predicted track")
ax.legend(loc="lower right")
ax.set_title(f"SpotGEO train sequence {seq}, frame 1, with all 5 frames' positions overlaid")
ax.axis("off")
out = ROOT / "docs" / "example.png"
fig.savefig(out, dpi=100)
print("saved", out)
