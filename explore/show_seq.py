import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pipeline import find_satellites, DATA

seq = int(sys.argv[1]) if len(sys.argv) > 1 else 1

anno = json.load(open(DATA / "train_anno.json"))
gt = {a["frame"]: a["object_coords"] for a in anno if a["sequence_id"] == seq}

final = find_satellites(seq)
for p0, v, m in final:
    print(p0.round(1), v.round(1), len(m), "frames:", [f + 1 for f, idx in m])

fig, axes = plt.subplots(1, 5, figsize=(22, 4), constrained_layout=True)
for f in range(1, 6):
    img = np.array(Image.open(DATA / "train" / str(seq) / f"{f}.png"))
    ax = axes[f - 1]
    ax.imshow(img, cmap="gray", vmin=np.percentile(img, 50), vmax=np.percentile(img, 99.5))
    for x, y in gt[f]:
        ax.plot(x, y, "o", mfc="none", mec="red", ms=14)
    for p0, v, m in final:
        p = p0 + (f - 1) * v
        ax.plot(p[0], p[1], "s", mfc="none", mec="cyan", ms=10)
    ax.set_title(f"seq {seq}  frame {f}")
    ax.axis("off")
plt.show()