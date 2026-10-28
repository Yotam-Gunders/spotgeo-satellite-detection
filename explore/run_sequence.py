from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detect import detect

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "SpotGEOv2"
anno = json.load(open(DATA / "train_anno.json"))

seq = 1
fig, axes = plt.subplots(1, 5, figsize=(20, 3.5), constrained_layout=True)
for f in range(1, 6):
    img = np.array(Image.open(DATA / "train" / str(seq) / f"{f}.png"))
    cands = detect(img)
    print(f"frame {f}: {len(cands)} candidates")

    ax = axes[f - 1]
    ax.imshow(img, cmap="gray", vmin=np.percentile(img, 50), vmax=np.percentile(img, 99.5))
    ax.plot([c["x"] for c in cands], [c["y"] for c in cands], "+", color="yellow", ms=6)
    for a in anno:
        if a["sequence_id"] == seq and a["frame"] == f:
            for x, y in a["object_coords"]:
                ax.plot(x, y, "o", mfc="none", mec="red", ms=12)
    ax.set_title(f"frame {f}")
    ax.axis("off")
plt.show()