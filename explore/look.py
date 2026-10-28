from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "SpotGEOv2"

anno = json.load(open(DATA / "train_anno.json"))

fig, axes = plt.subplots(1, 5, figsize=(20, 3.5), constrained_layout=True)
axes = axes.flatten()
for f in range(1, 6):
    img = np.array(Image.open(DATA / "train" / "1" / f"{f}.png"))
    ax = axes[f - 1]
    ax.imshow(img, cmap="gray", vmin=np.percentile(img, 50), vmax=np.percentile(img, 99.95))
    ax.axis("off")
    for a in anno:
        if a["sequence_id"] == 1 and a["frame"] == f:
            for x,y in a["object_coords"]:
                ax.plot(x, y, "o", mfc="none", mec="red", ms=12)
    ax.set_title(f"frame {f}")
plt.show()

