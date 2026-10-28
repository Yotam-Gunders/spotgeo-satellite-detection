from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detect import detect

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "SpotGEOv2"

seq = 1
img1 = np.array(Image.open(DATA / "train" / str(seq) / "1.png"))
img2 = np.array(Image.open(DATA / "train" / str(seq) / "2.png"))
c1 = detect(img1)
c2 = detect(img2)

dx, dy = [], []
for a in c1:
    for b in c2:
        dx.append(b["x"] - a["x"])
        dy.append(b["y"] - a["y"])
print("pairs:", len(dx))

plt.figure(figsize=(8, 8))
plt.hist2d(dx, dy, bins=100, range=[[-100, 100], [-100, 100]], cmap="hot")
plt.colorbar(label="votes")
plt.xlabel("dx")
plt.ylabel("dy")
plt.gca().invert_yaxis()
plt.show()