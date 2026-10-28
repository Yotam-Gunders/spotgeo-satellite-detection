from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "SpotGEOv2"
img = np.array(Image.open(DATA / "train" / "1" / "1.png"))
sx, sy = 502, 237
sat = img[sy - 15:sy + 15, sx - 15:sx + 15]

by, bx = np.unravel_index(img.argmax(), img.shape)
star = img[by - 15:by + 15, bx - 15:bx + 15]
print("brightest pixel at x =", bx, "y =", by)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(sat, cmap="gray", interpolation="nearest")
axes[0].set_title("satellite")
axes[1].imshow(star, cmap="gray", interpolation="nearest")
axes[1].set_title("star")

np.set_printoptions(linewidth=200)
print(img[sy - 4:sy + 5, sx - 4:sx + 5])
plt.show()