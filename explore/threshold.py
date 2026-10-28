from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import median_filter, label, find_objects, center_of_mass, maximum

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "SpotGEOv2"

img = np.array(Image.open(DATA / "train" / "1" / "1.png"))
img = img.astype(float)

# background
local_bg = median_filter(img, size=31)
resid = img - local_bg

# noise: iterative sigma clipping
noise = 10.0
for _ in range(3):
    core = resid[np.abs(resid) < 5 * noise]
    noise = core.std()
    print("iteration:", noise, "kept:", core.size / resid.size)

# threshold
k = 5
mask = resid > k * noise
print("pixels above threshold:", mask.sum(), "out of", mask.size)

# blobs
labels, n = label(mask)
print("blobs:", n)

sizes = np.bincount(labels.ravel())[1:]
boxes = find_objects(labels)
cents = center_of_mass(resid, labels, range(1, n + 1))
peaks = maximum(resid, labels, range(1, n + 1))

print("id    x    y  size  peak   h   w  ext  fill")
for i in range(n):
    ys, xs = boxes[i]
    h, w = ys.stop - ys.start, xs.stop - xs.start
    ext = max(h, w)
    fill = sizes[i] / (h * w)
    if sizes[i] >= 3:
        print(f"{i + 1:3d} {cents[i][1]:5.0f} {cents[i][0]:4.0f} {sizes[i]:5d} {peaks[i]:5.0f} {h:3d} {w:3d} {ext:4d} {fill:5.2f}")

# shape filter: drop streaks
keep = [i for i in range(n) if max(boxes[i][0].stop - boxes[i][0].start, boxes[i][1].stop - boxes[i][1].start) <= 6]
print("candidates after shape filter:", len(keep), "of", n)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(local_bg, cmap="gray")
axes[0].set_title("local background")
axes[1].imshow(mask, cmap="gray")
axes[1].set_title("mask")
plt.show()