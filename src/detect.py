"""Background removal, noise estimate and connected-component candidates.

residual() is the original scipy version (slow). find_blobs() is shared with detect_fast.py.
"""
import numpy as np
from scipy.ndimage import median_filter, label, find_objects, center_of_mass, maximum


def residual(img, bg_size=31):
    img = img.astype(float)
    resid = img - median_filter(img, size=bg_size)
    noise = 10.0
    for _ in range(3):
        core = resid[np.abs(resid) < 5 * noise]
        noise = core.std()
    return resid, noise


def detect(img, k=5, bg_size=31, max_ext=6):
    resid, noise = residual(img, bg_size)
    return find_blobs(resid, noise, k, max_ext)


def find_blobs(resid, noise, k=5, max_ext=6):
    mask = resid > k * noise
    labels, n = label(mask)
    if n == 0:
        return []

    sizes = np.bincount(labels.ravel())[1:]
    boxes = find_objects(labels)
    cents = center_of_mass(resid, labels, range(1, n + 1))
    peaks = maximum(resid, labels, range(1, n + 1))

    cands = []
    for i in range(n):
        ys, xs = boxes[i]
        if ys.start == 0 or xs.start == 0 or ys.stop == resid.shape[0] or xs.stop == resid.shape[1]:
            continue
        ext = max(ys.stop - ys.start, xs.stop - xs.start)
        if ext <= max_ext:
            cands.append({"x": cents[i][1], "y": cents[i][0],
                          "peak": peaks[i] / noise, "size": int(sizes[i])})
    return cands
