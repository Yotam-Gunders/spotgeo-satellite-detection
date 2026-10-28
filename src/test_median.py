import numpy as np
from PIL import Image
import detect
import detect_fast
from pipeline import DATA

for seq, f in [(1, 1), (30, 3), (144, 5)]:
    img = np.array(Image.open(DATA / "train" / str(seq) / f"{f}.png"))
    r1, n1 = detect.residual(img)
    r2, n2 = detect_fast.residual(img)
    print(seq, f, np.array_equal(r1, r2), n1 == n2)