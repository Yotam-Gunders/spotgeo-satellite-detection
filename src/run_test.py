"""Predict all 5,120 test sequences in parallel and write submission.json in the official format."""
import json
from multiprocessing import Pool
from pipeline import find_satellites, ROOT


def predict(seq):
    final = find_satellites(seq, split="test")[:30]
    rows = []
    for f in range(5):
        coords = [(p0 + f * v).tolist() for p0, v, m in final]
        rows.append({"sequence_id": seq, "frame": f + 1,
                     "num_objects": len(coords), "object_coords": coords})
    return rows


if __name__ == "__main__":
    with Pool(8) as pool:
        results = pool.map(predict, range(1, 5121))
    submission = [row for rows in results for row in rows]
    with open(ROOT / "submission.json", "w") as fp:
        json.dump(submission, fp)
    print(len(submission), "rows")
