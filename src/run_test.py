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
    json.dump(submission, open(ROOT / "submission.json", "w"))
    print(len(submission), "rows")