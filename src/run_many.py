"""Evaluate the pipeline on training sequences 1 to 100 with the official metric."""
import json
from pipeline import find_satellites, DATA
from metric import score_seq, f1

with open(DATA / "train_anno.json") as fh:
    anno = json.load(fh)
GT = {}
for a in anno:
    GT.setdefault(a["sequence_id"], {})[a["frame"]] = a["object_coords"]

TP = FN = FP = 0
MSE = 0.0
for seq in range(1, 101):
    final = find_satellites(seq)
    tp, fn, fp, mse = score_seq(final, GT[seq])
    TP += tp
    FN += fn
    FP += fp
    MSE += mse
    print(f"seq {seq:3d}: tp {tp}  fn {fn}  fp {fp}  mse {mse:.1f}")
print(f"total: TP {TP}  FN {FN}  FP {FP}  F1 {f1(TP, FN, FP):.3f}  MSE {MSE:.1f}")
