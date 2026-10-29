"""Official SpotGEO metric (as implemented in the organizers' validation.py): per-frame Hungarian matching,
tau = 10 px, eps = 3 px. F1 from summed TP/FN/FP; MSE summed over sequences.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def match_frame(pred, gt, tau=10.0, eps=3.0):
    pred = np.asarray(pred, float).reshape(-1, 2)
    gt = np.asarray(gt, float).reshape(-1, 2)
    if len(pred) == 0 or len(gt) == 0:
        return 0, len(gt), len(pred), (len(gt) + len(pred)) * tau ** 2
    d = np.hypot(pred[:, None, 0] - gt[None, :, 0], pred[:, None, 1] - gt[None, :, 1])
    cost = np.where(d <= tau, d, 1000.0)
    rows, cols = linear_sum_assignment(cost)
    tp = int(np.sum(d[rows, cols] <= tau))
    fn, fp = len(gt) - tp, len(pred) - tp
    m = d[rows, cols]
    m = m[m < tau]
    sse = np.sum(m[m >= eps]) + (fn + fp) * tau ** 2
    return tp, fn, fp, sse


def score_seq(final, gt):
    tp = fn = fp = 0
    sse = 0.0
    for f in range(5):
        pred = [p0 + f * v for p0, v, m in final]
        a, b, c, s = match_frame(pred, gt[f + 1])
        tp += a
        fn += b
        fp += c
        sse += s
    mse = sse / (tp + fn + fp) if tp + fn + fp else 0.0
    return tp, fn, fp, mse


def f1(tp, fn, fp):
    if tp == 0:
        return 0.0
    return 2 * tp / (2 * tp + fn + fp)


if __name__ == "__main__":
    print(match_frame([[100, 100]], [[150, 100]]))
    print(match_frame([[100, 100]], [[103, 104]]))
    print(match_frame([[0, 0], [1, 1]], []))
