# Spotting GEO satellites without deep learning

A classical, fully explainable pipeline for the ESA Kelvins **SpotGEO** challenge: find faint geostationary satellites in 5-frame sequences of ground-based telescope images, where stars streak across the frame and the satellites are often barely above the noise.

On the official test set (5,120 sequences, scored once with the official toolkit) it reaches **F1 = 0.879**, which would place **6th of 14** on the final leaderboard. The top three entries used ensembles of neural networks. This one uses medians, connected components, voting and a little physics, and runs in about 0.4 s per sequence on a laptop CPU.

![Example: predicted tracks vs. ground truth](docs/example.png)

*Train sequence 30: all six satellites found. Red circles are the labels in frames 1 to 5. Cyan squares and lines are the predicted constant-velocity tracks. The short white dashes are stars.*

## Results

| Split | Sequences | TP | FN | FP | Precision | Recall | **F1** | MSE |
|---|---|---|---|---|---|---|---|---|
| Train 1 to 100 (used for tuning) | 100 | 885 | 120 | 30 | 96.7% | 88.1% | 0.922 | 1,118 |
| Train 101 to 300 (held out) | 200 | 1,360 | 335 | 50 | 96.5% | 80.2% | 0.876 | 2,972 |
| **Official test set (run once)** | **5,120** | **36,580** | **7,970** | **2,085** | **94.6%** | **82.1%** | **0.879** | **71,192** |

Comparison with the [final leaderboard](https://kelvins.esa.int/spot-the-geo-satellites/final-ranking), ranked by 1 − F1:

| Rank | Team | 1 − F1 |
|---|---|---|
| 1 | AgeniumSPACE | 0.0517 |
| 2 | POTLAB@BUAA | 0.0557 |
| 3 | dwiuzila | 0.0711 |
| 4 | Magpies | 0.0957 |
| 5 | Mr huangLTZaaa | 0.1158 |
| **–** | **this repo** | **0.1208** |
| 6 | francescodg | 0.1211 |
| 7 | mhalford | 0.1230 |

How the evaluation was kept honest:

- Every design choice and threshold was tuned on train sequences 1 to 100 only.
- Sequences 101 to 300 were held out to estimate generalization. The held-out F1 (0.876) predicted the test F1 (0.879) closely.
- The test set was run exactly once, after the method was frozen, and scored with the organizers' own `validation.py`.
- The local metric (`src/metric.py`) reproduces the official scorer exactly. On the held-out predictions both give F1 0.876 and MSE 2972.41.

## How it works

Each sequence is 5 grayscale frames of 640 × 480. The telescope tracks the Earth's rotation, so GEO satellites drift slowly while the stars sweep across the frame. Most of the pipeline is about getting rid of stars and noise without throwing away faint satellites.

### 1. Background removal and candidate detection (`detect.py`, `detect_fast.py`)

- **Background:** subtract a 31 × 31 median filter, which removes sky gradients and clouds but keeps small sources.
- **Noise level:** estimated with iterative sigma clipping, so stars don't inflate it.
- **Candidates:** pixels above 4σ are grouped into connected components, and each component's intensity-weighted centroid becomes a candidate.
- **Rejected components:** anything wider than 6 px (a star streak) and anything touching the border.

![Raw frame vs. background-subtracted residual](docs/step1_background.png)

*Left: a raw frame with uneven sky glow. Right: the same frame after subtracting the median background. Only compact sources remain, on flat noise.*

### 2. Two physical priors, measured from the labels

Two facts about the training labels drive most of the later decisions:

1. **All satellites in a sequence move with almost the same velocity.** The median spread is 0.2 px/frame, because they all sit on the geostationary belt.
2. **Their velocity relative to the stars is almost constant, about 26.7 px/frame** (5th to 95th percentile: 21.8 to 27.1). This is the Earth's rotation as seen by the telescope.

<p align="center"><img src="docs/step2_velocities.png" width="520" alt="Velocities of all track hypotheses"></p>

*Every pair of candidates gives a velocity hypothesis (grey). The stars form one dense cluster (red star). All six real satellites share a single velocity (blue circle) inside the ring 18 to 32 px/frame away from the stars (orange), which is where the search is restricted.*

### 3. Star velocity and per-point star removal (`pipeline.py`)

- **Track hypotheses:** every pair of candidates in two different frames defines a constant-velocity hypothesis, which is kept if at least 3 frames have a candidate within 1.5 px of its prediction. This is a hypothesize-and-verify scheme in the RANSAC and Hough family.
- **Star velocity:** the most common hypothesis velocity, refined with the median of nearby frame-to-frame displacements.
- **Star test:** a candidate is labelled a star if any other frame has a candidate at the position the star velocity predicts (within 2.5 px). Testing each point individually removes far fewer satellite points than deleting every member of a star-like track.

![Candidates after the per-point star test](docs/step3_star_test.png)

*Red crosses are candidates identified as stars and removed. Green circles are candidates kept for the satellite search. Cyan squares are the labelled satellites, and all of them survive.*

### 4. Satellite tracks (`pipeline.py`)

- **Hypotheses:** constant-velocity hypotheses are rebuilt on the remaining points.
- **Speed ring:** only hypotheses whose speed relative to the stars lies in 18 to 32 px/frame are kept (prior 2).
- **Merging:** near-duplicate tracks are merged.
- **Confirmation from the pixels:**
  - A track found in all 5 frames is accepted.
  - Otherwise, the 3 × 3 residual sum is measured at the predicted position in every frame where no candidate was found. The track is accepted only if every one of those frames shows at least 2σ of signal.
  - Effect: fewer false alarms (a chance alignment has nothing at the missing positions) and more detections (a faint satellite that dipped below the 4σ threshold still shows up).

### 5. Shift-and-add stacking for faint satellites (`pipeline.py`)

When at least one satellite is confirmed, its velocity predicts the velocity of every other satellite in the sequence (prior 1).

- **Stacking:** the 5 noise-normalized residual frames are shifted along that velocity, over a 3 × 3 grid of ±1 px/frame around it, and summed. A satellite moving at that velocity adds up 5×, while the noise only adds up √5×.
- **Star suppression:** bright star streaks are masked first, and residuals are clipped at ±3σ so that a single bright pixel can't dominate.
- **Accepting a peak:** it needs about 11σ of stacked signal, signal in at least 4 of the 5 frames, and no single frame contributing more than 60% of it.

![Shift-and-add stacking reveals a faint satellite](docs/step5_stacking.png)

*Each panel is a 41 × 41 px window centred on the predicted position, on the same noise scale. The satellite is barely distinguishable in single frames, but stands out clearly once the five frames are aligned along its velocity and summed.*

### Progression on the tuning set

| Step | TP | FN | FP | F1 |
|---|---|---|---|---|
| Baseline: 5σ detection, vote, delete star-track members | 684 | 321 | 66 | 0.779 |
| + relative-speed ring 18 to 32 px/frame | 675 | 330 | 20 | 0.794 |
| + detection threshold 4σ | 745 | 260 | 60 | 0.823 |
| + per-point star test with refined star velocity | 805 | 200 | 50 | 0.866 |
| + confirmation from pixel signal | 860 | 145 | 30 | 0.908 |
| + shift-and-add stacking | **885** | **120** | **30** | **0.922** |

### Speed

The first working version took about 21 s per sequence, which meant about 30 hours for the test set. Three changes left every output bit-identical, each checked with a unit test (`test_median.py`, `test_tracks.py`) and a full rerun:

| Change | Time per sequence |
|---|---|
| Compute the background once instead of twice | 21 s → 13 s |
| OpenCV's histogram median (`cv2.medianBlur`) with explicit reflect padding instead of `scipy.ndimage.median_filter` | 13 s → 2.8 s |
| Vectorized hypothesis building with `scipy.spatial.cKDTree` | 2.8 s → **0.4 s** |

With 8 worker processes, the whole test set runs in a few minutes.

## Design notes

**Why not a Kalman filter?** With only 5 frames and a constant-velocity model, a Kalman filter reduces to a least-squares line fit. Fitting is not the hard part. The hard part is data association: deciding which of about 190 candidates per frame belong together. Exhaustive pairwise hypotheses plus voting solve that directly.

**Some "false alarms" are probably real satellites.** Several false-positive tracks in the training set move at exactly the shared velocity of the labelled satellites in the same sequence (for example train sequences 1 and 84). Random noise has no reason to pick that velocity. These are most likely faint satellites that the human labellers missed, so the measured precision is a lower bound.

**Known failure modes:**

- **Sequences with no confirmed satellite get no stacking.** Running the stack over the whole speed ring instead flooded the results with noise peaks.
- **Satellites that move unusually fast relative to the stars are dropped by the speed ring.** One example is train sequence 4, at 38.6 px/frame.
- **Satellites that overlap a star streak can be erased by the streak mask.**
- **Three test sequences (1040, 2353, 4047) produced 20 to 30 predictions per frame for 2 to 3 real objects.** This accounts for several hundred of the test false positives. It is left unfixed on purpose: fixing it after looking at test results would be tuning on the test set.

**About the official MSE:** the official `validation.py` differs from the challenge paper. It adds up unsquared distances, and it sums the per-sequence MSEs instead of averaging them. `src/metric.py` follows the code, since that is what produced the leaderboard.

## Running it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # on Linux or macOS: .venv/bin/pip
```

Download the [SpotGEO dataset](https://doi.org/10.5281/zenodo.4432143) and unpack it so the layout is `data/SpotGEOv2/{train,test}/<sequence>/<frame>.png`, with `train_anno.json` and `test_anno.json` next to the folders. Then run from `src/`:

```bash
cd src
python run_many.py          # score train sequences 1 to 100, sequence by sequence
python run_test.py          # predict all 5,120 test sequences -> ../submission.json
python score_official.py    # validate and score with the official kit (see below)
```

`score_official.py` expects the organizers' `validation.py` from the [starter kit](https://zenodo.org/records/3874368) in an `official_kit/` folder at the repo root. It is not redistributed here.

## Repository layout

```
src/
  detect.py          background, noise, connected-component candidates (reference version)
  detect_fast.py     same results, OpenCV median (used by the pipeline)
  pipeline.py        star velocity, star test, tracks, confirmation, stacking
  metric.py          official F1 and MSE, per frame and per sequence
  run_many.py        evaluate on training sequences
  run_test.py        parallel prediction on the test set, writes the submission file
  score_official.py  run the official validator and scorer
  test_median.py     unit test: fast median == reference median
  test_tracks.py     unit test: fast hypothesis builder == reference
explore/             early exploration scripts and the README figures (make_figure.py, make_figures.py)
docs/                figures
```

## Acknowledgements

- **Dataset:** B. Chen, D. Liu, T.-J. Chin, M. Rutten, D. Derksen, M. Märtens, M. von Looz, G. Lecuyer, D. Izzo. *SpotGEO dataset*, Zenodo, 2021, [doi:10.5281/zenodo.4432143](https://doi.org/10.5281/zenodo.4432143), CC BY 4.0.
- **Challenge and metric:** B. Chen et al., *Spot the GEO Satellites: From Dataset to Kelvins SpotGEO Challenge*, CVPR Workshops 2021.

## License

MIT, see [LICENSE](LICENSE). The dataset is distributed separately under CC BY 4.0.
