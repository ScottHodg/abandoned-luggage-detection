# Real-Time Abandoned-Luggage Detection  Code \& Dataset Package

MBAI 5600G — Applied Integrative Analytics Capstone Project
Group 3: Ali Abughamja, Scott Hodgins
Ontario Tech University, Summer 2026

\---

## Project Title

Honest Validation of a Real-Time Abandoned-Luggage Detection System for Transit Environments.

## Overview / Description

This project reproduces and extends a published real-time abandoned-luggage detection
system. A YOLOv8s object detector and a ByteTrack tracker feed a location-based
"ownership rule" that flags a bag as abandoned when no person remains within an
ownership radius (R) for a sustained dwell time (T). The emphasis of the work is a
rigorous three-level validation (single split → group k-fold cross-validation →
unseen benchmark) that distinguishes genuine capability from measurement artifact.
The validation revealed data leakage in the original evaluation, quantified the
resulting inflation, and established an honest out-of-domain event recall of 0.50.

The novel contribution is a multi-objective (NSGA-II) optimization of the ownership
rule's parameters, which the base system fixed by hand.

> Note on reproducibility: the analysis was developed on Windows with an NVIDIA
> RTX 4070 GPU (CUDA). The scripts auto-detect CUDA, Apple MPS, or CPU where
> applicable. Exact metric values depend on the trained weights and the dataset
> build; see "Results" for reference numbers.

## Project Structure

```
capstone\_package/
├── README.md                     ← this file
├── requirements.txt              ← pinned dependencies
│
├── data/                         ← datasets (see "Dependencies / Data" below)
│   ├── unified\_dataset/          ← merged 4-source training/val/test images + labels
│   ├── videos/                   ← evaluation clips (ABODA, AVSS2007, negatives)
│   └── clips\_gt.csv              ← per-clip ground-truth (event, leave/return frames)
│
├── notebooks/
│   └── Project2\_abandoned\_luggage.ipynb   ← main analysis notebook (data build,
│                                            training, detection cache, sweeps)
│
├── scripts/                      ← clean, standalone, importable modules
│   ├── m6\_validation.py          ← ownership rule + domain-aware evaluation
│   ├── realclip\_eval.py          ← builds the per-clip detection cache
│   ├── viper\_parse.py            ← parses AVSS2007 ViPER XML ground truth
│   ├── video\_adapter.py          ← video reading / frame helpers
│   ├── kfold\_cv.py               ← group k-fold cross-validation
│   ├── robustness\_tests.py       ← 17-condition degradation / sensitivity suite
│   ├── tune\_project2.py          ← evolutionary hyperparameter tuning
│   ├── make\_m6\_figures.py        ← regenerates report figures from result CSVs
│   ├── render\_annotated\_videos.py← writes colour-coded annotated demo videos
│   
└── runs/                         ← trained weights (or download link, see below)
    └── detect/v8s\_p2\_tuned/weights/best.pt
```

* **Data:** `data/unified\_dataset` (images + YOLO labels), `data/videos` (evaluation
clips), `data/clips\_gt.csv` (ground-truth table). Large files are hosted externally;
see the download link below.
* **Notebooks:** `notebooks/Project2\_abandoned\_luggage.ipynb` is the primary,
end-to-end analysis (dataset build, model training, detection-cache construction,
domain sweeps, NSGA-II optimization).
* **Scripts:** the `scripts/` folder holds the clean, tested, importable modules used
for validation, robustness, tuning, and figure generation. Each has a
header comment describing its purpose and how to run it.

## Dependencies / Libraries to Install

Developed with **Python 3.13**. Install everything with:

```
pip install -r requirements.txt
```

Core libraries:

* **Python** (3.10–3.13)
* **numpy**
* **pandas**
* **matplotlib**
* **scipy**            (statistical significance testing)
* **opencv-python**    (video / image processing)
* **ultralytics**      (YOLOv8 detector + ByteTrack tracker)
* **torch**            (deep-learning backend; CUDA build on the RTX 4070)
* **pymoo**            (NSGA-II multi-objective optimization)
* **pyyaml**           (dataset / tracker configs)



> GPU note: a CUDA-capable GPU is strongly recommended for training and 1280-px
> inference. The code runs on CPU or Apple MPS but far more slowly.

## How to Run

Set `KMP\_DUPLICATE\_LIB\_OK=TRUE` before importing torch/ultralytics on Windows
(the scripts do this automatically). Run scripts from the package root so the
relative paths (`data/`, `runs/`) resolve.

### 1\. Data preprocessing

Open `notebooks/Project2\_abandoned\_luggage.ipynb` and run the dataset-build cell,
which merges the four sources, remaps classes to `{0: bag, 1: person}`, and writes
`data/unified\_dataset/{train,valid,test}`.

### 2\. Model training

In the notebook, run the training cell (YOLOv8s, 150 epochs, 640 px, tuned
hyperparameters — see Appendix B of the report). To reproduce hyperparameter
tuning instead:



### Optional - If you wish to save time and computing power

Since there is weights included you can choose to skip to Milestone 8 section of the code file. There is also a pickle file containing the cached videos since caching the video is a long process without a proper machine. If you wish to do everything yourself I would replace the appropriate code cells containing the model(s) prefilled out in the cells above and below the Milestone 8 section. 

```
python scripts/tune\_project2.py
```

Trained weights are written to `runs/detect/v8s\_p2\_tuned/weights/best.pt`.

### 3\. Model evaluation

Cross-validation (group k-fold, leak-controlled):

```
python scripts/kfold\_cv.py
```

Robustness / sensitivity suite (set MODE = "leakfree" or "leaky" at the top):

```
python scripts/robustness\_tests.py
```

Abandonment evaluation (in-domain vs out-of-domain) is run from the notebook using
`scripts/m6\_validation.py` (imported as a module) against the detection cache built
by `scripts/realclip\_eval.py`.

Regenerate report figures from the result CSVs:

```
python scripts/make\_m6\_figures.py
```



## Results

### Description

Validation degrades at each honest test, confirming that the single-split metrics were
inflated by data leakage. The reduction from single-split to cross-validated performance
is statistically significant for every class (one-sample t-test, p < 0.05). Out-of-domain
event recall on the unseen AVSS2007 benchmark is 0.50 at zero false alarms. Robustness
testing found a catastrophic sensitivity to sensor noise; error analysis attributed
out-of-domain failure to object unfamiliarity rather than object size.

### Sample table

|Metric|Single split (leaky)|5-fold group CV|vs single (p)|
|-|-|-|-|
|mAP@0.50|0.9367|0.708 ± 0.080|0.0031|
|bag AP|0.9662|0.651 ± 0.176|0.0160|
|person AP|0.9071|0.766 ± 0.044|0.0020|

Abandonment system: in-domain (ABODA) event recall 1.00; out-of-domain (AVSS2007)
event recall 0.50; false-alarm rate 0.00 at a dwell threshold T ≥ 6 s.

### Plots

Generated by `scripts/make\_m6\_figures.py` into `figures/`:

* validation waterfall (0.937 → 0.708 → 0.50)
* per-fold cross-validation bars (bag-class instability)
* recall / false-alarm rate vs dwell threshold T
* robustness degradation curves (sensor-noise collapse)
* break-even analysis (net value vs recall)

## Data Download

The video datasets are too large to include inline. Download from:

https://drive.google.com/drive/folders/1GKiBEI0ClkMpLT3ECLa1uleYqgPpCKKg?usp=sharing

Place the contents so that `data/unified\_dataset/`, `data/videos/`, and
`data/clips\_gt.csv` exist at the package root. Trained weights:

/runs/detect/v8s\_p2\_tuned is included. If any other weights or models are needed please feel free to ask via the contact information below.

Original public sources: ABODA and three Roboflow luggage/person datasets (training);
AVSS2007 / i-LIDS (evaluation). See the report's References for citations.

## Contact

Scott Hodgins — scott.hodgins@ontariotechu.net
Group \[3], MBAI 5600G, Ontario Tech University.

