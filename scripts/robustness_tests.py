# ==============================================================================
# M6 §4 — ROBUSTNESS & RELIABILITY ASSESSMENT
#
# Answers the rubric's §4 requirements with measurements rather than assertions:
#   * "sensitivity to input variations"        -> brightness / blur / noise sweeps
#   * "performance under noisy or incomplete data" -> gaussian noise, JPEG artefacts,
#                                                     occlusion (random erasing)
#   * "model reliability under realistic operating conditions" -> CCTV-like conditions
#   * "stability of predictions"               -> degradation curves per class
#
# Method: the held-out test split is copied and degraded one factor at a time,
# then the SAME trained model is re-evaluated on each degraded copy. Everything
# else is held constant, so any metric change is attributable to that degradation.
#
# Run from the Project 2 folder:   python robustness_tests.py
# ==============================================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # must precede ultralytics import

import shutil, glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

# ---- settings ---------------------------------------------------------------
# MODE selects which (model, test-set) pair is assessed:
#
#   "leaky"    -> v8s_p2_tuned  on  unified_dataset/test
#                 The original single split. Consecutive ABODA frames straddle
#                 train/test, so the baseline (~0.937) is inflated by leakage.
#
#   "leakfree" -> kfold_5 model on  kfold_work/val
#                 Fold 5's held-out set: whole videos, none seen in training.
#                 Baseline ~0.708. This is the honest measurement.
#
# Run both and compare: if a degradation hurts in BOTH modes, the finding is real
# and not an artefact of memorised scenes.
MODE = "leakfree"

if MODE == "leaky":
    SRC_TEST     = Path("unified_dataset/test")
    WEIGHTS_GLOB = "runs/**/v8s_p2_tuned*/weights/best.pt"
    WORK         = Path("robust_work_leaky")
    OUT_CSV      = "robustness_results_leaky.csv"
elif MODE == "leakfree":
    SRC_TEST     = Path("kfold_work/val")
    WEIGHTS_GLOB = "runs/**/kfold_5/weights/best.pt"
    WORK         = Path("robust_work_leakfree")
    OUT_CSV      = "robustness_results_leakfree.csv"
else:
    raise ValueError(f"MODE must be 'leaky' or 'leakfree', got {MODE!r}")

IMGSZ    = 640
DEVICE   = "cuda:0"
CLASSES  = {0: "bag", 1: "person"}
SEED     = 42


# ---- degradations -----------------------------------------------------------
def d_identity(img):
    return img

def d_noise(sigma):
    def f(img):
        rng = np.random.default_rng(SEED)
        n = rng.normal(0, sigma, img.shape)
        return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return f

def d_blur(k):
    def f(img):
        return cv2.GaussianBlur(img, (k, k), 0)
    return f

def d_bright(factor):
    def f(img):
        return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return f

def d_jpeg(quality):
    def f(img):
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img
    return f

def d_occlude(frac):
    """Random erasing — stands in for partial occlusion / incomplete data."""
    def f(img):
        rng = np.random.default_rng(SEED)
        out = img.copy()
        h, w = img.shape[:2]
        bh, bw = int(h * frac), int(w * frac)
        y, x = rng.integers(0, max(1, h - bh)), rng.integers(0, max(1, w - bw))
        out[y:y+bh, x:x+bw] = 0
        return out
    return f

def d_downup(scale):
    """Downscale then upscale — mimics a low-resolution / far-field camera feed."""
    def f(img):
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, int(w*scale)), max(1, int(h*scale))),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return f


# grouped so the report can present one table per condition family
SUITE = [
    ("baseline",        "reference", d_identity),
    ("brightness_0.4",  "lighting",  d_bright(0.4)),   # dark / night
    ("brightness_0.7",  "lighting",  d_bright(0.7)),
    ("brightness_1.3",  "lighting",  d_bright(1.3)),
    ("brightness_1.6",  "lighting",  d_bright(1.6)),   # glare / overexposed
    ("blur_3",          "focus",     d_blur(3)),
    ("blur_7",          "focus",     d_blur(7)),
    ("blur_11",         "focus",     d_blur(11)),      # badly out of focus
    ("noise_sigma10",   "sensor",    d_noise(10)),
    ("noise_sigma25",   "sensor",    d_noise(25)),
    ("noise_sigma40",   "sensor",    d_noise(40)),     # high-ISO night CCTV
    ("jpeg_q50",        "compression", d_jpeg(50)),
    ("jpeg_q20",        "compression", d_jpeg(20)),    # heavy stream compression
    ("occlude_20pct",   "occlusion", d_occlude(0.20)),
    ("occlude_35pct",   "occlusion", d_occlude(0.35)),
    ("lowres_50pct",    "resolution", d_downup(0.50)),
    ("lowres_25pct",    "resolution", d_downup(0.25)),
]


def build_degraded(fn, out_dir):
    """Copy the test split through `fn`. Labels are unchanged (geometry preserved)."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    n = 0
    for img_path in sorted((SRC_TEST / "images").iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        cv2.imwrite(str(out_dir / "images" / f"{n:05d}.jpg"), fn(img))
        lbl = SRC_TEST / "labels" / f"{img_path.stem}.txt"
        (out_dir / "labels" / f"{n:05d}.txt").write_text(lbl.read_text() if lbl.exists() else "")
        n += 1
    for c in out_dir.rglob("*.cache"):
        c.unlink()
    cfg = {"path": str(out_dir.resolve()), "train": "images", "val": "images",
           "names": CLASSES}
    (out_dir / "d.yaml").write_text(yaml.dump(cfg, sort_keys=False))
    return n


def main():
    from ultralytics import YOLO

    if not SRC_TEST.exists():
        raise FileNotFoundError(f"{SRC_TEST}/ not found — run from the Project 2 folder.")
    cands = glob.glob(WEIGHTS_GLOB, recursive=True)
    if not cands:
        raise FileNotFoundError(f"no weights matched {WEIGHTS_GLOB}")
    best = max(cands, key=os.path.getmtime)

    n_imgs = len(list((SRC_TEST / "images").iterdir()))
    print("=" * 74)
    print(f"ROBUSTNESS / SENSITIVITY ANALYSIS   [MODE = {MODE}]")
    print("=" * 74)
    print(f"model      : {best}")
    print(f"test split : {SRC_TEST}  ({n_imgs} images)")
    if MODE == "leakfree" and n_imgs != 200:
        print(f"  [!] expected 200 images in fold 5 val, found {n_imgs}.")
        print("      kfold_work may have been rebuilt/deleted since the CV run.")
        print("      Re-run kfold_cv.py or use MODE='leaky' instead.")
    print()

    model = YOLO(best)
    rows = []
    for name, family, fn in SUITE:
        out_dir = WORK / name
        n = build_degraded(fn, out_dir)
        r = model.val(data=str(out_dir / "d.yaml"), split="val", imgsz=IMGSZ,
                      device=DEVICE, plots=False, verbose=False)
        per = {model.names[int(c)]: float(r.box.ap50[i])
               for i, c in enumerate(r.box.ap_class_index)}
        rows.append({"condition": name, "family": family, "n": n,
                     "mAP50": round(float(r.box.map50), 4),
                     "bag_AP": round(per.get("bag", float("nan")), 4),
                     "person_AP": round(per.get("person", float("nan")), 4),
                     "recall": round(float(r.box.mr), 4)})
        print(f"  {name:16s} mAP50={rows[-1]['mAP50']:.4f}  "
              f"bag={rows[-1]['bag_AP']:.4f}  person={rows[-1]['person_AP']:.4f}")
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    df = pd.DataFrame(rows)
    base = df[df.condition == "baseline"].iloc[0]
    for m in ["mAP50", "bag_AP", "person_AP"]:
        df[f"d_{m}"] = (df[m] - base[m]).round(4)      # signed change vs baseline

    print("\n" + "=" * 74)
    print("RESULTS  (d_ = change vs baseline; negative = degradation)")
    print("=" * 74)
    print(df[["condition", "family", "mAP50", "bag_AP", "person_AP",
              "d_mAP50", "d_bag_AP", "d_person_AP"]].to_string(index=False))

    print("\n" + "=" * 74)
    print("WORST CONDITION PER FAMILY (by mAP50 drop)")
    print("=" * 74)
    for fam in df[df.family != "reference"].family.unique():
        sub = df[df.family == fam].sort_values("d_mAP50")
        w = sub.iloc[0]
        print(f"  {fam:12s}: {w['condition']:16s} mAP50 {w['mAP50']:.4f} "
              f"({w['d_mAP50']:+.4f})   bag {w['d_bag_AP']:+.4f}  person {w['d_person_AP']:+.4f}")

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")
    print(f"Baseline reference: mAP50={base['mAP50']:.4f} bag={base['bag_AP']:.4f} "
          f"person={base['person_AP']:.4f}")


if __name__ == "__main__":
    main()
