# ==============================================================================
# M6 §3/§4 — GROUP K-FOLD CROSS-VALIDATION
#
# Trains k models on k different train/val partitions of the pooled dataset and
# reports mean +/- std of the metrics. This provides:
#   * §3 "cross-validation or resampling procedures"  (explicitly required)
#   * §3 "evidence supporting model reliability"      (confidence bounds)
#   * §4 "consistency across validation folds"        (the std IS the answer)
#
# IMPORTANT — leakage control: ABODA frames are consecutive video frames, so
# near-duplicates MUST NOT straddle a fold boundary or the metrics are inflated.
# We group by source video where the filename allows it (Roboflow appends a hash
# like "_jpg.rf.<hex>" which we strip first), and use GroupKFold so every frame
# from one video lands in exactly one fold.
#
# Run from the Project 2 folder:   python kfold_cv.py
# ==============================================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # must precede ultralytics import

import re, shutil, json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import yaml

# ---- settings ---------------------------------------------------------------
SRC_DIR   = Path("unified_dataset")   # built by the Project 2 build cell
WORK_DIR  = Path("kfold_work2")        # temp per-fold datasets
K         = 5
EPOCHS    = 150
IMGSZ     = 640
BATCH     = 16
DEVICE    = "cuda:0"
SEED      = 42
# Dataloader workers. Safe to raise here because this script has an
# `if __name__ == "__main__"` guard (Windows multiprocessing needs it) and runs
# from a terminal, not a notebook. Drop to 0 if you hit multiprocessing errors.
WORKERS   = 4
CLASSES   = {0: "bag", 1: "person"}

# tuned hyperparameters (match v8s_p2_tuned so folds are comparable to the main model)
HYP = dict(optimizer="AdamW", lr0=0.00061, lrf=0.01543, momentum=0.74953,
           weight_decay=0.00073, warmup_epochs=3.27654, box=5.71302, cls=0.37035)

AUG = dict(close_mosaic=10, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
           degrees=5.0, translate=0.1, scale=0.5, fliplr=0.5)


# ---- grouping ---------------------------------------------------------------
def group_key(filename):
    """Map an image filename to a source-video group.

    Roboflow appends '_jpg.rf.<hex>' to exported names; that is stripped first.
    We then group ONLY when the name is recognisably a video frame, e.g.
        'aboda-dataset_frame5_0040_jpg.rf.9c1a2b3d.jpg' -> 'aboda-dataset_frame5'
        'luggage-and-persom_Calle_1_mp4-17_jpg.rf.ab12.jpg' -> '..._calle_1_mp4'

    Anything else (independent stock photos such as 'photo-1506748686') keeps its
    own group. This is deliberately conservative: wrongly merging unrelated photos
    into one group would unbalance the folds, whereas failing to merge only costs
    us a little leakage protection on images that are not frames anyway.
    """
    stem = Path(filename).stem.lower()
    stem = re.sub(r"_(jpg|jpeg|png)\.rf\.[0-9a-f]+$", "", stem)   # strip roboflow hash

    m = re.match(r"^(.*frame\d+)[-_]\d+$", stem)                  # ..._frame5_0040
    if m: return m.group(1)
    m = re.match(r"^(.*video\d+)[-_]\d+$", stem)                  # ..._video5_0040
    if m: return m.group(1)
    m = re.match(r"^(.*_mp4)[-_]\d+$", stem)                      # ..._Calle_1_mp4-17
    if m: return m.group(1)
    return stem                                                   # independent image


def pool_images():
    """Collect every (image, label) pair across the existing train/valid/test dirs."""
    items = []
    for split in ["train", "valid", "test"]:
        idir, ldir = SRC_DIR / split / "images", SRC_DIR / split / "labels"
        if not idir.exists():
            continue
        for img in sorted(idir.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            items.append({"img": img, "lbl": ldir / f"{img.stem}.txt",
                          "group": group_key(img.name)})
    return items


def report_grouping(items):
    groups = Counter(x["group"] for x in items)
    multi = {g: n for g, n in groups.items() if n > 1}
    print(f"  {len(items)} images -> {len(groups)} groups")
    print(f"  groups holding >1 frame (i.e. real video groups): {len(multi)}")
    if multi:
        top = sorted(multi.items(), key=lambda kv: -kv[1])[:8]
        print("  largest groups:", ", ".join(f"{g}={n}" for g, n in top))
    else:
        print("  [!] NO multi-frame groups found. Every image is independent, so k-fold")
        print("      cannot protect against near-duplicate frame leakage. If the ABODA")
        print("      source contains video frames, inspect the filenames and adjust")
        print("      group_key() before trusting these results.")
    return groups


# ---- fold construction ------------------------------------------------------
def make_folds(items, k, seed):
    """GroupKFold-style split: whole groups are assigned to folds, balanced by size."""
    by_group = defaultdict(list)
    for it in items:
        by_group[it["group"]].append(it)
    groups = sorted(by_group, key=lambda g: -len(by_group[g]))   # largest first
    rng = np.random.default_rng(seed)
    order = list(groups)
    rng.shuffle(order)
    order.sort(key=lambda g: -len(by_group[g]))                  # greedy bin-packing
    folds = [[] for _ in range(k)]
    sizes = [0] * k
    for g in order:
        i = int(np.argmin(sizes))                                # put group in emptiest fold
        folds[i].extend(by_group[g])
        sizes[i] += len(by_group[g])
    return folds


def materialise_fold(folds, held_out, work):
    """Write a YOLO dataset where fold `held_out` is val and the rest are train.

    Images are copied under SHORT SEQUENTIAL names (00001.jpg ...) rather than the
    original 70+ char Roboflow names. This removes three failure modes at once:
    filename collisions silently overwriting a copy, Windows path-length limits,
    and stale label caches keyed to old names. Grouping is already computed from
    the original names before this point, so nothing is lost.
    """
    if work.exists():
        shutil.rmtree(work)
    for split in ["train", "val"]:
        (work / split / "images").mkdir(parents=True, exist_ok=True)
        (work / split / "labels").mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    manifest = []
    for fi, fold in enumerate(folds):
        split = "val" if fi == held_out else "train"
        for it in fold:
            counts[split] += 1
            short = f"{counts[split]:05d}"
            dst_img = work / split / "images" / f"{short}{it['img'].suffix.lower()}"
            dst_lbl = work / split / "labels" / f"{short}.txt"
            shutil.copy(it["img"], dst_img)
            dst_lbl.write_text(it["lbl"].read_text() if it["lbl"].exists() else "")
            manifest.append({"split": split, "short": dst_img.name,
                             "original": it["img"].name, "group": it["group"]})

    # verify what actually landed on disk matches what we intended
    for split in ["train", "val"]:
        on_disk = len(list((work / split / "images").iterdir()))
        if on_disk != counts[split]:
            raise RuntimeError(
                f"fold {held_out+1}: {split} expected {counts[split]} images but "
                f"{on_disk} are on disk — copy lost files (name collision?)")

    # nuke any label cache so YOLO rescans this fold from scratch
    for cache in work.rglob("*.cache"):
        cache.unlink()

    (work / "manifest.json").write_text(json.dumps(manifest, indent=1))
    cfg = {"path": str(work.resolve()), "train": "train/images",
           "val": "val/images", "names": CLASSES}
    ypath = work / "fold.yaml"
    ypath.write_text(yaml.dump(cfg, sort_keys=False))
    return ypath, counts


# ---- main -------------------------------------------------------------------
def main():
    from ultralytics import YOLO

    if not SRC_DIR.exists():
        raise FileNotFoundError(f"{SRC_DIR}/ not found. Run this from the Project 2 folder.")

    print("=" * 74)
    print(f"{K}-FOLD GROUP CROSS-VALIDATION")
    print("=" * 74)

    items = pool_images()
    report_grouping(items)

    folds = make_folds(items, K, SEED)
    print(f"\n  fold sizes: {[len(f) for f in folds]}")
    est = K * EPOCHS * len(items) * 0.8 / 1000 / 60
    print(f"  ~{K} trainings x {EPOCHS} epochs — expect a multi-hour run.\n")

    rows = []
    for fold_i in range(K):
        print("-" * 74)
        print(f"FOLD {fold_i + 1}/{K}")
        print("-" * 74)
        ypath, counts = materialise_fold(folds, fold_i, WORK_DIR)
        print(f"  train={counts['train']}  val={counts['val']}")

        model = YOLO("yolov8s.pt")
        model.train(data=str(ypath), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                    workers=WORKERS, device=DEVICE, cache=False, seed=SEED,
                    name=f"kfold_{fold_i+1}", plots=False, val=True,
                    verbose=False, **HYP, **AUG)

        r = model.val(data=str(ypath), split="val", imgsz=IMGSZ,
                      device=DEVICE, plots=False, verbose=False)
        per = {model.names[int(c)]: float(r.box.ap50[i])
               for i, c in enumerate(r.box.ap_class_index)}
        row = {"fold": fold_i + 1,
               "n_train": counts["train"], "n_val": counts["val"],
               "mAP50": round(float(r.box.map50), 4),
               "mAP50_95": round(float(r.box.map), 4),
               "bag_AP": round(per.get("bag", float("nan")), 4),
               "person_AP": round(per.get("person", float("nan")), 4),
               "precision": round(float(r.box.mp), 4),
               "recall": round(float(r.box.mr), 4)}
        rows.append(row)
        print(f"  -> mAP50={row['mAP50']}  bag={row['bag_AP']}  person={row['person_AP']}")
        pd.DataFrame(rows).to_csv("kfold_results.csv", index=False)   # save as we go

    df = pd.DataFrame(rows)
    print("\n" + "=" * 74)
    print("PER-FOLD RESULTS")
    print("=" * 74)
    print(df.to_string(index=False))

    print("\n" + "=" * 74)
    print("CROSS-VALIDATED SUMMARY  (mean +/- std)")
    print("=" * 74)
    for m in ["mAP50", "mAP50_95", "bag_AP", "person_AP", "precision", "recall"]:
        print(f"  {m:11s}: {df[m].mean():.4f} +/- {df[m].std():.4f}"
              f"   [min {df[m].min():.4f}, max {df[m].max():.4f}]")

    df.to_csv("kfold_results.csv", index=False)
    print("\nSaved -> kfold_results.csv")
    print("Single-split reference (v8s_p2_tuned, test): mAP50=0.9367 bag=0.9662 person=0.9071")


if __name__ == "__main__":
    main()
