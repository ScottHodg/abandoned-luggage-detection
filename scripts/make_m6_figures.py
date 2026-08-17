# ==============================================================================
# M6 FIGURE GENERATOR
# Produces every [fill in] figure in the Milestone 6 report from data you already
# have. Run from the Project 2 folder:   python make_m6_figures.py
#
# Reads (uses whatever is present; skips figures whose inputs are missing):
#   kfold_results.csv                 -> Fig 2.5, 2.8
#   robustness_results_leakfree.csv   -> Fig 3.2, 3.3
#   robustness_results.csv (leaky)    -> Fig 3.4 (leaky-vs-leakfree)
#   hard-coded summary values         -> Fig 2.8 waterfall, 4.3 cascade
#   kfold_cv.py (for fold composition)-> Fig 4.4
# Writes PNGs into ./figures/
# ==============================================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path("figures"); FIG.mkdir(exist_ok=True)
BLUE, ORANGE, GREY, RED = "#1F4E79", "#E8833A", "#7F7F7F", "#C00000"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.axisbelow": True, "figure.dpi": 150})

def save(name):
    plt.tight_layout(); plt.savefig(FIG / name, bbox_inches="tight"); plt.close()
    print("  wrote figures/" + name)

def have(f):
    if Path(f).exists(): return True
    print(f"  [skip] {f} not found"); return False


# ---- Fig 2.5 — per-fold mAP and bag AP with mean line ----------------------
def fig_kfold_bars():
    if not have("kfold_results.csv"): return
    df = pd.read_csv("kfold_results.csv")
    x = np.arange(len(df)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w/2, df["mAP50"], w, label="mAP@0.50", color=BLUE)
    ax.bar(x + w/2, df["bag_AP"], w, label="bag AP", color=ORANGE)
    ax.axhline(df["mAP50"].mean(), ls="--", color=BLUE, lw=1.3,
               label=f"mAP mean {df['mAP50'].mean():.3f}")
    ax.axhline(df["bag_AP"].mean(), ls="--", color=ORANGE, lw=1.3,
               label=f"bag mean {df['bag_AP'].mean():.3f}")
    ax.set_xticks(x); ax.set_xticklabels([f"Fold {i}" for i in df["fold"]])
    ax.set_ylabel("AP@0.50"); ax.set_ylim(0, 1)
    ax.set_title("Per-fold performance (group 5-fold CV)", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    # annotate the fold-4 collapse
    lo = df.loc[df["bag_AP"].idxmin()]
    ax.annotate("bag collapse", xy=(lo["fold"]-1+w/2, lo["bag_AP"]),
                xytext=(lo["fold"]-1, lo["bag_AP"]-0.18), fontsize=8, color=RED,
                ha="center", arrowprops=dict(arrowstyle="->", color=RED))
    save("fig_2_5_kfold_bars.png")


# ---- Fig 2.8 — validation waterfall: 0.937 -> 0.708 -> 0.50 ----------------
def fig_waterfall():
    labels = ["Single split\n(leaky)", "Group 5-fold CV\n(leak-controlled)", "Out-of-domain\n(AVSS event recall)"]
    vals = [0.9367, 0.708, 0.50]
    removed = ["", "removes\nframe leakage", "removes\ndomain overfitting"]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    colors = [GREY, BLUE, RED]
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontweight="bold")
    for i in (1, 2):
        ax.annotate(removed[i], xy=(i-0.5, (vals[i-1]+vals[i])/2),
                    ha="center", va="center", fontsize=8, color=RED, style="italic")
        ax.annotate("", xy=(i-0.08, vals[i]+0.03), xytext=(i-0.92, vals[i-1]-0.03),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.set_ylabel("performance"); ax.set_ylim(0, 1)
    ax.set_title("Performance across three validation levels", fontweight="bold")
    save("fig_2_8_waterfall.png")


# ---- Fig 2.7 — in-domain vs OOD recall-vs-T (edit if your numbers differ) --
def fig_generalization_gap():
    T = [0.5, 1, 2, 3, 4, 6, 10, 15]
    ind = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.70, 0.60]
    ood = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.00]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(T, ind, "o-", color=BLUE, lw=2, label="in-domain (ABODA, seen)")
    ax.plot(T, ood, "s-", color=RED, lw=2, label="out-of-domain (AVSS, unseen)")
    ax.fill_between(T, ood, ind, color=RED, alpha=0.10)
    ax.text(6, 0.76, "generalization\ngap", color=RED, fontsize=9, ha="center", style="italic")
    ax.set_xlabel("dwell threshold T (s)"); ax.set_ylabel("event recall")
    ax.set_ylim(-0.03, 1.05); ax.set_title("In-domain vs out-of-domain recall", fontweight="bold")
    ax.legend(fontsize=9)
    save("fig_2_7_generalization_gap.png")


# ---- Fig 3.2 — robustness degradation curves by family --------------------
def fig_robustness_curves():
    if not have("robustness_results_leakfree.csv"): return
    df = pd.read_csv("robustness_results_leakfree.csv")
    base = df[df.condition == "baseline"]["mAP50"].iloc[0]
    # severity ordering within each family (index = severity rank)
    fams = {
        "lighting":   ["brightness_1.3", "brightness_1.6"],
        "focus":      ["blur_3", "blur_7", "blur_11"],
        "sensor":     ["noise_sigma10", "noise_sigma25", "noise_sigma40"],
        "resolution": ["lowres_50pct", "lowres_25pct"],
        "occlusion":  ["occlude_20pct", "occlude_35pct"],
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for fam, conds in fams.items():
        ys = [base] + [df[df.condition == c]["mAP50"].iloc[0] for c in conds if (df.condition == c).any()]
        xs = list(range(len(ys)))
        ax.plot(xs, ys, "o-", lw=2, label=fam)
    ax.axhline(base, ls=":", color=GREY, lw=1, label=f"baseline {base:.3f}")
    ax.set_xlabel("degradation severity (0 = clean)"); ax.set_ylabel("mAP@0.50")
    ax.set_ylim(0, max(0.8, base+0.05))
    ax.set_title("Robustness: performance vs degradation severity", fontweight="bold")
    ax.legend(fontsize=8)
    ax.annotate("sensor noise\ncollapse", xy=(2, 0.08), xytext=(1.3, 0.30), color=RED,
                fontsize=9, ha="center", arrowprops=dict(arrowstyle="->", color=RED))
    save("fig_3_2_robustness_curves.png")


# ---- Fig 3.4 — leaky vs leak-free relative loss ---------------------------
def fig_leak_vs_free():
    if not (have("robustness_results.csv") and have("robustness_results_leakfree.csv")): return
    lk = pd.read_csv("robustness_results.csv").set_index("condition")["mAP50"]
    lf = pd.read_csv("robustness_results_leakfree.csv").set_index("condition")["mAP50"]
    bl_k, bl_f = lk["baseline"], lf["baseline"]
    conds = ["brightness_1.6", "blur_7", "blur_11", "lowres_25pct", "noise_sigma40"]
    conds = [c for c in conds if c in lk.index and c in lf.index]
    rl = [(bl_k - lk[c]) / bl_k * 100 for c in conds]
    rf = [(bl_f - lf[c]) / bl_f * 100 for c in conds]
    x = np.arange(len(conds)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(x - w/2, rl, w, label="leaky evaluation", color=GREY)
    ax.bar(x + w/2, rf, w, label="leak-controlled", color=BLUE)
    ax.set_xticks(x); ax.set_xticklabels(conds, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("% performance lost vs own baseline")
    ax.set_title("Leakage conceals fragility (relative loss per condition)", fontweight="bold")
    ax.legend(fontsize=9)
    save("fig_3_4_leak_vs_free.png")


# ---- Fig 4.3 — misclassification cascade funnel ---------------------------
def fig_cascade():
    stages = ["Frames total", "Bag detected", "Bag usable\n(no phantom owner)", "Timer-eligible"]
    vals = [5474, 161, 39, 39]
    pct = ["100%", "2.9%", "0.7%", "0.7%"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = np.arange(len(stages))[::-1]
    maxv = vals[0]
    for yi, (s, v, p) in zip(y, zip(stages, vals, pct)):
        width = v / maxv
        ax.barh(yi, width, color=BLUE, alpha=0.85, height=0.6)
        ax.text(width + 0.01, yi, f"{v}  ({p})", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(stages, fontsize=9)
    ax.set_xlim(0, 1.15); ax.set_xlabel("fraction of frames (log-like scale not applied)")
    ax.set_title("Misclassification cascade — AVSS2007 EASY", fontweight="bold")
    ax.annotate("timer needs ~180 consecutive\nframes at T=6s: impossible",
                xy=(0.05, 0), xytext=(0.35, 0.6), fontsize=8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    save("fig_4_3_cascade.png")


# ---- Fig 4.4 — fold composition (which videos in each fold) ----------------
def fig_fold_composition():
    if not have("kfold_cv.py"): return
    import importlib.util, sys
    from collections import Counter
    spec = importlib.util.spec_from_file_location("k", "kfold_cv.py")
    k = importlib.util.module_from_spec(spec); sys.modules["k"] = k
    try:
        spec.loader.exec_module(k)
        folds = k.make_folds(k.pool_images(), 5, 42)
    except Exception as e:
        print(f"  [skip] fold composition: {e}"); return
    # count aboda video frames per fold
    vids = sorted({it["group"] for f in folds for it in f if "aboda" in it["group"]})
    mat = np.zeros((len(vids), len(folds)))
    for fj, f in enumerate(folds):
        cnt = Counter(it["group"] for it in f)
        for vi, v in enumerate(vids):
            mat[vi, fj] = cnt.get(v, 0)
    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.4*len(vids))))
    im = ax.imshow(mat, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(folds))); ax.set_xticklabels([f"Fold {i+1}" for i in range(len(folds))])
    ax.set_yticks(range(len(vids)))
    ax.set_yticklabels([v.replace("aboda-dataset_", "").replace("_mp4", "") for v in vids], fontsize=8)
    for i in range(len(vids)):
        for j in range(len(folds)):
            if mat[i, j] > 0:
                ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if mat[i, j] > mat.max()*0.5 else "black")
    ax.set_title("ABODA video frames per fold (held-out sets)", fontweight="bold")
    plt.colorbar(im, label="frames")
    save("fig_4_4_fold_composition.png")


if __name__ == "__main__":
    print("Generating M6 figures into ./figures/ ...")
    fig_kfold_bars()
    fig_waterfall()
    fig_generalization_gap()
    fig_robustness_curves()
    fig_leak_vs_free()
    fig_cascade()
    fig_fold_composition()
    print("\nDone. Figures that need a real photo/still (not generated here):")
    print("  * Fig 3.3 noise stills  — save frames at noise 0/10/25/40 with boxes")
    print("  * Fig 4.2 suitcase misclassification — you already have easy_all_detections.jpg")
    print("  * Fig 7.4 deployment architecture diagram — draw manually")
