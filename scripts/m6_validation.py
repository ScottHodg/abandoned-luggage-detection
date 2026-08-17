# ==============================================================================
# M6 VALIDATION MODULE — Milestone 6 (Validation & Business Impact)
#
# Provides the abandonment rule + domain-aware evaluation so it can be IMPORTED
# rather than re-pasted into cells (survives kernel restarts).
#
# Key concept for M6: the training set includes ABODA frames, so ABODA clips are
# IN-DOMAIN (seen during training). AVSS2007 and the AI clips were never trained
# on, so they are OUT-OF-DOMAIN (OOD) and constitute the true generalization test.
# Results are reported separately for each; the gap is the generalization finding.
# ==============================================================================
import math
from collections import defaultdict

# ---- domain classification ---------------------------------------------------
def clip_domain(clip_name):
    """Classify a clip as 'in_domain' (ABODA: in training) or 'ood' (unseen)."""
    n = clip_name.lower()
    if n.startswith("avsss07") or n.startswith("avss"):
        return "ood"                      # AVSS2007 — never trained on
    if n.startswith("20250304_") or n.startswith("hailuo"):
        return "ood"                      # AI-generated clips — never trained on
    if n.startswith("video") and n.endswith((".avi", ".mp4")):
        return "in_domain"                # ABODA — frames present in training data
    return "unknown"

def split_gt_by_domain(gt):
    """Return (in_domain_gt, ood_gt, unknown_clips)."""
    ind, ood, unk = {}, {}, []
    for clip, g in gt.items():
        d = clip_domain(clip)
        if d == "in_domain": ind[clip] = g
        elif d == "ood":     ood[clip] = g
        else:                unk.append(clip)
    return ind, ood, unk

# ---- location-based abandonment rule (from Milestone 5) ----------------------
def _c(b): x1, y1, x2, y2 = b; return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
def _d(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])

def run_rule_loc(frames, fps, R_OWN, T_THRESHOLD, M_THRESHOLD,
                 LOC_MERGE=60, GAP_TOLERANCE=150):
    """Abandonment timer keyed to a persistent LOCATION cluster (robust to
    tracker-ID fragmentation). Returns {loc_id: first_frame_fired}."""
    locations, next_id, fired = {}, 0, {}
    for fi, fr in enumerate(frames):
        now = fi / fps
        persons = [_c(b) for _, b in fr["persons"]]
        seen = set()
        for _bid, b in fr["bags"]:
            c = _c(b)
            best, bestd = None, LOC_MERGE
            for lid, L in locations.items():
                dd = _d(c, L["pos"])
                if dd < bestd: best, bestd = lid, dd
            if best is None:
                best = next_id; next_id += 1
                locations[best] = {"pos": c, "unattended_since": None,
                                   "last_seen": fi, "ema": c}
            L = locations[best]
            L["ema"] = (0.8 * L["ema"][0] + 0.2 * c[0], 0.8 * L["ema"][1] + 0.2 * c[1])
            L["pos"] = L["ema"]; L["last_seen"] = fi; seen.add(best)
            nearest = min((_d(L["pos"], p) for p in persons), default=float("inf"))
            if nearest > R_OWN:
                if L["unattended_since"] is None:
                    L["unattended_since"] = now
                elif (now - L["unattended_since"]) >= T_THRESHOLD and best not in fired:
                    fired[best] = fi
            else:
                L["unattended_since"] = None
        for lid in list(locations):
            if lid not in seen and fi - locations[lid]["last_seen"] > GAP_TOLERANCE:
                del locations[lid]
    return fired

# ---- scoring -----------------------------------------------------------------
def eval_loc(cache, gt, R_OWN, T, M, LOC_MERGE=60, GAP_TOLERANCE=150):
    """Score the location rule over a cache/gt subset. Returns summary dict."""
    TP = FP = FN = TN = 0
    for clip, g in gt.items():
        if clip not in cache: continue
        fired = run_rule_loc(cache[clip]["frames"], cache[clip]["fps"],
                             R_OWN, T, M, LOC_MERGE, GAP_TOLERANCE)
        did = len(fired) > 0
        if g["event"] == 1: TP += int(did); FN += int(not did)
        else:               FP += int(did); TN += int(not did)
    rec = TP / (TP + FN) if (TP + FN) else float("nan")
    prec = TP / (TP + FP) if (TP + FP) else float("nan")
    far = FP / (FP + TN) if (FP + TN) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and prec + rec > 0) else float("nan")
    return {"recall": round(rec, 3) if rec == rec else float("nan"),
            "precision": round(prec, 3) if prec == prec else float("nan"),
            "FAR": round(far, 3) if far == far else float("nan"),
            "F1": round(f1, 3) if f1 == f1 else float("nan"),
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "n_pos": TP + FN, "n_neg": FP + TN}

def eval_per_clip(cache, gt, R_OWN, T, M, LOC_MERGE=60, GAP_TOLERANCE=150):
    """Per-clip outcomes with the domain label and detection diagnostic."""
    rows = []
    for clip, g in gt.items():
        if clip not in cache: continue
        frames = cache[clip]["frames"]
        fired = run_rule_loc(frames, cache[clip]["fps"], R_OWN, T, M, LOC_MERGE, GAP_TOLERANCE)
        did = len(fired) > 0
        outcome = ("TP" if did else "FN") if g["event"] == 1 else ("FP" if did else "TN")
        rows.append({"clip": clip[:40], "domain": clip_domain(clip),
                     "event": g["event"], "fired": did, "outcome": outcome,
                     "bag_frames": sum(1 for f in frames if f["bags"]),
                     "fire_frame": min(fired.values()) if fired else None})
    return rows

def sweep_T_by_domain(cache, gt, Ts, R_OWN=200, M=30, LOC_MERGE=60, GAP_TOLERANCE=150):
    """Recall/FAR vs T, reported separately for in-domain and OOD. Returns list of dicts."""
    ind, ood, unk = split_gt_by_domain(gt)
    if unk: print(f"[!] unclassified clips (ignored): {unk}")
    rows = []
    for T in Ts:
        a = eval_loc(cache, ind, R_OWN, T, M, LOC_MERGE, GAP_TOLERANCE)
        b = eval_loc(cache, ood, R_OWN, T, M, LOC_MERGE, GAP_TOLERANCE)
        rows.append({"T": T,
                     "in_recall": a["recall"], "in_FAR": a["FAR"], "in_n": a["n_pos"],
                     "ood_recall": b["recall"], "ood_FAR": b["FAR"],
                     "ood_pos": b["n_pos"], "ood_neg": b["n_neg"]})
    return rows

# ---- self-test ---------------------------------------------------------------
if __name__ == "__main__":
    def box(cx, cy): return (cx - 20, cy - 40, cx + 20, cy + 40)
    def make_pos(n=400, leave=50):
        fr = []
        for f in range(n):
            px = 640 if f < leave else min(640 + (f - leave) * 8, 1250)
            fr.append({"bags": [(1, box(640, 400))],
                       "persons": [(9, (px - 25, 360, px + 25, 510))]})
        return fr
    def make_neg(n=400):
        return [{"bags": [(1, box(640, 400))], "persons": [(9, box(640, 430))]} for _ in range(n)]

    cache = {"video1.avi":        {"fps": 30, "frames": make_pos()},   # ABODA -> in_domain
             "AVSSS07_EASY.mpg":  {"fps": 30, "frames": make_pos()},   # AVSS  -> ood
             "Hailuo_Video_x.mp4":{"fps": 25, "frames": make_neg()}}   # AI    -> ood neg
    gt = {"video1.avi":        {"fps":30,"event":1,"leave_frame":50,"return_frame":None},
          "AVSSS07_EASY.mpg":  {"fps":30,"event":1,"leave_frame":50,"return_frame":None},
          "Hailuo_Video_x.mp4":{"fps":25,"event":0,"leave_frame":None,"return_frame":None}}

    ind, ood, unk = split_gt_by_domain(gt)
    print("in_domain:", list(ind), "| ood:", list(ood), "| unknown:", unk)
    assert list(ind) == ["video1.avi"], "ABODA should be in_domain"
    assert len(ood) == 2, "AVSS + AI should be ood"

    rows = sweep_T_by_domain(cache, gt, [2, 6])
    for r in rows: print(r)
    assert rows[0]["in_recall"] == 1.0 and rows[0]["ood_recall"] == 1.0
    assert rows[0]["ood_FAR"] == 0.0, "owner-present negative must not fire"
    print("\nOK — domain split + rule + scoring all working")
