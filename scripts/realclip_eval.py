# ==============================================================================
# REAL-CLIP EVALUATION WRAPPER
# Connects: your videos + clips_gt.csv + trained detector + R/T/M rule -> scores.
# Returns the SAME keys as the synthetic evaluate(), so it drops into NSGA-II.
#
# Two-stage design for speed:
#   1) CACHE detections once per clip (slow: runs detector+tracker on GPU).
#   2) evaluate_real() scores cached tracks against GT for any (R,T,M) FAST (no GPU).
# This is what makes NSGA-II feasible on real video: detection runs ONCE,
# then the optimizer scores hundreds of R/T/M combos against the cache instantly.
# ==============================================================================
import csv, math
from pathlib import Path
from collections import defaultdict, deque

# ---- reuse the SAME rule as synthetic (import or paste run_rule alongside) ----
def _c(b): x1,y1,x2,y2=b; return ((x1+x2)/2.0,(y1+y2)/2.0)
def _d(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])

def run_rule(frames, fps, R_OWN, T_THRESHOLD, M_THRESHOLD, MOVE_WINDOW=15, OCCLUSION_GRACE=30):
    hist=defaultdict(lambda:deque(maxlen=MOVE_WINDOW)); since={}; seen={}; fired={}
    for fi,fr in enumerate(frames):
        now=fi/fps; ppl=[_c(b) for _,b in fr["persons"]]; active=set()
        for bid,b in fr["bags"]:
            c=_c(b); active.add(bid); seen[bid]=fi; hist[bid].append(c)
            moved=max((_d(c,p) for p in hist[bid]),default=0.0)
            near=min((_d(c,pc) for pc in ppl),default=float("inf"))
            if near>R_OWN and moved<M_THRESHOLD:
                since.setdefault(bid,now)
                if now-since[bid]>=T_THRESHOLD and bid not in fired: fired[bid]=fi
            else: since.pop(bid,None)
        for bid in list(since):
            if bid not in active and fi-seen.get(bid,fi)>OCCLUSION_GRACE: since.pop(bid,None)
    return fired

# ---- load GT ------------------------------------------------------------------
def load_gt(gt_csv):
    gt = {}
    with open(gt_csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("scored","").strip().lower() != "yes":
                continue
            def i(x): 
                x=(x or "").strip(); return int(x) if x not in ("","nan") else None
            gt[r["clip"].strip()] = {
                "fps": float(r["fps"]),
                "event": int(r["event"]),
                "leave_frame": i(r.get("leave_frame")),
                "return_frame": i(r.get("return_frame")),
                "category": r.get("category",""),
            }
    return gt

# ---- STAGE 1: cache detections per clip (run ONCE) ----------------------------
def build_detection_cache(video_dir, gt_csv, model, bag_cls=0, person_cls=1,
                          imgsz=640, conf=0.15, device="cuda:0", tracker="bytetrack.yaml"):
    """Returns {clip: {"fps":..., "frames":[{"bags":[(id,box)],"persons":[(id,box)]}...]}}"""
    gt = load_gt(gt_csv)
    # locate each clip (may be in ABODA/ or AI_Videos/ subfolders)
    all_videos = {p.name: p for p in Path(video_dir).rglob("*")
                  if p.suffix.lower() in (".avi",".mp4",".mov",".mkv",".mpg",".mpeg")}
    cache = {}
    for clip, g in gt.items():
        vp = all_videos.get(clip)
        if vp is None:
            print(f"  MISSING video for {clip}"); continue
        frames = []
        for res in model.track(source=str(vp), stream=True, persist=True, imgsz=imgsz,
                               conf=conf, device=device, tracker=tracker, verbose=False):
            bags, persons = [], []
            b = res.boxes
            if b is not None and b.id is not None:
                for tid, ccls, box in zip(b.id.int().tolist(), b.cls.int().tolist(), b.xyxy.tolist()):
                    if ccls == bag_cls: bags.append((int(tid), tuple(box)))
                    elif ccls == person_cls: persons.append((int(tid), tuple(box)))
            frames.append({"bags": bags, "persons": persons})
        cache[clip] = {"fps": g["fps"], "frames": frames}
        nb = sum(1 for fr in frames if fr["bags"])
        print(f"  cached {clip:30s} frames={len(frames):4d} frames_with_bag={nb:4d}")
    return cache, gt

# ---- STAGE 2: score cached tracks against GT for given (R,T,M) -- FAST, no GPU -
def evaluate_real(cache, gt, R_OWN, T_THRESHOLD, M_THRESHOLD, **kw):
    TP=FP=FN=TN=0; per_clip=[]
    for clip, g in gt.items():
        if clip not in cache: continue
        frames = cache[clip]["frames"]; fps = cache[clip]["fps"]
        fired = run_rule(frames, fps, R_OWN, T_THRESHOLD, M_THRESHOLD, **kw)
        did = len(fired) > 0
        # diagnostic: was a bag detected during the unattended window?
        lv = g["leave_frame"]; rt = g["return_frame"]
        win = range(lv or 0, rt or len(frames))
        bag_in_window = sum(1 for i in win if i < len(frames) and frames[i]["bags"])
        if g["event"] == 1:
            outcome = "TP" if did else "FN"
            TP += int(did); FN += int(not did)
        else:
            outcome = "FP" if did else "TN"
            FP += int(did); TN += int(not did)
        per_clip.append({"clip":clip, "outcome":outcome, "fired":did,
                         "bag_frames_in_window":bag_in_window, "category":g["category"]})
    prec=TP/(TP+FP) if TP+FP else float("nan")
    rec =TP/(TP+FN) if TP+FN else float("nan")
    f1  =2*prec*rec/(prec+rec) if (prec and rec and prec+rec>0) else float("nan")
    far =FP/(FP+TN) if (FP+TN) else 0.0
    summary = {"TP":TP,"FP":FP,"FN":FN,"TN":TN,
               "precision":round(prec,3),"recall":round(rec,3),"F1":round(f1,3),
               "false_alarm_rate":round(far,3)}
    return summary, per_clip

# adapter-compatible signature for NSGA-II (mirrors synthetic evaluate keys)
def make_nsga_evaluate(cache, gt):
    """Returns an evaluate(_, fps, R_OWN=,T_THRESHOLD=,M_THRESHOLD=) closure for NSGA-II."""
    def _ev(_bank_ignored, _fps_ignored, R_OWN, T_THRESHOLD, M_THRESHOLD, **kw):
        s, _ = evaluate_real(cache, gt, R_OWN, T_THRESHOLD, M_THRESHOLD, **kw)
        # add recall/FP/TN keys the optimizer reads
        return {"recall": s["recall"] if s["recall"]==s["recall"] else 0.0,
                "FP": s["FP"], "TN": s["TN"], "precision": s["precision"],
                "F1": s["F1"]}
    return _ev
