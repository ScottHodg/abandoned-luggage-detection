# ==============================================================================
# REAL-VIDEO ADAPTER for the abandonment evaluation harness.
# Bridges any abandonment video (ABODA / PETS2006 / AVSS / self-recorded) into
# the SAME per-frame {"bags":..., "persons":...} stream the synthetic harness used,
# then scores predicted events against a simple temporal ground-truth file.
#
# Requires: the evaluate()/event logic interface from the synthetic harness.
# Detector+tracker (YOLO + ByteTrack) runs on the real video to produce tracks.
# ==============================================================================
import csv, math
from pathlib import Path

# ---- ground-truth schema (ONE csv you create by watching the clips) ----------
# clips_gt.csv columns:
#   clip,fps,abandoned_frame,object_cls,event
# - clip            : video filename (matches a file in your video dir)
# - fps             : frames-per-second of that clip
# - abandoned_frame : frame index at which the object TRULY becomes abandoned
#                     (owner has left); use -1 for negative clips (no abandonment)
# - object_cls      : 0 (bag) — kept for generality
# - event           : 1 if this clip contains a true abandonment, else 0
#
# Example rows:
#   ABODA_video1.avi,25,640,0,1
#   ABODA_video3.avi,30,-1,0,0       <- a clip with no abandonment (negative)
def load_gt(path):
    gt = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            gt[row["clip"]] = {
                "fps": float(row["fps"]),
                "abandoned_frame": int(row["abandoned_frame"]),
                "event": int(row["event"]),
            }
    return gt

# ---- turn a real video into the harness frame-stream via YOLO+ByteTrack -------
# Produces: frames = [{"bags":[(track_id,(x1,y1,x2,y2))], "persons":[(track_id,box)]}, ...]
def video_to_frames(video_path, model, bag_cls=0, person_cls=1,
                    imgsz=640, conf=0.25, device="cuda:0", tracker="bytetrack.yaml"):
    frames = []
    # stream=True keeps memory flat over long clips; persist=True keeps track IDs stable
    for res in model.track(source=str(video_path), stream=True, persist=True,
                           imgsz=imgsz, conf=conf, device=device, tracker=tracker, verbose=False):
        bags, persons = [], []
        b = res.boxes
        if b is not None and b.id is not None:
            ids = b.id.int().tolist()
            clss = b.cls.int().tolist()
            xyxy = b.xyxy.tolist()
            for tid, c, box in zip(ids, clss, xyxy):
                if c == bag_cls:    bags.append((int(tid), tuple(box)))
                elif c == person_cls: persons.append((int(tid), tuple(box)))
        frames.append({"bags": bags, "persons": persons})
    return frames

# ---- evaluate one clip: did the rule fire, and was it correct/on-time? --------
def eval_clip(frames, fps, gt_row, run_rule, R_OWN, T_THRESHOLD, M_THRESHOLD, **kw):
    fired = run_rule(frames, fps, R_OWN, T_THRESHOLD, M_THRESHOLD, **kw)
    did_fire = len(fired) > 0
    fire_frame = min(fired.values()) if did_fire else None
    out = {"did_fire": did_fire, "fire_frame": fire_frame,
           "expected": bool(gt_row["event"]), "latency_s": None, "outcome": None}
    if gt_row["event"]:
        if did_fire:
            out["outcome"] = "TP"
            if gt_row["abandoned_frame"] >= 0:
                out["latency_s"] = round((fire_frame - gt_row["abandoned_frame"]) / fps, 2)
        else:
            out["outcome"] = "FN"
    else:
        out["outcome"] = "FP" if did_fire else "TN"
    return out

# ---- run the whole benchmark over a folder of clips --------------------------
def run_benchmark(video_dir, gt_csv, model, run_rule,
                  R_OWN, T_THRESHOLD, M_THRESHOLD,
                  imgsz=640, conf=0.25, device="cuda:0", **rule_kw):
    gt = load_gt(gt_csv)
    rows, conf_counts = [], {"TP":0,"FP":0,"FN":0,"TN":0}
    for clip, g in gt.items():
        vp = Path(video_dir) / clip
        if not vp.exists():
            print(f"  MISSING video: {vp}"); continue
        frames = video_to_frames(vp, model, imgsz=imgsz, conf=conf, device=device)
        r = eval_clip(frames, g["fps"], g, run_rule, R_OWN, T_THRESHOLD, M_THRESHOLD, **rule_kw)
        conf_counts[r["outcome"]] += 1
        rows.append({"clip":clip, **r})
        print(f"  {clip:28s} -> {r['outcome']:3s}  fired={r['did_fire']}  latency={r['latency_s']}")
    TP,FP,FN,TN = (conf_counts[k] for k in ("TP","FP","FN","TN"))
    prec = TP/(TP+FP) if TP+FP else float("nan")
    rec  = TP/(TP+FN) if TP+FN else float("nan")
    f1   = 2*prec*rec/(prec+rec) if (prec and rec and prec+rec>0) else float("nan")
    lat  = [x["latency_s"] for x in rows if x["latency_s"] is not None]
    summary = {"clips":len(rows), **conf_counts,
               "precision":round(prec,3), "recall":round(rec,3), "F1":round(f1,3),
               "mean_latency_s": round(sum(lat)/len(lat),2) if lat else None}
    return rows, summary

# --- structural self-test with a FAKE model + the synthetic rule (no GPU/video) ---
if __name__ == "__main__":
    import math
    from collections import defaultdict, deque
    # minimal copy of the rule so this file is self-testable
    def _c(b): x1,y1,x2,y2=b; return ((x1+x2)/2,(y1+y2)/2)
    def _d(a,b): return math.hypot(a[0]-b[0],a[1]-b[1])
    def run_rule(frames, fps, R_OWN, T_THRESHOLD, M_THRESHOLD, MOVE_WINDOW=15, OCCLUSION_GRACE=30):
        hist=defaultdict(lambda:deque(maxlen=MOVE_WINDOW)); since={}; seen={}; fired={}
        for fi,fr in enumerate(frames):
            now=fi/fps; ppl=[(_c(b)) for _,b in fr["persons"]]; active=set()
            for bid,b in fr["bags"]:
                c=_c(b); active.add(bid); seen[bid]=fi; hist[bid].append(c)
                moved=max((_d(c,p) for p in hist[bid]),default=0.0)
                near=min((_d(c,pc) for pc in ppl),default=float("inf"))
                if near>R_OWN and moved<M_THRESHOLD:
                    since.setdefault(bid,now)
                    if now-since[bid]>=T_THRESHOLD and bid not in fired: fired[bid]=fi
                else: since.pop(bid,None)
        return fired
    # fabricate a positive clip: bag static, person leaves at frame 50
    def box(cx,cy): return (cx-20,cy-40,cx+20,cy+40)
    frames=[]
    for f in range(400):
        bags=[(1,box(640,400))]
        px = 640 if f<50 else min(640+(f-50)*8, 1250)
        persons=[(9,(px-25,360,px+25,510))]
        frames.append({"bags":bags,"persons":persons})
    gt_row={"event":1,"abandoned_frame":50,"fps":25}
    r=eval_clip(frames,25,gt_row,run_rule,R_OWN=120,T_THRESHOLD=6,M_THRESHOLD=30)
    print("self-test positive clip:", r)
    assert r["outcome"]=="TP", "should detect abandonment"
    print("OK - adapter logic sound")
