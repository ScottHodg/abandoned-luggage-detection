# ==============================================================================
# RENDER ANNOTATED DEMO VIDEOS
#
# Runs the detector + abandonment rule over one or more clips and writes out
# annotated MP4s with colour-coded boxes:
#     green = bag attended · amber = unattended, timer counting · red = ABANDONED
#
# Pre-rendering means the presentation plays a video file instead of running the
# model live, which removes the main failure risk during a demo.
#
# Run from the Project 2 folder:   python render_annotated_videos.py
# ==============================================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # must precede torch/ultralytics

import glob, math
from pathlib import Path
import cv2
import numpy as np

# ---- settings ---------------------------------------------------------------
CLIPS = [                      # add/remove clips here; globs are fine
    "videos/**/video1.avi",
    "videos/**/video8.avi",
    "videos/**/video9.avi",
    "videos/**/video10.avi",
    "videos/**/AVSSS07_EASY.mpg",     # the honest failure case
]
OUT_DIR     = Path("annotated_videos")
MAX_FRAMES  = 3000             # cap per clip (keep demo clips short)
CONF        = 0.25
IMGSZ       = 1280
R_OWN       = 200              # ownership radius (px)
T_DWELL     = 5.0              # dwell threshold (s)
SIDE_BY_SIDE = True            # raw on the left, annotated on the right
OUT_WIDTH   = 960              # width of EACH panel in the output

COLOURS = {"attended": (60, 190, 60), "timing": (0, 190, 255),
           "abandoned": (40, 40, 235), "person": (230, 150, 40)}

# Ownership-radius overlay: a faint circle drawn around each bag showing the zone
# in which a person counts as the owner. Raise RADIUS_ALPHA to make it clearer,
# lower it (or set SHOW_RADIUS = False) to make it near-invisible.
SHOW_RADIUS   = True
RADIUS_ALPHA  = 0.30           # 0.10 = barely there, 0.60 = clearly visible
RADIUS_COLOUR = (205, 205, 205)  # light grey (BGR)


# ---- rule (same logic as the demo app) --------------------------------------
def _c(b): x1, y1, x2, y2 = b; return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
def _d(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])

def run_rule(frames, fps, R_OWN, T, LOC_MERGE=60, GAP_TOL=150):
    """Returns (first_fire_index, per_frame_states)."""
    locations, nxt, fired = {}, 0, None
    per_frame = []
    for fi, fr in enumerate(frames):
        now = fi / fps
        pcs = [_c(b) for _, b in fr["persons"]]
        seen, statuses = set(), []
        for _bid, b in fr["bags"]:
            c = _c(b); best, bd = None, LOC_MERGE
            for lid, L in locations.items():
                dd = _d(c, L["pos"])
                if dd < bd: best, bd = lid, dd
            if best is None:
                best = nxt; nxt += 1
                locations[best] = {"pos": c, "since": None, "last": fi, "ema": c}
            L = locations[best]
            L["ema"] = (0.8 * L["ema"][0] + 0.2 * c[0], 0.8 * L["ema"][1] + 0.2 * c[1])
            L["pos"] = L["ema"]; L["last"] = fi; seen.add(best)
            nearest = min((_d(L["pos"], p) for p in pcs), default=float("inf"))
            el = 0.0
            if nearest > R_OWN:
                if L["since"] is None: L["since"] = now
                el = now - L["since"]
                status = "abandoned" if el >= T else "timing"
                if el >= T and fired is None: fired = fi
            else:
                L["since"] = None; status = "attended"
            statuses.append((b, status, el))
        per_frame.append(statuses)
        for lid in list(locations):
            if lid not in seen and fi - locations[lid]["last"] > GAP_TOL:
                del locations[lid]
    return fired, per_frame


def draw_boxes(frame, dets, statuses, R_OWN=None, radius_alpha=RADIUS_ALPHA,
               show_radius=SHOW_RADIUS, show_link=True):
    """Colour-coded boxes, plus a faint ownership-radius circle.

    The circle is centred on the BAG because that is what the rule measures:
    "is any person within R pixels of this bag?". A thin link to the nearest
    person makes the measured distance visible. Both are alpha-blended so they
    sit under the boxes as a subtle underlay rather than competing with them.
    """
    out = frame.copy()
    pcs = [((p[0] + p[2]) / 2.0, (p[1] + p[3]) / 2.0) for _, p in dets["persons"]]

    if show_radius and R_OWN:
        ov = out.copy()
        for b, status, el in statuses:
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            cv2.circle(ov, (int(cx), int(cy)), int(R_OWN),
                       RADIUS_COLOUR, 1, cv2.LINE_AA)
            if show_link and pcs:
                nx, ny = min(pcs, key=lambda p: math.hypot(cx - p[0], cy - p[1]))
                cv2.line(ov, (int(cx), int(cy)), (int(nx), int(ny)),
                         RADIUS_COLOUR, 1, cv2.LINE_AA)
        cv2.addWeighted(ov, radius_alpha, out, 1 - radius_alpha, 0, out)

    for _, b in dets["persons"]:
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(out, (x1, y1), (x2, y2), COLOURS["person"], 2)
        cv2.putText(out, "person", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOURS["person"], 1, cv2.LINE_AA)
    for b, status, el in statuses:
        x1, y1, x2, y2 = [int(v) for v in b]
        col = COLOURS[status]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 4 if status == "abandoned" else 2)
        txt = ("ABANDONED %.1fs" % el if status == "abandoned"
               else "unattended %.1fs" % el if status == "timing" else "bag (attended)")
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), col, -1)
        cv2.putText(out, txt, (x1 + 4, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def banner(img, text, colour=(30, 30, 30)):
    """Title strip across the top so the viewer knows what they are looking at."""
    out = img.copy()
    h = 38
    cv2.rectangle(out, (0, 0), (out.shape[1], h), colour, -1)
    cv2.putText(out, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def fit(img, w):
    scale = w / img.shape[1]
    return cv2.resize(img, (w, int(round(img.shape[0] * scale))))


def render(clip_path, model):
    name = Path(clip_path).stem
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"  [skip] cannot open {clip_path}"); return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # pass 1 — detect
    print(f"  pass 1/2 detecting…", end="", flush=True)
    frames_meta, n = [], 0
    while n < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok: break
        res = model.predict(frame, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
        bags, persons = [], []
        for i, (c, xy) in enumerate(zip(res.boxes.cls.tolist(), res.boxes.xyxy.tolist())):
            (bags if model.names[int(c)] == "bag" else persons).append((i, tuple(xy)))
        frames_meta.append({"bags": bags, "persons": persons})
        n += 1
        if n % 100 == 0: print(".", end="", flush=True)
    cap.release()
    print(f" {n} frames")

    fire, per_frame = run_rule(frames_meta, fps, R_OWN, T_DWELL)
    bag_frames = sum(1 for f in frames_meta if f["bags"])
    verdict = (f"ALARM at frame {fire} ({fire/fps:.1f}s)" if fire is not None
               else "no alarm fired")
    print(f"  bags in {bag_frames}/{n} frames ({bag_frames/max(n,1)*100:.1f}%) · {verdict}")

    # pass 2 — draw and write
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{name}_annotated.mp4"
    cap = cv2.VideoCapture(clip_path)
    writer, i = None, 0
    print(f"  pass 2/2 rendering…", end="", flush=True)
    while i < n:
        ok, frame = cap.read()
        if not ok: break
        drawn = draw_boxes(frame, frames_meta[i], per_frame[i], R_OWN=R_OWN)
        state = ("ABANDONED" if any(s[1] == "abandoned" for s in per_frame[i])
                 else "unattended" if any(s[1] == "timing" for s in per_frame[i])
                 else "attended" if frames_meta[i]["bags"] else "no bag detected")
        drawn = banner(drawn, f"{name}  |  R={R_OWN}px  T={T_DWELL:.0f}s  |  {state}")
        if SIDE_BY_SIDE:
            left = banner(frame, f"{name}  |  raw footage")
            panel = np.hstack([fit(left, OUT_WIDTH), fit(drawn, OUT_WIDTH)])
        else:
            panel = fit(drawn, OUT_WIDTH)
        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (panel.shape[1], panel.shape[0]))
        writer.write(panel)
        i += 1
        if i % 100 == 0: print(".", end="", flush=True)
    cap.release()
    if writer: writer.release()
    print(f" -> {out_path}")
    return out_path


def main():
    from ultralytics import YOLO
    cands = glob.glob("runs/**/v8s_p2_tuned*/weights/best.pt", recursive=True)
    if not cands:
        raise FileNotFoundError("v8s_p2_tuned weights not found under runs/")
    weights = max(cands, key=os.path.getmtime)
    print(f"model: {weights}\n")
    model = YOLO(weights)

    made = []
    for pattern in CLIPS:
        hits = glob.glob(pattern, recursive=True)
        if not hits:
            print(f"[skip] no file matched {pattern}"); continue
        print(f"{Path(hits[0]).name}")
        out = render(hits[0], model)
        if out: made.append(out)
        print()

    print("=" * 60)
    print(f"rendered {len(made)} video(s) into {OUT_DIR}/")
    for m in made: print("  ", m)
    print("\nColour key: green = attended · amber = unattended (timer) · red = ABANDONED")


if __name__ == "__main__":
    main()
