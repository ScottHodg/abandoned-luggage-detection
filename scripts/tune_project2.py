# ==============================================================================
# Hyperparameter tuner for Project 2 (4-source expanded dataset).
# Run from a terminal:  python tune_project2.py
# Produces runs/detect/tune*/best_hyperparameters.yaml + tune_results.csv
# ==============================================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # OpenMP fix — must precede ultralytics import

from ultralytics import YOLO

# --- settings ---------------------------------------------------------------
DATA        = "dataset.yaml"   # must point at the Project 2 (4-source) unified_dataset
MODEL       = "yolov8s.pt"     # COCO-pretrained base
ITERATIONS  = 20               # number of hyperparameter combos to try
EPOCHS      = 40               # epochs per combo (screening budget)
IMGSZ       = 640
BATCH       = 16
DEVICE      = "cuda:0"

def main():
    # quick sanity check that the data config exists
    if not os.path.exists(DATA):
        raise FileNotFoundError(
            f"'{DATA}' not found in {os.getcwd()}. "
            f"Run this from the Project 2 folder that contains dataset.yaml.")

    print(f"Tuning {MODEL} on {DATA}")
    print(f"  {ITERATIONS} iterations x {EPOCHS} epochs  (this is a long, overnight-scale run)")
    print(f"  cwd: {os.getcwd()}\n")

    model = YOLO(MODEL)
    model.tune(
        data=DATA,
        epochs=EPOCHS,
        iterations=ITERATIONS,
        optimizer="AdamW",
        imgsz=IMGSZ,
        batch=BATCH,
        workers=0,          # Windows-safe
        cache=False,        # do NOT cache to RAM (caused crashes previously)
        device=DEVICE,
        seed=42,
        plots=True,
        save=True,
        val=True,
        # explicit search space (comment out to use Ultralytics defaults)
        space={
            "lr0":           (1e-5, 1e-1),
            "lrf":           (0.01, 1.0),
            "momentum":      (0.6,  0.98),
            "weight_decay":  (0.0,  0.001),
            "warmup_epochs": (0.0,  5.0),
            "box":           (1.0,  20.0),
            "cls":           (0.2,  4.0),
        },
    )
    print("\nDone. Best hyperparameters written to runs/detect/tune*/best_hyperparameters.yaml")
    print("Next: retrain at full 150 epochs using those values, then evaluate on the test split.")

if __name__ == "__main__":
    main()
