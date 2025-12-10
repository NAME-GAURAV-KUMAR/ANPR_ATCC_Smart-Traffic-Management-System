from ultralytics import YOLO
from pathlib import Path

DATASET = Path("ATCC_dataset")

TRAIN_IMAGES = DATASET / "images/train"
VAL_IMAGES   = DATASET / "images/val"
TRAIN_LABELS = DATASET / "labels/train"
VAL_LABELS   = DATASET / "labels/val"

TRAIN_LABELS.mkdir(parents=True, exist_ok=True)
VAL_LABELS.mkdir(parents=True, exist_ok=True)

# COCO → custom mapping
COCO_TO_CUSTOM = {
    3: 0,  # motorcycle
    2: 1,  # car
    0: 2,  # person
    1: 3,  # bicycle
    7: 4,  # truck
}

model = YOLO("yolov8n.pt")

def create_labels(img_dir, label_dir):
    imgs = sorted(list(img_dir.glob("*.jpg")))
    results = model(imgs, imgsz=640, conf=0.35, verbose=False)

    for img, res in zip(imgs, results):
        label_file = label_dir / (img.stem + ".txt")
        lines = []

        for box in res.boxes:
            cls = int(box.cls.item())
            if cls not in COCO_TO_CUSTOM:
                continue
            new_id = COCO_TO_CUSTOM[cls]
            x, y, w, h = box.xywhn[0].tolist()
            lines.append(f"{new_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        with open(label_file, "w") as f:
            f.write("\n".join(lines))

    print(f"Labels created in {label_dir}")

create_labels(TRAIN_IMAGES, TRAIN_LABELS)
create_labels(VAL_IMAGES, VAL_LABELS)

print("✅ Auto labeling complete!")
