# atcc_processor.py
from ultralytics import YOLO
import cv2
import pandas as pd
from pathlib import Path
import os


ATCC_MODEL_PATH = "models/atcc_detector.pt"
ANPR_MODEL_PATH = "models/anpr_detector.pt"   # you can hook your ANPR here later


def run_atcc(video_path: str):
    """
    Run ATCC on a video.
    Returns:
      out_video_path, tracking_log_df, class_counts_df
    """
    model = YOLO(ATCC_MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = "output_atcc_result.mp4"
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_idx = 0
    records = []   # for tracking log (simple: frame-wise)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        results = model(frame, imgsz=640, conf=0.4, verbose=False)
        r = results[0]

        for box in r.boxes:
            cls_id = int(box.cls.item())
            conf   = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_name = model.names[cls_id]

            # annotate
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # log for summary
            records.append({
                "frame": frame_idx,
                "class": cls_name,
                "conf": conf
            })

        out.write(frame)

    cap.release()
    out.release()

    # make summary tables
    log_df = pd.DataFrame(records)
    if len(log_df) == 0:
        counts_df = pd.DataFrame(columns=["class", "count"])
    else:
        counts_df = (
            log_df.groupby("class")
                  .size()
                  .reset_index(name="count")
                  .sort_values("count", ascending=False)
        )

    return out_path, log_df, counts_df
