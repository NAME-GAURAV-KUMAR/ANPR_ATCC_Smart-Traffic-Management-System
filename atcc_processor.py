import os
import cv2
import numpy as np
import pandas as pd
import tempfile
from ultralytics import YOLO
from datetime import timedelta


def seconds_to_hhmmss(sec):
    return str(timedelta(seconds=int(sec)))


def run_atcc(video_path, model_path="yolov8n.pt", conf=0.25):
    """
    ATCC processing using YOLOv8 tracking.
    Detects vehicles, tracks them, counts them, and generates log.
    """
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # counting line (horizontal)
    count_line_y = int(H * 0.6)

    # output video temporary file
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (W, H))

    tracked_info = {}  # track_id → {class, first_frame, last_frame}
    class_counts = {}  # class → count
    counted_ids = set()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        time_sec = frame_idx / fps
        time_str = seconds_to_hhmmss(time_sec)

        # YOLOv8 tracking
        results = model.track(frame, persist=True, conf=conf, verbose=False)

        if results[0].boxes.id is None:
            out.write(frame)
            continue

        boxes = results[0].boxes
        class_names = model.model.names

        for box in boxes:

            if box.id is None:
                continue

            track_id = int(box.id[0])
            cls_id = int(box.cls[0])
            cls_name = class_names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # register track
            if track_id not in tracked_info:
                tracked_info[track_id] = {
                    "class": cls_name,
                    "first_frame": frame_idx,
                    "last_frame": frame_idx
                }
            else:
                tracked_info[track_id]["last_frame"] = frame_idx

            # counting when object crosses line
            if cy > count_line_y and track_id not in counted_ids:
                counted_ids.add(track_id)
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            # draw tracking info
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{cls_name} ID:{track_id}",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,255,0), 2)

        # draw time + counting line
        cv2.putText(frame, f"Time: {time_str}", (10, H-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.line(frame, (0, count_line_y), (W, count_line_y), (255,0,0), 2)

        out.write(frame)

    cap.release()
    out.release()

    # build CSV log
    log_rows = []
    for tid, info in tracked_info.items():
        log_rows.append({
            "track_id": tid,
            "class": info["class"],
            "first_frame": info["first_frame"],
            "last_frame": info["last_frame"],
            "first_time": seconds_to_hhmmss(info["first_frame"]/fps),
            "last_time": seconds_to_hhmmss(info["last_frame"]/fps)
        })

    df_tracks = pd.DataFrame(log_rows)

    df_counts = pd.DataFrame(
        [{"class": k, "count": v} for k, v in class_counts.items()]
    )

    return temp_output.name, df_tracks, df_counts
