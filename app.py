import streamlit as st
import cv2
from ultralytics import YOLO
import easyocr
import tempfile
import numpy as np
import pandas as pd

st.title("Smart Traffic System – ATCC + ANPR")

# -----------------------------
# Sidebar
# -----------------------------
mode = st.sidebar.selectbox("Select Mode", ["ATCC - Vehicle Counting", "ANPR - Number Plate Recognition"])

uploaded_file = st.file_uploader("Upload Image / Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# Load models
VEHICLE_MODEL = YOLO("yolov8n.pt")    
PLATE_MODEL = YOLO("models/anpr_detector.pt")  
OCR = easyocr.Reader(['en'], gpu=False)

vehicle_classes = ["car", "truck", "bus", "motorcycle", "motorbike", "bicycle"]

def count_vehicles_yolo(frame):
    results = VEHICLE_MODEL(frame, verbose=False)
    boxes = results[0].boxes

    summary = {
        "car": 0,
        "truck": 0,
        "bus": 0,
        "motorcycle": 0,
        "bicycle": 0
    }

    for box in boxes:
        cls = int(box.cls[0])
        name = VEHICLE_MODEL.names[cls].lower()

        # Normalize motorbike name
        if name == "motorbike":
            name = "motorcycle"

        if name in summary:
            summary[name] += 1

            # Draw boxes
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,0), 2)

    total = sum(summary.values())
    return frame, summary, total


def run_anpr(frame):
    plate_results = PLATE_MODEL(frame, conf=0.3, verbose=False)

    if len(plate_results[0].boxes) == 0:
        return frame, "No Plate Found"

    box = plate_results[0].boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    plate_img = frame[y1:y2, x1:x2]

    text = OCR.readtext(plate_img, detail=0)
    plate_text = text[0] if text else "Not Readable"

    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
    cv2.putText(frame, plate_text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    return frame, plate_text



# -----------------------------
# RUN BUTTON
# -----------------------------
if uploaded_file:
    ext = uploaded_file.name.split(".")[-1].lower()

    # IMAGE MODE
    if ext in ["jpg", "jpeg", "png"]:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if mode == "ATCC - Vehicle Counting":
            out_img, summary, total = count_vehicles_yolo(frame)

            st.image(out_img, channels="BGR")

            st.subheader("Vehicle Summary")
            df = pd.DataFrame(summary.items(), columns=["Vehicle Type", "Count"])
            st.dataframe(df)

            st.success(f"Total Vehicles: {total}")

        elif mode == "ANPR - Number Plate Recognition":
            out_img, plate_num = run_anpr(frame)
            st.image(out_img, channels="BGR")
            st.success(f"Detected Plate: {plate_num}")


    # VIDEO MODE
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix="."+ext)
        tmp.write(uploaded_file.read())
        video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        # Running summary
        summary_total = {
            "car": 0,
            "truck": 0,
            "bus": 0,
            "motorcycle": 0,
            "bicycle": 0
        }

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if mode == "ATCC - Vehicle Counting":
                processed, summary, total = count_vehicles_yolo(frame)

                # Add totals
                for key in summary_total:
                    summary_total[key] += summary[key]

                stframe.image(processed, channels="BGR")

            else:
                processed, plate = run_anpr(frame)
                stframe.image(processed, channels="BGR")

        cap.release()

        if mode == "ATCC - Vehicle Counting":
            st.subheader("Final Vehicle Summary")
            df = pd.DataFrame(summary_total.items(), columns=["Vehicle Type", "Count"])
            st.dataframe(df)
            st.success(f"Total Vehicles: {sum(summary_total.values())}")
