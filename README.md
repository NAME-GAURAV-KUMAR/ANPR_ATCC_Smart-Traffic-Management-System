# 🚦 Smart Traffic Vision System  
## **ATCC (Traffic Count & Classification) + ANPR (Number Plate Recognition)**  
A complete computer-vision system built using **YOLOv8**, **EasyOCR**, and **Streamlit** for real-time traffic surveillance, number plate extraction, and analytics.

---

# 🧩 Project Overview

This repository contains **two full pipelines**:

---

## 1️⃣ **ATCC – Automatic Traffic Count & Classification**
- Detects & classifies **car, bus, truck, motorcycle, bicycle, pedestrian**  
- Tracks vehicles across frames  
- Generates automated **count summary + CSV logs**  
- Supports **image and video inputs**  
- Custom-trained YOLOv8 model: `yolo_ATCC.pt`

---

## 2️⃣ **ANPR – Automatic Number Plate Recognition**
From the folder structure you uploaded (ANPR dataset, annotations, yaml, notebooks), the ANPR part includes:

### ✔ YOLOv8 License Plate Detection  
### ✔ Dataset preparation (annotations + images)  
### ✔ Dedicated training notebook  
### ✔ YAML configuration  
### ✔ OCR using EasyOCR  
### ✔ Integration inside Streamlit App  

Your ANPR pipeline:

anpr_dataset/
├── annotations/ # YOLO txt label files
├── images/ # training & validation images
├── car_plate_data.yaml
├── ANPR.ipynb # training notebook
├── ANPR_DATASET.zip # original dataset
Inside the app, ANPR works like this:

1. YOLO detects the license plate  
2. The bounding box is extracted  
3. OCR reads the plate characters  
4. Output is shown on UI + downloadable logs  

The model trained: `yolo_ANPR.pt`

---

# 📁 Repository Structure
├── app.py # Streamlit app combining ATCC + ANPR
├── train_atcc.py # Training script for ATCC model
├── auto_label.py # Auto-labels ATCC frames using YOLO
├── process.py # ATCC inference pipeline
├── first.ipynb 

├── ATCC_dataset/
│ ├── images/train
│ ├── images/val
│ ├── labels/train
│ ├── labels/val

├── anpr_dataset/
│ ├── images/
│ ├── annotations/
│ ├── car_plate_data.yaml
│ ├── ANPR.ipynb
│ ├── ANPR_DATASET.zip

├── models/
│ ├── yolo_ATCC.pt
│ ├── yolo_ANPR.pt

├── notebooks/
│ ├── anpr-license-training.ipynb
│ ├── atcc-bdd100k.ipynb

---

# 🛠️ Installation (Both ATCC + ANPR)

## Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
Install Dependencies
pip install -U ultralytics streamlit opencv-python pillow pandas easyocr
🚀 Running the App
streamlit run app.py
🖥️ User Interface Workflow
Step 1 — Choose Mode

ATCC

ANPR

Step 2 — Upload a file

JPG

PNG

MP4

Step 3 — Get Results

Annotated image/video

Vehicle count summary

Number plate text

CSV logs

📦 ATCC Dataset Creation Workflow
1️⃣ Extract frames

You extracted 32,629 images from highway video.

2️⃣ Auto Label Using YOLO
python auto_label.py
3️⃣ Create Dataset (400 images)
python first.ipynb
4️⃣ Train ATCC Model
python train_atcc.py
🧠 ANPR Model Training Workflow

From your ANPR folder structure, training steps were:

Step 1 — Place images in:
anpr_dataset/images/

Step 2 — Place YOLO annotations in:
anpr_dataset/annotations/

Step 3 — Configure YAML

Example:

path: anpr_dataset
train: images
val: images
names:
  0: license_plate

Step 4 — Train YOLOv8 ANPR Model

Inside your ANPR.ipynb:

from ultralytics import YOLO  
model = YOLO("yolov8n.pt")
model.train(data="car_plate_data.yaml", epochs=20, imgsz=640)

Final Results

mAP50: ~0.85

mAP50-95: ~0.47

Model saved as:

yolo_ANPR.pt

📊 ATCC Model Training Summary

Dataset: BDD100K
Model: YOLOv8n
Results:

Metric	Score
mAP50	0.587
mAP50-95	0.325
Classes	Car, Truck, Bus, Bike, Person, Traffic Light, etc

Final model saved as:
yolo_ATCC.pt
🧪 Streamlit App – Combined Features
ATCC Output

Car count

Motorcycle count

Truck count

Bus count

Total vehicle count

Tracking table

Download CSV

ANPR Output

Detected plate

Plate image preview

OCR text result

Log download

Supports JPG, PNG, MP4, AVI.

🏁 Final Deliverables in This Project

✔ ATCC YOLOv8 Model
✔ ANPR YOLOv8 Model
✔ Auto-labeling pipeline
✔ Small dataset generation
✔ Full training notebooks
✔ Combined Streamlit Web UI
✔ Output videos + logs
✔ README documentation

👨‍💻 Author

Gaurav Kumar
Smart Traffic Vision System (ATCC + ANPR), 2025
