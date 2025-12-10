from ultralytics import YOLO

# Load base model 
model = YOLO("yolov8n.pt")  

# Start training
model.train(
    data="ATCC_dataset/atcc.yaml",  
    epochs=25,
    imgsz=640,
    batch=16,
    name="atcc_train"
)
