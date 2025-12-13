from ultralytics import YOLO

# Load the model
model = YOLO("runs/detect/train2/weights/best.pt")

# Run the evaluation
results = model.val(data="data.yaml")