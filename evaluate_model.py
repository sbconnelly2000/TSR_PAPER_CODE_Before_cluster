from ultralytics import YOLO

model = YOLO("runs/train/train/weights/best.pt")

results = model.val(data="data.yaml")