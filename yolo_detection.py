from ultralytics import YOLO

# Load trained model 
model = YOLO('/ultralytics/yolo_share/sevici_model.pt')

# Source to be analyzed
source=2

# Run interference (detection)
results = model(source=source, show=True, conf=0.6)

       






