from ultralytics import YOLO

# Load trained model
model = YOLO("/ultralytics/yolo_share/sevici_model.pt")  

# Validate the model with the test dataset
metrics = model.val( split='test', conf=0.6)   