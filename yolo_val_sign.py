from ultralytics import YOLO

# Load model
model = YOLO("/ultralytics/yolo_share/training/first_train_signs_11/weights/best.pt")  

# Validate the model
metrics = model.val( split='test') 
