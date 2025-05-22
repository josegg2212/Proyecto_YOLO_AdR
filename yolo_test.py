from ultralytics import YOLO

# Load trained model
model = YOLO("/ultralytics/yolo_share/training/first_train_1110/weights/best.pt")  

# Validate the model with the test dataset
metrics = model.val( split='test')   