from ultralytics import YOLO

# Load pre-trained model weights
model = YOLO('/ultralytics/yolo_share/training/first_train_114/weights/best.pt')

# Source to be analyzed
source='csm_foto_portada_horitzontal_3787176f58.jpg'

# Run interference (detection)
results = model(source=source, show=True, conf=0.2)

       






