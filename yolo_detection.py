from ultralytics import YOLO

# Load pre-trained model weights
model = YOLO('/ultralytics/yolo_share/training/first_train/weights/best.pt')

# Source to be analyzed
source='https://www.mairenadelaljarafe.es/export/sites/mairena/.galleries/noticias/DZM_AY_20201027_029.jpg_2090699297.jpg'

# Run interference (detection)
results = model(source=source, show=True)








