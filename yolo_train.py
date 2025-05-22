from ultralytics import YOLO
import os
import time 

# Load pretrained model weights for weight transfer
model = YOLO("/ultralytics/yolov8n.pt")  

# Dataset path
path_dataset = '/ultralytics/datasets/sevici_dataset/data.yaml'

while True :
    # Check if the path exists
    if path_dataset and os.path.exists(path_dataset):
        print(f"Ruta del dataset encontrada en: {path_dataset}")
        print("Iniciando entrenamiento...")
        
        # Train the model
        results = model.train(data=path_dataset, epochs=100, imgsz=640, workers=0, batch=4, project="/ultralytics/yolo_share/training", name="first_train_11")

        break
    else : 
        print("La ruta del dataset no existe o no está definida. Reintentando en 30 segundos...")
        time.sleep(30)