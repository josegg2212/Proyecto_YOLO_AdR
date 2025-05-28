import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from SignalNet import TrafficSignNet


imsize=128
num_clas=8
transform_eval = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
])
class_names = ['CEDA', 'NO ADELANTAR', 'PEATONES', 'PROHIBIDO', 'ROTONDA', 'SIN SALIDA', 'STOP', 'VELOCIDAD']

val_data = datasets.ImageFolder('/ultralytics/yolo_share/signals3/valid', transform=transform_eval)
val_loader = DataLoader(val_data, batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TrafficSignNet(num_clas=num_clas,fil=imsize,col=imsize).to(device)
# model.load_state_dict(torch.load('/ultralytics/yolo_share/Proyecto_YOLO_AdR/traffic_sign_net_8clases.pth', map_location=device))
model.load_state_dict(torch.load('/ultralytics/yolo_share/Proyecto_YOLO_AdR/traffic_sign_net.pth', map_location=device))

# Confussion Matrix per class
ConfMat = torch.zeros(num_clas, num_clas, dtype=torch.float32).to(device)

model.eval()
T_ini=time.time()
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        for t, p in zip(labels.view(-1), predicted.view(-1)):
            ConfMat[t.long(), p.long()] += 1

T_fin=time.time()
print(f"Total de imágenes procesadas: {len(val_data)}")
print(f"Tiempo de inferencia medio: {(T_fin - T_ini)/len(val_data)} segundos por imagen")

# Confusion matrix normalization
ConfMat = ConfMat / ConfMat.sum(dim=1, keepdim=True)

# Convert to numpy for visualization
ConfMat = ConfMat.cpu().numpy()

# Visualization of the confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(ConfMat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión Normalizada')
plt.colorbar()
plt.xticks(np.arange(num_clas), val_data.classes, rotation=45)
plt.yticks(np.arange(num_clas), val_data.classes)
plt.xlabel('Predicción')
plt.ylabel('Ground Truth')
# Numerical values in the confusion matrix
for i in range(num_clas):
    for j in range(num_clas):
        value = ConfMat[i, j]
        plt.text(j, i, str(value), ha="center", va="center", color="black" if value < 0.5 else "white", fontsize=16)
plt.tight_layout()
plt.show()

# Amount of True Positives, False Positives and False Negatives per class
tp = {}
fp = {}
fn = {}
for clase in class_names:
    tp[clase] = ConfMat[class_names.index(clase), class_names.index(clase)]
    fp[clase] = ConfMat[:, class_names.index(clase)].sum() - tp[clase]
    fn[clase] = ConfMat[class_names.index(clase), :].sum() - tp[clase]
    print(f"Clase {clase}:\nTrue Positives: {tp[clase]}, False Positives: {fp[clase]}, False Negatives: {fn[clase]}\n")
