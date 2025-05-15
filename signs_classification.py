import requests
from PIL import Image
from io import BytesIO
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from IdentSenales import TrafficSignNet

# Cargar imagen
# img_path = '/ultralytics/yolo_share/Ceda.jpg'
# img_path = '/ultralytics/yolo_share/Prohibido.jpg'
# img_path = '/ultralytics/yolo_share/Stop.jpg'
# img_path = '/ultralytics/yolo_share/Velocidad.jpg'
img_path = '/ultralytics/yolo_share/Peatones.jpg'
img = Image.open(img_path).convert('RGB')

#####
# img_url = 'https://reynober.net/wp-content/uploads/2024/06/marcado-ce-senalizacion-vertical-scaled.jpg'
# img_url = 'https://dfmocasion.es/wp-content/uploads/2019/09/stop.jpg'
# img_url = 'https://previews.123rf.com/images/malven/malven1707/malven170700177/82873132-una-entrada-de-cartel-de-la-calle-est%C3%A1-prohibida-por-un-primer.jpg'
# img_url = 'https://thumbs.dreamstime.com/b/se%C3%B1al-ceda-paso-una-calle-borrosa-y-un-cielo-blanco-signo-de-carretera-rojo-en-forma-tri%C3%A1ngulo-sin-inscripci%C3%B3n-leyes-tr%C3%A1fico-186375073.jpg'
# img_url = 'https://images.coches.com/_news_/2019/10/limite-velocidad-10-1.jpg?w=1280&h=544'
# response = requests.get(img_url)
# img = Image.open(BytesIO(response.content)).convert('RGB')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Preprocesar
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
img_tensor = transform(img).unsqueeze(0).to(device) 

# Cargar modelo
model = TrafficSignNet().to(device)
model.load_state_dict(torch.load('/ultralytics/yolo_share/traffic_sign_net.pth'))
model.eval()

# Predicción
with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output.data, 1)

class_names = ['CEDA', 'PEATONES', 'PROHIBIDO', 'STOP', 'VELOCIDAD']
predicted_class = class_names[predicted.item()]
print(f"Predicción: {predicted_class}")

# Mostrar imagen con predicción
plt.imshow(img)
plt.title(f"Predicción: {predicted_class}")
plt.axis('off')
plt.show()