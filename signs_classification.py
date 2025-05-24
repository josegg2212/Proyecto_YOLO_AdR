import requests
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
from IdentSenales import TrafficSignNet
import cv2
import numpy as np
import time

#####
# img_url = 'https://reynober.net/wp-content/uploads/2024/06/marcado-ce-senalizacion-vertical-scaled.jpg'
# img_url = 'https://dfmocasion.es/wp-content/uploads/2019/09/stop.jpg'
# img_url = 'https://previews.123rf.com/images/malven/malven1707/malven170700177/82873132-una-entrada-de-cartel-de-la-calle-est%C3%A1-prohibida-por-un-primer.jpg'
# img_url = 'https://thumbs.dreamstime.com/b/se%C3%B1al-ceda-paso-una-calle-borrosa-y-un-cielo-blanco-signo-de-carretera-rojo-en-forma-tri%C3%A1ngulo-sin-inscripci%C3%B3n-leyes-tr%C3%A1fico-186375073.jpg'
# img_url = 'https://images.coches.com/_news_/2019/10/limite-velocidad-10-1.jpg?w=1280&h=544'
# response = requests.get(img_url)
# img = Image.open(BytesIO(response.content)).convert('RGB')

# Cargar imagen
# img_path = '/ultralytics/yolo_share/Ceda.jpg'
# img_path = '/ultralytics/yolo_share/Prohibido.jpg'
# img_path = '/ultralytics/yolo_share/Stop.jpg'
# img_path = '/ultralytics/yolo_share/Velocidad.jpg'
# img_path = '/ultralytics/yolo_share/Peatones.jpg'
# img = Image.open(img_path).convert('RGB')


class SignClassifier():
    def __init__(self, model_path, num_clas=5, fil=128, col=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.num_clas = num_clas

        # Cargar modelo
        self.model = TrafficSignNet(num_clas=num_clas, fil=fil, col=col).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def process_image(self, img_path):
        # Cargar imagen original
        img = cv2.imread(img_path)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {img_path}")

        img_for_model = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_for_model = cv2.resize(img_for_model, (128, 128))

        img_tensor = transforms.ToTensor()(img_for_model).unsqueeze(0).to(self.device)

        # Predicción
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)  # Convertir logits a probabilidades
            confidence, predicted = torch.max(probabilities, 1)  # Obtener la confianza y la clase predicha

        if self.num_clas == 5:
            class_names = ['CEDA', 'PEATONES', 'PROHIBIDO', 'STOP', 'VELOCIDAD']
        elif self.num_clas == 8:
            class_names = ['CEDA', 'NO ADELANTAR', 'PEATONES', 'PROHIBIDO', 'ROTONDA', 'SIN SALIDA', 'STOP', 'VELOCIDAD']
        elif self.num_clas == 9:
            class_names = ['CEDA', 'DESCONOCIDO', 'NO ADELANTAR', 'PEATONES', 'PROHIBIDO', 'ROTONDA', 'SIN SALIDA', 'STOP', 'VELOCIDAD']
        predicted_class = class_names[predicted.item()]

        if confidence.item() < 0.1:
            predicted_class = "Desconocido"
            print("Confianza baja, no se puede clasificar la señal.")

        print(f"Predicción: {predicted_class} con confianza: {confidence.item():.2f}")

        # Dibuja el texto sobre la imagen original
        img = cv2.resize(img, (720, 720))
        font_scale = 2
        thickness = 5

        text = f"Prediccion: {predicted_class}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 0, 0)

        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = (img.shape[1] - text_width) // 2
        y = 60

        cv2.rectangle(
            img,
            (x - 10, y - text_height - 10),
            (x + text_width + 10, y + baseline + 10),
            (255, 255, 255),
            thickness=cv2.FILLED
        )

        cv2.putText(
            img,
            text,
            (x, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )

        return img, predicted_class


if __name__ == "__main__":
    model_path = "traffic_sign_net_5clases.pth"
    img_path = "/ultralytics/yolo_share/Prohibido.jpg"
    classifier = SignClassifier(model_path)
    img, predicted_class = classifier.process_image(img_path)

    # Redimensiona solo para mostrar si es necesario
    max_width = 800
    if img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        img_show = cv2.resize(img, (max_width, int(img.shape[0] * scale)))
    else:
        img_show = img

    try:
        cv2.imshow("Prediccion", img_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("No se pudo mostrar la imagen (posible entorno sin GUI).")

    # Guardar imagen
    cv2.imwrite(f"predicted_{predicted_class}.jpg", img)