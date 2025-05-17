import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from IdentSenales import TrafficSignNet

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
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

        # Preprocesamiento
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        # Cargar modelo
        self.model = TrafficSignNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def process_image(self, img_path):
        # Cargar imagen
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Predicción
        with torch.no_grad():
            output = self.model(img_tensor)
            _, predicted = torch.max(output.data, 1)

        class_names = ['CEDA', 'PEATONES', 'PROHIBIDO', 'STOP', 'VELOCIDAD']
        predicted_class = class_names[predicted.item()]

        print(f"Predicción: {predicted_class}")

        # Devolver y mostrar imagen con predicción
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 24) 
        except IOError:
            font = ImageFont.load_default()
        
        text = f"Predicción: {predicted_class}"
        padding = 10

        # Obtener bounding box del texto
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Calcular coordenadas centradas horizontalmente
        x = (img.width - text_width) // 2
        y = 20  # Puedes ajustar la altura como prefieras

        # Dibujar fondo blanco detrás del texto
        draw.rectangle(
            [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
            fill="white"
        )

        # Dibujar texto encima
        draw.text((x, y), text, fill="blue", font=font)

        plt.imshow(img)
        plt.title(f"Predicción: {predicted_class}")
        plt.axis('off')
        plt.show()

        return img, predicted_class
    

if __name__ == "__main__":
    model_path = "traffic_sign_net_5clases.pth"
    img_path = "/ultralytics/yolo_share/Prohibido.jpg"
    classifier = SignClassifier(model_path)
    img, predicted_class = classifier.process_image(img_path)

    # Guardar imagen
    img.save(f"predicted_{predicted_class}.jpg")