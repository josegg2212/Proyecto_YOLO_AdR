import torch
from torchvision import transforms
from IdentSenales import TrafficSignNet
import cv2


class SignClassifier():
    def __init__(self, model_path, num_clas=8, fil=128, col=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.num_clas = num_clas

        # Load model
        self.model = TrafficSignNet(num_clas=num_clas, fil=fil, col=col).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def process_image(self, img_path):
        # Load original image
        img = cv2.imread(img_path)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {img_path}")

        img_for_model = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_for_model = cv2.resize(img_for_model, (128, 128))

        img_tensor = transforms.ToTensor()(img_for_model).unsqueeze(0).to(self.device)

        # Prediction
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)  # Convert logits to probabilities
            confidence, predicted = torch.max(probabilities, 1)         # Get confidence and predicted class

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

        # Draws the text on the original image
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
    model_path = "traffic_sign_net_8clases.pth"
    img_path = "/ultralytics/yolo_share/Prohibido.jpg"
    classifier = SignClassifier(model_path)
    img, predicted_class = classifier.process_image(img_path)

    # Resize only for display if necessary
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

    # Save the image
    cv2.imwrite(f"predicted_{predicted_class}.jpg", img)