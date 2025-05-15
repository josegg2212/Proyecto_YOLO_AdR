# Sin probar

from PIL import Image

# Cargar imagen
img_path = '...'
img = Image.open(img_path).convert('RGB')  # Asegura que sea RGB

# Preprocesar
transform = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor(),
])
img_tensor = transform(img).unsqueeze(0).to(device)  # Añade dimensión batch

# Cargar modelo
model.load_state_dict(torch.load('traffic_sign_net.pth'))
model.eval()

# Predicción
with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output.data, 1)

predicted_class = class_names[predicted.item()]
print(f"Predicción: {predicted_class}")

# Mostrar imagen con predicción
plt.imshow(img)
plt.title(f"Predicción: {predicted_class}")
plt.axis('off')
plt.show()