import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF


class TrafficSignNet(nn.Module):
    def __init__(self,num_clas=5,fil=128,col=128):
        super().__init__()
        self.num_clas = num_clas
        self.fil = fil
        self.col = col
        

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)    # [3,fil,col] -> [16,fil,col]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                                  # [16,fil,col] -> [16,fil/2,col/2]
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)   # [16,fil/2,col/2] -> [32,fil/2,col/2]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                                  # [32,fil/2,col/2] -> [32,fil/4,col/4]
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)   # [32,fil/4,col/4] -> [64,fil/4,col/4]
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)                                  # [64,fil/4,col/4] -> [64,fil/8,col/8]
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # [64,fil/8,col/8] -> [128,fil/8,col/8]
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)                                  # [128,fil/8,col/8] -> [128,fil/16,col/16]

        self.dropout = nn.Dropout(0.25)          # Ayuda a evitar sobreajuste
        # self.fc1 = nn.Linear(32 * (self.fil // 4) * (self.col // 4), 256)
        # self.fc1 = nn.Linear(16 * (self.fil // 2) * (self.col // 2), 256)
        self.fc1 = nn.Linear(64 * (self.fil // 8) * (self.col // 8), 256)                   # [64*fil/4*col/4,1] -> [256,1]
        # self.fc1 = nn.Linear(128 * (self.fil // 16) * (self.col // 16), 1024)               # [128*fil/16*col/16,1] -> [1024,1]  ###
        # self.fc1p5 = nn.Linear(1024, 256)                                                   # [1024,1] -> [256,1]  ###
        self.fc2 = nn.Linear(256, self.num_clas)                                            # [256,1] -> [5,1]


    def forward(self, x):                       # Recibe la imagen y la hace atravesar la red
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x))) 
        x = self.pool3(F.relu(self.conv3(x)))
        # x = self.pool4(F.relu(self.conv4(x)))   ###
        x = x.view(x.size(0),-1)                # Equivalente a Flatten en tensorflow
        x = self.dropout(F.relu(self.fc1(x)))   
        # x = self.dropout(F.relu(self.fc1p5(x))) ###
        x = self.fc2(x)
        return x


if __name__=='__main__':
    imsize=128
    num_classes = 8

    # Dataset
    transform_train = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=5),
        transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
        # transforms.RandomCrop(imsize), 
        transforms.ToTensor(),
        # transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
    ])

    # train_data = datasets.ImageFolder('/ultralytics/yolo_share/signals/train', transform=transform_train)    # Dataset de entrenamiento
    # val_data = datasets.ImageFolder('/ultralytics/yolo_share/signals/valid', transform=transform_eval)       # Dataset de validacion
    # test_data = datasets.ImageFolder('/ultralytics/yolo_share/signals/test', transform=transform_eval)       # Dataset de test

    train_data = datasets.ImageFolder('/ultralytics/yolo_share/signals3/train', transform=transform_train)    # Dataset de entrenamiento
    val_data = datasets.ImageFolder('/ultralytics/yolo_share/signals3/valid', transform=transform_eval)       # Dataset de validacion
    test_data = datasets.ImageFolder('/ultralytics/yolo_share/signals3/test', transform=transform_eval)       # Dataset de test

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)


    # Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        # Usa GPU si es posible
    # model = TrafficSignNet(num_clas=5,fil=imsize,col=imsize).to(device)
    model = TrafficSignNet(num_clas=num_classes,fil=imsize,col=imsize).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    best_acc = 0.0
    best_loss = float('inf')

    for epoch in range(50):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
        
            # Resetear gradientes acumulados
            optimizer.zero_grad()

            outputs = model(images)  # Se llama al objeto como si fuera una funcion. El operador __call__() que hereda de nn.Module llama internamente a forward (no recomendado usar directamente model.forward())
            loss = criterion(outputs, labels)       # Funcion de perdida

            # Backpropagation y actualizacion de pesos de la red
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} complete. Loss: {loss.item():.4f}")


        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():                    # Se desactiva el calculo de gradientes durante la validacion
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
        
                outputs = model(images)
        
                _, predicted = torch.max(outputs.data, 1)    # Prediccion de la red
        
                total += labels.size(0)
                correct += (predicted == labels).sum().item()    # Numero de predicciones correctas

        accuracy = 100 * correct / total
        print(f'Accuracy: {100 * correct / total:.2f}%')

        if accuracy > best_acc:
            best_acc = accuracy
            if loss.item() < best_loss:
                best_loss = loss.item()
            torch.save(model.state_dict(), 'traffic_sign_net.pth')
            print(f"Modelo guardado con accuracy {accuracy:.2f}%")
        elif loss.item() < best_loss and accuracy >= best_acc * 0.96:
            best_loss = loss.item()
            torch.save(model.state_dict(), 'traffic_sign_net.pth')
            print(f"Modelo guardado con accuracy {accuracy:.2f}% y loss {loss.item():.2f}%")

    print("\nTest del modelo")
    model.load_state_dict(torch.load('traffic_sign_net.pth'))
    model.eval()

    class_names = test_data.classes
    print(class_names)

    correct = 0
    total = 0
    with torch.no_grad():   
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(images.size(0)):
                img = images[i].cpu()
                gt = class_names[labels[i]]
                pred = class_names[predicted[i]]

                print(f"GT: {gt:15s} | Pred: {pred:15s}")

                plt.imshow(TF.to_pil_image(img))
                plt.title(f"GT: {gt} | Pred: {pred}")
                plt.axis('off')
                plt.show()

    test_acc = 100 * correct / total
    print(f'\nTest Accuracy: {test_acc:.2f}%')
