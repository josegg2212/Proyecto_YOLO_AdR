import matplotlib.pyplot as plt
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

        self.dropout = nn.Dropout(0.25)          # To avoid overfitting
        self.fc1 = nn.Linear(64 * (self.fil // 8) * (self.col // 8), 256)                   # [64*fil/4*col/4,1] -> [256,1]
        self.fc2 = nn.Linear(256, self.num_clas)                                            # [256,1] -> [5,1]


    def forward(self, x):                       # Passes the image through the network
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x))) 
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0),-1)                # Equivalent to Flatten in TensorFlow
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


if __name__=='__main__':
    imsize=128
    num_classes = 8

    # Dataset
    # Image transformations
    transform_train = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, translate=(0.3, 0.3), scale=(0.8, 1.2), shear=5),
        transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
        transforms.ToTensor(),
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
    ])

    # Dataset loading
    train_data = datasets.ImageFolder('/ultralytics/yolo_share/signals3/train', transform=transform_train)    # Training dataset
    val_data = datasets.ImageFolder('/ultralytics/yolo_share/signals3/valid', transform=transform_eval)       # Validation dataset
    test_data = datasets.ImageFolder('/ultralytics/yolo_share/signals3/test', transform=transform_eval)       # Test dataset

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)


    # Train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')           # Uses GPU if available
    model = TrafficSignNet(num_clas=num_classes,fil=imsize,col=imsize).to(device)   # Network initialization

    criterion = nn.CrossEntropyLoss()                                               # Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)                     # Optimizer

    best_acc = 0.0
    best_loss = float('inf')
    acc_table = []      # To store accuracy for each epoch

    # Training loop
    for epoch in range(50):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Reset gradients
            optimizer.zero_grad()

            outputs = model(images)                 # Calls the model like a function (using forward method directly not recommended) 
            loss = criterion(outputs, labels)       # Loss function

            # Backpropagation and weight update
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1

        # Average loss for the epoch
        mean_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} complete. Loss: {mean_loss:.4f}")


        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():                    # Disables gradient calculation during validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
        
                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)        # Network prediction

                total += labels.size(0)
                correct += (predicted == labels).sum().item()    # Amount of correct predictions

        accuracy = 100 * correct / total
        print(f'Accuracy: {100 * correct / total:.2f}%')
        acc_table.append(accuracy)

        # Model save criteria
        if accuracy > best_acc:
            best_acc = accuracy
            if mean_loss < best_loss:
                best_loss = mean_loss
            torch.save(model.state_dict(), 'traffic_sign_net.pth')
            print(f"Modelo guardado con accuracy {accuracy:.2f}%")
        elif mean_loss < best_loss and accuracy >= best_acc * 0.96:
            best_loss = mean_loss
            torch.save(model.state_dict(), 'traffic_sign_net.pth')
            print(f"Modelo guardado con accuracy {accuracy:.2f}% y loss {mean_loss:.3f}")


    # Plot accuracy over epochs
    acc_table = [0] + acc_table
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(acc_table)), acc_table, linestyle='-', color='b')
    plt.title('Evolución Accuracy')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy (%)')
    plt.xticks(range(0, len(acc_table), max(1, len(acc_table)//10)))
    plt.grid()
    plt.show()


    # Model testing
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
