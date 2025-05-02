import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class TrafficSignNet(nn.Module):
    def __init__(self,num_clas=5,fil=32,col=32):
        super().__init__()
        self.num_clas = num_clas
        self.fil = fil
        self.col = col
        

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)    # [3,fil,col] -> [16,fil,col]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                                  # [16,fil,col] -> [16,fil/2,col/2]
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)   # [16,fil/2,col/2] -> [32,fil/2,col/2]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                                  # [32,fil/2,col/2] -> [32,fil/4,col/4]
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)   # [32,fil/4,col/4] -> [64,fil/4,col/4]

        self.fc1 = nn.Linear(64 * self.fil * self.col, 256)                                 # [64*8*8,1] -> [256,1]
        self.fc2 = nn.Linear(256, self.num_clas)                                            # [256,1] -> [5,1]


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x))) 
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder('dataset/train', transform=transform)
val_data = datasets.ImageFolder('dataset/val', transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)


# Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TrafficSignNet(num_classes=5).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} complete. Loss: {loss.item():.4f}")


# Test
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')

