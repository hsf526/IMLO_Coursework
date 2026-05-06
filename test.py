import os
import torch
from torch import nn
from torch.utils.data import Dataset,random_split,Subset
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor,Lambda
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_data = datasets.OxfordIIITPet(
        root="data",
        split="test",
        download=True,
        transform=transform
    )



class resnetBlock(nn.Module):
        
    def __init__(self, c):
        super().__init__()

        self.activation = nn.ReLU()

        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
        )

    def forward(self, x):
        return self.activation(self.conv(x) + x)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.features = nn.Sequential(
            
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), 

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2), 

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            resnetBlock(512),
            resnetBlock(512),
            resnetBlock(512),

            nn.MaxPool2d(2), 
        )

        self.classifyer = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Dropout(p=0.5),
            nn.Linear(512*1*1 ,37)
        )


    def forward(self, x):
        x = self.features(x)
        #print("Shape after features: ",x.shape)
        logits = self.classifyer(x)
        return logits
    
if __name__ == "__main__":
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model_weights.pth"))
    model.eval()

    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")