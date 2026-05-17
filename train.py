import os
import torch
from torch import nn
from torch.utils.data import Dataset,random_split,Subset
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor,Lambda
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


LEARNING_RATE = 5e-3
BATCH_SIZE = 32
EPOCHS = 30
WEIGHT_DECAY = 5e-5

def train_loop(train_dataloader, model, loss_fn, optimizer,scheduler):
        size = len(train_dataloader.dataset)
        num_batches = len(train_dataloader)
        print("Size: ",size,"Batches: ",num_batches)
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if int(batch/num_batches * 100) % 25 == 0 or batch == num_batches-1:
                loss, current = loss.item(), batch * BATCH_SIZE + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn,text):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    print("Num batches: ",num_batches)
    print("Test Loss: ",test_loss)
    test_loss /= num_batches
    correct /= size
    print(f"{text} Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":

    training_transform = transforms.Compose([
        transforms.RandomResizedCrop((224,224), scale=(0.8, 1.0), ratio=(0.9,1.1),antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4783, 0.4459, 0.3957], std=[0.2254, 0.2223, 0.2240]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.25))
    ])

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4783, 0.4459, 0.3957], std=[0.2254, 0.2223, 0.2240]),
    ])

    trainval_data = datasets.OxfordIIITPet(
        root="data",
        split="trainval",
        download=True,
        transform=training_transform,
    )

    unaugmented_trainval_data = datasets.OxfordIIITPet(
        root="data",
        split="trainval",
        download=True,
        transform=transform,
    )

    test_data = datasets.OxfordIIITPet(
        root="data",
        split="test",
        download=True,
        transform=transform,
    )


    train_dataloader = DataLoader(trainval_data, BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, BATCH_SIZE, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class resnetBlock(nn.Module):

        def __init__(self,c):
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
                nn.MaxPool2d(2),

                nn.Conv2d(256, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                resnetBlock(512),
                nn.MaxPool2d(2),

                resnetBlock(512),
                resnetBlock(512),

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

    model = NeuralNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_dataloader), epochs=EPOCHS)


    for t in range(EPOCHS):
        print(f"Epoch {t+1} / {EPOCHS}\n--------------------------------")
        ##apply training transforms each epoch as they are random
        train_loop(train_dataloader, model, loss_fn, optimizer,scheduler)
        print("Learning rate: ",scheduler.get_last_lr())

    train_test_dataloaders = DataLoader(unaugmented_trainval_data, BATCH_SIZE, shuffle=False)
    test_loop(train_dataloader, model, loss_fn,"Training")
    test_loop(test_dataloader, model, loss_fn,"Testing")
    torch.save(model.state_dict(), "model.pth")
    
    print("------------------\nModel Trained and saved to model.pth\n------------------")