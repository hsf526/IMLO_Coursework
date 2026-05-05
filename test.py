import os
import torch
from torch import nn
from torch.utils.data import Dataset,random_split
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor,Lambda
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 30

def train_loop(train_dataloader, val_dataloader, model, loss_fn, optimizer):
        size = len(train_dataloader.dataset)
        num_batches = len(train_dataloader)
        print("Size: ",size,"Batches: ",num_batches)
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch % 10 == 0 or batch == num_batches-1:
                loss, current = loss.item(), batch * BATCH_SIZE + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        val_loss = 0
        val_correct = 0
        val_total = 0
        model.eval()
        with torch.no_grad():
            for X,y in val_dataloader:
                X,y =  X.to(device), y.to(device)

                preds = model(X)
                val_loss += loss_fn(preds,y).item()
                val_correct += (preds.argmax(1) == y).sum().item()
                val_total += y.size(0)
        val_loss = val_loss/len(val_dataloader)
        accuracy = val_correct/val_total * 100
        print(f'Validation Loss: {val_loss:.20f} | Validation Accuracy: {accuracy:.2f}%')


def test_loop(dataloader, model, loss_fn):
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
            #print("Y: ",y)
            #print("Pred: ",pred)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    print("Num batches: ",num_batches)
    print("Test Loss: ",test_loss)
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":

    training_transform = transforms.Compose([
        transforms.RandomResizedCrop(224,antialias=True),
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.RandomRotation(degrees=30),
        #transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
        #transforms.RandomAffine(degrees=0,translate=(0.1,0.1),scale=(0.9,1.1),shear=10),
        transforms.ToTensor(),
    ])

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    train_data = datasets.OxfordIIITPet(
        root="data",
        split="trainval",
        download=True,
        transform=training_transform
    )

    validation_data = datasets.OxfordIIITPet(
        root="data",
        split="trainval",
        download=True,
        transform=transform
    )

    test_data = datasets.OxfordIIITPet(
        root="data",
        split="test",
        download=True,
        transform=transform
    )

    train_size = int(len(train_data) * 0.8)
    val_size = len(train_data) - train_size

    ##the split will be the same for both as we have a set seed
    #so we now have a transformed training set and untransformed validation set
    gen = torch.Generator().manual_seed(50)
    training_data,__ = random_split(train_data,[train_size,val_size],generator=gen)
    __,val_data = random_split(validation_data,[train_size,val_size],generator=gen)
    

    

    train_dataloader = DataLoader(training_data, BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_data,32,shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.relu = nn.ReLU()

            #convoloution layers from alexnet
            self.convL1 = nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=2)
            self.convL2 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2)
            self.convL3 = nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,stride=1,padding=1)
            self.convL4 = nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,stride=1,padding=1)
            self.convL5 = nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1)
        
            self.maxPool1 = nn.MaxPool2d(kernel_size=3,stride=2)
            self.maxPool2 = nn.MaxPool2d(kernel_size=3,stride=2)
            self.maxPool3 = nn.MaxPool2d(kernel_size=3,stride=2)

            self.features = nn.Sequential(

                nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=2),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3,stride=2),
                
                nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3,stride=2),

                nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU(),

                nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU(),

                nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3,stride=2)
            )

            self.fullyConectedLayers = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(9216,4096),
                nn.ReLU(),

                nn.Dropout(p=0.5),
                nn.Linear(4096,2048),
                nn.ReLU(),

                nn.Linear(2048,37)
            )


        def forward(self, x):
            """
            x = self.maxPool1(self.relu(self.convL1(x)))
            x = self.maxPool2(self.relu(self.convL2(x)))
            x = self.relu(self.convL3(x))
            x = self.relu(self.convL4(x))
            x = self.maxPool3(self.relu(self.convL5(x)))
            #print("Shape before flatten:", x.shape)
            #print("Feature map:", x.shape)
            """
            x = self.features(x)
            x = self.flatten(x)
            logits = self.fullyConectedLayers(x)
            return logits
        
    model = NeuralNetwork().to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler(optimizer,step_size=5,gamma=0.5)
    

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------a------------")
        train_loop(train_dataloader, val_dataloader, model, loss_fn, optimizer)
        scheduler.step()
    print("Testing...")
    test_loop(test_dataloader, model, loss_fn)
    print("Done!")