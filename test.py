import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor,Lambda
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 5

def train_loop(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        print("Size: ",size,"Batches: ",num_batches)
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch % 10 == 0 or batch == num_batches:
                loss, current = loss.item(), batch * BATCH_SIZE + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])

    training_data = datasets.OxfordIIITPet(
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
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    labels_map = training_data.classes

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                # 3 * 244 * 244 as ive standardised image size to 244x244
                # and each pixel has 3 channels
                nn.Linear(3*224*224, 5024),
                nn.ReLU(),
                nn.Linear(5024, 2056),
                nn.ReLU(),
                nn.Linear(2056, 512),
                nn.ReLU(),
                nn.Linear(512,37),
                nn.Softmax(dim=1)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits
        
    model = NeuralNetwork().to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), "model_weights.pth")
