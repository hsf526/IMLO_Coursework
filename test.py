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

    training_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224),antialias=True),
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    training_data = datasets.OxfordIIITPet(
        root="data",
        split="trainval",
        download=True,
        transform=training_transform
    )

    test_data = datasets.OxfordIIITPet(
        root="data",
        split="test",
        download=True,
        transform=test_transform
    )
    train_dataloader = DataLoader(training_data, BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, BATCH_SIZE, shuffle=True)
    labels_map = training_data.classes

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.relu = nn.ReLU()
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
            #convoloution layers from alexnet
            self.convL1 = nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=2)
            self.convL2 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2)
            self.convL3 = nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,stride=1,padding=1)
            self.convL4 = nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1)
            self.convL5 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=2,padding=1)
        
            self.maxPool1 = nn.MaxPool2d(kernel_size=3,stride=2)
            self.maxPool2 = nn.MaxPool2d(kernel_size=3,stride=2)
            self.maxPool3 = nn.MaxPool2d(kernel_size=3,stride=1)

            self.fullyConectedLayers = nn.Sequential(
                nn.Linear(9216,4096),
                nn.ReLU(),

                nn.Linear(4096,2048),
                nn.ReLU(),

                nn.Linear(2048,37),
                nn.ReLU(),

                nn.Softmax(dim=1)
            )


        def forward(self, x):
            x = self.maxPool1(self.relu(self.convL1(x)))
            x = self.maxPool2(self.relu(self.convL2(x)))
            x = self.maxPool3(self.relu(self.convL3(x)))
            x = self.relu(self.convL4(x))
            x = self.relu(self.convL5(x))
            x = self.flatten(x)
            logits = self.fullyConectedLayers(x)
            return logits
        
    model = NeuralNetwork().to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------a------------")

        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    X,y = next(iter(train_dataloader))
    print(X.shape)
    X = X[0].unsqueeze(0)
    kernal = torch.tensor([[[[1,2,1], [2, 4, 2], [1, 2, 1]]]], dtype=torch.float32) / 16
    print(kernal.shape)
    kernal = kernal.repeat(1,3,1,1)
    kernal = kernal.repeat(2, 1, 1, 1)
    conv = nn.Conv2d(in_channels=3,out_channels=2,kernel_size=3,padding=1)
    conv.weight = nn.Parameter(kernal)
    print(conv.weight.shape)
    output = conv(X)
    print(output.shape)
    torch.save(model.state_dict(), "model_weights.pth")

    img = X[0].detach().cpu()          # [3, 224, 224]
    img = img.permute(1, 2, 0)         # [224, 224, 3]

    # Undo normalization if you used Normalize((0.5,...),(0.5,...))
    img = img * 0.5 + 0.5              # back to [0,1]

    # --- Extract feature maps ---
    fm1 = output[0, 0].detach().cpu()  # [224, 224]
    fm2 = output[0, 1].detach().cpu()  # [224, 224]

    # --- Plot ---
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(fm1, cmap='gray')
    plt.title("Feature Map 1")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(fm2, cmap='gray')
    plt.title("Feature Map 2")
    plt.axis("off")

    plt.show()
