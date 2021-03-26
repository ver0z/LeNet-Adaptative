import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import requests
from PIL import Image
from io import BytesIO

import copy

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=5, bias=False
        )
        self.relu_1 = nn.ReLU(inplace=True)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_2 = nn.Conv2d(
            in_channels=32, out_channels=256, kernel_size=5, bias=False
        )
        self.relu_2 = nn.ReLU(inplace=True)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(in_features=4096, out_features=120, bias=False)
        self.fc_2 = nn.Linear(in_features=120, out_features=84)
        self.fc_3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, input):
        conv_1_output = self.conv_1(input)
        relu_1_output = self.relu_1(conv_1_output)
        maxpool_1_output = self.maxpool_1(relu_1_output)
        conv_2_output = self.conv_2(maxpool_1_output)
        relu_2_output = self.relu_2(conv_2_output)
        maxpool_2_output = self.maxpool_2(relu_2_output)
        avg_pool = self.avg(maxpool_2_output)
        flatten_output = self.flatten(avg_pool)
        fc_1_output = self.fc_1(flatten_output)
        fc_2_output = self.fc_2(fc_1_output)
        fc_3_output = self.fc_3(fc_2_output)

        return fc_3_output


model = LeNet5()
print(model)

use_gpu = torch.cuda.is_available()
if use_gpu:
    net = model.cuda()
    print('USE GPU')
else:
    print('USE CPU')

from torch import optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

numb_batch = 64
T = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_data = torchvision.datasets.MNIST('mnist_data', train=True, download=True, transform=T)
val_data = torchvision.datasets.MNIST('mnist_data', train=False, download=True, transform=T)

train_dl = torch.utils.data.DataLoader(train_data, batch_size = numb_batch)
val_dl = torch.utils.data.DataLoader(val_data, batch_size = numb_batch)

def train(numb_epoch=3, lr=1e-3, device="cpu"):
    accuracies = []
    cnn = LeNet5().to(device)
    cec = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=lr)
    max_accuracy = 0
    running_loss = 0
    for epoch in range(numb_epoch):
        for i, (images, labels) in enumerate(train_dl):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = cnn(images)
            loss = cec(pred, labels)
            loss.backward()
            optimizer.step()
        accuracy = float(validate(cnn, val_dl))
        accuracies.append(accuracy)
        if accuracy > max_accuracy:
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy
            running_loss += loss.item() * images.size(0)
            epoch_loss = running_loss / len(train_dl)
            print("Saving Best Model with Accuracy: ", accuracy)
            print(f"Valor de Loss: {loss}")

        print('Epoch:', epoch+1, "Accuracy :", accuracy, '%')
    plt.plot(accuracies)
    plt.show()
    return best_model

def validate(model, data):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(data):
        images = images.cuda()
        x = model(images)
        value, pred = torch.max(x,1)
        pred = pred.data.cpu()
        total += x.size(0)
        correct += torch.sum(pred == labels)
    return correct*100./total

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("No Cuda Available")

lenet = train(5, device=device)