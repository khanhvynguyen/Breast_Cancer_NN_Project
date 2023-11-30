import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, h: int, w: int):
        super(MLPModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(h * w * 3, 20)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(20, 10)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(10, 2)  # Change to 1 for binary classification
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        # x = self.softmax(x)
        return x
