import torch
import torch.nn as nn
import torch.nn.functional as f

class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv = nnConv2d(in_channels, 8, kernel_size=3, stride=1, padding=0)
        self.fc = nn.Linear(8 * 26 * 26, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x