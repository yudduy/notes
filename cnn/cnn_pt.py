import torch
import torch.nn as nn
import torch.nn.functional as f

class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_filters: int = 8, kernel_size: int = 3, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size, stride=1, padding=0)
        conv_out_hw = 28 - kernel_size + 1  
        self.fc = nn.Linear(num_filters * conv_out_hw * conv_out_hw, num_classes)

    # no autograd for forward
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x) # linear -> logits
        return x

if __name__ == "__main__":
    torch.manual_seed(0)
    model = SimpleCNN(in_channels=1, num_filters=8, kernel_size=3, num_classes=10)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    x_batch = torch.randn(4, 1, 28, 28)
    y_batch = torch.randint(0, 10, (4,))

    logits = model(x_batch)
    loss = criterion(logits, y_batch)
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item()}")

    