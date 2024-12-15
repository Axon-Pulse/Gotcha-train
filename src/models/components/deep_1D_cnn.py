import torch.nn as nn
from torch import Tensor

class Deeper1DCNN(nn.Module):
    """
    A deeper 1D CNN for binary classification.

    Input shape: (batch_size, input_dim)
    Output shape: (batch_size)
    """
    def __init__(self, input_dim=128, n_classes=2, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            # First block
            nn.Conv1d(1, 32, kernel_size=3, padding=1),  # 1 input channel -> 32 output channels
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Downsample by 2

            # Second block
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Third block
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 * (input_dim // 8), 256),  # Adjust based on pooling
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Binary classification (single output value)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, input_dim)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.fc(x)
        return self.sigmoid(x)

# Example usage
# model = Deeper1DCNN(input_dim=128)
# x = torch.randn(16, 128)  # Batch of 16, input size 128
# output = model(x)
# print(output.shape)
