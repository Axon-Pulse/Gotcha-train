import torch.nn as nn
from torch import Tensor
from torch.nn import Sequential

    
class Simple1DCNN_v2(nn.Module):
    """
    Input shape: (x, 134)
    Output shape: ( x)

    simple mlp    
    """
    def __init__(self, input_dim=128, n_classes=2, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),  # 1 input channel, 16 output channels
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * (input_dim // 4), 64),  # Adjust based on pooled size
            nn.ReLU(),
            nn.Linear(64, 1),  # Output 1 value for binary classification
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, input_size)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.fc(x)
        return self.sigmoid(x)

    


if __name__ == "__main__":
    model = Simple1DCNN_v2()
    pass