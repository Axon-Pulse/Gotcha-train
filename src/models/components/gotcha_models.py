import torch.nn as nn
from torch import Tensor
from torch.nn import Sequential


class SimplestMLP(nn.Module):
    """
    Input shape: (x, 134)
    Output shape: ( x)

    simple mlp
    """

    def __init__(self, input_dim=128, n_classes=2, p_dropout=0.1, **kwargs):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(input_dim, 64),
            # nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Dropout(p_dropout),
            nn.Linear(16, n_classes - 1),
            # nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x[:, :, 6:])[:, :, 0]


class Simplest1DCnn(nn.Module):
    """
    Input shape: (x, 134)
    Output shape: ( x)

    simple mlp
    """

    def __init__(self, input_dim=128, n_classes=2, **kwargs):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(1, 8, 7),
            nn.ReLU(),
            nn.Conv1d(8, 16, 5),
            nn.ReLU(),
            nn.Conv1d(16, 8, 3),
            nn.ReLU(),
            nn.Conv1d(8, n_classes - 1, 122),
            # nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x.unsqueeze(dim=2).squeeze(dim=0))[:, 0, 0].unsqueeze(dim=0)
        # return  self.seq(x.transpose(2,0).squeeze(dim=1))[:,0,0].unsqueeze(dim=0)

        # return self.seq(x[:,:,6:])[:,:,0]


if __name__ == "__main__":
    model = SimplestMLP()
    pass
