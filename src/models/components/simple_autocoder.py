import torch.nn as nn
from torch import Tensor
from torch.nn import Sequential


class ConvBlock(nn.Module):
    """A basic CNN building block with convolution, batch normalization, ReLU activation, and
    optional max pooling."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        maxpool_kernel=(2, 2),
    ):
        super().__init__()
        ops = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if maxpool_kernel:
            ops.append(nn.MaxPool2d(maxpool_kernel))
        self.seq = Sequential(*ops)

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)


class Encoder(nn.Module):
    """A baseline CNN encoder with modified pooling to match reference dimensions."""

    def __init__(self, input_channels: int,**kargs) -> None:
        super().__init__()
        # Block 0
        blocks = [
            self.cnn_block(input_channels, 32),
            self.cnn_block(32, 64, maxpool_kernel=2),
            self.cnn_block(64, 128, maxpool_kernel=2),
            self.cnn_block(128, 128, maxpool_kernel=2),
            self.cnn_block(128, 128, maxpool_kernel=2),
            self.cnn_block(128, 128, maxpool_kernel=2),
        ]
        self.seq = nn.Sequential(*blocks)

    def cnn_block(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding="same",
        maxpool_kernel=None,
    ):
        modules = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        ]
        if maxpool_kernel is not None:
            modules.append(nn.MaxPool2d(maxpool_kernel))
        return nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        # print("Encoder Input shape:", x.shape)
        x_s = self.seq(x)
        # print("Encoder Output shape:", x_s)
        return self.seq(x)


class ClassificationHead(nn.Module):
    def __init__(
        self, input_dim=8192, n_classes=2, p_dropout=0.3
    ):  # Adjusted input_dim to match reference
        super().__init__()

        self.seq = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(16, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)


if __name__ == "__main__":
    _ = ClassificationHead()
