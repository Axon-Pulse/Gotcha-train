import torch
import torch.nn as nn


class TodB(nn.Module):
    """
    simple transformation to turn tensor into log scale
        calc: output=10*log_10(input
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 10 * torch.log10(x)


class PerChannelNormalize(nn.Module):
    """applies a per channel image normalization 
    out_channel=(input[channel]-mean(input[channel]))/std(input[channel])
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """appliess a per channel image normalization 
        out_channel=(input[channel]-mean(input[channel]))/std(input[channel])


        Args:
            x (torch.Tensor): batch of images to apply the transform on ,assuming [N C H W]

        Returns:
            torch.Tensor: the per channel normalized batch [N C H W]
        """
        # Compute mean and std along the batch dimension
        mean = x.mean(dim=[-1, -2], keepdim=True)
        std = x.std(dim=[-1, -2], keepdim=True)
        # Avoid division by zero by setting std to 1 where it's zero
        std = std + 1e-7
        # Apply normalization
        return (x - mean) / std


