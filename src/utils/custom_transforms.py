import torch
import torch.nn as nn
import math

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




# Daniel
class FlipAugmentation(nn.Module):
    """Flip the doppler vector 

    Args:
        nn (_type_): _description_
    """
    def __init__(self, p=0.5, vector_offset=6):
        super().__init__()
        self.p = p
        self.vector_offset = vector_offset
        
    def forward(self, x):
        if (torch.rand(1) > self.p):
            return x
        if x.ndim == 1:
            x[self.vector_offset:] = torch.flip(x[self.vector_offset:])
        elif x.ndim == 2:
            n = math.floor(self.p * x.shape[0])
            indices = torch.randint(0, x.shape[0], (n,))        
            x[indices,6:] = torch.flip(x[indices,6:], dims=[1])  ##????
        # return x.squeeze(dim=0)
        return x


class GotchaNormalize(nn.Module):
    """ Normalize only the doppler vector

    Args:
        nn (_type_): _description_
    """
    def __init__(self,method='std_mean'):
        super().__init__()
        self.method = method

    def forward(self, x):

        if self.method == 'std_mean':
            if x.ndim == 1:
                # For 1D vector
                mean = x.mean()
                std = x.std()
                std = std + 1e-7
                x = (x- mean) / std
            elif x.ndim == 2:
                # For 2D array
                mean = x.mean(dim=-1, keepdim=True)
                std = x.std(dim=-1, keepdim=True)
                std = std + 1e-7
                x = (x - mean) / std
            return x

        elif self.method == 'min_max':
            if x.ndim == 1:
                # For 1D vector
                min_val = x.min()
                max_val = x.max()
                max_val = max_val - min_val + 1e-7
                x = (x - min_val) / max_val
            elif x.ndim == 2:
                # For 2D array
                min_val = x.min(dim=-1, keepdim=True)[0]
                max_val = x.max(dim=-1, keepdim=True)[0]
                max_val = max_val - min_val + 1e-7
                x = (x - min_val) / max_val
            return x

class LowerSnr(nn.Module):
    """TODO

    Args:
        nn (_type_): _description_
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self,x):
        pass


class AddNoise(nn.Module):
    """TODO

    Args:
        nn (_type_): _description_
    """
    def __init__(self, p=0.5,method='speckle', vector_offset=6):
        super().__init__()
        self.p = p
        self.method = method
        self.speckle_variance = 0.1
        self.vector_offset = vector_offset

    def forward(self,x):
        if (torch.rand(1) > self.p):
            return x
        
        if self.method=='speckle':
            if x.ndim == 1:
                speckle = torch.normal(1, self.speckle_variance, x[self.vector_offset :].shape)
                x[self.vector_offset :] = x[self.vector_offset :] * speckle
            elif x.ndim == 2:
                speckle = torch.normal(1, self.speckle_variance, x[:,6:].shape)
                x[:,self.vector_offset :] = x[:,self.vector_offset :] * speckle 
        elif self.method=='gaussian':
            pass
        
        return x


if __name__ == "__main__":
    x = torch.rand(1000,134)
    flip = FlipAugmentation()
    y = flip(x)

    x = torch.rand(1000,134)
    norm = GotchaNormalize('min_max')
    y = norm(x)

    pass