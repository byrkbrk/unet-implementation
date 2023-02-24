# implement U-net here
import torch
import torch.nn as nn



class UNet(nn.Module):
    def __init__(self, input_channels, out_channels, hidden_channels=32):
        super(UNet, self).__init__()
        pass

    def forward(self, x):
        pass


def FeatureMapBlock(nn.Module):
    """Reduce or increase  # channels of a given tensor"""
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.model = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        return self.model(x)


def ContractingBlock(nn.Module):
    """Reduce height and width of a given tensor"""
    def __init__(self, input_channels):
        super(ContractingBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, input_channels*2, kernel_size=3),
            nn.Conv2d(input_channels*2, input_channels*2, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.model(x)
