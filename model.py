# implement U-net here
import torch
import torch.nn as nn



class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels=32):
        super(UNet, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels*2)
        self.contract3 = ContractingBlock(hidden_channels*4)
        self.contract4 = ContractingBlock(hidden_channels*8)
        self.expand1 = ExpandingBlock(hidden_channels*16)
        self.expand2 = ExpandingBlock(hidden_channels*8)
        self.expand3 = ExpandingBlock(hidden_channels*4)
        self.expand4 = ExpandingBlock(hidden_channels*2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)

    def forward(self, x):
        pass


class FeatureMapBlock(nn.Module):
    """Reduce or increase  # channels of a given tensor"""
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.model = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        return self.model(x)


class ContractingBlock(nn.Module):
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


class ExpandingBlock(nn.Module):
    """Increase height and width of a given tensor"""
    def __init__(self, input_channels):
        super(ExpandingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear",
            align_corners=True)
        self.conv1 = nn.Conv2d(input_channels, input_channels//2, kernel_size=2)
        self.conv2 = nn.Conv2d(input_channels, input_channels//2, kernel_size=3)
        self.conv3 = nn.Conv2d(input_channels//2, input_channels//2, kernel_size=3)
        self.relu = nn.ReLU()

    def crop(self, tensor_images, new_shape):
        h = new_shape[2]
        w = new_shape[3]
        h_start = int((tensor_images.shape[2] - h) / 2)
        w_start = int((tensor_images.shape[3] - w) / 2)
        cropped_images = tensor_images[:, :, h_start:h_start+h, w_start:w_start+w]

        return cropped_images

    def forward(self, x, skip_x):
        x = self.upsample(x)
        x = self.conv1(x)
        skip_x_cropped = self.crop(skip_x, x.shape)
        x = torch.cat([x, skip_x_cropped], dim=1)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)

        return x
