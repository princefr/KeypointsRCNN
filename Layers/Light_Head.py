import torch
import torchvision
from torch import nn


class LightHead(torch.nn.Module):
    def __init__(self, in_, backbone, mode="S", c_out=10):
        super(LightHead, self).__init__()
        self.backbone = backbone
        if mode == "L":
            self.out_mode = 256
        else:
            self.out_mode = 64
        self.conv1 = torch.nn.Conv2d(in_channels=in_, out_channels=self.out_mode, kernel_size=(15, 1), stride=1,
                                     padding=(7, 0), bias=True)
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv2 = torch.nn.Conv2d(in_channels=self.out_mode, out_channels=c_out, kernel_size=(1, 15),  stride=1,
                                     padding=(0, 7), bias=True)
        self.conv3 = torch.nn.Conv2d(in_channels=in_, out_channels=self.out_mode, kernel_size=(15, 1), stride=1, padding=(7, 0), bias=True)
        self.conv4 = torch.nn.Conv2d(in_channels=self.out_mode, out_channels=c_out, kernel_size=(1, 15), stride=1,
                                     padding=(0, 7), bias=True)

    def forward(self, input):
        x_backbone = self.backbone(input)
        x = self.conv1(x_backbone)
        x = self.relu(x)
        x = self.conv2(x)
        x_relu_2 = self.relu(x)

        x = self.conv3(x_backbone)
        x = self.relu(x)
        x = self.conv4(x)
        x_relu_4 = self.relu(x)

        return x_relu_2 + x_relu_4