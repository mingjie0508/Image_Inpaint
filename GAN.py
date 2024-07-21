import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from UNet import UNet


class Discriminator(nn.Module):
    REAL = 1.0
    FAKE = 0.0

    def __init__(self, in_channels, out_channels=1):
        super(Discriminator, self).__init__()
        self.discriminator = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.discriminator.fc = nn.Linear(512, 1)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, x):
        return self.discriminator(x)
    
    def get_loss(self, output, label):
        return self.criterion(output, label)
