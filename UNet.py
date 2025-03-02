import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # Encoder
        self.e11 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1) # shape (64, 96, 96)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)      # shape (64, 96, 96)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)          # shape (64, 48, 48)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)     # shape (128, 48, 48)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)    # shape (128, 48, 48)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)          # shape (128, 24, 24)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)    # shape (256, 24, 24)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)    # shape (256, 24, 24)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)          # shape (256, 12, 12)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)    # shape (512, 12, 12)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)    # shape (512, 12, 12)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)          # shape (512, 6, 6)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)   # shape (1024, 6, 6)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)  # shape (1024, 6, 6)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)   # shape (512, 12, 12)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)   # shape (512, 12, 12)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)    # shape (512, 12, 12)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)    # shape (256, 24, 24)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)    # shape (256, 24, 24)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)    # shape (256, 24, 24)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)    # shape (128, 48, 48)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)    # shape (128, 48, 48)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)    # shape (128, 48, 48)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)     # shape (256, 96, 96)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)     # shape (256, 96, 96)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)      # shape (256, 96, 96)

        # Output layer
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)   # shape (3, 96, 96)
        self.relu = nn.ReLU()

        self.criterion = nn.L1Loss()

    def forward(self, x):
        # Encoder
        xe11 = self.relu(self.e11(x))
        xe12 = self.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = self.relu(self.e21(xp1))
        xe22 = self.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = self.relu(self.e31(xp2))
        xe32 = self.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = self.relu(self.e41(xp3))
        xe42 = self.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = self.relu(self.e51(xp4))
        xe52 = self.relu(self.e52(xe51))
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = self.relu(self.d11(xu11))
        xd12 = self.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = self.relu(self.d21(xu22))
        xd22 = self.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = self.relu(self.d31(xu33))
        xd32 = self.relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = self.relu(self.d41(xu44))
        xd42 = self.relu(self.d42(xd41))

        # Output layer
        output = self.outconv(xd42)

        return output
    
    def get_loss(self, output, target, mask):
        unmask = 1 - mask
        return (self.criterion(mask*output, mask*target) +
                self.criterion(unmask*output, unmask*target))
