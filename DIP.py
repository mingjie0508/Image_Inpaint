from UNet import UNet

class DIP(UNet):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__(in_channels, out_channels)
    
    def get_loss(self, output, target, mask):
        return self.criterion(mask*output, mask*target)
