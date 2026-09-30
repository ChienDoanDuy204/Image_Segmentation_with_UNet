import torch
import torch.nn as nn
class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels: int = None, out_channels: int = None ) -> None: 
        super().__init__()
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.BNorm1 = nn.BatchNorm2d(num_features=out_channels)
        self.BNorm2 = nn.BatchNorm2d(num_features=out_channels)
        self.activate_fn = nn.GELU()
    def forward(self, x, skip_connect):
        x = self.transpose_conv(x)
        x = torch.cat([x, skip_connect], dim=1)
        x = self.activate_fn(self.BNorm1(self.conv1(x)))
        x = self.activate_fn(self.BNorm2(self.conv2(x)))
        return x 