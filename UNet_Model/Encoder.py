import torch.nn as nn
class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels: int = None, out_channels: int = None) -> None:
        super().__init__()
        self.list_module = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
    def forward(self, x):
        for i in range(len(self.list_module) - 1):
            x = self.list_module[i](x)
        skip_connect = x.clone()
        x = self.list_module[-1](x)
        return x, skip_connect
