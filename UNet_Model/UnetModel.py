from UNet_Model.Encoder import UNetEncoderBlock
from UNet_Model.Decoder import UNetDecoderBlock
import torch.nn as nn
class UNetModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        input_channels: int = 3, 
        n_filters_list: list[int] = [64,128,256,512,1024]
    ) -> None: 
        super().__init__()
        self.encoder_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()

        # xây dựng encoder
        in_channels = input_channels
        for filters in n_filters_list:
            self.encoder_list.append(UNetEncoderBlock(in_channels, filters))
            in_channels = filters

        # xây dựng decoder 
        n_filters_list = n_filters_list[::-1]
        for filters in n_filters_list[1:]:
            self.decoder_list.append(UNetDecoderBlock(in_channels, filters))
            in_channels = filters

        # lớp cuối cùng
        self.final_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride = 1)
    def forward(self, x):
        skip_connect_list = []
        for i in range(len(self.encoder_list)):
            if i < len(self.encoder_list)-1:
                x, skip_connect = self.encoder_list[i](x)
                skip_connect_list.append(skip_connect)
            else:
                # Bottleneck: Không thực hiện downsampling, lấy feature trước pool làm input cho Decoder Block đầu tiên
                _, x = self.encoder_list[i](x)
        
        skip_connect_list = skip_connect_list[::-1]
        for i in range(len(self.decoder_list)):
            x = self.decoder_list[i](x, skip_connect_list[i])
        x = self.final_conv(x)
        return x 