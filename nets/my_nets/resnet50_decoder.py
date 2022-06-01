import torch.nn as nn
import torch


class Resnet50Decoder(nn.Module):
    def __init__(self, inplanes, bn_momentum=0.1, plan=0):
        super(Resnet50Decoder, self).__init__()
        self.bn_momentum = bn_momentum
        self.inplanes = inplanes
        self.plan = plan
        self.deconv_with_bias = False

        self.deconv_layer1 = nn.ConvTranspose2d(
            in_channels=self.inplanes,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
            bias=self.deconv_with_bias)
        # 简单通过1*1卷积减小通道
        self.change_channel_layer1 = nn.ConvTranspose2d(
            in_channels=1024 + 256,
            out_channels=256,
            kernel_size=1,
            stride=1,
            bias=False)

        self.deconv_layer2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
            bias=self.deconv_with_bias)
        # 简单通过1*1卷积减小通道
        self.change_channel_layer2 = nn.ConvTranspose2d(
            in_channels=512 + 128,
            out_channels=128,
            kernel_size=1,
            stride=1,
            bias=False)

        self.deconv_layer3 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
            bias=self.deconv_with_bias)
        # 简单通过1*1卷积减小通道
        self.change_channel_layer3 = nn.ConvTranspose2d(
            in_channels=256 + 64,
            out_channels=64,
            kernel_size=1,
            stride=1,
            bias=False)

        # ----------------------------------------------------------#
        #   16,16,2048 -> 32,32,256 -> 64,64,128 -> 128,128,64
        #   利用ConvTranspose2d进行上采样。
        #   每次特征层的宽高变为原来的两倍。
        # ----------------------------------------------------------#

    def forward(self, x, x_l1, x_l2, x_l3):
        x = self.deconv_layer1(x)
        x = torch.cat((x, x_l3), 1)
        x = self.change_channel_layer1(x)
        x = self.deconv_layer2(x)
        x = torch.cat((x, x_l2), 1)
        x = self.change_channel_layer2(x)
        x = self.deconv_layer3(x)
        x = torch.cat((x, x_l1), 1)
        x = self.change_channel_layer3(x)

        return x
