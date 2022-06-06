import torch.nn as nn
import torch


class Decoder(nn.Module):
    def __init__(self, inplanes, bn_momentum=0.1, plan=0):
        super(Decoder, self).__init__()
        self.bn_momentum = bn_momentum
        self.inplanes = inplanes
        self.plan = plan
        self.deconv_with_bias = False

        self.deconv_layer1 = None
        self.change_channel_layer1 = None  # 简单通过1*1卷积减小通道
        self.deconv_layer2 = None
        self.change_channel_layer2 = None  # 简单通过1*1卷积减小通道
        self.deconv_layer3 = None
        self.change_channel_layer3 = None  # 简单通过1*1卷积减小通道

        # 不同维度上下采用层
        self.layer1_layer2 = None
        self.layer1_layer1 = None
        self.layer2_layer1 = None
        self.layer2_layer3 = None
        self.layer3_layer2 = None
        self.layer3_layer3 = None

        self.make_layers()

        # ----------------------------------------------------------#
        #   16,16,2048 -> 32,32,256 -> 64,64,128 -> 128,128,64
        #   利用ConvTranspose2d进行上采样。
        #   每次特征层的宽高变为原来的两倍。
        # ----------------------------------------------------------#

    def forward(self, x, x_l1=None, x_l2=None, x_l3=None):
        if self.plan == 0:
            x = self.forward_plan_0(x)
        elif self.plan == 10:
            x = self.forward_plan_10(x, x_l1, x_l2, x_l3)
        elif self.plan == 11 or self.plan == 12:
            x = self.forward_plan_11(x, x_l1, x_l2, x_l3)

        return x

    def make_layers(self):
        if self.plan == 0:
            self.layers_plan_0()
        elif self.plan == 10:
            self.layers_plan_10()
        elif self.plan == 11 or self.plan == 12:
            self.layers_plan_11()

    # ##### 网络结构 #####
    def layers_plan_0(self):
        self.deconv_layer1 = nn.ConvTranspose2d(in_channels=self.inplanes, out_channels=256, kernel_size=4,
                                                stride=2, padding=1, output_padding=0, bias=self.deconv_with_bias)
        self.deconv_layer2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2,
                                                padding=1, output_padding=0, bias=self.deconv_with_bias)
        self.deconv_layer3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2,
                                                padding=1, output_padding=0, bias=self.deconv_with_bias)

    def layers_plan_10(self):
        self.layers_plan_0()

        # 1*1卷积减小通道
        self.change_channel_layer1 = nn.ConvTranspose2d(in_channels=1024 + 256, out_channels=256, kernel_size=1,
                                                        stride=1, bias=False)
        self.change_channel_layer2 = nn.ConvTranspose2d(in_channels=512 + 128, out_channels=128, kernel_size=1,
                                                        stride=1, bias=False)
        self.change_channel_layer3 = nn.ConvTranspose2d(in_channels=256 + 64, out_channels=64, kernel_size=1,
                                                        stride=1, bias=False)

    def layers_plan_11(self):
        self.layers_plan_0()

        # 1*1卷积减小通道
        self.change_channel_layer1 = nn.ConvTranspose2d(in_channels=1024 + 256 + 256 + 512, out_channels=256,
                                                        kernel_size=1, stride=1, bias=False)
        self.change_channel_layer2 = nn.ConvTranspose2d(in_channels=512 + 128 + 1024 + 256, out_channels=128,
                                                        kernel_size=1, stride=1, bias=False)
        self.change_channel_layer3 = nn.ConvTranspose2d(in_channels=256 + 64 + 1024 + 512, out_channels=64,
                                                        kernel_size=1, stride=1, bias=False)

        # 上下采样
        self.layer1_layer2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1_layer1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer2_layer1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        if self.plan == 11:
            self.layer2_layer3 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4,
                                                stride=2, padding=1, output_padding=0, bias=False)
        elif self.plan == 12:
            self.layer2_layer3 = nn.Upsample(scale_factor=2)

        if self.plan == 11:
            self.layer3_layer2 = nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=4,
                                                    stride=2, padding=1, output_padding=0, bias=False)
        elif self.plan == 12:
            self.layer3_layer2 = nn.Upsample(scale_factor=2)

        if self.plan == 11:
            self.layer3_layer3 = nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=4,
                                                    stride=2, padding=1, output_padding=0, bias=False)
        elif self.plan == 12:
            self.layer3_layer3 = nn.Upsample(scale_factor=2)

    # ##### 网络forward #####
    def forward_plan_0(self, x):
        x = self.deconv_layer1(x)
        x = self.deconv_layer2(x)
        x = self.deconv_layer3(x)
        return x

    def forward_plan_10(self, x, x_l1, x_l2, x_l3):
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

    def forward_plan_11(self, x, x_l1, x_l2, x_l3):
        # 上下采样
        l1_l2 = self.layer1_layer2(x_l1)
        l1_l1 = self.layer1_layer1(l1_l2)

        l2_l1 = self.layer2_layer1(x_l2)
        l2_l3 = self.layer2_layer3(x_l2)

        l3_l2 = self.layer3_layer2(x_l3)
        l3_l3 = self.layer3_layer3(l3_l2)

        # 解码 拼接
        x = self.deconv_layer1(x)
        x = torch.cat((x, x_l3, l1_l1, l2_l1), 1)
        x = self.change_channel_layer1(x)

        x = self.deconv_layer2(x)
        x = torch.cat((x, x_l2, l1_l2, l3_l2), 1)
        x = self.change_channel_layer2(x)

        x = self.deconv_layer3(x)
        x = torch.cat((x, x_l1, l2_l3, l3_l3), 1)
        x = self.change_channel_layer3(x)

        return x
