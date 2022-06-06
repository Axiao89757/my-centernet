import math
from torch import nn
from nets.sub_nets.resnet import ResNet
from nets.my_sub_nets.resnet50_decoder import Resnet50Decoder
from nets.my_sub_nets.resnet50_head import Resnet50Head
from nets.my_sub_nets.bottleneck import Bottleneck
from torch.hub import load_state_dict_from_url

model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


class CenterNetDenseConnection(nn.Module):
    def __init__(self, num_classes=1, backbone_pretrained=False, plan=0):
        """
        初始化一个密集连接的CenterNet
        :param num_classes: 类别数量
        :param backbone_pretrained: backbone是否预训练
        :param plan: 密集连接的计划，0：无密集连接；1：同尺寸跳接；2：不同尺度上下采样跳接
        """

        super(CenterNetDenseConnection, self).__init__()
        self.backbone_pretrained = backbone_pretrained
        self.plan = plan

        # <editor-folder desc="backbone: 512,512,3 -> 16,16,2048">
        # self.backbone = create_resnet50(pretrained=backbone_pretrained, plan=self.plan)
        self.backbone = ResNet(Bottleneck, [3, 4, 6, 3], plan=self.plan)
        if self.backbone_pretrained:
            state_dict = load_state_dict_from_url(model_urls['resnet50'], model_dir='model_data/')
            self.backbone.load_state_dict(state_dict)
        self.decoder = Resnet50Decoder(2048, plan=self.plan)
        # </editor-fold>

        # <editor-folder desc="header: 对获取到的特征进行上采样，进行分类预测和回归预测">
        # 128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
        #              -> 128, 128, 64 -> 128, 128, 2
        #              -> 128, 128, 64 -> 128, 128, 2
        self.head = Resnet50Head(channel=64, num_classes=num_classes)
        # </editor-fold>

        self._init_weights()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _init_weights(self):
        if not self.backbone_pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        self.head.cls_head[-1].weight.data.fill_(0)
        self.head.cls_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        x, x_l1, x_l2, x_l3 = self.backbone(x)
        x = self.decoder(x, x_l1=x_l1, x_l2=x_l2, x_l3=x_l3)
        return self.head(x)
