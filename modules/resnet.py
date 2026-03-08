import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from modules.GLFF import glff

from modules.SEattn import senet
__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Channel_Enhance(nn.Module):
    def __init__(self,channels):
        super(Channel_Enhance,self).__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.avgpool = nn.AvgPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.se = senet(channels)

    def forward(self, x):
        x_maxpool = self.maxpool(x)
        x_avgpool = self.avgpool(x)
        out = x_avgpool + x_maxpool
        #out = self.se(out)
        out = torch.sigmoid(out)
        return x + x*out



class SpatialAttention(nn.Module):
    """
    CBAM混合注意力机制的空间注意力
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv3d(2, 1, kernel_size=(1,7,7), padding=(0,3,3), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x


class DeconvUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            padding=(0, 0, 0),
            output_padding=(0, 0, 0),
            bias=False
        )

    def forward(self, x):
        return self.deconv(x)

class ConvDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
            bias=False
        )

    def forward(self, x):
        return self.down(x)

def conv3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=(1, stride, stride),
        padding=(0, 1, 1),
        bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.GL_Fusion1 = glff(128, 8, group_split=[8], kernel_sizes=[3])
        self.GL_Fusion2 = glff(128,8,group_split=[8],kernel_sizes=[5])


        self.ca1 = Channel_Enhance(channels=512)
        self.ca2 = Channel_Enhance(channels=256)


        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()

        self.convdup = DeconvUp(256,256)
        self.convdup1 = DeconvUp(128,128)
        self.convdown = ConvDown(128,128)
        #self.down28to7 = Down28to7(512,512)

        self.block1_conv = nn.Sequential(
            nn.Conv3d(64, 256, (1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(256),
            #nn.ReLU(inplace=True)
        )
        # 128 28 28
        self.block2_conv = nn.Sequential(
            nn.Conv3d(128, 256, (1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(256),
            #nn.ReLU(inplace=True)
        )
        # 256 14 14
        self.block3_conv = nn.Sequential(
            nn.Conv3d(256, 256, (1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(256),
           # nn.ReLU(inplace=True)
        )
        self.block4_conv = nn.Sequential(
            nn.Conv3d(512, 256, (1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(256),
            #nn.ReLU(inplace=True)
        )


        self.smooth1 = nn.Conv3d(256, 128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=1)
        self.smooth2 = nn.Conv3d(256, 128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=1)
        self.smooth3 = nn.Conv3d(256, 128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=1)
        self.smooth4 = nn.Conv3d(256, 512, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=1)
        self.channel_reduce_conv = nn.Conv3d(384, 128, kernel_size=(1, 1, 1))
        self.channel_increase_fe = nn.Conv3d(128, 512, kernel_size=(1, 1, 1))
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)


        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _upsample(self,x,target):
        return F.interpolate(x,size=target.shape[-3:],mode="nearest")


    def forward(self, x):
        N, C, T, H, W = x.size()
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c1 = self.maxpool(c1)

        c2 = self.layer1(c1)

        c3 = self.layer2(c2)

        c4 = self.layer3(c3)

        c5 = self.layer4(c4)

        c5 = self.ca1(c5)
        P5 = self.block4_conv(c5)

        c4 = self.ca2(c4)
       # P4 = self._upsample(P5, c4) + self.block3_conv(c4)
        P4 = self.convdup(P5) + self.block3_conv(c4)

        c3 = self.sa1(c3)
        #P3 = self._upsample(P4, c3) + self.block2_conv(c3)
        P3 = self.convdup(P4) + self.block2_conv(c3)

        c2 = self.sa2(c2)
        #P2 = self._upsample(P3, c2) + self.block1_conv(c2)
        P2 = self.convdup(P3) + self.block1_conv(c2)


        P5 = self.smooth4(P5)
        #print(P5.shape)
        P4 = self.smooth1(P4)
        P3 = self.smooth2(P3)
        P2 = self.smooth3(P2)

        #P5 = self._upsample(P5, P3)
        P4 = self.convdup1(P4)
       # P3 = self._upsample(P3, P2)
        P2 = self.convdown(P2)

        P_out = torch.cat([P2, P3, P4], dim=1)  # 512x56x56
        P_out = self.channel_reduce_conv(P_out)  ##128x56x56
        P_out1 = self.GL_Fusion1(P_out)
        P_out2 = self.GL_Fusion2(P_out)
        P_out = self.alpha * P_out1 + (1 - self.alpha) * P_out2
        P_out = self.channel_increase_fe(P_out) #512x56x56

        P_out = self._upsample(P_out,c5)  ##512x7x7
        #print(P_out.shape)
        P5 = self._upsample(P5,c5)
        x = P_out  + c5  +P5

        #print(x.shape)
        x = x.transpose(1, 2).contiguous()
        x = x.view((-1,) + x.size()[2:])  # bt,c,h,w

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # bt,c
        x = self.fc(x)  # bt,c

        return x

def resnet18(**kwargs):
    """Constructs a ResNet-18 based model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet18'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name :
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    model.load_state_dict(checkpoint, strict=False)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def test():
    net = resnet18()
    y = net(torch.randn(1,3,224,224))
    print(y.size())

#test()
if __name__ == "__main__":

    # 输入尺寸：Batch=1, Channel=3, Time=8, Height=224, Width=224
    #model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=32)  # ResNet18 结构
    model = Channel_Enhance(64)
    # 创建一个随机视频输入 (B, C, T, H, W)
    input_tensor = torch.rand(1, 64, 8, 224, 224)

    # 将模型设为 eval 模式
    model.eval()

    # 前向传播
    with torch.no_grad():
         x= model(input_tensor)

    # 打印每个 pyramid 特征的 shape
    # print("P2 shape:", P2.shape)  # [B, 256, T, H/4, W/4]  或者根据上采样后尺寸
    # print("P3 shape:", P3.shape)  # [B, 256, T, H/8, W/8]
    # print("P4 shape:", P4.shape)  # [B, 256, T, H/16, W/16]
    # print("P5 shape:", P5.shape)  # [B, 256, T, H/32, W/32]
    print("P_out shape:", x.shape)