from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url


class Basicblock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample=None
    ):
        super(Basicblock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)

        return out
    

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        stride: int = 1,
        downsample=None
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=mid_channels * self.expansion,
            kernel_size=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))

        out = self.relu(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))
        out += residual
        out = self.relu(out)

        return out


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    

class ResNet(nn.Module):

    def __init__(
        self,
        block: Union[Bottleneck, Basicblock],
        layers: List[int]
    ):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.downsample_ratio = 32
        self.feat_channels = 512 * block.expansion

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(
        self,
        block: Union[Basicblock, Bottleneck],
        mid_channels: int,
        num_blocks: int,
        stride: int = 1
    ):
        downsample = None
        if stride != 1 or self.in_channels != mid_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, mid_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(mid_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, mid_channels, stride, downsample))
        self.in_channels = mid_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, mid_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNetFPN(nn.Module):

    def __init__(
        self,
        block: Union[Bottleneck, Basicblock],
        layers: List[int]
    ):
        super(ResNetFPN, self).__init__()
        self.downsample_ratio = 4
        self.feat_channels = 64 * block.expansion
        self.base = ResNet(block, layers)

        # Top layer
        self.toplayer = nn.Conv2d(
            in_channels=512 * block.expansion,
            out_channels=64 * block.expansion,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Lateral layers
        self.latlayer1 = nn.Conv2d(
            in_channels=256 * block.expansion,
            out_channels=64 * block.expansion,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.latlayer2 = nn.Conv2d(
            in_channels=128 * block.expansion,
            out_channels=64 * block.expansion,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.latlayer3 = nn.Conv2d(
            in_channels=64 * block.expansion,
            out_channels=64 * block.expansion,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Smooth layers
        self.smooth = nn.Conv2d(
            in_channels=64 * block.expansion,
            out_channels=64 * block.expansion,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def _upsample_add(self, x: torch.Tensor, y: torch.Tensor):
        _, _, H, W = y.shape
        return F.interpolate(x, size=(H, W), mode="bilinear") + y
    
    def forward(self, x: torch.Tensor):
        # Bottom-up
        c1 = self.base.relu(self.base.bn1(self.base.conv1(x)))
        c1 = self.base.maxpool(c1)
        c2 = self.base.layer1(c1)
        c3 = self.base.layer2(c2)
        c4 = self.base.layer3(c3)
        c5 = self.base.layer4(c4)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p2 = self.smooth(p2)
        return p2


_resnet_weight_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
}

_resnet_specs = {
    "resnet18": (Basicblock, [2, 2, 2, 2]),
    "resnet34": (Basicblock, [3, 4, 6, 3]),
    "resnet50": (Bottleneck, [3, 4, 6, 3]),
    "resnet101": (Bottleneck, [3, 4, 23, 3]),
    "resnet152": (Bottleneck, [3, 8, 36, 3]),
}

def build_resnet(depth: int = 18, pretrained: bool = False):
    model_name = "resnet{}".format(depth)
    model = ResNet(*_resnet_specs[model_name])
    if pretrained:
        state_dict = load_url(
            url=_resnet_weight_urls[model_name],
            model_dir="./checkpoints",
        )
        model.load_state_dict(state_dict, strict=False)
    return model


def build_resnet_fpn(depth: int = 18):
    resnet_name = "resnet{}".format(depth)
    resnet_spec = _resnet_specs[resnet_name]
    model = ResNetFPN(*resnet_spec)
    return model
