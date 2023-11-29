import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Callable, List, Optional, Type, Union

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )
class BasicBlock(nn.Module):

    expansion: int = 1
    def __init__(self,
         inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class _ResNetLight(nn.Module):
    def __init__(self,         
                block: BasicBlock,
                layers: List[int],
                 num_classes: int = 1000

        ) -> None:
        super().__init__()

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 1, num_classes)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(
        self,         
        block: BasicBlock,
        planes: int,
        blocks: int,
        stride: int = 1,
        downsample = None
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        if stride != 1 or self.inplanes != planes :
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes , stride),
                norm_layer(planes ),
            )
        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, norm_layer
            )
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
def ResNetLight(num_classes: int):
    return _ResNetLight(BasicBlock, [2, 2, 2, 2], num_classes)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, num_conv_layers=2, dropout_prob=0.5):
        super(SimpleCNN, self).__init__()
        self.num_conv_layers = num_conv_layers

        # First Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Convolutional layers before the first pooling
        self.conv_before_pool1 = nn.ModuleList()
        self.bn_before_pool1 = nn.ModuleList()
        for _ in range(self.num_conv_layers):
            self.conv_before_pool1.append(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1))
            self.bn_before_pool1.append(nn.BatchNorm2d(16))

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second Convolutional Block
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Convolutional layers after the first pooling
        self.conv_after_pool1 = nn.ModuleList()
        self.bn_after_pool1 = nn.ModuleList()
        for _ in range(self.num_conv_layers):
            self.conv_after_pool1.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1))
            self.bn_after_pool1.append(nn.BatchNorm2d(32))


        # Second Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Third Convolutional Block
        self.bn3 = nn.BatchNorm2d(16)
        self.conv3_layers = nn.ModuleList()
        for _ in range(self.num_conv_layers):
            self.conv3_layers.append(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1))

        self.conv4 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        # Fourth Convolutional Block
        self.bn4 = nn.BatchNorm2d(8)
        self.conv4_layers = nn.ModuleList()
        for _ in range(self.num_conv_layers):
            self.conv4_layers.append(nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1))

        # Fully Connected Layers
        self.fc1 = nn.Linear(8 * 13 * 13, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout Layers
        self.dropout1 = nn.Dropout(dropout_prob / 2)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Convolutional layers before the first pooling
        x = F.relu(self.dropout(self.bn1(self.conv1(x))))

        residual1 = x
        for i in range(self.num_conv_layers):
            x = F.relu(self.dropout1(self.conv_before_pool1[i](self.bn_before_pool1[i](x))))

        x = x + residual1
        x = F.relu(self.dropout1(self.bn2(self.conv2(x))))
        x = self.pool1(x)

        # Convolutional layers after the first pooling
        residual2 = x
        for i in range(self.num_conv_layers):
            x = F.relu(self.dropout(self.conv_after_pool1[i](self.bn_after_pool1[i](x))))
        x = x + residual2
        x = self.pool1(x)
        x = F.relu(self.dropout(self.conv3(self.bn2(x))))

        # Convolutional layers before the second pooling
        residual3 = x
        for i in range(self.num_conv_layers):
            x = F.relu(self.dropout(self.conv3_layers[i](self.bn3(x))))
        x = x + residual3
        x = self.pool1(x)
        x = F.relu(self.dropout(self.conv4(self.bn3(x))))

        # Convolutional layers before the third pooling
        residual4 = x
        for i in range(self.num_conv_layers):
            x = F.relu(self.dropout(self.conv4_layers[i](self.bn4(x))))
        x = x + residual4
        x = self.pool1(x)

        x = x.view(-1, 8 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, x):
        out = self.forward(x)
        return torch.argmax(out), torch.max(F.softmax(out))