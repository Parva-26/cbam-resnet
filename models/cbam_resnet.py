
import torch
import torch.nn as nn
from modules.cbam import CBAM


class CBAMBottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, planes, stride=1, downsample=None, reduction_ratio=16):
        super(CBAMBottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)

        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.cbam = CBAM(planes * self.expansion, reduction_ratio)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class CBAMResNet50(nn.Module):

    def __init__(self, num_classes=10, reduction_ratio=16):
        super(CBAMResNet50, self).__init__()

        self.in_channels     = 64
        self.reduction_ratio = reduction_ratio  

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(planes=64,  num_blocks=3, stride=1)
        self.layer2 = self._make_layer(planes=128, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(planes=256, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(planes=512, num_blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * CBAMBottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride):
        downsample = None

        if stride != 1 or self.in_channels != planes * CBAMBottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * CBAMBottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * CBAMBottleneck.expansion)
            )

        blocks = [CBAMBottleneck(self.in_channels, planes, stride,
                                 downsample, self.reduction_ratio)]
        self.in_channels = planes * CBAMBottleneck.expansion

        for _ in range(1, num_blocks):
            blocks.append(CBAMBottleneck(self.in_channels, planes,
                                         reduction_ratio=self.reduction_ratio))

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  

        x = self.layer1(x)                      
        x = self.layer2(x)                      
        x = self.layer3(x)                      
        x = self.layer4(x)                      

        x = self.avgpool(x)                     
        x = torch.flatten(x, 1)                 
        x = self.fc(x)                          

        return x
