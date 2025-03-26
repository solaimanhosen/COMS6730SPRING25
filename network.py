import torch.nn as nn

class BasicBlockV1(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlockV1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class BottleneckBlockV2(nn.Module):
    expansion = 4

    def __init__(self, in_channels, base_channels, stride=1):
        super(BottleneckBlockV2, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=1, stride=1, bias=False)

        self.bn2 = nn.BatchNorm2d(base_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(base_channels)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(base_channels, base_channels * self.expansion, kernel_size=1, stride=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != base_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, base_channels * self.expansion, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        out += self.shortcut(identity)
        return out

class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        if args.resnet_version == 1:
            block = BasicBlockV1
        else:
            block = BottleneckBlockV2

        self.in_channels = 16  # Initial conv layer output channels
        num_blocks = [args.resnet_size] * 3  # Number of blocks per stage

        # Initial layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Residual layers
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, 10)

    def _make_layer(self, block, base_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, base_channels, s))
            self.in_channels = base_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
