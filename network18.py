import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        self.args = args
        
        # Standard ResNet-18 configuration
        if args.resnet_version == 1 and args.resnet_size == 18:
            self.stage_blocks = [2, 2, 2, 2]  # Four stages with 2 blocks each
            self.stage_filters = [64, 128, 256, 512]  # Original filter sizes
        else:
            raise ValueError("Unsupported configuration")

        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Four stages
        self.stage1 = self._make_stage(64, 64, self.stage_blocks[0], stride=1)
        self.stage2 = self._make_stage(64, 128, self.stage_blocks[1], stride=2)
        self.stage3 = self._make_stage(128, 256, self.stage_blocks[2], stride=2)
        self.stage4 = self._make_stage(256, 512, self.stage_blocks[3], stride=2)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, args.num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)       # 3x32x32 → 64x16x16
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # 64x8x8

        x = self.stage1(x)      # 64x8x8
        x = self.stage2(x)      # 128x4x4
        x = self.stage3(x)      # 256x2x2
        x = self.stage4(x)      # 512x1x1

        x = self.avgpool(x)     # 512x1x1 → 512x1x1
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x