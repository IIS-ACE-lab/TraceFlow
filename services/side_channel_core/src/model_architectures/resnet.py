import torch.nn as nn

class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)

        # if in and out dims differ, use a 1×1 conv to match
        self.downsample = None
        if stride!=1 or in_ch!=out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample:
            identity = self.downsample(identity)
        out += identity
        return self.relu(out)

class ResNet1D(nn.Module):
    """
    depth should be 18n+1 (e.g. 19, 37, 55)
    """
    def __init__(self, in_channels: int, depth: int, num_classes: int):
        super().__init__()
        # compute number of blocks per layer
        n = (depth - 1) // 18
        self.in_ch = 16
        # initial conv
        self.conv = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        # layers
        self.layer1 = self._make_layer(16,  n, stride=1)
        self.layer2 = self._make_layer(32,  n, stride=2)
        self.layer3 = self._make_layer(64,  n, stride=2)
        # global pool + classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc      = nn.Linear(64, num_classes)

    def _make_layer(self, out_ch, blocks, stride):
        layers = [BasicBlock1D(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch
        for _ in range(blocks - 1):
            layers.append(BasicBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x); x = self.bn(x); x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x).squeeze(-1)  # [B, C]
        return self.fc(x)
