from .resnet import ResNet1D
import torch.nn as nn

class MultiResNet(ResNet1D):
    def __init__(self, in_channels: int, depth: int):
        super().__init__(in_channels, depth, num_classes=256)
        # reuse the final feature vector for multiple heads:
        self.alpha_head = nn.Linear(256, 256)
        self.beta_head  = nn.Linear(256, 256)
        # e.g. 16 S-box heads:
        self.sbox_heads = nn.ModuleList([nn.Linear(256, 256) for _ in range(16)])
        self.perm_heads = nn.ModuleList([nn.Linear(256,  16) for _ in range(16)])

    def forward(self, x):
        features = super().forward(x)  # [B,256]
        out = {
            "alpha": nn.functional.softmax(self.alpha_head(features), dim=1),
            "beta":  nn.functional.softmax(self.beta_head(features),  dim=1),
        }
        for i, head in enumerate(self.sbox_heads):
            out[f"sbox_{i}"] = nn.functional.softmax(head(features), dim=1)
        for i, head in enumerate(self.perm_heads):
            out[f"perm_{i}"] = nn.functional.softmax(head(features), dim=1)
        return out

class MultiResNetNoPermind(MultiResNet):
    def forward(self, x):
        d = super().forward(x)
        # drop perm outputs
        return {k:v for k,v in d.items() if not k.startswith("perm_")}
