import torch.nn as nn

class MultiTest(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(input_dim, 200)
        self.relu    = nn.ReLU()
        # two heads: alpha and beta for instance
        self.alpha_head = nn.Linear(200, 256)
        self.beta_head  = nn.Linear(200, 256)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return {
            "alpha": nn.functional.softmax(self.alpha_head(x), dim=1),
            "beta":  nn.functional.softmax(self.beta_head(x),  dim=1),
        }
