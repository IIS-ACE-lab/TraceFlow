#services/side_channel_core/utils/flat_size.py

'''
If you run the hard coded CNN architectures, you will need to know the flattened feature size before the Linear layer.
This script calculates the flattened feature size for the ASCAD_cnn_best2.yaml architecture and updates the YAML file accordingly.
This is necessary because the in_features parameter for the Linear layer must match the flattened feature size.

Usage (paths are relative to this checkout, so it runs from any clone):
    python services/side_channel_core/utils/flat_size.py [path/to/architecture.yaml]


Author: Shraddha Mishra
'''

import sys, yaml, torch
from pathlib import Path

# services/side_channel_core and its src/ make core.model and its imports resolvable.
SCC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCC_ROOT / "src"))
sys.path.insert(0, str(SCC_ROOT))
from core.model import build_model

ARCH_PATH = (
    Path(sys.argv[1]).resolve()
    if len(sys.argv) > 1
    else SCC_ROOT / "config" / "architectures" / "ASCAD_cnn_best2.yaml"
)

# Load config
cfg = yaml.safe_load(ARCH_PATH.open())

# Temporarily set the in_features string to 1 to avoid crash
for layer in cfg["model"]["layers"]:
    if "Linear" in layer:
        layer["Linear"]["in_features"] = 1  # temporary placeholder
        break

# Build model up to flatten. Only shapes matter here: the model-only config
# has no deep_learning block, so it builds with PyTorch's default init.
model = build_model(cfg["model"])
x = torch.randn(1, 1, 700)

# Get flattened output size before Linear
with torch.no_grad():
    modules = list(model.children())
    pre_flatten = []
    for m in modules:
        pre_flatten.append(m)
        if isinstance(m, torch.nn.Flatten):
            break
    conv_net = torch.nn.Sequential(*pre_flatten)
    out = conv_net(x)
    flat_size = out.numel()

print(f"Flattened feature size before Linear: {flat_size}")

# Update the YAML file with correct in_features
for layer in cfg["model"]["layers"]:
    if "Linear" in layer:
        layer["Linear"]["in_features"] = flat_size
        break

# Save updated config
yaml.safe_dump(cfg, ARCH_PATH.open("w"))
print(f"Updated {ARCH_PATH.name} with in_features = {flat_size}")
