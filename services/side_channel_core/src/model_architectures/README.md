# services/side_channel_core/src/model_architectures
Hard-coded PyTorch networks that `core/model.py` exposes as single layer types (e.g. `ResNetV1`, `ResNet50_1D`, `InceptionV3_1D`, `MultiTest`, `MultiResNet`, `MultiResNetNoPermind`) or loads by name via `model_architectures.<architecture>`.

## Contents
```text
model_architectures/
├── __init__.py          # package marker (empty)
├── multi_resnet.py      # MultiResNet, MultiResNetNoPermind (ResNet1D subclasses)
├── multi_test.py        # MultiTest small test network
├── public_cnn_1d.py     # 1-D ResNet50_1D and InceptionV3_1D (ASCAD paper baselines)
├── resnet.py            # BasicBlock1D, ResNet1D (registered as ResNetV1)
├── README.md            # this file
└── README.pdf           # old rendered README
```

## Used by
- `src/core/model.py` (imports all four modules; dynamic import `model_architectures.<architecture>`).
- `multi_resnet.py` imports `resnet.py`.

## TODOs
None.
