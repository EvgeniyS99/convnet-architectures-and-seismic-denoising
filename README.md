# ConvNet Architectures and Seismic Denoising

This project explores convolutional neural network architectures implemented from PyTorch building blocks, without relying on pretrained model wrappers.

It covers image classification, semantic segmentation, and an applied seismic denoising task using several U-Net variants. The repository includes custom training and inference utilities, architectural experiments, and quantitative evaluation in both spatial and frequency domains.

## Projects

- [`models_from_scratch`](models_from_scratch) — CNN implementations, training utilities, and architecture experiments with AlexNet, VGG, ResNet, ResNeXt, SE-ResNet, EfficientNet, and U-Net.
- [`segmentation_notebooks`](segmentation_notebooks) — semantic segmentation experiments with U-Net, residual U-Net, and DeepLabV3.
- [`seismic_denoising`](seismic_denoising) — seismic image denoising experiments with baseline, residual, smaller, and larger U-Net variants.
