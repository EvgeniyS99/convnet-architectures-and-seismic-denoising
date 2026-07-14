# Seismic Denoising

Experiments with convolutional neural networks for removing noise from seismic images.

The notebooks cover data preparation, model training, inference, and quality evaluation using several U-Net variants.

## Contents

- `Denoise_simple_unet.ipynb` and `Denoise_simple_unet_800_iters.ipynb` — baseline U-Net experiments.
- `smaller_unet.ipynb` and `bigger_unet.ipynb` — comparisons of model capacity.
- `Denoise_res_unet.ipynb` — residual U-Net experiment.
- `prenoise.ipynb` — input noise preparation.
- `denoise_results.ipynb` — result analysis and visualization.
- `seismic_utils.py` — shared seismic-data utilities.
