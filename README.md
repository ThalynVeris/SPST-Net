# SPST-Net

This repository contains the research implementation and evaluation artifacts of
**SPST-Net** for single-image super-resolution. It accompanies the corresponding
manuscript and is intended to support methodological inspection and verification
of the reported experiments.

## Architecture

SPST-Net combines spatial feature modeling with complementary frequency-domain
processing. Its principal components are:

- **SPIB (Spectral Prior Injection Block):** transforms deep features into the
  Fourier domain, modulates the real and imaginary components through separately
  parameterized branches, recombines the complex spectrum, and injects the
  reconstructed spectral response through residual connections.
- **TAB (Token Aggregation Block):** aggregates content-related tokens to support
  non-local feature interaction.
- **FDCAB (Frequency-Decoupled Cross-Attention Block):** combines dedicated
  structure- and texture-oriented branches. Texture-derived queries interact with
  structure-derived keys and values while the structure representation serves as
  the residual base.

The main architecture is registered as `SPSTNet`, and the associated BasicSR
wrapper is registered as `SPSTNetModel`.

## Repository Structure

```text
basicsr/archs/spst_net_arch.py   SPST-Net architecture and core modules
basicsr/models/spst_net_model.py BasicSR model wrapper
options/train/                   Training configurations
options/test/                    Evaluation configurations
pretrained_models/               Experimental checkpoints
```

## Experimental Artifacts

Configurations and checkpoints are organized by upscaling factor:

| Scale | Training configuration | Evaluation configuration | Checkpoint |
|---|---|---|---|
| x2 | `train_spst_net_x2.yml` | `test_spst_net_x2.yml` | `spst_net_x2.pth` |
| x3 | `train_spst_net_x3.yml` | `test_spst_net_x3.yml` | `spst_net_x3.pth` |
| x4 | `train_spst_net_x4.yml` | `test_spst_net_x4.yml` | `spst_net_x4.pth` |

Checkpoint SHA-256 values:

```text
x2  bee23e6b3645759ad6de565097ba7fbbc15f07016766ac4e0ef6e19e3c5f3623
x3  ef8d7a2c59cd0beee85bde4e79876605f3afe760105db672f8f39bdced59c129
x4  9d095d6c7c327b3cd86a8f19f8f884ff41ddafd6e3e2228405519c8ffceb7a99
```

The repository emphasizes the proposed framework and preserves the experimental
artifacts used for manuscript review. It is not presented as a turnkey training
tutorial or a fully automated reproduction package.

## Environment

The recorded experiments used BasicSR 1.4.2, PyTorch 2.2.2+cu118, and
TorchVision 0.17.2+cu118. Required Python packages are listed in
`requirements.txt`.

## Quantitative Results

The manuscript reports evaluations on Set5, Set14, B100, Urban100, and Manga109
at the x2, x3, and x4 scales.

## Citation

TODO: Add the final BibTeX entry after publication metadata becomes available.

## Acknowledgements

This repository builds on the public
[CATANet](https://github.com/EquationWalker/CATANet) codebase and the
[BasicSR](https://github.com/XPixelGroup/BasicSR) framework. We thank the original
authors and retain the applicable Apache License 2.0 terms and attribution in
`LICENSE.txt`.

