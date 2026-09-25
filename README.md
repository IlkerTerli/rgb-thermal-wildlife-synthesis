# RGB-to-Thermal Wildlife Image Synthesis

This repository contains source code, experiment metadata, trained model
weights, and selected evaluation results associated with a bachelor thesis on
RGB-to-thermal image synthesis for wildlife imagery.

The experimental workflow combines generative image-to-image translation with
object detection. CycleGAN and pix2pix were investigated for the generation of
thermal-like images from RGB input images, while YOLOv8n was used for the
subsequent wildlife detection experiments.

## Project Overview

The repository covers three main experimental components:

1. **CycleGAN**
   - unpaired image-to-image translation
   - RGB-to-thermal image synthesis
   - experiments with different dataset sizes

2. **pix2pix**
   - paired image-to-image translation
   - RGB-to-thermal image synthesis
   - experiments with different dataset sizes and training durations
   - quantitative evaluation using PSNR and SSIM

3. **YOLOv8**
   - object detection on wildlife imagery
   - comparison between a manually assembled dataset and a generatively
     created dataset
   - archived training configurations, training curves, and trained weights

## Repository Structure

```text
rgb-thermal-wildlife-synthesis/
│
├── configs/
│   └── yolo/
│       ├── manual.yaml
│       └── generative.yaml
│
├── data/
│   └── README.md
│
├── experiments/
│   ├── cyclegan/
│   ├── pix2pix/
│   └── yolo/
│       ├── manual/
│       ├── generative/
│       └── README.md
│
├── models/
│   └── yolo/
│       ├── manual/
│       │   ├── best.pt
│       │   └── last.pt
│       └── generative/
│           ├── best.pt
│           └── last.pt
│
├── results/
│   ├── figures/
│   │   └── yolo/
│   └── tables/
│       └── pix2pix/
│
├── src/
│   ├── data_preparation/
│   │   ├── prepare_cyclegan_dataset.py
│   │   └── prepare_pix2pix_dataset.py
│   ├── evaluation/
│   │   └── evaluate_pix2pix_metrics.py
│   └── yolo/
│       ├── plot_training_curves.py
│       └── predict.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Data

The original image datasets are not included in this repository because of
their size and their original storage structure.

The `data/` directory therefore contains documentation only.

The scripts in `src/data_preparation/` provide reusable preparation workflows
for reconstructing the dataset layouts required by CycleGAN and pix2pix from
appropriately prepared source data.

## CycleGAN Experiments

Archived CycleGAN experiment metadata is stored in:

```text
experiments/cyclegan/
```

The archive contains the original training and test options as well as loss
logs for the documented experiments.

Further details are available in:

```text
experiments/cyclegan/README.md
```

CycleGAN uses unpaired image domains. Therefore, unlike pix2pix, there is no
pixel-aligned real thermal target for each generated image.

For this reason, pixel-wise metrics such as PSNR and SSIM are not used as
direct evaluation metrics for the CycleGAN outputs in this repository.

## pix2pix Experiments

Archived pix2pix experiment metadata is stored in:

```text
experiments/pix2pix/
```

The documented runs include:

```text
small
medium_20ep
medium_100ep
large_20ep
large_100ep
```

The experiment folders contain the original training configuration, test
configuration, and training loss logs where available.

Further details are available in:

```text
experiments/pix2pix/README.md
```

### Quantitative Evaluation

Because pix2pix operates on paired RGB/thermal data, its generated thermal
output (`fake_B`) can be compared directly with the corresponding real thermal
target (`real_B`).

The evaluation script is located at:

```text
src/evaluation/evaluate_pix2pix_metrics.py
```

It computes:

- PSNR
- SSIM

Archived summary results are stored in:

```text
results/tables/pix2pix/
```

The command-line interface can be inspected with:

```bash
python src/evaluation/evaluate_pix2pix_metrics.py --help
```

## YOLO Experiments

YOLOv8n was used for the object-detection experiments.

Two final dataset variants are archived:

```text
experiments/yolo/manual/
experiments/yolo/generative/
```

The final original Ultralytics runs correspond to:

- manually assembled dataset: `train7`
- generatively created dataset: `train5`

Each experiment directory contains:

- `args.yaml` — archived Ultralytics training configuration
- `results.csv` — epoch-wise training and validation results

The archived `args.yaml` files intentionally retain the absolute local paths
from the original experiments.

Portable dataset configuration files are provided separately under:

```text
configs/yolo/
```

Further details are available in:

```text
experiments/yolo/README.md
```

## Trained YOLO Models

The final trained YOLO checkpoints are included in:

```text
models/yolo/manual/
models/yolo/generative/
```

Each directory contains:

```text
best.pt
last.pt
```

`best.pt` contains the checkpoint selected by Ultralytics as the best model
during training, while `last.pt` contains the model state from the final
training epoch.

The original reproducible inference workflows used the corresponding
`last.pt` checkpoints.

Both checkpoints are retained to preserve the original trained model artifacts
and to allow further evaluation.

## YOLO Inference

A reusable inference implementation is provided at:

```text
src/yolo/predict.py
```

Available command-line options can be inspected with:

```bash
python src/yolo/predict.py --help
```

The original reproducible inference configuration used the following main
parameters:

- confidence threshold: `0.25`
- IoU threshold: `0.70`
- image size: `640`
- device: CPU

Model paths, image directories, output locations, and inference parameters are
provided through command-line arguments rather than being hard-coded into the
reusable script.

## YOLO Training Curves

Training and validation figures can be regenerated from the archived
`results.csv` files using:

```bash
python src/yolo/plot_training_curves.py
```

Generated figures are stored in:

```text
results/figures/yolo/
```

The repository currently includes:

```text
YOLO_Manual_Loss.png
YOLO_Manual_Metrics.png
YOLO_Generative_Loss.png
YOLO_Generative_Metrics.png
```

## Dataset Preparation

Reusable preparation scripts are located in:

```text
src/data_preparation/
```

### CycleGAN

The CycleGAN dataset preparation script can be inspected with:

```bash
python src/data_preparation/prepare_cyclegan_dataset.py --help
```

The script prepares RGB and thermal image domains in the directory structure
required for CycleGAN experiments.

### pix2pix

The pix2pix dataset preparation script can be inspected with:

```bash
python src/data_preparation/prepare_pix2pix_dataset.py --help
```

The script combines paired RGB and thermal images into the side-by-side image
format expected by the pix2pix framework.

Conceptually, each prepared sample has the form:

```text
[ RGB image | Thermal image ]
```

The RGB image forms domain A and the corresponding thermal image forms
domain B.

## Reproducibility

The repository separates original experiment evidence from reusable code:

- `experiments/` preserves original experiment metadata and recorded results.
- `configs/` contains portable configuration files.
- `src/` contains reusable and parameterized Python implementations.
- `models/` contains selected final trained YOLO weights.
- `results/` contains selected quantitative and visual outputs.
- raw datasets and temporary training artifacts are intentionally excluded.

This structure allows the original experimental settings to remain traceable
without embedding machine-specific paths into the reusable source code.

## Notes on Archived Experiment Files

Some archived experiment files contain absolute Windows or Linux paths from
the original development and training environments.

These paths are preserved intentionally because the files represent original
experiment metadata.

They should not be interpreted as paths required by the reusable source code.

## Software

The project was developed primarily in Python and uses tools and libraries
including:

- PyTorch
- CycleGAN and pix2pix
- Ultralytics YOLO
- NumPy
- Pillow
- scikit-image
- SciPy
- pandas
- matplotlib
- OpenCV

The dependencies required by the reusable repository utilities can be
installed with:

```bash
pip install -r requirements.txt
```

The reusable repository scripts were verified with Python 3.13.5 and the
package versions listed in `requirements.txt`.

The archived training experiments were originally executed in separate
experimental environments. Therefore, the versions in `requirements.txt`
should be interpreted as a verified environment for the reusable repository
utilities rather than as an exact reconstruction of every historical training
environment.

## Scope of the Repository

This repository is intended to provide the code and experimental artifacts
necessary to understand and reproduce the main computational steps of the
bachelor-thesis experiments.

Large raw datasets, intermediate training checkpoints, cache files, temporary
outputs, and local development artifacts are intentionally excluded.