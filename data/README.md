# Data

The datasets used in this project are not redistributed through this
repository.

The experiments are based on RGB and thermal wildlife image data that were
prepared and processed locally. The repository contains reusable scripts for
reconstructing the relevant dataset layouts where redistribution of the source
data is permitted.

## Dataset Categories

Several dataset configurations were used throughout the experiments:

- Small
- Medium
- Large

These configurations were used to investigate the influence of dataset size
on RGB-to-thermal image translation performance.

Not every dataset configuration was used by every model. Detailed experiment-
specific information is provided in the corresponding documentation under:

```text
experiments/cyclegan/
experiments/pix2pix/
experiments/yolo/
```

## Dataset Preparation

Reusable dataset preparation scripts are located in:

```text
src/data_preparation/
```

The CycleGAN preparation workflow is implemented in:

```text
src/data_preparation/prepare_cyclegan_dataset.py
```

The pix2pix preparation workflow is implemented in:

```text
src/data_preparation/prepare_pix2pix_dataset.py
```

CycleGAN uses separate RGB and thermal domains, while pix2pix requires
spatially corresponding paired images.

For pix2pix, the prepared samples use the side-by-side representation:

```text
[ RGB image | Thermal image ]
```

where the RGB image forms domain A and the thermal image forms domain B.

## Repository Policy

Raw image datasets and processed dataset copies are intentionally excluded from
the repository because of their size and original storage requirements.

Selected experiment metadata, evaluation results, figures, and final YOLO model
weights are stored separately in the appropriate repository directories.

The repository therefore contains the information required to understand the
dataset organization and reproduce the relevant preprocessing steps without
redistributing the original image datasets.