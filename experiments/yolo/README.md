# YOLO Experiments

This directory contains the archived configurations and training results of the
YOLO experiments conducted as part of the bachelor thesis.

Two dataset variants were evaluated:

- `manual/` – manually assembled dataset
- `generative/` – generatively created dataset

## Final Experiment Runs

### Manual Dataset

The final experiment corresponds to the original Ultralytics run `train7`.

Archived files:

- `manual/args.yaml` – original Ultralytics training configuration
- `manual/results.csv` – epoch-wise training and validation results

Main training configuration:

- Model: YOLOv8n
- Epochs: 50
- Batch size: 16
- Image size: 640
- Device: CPU
- Pretrained weights: enabled
- Optimizer: auto
- Seed: 0
- Deterministic training: enabled

### Generatively Created Dataset

The final experiment corresponds to the original Ultralytics run `train5`.

Archived files:

- `generative/args.yaml` – original Ultralytics training configuration
- `generative/results.csv` – epoch-wise training and validation results

Main training configuration:

- Model: YOLOv8n
- Epochs: 50
- Batch size: 16
- Image size: 640
- Device: CPU
- Pretrained weights: enabled
- Optimizer: auto
- Seed: 0
- Deterministic training: enabled

## Dataset Configuration

Portable dataset configurations are stored in:

```text
configs/yolo/manual.yaml
configs/yolo/generative.yaml

```

The archived `args.yaml` files contain the original local Windows paths used
during the experiments. These paths are preserved as experiment metadata and
are not intended to be portable.

## Training Curves

The training and validation curves can be reproduced from the archived
`results.csv` files using:

```powershell
py src\yolo\plot_training_curves.py
```

The generated figures are stored in:

```text
results/figures/yolo/
```

Generated figures:

- `YOLO_Manual_Loss.png`
- `YOLO_Manual_Metrics.png`
- `YOLO_Generative_Loss.png`
- `YOLO_Generative_Metrics.png`

## Model Weights

The trained YOLO model weights corresponding to the final experiments are
included in the repository.

### Manual Dataset

The final manual experiment corresponds to the original Ultralytics run
`train7`.

Model weights:

```text
models/yolo/manual/best.pt
models/yolo/manual/last.pt
```

### Generatively Created Dataset

The final generative experiment corresponds to the original Ultralytics run
`train5`.

Model weights:

```text
models/yolo/generative/best.pt
models/yolo/generative/last.pt
```

`best.pt` contains the checkpoint selected by Ultralytics as the best model
during training, while `last.pt` contains the model state from the final
training epoch.

The original reproducible inference scripts used the corresponding `last.pt`
weights. Both checkpoints are retained to preserve the original experiment
artifacts and to allow further evaluation.

## Reproducible Inference

A reusable inference script is provided at:

```text
src/yolo/predict.py
```

The script accepts the trained model, input image directory, output directory,
run name, confidence threshold, IoU threshold, image size, and inference device
as command-line arguments.

Its command-line interface can be inspected with:

```powershell
py src\yolo\predict.py --help
```

The original inference experiments used the following main parameters:

- Confidence threshold: 0.25
- IoU threshold: 0.70
- Image size: 640
- Device: CPU

## Note on the Archived Manual Precision Values

The original `manual/results.csv` contains a precision value of approximately
`0.00333` throughout the recorded epochs, while recall and mAP values are
substantially higher.

This value is preserved exactly as recorded by the original Ultralytics
experiment. No retrospective modification of the experimental results was
performed.

The plotting script therefore reproduces this unusually low precision curve
directly from the archived source data.

## Repository Structure

The YOLO experiment artifacts are separated according to their purpose:

```text
configs/yolo/
    Portable dataset configurations

experiments/yolo/
    Original training configurations and epoch-wise results

models/yolo/
    Final trained YOLO model weights

src/yolo/
    Reusable inference and plotting scripts

results/figures/yolo/
    Reproduced training and validation figures
```

This separation preserves the original experimental evidence while providing
portable configurations and reusable scripts for reproduction and further
evaluation.