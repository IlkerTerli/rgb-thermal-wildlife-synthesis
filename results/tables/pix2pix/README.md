# pix2pix Quantitative Evaluation

This directory contains archived quantitative evaluation results for the
pix2pix RGB-to-thermal experiments.

## Metrics

The generated thermal-like output (`fake_B`) was compared with the
corresponding real thermal target image (`real_B`).

Two full-reference image-similarity metrics were used:

- **PSNR (Peak Signal-to-Noise Ratio)**
  Measures pixel-level reconstruction similarity. Higher values indicate
  greater similarity.

- **SSIM (Structural Similarity Index)**
  Measures structural similarity with respect to luminance, contrast, and
  image structure. Higher values indicate greater similarity.

The evaluation was performed on grayscale images.

## Archived Results

| Experiment | N | PSNR (dB) | SSIM |
|---|---:|---:|---:|
| Small | 50 | 25.0103 ± 0.2273 | 0.7209 ± 0.0063 |
| Medium – 20 epochs | 374 | 11.7744 ± 1.1934 | 0.3559 ± 0.0212 |
| Medium – 100 epochs | 374 | 11.7714 ± 1.0275 | 0.3477 ± 0.0285 |
| Large – 20 epochs | 500 | 16.3709 ± 1.8302 | 0.3519 ± 0.0697 |
| Large – 100 epochs | 500 | 17.0366 ± 2.3719 | 0.3642 ± 0.0996 |

Values are reported as mean ± standard deviation.

## Files

The archived summary files are:

```text
metrics_small_summary.csv
metrics_medium_summary.csv
metrics_large_summary.csv
```

Their contents correspond to:

- `metrics_small_summary.csv` – Small dataset evaluation
- `metrics_medium_summary.csv` – Medium dataset, 20-epoch and 100-epoch
  experiments
- `metrics_large_summary.csv` – Large dataset, 20-epoch and 100-epoch
  experiments

These CSV files are archived outputs from the original bachelor-thesis
experiments.

## Reproducible Evaluation

The reusable evaluation implementation is located at:

```text
src/evaluation/evaluate_pix2pix_metrics.py
```

Its command-line interface can be inspected with:

```bash
python src/evaluation/evaluate_pix2pix_metrics.py --help
```

The evaluation searches for generated `fake_B` images and their corresponding
`real_B` targets and calculates the image-level PSNR and SSIM values.

Summary statistics are then calculated from the evaluated image pairs.

## Interpretation

PSNR and SSIM are full-reference similarity metrics. They quantify similarity
between the generated thermal-like image and its corresponding real thermal
target.

Higher values indicate greater similarity under the respective metric.

The metrics should be interpreted together with qualitative inspection of the
generated images rather than as standalone measures of perceptual quality.