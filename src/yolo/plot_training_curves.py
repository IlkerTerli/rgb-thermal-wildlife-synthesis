"""
Create training curves for the two YOLO experiments.

The script uses the archived Ultralytics results.csv files from:

    experiments/yolo/manual/results.csv
    experiments/yolo/generative/results.csv

It reproduces the training visualizations used during the
bachelor-thesis evaluation while avoiding machine-specific paths.

Generated figures:
- YOLO_Manual_Loss.png
- YOLO_Manual_Metrics.png
- YOLO_Generative_Loss.png
- YOLO_Generative_Metrics.png
"""

from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create YOLO training curves from Ultralytics results.csv files."
    )

    parser.add_argument(
        "--manual-csv",
        default="experiments/yolo/manual/results.csv",
        help="Path to results.csv of the manually assembled dataset.",
    )

    parser.add_argument(
        "--generative-csv",
        default="experiments/yolo/generative/results.csv",
        help="Path to results.csv of the generatively created dataset.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/figures/yolo",
        help="Directory in which the generated figures are stored.",
    )

    return parser.parse_args()


def load_results(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Results CSV not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    # Ultralytics CSV files may contain whitespace in column names.
    df.columns = df.columns.str.strip()

    return df


def plot_losses(df, title, output_path):
    """
    Plot training and validation losses.

    The three losses correspond to the values stored by Ultralytics:
    box loss, classification loss and distribution focal loss.
    """

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 10),
        sharex=True,
    )

    # --------------------------------------------------------
    # Box loss
    # --------------------------------------------------------

    axes[0].plot(
        df["epoch"],
        df["train/box_loss"],
        label="Training",
    )

    axes[0].plot(
        df["epoch"],
        df["val/box_loss"],
        label="Validation",
    )

    axes[0].set_ylabel("Box Loss")
    axes[0].set_title("Localization Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    # --------------------------------------------------------
    # Classification loss
    # --------------------------------------------------------

    axes[1].plot(
        df["epoch"],
        df["train/cls_loss"],
        label="Training",
    )

    axes[1].plot(
        df["epoch"],
        df["val/cls_loss"],
        label="Validation",
    )

    axes[1].set_ylabel("Classification Loss")
    axes[1].set_title("Classification Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    # --------------------------------------------------------
    # Distribution Focal Loss
    # --------------------------------------------------------

    axes[2].plot(
        df["epoch"],
        df["train/dfl_loss"],
        label="Training",
    )

    axes[2].plot(
        df["epoch"],
        df["val/dfl_loss"],
        label="Validation",
    )

    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("DFL Loss")
    axes[2].set_title("Distribution Focal Loss")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    fig.suptitle(
        title,
        fontsize=14,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def plot_metrics(df, title, output_path):
    """
    Plot the validation metrics reported by Ultralytics.
    """

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        df["epoch"],
        df["metrics/precision(B)"],
        label="Precision",
    )

    plt.plot(
        df["epoch"],
        df["metrics/recall(B)"],
        label="Recall",
    )

    plt.plot(
        df["epoch"],
        df["metrics/mAP50(B)"],
        label="mAP@0.5",
    )

    plt.plot(
        df["epoch"],
        df["metrics/mAP50-95(B)"],
        label="mAP@0.5:0.95",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.title(title)

    plt.ylim(
        0,
        1.05,
    )

    plt.legend()
    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def main():
    args = parse_args()

    manual_csv = Path(args.manual_csv)
    generative_csv = Path(args.generative_csv)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load archived experiment results
    # --------------------------------------------------------

    df_manual = load_results(
        manual_csv
    )

    df_generative = load_results(
        generative_csv
    )

    print(f"Manual experiment: {manual_csv}")
    print(f"Epochs: {len(df_manual)}")

    print(f"Generative experiment: {generative_csv}")
    print(f"Epochs: {len(df_generative)}")

    # --------------------------------------------------------
    # Manual YOLO dataset
    # --------------------------------------------------------

    plot_losses(
        df_manual,
        "YOLO Training - Manually Assembled Dataset",
        output_dir / "YOLO_Manual_Loss.png",
    )

    plot_metrics(
        df_manual,
        "YOLO Validation Metrics - Manually Assembled Dataset",
        output_dir / "YOLO_Manual_Metrics.png",
    )

    # --------------------------------------------------------
    # Generatively created YOLO dataset
    # --------------------------------------------------------

    plot_losses(
        df_generative,
        "YOLO Training - Generatively Created Dataset",
        output_dir / "YOLO_Generative_Loss.png",
    )

    plot_metrics(
        df_generative,
        "YOLO Validation Metrics - Generatively Created Dataset",
        output_dir / "YOLO_Generative_Metrics.png",
    )

    print("\nAll YOLO training figures were created.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()