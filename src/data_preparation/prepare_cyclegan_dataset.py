"""
Prepare FLIR RGB/thermal datasets for CycleGAN.

This script consolidates the dataset-preparation workflow used in the
original bachelor-thesis experiments.

The original preparation steps were executed interactively in the shell
and were later refactored into this reusable script for reproducibility.

Supported source layouts
------------------------

1. paired
   Each source image consists of an RGB image and its corresponding
   thermal image concatenated horizontally.

   The image is split at its horizontal midpoint.

2. separated
   RGB and thermal images already exist in separate directories.

CycleGAN convention used in this project
----------------------------------------

Domain A = RGB
Domain B = Thermal
"""

from pathlib import Path
import argparse
import shutil

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_image_files(directory):
    """Return all supported image files contained in a directory."""

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {directory}"
        )

    files = []

    for file_path in directory.iterdir():
        if (
            file_path.is_file()
            and file_path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            files.append(file_path)

    return sorted(files)


def create_target_directories(output_dir):
    """Create the directory structure expected by CycleGAN."""

    directories = [
        "trainA",
        "trainB",
        "valA",
        "valB",
        "testA",
        "testB",
    ]

    for directory in directories:
        target_dir = output_dir / directory
        target_dir.mkdir(parents=True, exist_ok=True)


def prepare_paired_dataset(source_dir, output_dir):
    """
    Prepare horizontally concatenated RGB/thermal pairs.

    Expected source structure:

        source_dir/
            train/
            val/
            test/

    Each image:

        [ RGB | Thermal ]

    Output structure:

        output_dir/
            trainA/
            trainB/
            valA/
            valB/
            testA/
            testB/
    """

    split_mapping = {
        "train": ("trainA", "trainB"),
        "val": ("valA", "valB"),
        "test": ("testA", "testB"),
    }

    create_target_directories(output_dir)

    for split_name, target_names in split_mapping.items():

        source_split = source_dir / split_name

        target_a = output_dir / target_names[0]
        target_b = output_dir / target_names[1]

        files = get_image_files(source_split)

        print(
            f"{split_name}: "
            f"{len(files)} paired images"
        )

        for file_path in files:

            with Image.open(file_path) as image:
                image = image.convert("RGB")

                width, height = image.size
                middle = width // 2

                rgb_image = image.crop(
                    (0, 0, middle, height)
                )

                thermal_image = image.crop(
                    (middle, 0, width, height)
                )

                rgb_image.save(
                    target_a / file_path.name
                )

                thermal_image.save(
                    target_b / file_path.name
                )


def prepare_separated_dataset(source_dir, output_dir):
    """
    Prepare an already separated RGB/thermal dataset.

    Expected source structure:

        source_dir/
            train_rgb/
            train_thermal/
            val_rgb/
            val_thermal/
            test_rgb/
            test_thermal/

    Mapping:

        RGB     -> Domain A
        Thermal -> Domain B
    """

    directory_mapping = {
        "train_rgb": "trainA",
        "train_thermal": "trainB",
        "val_rgb": "valA",
        "val_thermal": "valB",
        "test_rgb": "testA",
        "test_thermal": "testB",
    }

    create_target_directories(output_dir)

    for source_name, target_name in directory_mapping.items():

        source_path = source_dir / source_name
        target_path = output_dir / target_name

        files = get_image_files(source_path)

        print(
            f"{source_name}: "
            f"{len(files)} images"
        )

        for file_path in files:

            shutil.copy2(
                file_path,
                target_path / file_path.name,
            )


def print_dataset_summary(output_dir):
    """Print the number of images in every generated CycleGAN folder."""

    print()
    print("Dataset summary")
    print("-" * 40)

    directories = [
        "trainA",
        "trainB",
        "valA",
        "valB",
        "testA",
        "testB",
    ]

    for directory in directories:

        directory_path = output_dir / directory

        files = get_image_files(directory_path)

        print(
            f"{directory}: "
            f"{len(files)}"
        )


def main():
    """Parse command-line arguments and prepare the dataset."""

    parser = argparse.ArgumentParser(
        description=(
            "Prepare RGB/thermal image data "
            "for CycleGAN."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the source dataset.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the generated CycleGAN dataset.",
    )

    parser.add_argument(
        "--layout",
        choices=[
            "paired",
            "separated",
        ],
        required=True,
        help=(
            "'paired' for horizontally concatenated "
            "RGB/thermal images; "
            "'separated' for separate RGB and "
            "thermal directories."
        ),
    )

    args = parser.parse_args()

    source_dir = args.source.resolve()
    output_dir = args.output.resolve()

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: "
            f"{source_dir}"
        )

    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Layout: {args.layout}")
    print()

    if args.layout == "paired":

        prepare_paired_dataset(
            source_dir,
            output_dir,
        )

    elif args.layout == "separated":

        prepare_separated_dataset(
            source_dir,
            output_dir,
        )

    print_dataset_summary(output_dir)

    print()
    print(
        "CycleGAN dataset preparation completed."
    )


if __name__ == "__main__":
    main()