"""
Prepare paired RGB/thermal images for pix2pix.

This script is a refactored version of the dataset-preparation code used
during the bachelor-thesis experiments.

Original workflow
-----------------
1. RGB and thermal images were paired and assigned identical filenames.
2. The corresponding RGB and thermal images were loaded.
3. The thermal image was resized to the RGB image dimensions if necessary.
4. Both images were concatenated horizontally:

       [ RGB | Thermal ]

5. The combined image was written to disk in the paired format expected
   by the pix2pix framework.

The core image-processing logic is preserved from the original experiment
script. The command-line interface was added later for reproducibility.
"""

from pathlib import Path
import argparse

import cv2


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def get_image_files(directory):
    """Return supported image files from a directory."""

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


def combine_images(rgb_dir, thermal_dir, output_dir):
    """
    Combine corresponding RGB and thermal images horizontally.

    Matching is based on identical filenames.

    Output format:

        [ RGB | Thermal ]
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rgb_files = get_image_files(rgb_dir)

    created = 0
    missing_thermal = 0
    unreadable = 0
    write_errors = 0

    for rgb_path in rgb_files:

        thermal_path = thermal_dir / rgb_path.name

        if not thermal_path.exists():
            missing_thermal += 1
            continue

        rgb = cv2.imread(str(rgb_path))
        thermal = cv2.imread(str(thermal_path))

        if rgb is None or thermal is None:
            unreadable += 1
            continue

        # Ensure identical spatial dimensions before concatenation.
        thermal = cv2.resize(
            thermal,
            (
                rgb.shape[1],
                rgb.shape[0],
            ),
        )

        # Core preprocessing step used in the original experiments.
        combined = cv2.hconcat(
            [
                rgb,
                thermal,
            ]
        )

        output_path = output_dir / rgb_path.name

        success = cv2.imwrite(
            str(output_path),
            combined,
        )

        if success:
            created += 1
        else:
            write_errors += 1

    print(f"Source RGB:       {rgb_dir}")
    print(f"Source thermal:   {thermal_dir}")
    print(f"Output:           {output_dir}")
    print(f"RGB files:        {len(rgb_files)}")
    print(f"Created:          {created}")
    print(f"Missing thermal:  {missing_thermal}")
    print(f"Unreadable:       {unreadable}")
    print(f"Write errors:     {write_errors}")
    print()


def prepare_dataset(source_root, output_root):
    """
    Prepare train, validation, and test splits.

    Expected source structure:

        source_root/
            train_rgb/
            train_thermal/
            val_rgb/
            val_thermal/
            test_rgb/
            test_thermal/

    Generated structure:

        output_root/
            pix2pix_train/
            pix2pix_val/
            pix2pix_test/
    """

    splits = {
        "train": (
            "train_rgb",
            "train_thermal",
            "pix2pix_train",
        ),
        "val": (
            "val_rgb",
            "val_thermal",
            "pix2pix_val",
        ),
        "test": (
            "test_rgb",
            "test_thermal",
            "pix2pix_test",
        ),
    }

    for split_name, directories in splits.items():

        rgb_name, thermal_name, output_name = directories

        rgb_dir = source_root / rgb_name
        thermal_dir = source_root / thermal_name
        output_dir = output_root / output_name

        print("=" * 60)
        print(f"Preparing {split_name} split")
        print("=" * 60)

        combine_images(
            rgb_dir,
            thermal_dir,
            output_dir,
        )


def main():
    """Parse command-line arguments and prepare the pix2pix dataset."""

    parser = argparse.ArgumentParser(
        description=(
            "Combine paired RGB and thermal images "
            "into the side-by-side format required by pix2pix."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help=(
            "Root directory containing train_rgb, train_thermal, "
            "val_rgb, val_thermal, test_rgb and test_thermal."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output root directory. "
            "If omitted, the source directory is used."
        ),
    )

    args = parser.parse_args()

    source_root = args.source.resolve()

    if not source_root.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: {source_root}"
        )

    if args.output is None:
        output_root = source_root
    else:
        output_root = args.output.resolve()

    print()
    print("pix2pix dataset preparation")
    print("=" * 60)
    print(f"Source root: {source_root}")
    print(f"Output root: {output_root}")
    print()

    prepare_dataset(
        source_root,
        output_root,
    )

    print("pix2pix dataset preparation completed.")


if __name__ == "__main__":
    main()