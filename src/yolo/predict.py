"""
Run reproducible YOLO inference on a directory of images.

This script is a portable refactoring of the inference scripts used
during the bachelor-thesis experiments.

Original inference configuration:
- confidence threshold: 0.25
- IoU threshold (NMS): 0.70
- image size: 640
- device: CPU
- annotated images, labels and confidence values are stored

The trained model, source directory and output directory are supplied
through command-line arguments so that no machine-specific paths are
required.
"""

from pathlib import Path
import argparse
import csv
import json

from ultralytics import YOLO


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run reproducible YOLO inference."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the trained YOLO weight file (.pt).",
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Directory containing the input images.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Root directory for prediction results.",
    )

    parser.add_argument(
        "--run-name",
        required=True,
        help="Name of the prediction run.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold. Default: 0.25.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="IoU threshold for non-maximum suppression. Default: 0.70.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size. Default: 640.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device. Default: cpu.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    source_dir = Path(args.source)
    output_root = Path(args.output)

    # ------------------------------------------------------------
    # Validate input paths
    # ------------------------------------------------------------

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )

    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Source directory not found:\n{source_dir}"
        )

    source_images = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not source_images:
        raise FileNotFoundError(
            f"No supported images found in:\n{source_dir}"
        )

    print(f"Model: {model_path}")
    print(f"Source directory: {source_dir}")
    print(f"Images found: {len(source_images)}")

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------

    model = YOLO(str(model_path))

    print(f"Model classes: {model.names}")

    # ------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------

    results = model.predict(
        source=str(source_dir),

        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,

        save=True,
        save_txt=True,
        save_conf=True,

        show_labels=True,
        show_conf=True,
        show_boxes=True,

        project=str(output_root),
        name=args.run_name,
        exist_ok=True,

        verbose=True,
    )

    # ------------------------------------------------------------
    # Store inference configuration
    # ------------------------------------------------------------

    run_dir = output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    configuration = {
        "model_path": str(model_path),
        "source_directory": str(source_dir),
        "confidence_threshold": args.conf,
        "iou_threshold_nms": args.iou,
        "image_size": args.imgsz,
        "device": args.device,
        "show_labels": True,
        "show_confidence": True,
        "show_boxes": True,
    }

    config_path = run_dir / "inference_config.json"

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------
    # Store confidence values
    # ------------------------------------------------------------

    csv_path = run_dir / "confidence_values.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(
            csv_file,
            delimiter=";",
        )

        writer.writerow(
            [
                "image",
                "class_id",
                "class_name",
                "confidence",
            ]
        )

        print("\nDetected objects:")

        for result in results:
            image_name = Path(result.path).name

            if result.boxes is None or len(result.boxes) == 0:
                print(f"{image_name}: no detection")

                writer.writerow(
                    [
                        image_name,
                        "",
                        "No detection",
                        "",
                    ]
                )

                continue

            class_ids = result.boxes.cls.cpu().tolist()
            confidence_values = (
                result.boxes.conf.cpu().tolist()
            )

            for class_id, confidence in zip(
                class_ids,
                confidence_values,
            ):
                class_id = int(class_id)
                class_name = result.names[class_id]

                print(
                    f"{image_name}: "
                    f"{class_name}, "
                    f"confidence = {confidence:.4f}"
                )

                writer.writerow(
                    [
                        image_name,
                        class_id,
                        class_name,
                        f"{confidence:.6f}",
                    ]
                )

    # ------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------

    print("\nInference completed.")
    print(f"Results: {run_dir}")
    print(f"Configuration: {config_path}")
    print(f"Confidence table: {csv_path}")


if __name__ == "__main__":
    main()