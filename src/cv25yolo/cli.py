"""The command line: cv25yolo model.onnx -o out/ --calib images/."""

import argparse
import glob
import os

from cv25yolo.convert import convert


def main(arguments: list[str] | None = None) -> None:
    """Run the converter from the command line.

    Args:
        arguments: The command-line arguments. By default those of the process.
    """
    parser = argparse.ArgumentParser(
        prog="cv25yolo",
        description="Convert a YOLOv5n or YOLOv5s ONNX export into a network binary for the CV25.",
    )
    parser.add_argument("onnx", help="the ONNX file, exported with a fixed input size")
    parser.add_argument("-o", "--out", required=True, help="output directory")
    parser.add_argument("--calib", required=True, help="directory with calibration images (*.jpg)")
    parser.add_argument(
        "--calib-count", type=int, default=64, help="number of calibration images to use"
    )
    parser.add_argument(
        "--stretch",
        action="store_true",
        help="the application stretches frames to the input size instead of letterboxing them",
    )
    options = parser.parse_args(arguments)
    images = sorted(glob.glob(os.path.join(options.calib, "*.jpg")))[: options.calib_count]
    if not images:
        parser.error(f"no *.jpg files in {options.calib}")
    try:
        convert(options.onnx, options.out, images, options.stretch)
    except ValueError as error:
        parser.exit(1, f"cv25yolo: {error}\n")


if __name__ == "__main__":
    main()
