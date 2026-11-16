"""Rebuild known models and compare the binaries with the ones that were tested on a device.

The models and the calibration images are not part of this repository. The test runs when
two environment variables point to them, and is skipped otherwise:

    CV25YOLO_GOLDEN_MODELS   directory with <case>.onnx for the cases in tests/golden.txt
    CV25YOLO_GOLDEN_IMAGES   directory with the calibration images (the first 64 *.jpg by name)

The case names give the model and its input size as height x width. A change to the compiler
that alters a binary makes this test fail on purpose: such a binary has to be tried on a
device before its new hash goes into tests/golden.txt.
"""

import glob
import hashlib
import os

import pytest

from cv25yolo import convert

MODELS = os.environ.get("CV25YOLO_GOLDEN_MODELS")
IMAGES = os.environ.get("CV25YOLO_GOLDEN_IMAGES")
with open(os.path.join(os.path.dirname(__file__), "golden.txt")) as file:
    CASES = [line.split() for line in file if line.strip()]


@pytest.mark.skipif(not (MODELS and IMAGES), reason="the golden models are not available")
@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_binary_is_unchanged(case, tmp_path):
    """The binary of a known model has the hash of the binary that ran on the device."""
    name, expected = case
    images = sorted(glob.glob(os.path.join(IMAGES, "*.jpg")))[:64]
    convert(os.path.join(MODELS, name + ".onnx"), str(tmp_path), images, log=lambda message: None)
    with open(tmp_path / "model.bin", "rb") as file:
        assert hashlib.sha256(file.read()).hexdigest() == expected
