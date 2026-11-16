"""Tests of the published parts: import, calibration, letterboxing and the Detect head."""

import numpy as np
import onnx
import pytest
import tiny_model
from check import count_matches
from decode import decode
from PIL import Image

from cv25yolo import convert
from cv25yolo.compiler import emulator, onnx_import, quantize
from cv25yolo.image import load_image


@pytest.fixture
def tiny(tmp_path):
    """Write the tiny network and return its path."""
    path = str(tmp_path / "tiny.onnx")
    tiny_model.make_model(path)
    return path


def test_import_reads_the_body(tiny):
    """The body becomes layers; the Detect head after the last convolution is left out."""
    layers, weights, input_shape = onnx_import.read_layers(onnx.load(tiny))
    assert [layer.op for layer in layers] == [
        "input",
        "conv",
        "silu",
        "conv",
        "silu",
        "conv",
        "output",
    ]
    assert layers[-1].inputs == [layers[-2].id]
    assert layers[1].attributes == {"kernel": 6, "stride": 2, "padding": 2}
    assert weights["detect"][0].shape == (18, 32, 1, 1)
    assert input_shape == (3, tiny_model.SIZE, tiny_model.SIZE)


def test_import_reads_the_anchors(tiny):
    """The anchors come from the Detect head's constants."""
    (anchors,) = onnx_import.read_anchors(onnx.load(tiny))
    assert anchors.tolist() == tiny_model.ANCHORS


def test_calibration_follows_the_bulk_of_the_images(monkeypatch):
    """One unusual image in a hundred does not widen the range of a tensor."""
    monkeypatch.setattr(emulator, "run", lambda primitives, image: {"tensor": image})
    images = [np.array([-1.0, 1.0])] * 99 + [np.array([-50.0, 50.0])]
    assert quantize.calibrate([], images) == {"tensor": (-1.0, 1.0)}


def test_letterbox_keeps_the_aspect_ratio(tmp_path):
    """A wide white image is centred between grey bars."""
    path = str(tmp_path / "wide.png")
    Image.new("RGB", (200, 100), (255, 255, 255)).save(path)
    image = load_image(path, 64, 64)
    assert image.shape == (3, 64, 64) and image.dtype == np.uint8
    assert (image[:, :16] == 114).all() and (image[:, 48:] == 114).all()
    assert (image[:, 17:47] == 255).all()
    assert (load_image(path, 64, 64, stretch=True) == 255).all()


def test_decode_follows_the_yolov5_formulas():
    """With all logits at 0 every box sits in the centre of its cell, as large as its anchor."""
    anchors = [[10.0, 13.0], [16.0, 30.0], [33.0, 23.0]]
    description = {"outputs": [{"anchors": anchors, "stride": 8}]}
    rows = decode([np.zeros((3 * 6, 2, 2))], description)
    assert rows.shape == (3 * 2 * 2, 6)
    assert np.allclose(rows[0, :4], [4, 4, 10, 13])  # anchor 0, cell (0, 0)
    assert np.allclose(rows[3, :4], [12, 12, 10, 13])  # anchor 0, cell (1, 1)
    assert np.allclose(rows[4, 2:4], [16, 30])  # anchor 1
    assert np.allclose(rows[:, 4:], 0.5)


def test_count_matches_pairs_boxes_of_the_same_class():
    """A box matches one box of the other set: same class and enough overlap."""
    ours = np.array([[0, 0, 10, 10, 0.9, 0], [0, 0, 10, 10, 0.8, 0], [20, 20, 30, 30, 0.8, 1]])
    theirs = np.array([[1, 0, 10, 10, 0.7, 0], [20, 20, 30, 30, 0.9, 2]])
    assert count_matches(ours, theirs) == 1


def test_unpublished_steps_say_so(tiny, tmp_path):
    """The pipeline runs up to the first outlined step, which says that it is not published."""
    messages = []
    with pytest.raises(NotImplementedError, match="not published"):
        convert(tiny, str(tmp_path / "out"), ["unused.jpg"], log=messages.append)
    assert messages == ["read tiny.onnx: 7 layers, input 64 x 64"]
