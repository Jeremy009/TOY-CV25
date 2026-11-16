"""The converter refuses the models that it does not support."""

import pytest
import tiny_model

from cv25yolo import convert


def test_unsupported_input_size_is_refused(tmp_path, monkeypatch):
    """An input size that is not a multiple of 32 gives a clear error."""
    monkeypatch.setattr(tiny_model, "SIZE", 72)
    onnx_path = str(tmp_path / "tiny.onnx")
    tiny_model.make_model(onnx_path)
    with pytest.raises(ValueError, match="multiples of 32"):
        convert(onnx_path, str(tmp_path / "out"), ["unused.jpg"], log=lambda message: None)


def test_larger_model_is_refused(tmp_path, monkeypatch):
    """A model that is wider than YOLOv5s gives a clear error."""
    monkeypatch.setattr(tiny_model, "WIDTH", 48)
    onnx_path = str(tmp_path / "tiny.onnx")
    tiny_model.make_model(onnx_path)
    with pytest.raises(ValueError, match="only YOLOv5n and YOLOv5s"):
        convert(onnx_path, str(tmp_path / "out"), ["unused.jpg"], log=lambda message: None)
