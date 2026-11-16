"""Example: turn the output files of the network binary into bounding boxes.

The network binary stops at the three convolutions of YOLOv5's Detect head. The
rest of the head runs on the CPU: a sigmoid, the grid and anchor arithmetic, and
non-maximum suppression. This file does that with numpy only.

    python examples/decode.py out/model.json frame.rgb.out0 frame.rgb.out1 frame.rgb.out2

The .out files are what `larod-client` writes on the device, in any order. Boxes
are printed in pixels of the network input.
"""

import json
import sys

import numpy as np


def unpack(raw: bytes, output: dict) -> np.ndarray:
    """Turn one output file into a float tensor.

    Args:
        raw: The bytes of the file.
        output: The entry of this output in model.json.

    Returns:
        The tensor, shape (channels, height, width).
    """
    raise NotImplementedError("not published")


def decode(tensors: list[np.ndarray], description: dict) -> np.ndarray:
    """Apply the arithmetic of YOLOv5's Detect head.

    Each tensor has, per anchor, the channels x, y, width, height, objectness and one score
    per class. With s = sigmoid(tensor):

        centre = (2 * s_xy - 0.5 + cell position) * stride
        size   = (2 * s_wh) ** 2 * anchor

    Args:
        tensors: The output tensors (channels, height, width), in the order of model.json.
        description: The content of model.json.

    Returns:
        One row per candidate box: centre x, centre y, width, height, objectness and the
        class scores.
    """
    rows = []
    for tensor, output in zip(tensors, description["outputs"]):
        anchors = np.array(output["anchors"])
        _, height, width = tensor.shape
        values = tensor.reshape(len(anchors), -1, height, width).transpose(0, 2, 3, 1)
        s = 1.0 / (1.0 + np.exp(-values.astype(np.float64)))
        cell_y, cell_x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        cells = np.stack([cell_x, cell_y], -1)[None]
        centre = (2 * s[..., :2] - 0.5 + cells) * output["stride"]
        size = (2 * s[..., 2:4]) ** 2 * anchors[:, None, None, :]
        rows.append(np.concatenate([centre, size, s[..., 4:]], -1).reshape(-1, values.shape[-1]))
    return np.concatenate(rows)


def select(rows: np.ndarray, confidence: float = 0.25, overlap: float = 0.45) -> np.ndarray:
    """Keep the confident boxes and drop the ones that overlap a better box of the same class.

    Args:
        rows: The candidate boxes from `decode`.
        confidence: Lowest score (objectness * class score) to keep.
        overlap: Largest intersection over union that two kept boxes of a class may have.

    Returns:
        One row per detection: x1, y1, x2, y2, score, class. Best score first.
    """
    scores = rows[:, 5:] * rows[:, 4:5]
    classes = scores.argmax(1)
    scores = scores.max(1)
    corners = np.concatenate([rows[:, :2] - rows[:, 2:4] / 2, rows[:, :2] + rows[:, 2:4] / 2], 1)
    order = [i for i in np.argsort(-scores) if scores[i] >= confidence]
    kept = []
    while order:
        best = order.pop(0)
        kept.append(best)
        remaining = []
        for i in order:
            if classes[i] != classes[best] or iou(corners[i], corners[best]) <= overlap:
                remaining.append(i)
        order = remaining
    return np.column_stack([corners[kept], scores[kept], classes[kept]]).reshape(-1, 6)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Return the intersection over union of two boxes given as x1, y1, x2, y2."""
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / (union + 1e-9)


def main() -> None:
    """Decode the output files given on the command line and print the detections."""
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    with open(sys.argv[1]) as file:
        description = json.load(file)
    # The files are matched to the outputs by size, which differs per output.
    by_size = {}
    for path in sys.argv[2:]:
        with open(path, "rb") as file:
            raw = file.read()
        by_size[len(raw)] = raw
    tensors = [unpack(by_size[output["file_bytes"]], output) for output in description["outputs"]]
    for x1, y1, x2, y2, score, class_id in select(decode(tensors, description)):
        box = f"{x1:7.1f} {y1:7.1f} {x2:7.1f} {y2:7.1f}"
        print(f"class {int(class_id):3d}  score {score:.2f}  box {box}")


if __name__ == "__main__":
    main()
