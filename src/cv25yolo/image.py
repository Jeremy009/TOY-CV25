"""Turn an image file into the input the network binary expects."""

import numpy as np
from PIL import Image


def load_image(path: str, height: int, width: int, stretch: bool = False) -> np.ndarray:
    """Load an image and fit it to the network input.

    By default the image is letterboxed: scaled to fit while keeping its aspect ratio, centred,
    and padded with grey (value 114), as YOLOv5 does.

    Args:
        path: Path of the image file.
        height: Input height of the network.
        width: Input width of the network.
        stretch: True to resize to exactly height x width without keeping the aspect ratio.

    Returns:
        A planar RGB uint8 array of shape (3, height, width). The binary divides by 255 itself.
    """
    image = Image.open(path).convert("RGB")
    if stretch:
        resized = image.resize((width, height), Image.BILINEAR)
        return np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1).copy()
    ratio = min(width / image.width, height / image.height)
    new_width, new_height = round(image.width * ratio), round(image.height * ratio)
    canvas = Image.new("RGB", (width, height), (114, 114, 114))
    resized = image.resize((new_width, new_height), Image.BILINEAR)
    canvas.paste(resized, ((width - new_width) // 2, (height - new_height) // 2))
    return np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1).copy()
