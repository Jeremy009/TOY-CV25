"""Step 5 of the pipeline: cut the network into pieces and generate the binary.

The accelerator's memory is small, so the network runs as a chain of pieces. A
search cuts the network: each piece grows for as long as it fits, and the search
steps back when a cut turns out to be one the accelerator does not run correctly.
Each piece is then written as a program for the accelerator.

The code generator is not published. In the full project it is about 2,700 lines
in 12 modules.
"""

from collections.abc import Callable

from cv25yolo.compiler.records import Primitive


def generate(primitives: list[Primitive], path: str, log: Callable[[str], None]) -> int:
    """Cut the network into pieces and write the network binary.

    Args:
        primitives: The quantized primitives in program order.
        path: Path of the binary to write.
        log: Called with progress messages.

    Returns:
        The number of pieces.

    Raises:
        ValueError: If the network cannot be cut into pieces that fit the
            accelerator.
    """
    raise NotImplementedError("not published")
