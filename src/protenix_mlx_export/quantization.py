"""MLX affine-int8 conversion for PyTorch matrix parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

INT8_BITS = 8
INT8_GROUP_SIZE = 64
MATRIX_RANK = 2


@dataclass(frozen=True)
class QuantizedMatrix:
    """Packed affine matrix plus its logical and padded physical shapes."""

    weight: NDArray[np.uint32]
    scales: NDArray[np.floating]
    biases: NDArray[np.floating]
    logical_shape: tuple[int, int]
    physical_shape: tuple[int, int]


def quantize_affine_int8(
    weight: NDArray[np.floating],
    *,
    group_size: int = INT8_GROUP_SIZE,
) -> QuantizedMatrix:
    """Zero-pad and quantize one output-by-input matrix with MLX affine int8.

    Padding is on the input (contracted) axis and the pad value is zero, so the
    padded columns contribute nothing to a matmul whose activation is likewise
    zero-padded. The runtime still needs ``logical_shape`` to slice embeddings and
    to know the true input width, which is why both shapes are recorded.
    """
    if weight.ndim != MATRIX_RANK:
        message = f"quantized weight must be two-dimensional, found rank {weight.ndim}"
        raise ValueError(message)
    # The caller narrows to float16 before quantizing. No released Protenix matrix
    # comes close to that range, but clamping one silently would be indistinguishable
    # from a correct pack, so refuse instead.
    finite = weight[np.isfinite(weight)]
    # The cast is the probe, so its overflow is the answer rather than a warning.
    with np.errstate(over="ignore"):
        overflows = finite.size and not np.isfinite(finite.astype(np.float16)).all()
    if overflows:
        message = (
            "matrix does not fit in float16 and cannot be quantized without clamping; "
            f"max |w| = {np.abs(finite).max():.6g}"
        )
        raise ValueError(message)
    output_width, logical_input_width = (int(value) for value in weight.shape)
    physical_input_width = (
        (logical_input_width + group_size - 1) // group_size
    ) * group_size
    padded = np.zeros((output_width, physical_input_width), dtype=np.float16)
    padded[:, :logical_input_width] = weight.astype(np.float16, copy=False)

    packed, scales, biases = mx.quantize(
        mx.array(padded),
        group_size=group_size,
        bits=INT8_BITS,
        mode="affine",
    )
    mx.eval(packed, scales, biases)
    return QuantizedMatrix(
        weight=np.asarray(packed),
        scales=np.asarray(scales),
        biases=np.asarray(biases),
        logical_shape=(output_width, logical_input_width),
        physical_shape=(output_width, physical_input_width),
    )


def is_quantizable(name: str, shape: tuple[int, ...]) -> bool:
    """Return whether one state-dict entry is an affine-int8 matrix candidate.

    Protenix's graph makes this a purely structural test. Every rank-2 parameter in
    every released checkpoint is a ``LinearNoBias``/``Linear``/``Embedding`` matrix
    named ``*.weight``; rank-1 ``*.weight`` entries are LayerNorm gains and must stay
    dense. :func:`~protenix_mlx_export.model_export.export_state_dict` asserts that
    invariant against the checkpoint rather than trusting it, because a future
    variant could introduce a rank-2 parameter that is not a matmul operand.

    The two rank-3 stacks in the confidence head (``plddt_weight``,
    ``resolved_weight``) are deliberately excluded: ``mx.quantize`` is rank-2 only,
    and together they are under half a megabyte.
    """
    return len(shape) == MATRIX_RANK and name.endswith(".weight")
