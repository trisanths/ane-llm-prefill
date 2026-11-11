"""Route 2 <= M <= 8 row quantized matmuls (QuantizedLinear and the tied
lm_head) through qmv_h2. M = 1 stays on MLX's qmv, which is already at the
stream rate, and M > 8 stays on MLX's qmm."""
import mlx.core as mx
import mlx.nn as nn

from qmv_h2 import qmm_h2

BEST = {2: dict(rps=4, wpt=1, msplit=1), 3: dict(rps=4, wpt=1, msplit=1),
        4: dict(rps=4, wpt=1, msplit=1), 5: dict(rps=4, wpt=1, msplit=1),
        6: dict(rps=4, wpt=1, msplit=2), 7: dict(rps=4, wpt=1, msplit=1),
        8: dict(rps=4, wpt=1, msplit=2)}

_orig_linear = nn.QuantizedLinear.__call__
_orig_as_linear = nn.QuantizedEmbedding.as_linear


def _rows(layer, x):
    if getattr(layer, "mode", "affine") != "affine" or layer.bits != 4 or layer.group_size != 64:
        return 0
    k = x.shape[-1]
    m = x.size // k
    if m not in BEST or k % 256 or layer.weight.shape[0] % 4:
        return 0
    return m


PAD = {5: 6, 7: 8}


def _mm(self, x, m):
    x2 = x.reshape(m, x.shape[-1])
    if m in PAD:
        x2 = mx.concatenate([x2, mx.zeros((PAD[m] - m, x2.shape[1]), x2.dtype)], 0)
    y = qmm_h2(x2, self.weight, self.scales, self.biases, BEST[x2.shape[0]])
    return y[:m].reshape(*x.shape[:-1], -1)


def _linear(self, x):
    m = _rows(self, x)
    if not m:
        return _orig_linear(self, x)
    y = _mm(self, x, m)
    return y + self.bias if "bias" in self else y


def _as_linear(self, x):
    m = _rows(self, x)
    return _mm(self, x, m) if m else _orig_as_linear(self, x)


def patch():
    nn.QuantizedLinear.__call__ = _linear
    nn.QuantizedEmbedding.as_linear = _as_linear


def unpatch():
    nn.QuantizedLinear.__call__ = _orig_linear
    nn.QuantizedEmbedding.as_linear = _orig_as_linear
