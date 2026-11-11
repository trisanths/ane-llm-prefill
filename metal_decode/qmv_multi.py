"""Small-M affine 4-bit matmul that reads each weight once for all M rows.

MLX's qmv launches one set of threadgroups per input row, so M rows read the
weights M times; that makes the verify step of speculative decoding cost
nearly M single-token steps. This kernel keeps the qmv_fast layout (2 simd
groups x 4 output rows per threadgroup, x pre-scaled by 1/16^j so masked
nibbles need no shift). The M input rows are split across MSPLIT simd
groups that share one weight tile, so each weight comes from DRAM once and
the other simd groups hit cache, while each thread holds only M/MSPLIT rows.

patch() routes nn.QuantizedLinear and the tied lm_head through it for
2 <= M <= MAX_M rows.
"""
import mlx.core as mx
import mlx.nn as nn

MAX_M = 8

SRC = """
constexpr int MS = M / MSPLIT;
constexpr int BLOCK = VPT * 32;
constexpr int SSTEP = 64 / VPT;
const int KW = K / 2;
const int KG = K / 64;
uint tg = threadgroup_position_in_grid.x;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
int out_row = tg * RPS;
int m0 = sg * MS;
const device uint8_t* ws = (const device uint8_t*)w + out_row * KW + lane * (VPT / 2);
const device T* sc = scales + out_row * KG + lane / SSTEP;
const device T* bi = biases + out_row * KG + lane / SSTEP;
const device T* xp = x + m0 * K + lane * VPT;
float res[MS][RPS];
for (int m = 0; m < MS; ++m)
  for (int r = 0; r < RPS; ++r) res[m][r] = 0.0f;
for (int k = 0; k < K; k += BLOCK) {
  XT xt[MS][VPT];
  float xs[MS];
  for (int m = 0; m < MS; ++m) {
    float s = 0.0f;
    for (int i = 0; i < VPT; i += 4) {
      float a0 = float(xp[m * K + i]);
      float a1 = float(xp[m * K + i + 1]);
      float a2 = float(xp[m * K + i + 2]);
      float a3 = float(xp[m * K + i + 3]);
      s += a0 + a1 + a2 + a3;
      xt[m][i] = XT(a0);
      xt[m][i + 1] = XT(a1 * (1.0f / 16.0f));
      xt[m][i + 2] = XT(a2 * (1.0f / 256.0f));
      xt[m][i + 3] = XT(a3 * (1.0f / 4096.0f));
    }
    xs[m] = s;
  }
  for (int r = 0; r < RPS; ++r) {
    const device uint16_t* wl = (const device uint16_t*)(ws + r * KW);
    float s = float(sc[r * KG]);
    float b = float(bi[r * KG]);
    AT acc[MS];
    for (int m = 0; m < MS; ++m) acc[m] = AT(0);
    for (int i = 0; i < VPT / 4; ++i) {
      uint16_t wv = wl[i];
      XT q0 = XT(wv & 0x000f);
      XT q1 = XT(wv & 0x00f0);
      XT q2 = XT(wv & 0x0f00);
      XT q3 = XT(wv & 0xf000);
      for (int m = 0; m < MS; ++m) {
        acc[m] += AT(xt[m][4 * i] * q0 + xt[m][4 * i + 1] * q1 +
                     xt[m][4 * i + 2] * q2 + xt[m][4 * i + 3] * q3);
      }
    }
    for (int m = 0; m < MS; ++m) res[m][r] += s * float(acc[m]) + b * xs[m];
  }
  ws += BLOCK / 2;
  sc += BLOCK / 64;
  bi += BLOCK / 64;
  xp += BLOCK;
}
for (int m = 0; m < MS; ++m) {
  for (int r = 0; r < RPS; ++r) {
    float v = simd_sum(res[m][r]);
    if (lane == 0) y[(m0 + m) * N + out_row + r] = T(v);
  }
}
"""

_kernels = {}


def vpt_for(m):
    return 16 if m <= 4 else 8


CFG = {"rps": 4, "xt": "float", "at": "float", "vpt": None, "msplit": None}


def msplit_for(m):
    for sp in (4, 2):
        if m % sp == 0 and m // sp >= 1 and sp <= m:
            return sp
    return 1


def qmm_small(x, w, scales, biases, cfg=None):
    """x [M, K] with 1 <= M <= MAX_M; w, scales, biases in MLX affine 4-bit,
    group 64 layout. Returns [M, N]."""
    c = dict(CFG, **(cfg or {}))
    m, k = x.shape
    n = w.shape[0]
    vpt = c["vpt"] or vpt_for(m)
    rps = c["rps"]
    sp = c["msplit"] or msplit_for(m)
    assert m % sp == 0
    rows_per_tg = rps
    assert k % (vpt * 32) == 0 and n % rows_per_tg == 0, (k, n)
    kern = _kernels.get((c["xt"], c["at"]))
    if kern is None:
        src = SRC.replace("XT", c["xt"]).replace("AT", c["at"])
        kern = _kernels[(c["xt"], c["at"])] = mx.fast.metal_kernel(
            name=f"qmv_multi_affine4_{c['xt']}_{c['at']}", input_names=["x", "w", "scales", "biases"],
            output_names=["y"], source=src)
    return kern(
        inputs=[x, w, scales, biases],
        template=[("T", x.dtype), ("M", m), ("K", k), ("N", n), ("VPT", vpt), ("RPS", rps), ("MSPLIT", sp)],
        grid=(n // rows_per_tg * 32 * sp, 1, 1), threadgroup=(32 * sp, 1, 1),
        output_shapes=[(m, n)], output_dtypes=[x.dtype])[0]


def _usable(layer, x):
    if getattr(layer, "mode", "affine") != "affine" or layer.bits != 4 or layer.group_size != 64:
        return 0
    k = x.shape[-1]
    m = x.size // k
    if not (2 <= m <= MAX_M) or k % (vpt_for(m) * 32) != 0:
        return 0
    return m


_orig_linear = nn.QuantizedLinear.__call__
_orig_as_linear = nn.QuantizedEmbedding.as_linear


def _linear(self, x):
    m = _usable(self, x)
    if not m or self.weight.shape[0] % 8 != 0:
        return _orig_linear(self, x)
    y = qmm_small(x.reshape(m, x.shape[-1]), self.weight, self.scales, self.biases)
    y = y.reshape(*x.shape[:-1], -1)
    if "bias" in self:
        y = y + self.bias
    return y


def _as_linear(self, x):
    m = _usable(self, x)
    if not m or self.weight.shape[0] % 8 != 0:
        return _orig_as_linear(self, x)
    y = qmm_small(x.reshape(m, x.shape[-1]), self.weight, self.scales, self.biases)
    return y.reshape(*x.shape[:-1], -1)


def patch():
    nn.QuantizedLinear.__call__ = _linear
    nn.QuantizedEmbedding.as_linear = _as_linear


def unpatch():
    nn.QuantizedLinear.__call__ = _orig_linear
    nn.QuantizedEmbedding.as_linear = _orig_as_linear
