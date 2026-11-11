"""Small-M affine 4-bit matmul with packed-half dequant.

Each 32-bit word holds 8 nibbles. OR-ing (w >> 4j) & 0x000F000F into the
fp16 pattern 0x6400 (1024.0) gives the half2 (1024 + n_j, 1024 + n_{j+4})
in two integer ops; subtracting 1024 is exact. The M rows then use half2
FMAs, so the unpack is paid once per word and each row costs about 4 half2
FMAs per 8 weights. Products are summed in half only within one word (8
values, |sum| <= 120 max|x|) and then added in fp32, so activations up to
about 500 stay exact enough; larger blocks fall back to the caller.

Layout as qmv_multi: RPS output rows per threadgroup, MSPLIT simd groups
along M sharing one weight tile, VPT = 8 * WPT values per lane per block.
"""
import mlx.core as mx

SRC = """
constexpr int MS = M / MSPLIT;
constexpr int VPT = 8 * WPT;
constexpr int BLOCK = VPT * 32;
constexpr int SSTEP = 64 / VPT;
const int KW = K / 8;
const int KG = K / 64;
uint tg = threadgroup_position_in_grid.x;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
int out_row = tg * RPS;
int m0 = sg * MS;
const device uint32_t* wp = w + out_row * KW + lane * WPT;
const device T* sc = scales + out_row * KG + lane / SSTEP;
const device T* bi = biases + out_row * KG + lane / SSTEP;
const device T* xp = x + m0 * K + lane * VPT;
float res[MS][RPS];
for (int m = 0; m < MS; ++m)
  for (int r = 0; r < RPS; ++r) res[m][r] = 0.0f;
const half2 off = half2(1024.0h, 1024.0h);
for (int k = 0; k < K; k += BLOCK) {
  half2 xa[MS][WPT][4];
  float xs[MS];
  for (int m = 0; m < MS; ++m) {
    float s = 0.0f;
    for (int u = 0; u < WPT; ++u) {
      const device T* xq = xp + m * K + 8 * u;
      float v[8];
      for (int j = 0; j < 8; ++j) { v[j] = float(xq[j]); s += v[j]; }
      for (int j = 0; j < 4; ++j) xa[m][u][j] = half2(half(v[j]), half(v[j + 4]));
    }
    xs[m] = s;
  }
  for (int r = 0; r < RPS; ++r) {
    float s = float(sc[r * KG]);
    float b = float(bi[r * KG]);
    half2 q[WPT][4];
    for (int u = 0; u < WPT; ++u) {
      uint32_t wv = wp[r * KW + u];
      q[u][0] = as_type<half2>((wv & 0x000F000Fu) | 0x64006400u) - off;
      q[u][1] = as_type<half2>(((wv >> 4) & 0x000F000Fu) | 0x64006400u) - off;
      q[u][2] = as_type<half2>(((wv >> 8) & 0x000F000Fu) | 0x64006400u) - off;
      q[u][3] = as_type<half2>(((wv >> 12) & 0x000F000Fu) | 0x64006400u) - off;
    }
    for (int m = 0; m < MS; ++m) {
      float dot = 0.0f;
      for (int u = 0; u < WPT; ++u) {
        half2 a = xa[m][u][0] * q[u][0];
        a = fma(xa[m][u][1], q[u][1], a);
        a = fma(xa[m][u][2], q[u][2], a);
        a = fma(xa[m][u][3], q[u][3], a);
        dot += float(a.x) + float(a.y);
      }
      res[m][r] += s * dot + b * xs[m];
    }
  }
  wp += BLOCK / 8;
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

_kern = mx.fast.metal_kernel(name="qmv_h2_affine4", input_names=["x", "w", "scales", "biases"],
                             output_names=["y"], source=SRC)
CFG = {"rps": 4, "wpt": 1, "msplit": 1}


def qmm_h2(x, w, scales, biases, cfg=None):
    c = dict(CFG, **(cfg or {}))
    m, k = x.shape
    n = w.shape[0]
    sp, rps, wpt = c["msplit"], c["rps"], c["wpt"]
    assert m % sp == 0 and k % (256 * wpt) == 0 and n % rps == 0
    return _kern(inputs=[x, w, scales, biases],
                 template=[("T", x.dtype), ("M", m), ("K", k), ("N", n), ("RPS", rps), ("WPT", wpt), ("MSPLIT", sp)],
                 grid=(n // rps * 32 * sp, 1, 1), threadgroup=(32 * sp, 1, 1),
                 output_shapes=[(m, n)], output_dtypes=[x.dtype])[0]
