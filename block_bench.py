"""One transformer block (attention + MLP) at fixed M on the ANE via Core ML, vs MLX on the GPU.

D=4096, 32 heads of 128, causal attention, gated MLP 4096->12288->4096 built from
4096-wide chunks so no single weight is wider than 4096. Reports TFLOPS and the
per-op placement Core ML chose.
"""
import argparse
import json
import time

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

D, H, DH, FF = 4096, 32, 128, 12288
CHUNKS = FF // 4096


def block_flops(M):
    lin = 2.0 * M * D * (4 * D + 3 * FF)      # q,k,v,o + gate,up,down
    attn = 4.0 * M * M * D                     # QK^T and PV
    return lin + attn, lin, attn


def make_weights(seed=0):
    rng = np.random.default_rng(seed)
    w = lambda n, k: (rng.standard_normal((n, k)) / np.sqrt(k)).astype(np.float16)  # noqa: E731
    return {
        "q": w(D, D), "k": w(D, D), "v": w(D, D), "o": w(D, D),
        "gate": [w(4096, D) for _ in range(CHUNKS)],
        "up": [w(4096, D) for _ in range(CHUNKS)],
        "down": [w(D, 4096) for _ in range(CHUNKS)],
    }


def build_coreml(M, W, units, attn_impl, quant="none"):
    mask = np.triu(np.full((M, M), -1e4, dtype=np.float16), k=1)
    scale = np.float16(1.0 / np.sqrt(DH))
    opset = ct.target.iOS18 if attn_impl == "sdpa" else ct.target.iOS17

    @mb.program(input_specs=[mb.TensorSpec(shape=(M, D), dtype=types.fp16)],
                opset_version=opset)
    def prog(x):
        h = mb.layer_norm(x=x, axes=[-1], epsilon=1e-5)
        q = mb.linear(x=h, weight=W["q"])
        k = mb.linear(x=h, weight=W["k"])
        v = mb.linear(x=h, weight=W["v"])
        # (M, D) -> (H, M, DH)
        q = mb.transpose(x=mb.reshape(x=q, shape=[M, H, DH]), perm=[1, 0, 2])
        k = mb.transpose(x=mb.reshape(x=k, shape=[M, H, DH]), perm=[1, 0, 2])
        v = mb.transpose(x=mb.reshape(x=v, shape=[M, H, DH]), perm=[1, 0, 2])
        if attn_impl == "sdpa":
            ctx = mb.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=mask)
        else:
            s = mb.matmul(x=q, y=k, transpose_y=True)
            s = mb.mul(x=s, y=scale)
            s = mb.add(x=s, y=mask)
            p = mb.softmax(x=s, axis=-1)
            ctx = mb.matmul(x=p, y=v)
        ctx = mb.reshape(x=mb.transpose(x=ctx, perm=[1, 0, 2]), shape=[M, D])
        x = mb.add(x=x, y=mb.linear(x=ctx, weight=W["o"]))
        h = mb.layer_norm(x=x, axes=[-1], epsilon=1e-5)
        acc = None
        for i in range(CHUNKS):
            g = mb.silu(x=mb.linear(x=h, weight=W["gate"][i]))
            u = mb.linear(x=h, weight=W["up"][i])
            d = mb.linear(x=mb.mul(x=g, y=u), weight=W["down"][i])
            acc = d if acc is None else mb.add(x=acc, y=d)
        return mb.add(x=x, y=acc)

    model = ct.convert(
        prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
        compute_units=units, minimum_deployment_target=ct.target.macOS15,
        inputs=[ct.TensorType(name="x", shape=(M, D), dtype=np.float16)],
        outputs=[ct.TensorType(dtype=np.float16)],
    )
    if quant != "none":
        from ane_bench import quantize
        model = quantize(model, quant, M, D)
        model = ct.models.MLModel(model.get_spec(), weights_dir=model.weights_dir, compute_units=units)
    return model


def placement(model, units):
    from coremltools.models.compute_plan import MLComputePlan
    plan = MLComputePlan.load_from_path(path=model.get_compiled_model_path(), compute_units=units)
    fn = plan.model_structure.program.functions["main"]
    counts = {}
    for op in fn.block.operations:
        name = op.operator_name.split(".")[-1]
        if name.startswith("const"):
            continue
        u = plan.get_compute_device_usage_for_mlprogram_operation(op)
        dev = type(u.preferred_compute_device).__name__ if u else "None"
        dev = dev.replace("MLNeuralEngineComputeDevice", "ANE").replace(
            "MLCPUComputeDevice", "CPU").replace("MLGPUComputeDevice", "GPU")
        counts[f"{name}@{dev}"] = counts.get(f"{name}@{dev}", 0) + 1
    return counts


def ref_block(x, W):
    """fp32 numpy reference of the same block."""
    def ln(t):
        mu = t.mean(-1, keepdims=True)
        var = t.var(-1, keepdims=True)
        return (t - mu) / np.sqrt(var + 1e-5)
    M = x.shape[0]
    f = lambda a: a.astype(np.float32)  # noqa: E731
    h = ln(x)
    q, k, v = h @ f(W["q"]).T, h @ f(W["k"]).T, h @ f(W["v"]).T
    q = q.reshape(M, H, DH).transpose(1, 0, 2)
    k = k.reshape(M, H, DH).transpose(1, 0, 2)
    v = v.reshape(M, H, DH).transpose(1, 0, 2)
    s = q @ k.transpose(0, 2, 1) / np.sqrt(DH) + np.triu(np.full((M, M), -1e4, dtype=np.float32), k=1)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    ctx = (p @ v).transpose(1, 0, 2).reshape(M, D)
    x = x + ctx @ f(W["o"]).T
    h = ln(x)
    acc = 0
    for i in range(CHUNKS):
        g = h @ f(W["gate"][i]).T
        g = g / (1 + np.exp(-g))
        u = h @ f(W["up"][i]).T
        acc = acc + (g * u) @ f(W["down"][i]).T
    return x + acc


def run_coreml(M, W, units_name, attn_impl, iters, warmup, check, quant="none"):
    units = {"ane": ct.ComputeUnit.CPU_AND_NE, "gpu": ct.ComputeUnit.CPU_AND_GPU,
             "all": ct.ComputeUnit.ALL}[units_name]
    t0 = time.perf_counter()
    model = build_coreml(M, W, units, attn_impl, quant)
    build_s = time.perf_counter() - t0
    try:
        place = placement(model, units)
    except Exception as e:  # noqa: BLE001
        place = {"error": str(e)[:200]}
    rng = np.random.default_rng(1)
    x = rng.standard_normal((M, D)).astype(np.float16)
    out = model.get_spec().description.output[0].name
    for _ in range(warmup):
        model.predict({"x": x})
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        y = model.predict({"x": x})[out]
        ts.append(time.perf_counter() - t)
    ts = np.array(ts)
    total, lin, attn = block_flops(M)
    r = {"engine": f"coreml_{units_name}", "attn": attn_impl, "quant": quant, "M": M, "build_s": round(build_s, 2),
         "t_med_ms": float(np.median(ts)) * 1e3, "t_min_ms": float(ts.min()) * 1e3,
         "tflops_med": total / float(np.median(ts)) / 1e12, "tflops_best": total / float(ts.min()) / 1e12,
         "gflop_total": total / 1e9, "attn_frac": attn / total, "placement": place}
    if check:
        ref = ref_block(x.astype(np.float32), W)
        got = np.asarray(y, dtype=np.float32)
        r["max_rel_err"] = float(np.abs(got - ref).max() / np.abs(ref).max())
    return r


def run_mlx(M, W, iters, warmup):
    import mlx.core as mx
    import mlx.nn as nn
    w = {k: (mx.array(v) if not isinstance(v, list) else [mx.array(t) for t in v]) for k, v in W.items()}
    mx.eval(w)
    x = mx.array(np.random.default_rng(1).standard_normal((M, D)).astype(np.float16))
    mask = mx.array(np.triu(np.full((M, M), -1e4, dtype=np.float16), k=1))

    def ln(t):
        return mx.fast.layer_norm(t, None, None, 1e-5)

    def block(x):
        h = ln(x)
        q, k, v = h @ w["q"].T, h @ w["k"].T, h @ w["v"].T
        q = q.reshape(M, H, DH).transpose(1, 0, 2)
        k = k.reshape(M, H, DH).transpose(1, 0, 2)
        v = v.reshape(M, H, DH).transpose(1, 0, 2)
        ctx = mx.fast.scaled_dot_product_attention(q[None], k[None], v[None], scale=1 / np.sqrt(DH), mask=mask)[0]
        ctx = ctx.transpose(1, 0, 2).reshape(M, D)
        x = x + ctx @ w["o"].T
        h = ln(x)
        acc = None
        for i in range(CHUNKS):
            g = nn.silu(h @ w["gate"][i].T)
            u = h @ w["up"][i].T
            d = (g * u) @ w["down"][i].T
            acc = d if acc is None else acc + d
        return x + acc

    for _ in range(warmup):
        mx.eval(block(x))
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        mx.eval(block(x))
        ts.append(time.perf_counter() - t)
    ts = np.array(ts)
    total, lin, attn = block_flops(M)
    return {"engine": "mlx_gpu", "attn": "sdpa", "M": M,
            "t_med_ms": float(np.median(ts)) * 1e3, "t_min_ms": float(ts.min()) * 1e3,
            "tflops_med": total / float(np.median(ts)) / 1e12, "tflops_best": total / float(ts.min()) / 1e12,
            "gflop_total": total / 1e9, "attn_frac": attn / total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--engine", default="ane", choices=["ane", "gpu", "all", "mlx"])
    ap.add_argument("--attn", default="manual", choices=["manual", "sdpa"])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--quant", default="none", choices=["none", "w8", "w8a8"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    W = make_weights()
    for M in a.M:
        try:
            if a.engine == "mlx":
                r = run_mlx(M, W, a.iters, a.warmup)
            else:
                r = run_coreml(M, W, a.engine, a.attn, a.iters, a.warmup, a.check, a.quant)
        except Exception as e:  # noqa: BLE001
            r = {"engine": a.engine, "attn": a.attn, "M": M, "error": str(e)[:400]}
        print(json.dumps(r), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
