"""Measure achieved fp16 matmul throughput on the Apple Neural Engine via Core ML.

Builds a Core ML program that is `layers` chained linear ops with constant fp16
weights of shape (N, K), input (M, K). Times predict() and derives TFLOPS and
weight-streaming bandwidth. Verifies ANE placement through MLComputePlan.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import coremltools as ct  # noqa: E402
from coremltools.converters.mil import Builder as mb  # noqa: E402
from coremltools.converters.mil.mil import types  # noqa: E402

UNITS = {
    "ane": ct.ComputeUnit.CPU_AND_NE,
    "gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu": ct.ComputeUnit.CPU_ONLY,
    "all": ct.ComputeUnit.ALL,
}


def quantize(model, quant, M, K):
    """quant: none | w8 (weight-only int8) | w8a8 (int8 weights + int8 activations)."""
    if quant == "none":
        return model
    from coremltools.optimize.coreml import (
        OptimizationConfig, OpLinearQuantizerConfig, linear_quantize_weights,
        linear_quantize_activations)
    if quant == "w8a8":
        rng = np.random.default_rng(7)
        samples = [{"x": rng.standard_normal((M, K)).astype(np.float16)} for _ in range(4)]
        acfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric"))
        model = linear_quantize_activations(model, acfg, samples)
    wcfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel"))
    return linear_quantize_weights(model, wcfg)


TARGETS = {"macOS14": ct.target.macOS14, "macOS15": ct.target.macOS15}


def build_model(M, K, N, layers, units, seed=0, quant="none", target="macOS14"):
    rng = np.random.default_rng(seed)
    weights = []
    for _ in range(layers):
        # (N, K) layout for mb.linear; scaled so a chain of layers stays O(1)
        w = (rng.standard_normal((N, K)) / np.sqrt(K)).astype(np.float16)
        weights.append(w)

    @mb.program(input_specs=[mb.TensorSpec(shape=(M, K), dtype=types.fp16)],
                opset_version=ct.target.iOS17)
    def prog(x):
        for w in weights:
            x = mb.linear(x=x, weight=w)
        return x

    model = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=units,
        minimum_deployment_target=TARGETS[target],
        inputs=[ct.TensorType(name="x", shape=(M, K), dtype=np.float16)],
        outputs=[ct.TensorType(dtype=np.float16)],
    )
    model = quantize(model, quant, M, K)
    if quant != "none":
        # re-load with the requested compute units (optimize APIs return CPU-only handles)
        model = ct.models.MLModel(model.get_spec(), weights_dir=model.weights_dir, compute_units=units)
    return model, weights


def placement(model, units):
    """Return {device_name: count} for the linear ops in the compiled model."""
    from coremltools.models.compute_plan import MLComputePlan
    path = model.get_compiled_model_path()
    plan = MLComputePlan.load_from_path(path=path, compute_units=units)
    prog = plan.model_structure.program
    fn = prog.functions["main"]
    counts = {}
    for op in fn.block.operations:
        name = op.operator_name.split(".")[-1]
        if name.startswith("const"):
            continue
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        dev = type(usage.preferred_compute_device).__name__ if usage else "None"
        dev = dev.replace("MLNeuralEngineComputeDevice", "ANE").replace("MLCPUComputeDevice", "CPU").replace("MLGPUComputeDevice", "GPU")
        key = f"{name}@{dev}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def bench(M, K, N, layers, units_name, iters, warmup, check, quant="none", target="macOS14"):
    units = UNITS[units_name]
    t0 = time.perf_counter()
    model, weights = build_model(M, K, N, layers, units, quant=quant, target=target)
    build_s = time.perf_counter() - t0

    try:
        place = placement(model, units)
    except Exception as e:  # noqa: BLE001
        place = {"error": str(e)[:200]}

    rng = np.random.default_rng(1)
    x = rng.standard_normal((M, K)).astype(np.float16)
    out_name = model.get_spec().description.output[0].name

    for _ in range(warmup):
        model.predict({"x": x})

    times = []
    for _ in range(iters):
        t = time.perf_counter()
        y = model.predict({"x": x})[out_name]
        times.append(time.perf_counter() - t)
    times = np.array(times)
    med = float(np.median(times))
    best = float(times.min())

    flops = 2.0 * M * K * N * layers
    wbytes = 2.0 * K * N * layers
    io_bytes = 2.0 * M * K + 2.0 * M * N
    res = {
        "units": units_name, "M": M, "K": K, "N": N, "layers": layers,
        "quant": quant, "target": target, "build_s": round(build_s, 2),
        "t_med_ms": med * 1e3, "t_min_ms": best * 1e3,
        "tflops_med": flops / med / 1e12, "tflops_best": flops / best / 1e12,
        "weight_MB": wbytes / 1e6,
        "weight_GBps_med": wbytes / med / 1e9, "weight_GBps_best": wbytes / best / 1e9,
        "total_GBps_med": (wbytes + io_bytes) / med / 1e9,
        "placement": place,
    }
    if check:
        ref = x.astype(np.float32)
        for w in weights:
            ref = ref @ w.astype(np.float32).T
        got = np.asarray(y, dtype=np.float32)
        err = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-6))
        res["max_rel_err"] = err
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, nargs="+", default=[1, 8, 32, 128, 512, 1024, 2048, 4096])
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--units", default="ane", choices=list(UNITS))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--quant", default="none", choices=["none", "w8", "w8a8"])
    ap.add_argument("--target", default="macOS14", choices=["macOS14", "macOS15"])
    ap.add_argument("--out", default=None, help="append JSON lines here")
    a = ap.parse_args()

    for M in a.M:
        try:
            r = bench(M, a.K, a.N, a.layers, a.units, a.iters, a.warmup, a.check,
                      quant=a.quant, target=a.target)
        except Exception as e:  # noqa: BLE001
            r = {"units": a.units, "M": M, "K": a.K, "N": a.N, "layers": a.layers,
                 "quant": a.quant, "target": a.target, "error": str(e)[:300]}
        print(json.dumps(r), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
