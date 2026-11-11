"""Speculative decoding on mlx_lm: target alone vs target + draft, with and
without the qmv_h2 verify kernel, over several prompt types. Reports
generation tok/s, acceptance rate and the speedup over the target alone.
Conditions are interleaved per prompt; each prompt runs twice and the
second run counts."""
import sys
import time

import mlx.core as mx
from mlx_lm import load, stream_generate

import spec_patch

PROMPTS = {
    "code": "Write a Python function that parses an ISO 8601 date string and returns a datetime, with error handling and tests.",
    "prose": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "explain": "Explain how a transformer language model generates text, step by step, for a first-year student.",
    "list": "List twenty common kitchen tools and give one sentence on what each is used for.",
}


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                   enable_thinking=False)


def run(model, tok, prompt, draft=None, k=None, n=192):
    kw = dict(max_tokens=n)
    if draft is not None:
        kw.update(draft_model=draft, num_draft_tokens=k)
    acc = tot = 0
    last = None
    for r in stream_generate(model, tok, prompt, **kw):
        last = r
        tot += 1
        acc += bool(getattr(r, "from_draft", False))
    return last.generation_tps, acc / max(tot, 1), last.generation_tokens


def main(target, draft_path, ks=(2, 3, 4, 5)):
    model, tok = load(target)
    draft, _ = load(draft_path)
    conds = [("target", None, False)] + [(f"k{k}", k, False) for k in ks] + [(f"k{k}+h2", k, True) for k in ks]
    rows = {c[0]: [] for c in conds}
    for name, text in PROMPTS.items():
        p = chat(tok, text)
        for label, k, patched in conds:
            (spec_patch.patch if patched else spec_patch.unpatch)()
            for rep in range(2):
                tps, acc, ntok = run(model, tok, p, draft if k else None, k)
            rows[label].append((name, tps, acc))
        spec_patch.unpatch()
        base = rows["target"][-1][1]
        print(f"{name:8s} target {base:5.1f} tok/s | " + " | ".join(
            f"{l} {rows[l][-1][1]:5.1f} ({rows[l][-1][1]/base:.2f}x, acc {rows[l][-1][2]:.2f})" for l, _, _ in conds[1:]), flush=True)
    print("\nmean over prompts:")
    for l, _, _ in conds:
        sp = [r[1] / b[1] for r, b in zip(rows[l], rows["target"])]
        print(f"  {l:8s} {sum(r[1] for r in rows[l])/len(rows[l]):6.1f} tok/s  speedup {sum(sp)/len(sp):.2f}x  acc {sum(r[2] for r in rows[l])/len(rows[l]):.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
