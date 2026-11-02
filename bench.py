# /// script
# requires-python = ">=3.10"
# dependencies = ["mlx-lm"]
# ///
"""Compare Ollama and MLX on the same prompts.

Run: uv run bench.py --backend both
"""
import argparse
import csv
import json
import statistics
import time
import urllib.request
from datetime import datetime
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
FILLER = (
    "The platform team rolled out a change to the cluster autoscaler and "
    "recorded the effect on node churn, pod scheduling latency, and cost. "
)


def build_workloads(prompt_file):
    workloads = {
        # Decode speed: tiny prompt, long answer.
        "short_prompt_long_output": (
            "Write a detailed explanation of how DNS resolution works.",
            512,
        ),
        # Prefill speed: big prompt, tiny answer.
        "long_prompt_short_output": (
            FILLER * 250 + "\n\nIn two sentences, what is this text about?",
            64,
        ),
    }
    if prompt_file:
        workloads["real_task"] = (Path(prompt_file).read_text(), 300)
    return workloads


def ollama_run(model, prompt, max_tokens):
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 16384},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        r = json.load(resp)
    ns = 1e9
    prefill_s = r["prompt_eval_duration"] / ns
    decode_s = r["eval_duration"] / ns
    return {
        "ttft_s": (r["load_duration"] + r["prompt_eval_duration"]) / ns,
        "prefill_tps": r["prompt_eval_count"] / prefill_s if prefill_s else 0,
        "decode_tps": r["eval_count"] / decode_s if decode_s else 0,
        "prompt_tokens": r["prompt_eval_count"],
        "output_tokens": r["eval_count"],
        "peak_mem_gb": ollama_model_size_gb(model),
        "load_s": r["load_duration"] / ns,
    }


def ollama_model_size_gb(model):
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/ps") as resp:
            for m in json.load(resp).get("models", []):
                if m["name"].startswith(model.split(":")[0]):
                    return m["size"] / 1e9
    except Exception:
        pass
    return float("nan")


class MlxRunner:
    def __init__(self, model_id):
        from mlx_lm import load
        t0 = time.perf_counter()
        self.model, self.tokenizer = load(model_id)
        self.load_s = time.perf_counter() - t0

    def run(self, prompt, max_tokens):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False)
        sampler = make_sampler(temp=0.0)
        t0 = time.perf_counter()
        ttft = None
        last = None
        for last in stream_generate(self.model, self.tokenizer, chat,
                                    max_tokens=max_tokens, sampler=sampler):
            if ttft is None:
                ttft = time.perf_counter() - t0
        return {
            "ttft_s": ttft,
            "prefill_tps": last.prompt_tps,
            "decode_tps": last.generation_tps,
            "prompt_tokens": last.prompt_tokens,
            "output_tokens": last.generation_tokens,
            "peak_mem_gb": last.peak_memory,
            "load_s": self.load_s,
        }


def bench(backend, run_fn, workloads, runs):
    rows = []
    for name, (prompt, max_tokens) in workloads.items():
        run_fn(prompt, max_tokens)  # warm-up, not recorded
        for i in range(runs):
            result = run_fn(prompt, max_tokens)
            rows.append({"backend": backend, "workload": name, "run": i + 1, **result})
            print(f"  {backend:7} {name:26} run {i + 1}: "
                  f"decode {result['decode_tps']:.1f} tok/s, "
                  f"prefill {result['prefill_tps']:.0f} tok/s")
    return rows


def summarize(rows):
    keys = ["ttft_s", "prefill_tps", "decode_tps", "peak_mem_gb"]
    print(f"\n{'backend':8} {'workload':26} " + " ".join(f"{k:>12}" for k in keys))
    seen = []
    for r in rows:
        group = (r["backend"], r["workload"])
        if group in seen:
            continue
        seen.append(group)
        sel = [x for x in rows if (x["backend"], x["workload"]) == group]
        cells = []
        for k in keys:
            vals = [x[k] for x in sel]
            cells.append(f"{statistics.median(vals):>8.2f}±{max(vals) - min(vals):<3.1f}")
        print(f"{group[0]:8} {group[1]:26} " + " ".join(cells))
    print("\nColumns are median±range over the timed runs.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["ollama", "mlx", "both"], default="both")
    p.add_argument("--ollama-model", default="llama3.1:8b")
    p.add_argument("--mlx-model", default="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--prompt-file", help="Extra real-world prompt to benchmark")
    args = p.parse_args()

    workloads = build_workloads(args.prompt_file)
    rows = []
    if args.backend in ("ollama", "both"):
        print(f"Ollama: {args.ollama_model}")
        rows += bench("ollama", lambda pr, n: ollama_run(args.ollama_model, pr, n),
                      workloads, args.runs)
    if args.backend in ("mlx", "both"):
        print(f"MLX: {args.mlx_model}")
        runner = MlxRunner(args.mlx_model)
        rows += bench("mlx", runner.run, workloads, args.runs)

    out = Path(__file__).parent / "results" / f"{datetime.now():%Y%m%d-%H%M%S}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarize(rows)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
