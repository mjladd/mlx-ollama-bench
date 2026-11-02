# MLX vs Ollama benchmark

Plan: `mjwiki/notebooks/todos/in_prog/mlx-vs-ollama-benchmark-plan.md`.

## Setup

```sh
brew install ollama
ollama serve &            # or open the Ollama app
ollama pull llama3.1:8b   # default tag is 4-bit q4_K_M
```

`uv run` installs `mlx-lm` automatically from the script header. The MLX model downloads from Hugging Face on first run.

## Run

```sh
uv run bench.py --backend both
uv run bench.py --backend mlx --runs 3
uv run bench.py --prompt-file ~/some-real-prompt.txt
```

Each run does 1 warm-up and `--runs` timed runs per workload. Results go to `results/<timestamp>.csv` and a median table prints at the end.

## Caveats

- Ollama memory is the model size from `/api/ps`. MLX memory is peak GPU memory. They are close, not identical.
- Ollama TTFT is load time plus prefill time, since the non-streaming API gives no first-token timestamp.
- The two quantizations (GGUF q4_K_M and MLX 4-bit) are similar, not the same. Compare output quality too (phase 4 of the plan).
