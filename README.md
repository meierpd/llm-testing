# Local Gemma 7B Benchmark (MacBook Apple Silicon)

This project runs **Gemma 7B** locally using `llama-cpp-python` (Metal backend), then benchmarks speed with:
- TTFT (time to first token)
- Decode tokens/sec
- End-to-end tokens/sec

## 1) Create environment and install deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -U -r requirements.txt
```

## 2) Run benchmark (auto-download GGUF)

```bash
python run_gemma_benchmark.py
```

Defaults:
- Repo: `bartowski/gemma-7b-it-GGUF`
- File: `gemma-7b-it-Q4_K_M.gguf`

If you want a different quantization/model file:

```bash
python run_gemma_benchmark.py \
  --repo-id bartowski/gemma-7b-it-GGUF \
  --filename gemma-7b-it-Q5_K_M.gguf
```

If model access requires license acceptance on Hugging Face, set a token:

```bash
export HF_TOKEN=your_token_here
```

## 3) Custom prompts

Option A: edit `prompts.txt` (one prompt per line)

Option B: pass prompts directly:

```bash
python run_gemma_benchmark.py \
  --prompt "Explain retrieval-augmented generation in 4 bullet points" \
  --prompt "Write a Python function that merges two sorted lists"
```

## 4) Output

- Console: model responses + metrics per prompt
- File: `benchmark_results.json`

## Notes for MacBook Air M4 16GB

- Prefer 4-bit GGUF quantizations (`Q4_K_M`) for smoother memory usage.
- `--n-gpu-layers -1` uses Metal for all layers; usually fastest on Apple Silicon.
- If memory pressure appears, reduce:
  - `--n-ctx` (e.g. 2048)
  - quantization size (e.g. Q4 vs Q5/Q6)
