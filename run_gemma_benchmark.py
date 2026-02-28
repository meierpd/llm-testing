#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from llama_cpp import Llama


def download_model(repo_id: str, filename: str, model_dir: Path, token: str | None) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=model_dir,
        token=token,
    )
    return Path(model_path)


def load_prompts(prompts_file: Path | None, cli_prompts: list[str] | None) -> list[str]:
    if cli_prompts:
        return cli_prompts

    if prompts_file and prompts_file.exists():
        with prompts_file.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    return [
        "Explain transformers to a 12-year-old in 5 bullet points.",
        "Write a Python function to check if a string is a palindrome.",
        "Summarize why quantization helps local LLM inference in under 100 words.",
    ]


def run_one_prompt(
    llm: Llama,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    prompt_tokens = len(llm.tokenize(prompt.encode("utf-8"), add_bos=True))

    start = time.perf_counter()
    first_token_time = None
    output_chunks: list[str] = []
    completion_tokens = 0

    stream = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )

    for event in stream:
        token = event["choices"][0]["text"]
        if token:
            output_chunks.append(token)
            completion_tokens += 1
            if first_token_time is None:
                first_token_time = time.perf_counter()

    end = time.perf_counter()

    total_s = end - start
    ttft_s = (first_token_time - start) if first_token_time is not None else total_s
    decode_s = (end - first_token_time) if first_token_time is not None else 0.0

    decode_tps = (completion_tokens / decode_s) if decode_s > 0 else 0.0
    end_to_end_tps = (completion_tokens / total_s) if total_s > 0 else 0.0
    prompt_tps = (prompt_tokens / ttft_s) if ttft_s > 0 else 0.0

    return {
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": round(ttft_s, 4),
        "decode_s": round(decode_s, 4),
        "total_s": round(total_s, 4),
        "prompt_tps_est": round(prompt_tps, 2),
        "decode_tps": round(decode_tps, 2),
        "end_to_end_tps": round(end_to_end_tps, 2),
        "response": "".join(output_chunks).strip(),
    }


def print_summary(results: list[dict[str, Any]]) -> None:
    if not results:
        print("No results.")
        return

    decode_tps_vals = [r["decode_tps"] for r in results]
    ttft_vals = [r["ttft_s"] for r in results]
    e2e_vals = [r["end_to_end_tps"] for r in results]

    print("\n=== Benchmark Summary ===")
    print(f"Prompts run: {len(results)}")
    print(f"Avg TTFT (s): {statistics.mean(ttft_vals):.3f}")
    print(f"Avg decode tok/s: {statistics.mean(decode_tps_vals):.2f}")
    print(f"Avg end-to-end tok/s: {statistics.mean(e2e_vals):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemma 7B locally and benchmark speed.")
    parser.add_argument("--model-path", type=Path, default=None, help="Path to local .gguf model")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="bartowski/gemma-7b-it-GGUF",
        help="HF repo ID to download GGUF from",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="gemma-7b-it-Q4_K_M.gguf",
        help="GGUF filename in repo",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("./models"), help="Download/cache directory")
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face token")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="-1 = all layers on Metal")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--prompt", action="append", default=None, help="Pass multiple times for multiple prompts")
    parser.add_argument("--prompts-file", type=Path, default=Path("./prompts.txt"))
    parser.add_argument("--out-json", type=Path, default=Path("./benchmark_results.json"))
    args = parser.parse_args()

    model_path = args.model_path
    if model_path is None:
        print(f"Downloading model: {args.repo_id}/{args.filename}")
        model_path = download_model(args.repo_id, args.filename, args.model_dir, args.hf_token)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from: {model_path}")
    llm = Llama(
        model_path=str(model_path),
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.threads,
        verbose=False,
    )

    prompts = load_prompts(args.prompts_file, args.prompt)
    print(f"Running {len(prompts)} prompts...")

    # Warmup: reduces first-run JIT/cache noise.
    _ = llm("Warmup.", max_tokens=8, temperature=0.0)

    results = []
    for i, prompt in enumerate(prompts, start=1):
        print(f"\n--- Prompt {i} ---")
        print(prompt)
        result = run_one_prompt(
            llm=llm,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        results.append(result)
        print("\nResponse:\n")
        print(result["response"])
        print("\nMetrics:")
        print(json.dumps({k: v for k, v in result.items() if k not in {"prompt", "response"}}, indent=2))

    print_summary(results)

    args.out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved detailed results to: {args.out_json.resolve()}")


if __name__ == "__main__":
    main()
