"""
benchmark.py
------------
Measures inference latency and throughput for each model
at different input lengths. Useful for capacity planning.

Usage:
    python benchmark.py --model pegasus --iterations 50
"""

import argparse
import logging
import time
import statistics
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_TEXTS = {
    "short": "Artificial intelligence is transforming industries worldwide, from healthcare diagnostics to autonomous vehicles.",
    "medium": (
        "Machine learning is a subset of artificial intelligence that provides systems the ability to "
        "automatically learn and improve from experience without being explicitly programmed. "
        "It focuses on the development of computer programs that can access data and use it to learn for themselves. "
        "The process begins with observations or data, such as examples, direct experience, or instruction, "
        "to look for patterns in data and make better decisions in the future."
    ),
    "long": (
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks "
        "with representation learning. Learning can be supervised, semi-supervised or unsupervised. "
        "Deep-learning architectures such as deep neural networks, recurrent neural networks, convolutional neural "
        "networks and transformers have been applied to fields including computer vision, speech recognition, "
        "natural language processing, machine translation, bioinformatics, drug design, medical image analysis, "
        "climate science, material inspection and board game programs, where they have produced results comparable "
        "to and in some cases surpassing human expert performance. Artificial neural networks were inspired by "
        "information processing and distributed communication nodes in biological systems. ANNs have various "
        "differences from biological brains. Specifically, neural networks tend to be static and symbolic, "
        "while the biological brain of most living organisms is dynamic and analog."
    ),
}


def load_model(model_name: str, model_dir: Path):
    local = model_dir / model_name
    path = str(local) if local.exists() else {
        "pegasus": "google/pegasus-xsum",
        "bart": "facebook/bart-base",
        "t5": "t5-small",
    }[model_name]

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(DEVICE)
    model.eval()
    return model, tokenizer


def run_benchmark(model, tokenizer, text: str, prefix: str = None,
                  iterations: int = 20, num_beams: int = 4) -> dict:
    input_text = (prefix or "") + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
    token_count = inputs["input_ids"].shape[1]

    # Warmup
    with torch.no_grad():
        model.generate(inputs["input_ids"], max_length=64, num_beams=1)

    latencies = []
    for _ in range(iterations):
        start = time.time()
        with torch.no_grad():
            model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=128,
                num_beams=num_beams,
                early_stopping=True,
            )
        latencies.append((time.time() - start) * 1000)

    return {
        "input_tokens": token_count,
        "iterations": iterations,
        "mean_latency_ms": round(statistics.mean(latencies), 1),
        "p50_latency_ms": round(statistics.median(latencies), 1),
        "p95_latency_ms": round(sorted(latencies)[int(0.95 * len(latencies))], 1),
        "throughput_rps": round(1000 / statistics.mean(latencies), 2),
    }


def print_benchmark_table(results: dict, model_name: str):
    print(f"\n{'='*70}")
    print(f"  Benchmark: {model_name.upper()} | Device: {DEVICE}")
    print(f"{'='*70}")
    print(f"{'Input Size':<12} {'Tokens':>7} {'Mean(ms)':>10} {'P50(ms)':>9} {'P95(ms)':>9} {'RPS':>6}")
    print("-" * 70)
    for input_size, metrics in results.items():
        print(
            f"{input_size:<12} "
            f"{metrics['input_tokens']:>7} "
            f"{metrics['mean_latency_ms']:>10.1f} "
            f"{metrics['p50_latency_ms']:>9.1f} "
            f"{metrics['p95_latency_ms']:>9.1f} "
            f"{metrics['throughput_rps']:>6.2f}"
        )
    print("=" * 70 + "\n")


def run(args):
    model_dir = Path(args.model_dir)
    prefixes = {"t5": "summarize: ", "pegasus": None, "bart": None}

    logger.info(f"Loading {args.model}...")
    model, tokenizer = load_model(args.model, model_dir)
    prefix = prefixes.get(args.model)

    results = {}
    for size_name, text in SAMPLE_TEXTS.items():
        logger.info(f"Benchmarking {args.model} on '{size_name}' input ({args.iterations} iters)...")
        results[size_name] = run_benchmark(
            model, tokenizer, text, prefix=prefix,
            iterations=args.iterations, num_beams=args.num_beams
        )

    print_benchmark_table(results, args.model)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark model inference latency")
    parser.add_argument("--model", type=str, default="pegasus", choices=["pegasus", "bart", "t5"])
    parser.add_argument("--model_dir", type=str, default="../training/models")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--num_beams", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
