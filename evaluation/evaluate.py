"""
evaluate.py
-----------
Computes ROUGE-1, ROUGE-2, ROUGE-L scores for all three models
against the held-out test set.

Usage:
    python evaluate.py --model_dir ../training/models --test_data ../pipeline/data/processed/test.csv
"""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "pegasus": {"prefix": None, "hf_fallback": "google/pegasus-xsum"},
    "bart":    {"prefix": None, "hf_fallback": "facebook/bart-base"},
    "t5":      {"prefix": "summarize: ", "hf_fallback": "t5-small"},
}


def load_model_and_tokenizer(model_name: str, model_dir: Path):
    local_path = model_dir / model_name
    if local_path.exists() and any(local_path.iterdir()):
        path = str(local_path)
        logger.info(f"Loading fine-tuned {model_name} from {path}")
    else:
        path = MODELS[model_name]["hf_fallback"]
        logger.warning(f"Fine-tuned model not found, using HF fallback: {path}")

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(DEVICE)
    model.eval()
    return model, tokenizer


def generate_summary(model, tokenizer, text: str, prefix: str = None,
                     max_length: int = 128, num_beams: int = 4) -> str:
    input_text = (prefix or "") + text
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def evaluate_model(model_name: str, model, tokenizer, test_df: pd.DataFrame,
                   prefix: str = None, sample_size: int = 200) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Sample for speed
    eval_df = test_df.sample(min(sample_size, len(test_df)), random_state=42)

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    latencies = []

    logger.info(f"Evaluating {model_name} on {len(eval_df)} samples...")

    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=model_name):
        start = time.time()
        prediction = generate_summary(model, tokenizer, row["text"], prefix=prefix)
        latencies.append((time.time() - start) * 1000)

        result = scorer.score(row["title"], prediction)
        scores["rouge1"].append(result["rouge1"].fmeasure)
        scores["rouge2"].append(result["rouge2"].fmeasure)
        scores["rougeL"].append(result["rougeL"].fmeasure)

    return {
        "rouge1": round(sum(scores["rouge1"]) / len(scores["rouge1"]), 4),
        "rouge2": round(sum(scores["rouge2"]) / len(scores["rouge2"]), 4),
        "rougeL": round(sum(scores["rougeL"]) / len(scores["rougeL"]), 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "samples_evaluated": len(eval_df),
    }


def print_results_table(results: dict):
    print("\n" + "=" * 65)
    print(f"{'Model':<12} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'Latency(ms)':>12}")
    print("-" * 65)
    for model_name, metrics in results.items():
        print(
            f"{model_name:<12} "
            f"{metrics['rouge1']:>8.4f} "
            f"{metrics['rouge2']:>8.4f} "
            f"{metrics['rougeL']:>8.4f} "
            f"{metrics['avg_latency_ms']:>12.1f}"
        )
    print("=" * 65 + "\n")


def run_evaluation(model_dir: str, test_data: str, sample_size: int = 200, output_json: str = None):
    model_dir = Path(model_dir)
    test_df = pd.read_csv(test_data)[["text", "title"]].dropna()
    logger.info(f"Test set size: {len(test_df)}")

    all_results = {}

    for model_name, cfg in MODELS.items():
        try:
            model, tokenizer = load_model_and_tokenizer(model_name, model_dir)
            results = evaluate_model(
                model_name, model, tokenizer, test_df,
                prefix=cfg["prefix"], sample_size=sample_size
            )
            all_results[model_name] = results
            logger.info(f"{model_name}: ROUGE-2={results['rouge2']}")

            # Free memory between models
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")

    print_results_table(all_results)

    if output_json:
        with open(output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {output_json}")

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate summarization models with ROUGE")
    parser.add_argument("--model_dir", type=str, default="../training/models")
    parser.add_argument("--test_data", type=str, default="../pipeline/data/processed/test.csv")
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--output_json", type=str, default="results.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        model_dir=args.model_dir,
        test_data=args.test_data,
        sample_size=args.sample_size,
        output_json=args.output_json,
    )
