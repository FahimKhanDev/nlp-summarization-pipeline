"""
train_t5.py
-----------
Fine-tunes t5-small on the Wikipedia summarization dataset.
Runs on CPU — useful as a lightweight baseline or for environments
without GPU access.

Key characteristics:
- T5 treats all NLP tasks as text-to-text, so inputs are prefixed with "summarize:"
- Much smaller memory footprint than BART or PEGASUS
- Slower on CPU but stable and reproducible

Usage:
    python train_t5.py --epochs 5 --batch_size 2 --output_dir ./models/t5
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "t5-small"
DATA_DIR = Path(__file__).parent.parent / "pipeline" / "data" / "processed"
T5_PREFIX = "summarize: "

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 64


def load_split(split: str) -> Dataset:
    path = DATA_DIR / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Data not found at {path}. Run pipeline/flow.py first.")
    df = pd.read_csv(path)[["text", "title"]].dropna()
    # T5 requires task prefix
    df["text"] = T5_PREFIX + df["text"]
    return Dataset.from_pandas(df)


def preprocess_function(examples, tokenizer):
    model_inputs = tokenizer(
        examples["text"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["title"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length",
        )
    model_inputs["labels"] = [
        [(t if t != tokenizer.pad_token_id else -100) for t in label]
        for label in labels["input_ids"]
    ]
    return model_inputs


def train(args):
    logger.info(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    logger.info("Loading and tokenizing datasets...")
    train_dataset = load_split("train")
    val_dataset = load_split("val")

    tokenize_fn = lambda x: preprocess_function(x, tokenizer)
    train_tokenized = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text", "title"])
    val_tokenized = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text", "title"])

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        predict_with_generate=True,
        no_cuda=True,   # Force CPU
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting T5 fine-tuning on CPU...")
    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Model saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune T5-small for summarization")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--output_dir", type=str, default="./models/t5")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
