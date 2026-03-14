"""
train_pegasus.py
----------------
Fine-tunes google/pegasus-xsum on the Wikipedia summarization dataset.

PEGASUS is chosen as the primary model because its pretraining objective
(Gap Sentences Generation - GSG) directly mirrors abstractive summarization:
it learns to reconstruct masked-out sentences, which is structurally identical
to generating a summary. This gives it a head start over BART and T5 which
are general-purpose generative models.

Usage:
    python train_pegasus.py --epochs 3 --batch_size 2 --output_dir ./models/pegasus
"""

import os
import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "google/pegasus-xsum"
DATA_DIR = Path(__file__).parent.parent / "pipeline" / "data" / "processed"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 64


def load_split(split: str) -> Dataset:
    path = DATA_DIR / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Data not found at {path}. Run pipeline/flow.py first.")
    df = pd.read_csv(path)[["text", "title"]].dropna()
    return Dataset.from_pandas(df)


def preprocess_function(examples, tokenizer):
    inputs = tokenizer(
        examples["text"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(
            examples["title"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length",
        )
    inputs["labels"] = [
        [(t if t != tokenizer.pad_token_id else -100) for t in label]
        for label in targets["input_ids"]
    ]
    return inputs


def train(args):
    logger.info(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    logger.info("Loading datasets...")
    train_dataset = load_split("train")
    val_dataset = load_split("val")

    logger.info("Tokenizing datasets...")
    tokenize_fn = lambda x: preprocess_function(x, tokenizer)

    train_tokenized = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text", "title"])
    val_tokenized = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text", "title"])

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = torch.cuda.is_available() and args.fp16
    logger.info(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'} | fp16: {use_fp16}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=200,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        fp16=use_fp16,
        gradient_accumulation_steps=args.grad_accumulation,
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting PEGASUS fine-tuning...")
    trainer.train()

    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Training complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune PEGASUS for summarization")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--grad_accumulation", type=int, default=4)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default="./models/pegasus")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
