"""
train_bart.py
-------------
Fine-tunes facebook/bart-base on the Wikipedia summarization dataset.
Used as a strong baseline against PEGASUS.

Key differences from PEGASUS:
- BART uses a denoising autoencoding pretraining strategy (general purpose)
- Requires fp16 + gradient accumulation to fit in 6GB GPU memory
- Achieves good results but slightly lower ROUGE than PEGASUS

Usage:
    python train_bart.py --epochs 5 --batch_size 1 --fp16 --output_dir ./models/bart
"""

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

MODEL_NAME = "facebook/bart-base"
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
    model_inputs = tokenizer(
        examples["text"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        text_target=examples["title"],
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

    use_fp16 = torch.cuda.is_available() and args.fp16
    logger.info(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'} | fp16: {use_fp16}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        predict_with_generate=True,
        fp16=use_fp16,
        gradient_accumulation_steps=args.grad_accumulation,
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

    logger.info("Starting BART fine-tuning...")
    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Model saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BART for summarization")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--grad_accumulation", type=int, default=8)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default="./models/bart")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
