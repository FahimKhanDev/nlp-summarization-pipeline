"""
preprocess.py
-------------
Cleans raw Wikipedia articles and prepares train/val/test splits
with tokenized inputs ready for model fine-tuning.
"""

import re
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

MAX_INPUT_TOKENS = 512
MAX_TARGET_TOKENS = 64
MIN_TEXT_LENGTH = 100   # characters
MIN_TITLE_LENGTH = 3    # characters


def clean_text(text: str) -> str:
    """Remove HTML tags, excessive whitespace, and non-ASCII artifacts."""
    text = re.sub(r"<[^>]+>", "", text)               # strip HTML
    text = re.sub(r"\[\d+\]", "", text)                # strip citation refs like [1]
    text = re.sub(r"\s+", " ", text)                   # collapse whitespace
    text = re.sub(r"[^\x00-\x7F]+", " ", text)        # remove non-ASCII
    return text.strip()


def truncate_to_words(text: str, max_words: int = 400) -> str:
    """Truncate text to max_words to stay within token limits."""
    words = text.split()
    return " ".join(words[:max_words])


def load_raw(filepath: str) -> pd.DataFrame:
    logger.info(f"Loading raw data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning dataframe...")

    # Drop nulls
    df = df.dropna(subset=["text", "title"])

    # Clean text and title
    df["text"] = df["text"].apply(clean_text)
    df["title"] = df["title"].apply(clean_text)

    # Filter by minimum length
    df = df[df["text"].str.len() >= MIN_TEXT_LENGTH]
    df = df[df["title"].str.len() >= MIN_TITLE_LENGTH]

    # Truncate text to fit within token budget
    df["text"] = df["text"].apply(truncate_to_words)

    df = df.reset_index(drop=True)
    logger.info(f"After cleaning: {len(df)} rows remain")
    return df


def split_data(df: pd.DataFrame, train=0.8, val=0.1, test=0.1):
    """Split into train/val/test sets."""
    assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1.0"

    train_df, temp_df = train_test_split(df, test_size=(1 - train), random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=(test / (val + test)), random_state=42)

    logger.info(f"Split — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df):
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = PROCESSED_DATA_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.info(f"Saved {name} split → {path}")


def run_preprocessing(raw_file: str = "raw_articles.csv"):
    df = load_raw(RAW_DATA_DIR / raw_file)
    df = clean_dataframe(df)
    train_df, val_df, test_df = split_data(df)
    save_splits(train_df, val_df, test_df)
    logger.info("Preprocessing complete.")
    return train_df, val_df, test_df


if __name__ == "__main__":
    run_preprocessing()
