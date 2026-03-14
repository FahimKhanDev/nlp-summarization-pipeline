"""
ingest.py
---------
Streams Wikipedia articles from Hugging Face datasets without loading
the full corpus into memory. Saves batches as CSV to ./data/raw/.
"""

import os
import csv
import logging
from pathlib import Path
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).parent / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", 50000))
DEFAULT_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", 1000))


def stream_wikipedia(
    max_samples: int = DEFAULT_MAX_SAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    output_file: str = "raw_articles.csv",
) -> Path:
    """
    Streams Wikipedia articles from Hugging Face and writes them to CSV.

    Uses streaming=True to avoid downloading the full dataset (~20GB).
    Each row contains: id, title, text.

    Args:
        max_samples: Maximum number of articles to ingest.
        batch_size: Number of rows per write flush.
        output_file: Output CSV filename inside data/raw/.

    Returns:
        Path to the written CSV file.
    """
    output_path = RAW_DATA_DIR / output_file
    logger.info(f"Starting ingestion — max_samples={max_samples}, output={output_path}")

    dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    count = 0
    batch = []

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title", "text"])
        writer.writeheader()

        for article in dataset:
            if count >= max_samples:
                break

            # Skip articles with empty text or title
            if not article.get("text") or not article.get("title"):
                continue

            batch.append({
                "id": article.get("id", count),
                "title": article["title"].strip(),
                "text": article["text"].strip(),
            })
            count += 1

            if len(batch) >= batch_size:
                writer.writerows(batch)
                f.flush()
                batch = []
                logger.info(f"Ingested {count}/{max_samples} articles...")

        # Write remaining rows
        if batch:
            writer.writerows(batch)

    logger.info(f"Ingestion complete. {count} articles saved to {output_path}")
    return output_path


if __name__ == "__main__":
    stream_wikipedia()
