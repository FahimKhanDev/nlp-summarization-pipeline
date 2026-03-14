"""
flow.py
-------
Prefect 2.x orchestration flow for the full data pipeline:
  ingest → clean → split → done

Run locally:
    python flow.py

Or via Prefect UI after deploying:
    prefect deploy flow.py:summarization_pipeline
"""

import os
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta

from ingest import stream_wikipedia
from preprocess import run_preprocessing


@task(
    name="ingest-wikipedia",
    retries=2,
    retry_delay_seconds=30,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=24),
)
def ingest_task(max_samples: int, batch_size: int) -> str:
    logger = get_run_logger()
    logger.info(f"Ingesting up to {max_samples} Wikipedia articles...")
    output_path = stream_wikipedia(max_samples=max_samples, batch_size=batch_size)
    logger.info(f"Ingestion complete: {output_path}")
    return str(output_path)


@task(
    name="preprocess-articles",
    retries=1,
)
def preprocess_task(raw_file: str) -> dict:
    logger = get_run_logger()
    logger.info("Running preprocessing pipeline...")
    train_df, val_df, test_df = run_preprocessing(raw_file="raw_articles.csv")
    stats = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    }
    logger.info(f"Preprocessing done: {stats}")
    return stats


@task(name="validate-output")
def validate_task(stats: dict) -> bool:
    logger = get_run_logger()

    min_train_size = int(os.getenv("MIN_TRAIN_SIZE", 1000))

    if stats["train_size"] < min_train_size:
        logger.error(f"Training set too small: {stats['train_size']} < {min_train_size}")
        return False

    logger.info(f"Validation passed: {stats}")
    return True


@flow(
    name="nlp-summarization-pipeline",
    description="End-to-end data pipeline: Wikipedia ingestion → cleaning → train/val/test splits",
)
def summarization_pipeline(
    max_samples: int = int(os.getenv("MAX_SAMPLES", 50000)),
    batch_size: int = int(os.getenv("INGEST_BATCH_SIZE", 1000)),
):
    logger = get_run_logger()
    logger.info("=== NLP Summarization Pipeline Started ===")

    # Step 1: Ingest raw articles
    raw_path = ingest_task(max_samples=max_samples, batch_size=batch_size)

    # Step 2: Clean and split
    stats = preprocess_task(raw_file=raw_path)

    # Step 3: Validate output
    is_valid = validate_task(stats=stats)

    if is_valid:
        logger.info("Pipeline completed successfully. Data ready for training.")
    else:
        logger.warning("Pipeline completed with warnings. Check data quality.")

    return stats


if __name__ == "__main__":
    result = summarization_pipeline()
    print(f"\nPipeline result: {result}")
