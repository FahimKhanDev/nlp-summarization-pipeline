# NLP Summarization Pipeline

An end-to-end pipeline that ingests, processes, and serves summaries of large-scale text datasets using fine-tuned transformer models — containerized with Docker and served via FastAPI.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue)](https://docker.com)
[![Model](https://img.shields.io/badge/Model-PEGASUS-orange)](https://huggingface.co/google/pegasus-xsum)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

This project automates the summarization of Wikipedia-scale text datasets using transformer-based NLP models. It is architected as a modular data engineering system — not just a model training script — with clear separation between ingestion, processing, training, evaluation, and serving layers.

**Models evaluated:**
| Model | Task Alignment | Training Loss | ROUGE-2 | Notes |
|-------|---------------|---------------|---------|-------|
| T5-small | General (text-to-text) | 0.3877 | 0.21 | CPU-friendly, stable |
| BART-base | General (denoising) | 0.128 | 0.29 | GPU required, OOM-prone |
| **PEGASUS** | **Summarization-specific** | **0.094** | **0.38** | **Best performance** |

> PEGASUS was selected as the primary model because its pretraining objective (Gap Sentences Generation) directly mirrors the summarization task — it learns to reconstruct masked sentences, which is structurally identical to abstractive summarization. BART and T5 are retained as baselines.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  NLP Summarization Pipeline              │
├─────────────┬──────────────┬─────────────┬──────────────┤
│  Data        │  Training    │  Evaluation │  Serving     │
│  Pipeline    │  Scripts     │  Scripts    │  API         │
│  (Prefect)   │  (HuggingFace│  (ROUGE)    │  (FastAPI)   │
│              │   Trainer)   │             │              │
├─────────────┴──────────────┴─────────────┴──────────────┤
│              Docker Compose (local orchestration)        │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
nlp-summarization-pipeline/
├── pipeline/
│   ├── ingest.py          # Wikipedia dataset streaming
│   ├── preprocess.py      # Cleaning, tokenization, chunking
│   ├── flow.py            # Prefect orchestration flow
│   └── requirements.txt
├── training/
│   ├── train_pegasus.py   # PEGASUS fine-tuning (primary)
│   ├── train_bart.py      # BART-base fine-tuning (baseline)
│   ├── train_t5.py        # T5-small fine-tuning (baseline)
│   └── requirements.txt
├── api/
│   ├── main.py            # FastAPI inference server
│   ├── model_loader.py    # Model loading & caching
│   ├── schemas.py         # Pydantic request/response schemas
│   ├── Dockerfile
│   └── requirements.txt
├── evaluation/
│   ├── evaluate.py        # ROUGE scoring across all models
│   ├── benchmark.py       # Latency & throughput benchmarks
│   └── requirements.txt
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory Data Analysis
│   ├── 02_model_comparison.ipynb  # Model benchmarking
│   └── 03_error_analysis.ipynb    # Failure case analysis
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## Quick Start

### Prerequisites
- Docker & Docker Compose installed
- 8GB RAM minimum (16GB recommended)

### 1. Clone & configure
```bash
git clone https://github.com/YOUR_USERNAME/nlp-summarization-pipeline.git
cd nlp-summarization-pipeline
cp .env.example .env
```

### 2. Start the full stack
```bash
docker-compose up --build
```

### 3. Run the data pipeline
```bash
docker-compose run pipeline python flow.py
```

### 4. Call the API
```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Machine learning is a branch of artificial intelligence...", "model": "pegasus"}'
```

---

## API Reference

### `POST /summarize`
```json
// Request
{
  "text": "Your long article text here...",
  "model": "pegasus",        // "pegasus" | "bart" | "t5"
  "max_length": 128,
  "min_length": 30,
  "num_beams": 4
}

// Response
{
  "summary": "Concise summary here.",
  "model_used": "pegasus",
  "input_tokens": 312,
  "latency_ms": 420
}
```

### `GET /models`
Returns available models and their status.

### `GET /health`
Health check endpoint.

---

## Training Your Own Models

```bash
# Fine-tune PEGASUS (recommended)
cd training
pip install -r requirements.txt
python train_pegasus.py --epochs 3 --batch_size 2 --output_dir ./models/pegasus

# Fine-tune BART (baseline)
python train_bart.py --epochs 5 --batch_size 1 --fp16 --output_dir ./models/bart

# Fine-tune T5 (CPU fallback)
python train_t5.py --epochs 5 --batch_size 2 --output_dir ./models/t5
```

---

## Evaluation Results

```bash
cd evaluation
python evaluate.py --model_dir ../training/models --test_data ../pipeline/data/test.csv
```

Sample output:
```
Model       ROUGE-1   ROUGE-2   ROUGE-L   Latency(ms)
---------   -------   -------   -------   -----------
PEGASUS     0.4521    0.3812    0.4103    420
BART-base   0.3874    0.2934    0.3521    310
T5-small    0.3102    0.2145    0.2891    890
```

---

## Dataset

Uses the [Wikipedia summarization corpus](https://github.com/frefel/wikisum) streamed via Hugging Face `datasets` — no full download required. The pipeline streams and processes batches on-the-fly, making it memory-efficient for large-scale data.

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Data Pipeline | Prefect 2.x |
| Dataset Streaming | Hugging Face `datasets` |
| Model Training | Hugging Face `transformers` + `Trainer` API |
| Evaluation | `rouge-score` |
| API Serving | FastAPI + Uvicorn |
| Containerization | Docker + Docker Compose |
| Primary Model | `google/pegasus-xsum` |

---

## License
MIT
