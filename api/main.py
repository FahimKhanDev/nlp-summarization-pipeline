"""
main.py
-------
FastAPI inference server for the NLP Summarization Pipeline.

Endpoints:
  POST /summarize  — Generate summary from input text
  GET  /models     — List available models and their status
  GET  /health     — Health check

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import time
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model_loader import load_model, get_available_models, preload_all_models
from schemas import SummarizeRequest, SummarizeResponse, ModelsResponse, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    logger.info("=== API starting up — preloading models ===")
    preload_all_models()
    logger.info("=== All models ready ===")
    yield
    logger.info("=== API shutting down ===")


app = FastAPI(
    title="NLP Summarization API",
    description=(
        "End-to-end summarization pipeline serving fine-tuned PEGASUS, BART, and T5 models. "
        "PEGASUS is the primary model due to its summarization-specific pretraining objective."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/summarize", response_model=SummarizeResponse, summary="Summarize input text")
async def summarize(request: SummarizeRequest):
    """
    Generate a concise summary of the provided text using the selected model.

    - **text**: The article or passage to summarize (min 50 characters)
    - **model**: `pegasus` (best quality), `bart` (fast GPU), or `t5` (CPU-friendly)
    - **max_length**: Maximum token length of the generated summary
    - **num_beams**: Higher = better quality but slower (1–8)
    """
    try:
        model_bundle = load_model(request.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]
    device = model_bundle["device"]
    prefix = model_bundle["prefix"] or ""

    input_text = prefix + request.text

    start = time.time()

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    ).to(device)

    input_token_count = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=request.max_length,
            min_length=request.min_length,
            num_beams=request.num_beams,
            no_repeat_ngram_size=request.no_repeat_ngram_size,
            length_penalty=request.length_penalty,
            early_stopping=True,
        )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    latency_ms = round((time.time() - start) * 1000, 2)

    logger.info(f"Summarized {input_token_count} tokens → '{summary[:60]}...' [{latency_ms}ms]")

    return SummarizeResponse(
        summary=summary,
        model_used=request.model,
        input_tokens=input_token_count,
        latency_ms=latency_ms,
    )


@app.get("/models", response_model=ModelsResponse, summary="List available models")
async def list_models():
    """Returns available models, their load status, and which device they're running on."""
    return ModelsResponse(models=get_available_models())


@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health():
    """Returns API status and which models are currently loaded in memory."""
    available = get_available_models()
    loaded = [name for name, info in available.items() if info["loaded"]]
    return HealthResponse(
        status="ok",
        device="cuda" if torch.cuda.is_available() else "cpu",
        loaded_models=loaded,
    )


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "NLP Summarization API — visit /docs for interactive docs"}
