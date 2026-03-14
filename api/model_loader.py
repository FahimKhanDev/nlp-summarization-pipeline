"""
model_loader.py
---------------
Handles loading and caching of fine-tuned models for inference.
Models are loaded once at startup and cached in memory.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

logger = logging.getLogger(__name__)

# Map model names to their local fine-tuned paths or HuggingFace fallbacks
MODEL_REGISTRY = {
    "pegasus": {
        "local_path": "/models/pegasus",
        "hf_fallback": "google/pegasus-xsum",
        "prefix": None,
        "description": "PEGASUS fine-tuned on Wikipedia (primary model)",
    },
    "bart": {
        "local_path": "/models/bart",
        "hf_fallback": "facebook/bart-base",
        "prefix": None,
        "description": "BART-base fine-tuned on Wikipedia (baseline)",
    },
    "t5": {
        "local_path": "/models/t5",
        "hf_fallback": "t5-small",
        "prefix": "summarize: ",
        "description": "T5-small fine-tuned on Wikipedia (CPU-friendly baseline)",
    },
}

# In-memory model cache
_model_cache: Dict[str, dict] = {}
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_path(model_name: str) -> str:
    """Return local path if fine-tuned model exists, else fall back to HuggingFace hub."""
    config = MODEL_REGISTRY[model_name]
    local = Path(config["local_path"])
    if local.exists() and any(local.iterdir()):
        logger.info(f"Loading fine-tuned model from {local}")
        return str(local)
    logger.warning(f"Fine-tuned model not found at {local}. Using HuggingFace fallback: {config['hf_fallback']}")
    return config["hf_fallback"]


def load_model(model_name: str) -> dict:
    """Load a model into cache. Returns cached version if already loaded."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")

    if model_name in _model_cache:
        logger.debug(f"Returning cached model: {model_name}")
        return _model_cache[model_name]

    logger.info(f"Loading model: {model_name} on {_device}")
    start = time.time()

    path = _resolve_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(_device)
    model.eval()

    _model_cache[model_name] = {
        "model": model,
        "tokenizer": tokenizer,
        "prefix": MODEL_REGISTRY[model_name]["prefix"],
        "device": _device,
    }

    elapsed = time.time() - start
    logger.info(f"Model '{model_name}' loaded in {elapsed:.2f}s")
    return _model_cache[model_name]


def get_available_models() -> dict:
    return {
        name: {
            "description": cfg["description"],
            "loaded": name in _model_cache,
            "device": _device if name in _model_cache else None,
        }
        for name, cfg in MODEL_REGISTRY.items()
    }


def preload_all_models():
    """Preload all models at startup — call this in FastAPI lifespan."""
    for model_name in MODEL_REGISTRY:
        try:
            load_model(model_name)
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
