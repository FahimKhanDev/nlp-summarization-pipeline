"""
schemas.py
----------
Pydantic models for request/response validation in the FastAPI app.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=50, description="Input text to summarize (min 50 chars)")
    model: Literal["pegasus", "bart", "t5"] = Field(
        default="pegasus",
        description="Model to use for summarization"
    )
    max_length: int = Field(default=128, ge=20, le=512, description="Maximum summary length in tokens")
    min_length: int = Field(default=30, ge=5, le=100, description="Minimum summary length in tokens")
    num_beams: int = Field(default=4, ge=1, le=8, description="Beam search width")
    no_repeat_ngram_size: int = Field(default=3, ge=0, le=5)
    length_penalty: float = Field(default=1.0, ge=0.5, le=2.0)

    @field_validator("text")
    @classmethod
    def text_must_not_be_whitespace(cls, v):
        if not v.strip():
            raise ValueError("text must not be empty or whitespace")
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data...",
                "model": "pegasus",
                "max_length": 128,
                "min_length": 30,
                "num_beams": 4,
            }
        }
    }


class SummarizeResponse(BaseModel):
    summary: str
    model_used: str
    input_tokens: int
    latency_ms: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "summary": "Machine learning enables systems to learn from data automatically.",
                "model_used": "pegasus",
                "input_tokens": 312,
                "latency_ms": 420.5,
            }
        }
    }


class ModelInfo(BaseModel):
    name: str
    description: str
    loaded: bool
    device: Optional[str]


class ModelsResponse(BaseModel):
    models: dict
    default_model: str = "pegasus"


class HealthResponse(BaseModel):
    status: str
    device: str
    loaded_models: list[str]
