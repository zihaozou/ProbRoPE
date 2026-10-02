"""Vendored OpenAI CLIP backbone.

Re-exports the subset of upstream `clip` we use for CLIP-T / CLIP-F
metrics. See `clip.py` (trimmed) and `model.py` / `simple_tokenizer.py`
(verbatim) for upstream attribution.
"""
from .clip import available_models, load, tokenize

__all__ = ["available_models", "load", "tokenize"]
