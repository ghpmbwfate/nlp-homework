"""
Reranker backends.

Two interchangeable implementations share the same predict() interface:
    predict(pairs: list[tuple[str, str]]) -> list[float]

- BGEReranker: wraps sentence_transformers.CrossEncoder for bge-reranker-*.
- QwenReranker: wraps transformers.AutoModelForCausalLM for Qwen3-Reranker-*,
  using the official yes/no token logits scoring scheme.

A factory load_reranker(alias) reads src.config.RERANKER_REGISTRY and
returns the appropriate backend.
"""

from __future__ import annotations

import logging
from typing import List, Protocol, Tuple

from src.config import RERANKER_REGISTRY, _resolve_model_path

logger = logging.getLogger(__name__)


class RerankerBackend(Protocol):
    """Common interface for all reranker backends."""

    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score (query, document) pairs. Higher = more relevant."""
        ...


# ----------------------------------------------------------------------
# BGE CrossEncoder
# ----------------------------------------------------------------------

class BGEReranker:
    """sentence_transformers.CrossEncoder wrapper for bge-reranker-*."""

    def __init__(self, model_path: str, device: str):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_path, device=device)
        logger.info(f"[BGEReranker] Loaded {model_path} on {device}")

    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]


# ----------------------------------------------------------------------
# Qwen3-Reranker (CausalLM yes/no logits)
# ----------------------------------------------------------------------

class QwenReranker:
    """Qwen3-Reranker-* wrapper using CausalLM yes/no token logits.

    Follows the official Qwen3-Reranker inference recipe:
    - Wrap (query, doc) in a chat-style instruction template ending with a
      single-token "yes"/"no" classification head.
    - Read logits over the last token, take softmax over (yes_id, no_id),
      use P(yes) as the relevance score.
    """

    PREFIX = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query "
        "and the Instruct provided. Note that the answer can only be \"yes\" "
        "or \"no\".<|im_end|>\n<|im_start|>user\n"
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    INSTRUCTION = (
        "Given a Chinese financial-report question, retrieve the passage "
        "that best answers it."
    )

    def __init__(self, model_path: str, device: str, batch_size: int = 8,
                 max_length: int = 8192):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="left"
        )
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype
        ).to(device).eval()

        self._yes_id = self._tokenizer.convert_tokens_to_ids("yes")
        self._no_id = self._tokenizer.convert_tokens_to_ids("no")
        if self._yes_id is None or self._no_id is None:
            raise RuntimeError(
                "Qwen reranker tokenizer missing yes/no tokens"
            )

        # Pre-tokenized prefix/suffix to splice around (query, doc) text
        self._prefix_ids = self._tokenizer.encode(
            self.PREFIX, add_special_tokens=False
        )
        self._suffix_ids = self._tokenizer.encode(
            self.SUFFIX, add_special_tokens=False
        )
        logger.info(f"[QwenReranker] Loaded {model_path} on {device}")

    def _format_pair(self, query: str, doc: str) -> str:
        return (
            f"<Instruct>: {self.INSTRUCTION}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}"
        )

    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        torch = self._torch

        scores: List[float] = []
        for start in range(0, len(pairs), self._batch_size):
            batch = pairs[start:start + self._batch_size]
            texts = [self._format_pair(q, d) for q, d in batch]

            # Build full prompts: PREFIX + body + SUFFIX
            input_ids_list = []
            for txt in texts:
                body_ids = self._tokenizer.encode(
                    txt, add_special_tokens=False
                )
                # Reserve room for prefix + suffix
                max_body = self._max_length - len(self._prefix_ids) - len(self._suffix_ids)
                if max_body > 0 and len(body_ids) > max_body:
                    body_ids = body_ids[:max_body]
                ids = self._prefix_ids + body_ids + self._suffix_ids
                input_ids_list.append(ids)

            # Left-pad to max length in batch
            pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
            max_len = max(len(x) for x in input_ids_list)
            padded = [
                [pad_id] * (max_len - len(x)) + x for x in input_ids_list
            ]
            attention = [
                [0] * (max_len - len(x)) + [1] * len(x) for x in input_ids_list
            ]
            input_ids = torch.tensor(padded, device=self._device)
            attn = torch.tensor(attention, device=self._device)

            with torch.no_grad():
                logits = self._model(input_ids, attention_mask=attn).logits
            # Take logits at last position over yes/no tokens
            last = logits[:, -1, :]
            pair_logits = last[:, [self._no_id, self._yes_id]]
            probs = torch.softmax(pair_logits, dim=-1)
            yes_probs = probs[:, 1].detach().cpu().tolist()
            scores.extend(float(p) for p in yes_probs)

        return scores


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def load_reranker(alias: str) -> RerankerBackend:
    """Instantiate the reranker backend for *alias* (see RERANKER_REGISTRY)."""
    import torch

    if alias not in RERANKER_REGISTRY:
        raise ValueError(
            f"Unknown reranker alias '{alias}'. "
            f"Known: {list(RERANKER_REGISTRY.keys())}"
        )
    ms_id, backend = RERANKER_REGISTRY[alias]
    model_path = _resolve_model_path(ms_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if backend == "bge":
        return BGEReranker(model_path, device)
    if backend == "qwen":
        return QwenReranker(model_path, device)
    raise ValueError(f"Unsupported backend '{backend}' for alias '{alias}'")
