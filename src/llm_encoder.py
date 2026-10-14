from __future__ import annotations

from typing import Iterable, List

import torch
from transformers import AutoModel, AutoTokenizer


class SciBERTEncoder:
    """Thin wrapper around a frozen SciBERT encoder used for class descriptions."""

    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.embedding_dim = self.model.config.hidden_size
        self.max_tokens = 512

    def encode_texts(self, texts: Iterable[str], batch_size: int = 64) -> torch.Tensor:
        texts = list(texts)
        all_embeddings: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_tokens,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**encoded)
                hidden = outputs.last_hidden_state
                cls = hidden[:, 0, :]
                all_embeddings.append(cls.cpu())
        if not all_embeddings:
            return torch.empty((0, self.embedding_dim))
        return torch.cat(all_embeddings, dim=0)

    def encode_class_descriptions(self, descriptions: Iterable[str], batch_size: int = 64) -> torch.Tensor:
        return self.encode_texts(descriptions, batch_size=batch_size)
