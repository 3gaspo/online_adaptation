"""TS-RAG's Chronos-T5 representation and exact FAISS retrieval rule.

The experiment supplies only the candidate rows whose dates are causally
accessible. Embedding and ranking otherwise follow the pinned upstream
``retrieve.py`` implementation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class TSRAGRetriever(nn.Module):
    """Embed with Chronos-T5 EOS and rank candidates with ``IndexFlatL2``."""

    def __init__(
        self,
        weights_path: str | Path,
        *,
        device_map: str = "cuda",
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        from chronos import ChronosPipeline

        path = Path(weights_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        self.pipeline = ChronosPipeline.from_pretrained(
            str(path),
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
        )
        model = getattr(self.pipeline, "model", None)
        if model is not None:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad = False

    @torch.no_grad()
    def representation(
        self,
        x: torch.Tensor,
        *,
        pool: bool = False,
    ) -> torch.Tensor:
        del pool
        embeddings, _ = self.pipeline.embed(x.squeeze(1))
        return embeddings[:, -1, :].float()

    def search(
        self,
        query_vector: np.ndarray,
        candidate_vectors: np.ndarray,
        *,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply upstream TS-RAG's ``IndexFlatL2`` top-k search.

        The extra result and final-column removal intentionally mirror
        upstream ``Retriever.search(..., drop_first=False)``. Candidate date
        eligibility is deliberately absent from this external-model class.
        """
        import faiss

        query = np.ascontiguousarray(query_vector, dtype=np.float32)
        candidates = np.ascontiguousarray(candidate_vectors, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        index = faiss.IndexFlatL2(int(candidates.shape[1]))
        index.add(candidates)
        distances, indices = index.search(query, int(top_k) + 1)
        return distances[:, :-1], indices[:, :-1]
