"""Multi-stage reranking pipeline.

Stages:
1. CrossEncoder fine-ranking
2. MMR (Maximal Marginal Relevance) diversity reranking
3. Question-type based filtering/boosting
"""

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


_DEFAULT_MMR_LAMBDA = 0.7


class MultiStageReranker:
    """Reranker with cross-encoder + MMR + type filtering."""

    def __init__(
        self,
        reranker_model: str = "BAAI/bge-reranker-large",
        embed_model: str = "BAAI/bge-m3",
        mmr_lambda: float = _DEFAULT_MMR_LAMBDA,
    ):
        self.reranker = CrossEncoder(reranker_model)
        self.embedder = SentenceTransformer(embed_model)
        self.mmr_lambda = mmr_lambda

    def cross_encoder_rank(
        self, query: str, candidates: list[dict]
    ) -> list[dict]:
        """Stage 2: CrossEncoder fine-ranking."""
        if not candidates:
            return []
        pairs = [(query, c["content"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked = []
        for i, c in enumerate(candidates):
            new_c = {**c}
            new_c["rerank_score"] = float(scores[i])
            ranked.append(new_c)
        ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return ranked

    def mmr_rerank(
        self, query: str, candidates: list[dict], top_k: int = 10
    ) -> list[dict]:
        """Stage 3: MMR diversity reranking."""
        if not candidates:
            return []
        if len(candidates) <= top_k:
            return candidates

        contents = [c["content"] for c in candidates]
        embeddings = self.embedder.encode(contents, show_progress_bar=False)
        query_emb = self.embedder.encode([query], show_progress_bar=False)

        query_sims = cosine_similarity(query_emb, embeddings)[0]

        selected: list[int] = []
        remaining = set(range(len(candidates)))

        # Pick the most relevant first
        first_idx = int(np.argmax(query_sims))
        selected.append(first_idx)
        remaining.remove(first_idx)

        while len(selected) < top_k and remaining:
            best_idx = -1
            best_score = -float("inf")
            for idx in remaining:
                rel = query_sims[idx]
                max_sim = max(
                    cosine_similarity(
                        embeddings[idx].reshape(1, -1),
                        embeddings[s].reshape(1, -1),
                    )[0, 0]
                    for s in selected
                )
                score = self.mmr_lambda * rel - (1 - self.mmr_lambda) * max_sim
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return [{**candidates[i]} for i in selected]

    def type_boost(
        self,
        candidates: list[dict],
        question_type: str | None = None,
        table_boost: float = 0.1,
    ) -> list[dict]:
        """Stage 4: Boost table chunks for fact-extraction questions."""
        if question_type != "fact_extraction" or not candidates:
            return candidates

        boosted = []
        for c in candidates:
            new_c = {**c}
            if c.get("type") == "table":
                new_c["rerank_score"] = new_c.get("rerank_score", 0) + table_boost
            boosted.append(new_c)
        boosted.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return boosted

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        question_type: str | None = None,
        ce_top_k: int = 20,
        mmr_top_k: int = 10,
    ) -> list[dict]:
        """Full multi-stage reranking pipeline."""
        # Stage 2: CrossEncoder
        ce_ranked = self.cross_encoder_rank(query, candidates)
        ce_top = ce_ranked[:ce_top_k]

        # Stage 3: MMR
        mmr_ranked = self.mmr_rerank(query, ce_top, top_k=mmr_top_k)

        # Stage 4: Type boost
        final = self.type_boost(mmr_ranked, question_type)

        return final
