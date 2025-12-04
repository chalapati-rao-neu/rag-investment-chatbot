# src/rag_index.py

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from .data_prep import load_livermore_qa


class LivermoreIndex:
    """
    Holds the Q&A DataFrame, embedding model, and FAISS index,
    and exposes a retrieval method.
    """

    def __init__(
        self,
        excel_path: str = "data/Team Livermore.xlsx",
        embedding_model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2",
    ):
        # 1. Load data
        self.df = load_livermore_qa(excel_path)

        # 2. Load embedding model
        self.embedder = SentenceTransformer(embedding_model_name)

        # 3. Build FAISS index
        self.index, self.id_to_row = self._build_index()

    def _build_index(self) -> Tuple[faiss.IndexFlatIP, Dict[int, int]]:
        """
        Build a FAISS inner-product index on document embeddings.
        Returns the index and a mapping from FAISS id -> DataFrame row index.
        """
        # Prepare texts to embed: combine question + answer + label
        texts = (
            "Question: " + self.df["question"]
            + "\nAnswer: " + self.df["answer"]
            + "\nLabel: " + self.df["label"]
        ).tolist()

        # Compute embeddings
        embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        # Normalize for inner-product similarity (cosine-like)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product index
        index.add(embeddings)

        # Map FAISS index id -> DataFrame row index (aligned here)
        id_to_row = {i: i for i in range(len(self.df))}

        return index, id_to_row

    def _embed_query(self, query: str) -> np.ndarray:
        """
        Embed a user query and return a normalized 2D numpy array.
        """
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        return q_emb

    def retrieve(self, query: str, k: int = 3) -> List[dict]:
        """
        Retrieve top-k relevant QAs for the query across all labels.

        Returns a list of dicts with keys:
        'score', 'question', 'answer', 'label'.
        """
        if len(self.df) == 0:
            return []

        k = min(k, len(self.df))

        q_emb = self._embed_query(query)
        scores, ids = self.index.search(q_emb, k)

        ids = ids[0]
        scores = scores[0]

        results = []
        for idx, score in zip(ids, scores):
            row = self.df.iloc[self.id_to_row[idx]]
            results.append({
                "score": float(score),
                "question": row["question"],
                "answer": row["answer"],
                "label": row["label"],
            })

        return results


if __name__ == "__main__":
    # Quick test: python -m src.rag_index
    index = LivermoreIndex("data/Team Livermore.xlsx")
    query = "How should I handle fear after big losses?"
    res = index.retrieve(query, k=3)

    for r in res:
        print(f"[{r['label']}] score={r['score']:.3f}")
        print("Q:", r["question"])
        print("A:", r["answer"])
        print("-" * 60)