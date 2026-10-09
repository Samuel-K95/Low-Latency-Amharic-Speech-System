#!/usr/bin/env python3
"""
Hybrid Dense + Sparse Retrieval Service for Amharic Labour Proclamation.
Combines Multilingual-E5-Base dense embeddings with character-level TF-IDF sparse matching.
"""
import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from server.config import (
    INDEX_DIR,
    DATA_DIR,
    EMBEDDING_MODEL_ID,
    RETRIEVAL_DEVICE,
    ALPHA_HYBRID,
    TOP_K_RETRIEVAL,
    MIN_DENSE_THRESHOLD,
)
from src.normalize import normalize_query
from src.fidel_fold import fold

def _minmax(x: np.ndarray) -> np.ndarray:
    """Applies min-max scaling to an array of scores."""
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(x)

class RetrievalService:
    def __init__(
        self,
        index_dir: Path = INDEX_DIR,
        data_dir: Path = DATA_DIR,
        model_id: str = EMBEDDING_MODEL_ID,
        device: str = RETRIEVAL_DEVICE,
    ):
        self.index_dir = Path(index_dir)
        self.data_dir = Path(data_dir)
        self.model_id = model_id
        self.device = device
        self.embed_model = None
        self.chunks: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.tfidf: Optional[TfidfVectorizer] = None
        self.kw_matrix = None
        self.is_ready = False

    def load(self):
        """Loads dense embedding model, chunks metadata, FAISS/embeddings, and TF-IDF index."""
        if self.is_ready:
            return

        from sentence_transformers import SentenceTransformer

        print(f"[RetrievalService] Loading E5 Embedding Model {self.model_id} on {self.device}...")
        self.embed_model = SentenceTransformer(self.model_id, device=self.device)

        # 1. Load chunks metadata
        chunks_path = self.index_dir / "chunks.json"
        if not chunks_path.exists():
            chunks_path = self.data_dir / "chunks.json"
        assert chunks_path.exists(), f"Chunks file not found at {chunks_path}"
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"[RetrievalService] Loaded {len(self.chunks)} chunks metadata.")

        # 2. Load dense embeddings matrix
        emb_path = self.index_dir / "embeddings.npy"
        if emb_path.exists():
            self.embeddings = np.load(emb_path).astype("float32")
            print(f"[RetrievalService] Loaded dense embeddings matrix shape: {self.embeddings.shape}")
        else:
            print("[RetrievalService] embeddings.npy not found, generating embeddings now...")
            texts = ["passage: " + c.get("text_clean", c.get("text_spoken", "")) for c in self.chunks]
            self.embeddings = self.embed_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True
            ).astype("float32")
            np.save(emb_path, self.embeddings)

        # 3. Load or build TF-IDF sparse matrix
        tfidf_path = self.index_dir / "tfidf.pkl"
        if tfidf_path.exists():
            with open(tfidf_path, "rb") as f:
                self.tfidf = pickle.load(f)
            doc_texts = [c.get("text_folded", fold(c.get("text_spoken", ""))) for c in self.chunks]
            self.kw_matrix = self.tfidf.transform(doc_texts)
            print(f"[RetrievalService] Loaded TF-IDF model with {self.kw_matrix.shape[1]} features.")
        else:
            print("[RetrievalService] Building character n-gram TF-IDF vectorizer...")
            self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=1)
            doc_texts = [c.get("text_folded", fold(c.get("text_spoken", ""))) for c in self.chunks]
            self.kw_matrix = self.tfidf.fit_transform(doc_texts)
        # 4. Mark ready before warm-up to prevent recursive loop
        self.is_ready = True

        # 5. Warm-up
        try:
            for _ in range(2):
                self.retrieve("የሙከራ ጊዜ ስንት ነው?", k=TOP_K_RETRIEVAL)
        except Exception as e:
            print(f"[RetrievalService] Warm-up warning: {e}")

        print("[RetrievalService] Warm-up complete. Retrieval engine ready.")

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes query into normalized unit vector."""
        if not self.is_ready:
            self.load()
        q_norm = normalize_query(query)
        qv = self.embed_model.encode(
            ["query: " + q_norm],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")
        return qv[0]

    def retrieve(
        self,
        query: str,
        k: int = TOP_K_RETRIEVAL,
        alpha: float = ALPHA_HYBRID,
        mode: str = "hybrid"
    ) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Executes hybrid retrieval over proclamation articles.
        Returns:
            results: List of top matching chunks with scores and text
            timing: Timing breakdown in milliseconds
        """
        if not self.is_ready:
            self.load()

        t0 = time.perf_counter()
        q_norm = normalize_query(query)
        t1 = time.perf_counter()

        # Dense Cosine Similarity
        dense = np.zeros(len(self.chunks), dtype="float32")
        if mode in ("hybrid", "dense"):
            qv = self.embed_model.encode(
                ["query: " + q_norm],
                normalize_embeddings=True,
                convert_to_numpy=True
            ).astype("float32")
            dense = (self.embeddings @ qv[0])
        t2 = time.perf_counter()

        # Sparse TF-IDF Similarity
        kw = np.zeros(len(self.chunks), dtype="float32")
        if mode in ("hybrid", "keyword"):
            qk = self.tfidf.transform([q_norm])
            kw = (self.kw_matrix @ qk.T).toarray().ravel().astype("float32")
        t3 = time.perf_counter()

        # Score blending
        if mode == "dense":
            score = dense
        elif mode == "keyword":
            score = kw
        else:
            score = alpha * _minmax(dense) + (1.0 - alpha) * _minmax(kw)

        top_indices = np.argsort(-score)[:k]
        results = [
            {
                "id": self.chunks[i]["id"],
                "score": float(score[i]),
                "dense": float(dense[i]),
                "keyword": float(kw[i]),
                "text_spoken": self.chunks[i].get("text_spoken", ""),
                "text_clean": self.chunks[i].get("text_clean", ""),
                "article_id": self.chunks[i].get("article_id", self.chunks[i]["id"])
            }
            for i in top_indices
        ]
        t4 = time.perf_counter()

        timing = {
            "normalize_ms": (t1 - t0) * 1000.0,
            "embed_ms": (t2 - t1) * 1000.0,
            "keyword_ms": (t3 - t2) * 1000.0,
            "rank_ms": (t4 - t3) * 1000.0,
            "total_ms": (t4 - t0) * 1000.0,
        }
        return results, timing
