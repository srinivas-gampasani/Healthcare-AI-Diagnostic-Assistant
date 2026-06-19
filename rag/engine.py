"""
Clinical RAG Engine
===================
FAISS-backed retrieval-augmented generation over:
- Clinical guidelines (ADA, ACC/AHA, KDIGO, GINA)
- Drug reference information
- Clinical protocols
- Patient EHR data (structured + notes)

Uses sentence-transformers for embeddings (no OpenAI API key needed for embedding).
Falls back to TF-IDF-style keyword matching if embeddings unavailable.
"""
import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    id: str
    title: str
    category: str
    content: str
    score: float
    source_type: str  # guideline | protocol | ehr_note | drug_info


class ClinicalRAGEngine:
    """
    Production-grade RAG engine for clinical knowledge retrieval.
    
    Architecture:
    1. Documents are chunked and embedded at startup
    2. FAISS index enables fast ANN similarity search
    3. Retrieves top-k chunks and passes to LLM with patient context
    """

    def __init__(self):
        self.index = None
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings_model = None
        self._initialized = False

    def initialize(self):
        """Build FAISS index from clinical knowledge base."""
        if self._initialized:
            return

        from rag.knowledge_base import CLINICAL_GUIDELINES, CLINICAL_PROTOCOLS

        all_docs = []
        for doc in CLINICAL_GUIDELINES:
            # Chunk large docs into ~500 word segments
            chunks = self._chunk_text(doc["content"], max_words=400)
            for i, chunk in enumerate(chunks):
                all_docs.append({
                    "id": f"{doc['id']}_c{i}",
                    "title": doc["title"],
                    "category": doc.get("category", "general"),
                    "content": chunk,
                    "source_type": "guideline",
                    "icd10": doc.get("icd10", []),
                })

        for doc in CLINICAL_PROTOCOLS:
            all_docs.append({
                "id": doc["id"],
                "title": doc["title"],
                "category": doc.get("category", "protocol"),
                "content": doc["content"],
                "source_type": "protocol",
                "icd10": [],
            })

        self.chunks = all_docs

        # Try to build FAISS index with sentence-transformers
        try:
            self._build_faiss_index(all_docs)
            logger.info(f"FAISS index built with {len(all_docs)} chunks")
        except Exception as e:
            logger.warning(f"FAISS init failed ({e}), using keyword fallback")
            self.index = None

        self._initialized = True

    def _chunk_text(self, text: str, max_words: int = 400) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        step = max_words - 50  # 50-word overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + max_words])
            if chunk.strip():
                chunks.append(chunk)
        return chunks if chunks else [text]

    def _build_faiss_index(self, docs: List[Dict]):
        """Build FAISS index using sentence-transformers embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            import faiss

            model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embeddings_model = model

            texts = [f"{d['title']}. {d['content']}" for d in docs]
            embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            embeddings = embeddings.astype(np.float32)

            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine sim (normalized)
            self.index.add(embeddings)

        except ImportError:
            raise RuntimeError("faiss-cpu or sentence-transformers not installed")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        icd10_boost: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve top-k relevant clinical chunks for a query.
        
        Args:
            query: Clinical question or patient context
            top_k: Number of chunks to return
            category_filter: Optional category (diabetes, cardiology, etc.)
            icd10_boost: ICD-10 codes to boost relevance
        """
        if not self._initialized:
            self.initialize()

        if self.index is not None and self.embeddings_model is not None:
            results = self._faiss_retrieve(query, top_k * 2)
        else:
            results = self._keyword_retrieve(query, top_k * 2)

        # Apply category filter
        if category_filter:
            results = [r for r in results if r.category == category_filter] or results

        # ICD-10 boost: promote chunks that match the patient's conditions
        if icd10_boost:
            boosted = []
            normal = []
            for r in results:
                chunk_icd = self.chunks[next(
                    (i for i, c in enumerate(self.chunks) if c["id"] == r.id), 0
                )].get("icd10", [])
                if any(code in icd10_boost for code in chunk_icd):
                    r.score += 0.15  # boost score
                    boosted.append(r)
                else:
                    normal.append(r)
            results = sorted(boosted + normal, key=lambda x: x.score, reverse=True)

        return results[:top_k]

    def _faiss_retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """FAISS similarity search."""
        q_emb = self.embeddings_model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(q_emb, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append(RetrievedChunk(
                id=chunk["id"],
                title=chunk["title"],
                category=chunk["category"],
                content=chunk["content"],
                score=float(score),
                source_type=chunk["source_type"],
            ))
        return results

    def _keyword_retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """TF-IDF-style keyword fallback retrieval."""
        query_words = set(query.lower().split())
        scored = []
        for chunk in self.chunks:
            text = f"{chunk['title']} {chunk['content']}".lower()
            text_words = set(text.split())
            overlap = len(query_words & text_words)
            score = overlap / (len(query_words) + 1)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedChunk(
                id=c["id"], title=c["title"], category=c["category"],
                content=c["content"], score=s, source_type=c["source_type"]
            )
            for s, c in scored[:top_k]
        ]

    def format_context(self, chunks: List[RetrievedChunk], max_chars: int = 8000) -> str:
        """Format retrieved chunks into LLM context string."""
        parts = []
        total = 0
        for i, chunk in enumerate(chunks, 1):
            text = f"[Source {i}: {chunk.title} ({chunk.category})]\n{chunk.content.strip()}\n"
            if total + len(text) > max_chars:
                break
            parts.append(text)
            total += len(text)
        return "\n---\n".join(parts)


# Singleton
rag_engine = ClinicalRAGEngine()
