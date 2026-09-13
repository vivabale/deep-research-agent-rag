import asyncio
import logging
from pathlib import Path

from src.state import SearchResult
from .embeddings import LocalEmbedding
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class KnowledgeBaseRetriever:
    """Async wrapper around local vector retrieval.

    Embedding and vector search are CPU-bound operations, therefore they are
    executed in a worker thread to avoid blocking the LangGraph event loop.
    """

    def __init__(self, index_path=None):
        self.embedding = LocalEmbedding()
        self.store = VectorStore(path=index_path) if index_path else VectorStore()
        self.store.load()

        if not self.store.items:
            logger.warning(
                "RAG index is empty. Run `python -m src.rag.build_index` before retrieval."
            )

    def _search_sync(self, query, k, min_score):
        vec = self.embedding.embed_query(query)
        return self.store.search(vec, k=k, min_score=min_score)

    async def search(self, query, k=5, min_score=0.35):
        docs = await asyncio.to_thread(
            self._search_sync,
            query,
            k,
            min_score,
        )

        results = []
        for d in docs:
            meta = d.get("metadata", {})
            chunk_id = meta.get("chunk_id", "unknown")
            source = meta.get("source", meta.get("filename", "local"))
            score = d.get("score", 0.0)

            results.append(SearchResult(
                query=query,
                title=meta.get("title", Path(source).name),
                url=f"rag://{source}#{chunk_id}",
                snippet=d.get("content", "")[:300],
                content=d.get("content"),
                source_type="local_rag",
                metadata={
                    **meta,
                    "score": score,
                    "credibility": {
                        "score": 85,
                        "level": "high",
                        "factors": ["verified_local_knowledge_base"]
                    }
                }
            ))

        return results
