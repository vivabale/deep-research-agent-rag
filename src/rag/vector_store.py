import json
import math
from pathlib import Path


class VectorStore:
    """Lightweight persistent vector store.

    Uses real cosine similarity, so it works whether or not the
    embedding provider returns normalized vectors (Qwen API does not).
    """

    def __init__(self, path='storage/rag_index.json'):
        self.path = Path(path)
        self.items = []

    def add(self, chunks, vectors):
        self.items = [
            {
                'content': chunk.content,
                'metadata': chunk.metadata,
                'vector': vector,
            }
            for chunk, vector in zip(chunks, vectors)
        ]

    def persist(self):
        self.path.parent.mkdir(exist_ok=True, parents=True)
        self.path.write_text(
            json.dumps(self.items, ensure_ascii=False),
            encoding='utf8'
        )

    def load(self):
        if not self.path.exists():
            return
        try:
            self.items = json.loads(self.path.read_text(encoding='utf8'))
        except json.JSONDecodeError:
            self.items = []

    @staticmethod
    def _cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query_vector, k=5, min_score=0.35):
        scored = []

        for item in self.items:
            score = self._cosine(query_vector, item['vector'])

            if score >= min_score:
                scored.append({
                    **item,
                    'score': float(score)
                })

        return sorted(
            scored,
            key=lambda x: x['score'],
            reverse=True
        )[:k]