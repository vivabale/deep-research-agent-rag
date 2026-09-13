import re
import uuid
from ..schemas import DocumentChunk
from .base import BaseSplitter

class RecursiveSplitter(BaseSplitter):
    def __init__(self, chunk_size=500, overlap_sentences=1):
        self.chunk_size = chunk_size
        self.overlap_sentences = overlap_sentences

    def _sentences(self, text):
        return [s.strip() for s in re.split(r'(?<=[。！？.!?])\s*', text) if s.strip()]

    def split(self, document):
        sentences = self._sentences(document.content)
        chunks = []
        current = []
        current_len = 0
        position = 0

        for sentence in sentences:
            if current and current_len + len(sentence) > self.chunk_size:
                content = "".join(current)
                meta = dict(document.metadata)
                meta.update({
                    "chunk_id": str(uuid.uuid4()),
                    "position": position,
                    "parent_source": document.metadata.get("source")
                })
                chunks.append(DocumentChunk(content, meta))
                position += 1
                current = current[-self.overlap_sentences:]
                current_len = sum(len(x) for x in current)

            current.append(sentence)
            current_len += len(sentence)

        if current:
            meta = dict(document.metadata)
            meta.update({
                "chunk_id": str(uuid.uuid4()),
                "position": position,
                "parent_source": document.metadata.get("source")
            })
            chunks.append(DocumentChunk("".join(current), meta))

        return chunks
