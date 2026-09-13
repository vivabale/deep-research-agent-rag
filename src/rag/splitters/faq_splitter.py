import re, uuid
from ..schemas import DocumentChunk
from .base import BaseSplitter

class FAQSplitter(BaseSplitter):
    def split(self, document):
        chunks=[]
        for idx,part in enumerate(re.split(r"(?=\n?\d+[\.、])", document.content)):
            if part.strip():
                meta=dict(document.metadata); meta.update({"chunk_id":str(uuid.uuid4()),"position":idx})
                chunks.append(DocumentChunk(part.strip(),meta))
        return chunks
