from dataclasses import dataclass, field
from typing import Dict

@dataclass
class Document:
    content: str
    metadata: Dict = field(default_factory=dict)

@dataclass
class DocumentChunk(Document):
    """Embedding unit generated from a parent Document.

    metadata keys: source, filename, filetype, chunk_id, position, parent_source
    """
    pass
