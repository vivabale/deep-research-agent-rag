from pathlib import Path
from ..schemas import Document
from .base import BaseLoader

class TextLoader(BaseLoader):
    def load(self,path):
        p=Path(path)
        return Document(p.read_text(encoding="utf-8", errors="ignore"), {"source":p.name,"filename":p.name,"filetype":p.suffix.lower()})
